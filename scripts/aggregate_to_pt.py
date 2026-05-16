"""Aggregate per-event NPZ shards into a per-file .pt with the vBeta3 schema.

Run after a v_alpha_test batch (run_v_alpha_test_pt_parallel.sh).  The
v_alpha_test python module writes one ``<basename>__ev<NNNN>.npz`` per event
plus a sibling ``__ev<NNNN>.json``.  This script:

1. Groups them by source file (using the basename in the filename).
2. For each file, opens the source HDF5 to read the calib_prompt_hits length.
3. Allocates per-file output arrays sized to the full prompt-hit table.
4. Scatters per-event ``hit_timestamps_post_phase3`` and ``labels_global`` into
   the per-file arrays via each event's ``hit_refs``.
5. Writes ``<basename>.v_alpha_test.pt`` next to the NPZ shards (or to
   ``--output-dir`` if given) with the full schema described in
   ``v_alpha_test/config.yaml``.

CLI::

    python aggregate_to_pt.py \\
        --shard-dir /pscratch/.../valpha_runs/test10_v_alpha_test \\
        --output-dir /pscratch/.../valpha_runs/test10_v_alpha_test/pt_outputs

If ``--output-dir`` is omitted, .pt files land in ``--shard-dir`` next to
the per-event NPZs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


# Default sentinel values, matching the conventions used by vBeta2/vBeta3.
T0_RECO_SENTINEL = -1.0       # float32, "no t0 assigned"
CLUSTER_ID_SENTINEL = -1       # int16,   "no cluster"

# vBeta3 convention: each calib_final_hit row maps to one source prompt-hit
# via column 0 of this ref dataset.  Fallback: calib_final_hits/data['id'].
_CALIB_FINAL_TO_PROMPT_REF = "charge/calib_prompt_hits/ref/charge/calib_final_hits/ref"


def _calib_final_to_prompt_indices(h5: h5py.File) -> np.ndarray:
    """Return one source calib_prompt_hits row index per calib_final_hits row.

    Matches the vBeta3AllHits convention so the schemas are interoperable.
    """
    final_hits = h5["charge/calib_final_hits/data"]
    n_final = int(final_hits.shape[0])
    if _CALIB_FINAL_TO_PROMPT_REF in h5 and int(h5[_CALIB_FINAL_TO_PROMPT_REF].shape[0]) == n_final:
        return np.asarray(h5[_CALIB_FINAL_TO_PROMPT_REF][:, 0], dtype=np.int64)
    if "id" in final_hits.dtype.names:
        return np.asarray(final_hits["id"], dtype=np.int64)
    raise RuntimeError(
        "Could not derive calib_final_hits -> calib_prompt_hits mapping. "
        f"Missing or malformed {_CALIB_FINAL_TO_PROMPT_REF}, and "
        "calib_final_hits/data has no 'id' field."
    )


# Per-prompt-hit fields produced by this aggregator.
SCHEMA_VERSION = "v_alpha_test.1"
ALGORITHM = (
    "v_alpha_test: front-stage + Phase 2 + V2 light rescue + Phase 3 small-cluster matrix; "
    "torch.compile + TF32 + cudnn.benchmark + h5 prefetch; per-event NPZ shards "
    "merged into per-file .pt by aggregate_to_pt.py."
)


def _gather_shards_by_file(shard_dir: Path) -> dict[str, list[Path]]:
    """Return {file_basename: [event_npz, event_npz, ...]}."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for npz in sorted(shard_dir.glob("*__ev*.npz")):
        # Filename pattern: <basename>__ev<NNNN>.npz
        stem = npz.stem
        if "__ev" not in stem:
            continue
        base = stem.split("__ev")[0]
        groups[base].append(npz)
    return groups


def _resolve_source_file(file_basename: str, search_root: str | None) -> str | None:
    """Locate the source HDF5 by filename, used for the prompt-hit length."""
    candidate_name = f"{file_basename}.hdf5"
    if search_root and Path(search_root).is_dir():
        for candidate in Path(search_root).rglob(candidate_name):
            return str(candidate)
    # Fallback: the JSON sibling next to the NPZ records the absolute file path.
    return None


def _file_path_from_jsons(jsons: list[Path]) -> str | None:
    for jp in jsons:
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        fp = d.get("file") or d.get("input_file")
        if fp and Path(fp).exists():
            return str(fp)
    return None


def _aggregate_one_file(
    file_basename: str,
    shard_npzs: list[Path],
    *,
    output_dir: Path,
    extra_search_root: str | None,
    overwrite: bool,
    verbose: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_path = output_dir / f"{file_basename}.v_alpha_test.pt"
    if out_path.exists() and not overwrite:
        return {
            "file_basename": file_basename,
            "status": "skipped_existing",
            "out_path": str(out_path),
        }

    # Sibling JSONs carry the absolute source-file path.
    json_paths = [Path(str(p).replace(".npz", ".json")) for p in shard_npzs]
    json_paths = [p for p in json_paths if p.exists()]
    src_file = _file_path_from_jsons(json_paths) or _resolve_source_file(
        file_basename, extra_search_root
    )
    if src_file is None:
        return {
            "file_basename": file_basename,
            "status": "no_source_file",
            "out_path": None,
            "n_event_shards": len(shard_npzs),
        }

    # Get the file-global prompt-hit count + processed event ids from the h5.
    with h5py.File(src_file, "r") as h:
        n_calib_hits = int(h["charge/calib_prompt_hits/data"].shape[0])
        all_event_ids = np.asarray(h["charge/events/data"]["id"], dtype=np.int64)

    # Allocate file-level outputs.
    calib_hit_t0_reco = np.full(n_calib_hits, T0_RECO_SENTINEL, dtype=np.float32)
    prompt_hit_t_cluster_id = np.full(n_calib_hits, CLUSTER_ID_SENTINEL, dtype=np.int16)

    processed_event_ids: list[int] = []
    event_summaries: list[dict[str, Any]] = []
    failed_events: list[dict[str, Any]] = []
    n_assigned = 0

    for npz_path in sorted(shard_npzs):
        json_path = Path(str(npz_path).replace(".npz", ".json"))
        try:
            meta = json.load(open(json_path))
        except Exception as exc:
            meta = {"event_id": -1, "ok": False, "error": f"json read: {exc}"}

        if meta.get("ok") is False:
            failed_events.append({
                "event_id": int(meta.get("event_id", -1)),
                "error": str(meta.get("error", ""))[:300],
            })
            continue

        try:
            d = np.load(npz_path)
        except Exception as exc:
            failed_events.append({
                "event_id": int(meta.get("event_id", -1)),
                "error": f"npz read: {exc}",
            })
            continue

        if "hit_refs" not in d.files or "hit_timestamps_post_phase3" not in d.files:
            failed_events.append({
                "event_id": int(meta.get("event_id", -1)),
                "error": "shard missing hit_refs / hit_timestamps_post_phase3",
            })
            continue

        hit_refs = np.asarray(d["hit_refs"], dtype=np.int64)
        ts_post = np.asarray(d["hit_timestamps_post_phase3"], dtype=np.float32)
        # Prefer the post-V2 cluster id (with re-labeled moved hits); fall
        # back to the original front-stage labels_global if the shard was
        # produced by an older module version.
        if "t_cluster_id" in d.files:
            labels = np.asarray(d["t_cluster_id"], dtype=np.int64)
        elif "labels_global" in d.files:
            labels = np.asarray(d["labels_global"], dtype=np.int64)
        else:
            labels = None

        if hit_refs.size != ts_post.size:
            failed_events.append({
                "event_id": int(meta.get("event_id", -1)),
                "error": f"shape mismatch: hit_refs={hit_refs.size}, ts_post={ts_post.size}",
            })
            continue

        # Scatter per-event arrays into file-level output via hit_refs.
        ts_clean = ts_post.copy()
        # Hits with NaN or pre-stage-unassigned remain at T0_RECO_SENTINEL.
        valid_mask = np.isfinite(ts_clean) & (ts_clean >= 0)
        calib_hit_t0_reco[hit_refs[valid_mask]] = ts_clean[valid_mask]
        n_assigned += int(valid_mask.sum())

        if labels is not None:
            # Coerce to int16.  V2 re-labeling pushes ids past the original
            # cluster count; warn (in failed_events) if any id overflows.
            lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
            n_overflow = int(((labels > hi) | (labels < lo)).sum())
            if n_overflow > 0:
                failed_events.append({
                    "event_id": int(meta.get("event_id", -1)),
                    "error": (
                        f"int16 cluster-id overflow: {n_overflow} hits had "
                        f"label outside [{lo},{hi}] (max was {int(labels.max())}); "
                        "consider widening prompt_hit_t_cluster_id to int32."
                    ),
                })
            lbl16 = np.clip(labels, lo, hi).astype(np.int16)
            prompt_hit_t_cluster_id[hit_refs] = lbl16

        ev_id = int(meta.get("event_id", -1))
        processed_event_ids.append(ev_id)
        event_summaries.append({
            "event_id": ev_id,
            "n_hits": int(meta.get("n_hits", hit_refs.size)),
            "elapsed_s": float(meta.get("elapsed_s", 0.0)),
            "summary": meta.get("summary", {}),
        })

    n_unassigned = int(n_calib_hits - n_assigned)

    # ---- Derive merged (calib_final) hit fields ----
    # vBeta3 convention: per-final-hit value = value of column-0 prompt hit
    # from charge/calib_prompt_hits/ref/charge/calib_final_hits/ref.
    final_t0 = np.full(0, T0_RECO_SENTINEL, dtype=np.float32)
    final_cluster = np.full(0, CLUSTER_ID_SENTINEL, dtype=np.int16)
    final_prompt_index = np.zeros(0, dtype=np.int64)
    n_calib_final_hits = 0
    n_calib_final_assigned = 0
    final_source_note = ""
    try:
        with h5py.File(src_file, "r") as h:
            prompt_idx = _calib_final_to_prompt_indices(h)
            n_calib_final_hits = int(h["charge/calib_final_hits/data"].shape[0])
        final_prompt_index = prompt_idx.astype(np.int64, copy=False)
        final_t0 = np.full(n_calib_final_hits, T0_RECO_SENTINEL, dtype=np.float32)
        final_cluster = np.full(n_calib_final_hits, CLUSTER_ID_SENTINEL, dtype=np.int16)
        in_range = (prompt_idx >= 0) & (prompt_idx < n_calib_hits)
        final_t0[in_range] = calib_hit_t0_reco[prompt_idx[in_range]]
        final_cluster[in_range] = prompt_hit_t_cluster_id[prompt_idx[in_range]]
        n_calib_final_assigned = int(
            np.count_nonzero((final_t0 != T0_RECO_SENTINEL) & np.isfinite(final_t0) & (final_t0 >= 0))
        )
        final_source_note = (
            "charge/calib_final_hits/data, derived from calib_hit_t0_reco and "
            "prompt_hit_t_cluster_id through "
            "charge/calib_prompt_hits/ref/charge/calib_final_hits/ref[:, 0]"
        )
    except Exception as exc:
        failed_events.append({
            "event_id": -1,
            "error": f"final-hit derivation failed: {exc}",
        })

    elapsed = float(time.perf_counter() - t0)

    out = {
        "version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "input_file": str(src_file),
        # Per-prompt-hit (size = n_calib_hits)
        "calib_hit_t0_reco": torch.from_numpy(calib_hit_t0_reco),
        "prompt_hit_t_cluster_id": torch.from_numpy(prompt_hit_t_cluster_id),
        "n_calib_hits": int(n_calib_hits),
        "n_assigned": int(n_assigned),
        "n_unassigned": int(n_unassigned),
        # Per-merged-hit (size = n_calib_final_hits)
        "calib_final_hit_t0_reco": torch.from_numpy(final_t0),
        "calib_final_hit_cluster_id": torch.from_numpy(final_cluster),
        "calib_final_hit_prompt_index": torch.from_numpy(final_prompt_index),
        "calib_final_hit_source": final_source_note,
        "calib_final_hit_t0_unassigned_value": T0_RECO_SENTINEL,
        "calib_final_hit_cluster_id_unassigned_value": CLUSTER_ID_SENTINEL,
        "n_calib_final_hits": int(n_calib_final_hits),
        "n_calib_final_assigned": int(n_calib_final_assigned),
        "n_calib_final_unassigned": int(max(n_calib_final_hits - n_calib_final_assigned, 0)),
        # Event metadata
        "processed_event_ids": torch.from_numpy(np.asarray(sorted(processed_event_ids), dtype=np.int64)),
        "all_event_ids": torch.from_numpy(all_event_ids),
        "event_summaries": event_summaries,
        "failed_events": failed_events,
        "n_event_shards": int(len(shard_npzs)),
        "aggregator_elapsed_s": float(elapsed),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    if verbose:
        if n_calib_final_hits > 0:
            final_str = (
                f"  merged={n_calib_final_assigned}/{n_calib_final_hits}"
                f" ({100.0*n_calib_final_assigned/max(n_calib_final_hits,1):.2f}%)"
            )
        else:
            final_str = "  merged=skipped"
        print(
            f"  wrote {out_path.name}  "
            f"events={len(processed_event_ids)}  "
            f"prompt={n_assigned}/{n_calib_hits}"
            f" ({100.0*n_assigned/max(n_calib_hits,1):.2f}%)"
            f"{final_str}  elapsed={elapsed:.1f}s",
            flush=True,
        )
    return {
        "file_basename": file_basename,
        "status": "ok",
        "out_path": str(out_path),
        "n_event_shards": int(len(shard_npzs)),
        "n_assigned": int(n_assigned),
        "n_calib_hits": int(n_calib_hits),
        "n_calib_final_hits": int(n_calib_final_hits),
        "n_calib_final_assigned": int(n_calib_final_assigned),
        "elapsed_s": elapsed,
    }


def aggregate_dir(
    *,
    shard_dir: str,
    output_dir: str | None = None,
    extra_search_root: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    shard_p = Path(shard_dir)
    out_p = Path(output_dir) if output_dir else shard_p
    out_p.mkdir(parents=True, exist_ok=True)

    groups = _gather_shards_by_file(shard_p)
    if verbose:
        print(f"aggregator: {len(groups)} source files in {shard_p}", flush=True)

    results: list[dict[str, Any]] = []
    for file_basename in sorted(groups.keys()):
        if verbose:
            print(f"  aggregating {file_basename}: {len(groups[file_basename])} shards", flush=True)
        res = _aggregate_one_file(
            file_basename,
            groups[file_basename],
            output_dir=out_p,
            extra_search_root=extra_search_root,
            overwrite=overwrite,
            verbose=verbose,
        )
        results.append(res)

    summary_path = out_p / "v_alpha_test_aggregator_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "version": SCHEMA_VERSION,
                "shard_dir": str(shard_p),
                "output_dir": str(out_p),
                "n_files": len(groups),
                "results": results,
            },
            f,
            indent=1,
            default=float,
        )
    if verbose:
        print(f"aggregator: summary -> {summary_path}", flush=True)
    return {"results": results, "summary_path": str(summary_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--shard-dir", required=True,
                        help="Directory containing per-event __ev<NNNN>.npz/.json shards.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write per-file .pt outputs (default: --shard-dir).")
    parser.add_argument("--extra-search-root", default=None,
                        help="If JSONs don't carry an absolute file path, "
                             "rglob *.hdf5 under this root to find the source.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing per-file .pt outputs.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    res = aggregate_dir(
        shard_dir=args.shard_dir,
        output_dir=args.output_dir,
        extra_search_root=args.extra_search_root,
        overwrite=bool(args.overwrite),
        verbose=not bool(args.quiet),
    )
    n_ok = sum(1 for r in res["results"] if r["status"] == "ok")
    n_skip = sum(1 for r in res["results"] if r["status"] == "skipped_existing")
    n_err = len(res["results"]) - n_ok - n_skip
    print(f"aggregator: ok={n_ok} skipped={n_skip} err={n_err}", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
