#!/usr/bin/env python3
"""
Aggregate 2x2 per-event NPZ shards into one per-file ``.pt``.

The 2x2 worker (``run_2x2_worker.py``) writes one ``<basename>__ev<NNNN>.npz`` per
event with keys ``hit_refs`` (rows into ``charge/calib_prompt_hits/data``),
``hit_timestamps`` (matched t0, sentinel ``-1`` / NaN when unassigned) and
``labels`` (per-hit cluster id).  This script scatters them into file-global
arrays and writes ``<basename>.qlmatch2x2.pt`` with the SAME schema as the ND
``v_alpha_test`` release (``calib_hit_t0_reco`` + ``calib_final_hit_t0_reco``),
so the two detectors are interoperable downstream.

Usage::

    python TwoByTwo/aggregate_2x2_to_pt.py \
        --shard-dir output/2x2_sim --output-dir output/2x2_sim/pt_outputs --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import h5py

# Uniform sentinel -10000 for all three writeback fields (t_0, t_cluster_id,
# t_confidence). Unphysical for all of them so we can safely distinguish
# "CL matching ran and this hit was unassigned" from the pre-CL-matching
# HDF5 default of 0. Matches clmatchND_v1's convention.
UNASSIGNED = -10000
T0_SENTINEL = float(UNASSIGNED)
CID_SENTINEL = int(UNASSIGNED)
CONF_UNAVAILABLE = float(UNASSIGNED)
NS_PER_TICK = 16.0      # LArPix clock: 1 tick = 16 ns
# Schema v0.3 (this file):
#   * t_0 stored as NANOSECONDS (ticks*16), matching the ND clmatchND_v1 schema
#     so apply_pt_to_hdf5.py (in ND_Production) can fill both detectors uniformly.
#   * `calib_prompt_hits` and `calib_final_hits` nested dicts of {t_0, t_cluster_id,
#     t_confidence} added alongside the legacy flat keys for schema parity with ND.
#   * `t0_units: "ns"` and `ticks_per_ns: 16.0` metadata added; apply_pt_to_hdf5
#     refuses to write a PT that doesn't declare these.
SCHEMA_VERSION = "qlmatch2x2.4"  # bump: t_confidence uses real matched-filter cos
_FINAL_TO_PROMPT_REF = "charge/calib_prompt_hits/ref/charge/calib_final_hits/ref"


def _final_to_prompt_indices(h5: h5py.File) -> np.ndarray:
    final_hits = h5["charge/calib_final_hits/data"]
    n_final = int(final_hits.shape[0])
    if _FINAL_TO_PROMPT_REF in h5 and int(h5[_FINAL_TO_PROMPT_REF].shape[0]) == n_final:
        return np.asarray(h5[_FINAL_TO_PROMPT_REF][:, 0], dtype=np.int64)
    if "id" in final_hits.dtype.names:
        return np.asarray(final_hits["id"], dtype=np.int64)
    raise RuntimeError("cannot derive calib_final -> calib_prompt mapping")


def _src_from_jsons(jsons: list[Path]) -> str | None:
    for jp in jsons:
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        fp = d.get("file") or d.get("input_file")
        if fp and Path(fp).exists():
            return str(fp)
    return None


def _gather_by_file(shard_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for npz in sorted(shard_dir.glob("*__ev*.npz")):
        base = npz.name.split("__ev")[0]
        groups.setdefault(base, []).append(npz)
    return groups


def _aggregate_one(base: str, shards: list[Path], out_dir: Path,
                   algo: str, overwrite: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_path = out_dir / f"{base}.qlmatch2x2.pt"
    if out_path.exists() and not overwrite:
        return {"file": base, "status": "skipped_existing", "out": str(out_path)}

    jsons = [Path(str(p).replace(".npz", ".json")) for p in shards]
    jsons = [p for p in jsons if p.exists()]
    src = _src_from_jsons(jsons)
    if src is None:
        return {"file": base, "status": "no_source_file"}

    with h5py.File(src, "r") as h:
        n_prompt = int(h["charge/calib_prompt_hits/data"].shape[0])
        all_event_ids = np.asarray(h["charge/events/data"]["id"], dtype=np.int64)

    calib_hit_t0_reco = np.full(n_prompt, T0_SENTINEL, dtype=np.float32)
    prompt_hit_cluster_id = np.full(n_prompt, CID_SENTINEL, dtype=np.int16)
    # v0.4: per-hit t_confidence is the MATCHED-FILTER COS at the accepted t0,
    # averaged over the cluster's TPCs. Computed by pipeline_2x2.py as a post-
    # pass and saved in each NPZ shard as (cluster_cos_labels, cluster_cos_values).
    # Fallback (older shards without those arrays): support-size proxy.
    # Value is in [~0, 1] for assigned hits (higher = better light-charge match);
    # unassigned hits get UNASSIGNED (-10000) sentinel.
    prompt_hit_t_confidence = np.full(n_prompt, CONF_UNAVAILABLE, dtype=np.float32)

    processed, summaries, failed = [], [], []
    n_assigned = 0
    SUPPORT_REF = 20.0  # legacy fallback only (used when NPZ lacks cluster_cos_*)
    for npz in sorted(shards):
        jp = Path(str(npz).replace(".npz", ".json"))
        meta = {}
        try:
            meta = json.load(open(jp))
        except Exception:
            pass
        if meta.get("ok") is False:
            failed.append({"event_id": int(meta.get("event_id", -1)),
                           "error": str(meta.get("error", ""))[:300]})
            continue
        try:
            d = np.load(npz)
        except Exception as exc:
            failed.append({"event_id": int(meta.get("event_id", -1)),
                           "error": f"npz read: {exc}"})
            continue
        if int(d.get("ok", 1)) == 0 or d["hit_refs"].size == 0:
            continue
        hit_refs = np.asarray(d["hit_refs"], dtype=np.int64)
        ts = np.asarray(d["hit_timestamps"], dtype=np.float32)
        labels = np.asarray(d["labels"], dtype=np.int64) if "labels" in d.files else None
        if hit_refs.size != ts.size:
            failed.append({"event_id": int(meta.get("event_id", -1)),
                           "error": f"shape mismatch {hit_refs.size}!={ts.size}"})
            continue
        valid = np.isfinite(ts) & (ts >= 0)
        # Convert ticks -> ns for storage. Pipeline emits `hit_timestamps` in
        # ticks (see pipeline_2x2.py docstring: "assigns every hit a t0
        # (matching ticks)"); the ndlar_flow t_0 field is float32 ns.
        calib_hit_t0_reco[hit_refs[valid]] = (ts[valid].astype(np.float32)
                                              * np.float32(NS_PER_TICK))
        n_assigned += int(valid.sum())
        if labels is not None and labels.size == hit_refs.size:
            lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
            prompt_hit_cluster_id[hit_refs] = np.clip(labels, lo, hi).astype(np.int16)
            # ---- per-hit confidence ----
            # Preferred: use the matched-filter cos score the pipeline saved
            # per cluster (cluster_cos_labels + cluster_cos_values). Assigned
            # hits get their cluster's cos; unassigned hits get UNASSIGNED
            # sentinel.
            # Fallback (older shards or a cluster whose cos couldn't be
            # computed): support-size proxy min(1, n_hits_in_cluster/SUPPORT_REF).
            cos_lab = np.asarray(d["cluster_cos_labels"], np.int64) \
                if "cluster_cos_labels" in d.files else np.zeros(0, np.int64)
            cos_val = np.asarray(d["cluster_cos_values"], np.float32) \
                if "cluster_cos_values" in d.files else np.zeros(0, np.float32)
            lab_to_cos = dict(zip(cos_lab.tolist(), cos_val.tolist()))
            # Fallback support-size dict for any assigned cluster the pipeline
            # didn't score (e.g. cos calculation raised or all TPCs were empty).
            uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
            lab_to_fallback = {int(l): float(min(1.0, c / SUPPORT_REF))
                               for l, c in zip(uniq, counts)}
            per_hit_conf = np.array(
                [(lab_to_cos.get(int(l), lab_to_fallback.get(int(l), CONF_UNAVAILABLE))
                  if v else CONF_UNAVAILABLE)
                 for l, v in zip(labels, valid)],
                dtype=np.float32,
            )
            prompt_hit_t_confidence[hit_refs] = per_hit_conf
        ev_id = int(meta.get("event_id", d["ev_id"]) if "ev_id" in d.files else meta.get("event_id", -1))
        processed.append(ev_id)
        summaries.append({"event_id": ev_id, "n_hits": int(hit_refs.size),
                          "n_matched": int(valid.sum()),
                          "elapsed_s": float(meta.get("elapsed_s", 0.0))})

    # ---- derive merged (calib_final) hit fields ----
    with h5py.File(src, "r") as h:
        prompt_idx = _final_to_prompt_indices(h)
        n_final = int(h["charge/calib_final_hits/data"].shape[0])
    final_t0 = np.full(n_final, T0_SENTINEL, dtype=np.float32)
    final_cluster = np.full(n_final, CID_SENTINEL, dtype=np.int16)
    final_confidence = np.full(n_final, CONF_UNAVAILABLE, dtype=np.float32)
    in_range = (prompt_idx >= 0) & (prompt_idx < n_prompt)
    final_t0[in_range] = calib_hit_t0_reco[prompt_idx[in_range]]
    final_cluster[in_range] = prompt_hit_cluster_id[prompt_idx[in_range]]
    final_confidence[in_range] = prompt_hit_t_confidence[prompt_idx[in_range]]
    n_final_assigned = int(np.count_nonzero((final_t0 != T0_SENTINEL) & np.isfinite(final_t0) & (final_t0 >= 0)))

    out = {
        "version": SCHEMA_VERSION,
        "algorithm": algo,
        "detector": "2x2",
        "input_file": str(src),
        "src_basename": Path(src).name,
        # units contract (matches ND clmatchND_v1)
        "t0_units": "ns",
        "ticks_per_ns": NS_PER_TICK,
        "unassigned_sentinel": UNASSIGNED,
        # ---- HDF5-fill schema (nested dicts; apply_pt_to_hdf5.py reads these) ----
        # sizes: prompt = n_calib_hits, final = n_calib_final_hits
        "calib_prompt_hits": {
            "t_0": torch.from_numpy(calib_hit_t0_reco),           # float32 ns
            "t_cluster_id": torch.from_numpy(prompt_hit_cluster_id),  # int16
            "t_confidence": torch.from_numpy(prompt_hit_t_confidence),  # float32
        },
        "calib_final_hits": {
            "t_0": torch.from_numpy(final_t0),
            "t_cluster_id": torch.from_numpy(final_cluster),
            "t_confidence": torch.from_numpy(final_confidence),
        },
        "n_calib_prompt_hits": int(n_prompt),
        "n_prompt_assigned": int(n_assigned),
        "n_calib_final_hits": int(n_final),
        "n_final_assigned": int(n_final_assigned),
        # ---- legacy flat keys kept for backward compat with earlier 2x2 consumers ----
        "calib_hit_t0_reco": torch.from_numpy(calib_hit_t0_reco),
        "prompt_hit_t_cluster_id": torch.from_numpy(prompt_hit_cluster_id),
        "prompt_hit_t_confidence": torch.from_numpy(prompt_hit_t_confidence),
        "confidence_source": "matched-filter cos at accepted t0 per cluster "
                            "(pipeline_2x2 post-pass); fallback: support-size proxy",
        "confidence_support_ref": SUPPORT_REF,
        "n_calib_hits": int(n_prompt),
        "n_assigned": int(n_assigned),
        "n_unassigned": int(n_prompt - n_assigned),
        "calib_final_hit_t0_reco": torch.from_numpy(final_t0),
        "calib_final_hit_cluster_id": torch.from_numpy(final_cluster),
        "calib_final_hit_t_confidence": torch.from_numpy(final_confidence),
        "calib_final_hit_prompt_index": torch.from_numpy(prompt_idx.astype(np.int64)),
        "calib_final_hit_source": ("derived from calib_hit_t0_reco via "
                                   + _FINAL_TO_PROMPT_REF + "[:,0]"),
        "n_calib_final_hits": int(n_final),
        "n_calib_final_assigned": int(n_final_assigned),
        "n_calib_final_unassigned": int(n_final - n_final_assigned),
        # metadata
        "processed_event_ids": torch.from_numpy(np.asarray(sorted(set(processed)), dtype=np.int64)),
        "all_event_ids": torch.from_numpy(all_event_ids),
        "event_summaries": summaries,
        "failed_events": failed,
        "n_event_shards": int(len(shards)),
        "aggregator_elapsed_s": float(time.perf_counter() - t0),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return {"file": base, "status": "ok", "out": str(out_path),
            "n_prompt": n_prompt, "n_assigned": n_assigned,
            "n_events": len(processed), "n_failed": len(failed)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--algorithm", default="2x2 charge-light matching (QLMatching2x2)")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.output_dir) if args.output_dir else shard_dir / "pt_outputs"
    groups = _gather_by_file(shard_dir)
    if not groups:
        print(f"no shards (*__ev*.npz) under {shard_dir}")
        return
    for base, shards in groups.items():
        res = _aggregate_one(base, shards, out_dir, args.algorithm, args.overwrite)
        cov = (f" assigned={res.get('n_assigned')}/{res.get('n_prompt')}"
               f" events={res.get('n_events')} failed={res.get('n_failed')}"
               if res.get("status") == "ok" else "")
        print(f"[{res['status']}] {base} -> {res.get('out','')}{cov}", flush=True)


if __name__ == "__main__":
    main()
