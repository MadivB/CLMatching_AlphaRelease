#!/usr/bin/env python3
"""
2x2 charge-light matching worker — one GPU worker of an N-worker fan-out.

Mirrors the ND ``M5p1.phase25_trial2_v_alpha_test`` worker, but drives the 2x2
matcher (``TwoByTwo/matcher`` = the QLMatching2x2 package) and the 2x2 perceiver
(``TwoByTwo/perceiver`` = ML_2x2_perceiver).  Each worker processes the events
``(idx - offset) % stride == 0`` of every input FLOW file and writes one
``<basename>__ev<NNNN>.npz`` shard per event.  ``aggregate_2x2_to_pt.py`` then
merges the shards into one per-file ``<basename>.qlmatch2x2.pt`` whose schema
matches the ND release (``calib_hit_t0_reco`` scattered via ``hit_refs``).

Algorithm versions
------------------
* ``--version v1.0``  (DEFAULT) — the **error-matrix** small-cluster association:
  greedy per-TPC brightest-first placement, unit-variance chi2.  This is the
  validated, conservative baseline (== the ND vAlpha formulation).
* ``--version v0.1``  — the development line toward the first serious release:
  cluster-guided **region-grow** after-track association (tuned: conf_cos=0.55,
  light_margin=0.04) plus the learned-variance **tiebreaker** that re-ranks
  ambiguous t0 candidates with the 2x2 variance model.  Higher efficiency on the
  validation sample but not yet the default.  (``v2.0`` = legacy alias.)
* ``--version v0.1-fx``  (EXPERIMENTAL) — cosine-free **chi2 family expansion**
  post-pass (family_expand_2x2.py): agglomerative spatial families arbitrated by
  the error matrix.  Beats v0.1 on the 2x2 aggregate but the 2x2 prototype
  scores at base=0 (known multi-flash caveat; the ND port fixes this with
  remove-and-rescore — see M5p1/postpass_v01.py).

All versions use unit-variance for the core matching (the established best for
2x2 t0 chi2 — bright channels carry the flash-discriminating signal).

Run directly (sys.path is wired up below); not as ``-m`` (the matcher uses flat
imports).  Example::

    CUDA_VISIBLE_DEVICES=0 python TwoByTwo/run_2x2_worker.py \
        --files /path/to/file.FLOW.hdf5 --out-dir output/2x2_sim \
        --mode sim --version v1.0 --event-stride 8 --event-offset 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Wire up sys.path so the flat-import matcher + perceiver resolve from in-repo.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # .../CLMatching_AlphaRelease/TwoByTwo
_MATCHER = _HERE / "matcher"
_PERCEIVER = _HERE / "perceiver"
_VARMODEL = _HERE / "var_model"
for _p in (_MATCHER, _PERCEIVER, _VARMODEL):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
# light_model_2x2 reads C2L_DIR at import time for the perceiver + default ckpts.
os.environ.setdefault("C2L_DIR", str(_PERCEIVER))

_PULSE_SIM = _HERE / "assets" / "avg_pulse_2x2.npy"
_PULSE_DATA = _HERE / "assets" / "avg_pulse_data.npy"   # data-derived template (2026-08-29)
_VARCKPT = _VARMODEL / "var_model_2x2.pt"


def _list_events(h5):
    """Charge event ids that have a light match (the matchable population)."""
    try:
        ref = np.asarray(h5["charge/events/ref/light/events/ref"][()], np.int64)
        return np.unique(ref[:, 0]).tolist()
    except Exception:
        # fall back to all charge events
        return np.asarray(h5["charge/events/data"]["id"][()], np.int64).tolist()


def _version_config(version: str, var_wrapper):
    """Map an algorithm-version string to run_pipeline_for_event kwargs.

    Naming: the DEFAULT is the validated error-matrix baseline; "v0.1" is the
    development line (spatial-guided assignment + tiebreaker) that will mature
    into the first serious release. "v2.0" is kept as a legacy alias of v0.1.
    """
    v = (version or "v1.0").strip().lower()
    if v in ("v1", "v1.0", "threshold_family", "conclusion", "default"):
        # DEFAULT v1.0-conclusion: pipeline provides clusters/images; the final
        # t0 assignment comes from assoc_threshold_2x2.threshold_family_
        # association with the benchmark-final config (2-tick family merge +
        # ND-style seed snap tol 12 + quadratic sub-tick refine + light-model
        # energy discrimination). Passed the 95% benchmark:
        # eff@5t 96.15%, eff@1t 95.91%, core sigma 2.31 ns.
        return dict(enable_region_grow=False, tiebreak_variance_model=None,
                    _threshold_family=True)
    if v in ("v1.0-legacy", "error_matrix", "errormatrix", "baseline"):
        # Legacy v1.0: error-matrix (greedy unit-variance) association; final
        # t0s come from the pipeline stages themselves.
        return dict(enable_region_grow=False, tiebreak_variance_model=None)
    if v in ("v0.1", "v0.1-rg", "v2", "v2.0", "region_grow", "regiongrow"):
        # v0.1: region-grow (tuned) + variance tiebreaker.
        return dict(enable_region_grow=True,
                    tiebreak_variance_model=var_wrapper, tie_frac=0.10)
    if v in ("v0.1-fx", "family_expand", "familyexpand"):
        # v0.1-fx (EXPERIMENTAL): cosine-free chi2 family expansion applied as a
        # post-pass on the greedy result (see family_expand_2x2.py; the 2x2
        # prototype scores at base=0 — known multi-flash caveat, documented).
        return dict(enable_region_grow=False, tiebreak_variance_model=None,
                    _family_expand=True)
    raise SystemExit(
        f"unknown --version {version!r} (use v1.0 [default], v0.1, or v0.1-fx)")


def _build_var_wrapper(device):
    """Load the 2x2 variance model wrapped for the matcher tiebreaker (v0.1)."""
    import var_model as vm           # TwoByTwo/var_model/var_model.py
    import data_2x2 as data          # matcher

    class _NewVar:
        includes_model_error = True

        def __init__(self, ckpt):
            self.model = vm.load_model(ckpt, device=device,
                                       num_channels=48, num_tpcs=8)

        def predict_variance(self, raw_sub, tbl, dead_mask=None, big_var=1e12):
            wf = data.format_from_sub(tbl, raw_sub)
            sig = vm.predict(self.model, wf, np.arange(8), batch_size=8,
                             input_scale=1e-3, device=device)
            var = np.maximum((np.asarray(sig, np.float32) * 1000.0) ** 2, 1.0)
            if dead_mask is not None:
                var[dead_mask] = big_var
            return var.astype(np.float32)

    return _NewVar(str(_VARCKPT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--files", nargs="+", required=True,
                    help="FLOW hdf5 file(s) or globs")
    ap.add_argument("--out-dir", required=True, help="per-event NPZ shard dir")
    ap.add_argument("--mode", choices=["sim", "data"], default="sim",
                    help="perceiver checkpoint to use")
    ap.add_argument("--version", default="v1.0",
                    help="v1.0 = threshold-family association, benchmark-final "
                         "config (default) | v1.0-legacy = error-matrix | "
                         "v0.1 = region-grow + tiebreaker (alias v2.0) | "
                         "v0.1-fx = chi2 family-expand (experimental)")
    ap.add_argument("--event-stride", type=int, default=1)
    ap.add_argument("--event-offset", type=int, default=0)
    ap.add_argument("--max-events-per-file", type=int, default=0,
                    help="0 = all events")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dead-yaml", default="", help="optional dead-channel yaml")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import h5py
    import light_model_2x2 as lm
    import pipeline_2x2 as pipe

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand globs
    files = []
    for f in args.files:
        files.extend(sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f])
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit("no input files found")

    # ---- one-time model load ----
    t0 = time.time()
    pulse = _PULSE_DATA if args.mode == "data" else _PULSE_SIM
    light_model = lm.load_light_model(args.mode, device=args.device,
                                      pulse_path=str(pulse))
    var_wrapper = None
    cfg = _version_config(args.version, None)
    if cfg.get("tiebreak_variance_model", "sentinel") is None and \
            args.version.strip().lower() in ("v0.1", "v0.1-rg", "v2", "v2.0",
                                             "region_grow", "regiongrow"):
        var_wrapper = _build_var_wrapper(args.device)
        cfg = _version_config(args.version, var_wrapper)
    family_expand = bool(cfg.pop("_family_expand", False))
    threshold_family = bool(cfg.pop("_threshold_family", False))
    if args.verbose:
        print(f"[w{args.event_offset}] models loaded ({args.mode}, {args.version}) "
              f"in {time.time()-t0:.1f}s", flush=True)

    n_ok = n_err = n_skip = 0
    for fp in files:
        try:
            h5 = h5py.File(fp, "r")
        except Exception as e:
            print(f"[w{args.event_offset}] cannot open {fp}: {e}", flush=True)
            continue
        if "light" not in h5:
            h5.close()
            continue
        base = os.path.basename(fp)
        ev_ids = _list_events(h5)
        if args.max_events_per_file > 0:
            ev_ids = ev_ids[:args.max_events_per_file]
        my_ids = ev_ids[args.event_offset::args.event_stride]
        for ev_id in my_ids:
            tag = f"{base}__ev{int(ev_id):05d}"
            npz_path = out_dir / f"{tag}.npz"
            if npz_path.exists():
                n_skip += 1
                continue
            te = time.time()
            try:
                r = pipe.run_pipeline_for_event(h5, int(ev_id),
                                                light_model=light_model,
                                                dead_yaml=args.dead_yaml,
                                                unit_variance=True, **cfg)
                if r is None:
                    # no light / no hits — write an empty marker so we don't retry
                    np.savez_compressed(npz_path,
                                        hit_refs=np.zeros(0, np.int64),
                                        hit_timestamps=np.zeros(0, np.float32),
                                        labels=np.zeros(0, np.int64),
                                        hitTPCid=np.zeros(0, np.int64),
                                        ev_id=np.int64(ev_id), ok=np.int64(0))
                    n_skip += 1
                    continue
                if threshold_family:
                    # v1.0-conclusion: full threshold-family association. The
                    # pipeline's own hit_timestamps are superseded -- the
                    # association re-derives every cluster's t0 from scratch
                    # (families merge at 2 ticks, snap to flash seeds within
                    # 12 ticks, quadratic sub-tick refine, light-model energy
                    # discrimination). OVERWRITES r["hit_timestamps"] in place
                    # so the NPZ format below is unchanged.
                    import assoc_threshold_2x2 as at
                    ev = r["event"]
                    at.threshold_family_association(
                        labels=r["labels"], xset=ev.xset, yset=ev.yset,
                        zset=ev.zset, Eset=ev.Eset, hitTPCid=ev.hitTPCid,
                        hit_t0=r["hit_timestamps"],
                        cluster_energies=r["cluster_energies"],
                        image_maps=r["image_maps"],
                        labels_noisy=r["labels_noisy"],
                        full_wvfm=ev.fullLightWaveform,
                        full_var=ev.fullLightVar,
                        flash_seeds=ev.flash_seeds,
                        light_model=light_model,
                        refine_quad=True,
                        group_merge_ticks=2.0,
                        group_snap_seeds=True,
                        group_snap_tol=12.0)
                    # Confidence must reflect the FINAL t0s: recompute the
                    # per-cluster matched-filter cos at the association times.
                    r["cluster_cos"] = pipe.compute_cluster_cos(
                        labels=r["labels"], image_maps=r["image_maps"],
                        hit_t0=r["hit_timestamps"],
                        full_wvfm=ev.fullLightWaveform,
                        full_var=ev.fullLightVar,
                        cluster_to_tpcs=r["cluster_to_tpcs"])
                if family_expand:
                    # v0.1-fx: chi2 family-expansion post-pass (in-place on
                    # hit_timestamps; uses the pipeline's own structures).
                    import family_expand_2x2 as fx
                    ev = r["event"]
                    fx.family_expand_association(
                        labels=r["labels"], xset=ev.xset, yset=ev.yset,
                        zset=ev.zset, Eset=ev.Eset, hitTPCid=ev.hitTPCid,
                        hit_t0=r["hit_timestamps"],
                        cluster_energies=r["cluster_energies"],
                        image_maps=r["image_maps"], base_image=r["base_image"],
                        full_wvfm=ev.fullLightWaveform,
                        full_var=ev.fullLightVar,
                        track_labels=r["track_labels"],
                        flash_seeds=ev.flash_seeds)
                hit_refs = np.asarray(r["hit_refs"], np.int64)
                hit_ts = np.asarray(r["hit_timestamps"], np.float32)
                labels = np.asarray(r.get("labels", np.full(hit_refs.size, -1)), np.int64)
                tpcid = np.asarray(r["event"].hitTPCid, np.int64)
                if labels.shape[0] != hit_refs.shape[0]:
                    labels = np.full(hit_refs.size, -1, np.int64)
                # Per-cluster matched-filter cos in [0,1]; aggregator uses this
                # to fill t_confidence. dict {cid: cos} -> two aligned arrays.
                cos_dict = r.get("cluster_cos") or {}
                cos_labels = np.asarray(sorted(cos_dict.keys()), dtype=np.int64)
                cos_values = np.asarray([cos_dict[int(l)] for l in cos_labels],
                                        dtype=np.float32)
                np.savez_compressed(npz_path,
                                    hit_refs=hit_refs,
                                    hit_timestamps=hit_ts,
                                    labels=labels,
                                    hitTPCid=tpcid,
                                    cluster_cos_labels=cos_labels,
                                    cluster_cos_values=cos_values,
                                    ev_id=np.int64(ev_id), ok=np.int64(1))
                with open(out_dir / f"{tag}.json", "w") as jf:
                    json.dump({"ok": True, "file": fp, "event_id": int(ev_id),
                               "n_hits": int(hit_refs.size),
                               "n_matched": int(np.isfinite(hit_ts).sum()),
                               "version": args.version, "mode": args.mode,
                               "elapsed_s": round(time.time() - te, 3)}, jf)
                n_ok += 1
                if args.verbose:
                    print(f"[w{args.event_offset}] {tag}: {hit_refs.size} hits "
                          f"({time.time()-te:.2f}s)", flush=True)
            except Exception as e:
                n_err += 1
                with open(out_dir / f"{tag}.json", "w") as jf:
                    json.dump({"ok": False, "file": fp, "event_id": int(ev_id),
                               "error": str(e)}, jf)
                if args.verbose:
                    print(f"[w{args.event_offset}] ERR {tag}: {e}", flush=True)
                    traceback.print_exc()
            finally:
                try:
                    pipe.data.clear_cache()
                except Exception:
                    pass
        h5.close()

    print(f"[w{args.event_offset}] done: ok={n_ok} err={n_err} skip={n_skip} "
          f"wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
