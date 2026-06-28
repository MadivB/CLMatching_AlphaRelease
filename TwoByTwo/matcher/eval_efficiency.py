#!/usr/bin/env python3
"""
Efficiency + t0-residual histogram for the 2x2 charge-light matcher, using the
SAME truth method and plot as the ND vAlpha
(`export_valpha_final_four_plots.py` / `truth_plotting.py`):

  * per-hit truth t0 from the dominant backtrack segment's true `t0`
    (NOT detected-photon counts),
  * energy-weighted histogram of (truth_t0 - reco_t0) in ns over ASSIGNED hits,
  * headline efficiency = energy within +/-160 ns (= +/-10 ticks).

Output: residual PNG + a per-hit .npz for re-plotting.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py

import geometry_2x2 as geo
import pipeline_2x2 as pipe
import light_model_2x2 as lm
import truth_2x2 as truth

NS = geo.NS_PER_TICK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5", required=True)
    ap.add_argument("--max-events", type=int, default=60)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window-ns", type=float, default=160.0, help="±10 ticks")
    ap.add_argument("--dead-yaml", default="")
    ap.add_argument("--no-phase25", action="store_true")
    ap.add_argument("--no-varmodel", action="store_true",
                    help="use the self-contained noise model instead of the learned one")
    ap.add_argument("--no-unit-variance", action="store_true",
                    help="disable the default unit variance (use self-contained / learned)")
    ap.add_argument("--out-prefix", default="qlm2x2_eff")
    args = ap.parse_args()

    model = lm.load_light_model(args.mode, device=args.device)
    varmodel = None
    if not args.no_varmodel:
        import variance_model_2x2 as vm
        varmodel = vm.load_variance_model(device=args.device)
    h5 = h5py.File(args.in_h5, "r")
    tt = truth.TruthTables(h5)
    ref = np.asarray(h5["charge/events/ref/light/events/ref"][()], np.int64)
    ev_ids = np.unique(ref[:, 0])[: args.max_events].tolist()
    print(f"[eval] {len(ev_ids)} events, mode={args.mode}")

    truth_all, reco_all, ener_all, tpc_all = [], [], [], []
    n_done = 0
    for ev in ev_ids:
        res = pipe.run_pipeline_for_event(h5, int(ev), light_model=model,
                                          dead_yaml=args.dead_yaml,
                                          variance_model=varmodel,
                                          unit_variance=not args.no_unit_variance,
                                          enable_phase25=not args.no_phase25)
        if res is None:
            continue
        n_done += 1
        th = tt.per_hit_truth_t0(res["hit_refs"], int(ev))       # per-hit ticks
        truth_all.append(th)
        reco_all.append(np.asarray(res["hit_timestamps"], np.float64))
        ener_all.append(np.asarray(res["event"].Eset, np.float64))
        tpc_all.append(np.asarray(res["event"].hitTPCid, np.int64))

    truth_t0 = np.concatenate(truth_all)
    reco_t0 = np.concatenate(reco_all)
    energy = np.concatenate(ener_all)
    tpc = np.concatenate(tpc_all)
    np.savez(args.out_prefix + "_hits.npz", truth_t0=truth_t0, reco_t0=reco_t0,
             energy=energy, tpc=tpc)

    summ = truth.residual_summary(reco_t0, truth_t0, energy, window_ns=args.window_ns)
    tol_ticks = args.window_ns / NS
    print(f"\n===== 2x2 charge-light matching efficiency "
          f"(energy within ±{args.window_ns:.0f} ns = ±{tol_ticks:.0f} ticks) =====")
    print(f"  events processed   : {n_done}")
    print(f"  assigned hits      : {summ['n']}   assigned energy: {summ['assigned_energy']:.1f} MeV")
    print(f"  TOTAL EFFICIENCY   : {summ['within_pct']:.2f}%   (energy-weighted, ND method)")
    print(f"  residual median/std: {summ['median_ns']:+.2f} / {summ['std_ns']:.2f} ns")

    # efficiency vs energy (energy-weighted within each bin)
    print("\n  efficiency vs hit energy (energy-weighted):")
    assigned = (np.isfinite(truth_t0) & np.isfinite(reco_t0) & (reco_t0 >= 0)
                & np.isfinite(energy) & (energy > 0))
    d = np.abs(truth_t0 - reco_t0) * NS
    for lo, hi, nm in [(0, 1, "<1"), (1, 5, "1-5"), (5, 20, "5-20"), (20, 1e9, ">20")]:
        mb = assigned & (energy >= lo) & (energy < hi)
        if not mb.any():
            continue
        w = energy[mb]
        eff = 100.0 * w[d[mb] <= args.window_ns].sum() / w.sum()
        print(f"    {nm:>5} MeV: hits={int(mb.sum()):6d}  energy-eff={eff:6.2f}%")

    png = args.out_prefix + "_t0_residual.png"
    truth.save_t0_residual_hist(summ["diff_ns"], summ["weights"], output_path=png,
                                window_ns=args.window_ns)
    print(f"\n[plot] {png}")


if __name__ == "__main__":
    main()
