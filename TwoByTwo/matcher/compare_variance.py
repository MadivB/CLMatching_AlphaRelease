#!/usr/bin/env python3
"""
Compare t0-residual / efficiency across variance choices for the chi2 denominator.

Loads the per-hit .npz dumps written by eval_efficiency.py (truth_t0, reco_t0,
energy in matching ticks) for several variance configurations and:
  * tabulates the energy-weighted efficiency (energy within +/-160 ns) + the
    residual median / RMS / 68% half-width,
  * overlays the energy-weighted t0-residual distributions on one figure
    (zoom on the +/-200 ns core) so the shapes are directly comparable.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = 16.0
WINDOW = 160.0


def stats(npz):
    d = np.load(npz)
    truth, reco, E = d["truth_t0"], d["reco_t0"], d["energy"]
    asg = np.isfinite(truth) & np.isfinite(reco) & (reco >= 0) & np.isfinite(E) & (E > 0)
    res = (truth[asg] - reco[asg]) * NS
    w = E[asg]
    wsum = w.sum()
    eff = 100.0 * w[np.abs(res) <= WINDOW].sum() / wsum
    # energy-weighted median + 68% half-width
    o = np.argsort(res)
    cw = np.cumsum(w[o]) / wsum
    med = float(res[o][np.searchsorted(cw, 0.5)])
    p16 = float(res[o][np.searchsorted(cw, 0.16)])
    p84 = float(res[o][np.searchsorted(cw, 0.84)])
    rms = float(np.sqrt((w * (res - (w * res).sum() / wsum) ** 2).sum() / wsum))
    return dict(res=res, w=w, eff=eff, med=med, half=(p84 - p16) / 2, rms=rms,
                n=int(asg.sum()), E=float(wsum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding the *_hits.npz dumps")
    ap.add_argument("--out", default="variance_comparison.png")
    args = ap.parse_args()

    configs = [
        ("unit (== ND vAlpha)", "qlm_unit_hits.npz", "#000000"),
        ("self-contained (current)", "qlm_selfvar_hits.npz", "#1f77b4"),
        ("learned var_prediction", "qlm_varmodel_hits.npz", "#d62728"),
        ("total (learned+model err)", "qlm_totvar_hits.npz", "#2ca02c"),
    ]
    have = [(n, os.path.join(args.dir, f), c) for n, f, c in configs
            if os.path.exists(os.path.join(args.dir, f))]

    print(f"{'variance model':<28} {'eff%(±160ns)':>13} {'median(ns)':>11} "
          f"{'68% half(ns)':>13} {'RMS(ns)':>9}")
    print("-" * 78)
    results = []
    for name, path, color in have:
        s = stats(path)
        results.append((name, s, color))
        print(f"{name:<28} {s['eff']:>12.2f}  {s['med']:>10.2f} "
              f"{s['half']:>12.2f} {s['rms']:>8.1f}")

    fig, (axc, axf) = plt.subplots(1, 2, figsize=(14, 5))
    bins_core = np.arange(-200, 205, 5.0)
    bins_full = np.linspace(-800, 800, 160)
    for name, s, color in results:
        wn = s["w"] / s["w"].sum()
        axc.hist(s["res"], bins=bins_core, weights=wn, histtype="step", lw=2,
                 color=color, label=f"{name}: {s['eff']:.1f}%")
        axf.hist(s["res"], bins=bins_full, weights=wn, histtype="step", lw=1.6,
                 color=color)
    for ax in (axc, axf):
        ax.axvspan(-WINDOW, WINDOW, color="green", alpha=0.06)
        ax.axvline(0, color="0.5", ls=":", lw=1)
        ax.set_xlabel("Truth t0 - Reco t0 [ns]")
        ax.set_ylabel("proportion of assigned energy")
    axc.set_title("core (±200 ns)")
    axc.legend(fontsize=9, title="χ² variance: efficiency")
    axf.set_yscale("log")
    axf.set_title("full range (log)")
    fig.suptitle("2x2 t0-residual vs χ² variance model (energy-weighted)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\n[plot] {args.out}")


if __name__ == "__main__":
    main()
