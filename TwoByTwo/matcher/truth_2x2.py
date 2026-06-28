"""
mc_truth validation for the 2x2 charge-light matcher — ND-vAlpha method.

Truth t0 is taken the SAME way as the ND vAlpha plots (`truth_plotting.py`
`extract_event_hit_energy_and_truth_t0`), NOT from detected-photon counts:

  per hit: dominant backtrack segment (highest `fraction`) -> its true `t0`
  (us, `mc_truth/segments/data['t0']`), then convert to matching ticks

      truth_t0 = (t0_us * 1000 - ref_ns) / 16            (offset 0)

  ref_ns = charge/events['unix_ts'] * 1e9  (== light utime_ms * 1e6); this is
  the detector event-start reference.  Verified on 2x2 sim: for a bright track
  (reco t0 = 513) the segment method gives 513.04.

The efficiency / residual histogram reproduce `export_valpha_final_four_plots.py`
`save_t0_hist`: an ENERGY-weighted histogram of (truth_t0 - reco_t0) in ns over
ASSIGNED hits, with the headline number = energy within +/-160 ns (= +/-10 ticks).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

import geometry_2x2 as geo

NS_PER_TICK = geo.NS_PER_TICK            # 16
TICK_SCALE = 1000.0 / NS_PER_TICK        # us -> matching ticks (62.5)


class TruthTables:
    """Cached truth datasets for one open file (segment-t0 method)."""

    def __init__(self, h5, hits_dset: str = "calib_prompt_hits"):
        self.h5 = h5
        self.backtrack = h5["mc_truth/calib_prompt_hit_backtrack/data"]
        seg = h5["mc_truth/segments/data"]
        seg_id = np.asarray(seg["segment_id"][:], dtype=np.int64)
        seg_t0 = np.asarray(seg["t0"][:], dtype=np.float64)        # us (absolute)
        order = np.argsort(seg_id, kind="mergesort")
        self.seg_id_sorted = seg_id[order]
        self.seg_t0_sorted = seg_t0[order]
        # Event-start reference: light utime_ms (per board, sub-second precise).
        # charge unix_ts is integer-second and too coarse (events are ~1.2 s
        # apart, so it is wrong by up to ~1 s).
        utime = np.asarray(h5["light/events/data"]["utime_ms"][:], dtype=np.float64)
        self.light_utime_ms = utime.reshape(utime.shape[0], -1)[:, 0]
        clr = np.asarray(h5["charge/events/ref/light/events/ref"][()], dtype=np.int64)
        self.charge2light = {int(c): int(l) for c, l in clr}

    def _ref_ns(self, ev_id: int) -> float:
        light_id = self.charge2light.get(int(ev_id))
        if light_id is None or not (0 <= light_id < self.light_utime_ms.size):
            return np.nan
        return float(self.light_utime_ms[light_id]) * 1.0e6

    def per_hit_truth_t0(self, hit_refs: np.ndarray, ev_id: int) -> np.ndarray:
        """Per-hit truth t0 (matching ticks), aligned to ``hit_refs`` order.

        NaN where the hit has no backtrack segment.
        """
        hit_refs = np.asarray(hit_refs, dtype=np.int64)
        n = hit_refs.size
        out = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return out
        # h5py fancy-index needs increasing order -> sort then invert
        order = np.argsort(hit_refs, kind="mergesort")
        bt = self.backtrack[hit_refs[order]]
        seg_ids = np.asarray(bt["segment_ids"], dtype=np.int64).reshape(n, -1)
        frac = np.asarray(bt["fraction"], dtype=np.float64).reshape(n, -1)

        valid = seg_ids >= 0
        has = valid.any(axis=1)
        safe = np.where(valid, frac, -np.inf)
        best = np.argmax(safe, axis=1)
        best_sid = seg_ids[np.arange(n), best]

        ref_ns = self._ref_ns(ev_id)
        t0_sorted = np.full(n, np.nan, dtype=np.float64)
        rows = np.flatnonzero(has)
        if rows.size and np.isfinite(ref_ns):
            pos = np.searchsorted(self.seg_id_sorted, best_sid[rows])
            pos = np.clip(pos, 0, self.seg_id_sorted.size - 1)
            ok = self.seg_id_sorted[pos] == best_sid[rows]
            good = rows[ok]
            t0_us = self.seg_t0_sorted[pos[ok]]
            t0_sorted[good] = (t0_us * 1000.0 - ref_ns) / NS_PER_TICK
        out[order] = t0_sorted
        return out


# ---------------------------------------------------------------------------
# Energy-weighted residual summary (mirrors export_valpha_final_four_plots)
# ---------------------------------------------------------------------------
def _weighted_quantile(x, w, q):
    if x.size == 0:
        return float("nan")
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    cdf = np.cumsum(ws)
    if float(cdf[-1]) <= 0:
        return float("nan")
    idx = int(np.searchsorted(cdf / cdf[-1], float(q), side="left"))
    return float(xs[min(idx, xs.size - 1)])


def residual_summary(reco_t0, truth_t0, energy, *, window_ns: float = 160.0):
    """Energy-weighted (truth - reco) residual stats over assigned hits.

    'assigned' = finite truth & finite reco & reco>=0 & energy>0.
    Returns residuals in ns + weights + median/std + within-window fraction.
    """
    reco = np.asarray(reco_t0, np.float64)
    truth = np.asarray(truth_t0, np.float64)
    e = np.asarray(energy, np.float64)
    assigned = (np.isfinite(truth) & np.isfinite(reco) & (reco >= 0)
                & np.isfinite(e) & (e > 0))
    diff_ns = (truth[assigned] - reco[assigned]) * NS_PER_TICK
    w = e[assigned]
    wsum = float(w.sum())
    if diff_ns.size == 0 or wsum <= 0:
        return {"diff_ns": diff_ns, "weights": w, "n": 0, "assigned_energy": 0.0,
                "median_ns": np.nan, "std_ns": np.nan, "within_pct": np.nan,
                "window_ns": window_ns}
    median = _weighted_quantile(diff_ns, w, 0.5)
    mean = float((w * diff_ns).sum() / wsum)
    std = float(np.sqrt((w * (diff_ns - mean) ** 2).sum() / wsum))
    within = 100.0 * float(w[np.abs(diff_ns) <= window_ns].sum()) / wsum
    return {"diff_ns": diff_ns, "weights": w, "n": int(diff_ns.size),
            "assigned_energy": wsum, "median_ns": median, "std_ns": std,
            "within_pct": within, "window_ns": float(window_ns)}


def save_t0_residual_hist(diff_ns, weights, *, output_path, window_ns=160.0,
                          xlim_ns=(-200.0, 200.0), bin_width_ns=5.0,
                          figsize=(10.8, 5.4),
                          title="Energy-Weighted Truth t0 - Reco t0 (±10 ticks)"):
    """Reproduce export_valpha_final_four_plots.save_t0_hist (ppt-wide variant)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff_ns = np.asarray(diff_ns, np.float64)
    w = np.asarray(weights, np.float64)
    m = np.isfinite(diff_ns) & np.isfinite(w) & (w > 0)
    diff_ns, w = diff_ns[m], w[m]
    wsum = float(w.sum())
    median = _weighted_quantile(diff_ns, w, 0.5)
    mean = float((w * diff_ns).sum() / wsum)
    std = float(np.sqrt((w * (diff_ns - mean) ** 2).sum() / wsum))
    within = 100.0 * float(w[np.abs(diff_ns) <= window_ns].sum()) / max(wsum, 1e-12)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    bins = np.arange(xlim_ns[0], xlim_ns[1] + bin_width_ns, bin_width_ns)
    counts, edges = np.histogram(diff_ns, bins=bins, weights=w / wsum)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=edges[1] - edges[0], color="#4C78A8",
           edgecolor="white", linewidth=0.7, alpha=0.95)
    ax.axvline(0.0, color="black", ls="--", lw=1.2)
    ax.axvline(median, color="#D62728", ls="--", lw=1.8, label=f"Median = {median:.2f} ns")
    ax.set_xlim(*xlim_ns)
    ax.set_xlabel("Truth t0 - Reco t0 [ns]", fontsize=11)
    ax.set_ylabel("Proportion of assigned energy", fontsize=11)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(alpha=0.22, ls=":")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stats = (f"Assigned energy: {wsum:.1f} MeV\n"
             f"Weighted median: {median:.2f} ns\n"
             f"Weighted std:    {std:.2f} ns\n"
             f"Energy within ±{window_ns:.0f} ns: {within:.1f}%")
    ax.text(0.985, 0.83, stats, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                  edgecolor="0.8", alpha=0.92))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return {"assigned_energy_mev": wsum, "median_ns": median, "std_ns": std,
            "within_window_percent": within, "window_ns": window_ns}


# ---------------------------------------------------------------------------
# Per-cluster energy-binned report (for quick CLI feedback)
# ---------------------------------------------------------------------------
def evaluate_clusters(truth: TruthTables, *, ev_id, hit_refs, labels, hit_t0,
                      Eset, hitTPCid, tolerance_ticks: float = 10.0,
                      **_ignored) -> Dict[str, Any]:
    """Per-cluster + per-hit truth comparison.  Cluster truth = energy-weighted
    mean of its hits' truth t0.  Adds per-hit arrays for the residual histogram.
    """
    labels = np.asarray(labels, np.int64)
    hit_t0 = np.asarray(hit_t0, np.float64)
    Eset = np.asarray(Eset, np.float64)
    hit_refs = np.asarray(hit_refs, np.int64)
    truth_hit = truth.per_hit_truth_t0(hit_refs, ev_id)         # per-hit ticks

    rows = []
    for cid in np.unique(labels):
        if cid < 0:
            continue
        m = labels == cid
        rec = hit_t0[m]
        fin = np.isfinite(rec) & (rec >= 0)
        reco_t0 = float(np.median(rec[fin])) if fin.any() else float("nan")
        th = truth_hit[m]
        we = Eset[m]
        tmask = np.isfinite(th)
        if tmask.any() and we[tmask].sum() > 0:
            truth_t0 = float((th[tmask] * we[tmask]).sum() / we[tmask].sum())
        else:
            truth_t0 = float("nan")
        diff = (reco_t0 - truth_t0) if (np.isfinite(reco_t0) and np.isfinite(truth_t0)) else float("nan")
        rows.append({"clusterid": int(cid), "n_hits": int(m.sum()),
                     "energy_mev": float(we.sum()), "reco_t0": reco_t0,
                     "truth_t0": truth_t0, "diff_ticks": diff,
                     "matched": bool(np.isfinite(reco_t0)),
                     "has_truth": bool(np.isfinite(truth_t0)),
                     "correct": bool(np.isfinite(diff) and abs(diff) <= tolerance_ticks)})
    return {"rows": rows, "summary": _summarize(rows, tolerance_ticks),
            "tolerance_ticks": float(tolerance_ticks),
            "per_hit": {"truth_t0": truth_hit, "reco_t0": hit_t0,
                        "energy": Eset}}


def _summarize(rows, tol):
    have = [r for r in rows if r["has_truth"]]

    def frac(sub, key=None):
        if not sub:
            return float("nan")
        c = np.array([1.0 if r["correct"] else 0.0 for r in sub])
        if key is None:
            return float(c.mean())
        w = np.array([r[key] for r in sub], float)
        return float((w * c).sum() / w.sum()) if w.sum() > 0 else float("nan")

    edges = [0, 1, 5, 20, 1e9]
    names = ["<1MeV", "1-5MeV", "5-20MeV", ">20MeV"]
    bins = {}
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        sub = [r for r in have if lo <= r["energy_mev"] < hi]
        bins[nm] = {"n_clusters": len(sub), "success_count": frac(sub),
                    "success_energy": frac(sub, "energy_mev"),
                    "median_abs_diff": (float(np.median([abs(r["diff_ticks"]) for r in sub
                                                         if np.isfinite(r["diff_ticks"])]))
                                        if sub else float("nan"))}
    return {"n_clusters_total": len(rows), "n_clusters_with_truth": len(have),
            "success_count_all": frac(have), "success_energy_all": frac(have, "energy_mev"),
            "by_energy": bins}


def format_report(eval_out, *, max_rows=30):
    s = eval_out["summary"]
    tol = eval_out["tolerance_ticks"]
    L = [f"=== mc_truth (segment-t0) cluster report (|reco-truth| <= {tol:.0f} ticks) ===",
         f"clusters: {s['n_clusters_total']} total, {s['n_clusters_with_truth']} with truth",
         f"success (count) : {s['success_count_all']:.3f}   success (energy): {s['success_energy_all']:.3f}",
         f"  {'bin':>9} {'N':>5} {'succ(N)':>8} {'succ(E)':>8} {'medD':>7}"]
    for nm, b in s["by_energy"].items():
        L.append(f"  {nm:>9} {b['n_clusters']:>5} {b['success_count']:>8.3f} "
                 f"{b['success_energy']:>8.3f} {b['median_abs_diff']:>7.2f}")
    return "\n".join(L)


__all__ = ["TruthTables", "residual_summary", "save_t0_residual_hist",
           "evaluate_clusters", "format_report"]
