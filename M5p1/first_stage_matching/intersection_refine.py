"""Intersection refinement: hit-level relabeling at track/cluster crossings.

Runs immediately after global track clustering (opt-in via
``ClusteringConfig.intersection_refine_enable``).  Two cases:

(a) track x hit-cloud (blob cluster or shower): the track keeps an
    approximately uniform energy linear density (MeV/cm) along its axis;
    inside the geometric contact window it may exceed its own baseline by a
    soft factor, and only the excess above that cap is shed to the cloud.
    A per-bin hard floor at the baseline plus a core-radius protection mean
    the track can never be severed or dropped below its own dE/dx level.

(b) track x track (different labels): hits in the crossing window are first
    geometrically pinned (stage-2.5-style scoring, large margin), then the
    remaining truly-ambiguous hits are apportioned so both tracks continue
    through the crossing at the same level relative to their own baselines.
    A donor-floor keeps every track at >= its baseline continuation and a
    new-gap veto rolls back any pair that would sever a track.

Structural guarantees (by construction, not tuning):
- hits move only between EXISTING labels; noise (-1) is never touched;
- no label is created, renumbered, emptied, or retyped;
- no hit outside a detected intersection window can ever be relabeled;
- a hit moves at most once per event (no oscillation);
- fully deterministic (no RNG; all ties broken by hit index);
- baselines are measured on bins outside the union of ALL candidate windows,
  so they are invariant under the entire pass (no self-referential drift);
- flag off -> this module is never imported into the hot path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["refine_intersections"]


# ---------------------------------------------------------------------------
# Geometry helpers


def _fixed_sign(u: np.ndarray) -> np.ndarray:
    """Deterministic direction sign: largest-|component| axis made positive."""
    k = int(np.argmax(np.abs(u)))
    return -u if u[k] < 0 else u


def _svd_axis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    u = vt[0]
    u = u / (np.linalg.norm(u) + 1e-12)
    return c, _fixed_sign(u)


def _project(pts: np.ndarray, p: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dif = pts - p
    t = dif @ u
    perp2 = np.sum(dif * dif, axis=1) - t * t
    return t, np.sqrt(np.maximum(perp2, 0.0))


def _segment_closest_approach(
    p1: np.ndarray, u1: np.ndarray, lo1: float, hi1: float,
    p2: np.ndarray, u2: np.ndarray, lo2: float, hi2: float,
) -> tuple[float, float, float]:
    """Closest approach of two finite 3D segments, parameterized by the axial
    coordinate of each track's own model.  Returns (t1*, t2*, distance)."""
    a1 = p1 + lo1 * u1
    a2 = p2 + lo2 * u2
    d1 = (hi1 - lo1) * u1
    d2 = (hi2 - lo2) * u2
    r = a1 - a2
    aa = float(d1 @ d1)
    bb = float(d1 @ d2)
    cc = float(d2 @ d2)
    dd = float(d1 @ r)
    ee = float(d2 @ r)
    den = aa * cc - bb * bb
    if den > 1e-12:
        s = np.clip((bb * ee - cc * dd) / den, 0.0, 1.0)
    else:
        s = 0.0
    t = (bb * s + ee) / cc if cc > 1e-12 else 0.0
    t = float(np.clip(t, 0.0, 1.0))
    # re-clamp s given clamped t
    if aa > 1e-12:
        s = float(np.clip((bb * t - dd) / aa, 0.0, 1.0))
    q1 = a1 + s * d1
    q2 = a2 + t * d2
    dist = float(np.linalg.norm(q1 - q2))
    return lo1 + s * (hi1 - lo1), lo2 + t * (hi2 - lo2), dist


def _max_internal_gap(t_sorted: np.ndarray) -> float:
    if t_sorted.size < 2:
        return 0.0
    return float(np.max(np.diff(t_sorted)))


# ---------------------------------------------------------------------------
# Per-(track,TPC) model


@dataclass
class _TrackModel:
    label: int
    tpc: int
    idx: np.ndarray          # pristine member hit indices (this TPC)
    p: np.ndarray            # axis point
    u: np.ndarray            # unit direction (sign-fixed)
    tmin: float
    tmax: float
    length: float
    r_base: float
    w_score: float
    r_core: float
    bin_edges: np.ndarray    # nbins+1 edges over [tmin, tmax]
    bin_w: np.ndarray        # per-bin widths
    rho: np.ndarray          # pristine per-bin MeV/cm
    windows: list[tuple[float, float]] = field(default_factory=list)  # all candidate windows
    rho0: float = np.nan
    sigma: float = np.nan
    baseline_valid: bool = False

    def t_perp(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _project(pts, self.p, self.u)

    def bin_of(self, t: np.ndarray) -> np.ndarray:
        nb = self.bin_w.size
        j = np.searchsorted(self.bin_edges, t, side="right") - 1
        return np.clip(j, 0, nb - 1)


def _build_bins(t: np.ndarray, e_pos: np.ndarray, tmin: float, tmax: float, bin_cm: float):
    length = tmax - tmin
    nbins = max(1, int(np.ceil(length / bin_cm)))
    edges = tmin + bin_cm * np.arange(nbins + 1, dtype=np.float64)
    edges[-1] = tmax
    widths = np.diff(edges)
    if nbins >= 2 and widths[-1] < 0.5 * bin_cm:
        edges = np.delete(edges, nbins - 1)
        nbins -= 1
        widths = np.diff(edges)
    j = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1)
    esum = np.bincount(j, weights=e_pos, minlength=nbins)
    rho = esum / np.maximum(widths, 1e-9)
    return edges, widths, rho


# ---------------------------------------------------------------------------
# Main entry


def refine_intersections(
    *,
    labels_global: np.ndarray,
    split_index: int,
    label_info: dict[int, dict[str, Any]],
    debug: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    io_group: np.ndarray,
    energy: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (possibly-refined labels array, stats dict).  Never mutates
    the input labels array; on any internal invariant failure the ORIGINAL
    labels are returned with the failure recorded in stats."""
    t_start = time.perf_counter()
    cfg = config
    stats: dict[str, Any] = {
        "enabled": True,
        "n_models": 0,
        "n_pairs_a_found": 0, "n_pairs_b_found": 0,
        "n_pairs_a_moved": 0, "n_pairs_b_moved": 0,
        "n_hits_moved_a": 0, "n_hits_moved_b": 0,
        "energy_moved_a_mev": 0.0, "energy_moved_b_mev": 0.0,
        "vetoes": {}, "pairs": [],
        "truncated_pairs": 0, "commit_ok": True,
    }

    def _veto(reason: str) -> None:
        stats["vetoes"][reason] = stats["vetoes"].get(reason, 0) + 1

    labels_old = np.asarray(labels_global)
    if not label_info or not debug:
        stats["elapsed_s"] = time.perf_counter() - t_start
        return labels_old, stats

    pts = np.column_stack([
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    ])
    tpc_ids = (np.asarray(io_group, dtype=np.int64) - 1) // 2
    e_pos = np.clip(np.asarray(energy, dtype=np.float64), 0.0, None)
    labels = labels_old.copy()
    n_hits = labels.size

    # ---- Phase 1: per-(track,TPC) models -------------------------------
    models: list[_TrackModel] = []
    for lab in sorted(int(k) for k in label_info):
        info = label_info[lab]
        if str(info.get("type", "")) != "track" or int(info.get("n_tracks", 0)) != 1:
            continue
        members = np.flatnonzero(labels == lab)
        if members.size == 0:
            continue
        for tpc in sorted(int(v) for v in np.unique(tpc_ids[members])):
            idx = members[tpc_ids[members] == tpc]
            if idx.size < int(cfg.ir_min_track_hits):
                continue
            p, u = _svd_axis(pts[idx])
            t, perp = _project(pts[idx], p, u)
            tmin, tmax = float(t.min()), float(t.max())
            length = tmax - tmin
            if length < float(cfg.ir_min_track_len_cm):
                continue
            r_base = max(1.2, 1.2, float(np.percentile(perp, 90.0)))
            w_score = max(float(np.quantile(perp, 0.68)), 0.55)
            r_core = max(float(cfg.ir_core_min_cm), float(cfg.ir_core_frac) * r_base)
            edges, widths, rho = _build_bins(t, e_pos[idx], tmin, tmax, float(cfg.ir_bin_cm))
            models.append(_TrackModel(
                label=lab, tpc=tpc, idx=idx, p=p, u=u,
                tmin=tmin, tmax=tmax, length=length,
                r_base=r_base, w_score=w_score, r_core=r_core,
                bin_edges=edges, bin_w=widths, rho=rho,
            ))
    stats["n_models"] = len(models)
    if not models:
        stats["elapsed_s"] = time.perf_counter() - t_start
        return labels_old, stats

    models_by_tpc: dict[int, list[_TrackModel]] = {}
    for m in models:
        models_by_tpc.setdefault(m.tpc, []).append(m)

    recipient_types = set(cfg.ir_recipient_types)
    recipient_labels = [
        int(lab) for lab, info in label_info.items()
        if str(info.get("type", "")) in recipient_types
    ]

    # ---- Phase 2: window detection (pristine labels) --------------------
    margin = float(cfg.ir_window_margin_cm)
    pairs_a: list[dict[str, Any]] = []   # track x cloud
    pairs_b: list[dict[str, Any]] = []   # track x track

    # (a) track x cloud
    if recipient_labels:
        recipient_set = np.asarray(sorted(recipient_labels), dtype=labels.dtype)
        is_recipient = np.isin(labels, recipient_set)
        for m in models:
            cand = np.flatnonzero(is_recipient & (tpc_ids == m.tpc))
            if cand.size == 0:
                continue
            t_c, perp_c = m.t_perp(pts[cand])
            near = (perp_c <= 1.5 * m.r_base) & (t_c >= m.tmin) & (t_c <= m.tmax)
            if not np.any(near):
                continue
            near_idx = cand[near]
            near_t = t_c[near]
            for b_lab in sorted(int(v) for v in np.unique(labels[near_idx])):
                if b_lab == m.label:
                    continue
                sel = labels[near_idx] == b_lab
                if int(np.count_nonzero(sel)) < int(cfg.ir_min_recipient_hits):
                    _veto("a_too_few_contact_hits")
                    continue
                ts = near_t[sel]
                w_lo = max(m.tmin, float(ts.min()) - margin)
                w_hi = min(m.tmax, float(ts.max()) + margin)
                # Register the contact stretch for Phase-3 baseline masking
                # even when the pair itself is vetoed: an absorbed parallel
                # cloud must never inflate the track's own baseline.
                m.windows.append((w_lo, w_hi))
                if (w_hi - w_lo) > float(cfg.ir_max_window_frac) * m.length:
                    _veto("a_window_too_long")
                    continue
                pairs_a.append({"model": m, "recipient": b_lab, "w": (w_lo, w_hi)})
                stats["n_pairs_a_found"] += 1

    # (b) track x track (same TPC, different labels)
    theta_min = np.deg2rad(float(cfg.ir_theta_min_deg))
    for tpc, ms in sorted(models_by_tpc.items()):
        for i in range(len(ms)):
            for k in range(i + 1, len(ms)):
                m1, m2 = ms[i], ms[k]
                if m1.label == m2.label:
                    continue
                cosang = abs(float(m1.u @ m2.u))
                theta = float(np.arccos(np.clip(cosang, -1.0, 1.0)))
                if theta < theta_min:
                    _veto("b_angle_too_shallow")
                    continue
                t1s, t2s, d12 = _segment_closest_approach(
                    m1.p, m1.u, m1.tmin, m1.tmax, m2.p, m2.u, m2.tmin, m2.tmax)
                if d12 > float(cfg.ir_cross_dist_scale) * max(1.5 * m1.r_base, 1.5 * m2.r_base):
                    _veto("b_no_contact")
                    continue
                if not (m1.tmin - margin <= t1s <= m1.tmax + margin):
                    _veto("b_crossing_outside")
                    continue
                if not (m2.tmin - margin <= t2s <= m2.tmax + margin):
                    _veto("b_crossing_outside")
                    continue
                # Endpoint/Bragg guard (mirror of case (a)): a track ENDING
                # on/near another track has a legitimate dE/dx rise there --
                # apportioning would strip its Bragg peak, so skip the pair.
                guard = float(cfg.ir_endpoint_guard_cm)
                if (t1s - m1.tmin <= guard or m1.tmax - t1s <= guard
                        or t2s - m2.tmin <= guard or m2.tmax - t2s <= guard):
                    _veto("b_endpoint_guard")
                    continue
                h = float(np.clip((m1.r_base + m2.r_base) / max(np.sin(theta), 1e-3),
                                  float(cfg.ir_halflen_min_cm), float(cfg.ir_halflen_max_cm)))
                w1 = (max(m1.tmin, t1s - h), min(m1.tmax, t1s + h))
                w2 = (max(m2.tmin, t2s - h), min(m2.tmax, t2s + h))
                m1.windows.append(w1)
                m2.windows.append(w2)
                pairs_b.append({"m1": m1, "m2": m2, "w1": w1, "w2": w2,
                                "t1s": t1s, "t2s": t2s})
                stats["n_pairs_b_found"] += 1

    if not pairs_a and not pairs_b:
        stats["elapsed_s"] = time.perf_counter() - t_start
        return labels_old, stats

    # ---- Phase 3: move-invariant baselines ------------------------------
    for m in models:
        nb = m.bin_w.size
        masked = np.zeros(nb, dtype=bool)
        for (w_lo, w_hi) in m.windows:
            j_lo = max(0, int(m.bin_of(np.asarray([w_lo]))[0]) - 1)
            j_hi = min(nb - 1, int(m.bin_of(np.asarray([w_hi]))[0]) + 1)
            masked[j_lo:j_hi + 1] = True
        clean = m.rho[~masked]
        if clean.size >= int(cfg.ir_min_baseline_bins):
            rho0 = float(np.median(clean))
            if rho0 >= float(cfg.ir_min_baseline_mevcm):
                m.rho0 = rho0
                m.sigma = float(1.4826 * np.median(np.abs(clean - rho0)))
                m.baseline_valid = True

    # ---- Phase 4: sequential pair processing ----------------------------
    pairs_b.sort(key=lambda p: (p["m1"].tpc, min(p["m1"].label, p["m2"].label),
                                max(p["m1"].label, p["m2"].label), p["t1s"]))
    pairs_a.sort(key=lambda p: (p["model"].tpc, p["model"].label, p["recipient"]))
    all_pairs = [("b", p) for p in pairs_b] + [("a", p) for p in pairs_a]
    if len(all_pairs) > int(cfg.ir_max_pairs):
        stats["truncated_pairs"] = len(all_pairs) - int(cfg.ir_max_pairs)
        all_pairs = all_pairs[: int(cfg.ir_max_pairs)]

    moved_once = np.zeros(n_hits, dtype=bool)

    def _members(lab: int, tpc: int) -> np.ndarray:
        return np.flatnonzero((labels == lab) & (tpc_ids == tpc))

    def _context_score_model(m: _TrackModel, w: tuple[float, float]):
        """Score support fit on OUT-OF-WINDOW members (context fit); falls
        back to all current members when too few remain outside."""
        mem = _members(m.label, m.tpc)
        if mem.size == 0:
            return None
        t, perp = m.t_perp(pts[mem])
        out = (t < w[0]) | (t > w[1])
        use = mem[out] if int(np.count_nonzero(out)) >= 10 else mem
        perp_use = perp[out] if int(np.count_nonzero(out)) >= 10 else perp
        w_ctx = max(float(np.quantile(perp_use, 0.68)), 0.55) if perp_use.size else m.w_score
        tree = cKDTree(pts[use])
        return {"tree": tree, "w": w_ctx}

    def _score(m: _TrackModel, ctx: dict[str, Any], hit_idx: np.ndarray) -> np.ndarray:
        t, perp = m.t_perp(pts[hit_idx])
        nearest, _ = ctx["tree"].query(pts[hit_idx], k=1)
        gap = np.maximum(m.tmin - t - 10.0, 0.0) + np.maximum(t - m.tmax - 10.0, 0.0)
        return 0.70 * perp / ctx["w"] + 0.20 * np.asarray(nearest) / 2.2 + 0.10 * gap / 6.0

    for kind, pair in all_pairs:
        if kind == "b":
            m1, m2 = pair["m1"], pair["m2"]
            w1, w2 = pair["w1"], pair["w2"]
            if not (m1.baseline_valid and m2.baseline_valid):
                _veto("b_baseline_invalid")
                continue

            mem1 = _members(m1.label, m1.tpc)
            mem2 = _members(m2.label, m2.tpc)
            if mem1.size == 0 or mem2.size == 0:
                _veto("b_empty_side")
                continue

            # snapshot for gap veto / rollback
            pre_gap1 = _max_internal_gap(np.sort(m1.t_perp(pts[mem1])[0]))
            pre_gap2 = _max_internal_gap(np.sort(m2.t_perp(pts[mem2])[0]))
            snapshot: list[tuple[int, int]] = []   # (hit, old_label)

            ctx1 = _context_score_model(m1, w1)
            ctx2 = _context_score_model(m2, w2)
            if ctx1 is None or ctx2 is None:
                _veto("b_empty_side")
                continue

            def _pool_side(m_own, m_oth, w_own, mem_own):
                t_own, perp_own = m_own.t_perp(pts[mem_own])
                _, perp_oth = m_oth.t_perp(pts[mem_own])
                elig = (
                    (t_own >= w_own[0]) & (t_own <= w_own[1])
                    & (perp_oth <= 1.5 * m_oth.r_base)
                    & (e_pos[mem_own] > 0.0)
                    & ~moved_once[mem_own]
                    & ~((perp_own <= m_own.r_core) & (perp_oth > perp_own))
                )
                return mem_own[elig]

            pool1 = _pool_side(m1, m2, w1, mem1)
            pool2 = _pool_side(m2, m1, w2, mem2)
            if pool1.size + pool2.size < int(cfg.ir_min_pool_hits):
                _veto("b_pool_too_small")
                continue

            pool = np.concatenate([pool1, pool2])
            pool.sort()
            s1 = _score(m1, ctx1, pool)
            s2 = _score(m2, ctx2, pool)
            own_is_1 = labels[pool] == m1.label
            s_own = np.where(own_is_1, s1, s2)
            s_oth = np.where(own_is_1, s2, s1)

            moved_hits_e = 0.0

            # b1: geometric pinning (blatant misassignments only)
            pin = s_oth + float(cfg.ir_clear_margin) < s_own
            for h_local in np.flatnonzero(pin):
                h = int(pool[h_local])
                new_lab = m2.label if labels[h] == m1.label else m1.label
                snapshot.append((h, int(labels[h])))
                labels[h] = new_lab
                moved_once[h] = True
                moved_hits_e += float(e_pos[h])

            # b2: density equalization on the truly-ambiguous remainder
            amb_mask = (np.abs(s1 - s2) <= float(cfg.ir_clear_margin)) & ~pin & ~moved_once[pool]
            amb = pool[amb_mask]
            lean = (s2 - s1)[amb_mask]  # negative -> prefers m2
            # In-window membership on EACH axis: a moved hit only counts
            # toward the recipient's window energy if it actually lands
            # inside the recipient's own-axis window (keeps the incremental
            # e_cur updates exactly consistent with _window_energy, so the
            # donor floor is always evaluated on true in-window energy).
            if amb.size:
                t1_amb, _ = m1.t_perp(pts[amb])
                t2_amb, _ = m2.t_perp(pts[amb])
                in_w1_amb = (t1_amb >= w1[0]) & (t1_amb <= w1[1])
                in_w2_amb = (t2_amb >= w2[0]) & (t2_amb <= w2[1])
            else:
                in_w1_amb = in_w2_amb = np.zeros(0, dtype=bool)

            def _window_energy(m, w):
                mem = _members(m.label, m.tpc)
                if mem.size == 0:
                    return 0.0
                t, _ = m.t_perp(pts[mem])
                return float(np.sum(e_pos[mem[(t >= w[0]) & (t <= w[1])]]))

            e_exp1 = m1.rho0 * max(w1[1] - w1[0], 1e-9)
            e_exp2 = m2.rho0 * max(w2[1] - w2[0], 1e-9)
            e_cur1 = _window_energy(m1, w1)
            e_cur2 = _window_energy(m2, w2)
            amb_open = np.ones(amb.size, dtype=bool)

            while True:
                r1 = e_cur1 / e_exp1
                r2 = e_cur2 / e_exp2
                if abs(r1 - r2) <= float(cfg.ir_ratio_tol):
                    break
                donor_is_1 = r1 > r2 or (r1 == r2 and m1.label < m2.label)
                d_lab = m1.label if donor_is_1 else m2.label
                cand_mask = amb_open & (labels[amb] == d_lab)
                if not np.any(cand_mask):
                    break
                # most recipient-leaning candidate; ties by hit index
                key = lean if donor_is_1 else -lean
                order = np.lexsort((amb, key))
                pick = -1
                for oi in order:
                    if cand_mask[oi]:
                        pick = int(oi)
                        break
                if pick < 0:
                    break
                h = int(amb[pick])
                eh = float(e_pos[h])
                if donor_is_1:
                    n1 = e_cur1 - eh
                    n2 = e_cur2 + (eh if in_w2_amb[pick] else 0.0)
                    donor_new_r = n1 / e_exp1
                else:
                    n1 = e_cur1 + (eh if in_w1_amb[pick] else 0.0)
                    n2 = e_cur2 - eh
                    donor_new_r = n2 / e_exp2
                if abs(n1 / e_exp1 - n2 / e_exp2) >= abs(r1 - r2):
                    break
                if donor_new_r < float(cfg.ir_donor_floor_ratio):
                    break
                if moved_hits_e + eh > float(cfg.ir_pair_energy_cap_mev):
                    break
                snapshot.append((h, int(labels[h])))
                labels[h] = m2.label if donor_is_1 else m1.label
                moved_once[h] = True
                amb_open[pick] = False
                e_cur1, e_cur2 = n1, n2
                moved_hits_e += eh

            if not snapshot:
                continue

            # gap-continuity veto: a pair must not sever either track
            post_gap1 = _max_internal_gap(np.sort(m1.t_perp(pts[_members(m1.label, m1.tpc)])[0]))
            post_gap2 = _max_internal_gap(np.sort(m2.t_perp(pts[_members(m2.label, m2.tpc)])[0]))
            if (post_gap1 > max(pre_gap1, float(cfg.ir_gap_guard_cm))
                    or post_gap2 > max(pre_gap2, float(cfg.ir_gap_guard_cm))):
                for h, old in snapshot:
                    labels[h] = old
                    moved_once[h] = False
                _veto("b_gap_veto_rollback")
                continue

            stats["n_pairs_b_moved"] += 1
            stats["n_hits_moved_b"] += len(snapshot)
            stats["energy_moved_b_mev"] += moved_hits_e
            stats["pairs"].append({
                "kind": "b", "tpc": m1.tpc, "labels": [m1.label, m2.label],
                "n_moved": len(snapshot), "energy_mev": round(moved_hits_e, 3),
            })

        else:  # kind == "a"
            m = pair["model"]
            b_lab = pair["recipient"]
            w = pair["w"]
            if not m.baseline_valid:
                _veto("a_baseline_invalid")
                continue

            endpoint = (w[0] <= m.tmin + float(cfg.ir_endpoint_guard_cm)
                        or w[1] >= m.tmax - float(cfg.ir_endpoint_guard_cm))
            if endpoint:
                if cfg.ir_endpoint_soft_factor is None:
                    _veto("a_endpoint_guard")
                    continue
                cap = float(cfg.ir_endpoint_soft_factor) * m.rho0
            else:
                cap = float(cfg.ir_soft_factor) * m.rho0

            mem = _members(m.label, m.tpc)
            if mem.size == 0:
                _veto("a_empty_track")
                continue
            t_m, perp_m = m.t_perp(pts[mem])
            in_w = (t_m >= w[0]) & (t_m <= w[1])
            if not np.any(in_w):
                _veto("a_empty_window")
                continue

            # current window bin densities
            j_all = m.bin_of(t_m[in_w])
            nb = m.bin_w.size
            e_bins = np.bincount(j_all, weights=e_pos[mem[in_w]], minlength=nb)
            rho_cur = e_bins / np.maximum(m.bin_w, 1e-9)
            win_bins = sorted(int(v) for v in np.unique(j_all))
            excess = float(sum(max(rho_cur[j] - cap, 0.0) * m.bin_w[j] for j in win_bins))
            if excess < float(cfg.ir_min_excess_mev):
                _veto("a_no_excess")
                continue

            b_mem = _members(b_lab, m.tpc)
            if b_mem.size == 0:
                _veto("a_recipient_gone")
                continue
            b_tree = cKDTree(pts[b_mem])
            e_blob = float(np.sum(e_pos[np.flatnonzero(labels == b_lab)]))
            e_budget = min(float(cfg.ir_pair_energy_cap_mev),
                           float(cfg.ir_recipient_gain_frac) * max(e_blob, 0.0))

            cand_local = np.flatnonzero(
                in_w & (perp_m > m.r_core) & (e_pos[mem] > 0.0) & ~moved_once[mem]
            )
            if cand_local.size == 0:
                _veto("a_no_eligible")
                continue
            d_blob, _ = b_tree.query(pts[mem[cand_local]], k=1)
            linked = np.asarray(d_blob) <= float(cfg.ir_halo_link_cm)
            cand_local = cand_local[linked]
            if cand_local.size == 0:
                _veto("a_no_linked")
                continue
            d_blob = np.asarray(d_blob)[linked]

            j_cand = m.bin_of(t_m[cand_local])
            moved_e = 0.0
            n_moved = 0
            for j in win_bins:
                sel = np.flatnonzero(j_cand == j)
                if sel.size == 0:
                    continue
                order = np.lexsort((mem[cand_local[sel]], d_blob[sel],
                                    -perp_m[cand_local[sel]]))
                for oi in order:
                    if rho_cur[j] <= cap:
                        break
                    h = int(mem[cand_local[sel[oi]]])
                    eh = float(e_pos[h])
                    if rho_cur[j] - eh / m.bin_w[j] < m.rho0:
                        continue  # hard floor: bin never ends below baseline
                    if moved_e + eh > e_budget:
                        break
                    labels[h] = b_lab
                    moved_once[h] = True
                    rho_cur[j] -= eh / m.bin_w[j]
                    moved_e += eh
                    n_moved += 1
                if moved_e >= e_budget:
                    break

            if n_moved:
                stats["n_pairs_a_moved"] += 1
                stats["n_hits_moved_a"] += n_moved
                stats["energy_moved_a_mev"] += moved_e
                stats["pairs"].append({
                    "kind": "a", "tpc": m.tpc, "labels": [m.label, int(b_lab)],
                    "n_moved": n_moved, "energy_mev": round(moved_e, 3),
                    "endpoint_window": bool(endpoint),
                })

    # ---- Phase 5: commit with invariant checks --------------------------
    noise_ok = bool(np.array_equal(labels_old < 0, labels < 0)) and bool(
        np.array_equal(labels[labels_old < 0], labels_old[labels_old < 0]))
    label_set_ok = bool(np.isin(np.unique(labels), np.unique(labels_old)).all())
    if not (noise_ok and label_set_ok):
        stats["commit_ok"] = False
        stats["elapsed_s"] = time.perf_counter() - t_start
        return labels_old, stats

    stats["elapsed_s"] = time.perf_counter() - t_start
    return labels, stats
