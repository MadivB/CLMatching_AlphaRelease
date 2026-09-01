"""
Threshold-permutation association (the user's redesign, 2026-07-15).

Replaces the greedy Stage-4/5 association wholesale. Energy threshold (30 MeV
default) splits the problem:

STAGE A — big clusters (E >= big_mev), permutation placement:
  Sequential placement, energy-descending: each cluster is placed by a FULL
  residual chi2 scan over the whole t0 range (no flash-seed reliance — the 2x2
  has only 8 TPCs, we can afford it), its predicted light added onto the
  running plateau. Then the SWAP stage: for every pair of big clusters sharing
  a TPC, try the reversed placement order (remove both light images, place in
  reverse, re-scan); keep whichever ordering reaches the smaller total light
  loss — the light image is swapped along with the charge. (Orders often agree;
  the swap costs little.)

STAGE B — family expansion from each big cluster, largest first:
  The placed cluster seeds a FAMILY. Repeatedly take the spatially most
  relevant (nearest to the family frontier, within contact_cm) still-unplaced
  cluster with E < big_mev and test whether it PREFERS the family's t: residual
  chi2 on its support channels at t vs its own free-scan best. If it prefers t:
  place it at t, add its light to the plateau, its hits join the family
  (the frontier grows). If it does NOT: STOP this family's expansion
  immediately WITHOUT placing that cluster (it stays for stage C).

STAGE C — leftovers:
  Error-matrix association + family-based spatial expansion for everything
  still unplaced, via family_expand_association(respect_greedy=True): stage-A/B
  placements are frozen anchors (their t0s enter the candidate set); leftovers
  are adopted into adjacent families or seeded from their error-matrix best.

INVARIANT: no stage ever revises an earlier stage's decision. The only revision
mechanism is the stage-A swap, which is internal to stage A.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import geometry_2x2 as geo
from matching_2x2 import (full_integer_scan, shift_frac, support_channels,
                          score_at)
from family_expand_2x2 import family_expand_association

ADC_CLIP = geo.ADC_CLIP


def _cluster_tpcs(image_maps, cid):
    return sorted(tp for (c, tp) in image_maps if c == cid)


def _scan_cluster(cid, tpcs, image_maps, base, act, var, search_range,
                  support=None):
    """Summed-over-TPCs residual chi2 scan; returns (best_t0, errs_total)."""
    errs_total = None
    for tp in tpcs:
        img = image_maps[(cid, tp)]
        sup = support.get(tp) if isinstance(support, dict) else support
        _, errs = full_integer_scan(img, base[tp], act[tp], var[tp],
                                    search_range, sup)
        errs_total = errs if errs_total is None else errs_total + errs
    if errs_total is None:
        return None, None
    return float(int(np.argmin(errs_total))), errs_total


def _add_light(base, image_maps, cid, tpcs, t0, sign=+1.0):
    for tp in tpcs:
        blk = base[tp] + sign * shift_frac(image_maps[(cid, tp)], t0)
        base[tp] = np.clip(blk, 0.0, ADC_CLIP)


def _total_loss(base, act, var, tpcs):
    return sum(float(((np.clip(base[tp], None, ADC_CLIP) - act[tp]) ** 2
                      / np.maximum(var[tp], 1e-6)).sum()) for tp in tpcs)


def threshold_family_association(*, labels, xset, yset, zset, Eset, hitTPCid,
                                 hit_t0, cluster_energies, image_maps,
                                 full_wvfm, full_var, flash_seeds,
                                 labels_noisy=None,
                                 big_mev: float = 30.0, contact_cm: float = 3.5,
                                 accept_margin: float = 0.15,
                                 prefer_tol: float = 5.0,
                                 search_range: int = None,
                                 support_fraction: float = 0.90,
                                 support_floor: float = 25.0,
                                 swap_passes: int = 2,
                                 adopt_max_cm: float = 10.0,
                                 adopt_ratio: float = 3.0,
                                 light_model=None,
                                 refine_step: float = 0.25,
                                 refine_span: float = 3.0,
                                 refine_quad: bool = True,
                                 refine_log=None,
                                 hit_refine: str = None,
                                 sipm_xyz=None,
                                 chan_log=None,
                                 chan_min_amp: float = 600.0,
                                 chan_knn: int = 6,
                                 chan_pow: float = 2.0,
                                 verbose: bool = False) -> List[dict]:
    """Full association from scratch. hit_t0 is OVERWRITTEN in place (start
    from an unassigned array); returns the decision log."""
    from scipy.spatial import cKDTree

    labels = np.asarray(labels, np.int64)
    hit_t0 = np.asarray(hit_t0)
    XYZ = np.column_stack([xset, yset, zset]).astype(np.float64)
    act = np.asarray(full_wvfm, np.float32)
    var = np.asarray(full_var, np.float32)
    if search_range is None:
        search_range = int(geo.SEARCH_RANGE)
    hit_t0[:] = np.nan                                  # clean slate

    ids = [int(c) for c in np.unique(labels) if c >= 0]
    tpcs_of = {c: _cluster_tpcs(image_maps, c) for c in ids}
    E_of = {c: float(cluster_energies.get(c, 0.0)) for c in ids}
    sup_of = {c: {tp: support_channels(image_maps[(c, tp)],
                                       light_fraction=support_fraction,
                                       abs_floor=support_floor)
                  for tp in tpcs_of[c]} for c in ids}

    base = np.zeros_like(act)
    rows: List[dict] = []
    placed_t0: Dict[int, float] = {}

    # ================= STAGE A: big clusters, sequential + swap =============
    big = sorted((c for c in ids if E_of[c] >= big_mev and tpcs_of[c]),
                 key=lambda c: -E_of[c])
    for c in big:
        t0, _ = _scan_cluster(c, tpcs_of[c], image_maps, base, act, var,
                              search_range)
        placed_t0[c] = t0
        _add_light(base, image_maps, c, tpcs_of[c], t0)
        rows.append({"event": "placeA", "cluster": c, "t0": t0,
                     "E": E_of[c]})

    for _ in range(int(swap_passes)):
        improved = False
        for i in range(len(big)):
            for j in range(i + 1, len(big)):
                a, b = big[i], big[j]
                if not set(tpcs_of[a]) & set(tpcs_of[b]):
                    continue
                tps = sorted(set(tpcs_of[a]) | set(tpcs_of[b]))
                loss_now = _total_loss(base, act, var, tps)
                # remove both, place b first then a (reversed order)
                _add_light(base, image_maps, a, tpcs_of[a], placed_t0[a], -1)
                _add_light(base, image_maps, b, tpcs_of[b], placed_t0[b], -1)
                tb, _ = _scan_cluster(b, tpcs_of[b], image_maps, base, act,
                                      var, search_range)
                _add_light(base, image_maps, b, tpcs_of[b], tb)
                ta, _ = _scan_cluster(a, tpcs_of[a], image_maps, base, act,
                                      var, search_range)
                _add_light(base, image_maps, a, tpcs_of[a], ta)
                loss_swapped = _total_loss(base, act, var, tps)
                if loss_swapped < loss_now - 1e-6:
                    rows.append({"event": "swap", "pair": [a, b],
                                 "from": [placed_t0[a], placed_t0[b]],
                                 "to": [ta, tb]})
                    placed_t0[a], placed_t0[b] = ta, tb
                    improved = True
                else:                                   # revert to original
                    _add_light(base, image_maps, a, tpcs_of[a], ta, -1)
                    _add_light(base, image_maps, b, tpcs_of[b], tb, -1)
                    _add_light(base, image_maps, a, tpcs_of[a], placed_t0[a])
                    _add_light(base, image_maps, b, tpcs_of[b], placed_t0[b])
        if not improved:
            break

    # ================= STAGE B: family expansion, strict frontier ===========
    small_unplaced = set(c for c in ids
                         if c not in placed_t0 and tpcs_of[c])
    small_trees = {c: cKDTree(XYZ[labels == c]) for c in small_unplaced}
    for seed in big:                                    # largest first
        fam_pts = [XYZ[labels == seed]]
        T = placed_t0[seed]
        while True:
            fam_tree = cKDTree(np.vstack(fam_pts))
            best_d, best_e = np.inf, None
            for e in small_unplaced:
                d = float(np.min(fam_tree.query(XYZ[labels == e])[0]))
                if d < best_d:
                    best_d, best_e = d, e
            if best_e is None or best_d > contact_cm:
                break                                   # no frontier neighbour
            e = best_e
            # preference test — STRICT: the cluster PREFERS the family's t only
            # if its OWN free-scan optimum (residual base, support channels)
            # coincides with T (within the t0 resolution). "Close enough in
            # chi2" is NOT preference — that was the bright-flash trap.
            t_free, errs = _scan_cluster(e, tpcs_of[e], image_maps, base, act,
                                         var, search_range, support=sup_of[e])
            chi_free = float(errs[int(t_free)])
            chi_T = float(errs[int(round(min(max(T, 0), search_range)))])
            if abs(t_free - T) <= prefer_tol:
                placed_t0[e] = float(T)
                _add_light(base, image_maps, e, tpcs_of[e], float(T))
                fam_pts.append(XYZ[labels == e])
                small_unplaced.discard(e)
                rows.append({"event": "placeB", "cluster": e, "t0": float(T),
                             "family": seed, "dist_cm": round(best_d, 2),
                             "E": E_of[e]})
            else:
                rows.append({"event": "stopB", "cluster": e, "family": seed,
                             "dist_cm": round(best_d, 2),
                             "chi_T": chi_T, "chi_free": chi_free})
                break                                   # STRICT stop, e NOT placed

    # write A+B placements (these become frozen anchors for stage C)
    for c, t0 in placed_t0.items():
        hit_t0[labels == c] = np.float32(t0)

    # ================= STAGE C: error-matrix + spatial expansion ============
    rows_c = family_expand_association(
        labels=labels, xset=xset, yset=yset, zset=zset, Eset=Eset,
        hitTPCid=hitTPCid, hit_t0=hit_t0, cluster_energies=cluster_energies,
        image_maps=image_maps, base_image=base, full_wvfm=full_wvfm,
        full_var=full_var,
        track_labels=[],                # no special-casing: anchors rule
        flash_seeds=flash_seeds, contact_cm=contact_cm,
        accept_margin=accept_margin, respect_greedy=True, use_base=True)
    rows.extend({"stage": "C", **r} for r in rows_c)

    # ---- noise-absorbed extras: the greedy pipeline stamps hits that DBSCAN
    # called noise but that were absorbed into a cluster (labels_noisy). Extend
    # each placed cluster's t0 to its still-unassigned noisy extras, so tiny
    # 1-3-hit deposits attached to a placed cluster are not left behind. ----
    if labels_noisy is not None:
        ln = np.asarray(labels_noisy, np.int64)
        for c in ids:
            m = labels == c
            tv = hit_t0[m]
            tv = tv[np.isfinite(tv) & (tv >= 0)]
            if not tv.size:
                continue
            extra = (ln == c) & ~(np.isfinite(hit_t0) & (hit_t0 >= 0))
            if extra.any():
                hit_t0[extra] = np.float32(np.median(tv))
    # ============ STAGE D: UNAMBIGUOUS orphan adoption (not forced) =========
    # An orphan hit (never clustered, never assigned) is adopted ONLY when it
    # unambiguously belongs to one t0 group: the nearest assigned hit (searched
    # over the FULL detector, all TPCs) must be much closer (adopt_ratio x)
    # than the nearest assigned hit at any DIFFERENT t0. Single-interaction
    # events have no second group to compare against, so an absolute reach cap
    # (adopt_max_cm) still refuses far-out hits. Ambiguous -> left unassigned.
    if adopt_max_cm > 0:
        assigned = np.isfinite(hit_t0) & (hit_t0 >= 0)
        fin = np.isfinite(XYZ).all(axis=1)
        orph = (~assigned) & fin
        if orph.any() and assigned.any():
            # t0 groups over assigned hits (merge within 10 ticks)
            avals = hit_t0[assigned]
            aidx = np.flatnonzero(assigned)
            centers: List[float] = []
            gid = np.full(avals.size, -1, np.int64)
            for i in np.argsort(avals):
                for gi, c0 in enumerate(centers):
                    if abs(avals[i] - c0) <= 10.0:
                        gid[i] = gi
                        break
                else:
                    centers.append(float(avals[i]))
                    gid[i] = len(centers) - 1
            gtrees = [cKDTree(XYZ[aidx[gid == gi]]) for gi in range(len(centers))]
            gt0 = [float(np.median(avals[gid == gi])) for gi in range(len(centers))]
            n_ad = n_amb = 0
            for hi in np.flatnonzero(orph):
                ds = np.array([float(t.query(XYZ[hi])[0]) for t in gtrees])
                o = np.argsort(ds)
                d1 = ds[o[0]]
                if d1 > adopt_max_cm:
                    continue                        # too far even from nearest
                d2 = ds[o[1]] if ds.size > 1 else np.inf
                if d2 >= adopt_ratio * d1:
                    hit_t0[hi] = np.float32(gt0[int(o[0])])
                    n_ad += 1
                else:
                    n_amb += 1                      # ambiguous -> stay unassigned
            rows.append({"event": "adoptD", "n_orphans": int(orph.sum()),
                         "n_adopted": int(n_ad), "n_ambiguous": int(n_amb),
                         "adopt_max_cm": float(adopt_max_cm),
                         "adopt_ratio": float(adopt_ratio)})

    # ===== STAGE E: group re-prediction + 0.25-tick interpolated refine =====
    # Group ALL assigned hits by t0, re-predict each group's full waveform with
    # the perceiver (one bag per interaction), then refine each group t0 on a
    # refine_step (0.25 tick) grid minimizing the residual chi2 with the other
    # groups' re-predicted light as base. Membership is untouched — this only
    # sharpens each interaction's time.
    if light_model is not None:
        assigned = np.isfinite(hit_t0) & (hit_t0 >= 0)
        if assigned.any():
            avals = hit_t0[assigned]
            aidx = np.flatnonzero(assigned)
            centers2: List[float] = []
            gid2 = np.full(avals.size, -1, np.int64)
            for i in np.argsort(avals):
                for gi, c0 in enumerate(centers2):
                    if abs(avals[i] - c0) <= 10.0:
                        gid2[i] = gi
                        break
                else:
                    centers2.append(float(avals[i]))
                    gid2[i] = len(centers2) - 1
            G = len(centers2)
            gmaps, _ = light_model.predict_image_maps(
                np.asarray(xset)[aidx], np.asarray(yset)[aidx],
                np.asarray(zset)[aidx], np.asarray(Eset)[aidx],
                np.asarray(hitTPCid)[aidx], gid2)
            gt0v = [float(np.median(avals[gid2 == gi])) for gi in range(G)]
            gE = [float(np.asarray(Eset)[aidx][gid2 == gi].sum()) for gi in range(G)]
            gtpcs = {gi: sorted(tp for (g, tp) in gmaps if g == gi)
                     for gi in range(G)}
            for gi in sorted(range(G), key=lambda g: -gE[g]):
                if not gtpcs[gi]:
                    continue
                grid = np.arange(gt0v[gi] - refine_span,
                                 gt0v[gi] + refine_span + 1e-6, refine_step)
                grid = grid[(grid >= 0) & (grid <= search_range)]
                losses = np.empty(grid.size, np.float64)
                for k, tcand in enumerate(grid):
                    loss = 0.0
                    for tp in gtpcs[gi]:
                        bo = np.zeros_like(act[tp])
                        for gj in range(G):
                            if gj == gi or (gj, tp) not in gmaps:
                                continue
                            bo += shift_frac(gmaps[(gj, tp)], gt0v[gj])
                        model = np.clip(bo + shift_frac(gmaps[(gi, tp)], float(tcand)),
                                        None, ADC_CLIP)
                        loss += float(((model - act[tp]) ** 2
                                       / np.maximum(var[tp], 1e-6)).sum())
                    losses[k] = loss
                im = int(np.argmin(losses))
                best_t = float(grid[im])
                # analytic parabola through the 7 points around the minimum:
                # symmetric x-grid -> decoupled least squares,
                # b = sum(xL)/S2, a = (sum(x^2 L) - S2/7 sum L)/(S4 - S2^2/7),
                # continuous minimum at x* = -b/(2a). No c needed.
                if refine_quad and 3 <= im <= grid.size - 4:
                    xw = (grid[im - 3:im + 4] - grid[im])
                    Lw = losses[im - 3:im + 4]
                    S2 = float(np.sum(xw ** 2))
                    S4 = float(np.sum(xw ** 4))
                    b = float(np.sum(xw * Lw)) / S2
                    a = (float(np.sum(xw ** 2 * Lw)) - S2 / 7.0 * float(np.sum(Lw))) \
                        / (S4 - S2 ** 2 / 7.0)
                    if a > 0:
                        xstar = -b / (2.0 * a)
                        if abs(xstar) <= 0.5:
                            best_t = float(grid[im] + xstar)
                if refine_log is not None:
                    refine_log.append({
                        "group": int(gi), "E": float(gE[gi]),
                        "t_int": float(gt0v[gi]),
                        "grid": grid.tolist(), "losses": losses.tolist(),
                        "im": im, "t_disc": float(grid[im]),
                        "t_quad": float(best_t)})
                if abs(best_t - gt0v[gi]) > 1e-9:
                    hit_t0[aidx[gid2 == gi]] = np.float32(best_t)
                    rows.append({"event": "refineE", "group": int(gi),
                                 "from": gt0v[gi], "to": best_t,
                                 "E": round(gE[gi], 1)})
                    gt0v[gi] = best_t

            # ===== STAGE F: per-hit t0 from per-channel fits ================
            # For each group and TPC, fit every light channel's OWN t0
            # (same 0.25-tick grid + analytic quadratic as stage E, chi2
            # restricted to that channel), then give each charge hit the
            # inverse-distance-weighted average of its chan_knn nearest
            # channels' times. Dim channels (predicted peak < chan_min_amp)
            # are excluded, but at least chan_knn channels are kept per TPC
            # (topped up by amplitude). Group membership and group times are
            # untouched; only member-hit t0s fan out around the group time.
            if hit_refine in ("channel6", "plane") and sipm_xyz is not None:
                sxyz = np.asarray(sipm_xyz, np.float64)
                tpc_of = np.asarray(hitTPCid)[aidx]
                for gi in range(G):
                    members = np.flatnonzero(gid2 == gi)
                    if members.size == 0 or not gtpcs[gi]:
                        continue
                    grid = np.arange(gt0v[gi] - refine_span,
                                     gt0v[gi] + refine_span + 1e-6, refine_step)
                    grid = grid[(grid >= 0) & (grid <= search_range)]
                    if grid.size < 7:
                        continue
                    for tp in gtpcs[gi]:
                        img = gmaps[(gi, tp)]
                        pos = sxyz[tp]
                        okpos = np.isfinite(pos).all(axis=1)
                        amps = img.max(axis=1)
                        sel = okpos & (amps >= chan_min_amp)
                        if sel.sum() < chan_knn:
                            cand = np.flatnonzero(okpos)
                            if cand.size == 0:
                                continue
                            top = cand[np.argsort(amps[cand])[::-1][:chan_knn]]
                            sel = np.zeros(amps.size, bool)
                            sel[top] = True
                        chans = np.flatnonzero(sel)
                        # with chan_log we fit EVERY mapped channel (offline
                        # estimator studies subset them later); assignment
                        # still uses only the amp-selected subset
                        fitset = np.flatnonzero(okpos) if chan_log is not None \
                            else chans
                        bo = np.zeros_like(act[tp])
                        for gj in range(G):
                            if gj != gi and (gj, tp) in gmaps:
                                bo += shift_frac(gmaps[(gj, tp)], gt0v[gj])
                        a_c = act[tp][fitset]
                        v_c = np.maximum(var[tp][fitset], 1e-6)
                        b_c = bo[fitset]
                        i_c = img[fitset]
                        L = np.empty((grid.size, fitset.size), np.float64)
                        for k, tcand in enumerate(grid):
                            model = np.clip(b_c + shift_frac(i_c, float(tcand)),
                                            None, ADC_CLIP)
                            L[k] = ((model - a_c) ** 2 / v_c).sum(axis=1)
                        tch_fit = np.empty(fitset.size)
                        for ci in range(fitset.size):
                            im = int(np.argmin(L[:, ci]))
                            bt = float(grid[im])
                            if refine_quad and 3 <= im <= grid.size - 4:
                                xw = grid[im - 3:im + 4] - grid[im]
                                Lw = L[im - 3:im + 4, ci]
                                S2 = float(np.sum(xw ** 2))
                                S4 = float(np.sum(xw ** 4))
                                b2 = float(np.sum(xw * Lw)) / S2
                                a2 = (float(np.sum(xw ** 2 * Lw))
                                      - S2 / 7.0 * float(np.sum(Lw))) \
                                    / (S4 - S2 ** 2 / 7.0)
                                if a2 > 0:
                                    xs = -b2 / (2.0 * a2)
                                    if abs(xs) <= 0.5:
                                        bt = float(grid[im] + xs)
                            tch_fit[ci] = bt
                        tch = tch_fit[np.searchsorted(fitset, chans)] \
                            if chan_log is not None else tch_fit
                        hm = members[tpc_of[members] == tp]
                        if hm.size == 0:
                            continue
                        if chan_log is not None:
                            chan_log.append(dict(
                                group=int(gi), tpc=int(tp),
                                chans=fitset.tolist(),
                                tch=tch_fit.tolist(),
                                amps=amps[fitset].tolist(),
                                pos=pos[fitset].tolist(),
                                hits=aidx[hm].tolist(),
                                grid=grid.tolist(),
                                L=L.T.tolist()))
                        hxyz = XYZ[aidx[hm]]
                        d = np.linalg.norm(
                            hxyz[:, None, :] - pos[chans][None, :, :], axis=2)
                        if hit_refine == "plane" and chans.size >= 6 and \
                                np.unique(np.round(pos[chans][:, 2], 1)).size >= 2:
                            # first-order: weighted local plane t(y,z) through
                            # ALL selected channels, evaluated at each hit.
                            # (zeroth-order IDW plateaus along z because the
                            # nearest set is single-wall; the plane uses the
                            # two walls as a z lever arm.)  Steep 1/d^3 kernel:
                            # 1/d oversmooths — distant channels drag every
                            # intercept toward the TPC mean and attenuate real
                            # time gradients; d^-3 keeps the fit local while
                            # remaining fully continuous in the hit position.
                            u = 1.0 / np.maximum(d, 1e-3) ** 3     # (N, C)
                            dy = pos[chans][None, :, 1] - hxyz[:, 1:2]
                            dz = pos[chans][None, :, 2] - hxyz[:, 2:3]
                            one = np.ones_like(dy)
                            A = np.stack([one, dy, dz], axis=2)    # (N, C, 3)
                            ATA = np.einsum("nci,nc,ncj->nij", A, u, A)
                            ATb = np.einsum("nci,nc,c->ni", A, u, tch)
                            lam = 1e-3 * u.sum(axis=1)
                            ATA[:, 1, 1] += lam
                            ATA[:, 2, 2] += lam
                            beta = np.linalg.solve(ATA, ATb[..., None])[..., 0]
                            hit_t0[aidx[hm]] = beta[:, 0].astype(np.float32)
                        else:
                            k6 = min(chan_knn, chans.size)
                            nn = np.argpartition(d, k6 - 1, axis=1)[:, :k6]
                            dn = np.take_along_axis(d, nn, axis=1)
                            w = 1.0 / np.maximum(dn, 1e-3) ** chan_pow
                            hit_t0[aidx[hm]] = ((w * tch[nn]).sum(axis=1)
                                                / w.sum(axis=1)).astype(np.float32)
                        rows.append({"event": "perhitF", "group": int(gi),
                                     "tpc": int(tp), "n_ch": int(chans.size),
                                     "mode": hit_refine,
                                     "ch_spread": round(float(tch.max()
                                                              - tch.min()), 3)})

    if verbose:
        nA = sum(1 for r in rows if r.get("event") == "placeA")
        nB = sum(1 for r in rows if r.get("event") == "placeB")
        print(f"[threshold] A placed {nA}, B adopted {nB}, "
              f"C rows {len(rows_c)}", flush=True)
    return rows
