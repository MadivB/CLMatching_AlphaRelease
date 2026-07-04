"""
Cosine-free, error-matrix-driven agglomerative FAMILY EXPANSION (testing version).

Motivation
----------
The whitened-cosine region-grow (``matching_2x2.region_grow_association``) is
scale-invariant, which is dangerous for a small DISPLACED fragment: it can be
pulled into a bright flash on spatial pattern alone, even though the fragment is
physically detached and dim.  This version replaces the cosine with the **error
matrix (chi2)** — chi2 = sum (model - observed)^2 / var over the cluster's
support — which penalises the magnitude mismatch cosine ignores, so a faint
fragment does NOT fit a bright flash.

Procedure (per the design):
  1. SEED    — via the error matrix, pick the most *decisive* unassigned cluster
               and open a family at its preferred t0.
  2. EXPAND  — find the spatially most-relevant external cluster to the family
               (single-linkage nearest-hit contact; fragmentation-robust).
  3. DECIDE  — via the error matrix, does that cluster prefer the family's t0?
       (a) yes & unassigned  -> absorb it, re-derive the family, re-expand;
       (b) no  & unassigned   -> it's a boundary (stop, or try the next-nearest);
       (c) already in family2 -> the two families touch: reconcile with a
           VARIANCE-weighted chi2 tiebreaker (may move the cluster 2<->1).
  Iterate until every cluster belongs to a family; write per-family hit t0.

Tracks are pre-seeded as their own families and never absorbed or moved (the
backbone).  The family PCA is computed for logging only here — the growth metric
is single-linkage contact, pending the boundary-method research.

This is a standalone testing module; the production path is unaffected.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from matching_2x2 import score_at, support_channels, _pca_direction  # noqa


def _cluster_min_dist(pts_a: np.ndarray, tree_b) -> float:
    """min hit-hit distance from point set a to cluster b (via b's KDTree)."""
    return float(tree_b.query(pts_a)[0].min())


def _dimensionality(pts: np.ndarray) -> dict:
    """Structure-tensor dimensionality features (Weinmann et al.): linearity,
    planarity, scattering from the local covariance eigenvalues l1>=l2>=l3.
    The research's dependency-light replacement for a global-PCA linearity."""
    if pts.shape[0] < 3:
        return {"lin": 0.0, "plan": 0.0, "scat": 1.0}
    d = pts - pts.mean(axis=0)
    ev = np.sort(np.linalg.eigvalsh(d.T @ d / pts.shape[0]))[::-1]
    l1 = max(float(ev[0]), 1e-12)
    return {"lin": float((ev[0] - ev[1]) / l1),
            "plan": float((ev[1] - ev[2]) / l1),
            "scat": float(ev[2] / l1)}


def _median_nn(pts: np.ndarray) -> Optional[float]:
    """Median nearest-neighbour spacing of a point cloud (the family's scale)."""
    if pts.shape[0] < 2:
        return None
    from scipy.spatial import cKDTree
    return float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))


def family_expand_association(*, labels, xset, yset, zset, Eset, hitTPCid, hit_t0,
                              cluster_energies, image_maps, base_image,
                              full_wvfm, full_var, track_labels, flash_seeds,
                              reconcile_var=None, contact_cm: float = 3.5,
                              accept_margin: float = 0.15, reconcile_hyst: float = 0.05,
                              support_fraction: float = 0.90, support_floor: float = 25.0,
                              try_next_neighbor: bool = True, max_outer: int = 2000,
                              adaptive_gap: bool = False, gap_k: float = 3.0
                              ) -> List[dict]:
    from scipy.spatial import cKDTree

    labels = np.asarray(labels, np.int64)
    tpcid = np.asarray(hitTPCid, np.int64)
    hit_t0 = np.asarray(hit_t0)                       # modify caller array IN PLACE
    XYZ = np.column_stack([xset, yset, zset]).astype(np.float64)
    track_set = set(int(c) for c in track_labels)
    ids = [int(c) for c in np.unique(labels) if c >= 0]
    if not ids:
        return []

    # ---- candidate t0 set: flash seeds + track t0s only (cosine-free, physical) ----
    cand = set()
    for tp in range(len(flash_seeds)):
        for s in flash_seeds[tp]:
            cand.add(round(float(s)))
    track_t0 = {}
    for c in ids:
        if c in track_set:
            t0s = hit_t0[labels == c]
            t0s = t0s[np.isfinite(t0s) & (t0s >= 0)]
            if t0s.size:
                tt = float(np.median(t0s))
                track_t0[c] = tt
                cand.add(round(tt))
    cand = sorted(cand)
    if not cand:
        return []
    cand_ix = {t: i for i, t in enumerate(cand)}

    # ---- per-cluster geometry, support, points, KDTree ----
    info: Dict[int, dict] = {}
    for c in ids:
        m = labels == c
        pts = XYZ[m]
        tpcs = sorted(set(int(t) for t in tpcid[m]))
        sup = {tp: support_channels(image_maps[(c, tp)],
                                    light_fraction=support_fraction, abs_floor=support_floor)
               for tp in tpcs if (c, tp) in image_maps}
        cen, direction, lin = (_pca_direction(pts, Eset[m]) if pts.shape[0] >= 2
                               else (pts[0], np.zeros(3), 0.0))
        dim = _dimensionality(pts)          # local structure-tensor (research rec.)
        info[c] = {"pts": pts, "tree": cKDTree(pts), "tpcs": tpcs, "sup": sup,
                   "E": float(cluster_energies.get(c, 0.0)), "track": c in track_set,
                   "cen": cen, "dir": direction, "lin": lin, "dim": dim}

    # ---- error matrix: chi2(c, t) at base=0 over the cluster's support ----
    def chi2_at(c: int, t: float, var_src) -> float:
        s, any_tp = 0.0, False
        for tp in info[c]["tpcs"]:
            if (c, tp) not in image_maps:
                continue
            z0 = np.zeros_like(full_wvfm[tp])
            s += score_at(image_maps[(c, tp)], z0, full_wvfm[tp], var_src[tp],
                          t, info[c]["sup"].get(tp))
            any_tp = True
        return s if any_tp else np.inf

    Emat = {c: np.array([chi2_at(c, t, full_var) for t in cand], float) for c in ids}
    best_i = {c: int(np.argmin(Emat[c])) for c in ids}
    best_t0 = {c: float(cand[best_i[c]]) for c in ids}
    best_chi2 = {c: float(Emat[c][best_i[c]]) for c in ids}

    def decisiveness(c: int) -> float:
        r = np.sort(Emat[c])
        return float((r[1] - r[0]) / max(r[0], 1e-6)) if r.size >= 2 else 0.0

    def wants(c: int, t0: float, var_src=None) -> bool:
        """does cluster c accept t0 (chi2 within accept_margin of its own best)?"""
        chi = (Emat[c][cand_ix[round(t0)]] if (var_src is None and round(t0) in cand_ix)
               else chi2_at(c, t0, full_var if var_src is None else var_src))
        return chi <= best_chi2[c] * (1.0 + accept_margin)

    # ---- cluster-cluster single-linkage min distance ----
    D: Dict[int, Dict[int, float]] = {c: {} for c in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = _cluster_min_dist(info[a]["pts"], info[b]["tree"])
            D[a][b] = D[b][a] = d

    # ---- families ----
    fam_of: Dict[int, Optional[int]] = {c: None for c in ids}
    families: List[dict] = []
    for c in ids:                              # tracks first, as fixed anchors
        if c in track_set:
            families.append({"clusters": {c}, "t0": track_t0.get(c, best_t0[c]),
                             "track": True})
            fam_of[c] = len(families) - 1

    def nearest_externals(fam_clusters, reach):
        out = []
        for e in ids:
            if e in fam_clusters:
                continue
            d = min(D[fc][e] for fc in fam_clusters)
            if d <= reach:
                out.append((d, e))
        out.sort()
        return out

    rows: List[dict] = []
    unassigned = [c for c in ids if fam_of[c] is None]
    guard = 0
    while unassigned and guard < max_outer:
        guard += 1
        # STEP 1 — seed: most decisive unassigned cluster (error matrix), E tiebreak
        seed = max(unassigned, key=lambda c: (decisiveness(c), info[c]["E"]))
        fidx = len(families)
        families.append({"clusters": {seed}, "t0": best_t0[seed], "track": False})
        fam_of[seed] = fidx
        unassigned.remove(seed)
        rows.append({"event": "seed", "cluster": seed, "t0": best_t0[seed],
                     "decisiveness": round(decisiveness(seed), 3),
                     "dim": {k: round(v, 2) for k, v in info[seed]["dim"].items()},
                     "energy_mev": info[seed]["E"]})

        # STEP 2/3 — expand this family
        while guard < max_outer:
            guard += 1
            F = families[fidx]
            # scale-aware reach: a sparse (shower) family may bridge larger gaps
            # than a dense (track) one (research: adaptive/scale-aware linking).
            if adaptive_gap:
                fam_pts = np.vstack([info[c]["pts"] for c in F["clusters"]])
                sp = _median_nn(fam_pts)
                reach = max(contact_cm, gap_k * sp) if sp else contact_cm
            else:
                reach = contact_cm
            progressed = False
            for d, e in nearest_externals(F["clusters"], reach):
                fe = fam_of[e]
                if fe is None:                         # ----- case a / b -----
                    if wants(e, F["t0"]):              # (a) absorb
                        F["clusters"].add(e); fam_of[e] = fidx
                        if e in unassigned:
                            unassigned.remove(e)
                        rows.append({"event": "absorb", "cluster": e, "into": fidx,
                                     "t0": F["t0"], "dist_cm": round(d, 2),
                                     "energy_mev": info[e]["E"]})
                        progressed = True
                        break
                    else:                              # (b) boundary
                        rows.append({"event": "reject", "cluster": e, "family": fidx,
                                     "dist_cm": round(d, 2), "best_t0_e": best_t0[e],
                                     "fam_t0": F["t0"]})
                        if try_next_neighbor:
                            continue                   # try the next-nearest
                        break                          # strict spec: stop growth
                else:                                  # ----- case c: e in family fe -----
                    if fe == fidx:
                        continue
                    F2 = families[fe]
                    # variance-weighted chi2 tiebreaker at the contact
                    vsrc = reconcile_var if reconcile_var is not None else full_var
                    chi_F = chi2_at(e, F["t0"], vsrc)
                    chi_F2 = chi2_at(e, F2["t0"], vsrc)
                    move = (not F2["track"]) and chi_F < chi_F2 * (1.0 - reconcile_hyst)
                    rows.append({"event": "contact", "cluster": e, "famA": fidx,
                                 "famB": fe, "chi_A": round(chi_F, 1), "chi_B": round(chi_F2, 1),
                                 "moved": bool(move), "dist_cm": round(d, 2)})
                    if move:
                        F2["clusters"].discard(e)
                        F["clusters"].add(e); fam_of[e] = fidx
                        progressed = True
                        break
                    # else: boundary between two families; keep scanning neighbours
                    continue
            if not progressed:
                break                                  # family complete -> back to STEP 1

    # ---- write per-family t0 to hits (tracks unchanged: fam t0 == track t0) ----
    for fam in families:
        for c in fam["clusters"]:
            hit_t0[labels == c] = np.float32(fam["t0"])

    return rows
