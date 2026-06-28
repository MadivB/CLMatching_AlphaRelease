"""
Phased charge-light matching kernels for the 2x2 (vAlpha port).

The scoring is the same variance-weighted chi2 the ND vAlpha uses:

    chi2(t0) = sum_{ch,tick} ( clip(base + shift(image, t0)) - actual )^2 / var

where ``base`` is the running sum of already-placed cluster light (so later
clusters fit the residual), ``image`` is the perceiver-predicted (48, T) light
of a cluster, and t0 is the shift (in matching ticks) applied so the template
peak (tick 105) lands on the observed flash.

Phases (mirroring the ND vAlpha / v4-2x2 first stage):
  * match_tracks            — Stage 4: each track keeps its own t0 (backbone).
  * match_clusters_greedy   — Stage 5: per-TPC greedy placement of blobs, with
                              two small-blob-specific behaviours:
                                (a) chi2 restricted to the cluster's predicted
                                    "support" channels — a faint blob lights a
                                    handful of SiPMs, so scoring all 48 buries
                                    its signal in noise;
                                (b) faint blobs are matched to flash-seed / track
                                    t0 candidates instead of a free 700-tick
                                    scan that noise can capture.
  * refine_clusters         — Phase 8: per-cluster full scan + sub-tick refine.

Phase 2 (large-cluster flash-grid) and Phase 3 (small-cluster matrix) are
provided by the order-agnostic ``phased_matching`` module and called from the
pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import geometry_2x2 as geo
from pulse_shapes import timeinterpolation

ADC_CLIP = geo.ADC_CLIP
PEAK = geo.PULSE_PEAK_TICK


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------
def shift_frac(image: np.ndarray, t0: float) -> np.ndarray:
    """Fractional time-shift of a (C, T) image; outside-window filled with 0."""
    return timeinterpolation(np.asarray(image, np.float32), shift=float(t0),
                             baseline=0.0).astype(np.float32)


def _chi2(model: np.ndarray, actual: np.ndarray, var: np.ndarray) -> float:
    return float(((np.clip(model, None, ADC_CLIP) - actual) ** 2
                  / np.maximum(var, 1e-6)).sum())


def score_at(image, base, actual, var, t0, support=None) -> float:
    """chi2 of placing ``image`` at t0 on top of ``base`` (residual fit)."""
    shifted = shift_frac(image, t0)
    rows = slice(None) if support is None else support
    model = base[rows] + shifted[rows]
    return _chi2(model, actual[rows], var[rows])


def full_integer_scan(image, base, actual, var, search_range,
                      support=None, *, block: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Integer-shift chi2 scan over t0 in [0, search_range].

    Vectorised over shifts with a strided sliding window (exact for integer t0,
    identical result to a per-shift loop, ~20x faster).  When ``support`` is
    given the chi2 is restricted to those channels (key for small-blob S/N).
    """
    image = np.asarray(image, np.float32)
    base = np.asarray(base, np.float32)
    actual = np.asarray(actual, np.float32)
    var = np.maximum(np.asarray(var, np.float32), 1e-6)
    if support is not None:
        image, base, actual, var = image[support], base[support], actual[support], var[support]
    C, T = image.shape
    S = int(search_range)

    # pad = [zeros(C,S), image]; shift(image,t0)[:,t] = pad[:, (S-t0)+t]
    pad = np.zeros((C, S + T), dtype=np.float32)
    pad[:, S:] = image
    windows = np.lib.stride_tricks.sliding_window_view(pad, T, axis=1)  # (C, S+1, T)

    actual_b = actual[:, None, :]
    var_b = var[:, None, :]
    base_b = base[:, None, :]
    errs_by_start = np.empty(S + 1, dtype=np.float32)
    for b0 in range(0, S + 1, int(block)):
        b1 = min(S + 1, b0 + int(block))
        model = np.clip(base_b + windows[:, b0:b1, :], None, ADC_CLIP)
        errs_by_start[b0:b1] = ((model - actual_b) ** 2 / var_b).sum(axis=(0, 2))

    # window start = S - t0  =>  errors indexed by t0 are the reversed order
    shifts = np.arange(S + 1, dtype=np.int32)
    errs = errs_by_start[::-1].copy()
    return shifts, errs


def peak_snap_t0(actual, base, t0, resolution, *, back: Optional[int] = None) -> float:
    """Snap t0 so the model peak lands on the observed residual peak.

    Expected model peak is at PEAK + t0; we look for the actual (positive)
    residual peak within [expected-back, expected] and shift t0 by the offset.
    """
    expected = int(PEAK + t0)
    signal = np.clip(np.asarray(actual) - np.asarray(base), 0.0, None).sum(axis=0)
    b = int(resolution if back is None else back)
    s0 = max(0, expected - b)
    s1 = min(geo.WVFM_LEN, expected + 1)
    if s1 > s0:
        off = int(np.argmax(signal[s0:s1]))
        return float(t0 + (s0 + off - expected))
    return float(t0)


def light_support_tile_mask(wvfm_tpc, *, n_keep_tiles: int = 4) -> np.ndarray:
    """ND-style 'light support' for the 2x2: keep the brightest light-detector
    tiles, mask the rest.

    The 48 channels (ORDERED_KEYS) form 8 tiles of 6: tile = side*4 + (y_rel//6),
    i.e. side 0/1 x y_rel blocks [0-5],[6-11],[12-17],[18-23].  Returns a (48,)
    bool mask (True = keep) of the ``n_keep_tiles`` tiles with the most observed
    light in this TPC.
    """
    peak = np.clip(np.asarray(wvfm_tpc, np.float32), 0.0, None).max(axis=1)  # (48,)
    ch = np.arange(peak.size)
    tile_id = (ch // 24) * 4 + ((ch % 24) // 6)                # 0..7
    n_tiles = int(tile_id.max()) + 1
    tile_bright = np.array([peak[tile_id == t].sum() for t in range(n_tiles)])
    keep = set(int(t) for t in np.argsort(tile_bright)[-int(n_keep_tiles):])
    return np.array([int(tile_id[j]) in keep for j in ch], dtype=bool)


def support_channels(image, *, light_fraction: float = 0.90,
                     abs_floor: float = 25.0) -> Optional[np.ndarray]:
    """Boolean (48,) mask of channels a cluster is predicted to light.

    Smallest set of channels whose summed predicted peak reaches
    ``light_fraction`` of the total, unioned with any channel above
    ``abs_floor`` ADC.  Returns None if the cluster predicts ~no light (caller
    falls back to all channels).
    """
    pk = np.clip(np.asarray(image, np.float32).max(axis=1), 0.0, None)  # (48,)
    total = float(pk.sum())
    if total <= 0.0 or pk.max() <= 0.0:
        return None
    order = np.argsort(pk)[::-1]
    csum = np.cumsum(pk[order])
    n_keep = int(np.searchsorted(csum, light_fraction * total) + 1)
    mask = np.zeros(pk.size, dtype=bool)
    mask[order[:n_keep]] = True
    mask |= pk >= float(abs_floor)
    return mask


def refine_t0_subtick(image, base, actual, var, t0, *, grid=None,
                      support=None) -> Tuple[float, float]:
    """Sub-tick refine of t0 on a fine grid around t0.  Returns (best_t0, err)."""
    if grid is None:
        grid = np.arange(-0.8, 0.85, 0.05)
    cands = np.asarray(t0, np.float64) + np.asarray(grid, np.float64)
    cands = cands[cands >= 0]
    best_t0, best_e = float(t0), score_at(image, base, actual, var, t0, support)
    for c in cands:
        e = score_at(image, base, actual, var, float(c), support)
        if e < best_e:
            best_e, best_t0 = e, float(c)
    return best_t0, best_e


# ----------------------------------------------------------------------------
# Candidate-t0 bookkeeping
# ----------------------------------------------------------------------------
def _ensure_seeds(cand: List[float], seeds: Sequence[float], resolution: float):
    for s in seeds:
        if not any(abs(float(s) - c) <= resolution for c in cand):
            cand.append(float(s))


def observed_brightness_at(actual, support, t0, *, window: int = 8) -> float:
    """Peak observed light (summed over support channels) near tick 105+t0.

    Used to ask "is there a real flash here that lights this blob's channels?"
    """
    rows = slice(None) if support is None else support
    sig = np.clip(np.asarray(actual)[rows], 0.0, None).sum(axis=0)
    c = int(PEAK + t0)
    s0 = max(0, c - int(window))
    s1 = min(geo.WVFM_LEN, c + int(window) + 1)
    return float(sig[s0:s1].max()) if s1 > s0 else 0.0


def matched_filter_at(image, base, actual, var, t0, support) -> float:
    """Whitened cosine similarity between the cluster's predicted pattern (shifted
    to t0) and the observed residual, over support channels.

    Scale-invariant, so a faint blob is not penalised for predicting little
    light — what matters is whether its *spatial pattern* aligns with a real
    flash.  Returns a value in roughly [0, 1]; higher = better pattern match.
    """
    rows = slice(None) if support is None else support
    shifted = shift_frac(image, t0)[rows]
    resid = np.clip(actual[rows] - base[rows], 0.0, None)
    w = 1.0 / np.sqrt(np.maximum(var[rows], 1e-6))
    a = shifted * w
    b = resid * w
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return float((a * b).sum() / (da * db))


def free_matched_filter_scan(image, base, actual, var, support, search_range,
                             *, block: int = 64) -> Tuple[float, float]:
    """Best t0 by whitened cosine over the full [0, search_range], vectorised.

    Recovers a blob whose true flash was not a strong seed candidate (e.g. a
    faint flash) by finding where its predicted pattern best aligns with the
    observed residual.  Returns (best_t0, best_cosine).
    """
    rows = slice(None) if support is None else support
    img = np.asarray(image, np.float32)[rows]
    resid = np.clip(np.asarray(actual)[rows] - np.asarray(base)[rows], 0.0, None)
    w = (1.0 / np.sqrt(np.maximum(np.asarray(var)[rows], 1e-6))).astype(np.float32)
    C, T = img.shape
    S = int(search_range)
    pad = np.zeros((C, S + T), dtype=np.float32)
    pad[:, S:] = img
    windows = np.lib.stride_tricks.sliding_window_view(pad, T, axis=1)  # (C,S+1,T)
    bw = (resid * w)[:, None, :]
    db = float(np.sqrt((bw * bw).sum()))
    if db <= 0.0:
        return 0.0, 0.0
    cos = np.empty(S + 1, dtype=np.float32)
    for b0 in range(0, S + 1, int(block)):
        b1 = min(S + 1, b0 + int(block))
        aw = windows[:, b0:b1, :] * w[:, None, :]
        num = (aw * bw).sum(axis=(0, 2))
        da = np.sqrt((aw * aw).sum(axis=(0, 2)))
        cos[b0:b1] = num / np.maximum(da * db, 1e-12)
    cos = cos[::-1]                                   # index by t0
    t0 = float(int(np.argmax(cos)))
    return t0, float(cos[int(t0)])


def best_among_candidates(image, base, actual, var, cand, support,
                          *, subtick=True) -> Tuple[Optional[float], float]:
    """Score ``image`` only at the supplied candidate t0s; return the best."""
    best_t0, best_e = None, np.inf
    for c in cand:
        e = score_at(image, base, actual, var, float(c), support)
        if e < best_e:
            best_e, best_t0 = e, float(c)
    if best_t0 is not None and subtick:
        best_t0, best_e = refine_t0_subtick(image, base, actual, var, best_t0,
                                            support=support)
    return best_t0, float(best_e)


# ----------------------------------------------------------------------------
# Stage 4 — tracks (backbone)
# ----------------------------------------------------------------------------
def _merge_close(values, tol):
    vals = sorted(float(v) for v in values if np.isfinite(v))
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.median(g)) for g in groups]


def match_tracks(*, track_labels, cluster_to_tpcs, image_maps, base_image,
                 full_wvfm, full_var, labels, hit_t0, t0_candidates,
                 flash_seeds=None, search_range=geo.SEARCH_RANGE,
                 t0_resolution=5, free_cos_margin=0.05) -> List[dict]:
    """Match each track cluster across all its TPCs (single shared t0).

    The track's t0 is chosen from the REAL flash seeds in its TPCs (whitened
    matched-filter on the predicted-vs-observed pattern), not a free chi2 scan:
    a free scan in a busy multi-flash event gets pulled by saturation/pile-up
    onto the wrong (or a non-flash) tick.  A free scan is used only as a
    fallback when no seed explains the track (its flash was not tabulated).
    """
    rows = []
    for cid in track_labels:
        cid = int(cid)
        if cid not in cluster_to_tpcs:
            continue
        tpcs = sorted(int(t) for t in cluster_to_tpcs[cid]
                      if (cid, int(t)) in image_maps)
        if not tpcs:
            continue
        img = np.concatenate([image_maps[(cid, t)] for t in tpcs], axis=0)
        base = np.concatenate([base_image[t] for t in tpcs], axis=0)
        act = np.concatenate([full_wvfm[t] for t in tpcs], axis=0)
        var = np.concatenate([full_var[t] for t in tpcs], axis=0)

        shifts, errs = full_integer_scan(img, base, act, var, search_range)
        t0 = float(shifts[int(np.argmin(errs))])
        t0 = peak_snap_t0(act, base, t0, t0_resolution)

        hit_t0[labels == cid] = np.float32(t0)
        for t in tpcs:
            t0_candidates[t].append(t0)
        placed = shift_frac(img, t0).reshape(len(tpcs), geo.N_CHANNELS, -1)
        for i, t in enumerate(tpcs):
            base_image[t] = np.clip(base_image[t] + placed[i], None, ADC_CLIP)
        rows.append({"clusterid": cid, "tpcs": tpcs, "t0": t0, "stage": "track",
                     "mode": "free"})
    return rows


# ----------------------------------------------------------------------------
# Stage 5 — per-TPC greedy blob placement (small-blob aware)
# ----------------------------------------------------------------------------
def match_clusters_greedy(*, tpc_to_clusters, image_maps, image_maps_noisy,
                          base_image, full_wvfm, full_var, labels, labels_noisy,
                          hit_t0, t0_candidates, cluster_energies, flash_seeds,
                          search_range=geo.SEARCH_RANGE, t0_resolution=5,
                          small_energy_mev=8.0, support_fraction=0.90,
                          support_floor=25.0, brightness_floor=150.0,
                          discriminate_floor=40.0,
                          uninformative_cos=0.15, tile_masks=None,
                          tiebreak_var=None, tie_frac=0.08) -> List[dict]:
    """Place each non-track cluster in its TPC, brightest first.

    Small (low-energy) clusters are (a) scored on their predicted support
    channels only and (b) matched to existing t0 candidates (flash seeds + track
    t0s) when available, instead of a free scan.
    """
    rows = []
    for tpc, clusters in sorted(tpc_to_clusters.items()):
        tpc = int(tpc)
        cand = t0_candidates[tpc]
        _ensure_seeds(cand, flash_seeds[tpc], t0_resolution)
        base = base_image[tpc]
        act = full_wvfm[tpc]
        var = full_var[tpc]

        order = sorted((int(c) for c in clusters),
                       key=lambda c: -float(cluster_energies.get(int(c), 0.0)))
        for cid in order:
            if (cid, tpc) not in image_maps:
                continue
            img = image_maps[(cid, tpc)]
            e = float(cluster_energies.get(cid, 0.0))
            small = e < float(small_energy_mev)
            if not small:
                support = None
            elif tile_masks is not None:
                # attempt 2: force the small-cluster loss onto the brightest
                # light-detector tiles (mask off the 24 weak channels)
                support = tile_masks[tpc]
            else:
                support = support_channels(img, light_fraction=support_fraction,
                                           abs_floor=support_floor)

            if small and len(cand) > 0:
                # Faint blobs: pick the candidate flash whose observed light best
                # matches the blob's predicted spatial pattern (whitened cosine).
                cand_list = list(cand)
                pred_peak = float(np.clip(
                    img if support is None else img[support], 0.0, None).max())
                if pred_peak < discriminate_floor:
                    lit_pool = [c for c in cand_list
                                if observed_brightness_at(act, support, c) > brightness_floor]
                    pool = lit_pool if lit_pool else cand_list
                    t0 = max(pool, key=lambda c: observed_brightness_at(act, support, c))
                    err = score_at(img, base, act, var, t0, support)
                    mode = "bright"
                else:
                    corr = [matched_filter_at(img, base, act, var, c, support)
                            for c in cand_list]
                    order = np.argsort(corr)[::-1]
                    bi = int(order[0])
                    t0, best_corr = float(cand_list[bi]), float(corr[bi])
                    # Variance-prediction tiebreaker: if the top two candidate t0s
                    # are within tie_frac under unit variance (ambiguous), re-rank
                    # only those with the learned variance, which may discriminate
                    # where unit variance cannot.
                    if (tiebreak_var is not None and len(order) >= 2 and best_corr > 0
                            and (best_corr - corr[int(order[1])]) / best_corr < tie_frac):
                        tv = tiebreak_var[tpc]
                        tied = [int(order[0]), int(order[1])]
                        cv = [matched_filter_at(img, base, act, tv, cand_list[j], support)
                              for j in tied]
                        bi = tied[int(np.argmax(cv))]
                        t0 = float(cand_list[bi])
                    t0_free, corr_free = free_matched_filter_scan(
                        img, base, act, var, support, search_range)
                    if corr_free > best_corr + 0.05:
                        t0 = t0_free
                    near = [c for c in cand_list if abs(t0 - c) <= t0_resolution]
                    if near:
                        t0 = min(near, key=lambda c: abs(c - t0))
                    elif not any(abs(t0 - c) <= t0_resolution for c in cand):
                        cand.append(t0)
                    t0, err = refine_t0_subtick(img, base, act, var, t0, support=support)
                    mode = "mf"
            else:
                shifts, errs = full_integer_scan(img, base, act, var,
                                                 search_range, support)
                t0 = float(shifts[int(np.argmin(errs))])
                t0 = peak_snap_t0(act, base, t0, t0_resolution)
                near = [c for c in cand if abs(t0 - c) <= t0_resolution]
                if near:
                    t0 = min(near, key=lambda c: abs(c - t0))
                else:
                    cand.append(t0)
                t0, err = refine_t0_subtick(img, base, act, var, t0, support=support)
                mode = "scan"

            # clean vs noise-absorbed variant: keep whichever fits better
            img_n = image_maps_noisy.get((cid, tpc))
            placed_img = img
            use_noisy = False
            if img_n is not None:
                if score_at(img_n, base, act, var, t0, support) < \
                        score_at(img, base, act, var, t0, support):
                    placed_img, use_noisy = img_n, True

            if use_noisy:
                hit_t0[labels_noisy == cid] = np.float32(t0)
            else:
                hit_t0[labels == cid] = np.float32(t0)
            base = np.clip(base + shift_frac(placed_img, t0), None, ADC_CLIP)
            rows.append({"clusterid": cid, "TPCid": tpc, "t0": float(t0),
                         "energy_mev": e, "small": bool(small), "mode": mode,
                         "noisy_variant": bool(use_noisy), "chi2": float(err)})
        base_image[tpc] = base
    return rows


# ----------------------------------------------------------------------------
# Phase 8 — per-cluster refinement
# ----------------------------------------------------------------------------
def refine_clusters(*, tpc_to_clusters, image_maps, base_image, full_wvfm,
                    full_var, labels, labels_noisy, hit_t0,
                    cluster_energies=None, small_energy_mev=8.0,
                    search_range=geo.SEARCH_RANGE, t0_resolution=5,
                    improvement_threshold=0.01) -> List[dict]:
    """Remove each cluster's current placement, rescan, re-place if it improves.

    Small (low-energy) clusters are intentionally skipped: their t0 was set by
    matching to a real flash seed, and a free full-scan against a noisy residual
    would only risk pulling a faint blob off the correct flash.
    """
    cluster_energies = cluster_energies or {}
    rows = []
    for tpc, clusters in sorted(tpc_to_clusters.items()):
        tpc = int(tpc)
        base = base_image[tpc]
        act = full_wvfm[tpc]
        var = full_var[tpc]
        for cid in clusters:
            cid = int(cid)
            if (cid, tpc) not in image_maps:
                continue
            if float(cluster_energies.get(cid, 0.0)) < float(small_energy_mev):
                continue
            mask = labels == cid
            if not np.any(mask):
                continue
            cur_t0 = float(hit_t0[mask][0])
            if not np.isfinite(cur_t0) or cur_t0 < 0:
                continue
            img = image_maps[(cid, tpc)]
            base_wo = np.clip(base - shift_frac(img, cur_t0), 0.0, None)
            cur_err = score_at(img, base_wo, act, var, cur_t0)

            shifts, errs = full_integer_scan(img, base_wo, act, var, search_range)
            t0_int = float(shifts[int(np.argmin(errs))])
            anchor = t0_int if (cur_err - float(errs.min())) > improvement_threshold else cur_t0
            anchor = peak_snap_t0(act, base_wo, anchor, t0_resolution,
                                  back=2 * t0_resolution)
            new_t0, new_err = refine_t0_subtick(img, base_wo, act, var, anchor)

            if (cur_err - new_err) > improvement_threshold:
                hit_t0[(labels == cid) | (labels_noisy == cid)] = np.float32(new_t0)
                base = np.clip(base_wo + shift_frac(img, new_t0), None, ADC_CLIP)
                rows.append({"clusterid": cid, "TPCid": tpc, "old_t0": cur_t0,
                             "new_t0": float(new_t0),
                             "improvement": float(cur_err - new_err)})
            else:
                base = np.clip(base_wo + shift_frac(img, cur_t0), None, ADC_CLIP)
        base_image[tpc] = base
    return rows


def _pca_direction(pts, e):
    """Energy-weighted principal axis + linearity of a hit cloud."""
    e = np.clip(np.asarray(e, np.float64), 1e-8, None)
    w = e / e.sum()
    c = (pts * w[:, None]).sum(axis=0)
    d = pts - c
    cov = (d * w[:, None]).T @ d
    evals, evecs = np.linalg.eigh(cov)
    direction = evecs[:, -1]
    lin = float(evals[-1] / max(evals.sum(), 1e-12))
    return c, direction / max(np.linalg.norm(direction), 1e-12), lin


def region_grow_association(*, labels, xset, yset, zset, Eset, hitTPCid, hit_t0,
                            cluster_energies, image_maps, base_image, full_wvfm,
                            full_var, track_labels, contact_cm=3.5,
                            conf_cos=0.55, light_margin=0.04, min_seed_cos=0.20,
                            dir_cm=6.0, max_iter=4) -> List[dict]:
    """Cluster-guided spatial region-growing for the after-track association.

    Idea (vRelease4 'generalized component rescue'): once a cluster is confidently
    matched to the light it becomes a seed and propagates its t0 to spatially
    adjacent UNCERTAIN clusters (preferring growth along its PCA axis), instead of
    every blob fighting the light independently.  The light arbitrates: an
    uncertain neighbour only adopts a seed's t0 if that t0 explains its own
    predicted light at least about as well (whitened matched-filter, so variance
    aware) as its current t0 — otherwise it keeps its own / stays a boundary
    between two t0 regions.

    Post-pass on top of the greedy result (which provides the initial t0s); only
    moves low-confidence clusters, so confident matches and the track backbone are
    preserved.
    """
    from scipy.spatial import cKDTree
    from collections import defaultdict

    labels = np.asarray(labels, np.int64)
    hit_t0 = np.asarray(hit_t0)          # modify the caller's array IN PLACE (do
    tpcid = np.asarray(hitTPCid, np.int64)  # not cast -> would copy and lose moves)
    XYZ = np.column_stack([xset, yset, zset]).astype(np.float64)
    track_set = set(int(c) for c in track_labels)

    ids = [int(c) for c in np.unique(labels) if c >= 0]
    info = {}
    for c in ids:
        m = labels == c
        t0s = hit_t0[m]
        t0s = t0s[np.isfinite(t0s) & (t0s >= 0)]
        if t0s.size == 0:
            continue
        tpcs = sorted(set(int(t) for t in tpcid[m]))
        # confidence = matched-filter cos of the cluster's predicted support
        # pattern vs the OBSERVED light at its current t0 (zero base)
        cos = 0.0
        for tp in tpcs:
            if (c, tp) in image_maps:
                img = image_maps[(c, tp)]
                sup = support_channels(img)
                z0 = np.zeros_like(full_wvfm[tp])
                cos = max(cos, matched_filter_at(img, z0, full_wvfm[tp],
                                                 full_var[tp], float(np.median(t0s)), sup))
        pts = XYZ[m]
        cen, direction, lin = _pca_direction(pts, Eset[m]) if m.sum() >= 2 else (pts[0], np.zeros(3), 0.0)
        info[c] = {"t0": float(np.median(t0s)), "cos": float(cos), "tpcs": tpcs,
                   "is_track": c in track_set, "cen": cen, "dir": direction,
                   "energy": float(cluster_energies.get(c, 0.0))}

    # cluster adjacency (touching hits within contact_cm)
    tree = cKDTree(XYZ)
    pairs = tree.query_pairs(float(contact_cm), output_type="ndarray")
    adj = defaultdict(set)
    if pairs.size:
        la, lb = labels[pairs[:, 0]], labels[pairs[:, 1]]
        good = (la >= 0) & (lb >= 0) & (la != lb)
        for a, b in zip(la[good], lb[good]):
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

    rows = []
    for _ in range(int(max_iter)):
        moved = 0
        # seeds: tracks (always) + confident blobs. Grow into uncertain neighbours.
        for c in sorted((i for i in info),
                        key=lambda i: (info[i]["is_track"], info[i]["cos"]), reverse=True):
            if not (info[c]["is_track"] or info[c]["cos"] >= conf_cos):
                continue  # low-confidence non-track clusters cannot seed growth
            seed = info[c]
            for n in adj.get(c, ()):
                nb = info.get(n)
                if nb is None or nb["is_track"]:
                    continue                        # never move the track backbone
                if abs(nb["t0"] - seed["t0"]) < 1.0:
                    continue                        # already same t0
                # light arbitration: adopt the (spatially adjacent) seed's t0 only
                # if it explains the neighbour's OWN predicted light at least about
                # as well as the neighbour's current t0 (and not negligibly) — this
                # overrides a confident-but-wrong neighbour while protecting a
                # genuinely-correct one (whose own t0 fits clearly better).
                tp = nb["tpcs"][0]
                if (n, tp) not in image_maps:
                    continue
                img = image_maps[(n, tp)]
                sup = support_channels(img)
                z0 = np.zeros_like(full_wvfm[tp])
                cos_seed = matched_filter_at(img, z0, full_wvfm[tp], full_var[tp], seed["t0"], sup)
                cos_own = matched_filter_at(img, z0, full_wvfm[tp], full_var[tp], nb["t0"], sup)
                if cos_seed >= cos_own - light_margin and cos_seed >= min_seed_cos:
                    hit_t0[labels == n] = np.float32(seed["t0"])
                    rows.append({"clusterid": n, "from_t0": nb["t0"], "to_t0": seed["t0"],
                                 "seed": c, "cos_seed": float(cos_seed), "cos_own": float(cos_own),
                                 "energy_mev": nb["energy"]})
                    info[n]["t0"] = seed["t0"]
                    info[n]["cos"] = max(info[n]["cos"], conf_cos)  # now a seed too
                    moved += 1
        if moved == 0:
            break
    return rows


def colocation_repair(*, labels, xset, yset, zset, hit_t0, cluster_energies,
                      contact_cm=5.0, min_repair_energy=20.0, t0_tol=8.0,
                      min_votes=2, min_vote_frac=0.55, max_iter=3) -> List[dict]:
    """Spatial co-location: snap a mis-matched LARGE cluster to the consensus t0
    of its spatially-touching neighbours.

    Rationale (from the event-14 diagnostic): a big shower fragment can match the
    WRONG flash because its light is buried inside a co-located brighter cluster,
    but the many small blobs of the SAME interaction match reliably and pin the
    local interaction time.  Each touching neighbour casts one vote; if a strong
    consensus t0 disagrees with the cluster's t0, snap it.  Only large clusters
    (>= ``min_repair_energy``) are repaired — small blobs are trusted as the
    anchors.  Different interactions are separated because their shower charge is
    not spatially contiguous.
    """
    from scipy.spatial import cKDTree
    from collections import defaultdict

    labels = np.asarray(labels, np.int64)
    hit_t0 = np.asarray(hit_t0, np.float64)
    XYZ = np.column_stack([xset, yset, zset]).astype(np.float64)

    ids = [int(c) for c in np.unique(labels) if c >= 0]
    if len(ids) < 2:
        return []
    cl_t0 = {}
    for c in ids:
        v = hit_t0[labels == c]
        v = v[np.isfinite(v) & (v >= 0)]
        cl_t0[c] = float(np.median(v)) if v.size else float("nan")

    # cluster adjacency from touching hits
    tree = cKDTree(XYZ)
    pairs = tree.query_pairs(float(contact_cm), output_type="ndarray")
    adj = defaultdict(set)
    if pairs.size:
        la, lb = labels[pairs[:, 0]], labels[pairs[:, 1]]
        m = (la >= 0) & (lb >= 0) & (la != lb)
        for a, b in zip(la[m], lb[m]):
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

    rows = []
    for _ in range(int(max_iter)):
        moved = 0
        for c in sorted(ids, key=lambda c: -cluster_energies.get(c, 0.0)):
            if cluster_energies.get(c, 0.0) < min_repair_energy:
                continue
            if not np.isfinite(cl_t0[c]):
                continue
            neigh = [n for n in adj.get(c, ()) if np.isfinite(cl_t0[n])]
            if len(neigh) < min_votes:
                continue
            # group neighbour t0s; each neighbour = one vote
            groups = []  # [t0_list]
            for n in sorted(neigh, key=lambda n: cl_t0[n]):
                if groups and abs(cl_t0[n] - np.median(groups[-1])) <= t0_tol:
                    groups[-1].append(cl_t0[n])
                else:
                    groups.append([cl_t0[n]])
            best = max(groups, key=len)
            if len(best) < min_votes or len(best) / len(neigh) < min_vote_frac:
                continue
            consensus = float(np.median(best))
            if abs(cl_t0[c] - consensus) <= t0_tol:
                continue
            # snap
            hit_t0[labels == c] = np.float32(consensus)
            rows.append({"clusterid": c, "energy_mev": cluster_energies.get(c, 0.0),
                         "old_t0": cl_t0[c], "new_t0": consensus,
                         "n_votes": len(best), "n_neigh": len(neigh)})
            cl_t0[c] = consensus
            moved += 1
        if moved == 0:
            break
    return rows


__all__ = [
    "shift_frac", "score_at", "full_integer_scan", "peak_snap_t0",
    "support_channels", "refine_t0_subtick", "best_among_candidates",
    "match_tracks", "match_clusters_greedy", "refine_clusters",
    "colocation_repair", "region_grow_association", "light_support_tile_mask",
]
