"""
Trial2 Combined Rescue (Phase 2.5) for the 2x2 charge-light matching v4 pipeline.

Adapted from M5p1/phase25_trial2.py + phase25_trial2_combined.py +
phase25_amendment.py, simplified for the 2x2 detector:
  - 8 TPCs, 48 channels, 1000-tick waveforms (no light<->charge remap)
  - No YAML geometry; no shower workflow; no streaming voxelization
  - Default ``light_veto_track_min_tpcs = 2`` (down from 4) so even a
    cosmic that crosses just two TPCs is protected from light-only moves

Three sub-stages, applied in order to the namespace state:

  (a) ``run_large_cluster_flash_grid_correction`` — for every large
      (>= ``large_cluster_min_energy_mev``) non-track cluster, fine-correct
      its (cluster, tpc) t0 against the flash table on a sub-tick grid.

  (b) ``run_spatial_mixed_t0_repair`` — find pairs of nearby t0s that share
      charge in the same TPC, fit per-t0 trimmed-PCA spatial models, and
      reassign each connected component to whichever t0 best explains it.

  (c) ``run_physical_light_repair`` — for every (tpc, source_t0) where the
      current model overflows the measured light, propose moves of donor
      components to other candidate t0s; accept only those that pass the
      physical chi^2 dChi gate (srcRed >= ``phys_min_source_ofch_reduction``,
      dChi >= ``phys_min_dchi2_improvement``, dChi/E >= ``phys_min_dchi2_per_mev``)
      AND the multi-TPC track veto.

Each accepted move triggers an exact family rebuild via the supplied CNN
``predict_family_image_fn`` so ``baseImage`` stays consistent with the
per-hit ``hit_timestamps``.

Dependencies (caller passes in):
  - ``predict_family_image_fn(xs, ys, zs, es, tpc, t0_label) -> (48, 1000)``
    A function that voxelises the supplied hits and returns the unshifted
    predicted ADC waveform image for one (tpc, family) group. The notebook
    typically wraps ``ML_3DCNN_v3_1.process_clusters_to_imageMaps`` for this.
  - ``imageMaps[(cluster_id, tpc_id)]`` from the first-stage CNN run, used by
    the light-repair component proxy when no rebuild is needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Trial2Config:
    # Generic.
    verbose: bool = True
    pulse_peak_tick: int = 105
    half_window_ticks: int = 18
    pad_ticks: int = 12
    adc_clip: float = 60780.0
    overflow_weight: float = 4.0
    nt: int = 1000
    n_channels: int = 48
    n_tpcs: int = 8

    # Sub-step (a): large-cluster flash-grid fine correction.
    enable_large_flash_grid_correction: bool = True
    large_cluster_min_energy_mev: float = 50.0
    large_grid_offsets_ticks: tuple = (-1.0, -0.5, 0.0, 0.5, 1.0)
    large_min_loss_improvement: float = 0.0
    large_max_clusters: int | None = None

    # Sub-step (b): spatial mixed-t0 repair.
    enable_spatial: bool = True
    spatial_t0_match_ticks: float = 10.0
    spatial_different_t0_ticks: float = 10.0
    spatial_contact_radius_cm: float = 4.0
    spatial_max_pairs: int = 24
    spatial_min_pair_hits: int = 24
    spatial_min_pair_energy_mev: float = 3.0
    spatial_component_radius_cm: float = 2.6
    spatial_smooth_component_radius_cm: float = 2.8
    spatial_min_model_hits: int = 8
    spatial_min_model_energy_mev: float = 0.05
    spatial_max_models_per_t0: int = 16
    spatial_trim_model_quantile: float = 0.85
    spatial_trim_iterations: int = 2
    spatial_axis_width_floor_cm: float = 0.55
    spatial_nearest_scale_cm: float = 2.2
    spatial_endpoint_gap_scale_cm: float = 6.0
    spatial_endpoint_margin_cm: float = 10.0
    spatial_max_accept_score: float = 3.0
    spatial_rescan_pool_margin: float = 0.75
    spatial_component_strong_margin: float = 0.08
    spatial_move_margin: float = 0.08
    spatial_keep_inertia: float = 0.0
    spatial_min_moved_hits: int = 1

    # Sub-step (c): physical-chi^2 light repair.
    enable_light: bool = True
    light_t0_match_ticks: float = 5.0
    light_overflow_sigma: float = 3.0
    light_overflow_abs_adc: float = 400.0
    light_model_activity_adc: float = 400.0
    light_min_overflow_channels: int = 6
    light_max_total_moves: int = 24
    light_max_moves_per_tpc: int = 3
    light_component_radius_cm: float = 3.0
    light_min_component_hits: int = 4
    light_min_component_energy_mev: float = 0.20
    light_max_components_per_t0: int = 80
    light_t0_merge_ticks: float = 5.0
    light_flash_cluster_match_ticks: float = 5.0

    # Physical chi^2 gate (the headline acceptance criterion).
    phys_min_source_ofch_reduction: int = 8
    phys_min_dchi2_improvement: float = 5.0e2
    phys_min_dchi2_per_mev: float = 10.0
    phys_std_floor: float = 1.0e-6

    # Multi-TPC track veto.
    light_veto_multitpc_track: bool = True
    # 2x2 default = 2 (cosmic crossing 2 of 8 TPCs is already track-like).
    light_veto_track_min_tpcs: int = 2
    light_veto_override_min_candidates: int = 3


# ---------------------------------------------------------------------------
# Image shift / loss helpers (kept self-contained)
# ---------------------------------------------------------------------------

def _shift_fractional(image, shift_ticks, *, nt):
    img = np.asarray(image, dtype=np.float32)
    out = np.zeros((img.shape[0], int(nt)), dtype=np.float32)
    src_x = np.arange(img.shape[1], dtype=np.float32)
    dst_x = np.arange(int(nt), dtype=np.float32) - float(shift_ticks)
    for ch in range(img.shape[0]):
        out[ch] = np.interp(dst_x, src_x, img[ch], left=0.0, right=0.0).astype(np.float32)
    return out


def _shift_integer(image, t0, *, nt=None):
    img = np.asarray(image, dtype=np.float32)
    if nt is None:
        nt = img.shape[1]
    out = np.zeros((img.shape[0], int(nt)), dtype=np.float32)
    t0 = int(round(float(t0)))
    if t0 >= 0:
        if t0 < nt:
            n = min(img.shape[1], nt - t0)
            out[:, t0:t0 + n] = img[:, :n]
    else:
        src0 = -t0
        if src0 < img.shape[1]:
            n = min(img.shape[1] - src0, nt)
            out[:, :n] = img[:, src0:src0 + n]
    return out


def _window_mask(nt, t0_values, cfg: Trial2Config):
    mask = np.zeros(int(nt), dtype=bool)
    for t0 in t0_values:
        tick = int(round(float(t0))) + int(cfg.pulse_peak_tick)
        lo = max(0, tick - int(cfg.half_window_ticks) - int(cfg.pad_ticks))
        hi = min(int(nt), tick + int(cfg.half_window_ticks) + int(cfg.pad_ticks) + 1)
        if hi > lo:
            mask[lo:hi] = True
    if not mask.any():
        mask[:] = True
    return mask


def _weighted_loss_window(model_tpc, actual_tpc, std_tpc, t0_values, cfg: Trial2Config):
    tmask = _window_mask(model_tpc.shape[-1], t0_values, cfg)
    m = model_tpc[:, tmask].astype(np.float32)
    a = actual_tpc[:, tmask].astype(np.float32)
    s = np.maximum(std_tpc[:, tmask].astype(np.float32), 1e-6)
    w = np.where(m > a, float(cfg.overflow_weight), 1.0).astype(np.float32)
    return float(np.sum(((m - a) ** 2 / s) * w))


def _physical_chi2_window_loss(*, model_tpc, actual_tpc, std_tpc, keep,
                               t0_values, cfg: Trial2Config, std_floor):
    n_ticks = int(actual_tpc.shape[1])
    tmask = _window_mask(n_ticks, t0_values, cfg)
    if not tmask.any() or len(keep) == 0:
        return 0.0, 0
    pred = model_tpc[keep][:, tmask].astype(np.float64)
    actual = actual_tpc[keep][:, tmask].astype(np.float64)
    std = np.maximum(std_tpc[keep][:, tmask].astype(np.float64), float(std_floor))
    chi2 = float(np.sum(((pred - actual) / std) ** 2))
    return chi2, int(pred.size)


def _keep_channel_indices(tpc, actual_full, saturated_channel_cache=None):
    if saturated_channel_cache is not None:
        try:
            veto = np.asarray(saturated_channel_cache["veto_mask"][int(tpc)], dtype=bool)
            keep = np.flatnonzero(~veto).astype(np.int32)
            if keep.size:
                return keep
        except Exception:
            pass
    actual = np.asarray(actual_full[int(tpc)], dtype=np.float32)
    veto = np.sum(actual > 60700.0, axis=1) > 6
    return np.flatnonzero(~veto).astype(np.int32)


# ---------------------------------------------------------------------------
# Spatial primitives
# ---------------------------------------------------------------------------

def _connected_components(points, radius_cm):
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=np.int64)]
    pairs = cKDTree(pts).query_pairs(float(radius_cm), output_type="ndarray")
    parent = np.arange(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return int(a)

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def _fit_trimmed_pca(points, energies, cfg: Trial2Config):
    pts_all = np.asarray(points, dtype=np.float64)
    if pts_all.shape[0] == 0:
        return None
    e_all = np.clip(np.asarray(energies, dtype=np.float64), 1e-9, None)
    keep = np.ones(pts_all.shape[0], dtype=bool)
    centroid = np.mean(pts_all, axis=0)
    direction = np.array([1.0, 0.0, 0.0])
    linearity = 0.0

    for _ in range(max(1, int(cfg.spatial_trim_iterations))):
        pts = pts_all[keep]
        e = e_all[keep]
        if pts.shape[0] < 2:
            centroid = pts[0].copy() if pts.shape[0] == 1 else centroid
            break
        w = e / max(float(np.sum(e)), 1e-12)
        centroid = np.sum(pts * w[:, None], axis=0)
        centered = pts - centroid
        cov = (centered * w[:, None]).T @ centered
        try:
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            direction = evecs[:, 0]
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            linearity = float(evals[0] / max(float(np.sum(evals)), 1e-12))
        except np.linalg.LinAlgError:
            direction = np.array([1.0, 0.0, 0.0])
            linearity = 0.0

        rel = pts_all - centroid
        proj_all = rel @ direction
        perp_all = rel - proj_all[:, None] * direction[None, :]
        perp_dist = np.sqrt(np.sum(perp_all * perp_all, axis=1))
        if pts_all.shape[0] >= 8:
            cut = float(np.quantile(perp_dist, float(cfg.spatial_trim_model_quantile)))
            keep_new = perp_dist <= max(cut, float(cfg.spatial_axis_width_floor_cm))
            if np.count_nonzero(keep_new) >= max(3, int(0.40 * pts_all.shape[0])):
                keep = keep_new

    rel = pts_all - centroid
    proj = rel @ direction
    perp = rel - proj[:, None] * direction[None, :]
    perp_dist = np.sqrt(np.sum(perp * perp, axis=1))
    fit_perp = perp_dist[keep] if keep.any() else perp_dist
    width = max(float(np.quantile(fit_perp, 0.68)),
                float(cfg.spatial_axis_width_floor_cm))
    fit_proj = proj[keep] if keep.any() else proj
    return {
        "centroid": centroid, "direction": direction,
        "proj_min": float(np.min(fit_proj)) if fit_proj.size else 0.0,
        "proj_max": float(np.max(fit_proj)) if fit_proj.size else 0.0,
        "width_cm": float(width), "linearity": float(linearity),
        "tree": cKDTree(pts_all), "points": pts_all,
        "energy_mev": float(np.sum(e_all)),
        "n_hits": int(pts_all.shape[0]),
        "n_fit_hits": int(np.count_nonzero(keep)),
    }


def _score_points_to_model(points, model, cfg: Trial2Config):
    pts = np.asarray(points, dtype=np.float64)
    if model is None:
        return np.full(pts.shape[0], 1e9)
    rel = pts - model["centroid"][None, :]
    proj = rel @ model["direction"]
    perp = rel - proj[:, None] * model["direction"][None, :]
    perp_dist = np.sqrt(np.sum(perp * perp, axis=1))
    nearest, _ = model["tree"].query(pts, k=1)
    nearest = np.asarray(nearest, dtype=np.float64)
    margin = float(cfg.spatial_endpoint_margin_cm)
    low_gap = np.maximum(model["proj_min"] - proj - margin, 0.0)
    high_gap = np.maximum(proj - model["proj_max"] - margin, 0.0)
    endpoint_gap = low_gap + high_gap
    width = max(float(model["width_cm"]), float(cfg.spatial_axis_width_floor_cm))
    if model["linearity"] >= 0.85:
        return (0.70 * (perp_dist / width)
                + 0.20 * (nearest / float(cfg.spatial_nearest_scale_cm))
                + 0.10 * (endpoint_gap / float(cfg.spatial_endpoint_gap_scale_cm)))
    return (0.45 * (perp_dist / width)
            + 0.45 * (nearest / float(cfg.spatial_nearest_scale_cm))
            + 0.10 * (endpoint_gap / float(cfg.spatial_endpoint_gap_scale_cm)))


def _score_points_to_models(points, models, cfg: Trial2Config):
    if not models:
        return (np.full(points.shape[0], 1e9),
                np.full(points.shape[0], -1, dtype=np.int32))
    score_mat = np.vstack([_score_points_to_model(points, m, cfg) for m in models])
    best = np.argmin(score_mat, axis=0).astype(np.int32)
    return score_mat[best, np.arange(points.shape[0])], best


def _build_spatial_models(indices, x, y, z, energy, cfg: Trial2Config):
    if indices.size == 0:
        return []
    pts = np.column_stack((x[indices], y[indices], z[indices])).astype(np.float64)
    comps = _connected_components(pts, float(cfg.spatial_component_radius_cm))
    rows = []
    for cid, comp_local in enumerate(comps):
        gidx = indices[comp_local]
        if gidx.size < int(cfg.spatial_min_model_hits):
            continue
        if float(np.sum(energy[gidx])) < float(cfg.spatial_min_model_energy_mev):
            continue
        m = _fit_trimmed_pca(
            np.column_stack((x[gidx], y[gidx], z[gidx])),
            energy[gidx], cfg,
        )
        if m is None:
            continue
        m["global_indices"] = gidx
        m["component_id"] = int(cid)
        rows.append(m)
    rows.sort(key=lambda m: (-m["energy_mev"], -m["n_hits"]))
    return rows[: int(cfg.spatial_max_models_per_t0)]


# ---------------------------------------------------------------------------
# Family rebuild (exact CNN re-prediction for affected (tpc, t0) groups)
# ---------------------------------------------------------------------------

def _predict_family_images_batch(specs, *, hit_timestamps, hit_tpc_ids,
                                 x, y, z, energy, predict_family_image_fn,
                                 t0_match_ticks, n_channels, nt):
    """Predict the unshifted (channels, nt) image for each (tpc, t0) group.

    Hits assigned to (tpc=spec[0], t0~spec[1]) are voxelised together.
    ``predict_family_image_fn(xs, ys, zs, es, tpc) -> (n_channels, nt)``.
    """
    out = {}
    finite = np.isfinite(hit_timestamps) & (hit_timestamps >= 0)
    for spec in specs:
        tpc, t0 = int(spec[0]), int(spec[1])
        mask = ((np.asarray(hit_tpc_ids, dtype=np.int32) == tpc)
                & finite
                & (np.abs(hit_timestamps - float(t0)) <= float(t0_match_ticks)))
        if not mask.any():
            out[(tpc, t0)] = np.zeros((int(n_channels), int(nt)), dtype=np.float32)
            continue
        idx = np.flatnonzero(mask)
        try:
            img = predict_family_image_fn(
                x[idx].astype(np.float32),
                y[idx].astype(np.float32),
                z[idx].astype(np.float32),
                energy[idx].astype(np.float32),
                int(tpc),
            )
        except Exception as exc:
            print(f"  ! predict_family_image_fn failed for tpc={tpc} t0={t0}: {exc}")
            img = np.zeros((int(n_channels), int(nt)), dtype=np.float32)
        out[(tpc, t0)] = np.asarray(img, dtype=np.float32)
    return out


def _exact_update_affected_families(*, base_image, old_hit_timestamps,
                                    new_hit_timestamps, affected_specs,
                                    hit_tpc_ids, x, y, z, energy,
                                    predict_family_image_fn, cfg: Trial2Config):
    specs = sorted({(int(s[0]), int(s[1])) for s in affected_specs})
    if not specs:
        return base_image, []
    old_imgs = _predict_family_images_batch(
        specs, hit_timestamps=old_hit_timestamps, hit_tpc_ids=hit_tpc_ids,
        x=x, y=y, z=z, energy=energy,
        predict_family_image_fn=predict_family_image_fn,
        t0_match_ticks=cfg.spatial_t0_match_ticks,
        n_channels=int(cfg.n_channels), nt=int(cfg.nt),
    )
    new_imgs = _predict_family_images_batch(
        specs, hit_timestamps=new_hit_timestamps, hit_tpc_ids=hit_tpc_ids,
        x=x, y=y, z=z, energy=energy,
        predict_family_image_fn=predict_family_image_fn,
        t0_match_ticks=cfg.spatial_t0_match_ticks,
        n_channels=int(cfg.n_channels), nt=int(cfg.nt),
    )
    out = np.asarray(base_image, dtype=np.float32).copy()
    nt = out.shape[-1]
    records = []
    for tpc, t0 in specs:
        old_shift = _shift_integer(old_imgs[(tpc, t0)], t0, nt=nt)
        new_shift = _shift_integer(new_imgs[(tpc, t0)], t0, nt=nt)
        delta = new_shift - old_shift
        out[tpc] = np.clip(out[tpc] + delta, 0.0, float(cfg.adc_clip))
        records.append({"TPCid": int(tpc), "t0": int(t0),
                        "old_sum": float(np.sum(old_shift)),
                        "new_sum": float(np.sum(new_shift)),
                        "delta_sum": float(np.sum(delta))})
    return out, records


# ---------------------------------------------------------------------------
# Sub-step (a): large-cluster flash-grid fine correction
# ---------------------------------------------------------------------------

def run_large_cluster_flash_grid_correction(
    *, namespace, cfg: Trial2Config,
):
    """Sub-step (a). Mutates namespace baseImage / hit_timestamps / t0Candidates."""
    base = np.asarray(namespace["baseImage"], dtype=np.float32).copy()
    hit_ts = np.asarray(namespace["hit_timestamps"], dtype=np.float32).copy()
    actual = np.asarray(namespace["fullLightWaveform"], dtype=np.float32)
    std = np.maximum(np.asarray(namespace.get(
        "fullLightStd", np.ones_like(actual)), dtype=np.float32), 1e-6)
    labels = np.asarray(namespace["labels_global"], dtype=np.int64)
    tpcs = np.asarray(namespace["hitTPCid"], dtype=np.int32)
    energy = np.asarray(namespace["Eset"], dtype=np.float64)
    image_maps = namespace["imageMaps"]
    cluster_to_tpcs = namespace.get("cluster_to_tpcs", {})
    track_shower = set(int(v) for v in namespace.get("track_shower_labels", []))
    t0_cands = namespace["t0Candidates"]

    # Filter & sort clusters by per-cluster total energy (largest first).
    candidates = []
    for cid_key in cluster_to_tpcs.keys():
        cid = int(cid_key)
        if cid in track_shower:
            continue
        cmask = labels == cid
        e = float(np.sum(energy[cmask]))
        if e < float(cfg.large_cluster_min_energy_mev):
            continue
        candidates.append((cid, e))
    candidates.sort(key=lambda r: (-r[1], r[0]))
    if cfg.large_max_clusters is not None:
        candidates = candidates[: int(cfg.large_max_clusters)]

    rows = []
    for cid, _ in candidates:
        for tpc in sorted(int(t) for t in cluster_to_tpcs.get(cid, [])):
            key = (cid, int(tpc))
            if key not in image_maps:
                continue
            hit_idx = np.flatnonzero((labels == cid) & (tpcs == tpc)).astype(np.int64)
            if hit_idx.size == 0:
                continue
            cur = hit_ts[hit_idx]
            finite = np.isfinite(cur) & (cur >= 0)
            if not finite.any():
                continue
            current_t0 = float(np.median(cur[finite]))

            flash_t0s = [float(v) for v in t0_cands[int(tpc)]
                         if v is not None and np.isfinite(float(v))]
            if not flash_t0s:
                continue

            cluster_img = np.asarray(image_maps[key], dtype=np.float32)
            old_shifted = _shift_fractional(cluster_img, current_t0, nt=int(cfg.nt))
            base_without = np.clip(base[int(tpc)] - old_shifted, 0.0, None)
            before_loss = _weighted_loss_window(
                base[int(tpc)], actual[int(tpc)], std[int(tpc)],
                [current_t0], cfg)

            best = None  # (loss, cand_t0, model)
            for ft0 in flash_t0s:
                for off in cfg.large_grid_offsets_ticks:
                    cand = float(ft0) + float(off)
                    new_shifted = _shift_fractional(cluster_img, cand, nt=int(cfg.nt))
                    trial_model = np.clip(base_without + new_shifted, 0.0, float(cfg.adc_clip))
                    loss = _weighted_loss_window(
                        trial_model, actual[int(tpc)], std[int(tpc)],
                        [current_t0, cand], cfg)
                    if best is None or loss < best[0]:
                        best = (loss, cand, trial_model)
            if best is None:
                continue
            best_loss, best_t0, best_model = best
            improvement = before_loss - best_loss
            accepted = (improvement >= float(cfg.large_min_loss_improvement)
                        and abs(best_t0 - current_t0) > 1e-3)
            row = {
                "clusterid": int(cid), "TPCid": int(tpc),
                "current_t0": float(current_t0), "best_t0": float(best_t0),
                "before_loss": float(before_loss), "best_loss": float(best_loss),
                "improvement": float(improvement),
                "accepted": bool(accepted),
            }
            rows.append(row)
            if accepted:
                base[int(tpc)] = best_model.astype(np.float32)
                hit_ts[hit_idx] = np.float32(best_t0)
                _canonicalize_candidate_t0(
                    t0_cands, int(tpc), float(best_t0),
                    merge_ticks=float(cfg.light_flash_cluster_match_ticks),
                )

    namespace["baseImage"] = base
    namespace["hit_timestamps"] = hit_ts
    namespace["t0Candidates"] = t0_cands
    return {"rows": rows, "accepted_rows": [r for r in rows if r["accepted"]]}


# ---------------------------------------------------------------------------
# Sub-step (b): spatial mixed-t0 repair
# ---------------------------------------------------------------------------

def _detect_mixed_t0_pairs(*, hit_timestamps, hit_tpc_ids, x, y, z, energy,
                           allowed_tpcs, cfg: Trial2Config,
                           locked_hit_mask=None):
    """``locked_hit_mask`` (bool, len = n_hits): True for hits that must NOT
    be moved (e.g. backbone track hits). Locked hits may still anchor the
    detection of a mixed-t0 pair, but are excluded from the candidate hit
    set returned per pair, and pairs whose endpoints are *both* locked are
    skipped entirely.
    """
    rows = []
    finite = np.isfinite(hit_timestamps) & (hit_timestamps >= 0)
    locked = (np.zeros_like(finite) if locked_hit_mask is None
              else np.asarray(locked_hit_mask, dtype=bool))
    for tpc in sorted(int(v) for v in allowed_tpcs):
        idx = np.flatnonzero((hit_tpc_ids == tpc) & finite).astype(np.int64)
        if idx.size < 2:
            continue
        pts = np.column_stack((x[idx], y[idx], z[idx])).astype(np.float64)
        pairs = cKDTree(pts).query_pairs(float(cfg.spatial_contact_radius_cm),
                                         output_type="ndarray")
        if pairs.size == 0:
            continue
        t0_int = np.round(hit_timestamps[idx]).astype(np.int32)

        pair_hits = {}
        for a, b in pairs:
            ga, gb = int(idx[a]), int(idx[b])
            if locked[ga] and locked[gb]:
                continue
            ta, tb = int(t0_int[a]), int(t0_int[b])
            if abs(ta - tb) <= float(cfg.spatial_different_t0_ticks):
                continue
            key = (min(ta, tb), max(ta, tb))
            s = pair_hits.setdefault(key, set())
            if not locked[ga]:
                s.add(ga)
            if not locked[gb]:
                s.add(gb)

        for (ta, tb), hit_set in pair_hits.items():
            pair_idx = np.asarray(sorted(hit_set), dtype=np.int64)
            if pair_idx.size < int(cfg.spatial_min_pair_hits):
                continue
            e = float(np.sum(energy[pair_idx]))
            if e < float(cfg.spatial_min_pair_energy_mev):
                continue
            rows.append({
                "TPCid": int(tpc), "t0_a": int(ta), "t0_b": int(tb),
                "n_hits": int(pair_idx.size), "energy_mev": float(e),
                "hit_indices": pair_idx,
                "severity": float(e * pair_idx.size),
            })
    rows.sort(key=lambda r: (-r["severity"], -r["energy_mev"], r["TPCid"]))
    return rows[: int(cfg.spatial_max_pairs)]


def _restore_mixed_pair_spatially(row, *, hit_timestamps, hit_tpc_ids,
                                  x, y, z, energy, cfg: Trial2Config,
                                  locked_hit_mask=None):
    """Locked hits (e.g. backbone tracks) are still used to *fit* the per-t0
    PCA spatial models, but are excluded from the candidate set that can be
    moved between t0s.
    """
    tpc = int(row["TPCid"]); t0_a = int(row["t0_a"]); t0_b = int(row["t0_b"])
    tpc_mask = hit_tpc_ids == tpc
    finite = np.isfinite(hit_timestamps) & (hit_timestamps >= 0)
    locked = (np.zeros(hit_timestamps.size, dtype=bool) if locked_hit_mask is None
              else np.asarray(locked_hit_mask, dtype=bool))
    mask_a = tpc_mask & finite & (np.abs(hit_timestamps - t0_a)
                                  <= float(cfg.spatial_t0_match_ticks))
    mask_b = tpc_mask & finite & (np.abs(hit_timestamps - t0_b)
                                  <= float(cfg.spatial_t0_match_ticks))
    idx_a = np.flatnonzero(mask_a).astype(np.int64)
    idx_b = np.flatnonzero(mask_b).astype(np.int64)
    if idx_a.size == 0 or idx_b.size == 0:
        return None

    models_a = _build_spatial_models(idx_a, x, y, z, energy, cfg)
    models_b = _build_spatial_models(idx_b, x, y, z, energy, cfg)
    if not models_a or not models_b:
        return None

    idx_pair = np.unique(np.concatenate([idx_a, idx_b])).astype(np.int64)
    pair_pts = np.column_stack((x[idx_pair], y[idx_pair], z[idx_pair])).astype(np.float64)
    score_a, _ = _score_points_to_models(pair_pts, models_a, cfg)
    score_b, _ = _score_points_to_models(pair_pts, models_b, cfg)
    current_is_a = np.abs(hit_timestamps[idx_pair] - t0_a) <= float(cfg.spatial_t0_match_ticks)
    score_current = np.where(current_is_a, score_a, score_b)
    score_other = np.where(current_is_a, score_b, score_a)
    plausible = np.minimum(score_a, score_b) <= float(cfg.spatial_max_accept_score)
    competitive = score_other <= (score_current + float(cfg.spatial_rescan_pool_margin))
    candidate_local = np.flatnonzero(plausible | competitive).astype(np.int64)
    if candidate_local.size == 0:
        return None

    candidate_idx = idx_pair[candidate_local]
    # Exclude locked (backbone) hits from the movable candidate set.
    candidate_idx = candidate_idx[~locked[candidate_idx]]
    if candidate_idx.size == 0:
        return None
    candidate_pts = np.column_stack(
        (x[candidate_idx], y[candidate_idx], z[candidate_idx]))
    comps = _connected_components(candidate_pts,
                                  float(cfg.spatial_smooth_component_radius_cm))

    pair_score_a = {int(g): float(s) for g, s in zip(idx_pair, score_a)}
    pair_score_b = {int(g): float(s) for g, s in zip(idx_pair, score_b)}
    new_t0_by_hit = {}
    move_rows = []

    for comp_id, comp_local in enumerate(comps, start=1):
        comp_idx = candidate_idx[comp_local].astype(np.int64)
        if comp_idx.size == 0:
            continue
        s_a = np.array([pair_score_a[int(i)] for i in comp_idx])
        s_b = np.array([pair_score_b[int(i)] for i in comp_idx])
        cur_a = np.abs(hit_timestamps[comp_idx] - t0_a) <= float(cfg.spatial_t0_match_ticks)
        cur_b = np.abs(hit_timestamps[comp_idx] - t0_b) <= float(cfg.spatial_t0_match_ticks)
        s_a_inertial = s_a.copy(); s_b_inertial = s_b.copy()
        s_a_inertial[cur_b] += float(cfg.spatial_keep_inertia)
        s_b_inertial[cur_a] += float(cfg.spatial_keep_inertia)
        weights = np.clip(energy[comp_idx], 1e-9, None)
        mean_a = float(np.average(s_a_inertial, weights=weights))
        mean_b = float(np.average(s_b_inertial, weights=weights))
        n_cur_a = int(np.count_nonzero(cur_a))
        n_cur_b = int(np.count_nonzero(cur_b))
        current_t0 = t0_a if n_cur_a >= n_cur_b else t0_b
        current_mean = mean_a if current_t0 == t0_a else mean_b
        other_t0 = t0_b if current_t0 == t0_a else t0_a
        other_mean = mean_b if current_t0 == t0_a else mean_a

        old_vals = np.where(cur_a, t0_a, t0_b).astype(np.int32)
        targets = old_vals.copy()
        if other_mean + float(cfg.spatial_component_strong_margin) < current_mean:
            targets[:] = int(other_t0)
        else:
            for j in range(comp_idx.size):
                old_is_a = bool(cur_a[j])
                s_old = s_a_inertial[j] if old_is_a else s_b_inertial[j]
                s_new = s_b_inertial[j] if old_is_a else s_a_inertial[j]
                if s_new + float(cfg.spatial_move_margin) < s_old:
                    targets[j] = t0_b if old_is_a else t0_a

        moved = targets != old_vals
        if not moved.any():
            continue
        moved_idx = comp_idx[moved]
        for g, nt0 in zip(moved_idx, targets[moved]):
            new_t0_by_hit[int(g)] = int(nt0)
        for old_t0, new_t0 in sorted({(int(o), int(n))
                                      for o, n in zip(old_vals[moved], targets[moved])}):
            sel = moved_idx[(old_vals[moved] == old_t0) & (targets[moved] == new_t0)]
            move_rows.append({"component_id": int(comp_id),
                              "old_t0": int(old_t0), "new_t0": int(new_t0),
                              "n_hits": int(sel.size),
                              "energy_mev": float(np.sum(energy[sel])),
                              "hit_indices": sel.astype(np.int64)})

    if len(new_t0_by_hit) < int(cfg.spatial_min_moved_hits):
        return None
    moved_idx = np.asarray(sorted(new_t0_by_hit), dtype=np.int64)
    return {
        "TPCid": int(tpc), "t0_a": int(t0_a), "t0_b": int(t0_b),
        "moved_idx": moved_idx,
        "new_t0": np.array([new_t0_by_hit[int(i)] for i in moved_idx], dtype=np.float32),
        "move_rows": move_rows, "n_moved": int(moved_idx.size),
        "energy_moved_mev": float(np.sum(energy[moved_idx])),
    }


def run_spatial_mixed_t0_repair(*, namespace, cfg: Trial2Config,
                                predict_family_image_fn):
    base = np.asarray(namespace["baseImage"], dtype=np.float32).copy()
    hit_ts = np.asarray(namespace["hit_timestamps"], dtype=np.float32).copy()
    tpcs = np.asarray(namespace["hitTPCid"], dtype=np.int32)
    x = np.asarray(namespace["xset"], dtype=np.float64)
    y = np.asarray(namespace["yset"], dtype=np.float64)
    z = np.asarray(namespace["zset"], dtype=np.float64)
    energy = np.asarray(namespace["Eset"], dtype=np.float64)
    labels = np.asarray(namespace["labels_global"], dtype=np.int64)

    # Build the lock mask: any hit whose cluster id is in track_shower_labels
    # is "locked" — it can shape spatial models but cannot be reassigned.
    track_shower = np.fromiter(
        (int(v) for v in namespace.get("track_shower_labels", [])),
        dtype=np.int64,
    )
    if track_shower.size:
        locked_hit_mask = np.isin(labels, track_shower)
    else:
        locked_hit_mask = np.zeros(labels.size, dtype=bool)

    allowed_tpcs = sorted({int(v) for v in np.unique(tpcs) if int(v) >= 0})
    pair_rows = _detect_mixed_t0_pairs(
        hit_timestamps=hit_ts, hit_tpc_ids=tpcs,
        x=x, y=y, z=z, energy=energy,
        allowed_tpcs=allowed_tpcs, cfg=cfg,
        locked_hit_mask=locked_hit_mask,
    )
    spatial_moves = []
    spatial_trials = []
    family_records = []
    for row in pair_rows:
        trial = _restore_mixed_pair_spatially(
            row, hit_timestamps=hit_ts, hit_tpc_ids=tpcs,
            x=x, y=y, z=z, energy=energy, cfg=cfg,
            locked_hit_mask=locked_hit_mask,
        )
        spatial_trials.append({"pair": row, "trial": trial})
        if trial is None or trial["n_moved"] == 0:
            continue
        before_ts = hit_ts.copy()
        before_base = base.copy()
        moved = trial["moved_idx"]
        new_t0_arr = trial["new_t0"]
        old_t0_arr = before_ts[moved]
        hit_ts[moved] = new_t0_arr.astype(np.float32)

        affected = set()
        for old_t, new_t in zip(old_t0_arr, new_t0_arr):
            try:
                affected.add((int(trial["TPCid"]), int(round(float(old_t)))))
            except Exception:
                pass
            affected.add((int(trial["TPCid"]), int(round(float(new_t)))))
        base, recs = _exact_update_affected_families(
            base_image=before_base,
            old_hit_timestamps=before_ts, new_hit_timestamps=hit_ts,
            affected_specs=affected,
            hit_tpc_ids=tpcs, x=x, y=y, z=z, energy=energy,
            predict_family_image_fn=predict_family_image_fn, cfg=cfg,
        )
        family_records.extend(recs)
        spatial_moves.append(trial)

    namespace["baseImage"] = base
    namespace["hit_timestamps"] = hit_ts
    return {"pair_rows": pair_rows, "spatial_moves": spatial_moves,
            "spatial_trials": spatial_trials,
            "family_update_records": family_records}


# ---------------------------------------------------------------------------
# Sub-step (c): physical-chi^2 light repair
# ---------------------------------------------------------------------------

def _scan_light_overflows_source_only(*, base_image, full_light_waveform,
                                      full_light_std, source_t0s_by_tpc,
                                      saturated_channel_cache, allowed_tpcs,
                                      cfg: Trial2Config):
    rows = []
    actual = np.asarray(full_light_waveform, dtype=np.float32)
    model = np.asarray(base_image, dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std, dtype=np.float32), 1e-6)

    for tpc in sorted(int(v) for v in allowed_tpcs):
        source_t0s = source_t0s_by_tpc.get(int(tpc), [])
        if not source_t0s:
            continue
        keep = _keep_channel_indices(int(tpc), actual, saturated_channel_cache)
        if keep.size == 0:
            continue
        a = actual[int(tpc), keep]
        m = model[int(tpc), keep]
        s = std[int(tpc), keep]
        nt = a.shape[1]
        for t0 in sorted({int(v) for v in source_t0s}):
            tick = int(t0) + int(cfg.pulse_peak_tick)
            if tick < 0 or tick >= nt:
                continue
            lo = max(0, tick - int(cfg.half_window_ticks))
            hi = min(nt, tick + int(cfg.half_window_ticks) + 1)
            threshold = np.maximum(float(cfg.light_overflow_sigma) * s[:, lo:hi],
                                   float(cfg.light_overflow_abs_adc))
            residual = m[:, lo:hi] - a[:, lo:hi]
            overflow = (residual > threshold) & (m[:, lo:hi] > float(cfg.light_model_activity_adc))
            n_of_ch = int(np.count_nonzero(np.any(overflow, axis=1)))
            if n_of_ch < int(cfg.light_min_overflow_channels):
                continue
            peak_threshold = np.maximum(float(cfg.light_overflow_sigma) * s[:, tick],
                                        float(cfg.light_overflow_abs_adc))
            peak_residual = m[:, tick] - a[:, tick]
            peak_overflow = (peak_residual > peak_threshold) & (m[:, tick] > float(cfg.light_model_activity_adc))
            win_of = float(np.sum(np.clip(residual[overflow], 0, None)))
            pk_of = float(np.sum(np.clip(peak_residual[peak_overflow], 0, None)))
            rows.append({
                "TPCid": int(tpc), "t0": int(t0), "peak_tick": int(tick),
                "overflow_channels": int(n_of_ch),
                "peak_overflow_channels": int(np.count_nonzero(peak_overflow)),
                "window_overflow": float(win_of),
                "peak_overflow": float(pk_of),
                "severity": float(pk_of / max(n_of_ch, 1) + 0.002 * win_of),
            })
    rows.sort(key=lambda r: (-r["severity"], -r["window_overflow"],
                             -r["overflow_channels"], r["TPCid"], r["t0"]))
    for rank, row in enumerate(rows, start=1):
        row["source_rank"] = int(rank)
    return rows


def _build_light_components(*, tpc, t0, hit_timestamps, hit_tpc_ids,
                            labels_global, x, y, z, energy, cfg: Trial2Config):
    finite = np.isfinite(hit_timestamps) & (hit_timestamps >= 0)
    mask = (hit_tpc_ids == tpc) & finite & (np.abs(hit_timestamps - t0)
                                            <= float(cfg.light_t0_match_ticks))
    if not mask.any():
        return []
    idx = np.flatnonzero(mask).astype(np.int64)
    pts = np.column_stack((x[idx], y[idx], z[idx])).astype(np.float64)
    comps = _connected_components(pts, float(cfg.light_component_radius_cm))
    rows = []
    labels = np.asarray(labels_global, dtype=np.int64)
    for comp_id, comp_local in enumerate(comps):
        gidx = idx[comp_local]
        if gidx.size < int(cfg.light_min_component_hits):
            continue
        e = float(np.sum(energy[gidx]))
        if e < float(cfg.light_min_component_energy_mev):
            continue
        # Majority parent label.
        comp_labels = labels[gidx]
        comp_labels = comp_labels[comp_labels >= 0]
        parent_label = -1
        if comp_labels.size:
            vals, counts = np.unique(comp_labels, return_counts=True)
            parent_label = int(vals[int(np.argmax(counts))])
        rows.append({
            "component_id": int(comp_id),
            "hit_indices": gidx,
            "n_hits": int(gidx.size),
            "energy_mev": float(e),
            "parent_label": int(parent_label),
        })
    rows.sort(key=lambda r: (-r["energy_mev"], -r["n_hits"]))
    return rows[: int(cfg.light_max_components_per_t0)]


def _component_proxy_image(*, component, image_maps, labels_global, hit_tpc_ids,
                           energy, tpc, n_channels, nt):
    """Energy-fraction scaled image for a sub-component of a parent cluster."""
    parent = int(component["parent_label"])
    if parent < 0 or (parent, int(tpc)) not in image_maps:
        return None
    parent_mask = (np.asarray(labels_global, dtype=np.int64) == parent) & \
                  (np.asarray(hit_tpc_ids, dtype=np.int32) == int(tpc))
    parent_e = float(np.sum(energy[parent_mask])) if parent_mask.any() else 0.0
    comp_e = float(component["energy_mev"])
    if parent_e <= 0:
        return None
    frac = comp_e / parent_e
    img = np.asarray(image_maps[(parent, int(tpc))], dtype=np.float32) * float(frac)
    if img.shape != (int(n_channels), int(nt)):
        # Pad/truncate just in case.
        out = np.zeros((int(n_channels), int(nt)), dtype=np.float32)
        ch = min(img.shape[0], int(n_channels))
        tk = min(img.shape[1], int(nt))
        out[:ch, :tk] = img[:ch, :tk]
        img = out
    return img


def _evaluate_trial2_light_gate(*, trial, base_image, full_light_waveform,
                                full_light_std, saturated_channel_cache,
                                cfg: Trial2Config):
    tpc = int(trial["TPCid"])
    old_t0 = int(trial["old_t0"])
    new_t0 = int(trial["new_t0"])
    keep = _keep_channel_indices(tpc, full_light_waveform, saturated_channel_cache)
    if keep.size == 0:
        return {"accepted": False, "reason": "no_keep_channels"}
    actual = np.asarray(full_light_waveform[tpc], dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std[tpc], dtype=np.float32), 1e-6)
    before_model = np.asarray(base_image[tpc], dtype=np.float32)
    delta = np.asarray(trial["delta"], dtype=np.float32)
    after_model = np.clip(before_model + delta, 0.0, float(cfg.adc_clip))

    def _ofch(model_tpc, t0):
        tick = int(t0) + int(cfg.pulse_peak_tick)
        if tick < 0 or tick >= actual.shape[1]:
            return 0
        lo = max(0, tick - int(cfg.half_window_ticks))
        hi = min(actual.shape[1], tick + int(cfg.half_window_ticks) + 1)
        a = actual[keep, lo:hi]
        m = model_tpc[keep, lo:hi]
        s = std[keep, lo:hi]
        threshold = np.maximum(float(cfg.light_overflow_sigma) * s,
                               float(cfg.light_overflow_abs_adc))
        overflow = ((m - a) > threshold) & (m > float(cfg.light_model_activity_adc))
        return int(np.count_nonzero(np.any(overflow, axis=1)))

    src_red = _ofch(before_model, old_t0) - _ofch(after_model, old_t0)
    chi2_before, dof = _physical_chi2_window_loss(
        model_tpc=before_model, actual_tpc=actual, std_tpc=std, keep=keep,
        t0_values=[old_t0, new_t0], cfg=cfg, std_floor=float(cfg.phys_std_floor))
    chi2_after, _ = _physical_chi2_window_loss(
        model_tpc=after_model, actual_tpc=actual, std_tpc=std, keep=keep,
        t0_values=[old_t0, new_t0], cfg=cfg, std_floor=float(cfg.phys_std_floor))
    dchi2 = chi2_before - chi2_after
    comp_e = float(trial.get("energy_mev", 0.0))
    dchi2_per_e = dchi2 / max(comp_e, 1e-6)

    src_ch_ok = src_red >= int(cfg.phys_min_source_ofch_reduction)
    dchi2_ok = dchi2 >= float(cfg.phys_min_dchi2_improvement)
    dchi2_per_e_ok = dchi2_per_e >= float(cfg.phys_min_dchi2_per_mev)
    accepted = bool(src_ch_ok and dchi2_ok and dchi2_per_e_ok)
    return {
        "accepted": accepted,
        "src_red": int(src_red),
        "chi2_before": float(chi2_before),
        "chi2_after": float(chi2_after),
        "dchi2": float(dchi2),
        "dchi2_per_mev": float(dchi2_per_e),
        "dof": int(dof),
        "channel_ok": bool(src_ch_ok),
        "dchi2_ok": bool(dchi2_ok),
        "dchi2_per_e_ok": bool(dchi2_per_e_ok),
    }


def _light_multitpc_track_veto(*, trial, labels_global, hit_tpc_ids, energy,
                               cfg: Trial2Config):
    idx = trial.get("hit_indices", np.array([], dtype=np.int64))
    if idx.size == 0:
        return {"track_veto": False, "track_veto_label": -1, "track_veto_n_tpcs": 0}
    labels = np.asarray(labels_global, dtype=np.int64)
    tpcs = np.asarray(hit_tpc_ids, dtype=np.int32)
    e = np.asarray(energy, dtype=np.float64)

    moved_labels = labels[idx]
    valid = moved_labels >= 0
    if not valid.any():
        return {"track_veto": False, "track_veto_label": -1, "track_veto_n_tpcs": 0}
    idx_valid = idx[valid]
    moved_labels = moved_labels[valid]

    vals = np.unique(moved_labels)
    label_energy = []
    label_hits = []
    for label in vals:
        m = moved_labels == int(label)
        h = idx_valid[m]
        label_energy.append(float(np.sum(e[h][np.isfinite(e[h])])))
        label_hits.append(int(h.size))
    order = np.lexsort((-np.asarray(label_hits), -np.asarray(label_energy)))
    majority_label = int(vals[order[0]])

    full_idx = np.flatnonzero(labels == majority_label)
    full_tpcs = tpcs[full_idx]
    label_tpcs = sorted({int(v) for v in np.unique(full_tpcs[full_tpcs >= 0])})

    veto = bool(len(label_tpcs) >= int(cfg.light_veto_track_min_tpcs))
    return {
        "track_veto": veto,
        "track_veto_label": majority_label,
        "track_veto_n_tpcs": int(len(label_tpcs)),
        "track_veto_label_tpcs": label_tpcs,
    }


def run_physical_light_repair(*, namespace, cfg: Trial2Config,
                              predict_family_image_fn):
    """Sub-step (c): physical chi^2 light repair."""
    base = np.asarray(namespace["baseImage"], dtype=np.float32).copy()
    hit_ts = np.asarray(namespace["hit_timestamps"], dtype=np.float32).copy()
    actual = np.asarray(namespace["fullLightWaveform"], dtype=np.float32)
    std = np.maximum(np.asarray(namespace.get(
        "fullLightStd", np.ones_like(actual)), dtype=np.float32), 1e-6)
    labels = np.asarray(namespace["labels_global"], dtype=np.int64)
    tpcs = np.asarray(namespace["hitTPCid"], dtype=np.int32)
    x = np.asarray(namespace["xset"], dtype=np.float64)
    y = np.asarray(namespace["yset"], dtype=np.float64)
    z = np.asarray(namespace["zset"], dtype=np.float64)
    energy = np.asarray(namespace["Eset"], dtype=np.float64)
    image_maps = namespace["imageMaps"]
    t0_cands = namespace["t0Candidates"]
    saturated_channel_cache = namespace.get("saturated_channel_cache")

    allowed_tpcs = sorted({int(v) for v in np.unique(tpcs) if int(v) >= 0})

    # Source-only flash-table pruning: only consider t0 candidates that have at
    # least one assigned charge hit nearby.
    from flash_cluster_table_2x2 import associated_source_t0s_by_tpc
    source_t0s_by_tpc = associated_source_t0s_by_tpc(
        hit_timestamps=hit_ts, hit_tpc_ids=tpcs,
        t0_candidates=t0_cands, allowed_tpcs=allowed_tpcs,
        labels_global=labels,
        match_ticks=float(cfg.light_flash_cluster_match_ticks),
    )

    overflow_rows = _scan_light_overflows_source_only(
        base_image=base, full_light_waveform=actual,
        full_light_std=std, source_t0s_by_tpc=source_t0s_by_tpc,
        saturated_channel_cache=saturated_channel_cache,
        allowed_tpcs=allowed_tpcs, cfg=cfg,
    )

    light_moves = []
    light_trials = []
    family_records = []
    moves_by_tpc = {}
    track_veto_candidates = []

    nt = base.shape[-1]

    for row in overflow_rows:
        if len(light_moves) >= int(cfg.light_max_total_moves):
            break
        tpc = int(row["TPCid"])
        if moves_by_tpc.get(tpc, 0) >= int(cfg.light_max_moves_per_tpc):
            continue
        source_t0 = int(row["t0"])

        # Candidate destination t0s.
        cand_dests = sorted({int(round(float(v))) for v in t0_cands[int(tpc)]
                             if v is not None and np.isfinite(float(v))
                             and abs(int(round(float(v))) - source_t0) > float(cfg.light_t0_merge_ticks)})
        if not cand_dests:
            continue

        # Donor components on the source side.
        donor_components = _build_light_components(
            tpc=tpc, t0=source_t0,
            hit_timestamps=hit_ts, hit_tpc_ids=tpcs,
            labels_global=labels, x=x, y=y, z=z, energy=energy, cfg=cfg,
        )
        if not donor_components:
            continue

        # Build trials for every (donor, dest_t0) pair, evaluate gate, take best.
        best_trial = None
        best_score = None
        for component in donor_components:
            proxy = _component_proxy_image(
                component=component, image_maps=image_maps,
                labels_global=labels, hit_tpc_ids=tpcs, energy=energy,
                tpc=tpc, n_channels=int(cfg.n_channels), nt=int(cfg.nt),
            )
            if proxy is None:
                continue
            old_shift = _shift_integer(proxy, source_t0, nt=nt)
            for dest_t0 in cand_dests:
                new_shift = _shift_integer(proxy, dest_t0, nt=nt)
                delta = (new_shift - old_shift).astype(np.float32)
                trial = {
                    "TPCid": tpc, "old_t0": int(source_t0), "new_t0": int(dest_t0),
                    "component_id": int(component["component_id"]),
                    "hit_indices": component["hit_indices"],
                    "n_hits": int(component["n_hits"]),
                    "energy_mev": float(component["energy_mev"]),
                    "parent_label": int(component["parent_label"]),
                    "delta": delta,
                    "source_rank": int(row["source_rank"]),
                }
                gate = _evaluate_trial2_light_gate(
                    trial=trial, base_image=base,
                    full_light_waveform=actual, full_light_std=std,
                    saturated_channel_cache=saturated_channel_cache, cfg=cfg,
                )
                trial.update(gate)
                light_trials.append({k: v for k, v in trial.items() if k != "delta"})
                if not gate["accepted"]:
                    continue
                if cfg.light_veto_multitpc_track:
                    veto = _light_multitpc_track_veto(
                        trial=trial, labels_global=labels,
                        hit_tpc_ids=tpcs, energy=energy, cfg=cfg,
                    )
                    trial.update(veto)
                    if veto["track_veto"]:
                        track_veto_candidates.append(trial)
                        continue
                # Keep the trial with the largest dChi2/E.
                score = trial["dchi2_per_mev"]
                if best_score is None or score > best_score:
                    best_trial = trial
                    best_score = score

        if best_trial is None:
            continue

        # Apply the best accepted trial.
        before_ts = hit_ts.copy()
        before_base = base.copy()
        moved = best_trial["hit_indices"].astype(np.int64)
        old_t0 = int(best_trial["old_t0"])
        new_t0 = int(best_trial["new_t0"])
        hit_ts[moved] = np.float32(new_t0)
        affected = {(tpc, old_t0), (tpc, new_t0)}
        base, recs = _exact_update_affected_families(
            base_image=before_base,
            old_hit_timestamps=before_ts, new_hit_timestamps=hit_ts,
            affected_specs=affected,
            hit_tpc_ids=tpcs, x=x, y=y, z=z, energy=energy,
            predict_family_image_fn=predict_family_image_fn, cfg=cfg,
        )
        family_records.extend(recs)
        light_moves.append({k: v for k, v in best_trial.items() if k != "delta"})
        moves_by_tpc[tpc] = moves_by_tpc.get(tpc, 0) + 1
        _canonicalize_candidate_t0(
            t0_cands, tpc, float(new_t0),
            merge_ticks=float(cfg.light_flash_cluster_match_ticks))

    # Track-veto override pass: replay vetoed trials sharing a popular label.
    if track_veto_candidates:
        from collections import Counter
        labels_count = Counter(int(t["track_veto_label"]) for t in track_veto_candidates)
        override_labels = {l for l, c in labels_count.items()
                           if c >= int(cfg.light_veto_override_min_candidates)}
        for trial in sorted(track_veto_candidates,
                            key=lambda t: t.get("source_rank", 1 << 30)):
            if int(trial.get("track_veto_label", -1)) not in override_labels:
                continue
            if len(light_moves) >= int(cfg.light_max_total_moves):
                break
            tpc = int(trial["TPCid"])
            if moves_by_tpc.get(tpc, 0) >= int(cfg.light_max_moves_per_tpc):
                continue
            # Re-evaluate gate against the (possibly updated) base.
            gate = _evaluate_trial2_light_gate(
                trial=trial, base_image=base,
                full_light_waveform=actual, full_light_std=std,
                saturated_channel_cache=saturated_channel_cache, cfg=cfg,
            )
            if not gate["accepted"]:
                continue
            before_ts = hit_ts.copy()
            before_base = base.copy()
            moved = trial["hit_indices"].astype(np.int64)
            old_t0 = int(trial["old_t0"])
            new_t0 = int(trial["new_t0"])
            hit_ts[moved] = np.float32(new_t0)
            affected = {(tpc, old_t0), (tpc, new_t0)}
            base, recs = _exact_update_affected_families(
                base_image=before_base,
                old_hit_timestamps=before_ts, new_hit_timestamps=hit_ts,
                affected_specs=affected,
                hit_tpc_ids=tpcs, x=x, y=y, z=z, energy=energy,
                predict_family_image_fn=predict_family_image_fn, cfg=cfg,
            )
            family_records.extend(recs)
            override_record = {k: v for k, v in trial.items() if k != "delta"}
            override_record["track_veto_overridden"] = True
            light_moves.append(override_record)
            moves_by_tpc[tpc] = moves_by_tpc.get(tpc, 0) + 1

    namespace["baseImage"] = base
    namespace["hit_timestamps"] = hit_ts
    namespace["t0Candidates"] = t0_cands
    return {
        "overflow_rows": overflow_rows,
        "light_trials": light_trials,
        "light_moves": light_moves,
        "family_update_records": family_records,
        "moves_by_tpc": moves_by_tpc,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def _canonicalize_candidate_t0(t0_candidates, tpc, t0, *, merge_ticks):
    from flash_cluster_table_2x2 import canonicalize_candidate_t0 as _impl
    return _impl(t0_candidates, tpc, t0, merge_ticks=merge_ticks)


def run_trial2_combined_rescue(*, namespace, predict_family_image_fn,
                               cfg: Trial2Config | None = None,
                               commit: bool = True):
    """Run sub-steps (a), (b), (c) on the namespace.

    The namespace must contain:
      - baseImage              (8, 48, 1000) float32
      - hit_timestamps         (n_hits,)     float32  (== t0_for_all_hits)
      - fullLightWaveform      (8, 48, 1000) float32
      - fullLightStd           (8, 48, 1000) float32  (variance map)
      - labels_global          (n_hits,)     int64    (== labels)
      - hitTPCid               (n_hits,)     int32
      - xset, yset, zset       (n_hits,)     float
      - Eset                   (n_hits,)     float
      - imageMaps              dict[(cluster_id, tpc) -> (48, 1000)]
      - cluster_to_tpcs        dict[cluster_id -> list[tpc]]
      - t0Candidates           list[list[float]] indexed by tpc
      - track_shower_labels    optional iterable of cluster ids that are tracks/showers
      - saturated_channel_cache optional dict with veto_mask[tpc] -> bool[48]

    Returns a result dict with sub-step records. If ``commit=False``, restores
    the namespace to its pre-call state (snapshots taken at entry).
    """
    cfg = cfg or Trial2Config()

    before_ts = np.asarray(namespace["hit_timestamps"], dtype=np.float32).copy()
    before_base = np.asarray(namespace["baseImage"], dtype=np.float32).copy()
    before_t0_cands = copy.deepcopy(namespace["t0Candidates"])

    # (a) Large-cluster flash-grid fine correction.
    large_result = {"rows": [], "accepted_rows": []}
    if cfg.enable_large_flash_grid_correction:
        if cfg.verbose:
            print("=== Trial2 (a): large-cluster flash-grid fine correction ===")
        large_result = run_large_cluster_flash_grid_correction(
            namespace=namespace, cfg=cfg)
        if cfg.verbose:
            print(f"   accepted: {len(large_result['accepted_rows'])} / {len(large_result['rows'])}")

    # (b) Spatial mixed-t0 repair.
    spatial_result = {"pair_rows": [], "spatial_moves": [],
                      "spatial_trials": [], "family_update_records": []}
    if cfg.enable_spatial:
        if cfg.verbose:
            print("=== Trial2 (b): spatial mixed-t0 repair ===")
        spatial_result = run_spatial_mixed_t0_repair(
            namespace=namespace, cfg=cfg,
            predict_family_image_fn=predict_family_image_fn)
        if cfg.verbose:
            print(f"   moved {len(spatial_result['spatial_moves'])} components "
                  f"across {len(spatial_result['pair_rows'])} mixed-t0 pairs")

    # (c) Physical-chi^2 light repair.
    light_result = {"light_moves": [], "light_trials": [], "overflow_rows": [],
                    "family_update_records": [], "moves_by_tpc": {}}
    if cfg.enable_light:
        if cfg.verbose:
            print("=== Trial2 (c): physical-chi2 light repair ===")
        light_result = run_physical_light_repair(
            namespace=namespace, cfg=cfg,
            predict_family_image_fn=predict_family_image_fn)
        if cfg.verbose:
            print(f"   accepted {len(light_result['light_moves'])} light moves "
                  f"across {len(light_result['overflow_rows'])} candidate (tpc, t0) rows")

    result = {
        "large_flash_grid": large_result,
        "spatial": spatial_result,
        "light": light_result,
        "config": cfg,
        "before": {"hit_timestamps": before_ts, "baseImage": before_base,
                   "t0Candidates": before_t0_cands},
    }
    if not commit:
        # Restore namespace.
        namespace["hit_timestamps"] = before_ts
        namespace["baseImage"] = before_base
        namespace["t0Candidates"] = copy.deepcopy(before_t0_cands)
    return result


__all__ = [
    "Trial2Config",
    "run_trial2_combined_rescue",
    "run_large_cluster_flash_grid_correction",
    "run_spatial_mixed_t0_repair",
    "run_physical_light_repair",
]
