"""Phase 2.5 charge-light amendment stage.

This module packages the two notebook-level amendment prototypes into a
non-plotting stage:

1. spatial mixed-t0 repair for assigned hit families in the same TPC;
2. light-overflow repair by moving whole connected components between flash t0s.

After hit-level t0 changes, affected t0-family light images are re-predicted with
the charge-light model for the full family, not just moved hits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

try:
    from NewMLSection.ML_NDfull_perceiver import group_voxelize_pairs, predict_phi
except ModuleNotFoundError:
    from M5p1ReleaseVersion.NewMLSection.ML_NDfull_perceiver import group_voxelize_pairs, predict_phi


__all__ = [
    "Phase25Config",
    "infer_shower_tpcs",
    "infer_long_track_hit_mask",
    "run_phase25_amendment",
    "run_phase25_amendment_from_namespace",
]


@dataclass
class Phase25Config:
    """Tunable parameters for Phase 2.5."""

    # General
    skip_shower_tpcs: bool = True
    shower_keywords: tuple[str, ...] = ("shower",)
    freeze_long_track_hits: bool = False
    long_track_keywords: tuple[str, ...] = ("track",)
    long_track_exclude_keywords: tuple[str, ...] = ("shower",)
    long_track_min_tpcs: int = 3
    adc_clip: float = 60780.0
    verbose: bool = True
    max_move_energy_per_tpc_mev: float | None = 80.0

    # Timing convention
    t0_match_ticks: int = 10
    t0_merge_ticks: int = 5
    pulse_peak_tick: int = 105
    half_window_ticks: int = 18
    pad_ticks: int = 12

    # Exact GPU family re-prediction
    target_scale: float = 1.0e-3
    prediction_batch_size: int = 8
    raw_clip: tuple[float, float] = (0.0, 60780.0)
    min_prediction_threshold: float = 100.0
    device_policy: str = "auto"

    # Spatial mixed-t0 detection
    enable_spatial: bool = True
    spatial_contact_radius_cm: float = 4.0
    spatial_different_t0_ticks: float = 10.0
    spatial_min_pair_hits: int = 24
    spatial_min_pair_energy_mev: float = 3.0
    spatial_max_pairs: int = 24

    # Spatial repair modeling
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
    spatial_max_loss_increase: float = 1.0e4

    # Light overflow scan / repair
    enable_light: bool = True
    light_overflow_sigma: float = 3.0
    light_overflow_abs_adc: float = 400.0
    light_model_activity_adc: float = 400.0
    light_min_overflow_channels: int = 6
    light_component_radius_cm: float = 3.0
    light_min_component_hits: int = 4
    light_min_component_energy_mev: float = 0.20
    light_max_components_per_t0: int = 80
    light_max_dest_t0s: int = 10
    light_max_total_moves: int = 24
    light_max_moves_per_tpc: int = 3
    light_min_dest_deficit_sum: float = 5000.0
    light_min_loss_improvement: float = 1.0e5
    light_min_old_overflow_reduction: float = 5000.0
    light_min_dest_deficit_reduction: float = 5000.0
    light_max_dest_new_overflow: float = 2.5e5
    light_max_dest_new_overflow_frac: float = 0.50
    light_use_spatial_prior: bool = True
    light_spatial_prior_weight: float = 0.05
    light_overflow_rows_per_pass: int = 120
    overflow_weight: float = 4.0

    # Internal safeguards
    max_family_prediction_hits: int | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def _get(namespace: dict[str, Any], name: str) -> Any:
    if name not in namespace:
        raise KeyError(f"Phase25 requires `{name}` in the provided namespace.")
    return namespace[name]


def _shift_image(image: np.ndarray, t0: int, *, nt: int | None = None) -> np.ndarray:
    """Shift an unshifted (channels, ticks) image by integer t0 ticks."""
    img = np.asarray(image, dtype=np.float32)
    if nt is None:
        nt = img.shape[1]
    out = np.zeros((img.shape[0], int(nt)), dtype=np.float32)
    t0 = int(round(float(t0)))
    if t0 >= 0:
        if t0 < nt:
            n = min(img.shape[1], nt - t0)
            out[:, t0 : t0 + n] = img[:, :n]
    else:
        src0 = -t0
        if src0 < img.shape[1]:
            n = min(img.shape[1] - src0, nt)
            out[:, :n] = img[:, src0 : src0 + n]
    return out


def _unshift_image_window(model_image: np.ndarray, t0: int) -> np.ndarray:
    """Approximate the unshifted source-family image from the shifted TPC model."""
    model = np.asarray(model_image, dtype=np.float32)
    out = np.zeros_like(model, dtype=np.float32)
    nt = model.shape[1]
    t0 = int(round(float(t0)))
    if t0 >= 0:
        if t0 < nt:
            n = nt - t0
            out[:, :n] = model[:, t0 : t0 + n]
    else:
        dst0 = -t0
        if dst0 < nt:
            n = nt - dst0
            out[:, dst0 : dst0 + n] = model[:, :n]
    return out


def _finite_assigned(t0: np.ndarray) -> np.ndarray:
    return np.isfinite(t0) & (t0 >= 0)


def _merge_close_t0s(values: Iterable[Any], merge_ticks: int) -> list[int]:
    vals = sorted(
        int(round(float(v)))
        for v in values
        if v is not None and np.isfinite(float(v)) and int(round(float(v))) != 0
    )
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= int(merge_ticks):
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    return [int(round(float(np.median(g)))) for g in groups]


def _keep_channel_indices(
    TPCid: int,
    actual_full: np.ndarray,
    saturated_channel_cache: Any | None,
) -> np.ndarray:
    if saturated_channel_cache is not None:
        try:
            veto = np.asarray(saturated_channel_cache["veto_mask"][int(TPCid)], dtype=bool)
            keep = np.flatnonzero(~veto).astype(np.int32)
            if keep.size:
                return keep
        except Exception:
            pass
    actual = np.asarray(actual_full[int(TPCid)], dtype=np.float32)
    veto = np.sum(actual > 60700.0, axis=1) > 6
    return np.flatnonzero(~veto).astype(np.int32)


def _weighted_loss(
    model_kept: np.ndarray,
    actual_kept: np.ndarray,
    std_kept: np.ndarray,
    tmask: np.ndarray,
    overflow_weight: float,
) -> float:
    m = model_kept[:, tmask].astype(np.float32)
    a = actual_kept[:, tmask].astype(np.float32)
    s = np.maximum(std_kept[:, tmask].astype(np.float32), 1e-6)
    w = np.where(m > a, float(overflow_weight), 1.0).astype(np.float32)
    return float(np.sum(((m - a) ** 2 / s) * w))


def _move_energy_within_limit(energy_mev: float, cfg: Phase25Config) -> bool:
    if cfg.max_move_energy_per_tpc_mev is None:
        return True
    return float(energy_mev) <= float(cfg.max_move_energy_per_tpc_mev)


def _window_mask(nt: int, t0s: Iterable[int], cfg: Phase25Config) -> np.ndarray:
    mask = np.zeros(int(nt), dtype=bool)
    for t0 in t0s:
        tick = int(t0) + int(cfg.pulse_peak_tick)
        lo = max(0, tick - int(cfg.half_window_ticks) - int(cfg.pad_ticks))
        hi = min(nt, tick + int(cfg.half_window_ticks) + int(cfg.pad_ticks) + 1)
        if lo < hi:
            mask[lo:hi] = True
    if not np.any(mask):
        mask[:] = True
    return mask


def infer_shower_tpcs(
    *,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    label_info: Any | None = None,
    shower_keywords: Iterable[str] = ("shower",),
) -> set[int]:
    """Infer shower-related TPCs without changing the Phase-1 return schema."""
    labels = np.asarray(labels_global, dtype=np.int64)
    tpcs = np.asarray(hit_tpc_ids, dtype=np.int32)
    keywords = tuple(str(k).lower() for k in shower_keywords)
    out: set[int] = set()

    def _maybe_add_from_label(cid: int, info: Any) -> None:
        if isinstance(info, dict):
            text = " ".join(
                str(info.get(k, ""))
                for k in ("type", "class", "classification", "kind", "stage", "name", "label")
            ).lower()
        else:
            text = str(info).lower()
        if not any(k in text for k in keywords):
            return

        tpc_values = None
        if isinstance(info, dict):
            for key in ("tpcs", "TPCs", "hit_tpcs", "active_tpcs", "charge_tpcs"):
                if key in info:
                    tpc_values = info[key]
                    break
        if tpc_values is not None:
            try:
                for t in np.asarray(tpc_values).ravel():
                    if np.isfinite(float(t)):
                        out.add(int(t))
                return
            except Exception:
                pass

        mask = labels == int(cid)
        if np.any(mask):
            out.update(int(v) for v in np.unique(tpcs[mask]))

    if isinstance(label_info, dict):
        for key, info in label_info.items():
            cid = None
            try:
                cid = int(key)
            except Exception:
                if isinstance(info, dict):
                    for candidate_key in ("label", "clusterid", "cluster_id", "id"):
                        if candidate_key in info:
                            try:
                                cid = int(info[candidate_key])
                                break
                            except Exception:
                                pass
            if cid is not None:
                _maybe_add_from_label(cid, info)
    elif label_info is not None:
        try:
            iterator = enumerate(label_info)
        except TypeError:
            iterator = []
        for fallback_cid, info in iterator:
            cid = fallback_cid
            if isinstance(info, dict):
                for candidate_key in ("label", "clusterid", "cluster_id", "id"):
                    if candidate_key in info:
                        try:
                            cid = int(info[candidate_key])
                            break
                        except Exception:
                            pass
            _maybe_add_from_label(int(cid), info)

    return out


def _iter_label_info_items(label_info: Any | None) -> Iterable[tuple[int, Any]]:
    if isinstance(label_info, dict):
        for key, info in label_info.items():
            cid = None
            try:
                cid = int(key)
            except Exception:
                if isinstance(info, dict):
                    for candidate_key in ("label", "clusterid", "cluster_id", "id"):
                        if candidate_key in info:
                            try:
                                cid = int(info[candidate_key])
                                break
                            except Exception:
                                pass
            if cid is not None:
                yield int(cid), info
    elif label_info is not None:
        try:
            iterator = enumerate(label_info)
        except TypeError:
            iterator = []
        for fallback_cid, info in iterator:
            cid = int(fallback_cid)
            if isinstance(info, dict):
                for candidate_key in ("label", "clusterid", "cluster_id", "id"):
                    if candidate_key in info:
                        try:
                            cid = int(info[candidate_key])
                            break
                        except Exception:
                            pass
            yield int(cid), info


def _label_info_text(info: Any) -> str:
    if isinstance(info, dict):
        keys = (
            "type",
            "class",
            "classification",
            "kind",
            "stage",
            "name",
            "label",
            "category",
            "topology",
        )
        return " ".join(str(info.get(k, "")) for k in keys).lower()
    return str(info).lower()


def infer_long_track_hit_mask(
    *,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    label_info: Any | None = None,
    track_keywords: Iterable[str] = ("track",),
    exclude_keywords: Iterable[str] = ("shower",),
    min_tpcs: int = 3,
    return_records: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Mark hits belonging to explicit track-like labels spanning many TPCs.

    This intentionally uses label metadata to identify tracks. It does not use a
    pure-linearity fallback, because shower fragments can also look linear over
    a short range and should not be silently frozen.
    """
    labels = np.asarray(labels_global, dtype=np.int64)
    tpcs = np.asarray(hit_tpc_ids, dtype=np.int32)
    track_words = tuple(str(v).lower() for v in track_keywords)
    exclude_words = tuple(str(v).lower() for v in exclude_keywords)
    min_tpcs = max(int(min_tpcs), 3)

    track_labels: set[int] = set()
    for cid, info in _iter_label_info_items(label_info):
        text = _label_info_text(info)
        if any(word in text for word in exclude_words):
            continue
        if any(word in text for word in track_words):
            track_labels.add(int(cid))

    locked = np.zeros(labels.shape[0], dtype=bool)
    records: list[dict[str, Any]] = []

    for cid in sorted(track_labels):
        mask = labels == int(cid)
        if not np.any(mask):
            continue
        active_tpcs = sorted(int(v) for v in np.unique(tpcs[mask]))
        if len(active_tpcs) < min_tpcs:
            continue
        n_hits = int(np.count_nonzero(mask))
        locked |= mask
        records.append(
            {
                "label": int(cid),
                "active_tpcs": active_tpcs,
                "n_tpcs": int(len(active_tpcs)),
                "n_locked_hits": n_hits,
            }
        )

    if return_records:
        return locked.astype(bool), records
    return locked.astype(bool)


def _connected_components(points: np.ndarray, radius_cm: float) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    n = int(pts.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=np.int64)]

    tree = cKDTree(pts)
    pairs = tree.query_pairs(float(radius_cm), output_type="ndarray")
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return int(a)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(int(a), int(b))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def _fit_trimmed_pca(
    points: np.ndarray,
    energies: np.ndarray,
    cfg: Phase25Config,
) -> dict[str, Any] | None:
    pts_all = np.asarray(points, dtype=np.float64)
    if pts_all.shape[0] == 0:
        return None
    e_all = np.clip(np.asarray(energies, dtype=np.float64), 1e-9, None)
    keep = np.ones(pts_all.shape[0], dtype=bool)
    centroid = np.mean(pts_all, axis=0)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    linearity = 0.0

    for _ in range(max(1, int(cfg.spatial_trim_iterations))):
        pts = pts_all[keep]
        e = e_all[keep]
        if pts.shape[0] < 2:
            centroid = pts[0].copy()
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
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            linearity = 0.0

        rel = pts_all - centroid
        proj_all = rel @ direction
        perp_all = rel - proj_all[:, None] * direction[None, :]
        perp_dist_all = np.sqrt(np.sum(perp_all * perp_all, axis=1))
        if pts_all.shape[0] >= 8:
            cut = float(np.quantile(perp_dist_all, float(cfg.spatial_trim_model_quantile)))
            keep_new = perp_dist_all <= max(cut, float(cfg.spatial_axis_width_floor_cm))
            if np.count_nonzero(keep_new) >= max(3, int(0.40 * pts_all.shape[0])):
                keep = keep_new

    rel = pts_all - centroid
    proj = rel @ direction
    perp = rel - proj[:, None] * direction[None, :]
    perp_dist = np.sqrt(np.sum(perp * perp, axis=1))
    fit_perp = perp_dist[keep] if np.any(keep) else perp_dist
    width = max(float(np.quantile(fit_perp, 0.68)), float(cfg.spatial_axis_width_floor_cm))
    fit_proj = proj[keep] if np.any(keep) else proj

    return {
        "centroid": centroid.astype(np.float64),
        "direction": direction.astype(np.float64),
        "proj_min": float(np.min(fit_proj)) if fit_proj.size else 0.0,
        "proj_max": float(np.max(fit_proj)) if fit_proj.size else 0.0,
        "width_cm": float(width),
        "linearity": float(linearity),
        "tree": cKDTree(pts_all),
        "points": pts_all,
        "energy_mev": float(np.sum(e_all)),
        "n_hits": int(pts_all.shape[0]),
        "n_fit_hits": int(np.count_nonzero(keep)),
    }


def _score_points_to_model(points: np.ndarray, model: dict[str, Any] | None, cfg: Phase25Config) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if model is None:
        return np.full(pts.shape[0], 1.0e9, dtype=np.float64)

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

    if float(model["linearity"]) >= 0.85:
        score = (
            0.70 * (perp_dist / width)
            + 0.20 * (nearest / float(cfg.spatial_nearest_scale_cm))
            + 0.10 * (endpoint_gap / float(cfg.spatial_endpoint_gap_scale_cm))
        )
    else:
        score = (
            0.45 * (perp_dist / width)
            + 0.45 * (nearest / float(cfg.spatial_nearest_scale_cm))
            + 0.10 * (endpoint_gap / float(cfg.spatial_endpoint_gap_scale_cm))
        )
    return score.astype(np.float64)


def _score_points_to_models(
    points: np.ndarray,
    models: list[dict[str, Any]],
    cfg: Phase25Config,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if not models:
        return (
            np.full(pts.shape[0], 1.0e9, dtype=np.float64),
            np.full(pts.shape[0], -1, dtype=np.int32),
        )
    score_mat = np.vstack([_score_points_to_model(pts, m, cfg) for m in models])
    best = np.argmin(score_mat, axis=0).astype(np.int32)
    return score_mat[best, np.arange(pts.shape[0])].astype(np.float64), best


def _build_spatial_models(
    indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    cfg: Phase25Config,
) -> list[dict[str, Any]]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return []
    pts = np.column_stack((x[idx], y[idx], z[idx])).astype(np.float64)
    comps = _connected_components(pts, cfg.spatial_component_radius_cm)
    rows: list[dict[str, Any]] = []
    for comp_id, comp_local in enumerate(comps):
        gidx = idx[comp_local]
        e_sum = float(np.sum(energy[gidx]))
        if gidx.size < int(cfg.spatial_min_model_hits):
            continue
        if e_sum < float(cfg.spatial_min_model_energy_mev):
            continue
        model = _fit_trimmed_pca(
            np.column_stack((x[gidx], y[gidx], z[gidx])).astype(np.float64),
            energy[gidx],
            cfg,
        )
        if model is None:
            continue
        model["global_indices"] = gidx.astype(np.int64)
        model["component_id"] = int(comp_id)
        rows.append(model)
    rows.sort(key=lambda m: (-float(m["energy_mev"]), -int(m["n_hits"])))
    return rows[: int(cfg.spatial_max_models_per_t0)]


def _label_summary(indices: np.ndarray, labels: np.ndarray, max_items: int = 8) -> str:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return "[]"
    vals, counts = np.unique(labels[idx].astype(np.int32), return_counts=True)
    order = np.argsort(counts)[::-1]
    return "[" + ", ".join(f"{int(vals[k])}:{int(counts[k])}" for k in order[:max_items]) + "]"


def _pca_geometry(
    indices: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64)
    pts = np.column_stack((x[idx], y[idx], z[idx])).astype(np.float64)
    weights = np.clip(np.asarray(energy[idx], dtype=np.float64), 1e-9, None)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    centroid = np.sum(pts * weights[:, None], axis=0)

    if pts.shape[0] < 2:
        return {
            "centroid": centroid.astype(np.float64),
            "direction": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "linearity": 0.0,
        }

    centered = pts - centroid
    cov = (centered * weights[:, None]).T @ centered
    try:
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        direction = evecs[:, 0]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        linearity = float(evals[0] / max(float(np.sum(evals)), 1e-12))
    except np.linalg.LinAlgError:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        linearity = 0.0

    return {
        "centroid": centroid.astype(np.float64),
        "direction": direction.astype(np.float64),
        "linearity": float(linearity),
    }


def _detect_mixed_t0_pairs(
    *,
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    allowed_tpcs: set[int],
    locked_hit_mask: np.ndarray | None,
    cfg: Phase25Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    finite = _finite_assigned(hit_timestamps)
    locked = np.zeros(hit_timestamps.shape[0], dtype=bool) if locked_hit_mask is None else np.asarray(locked_hit_mask, dtype=bool)

    for tpc in sorted(allowed_tpcs):
        tmask = (hit_tpc_ids == int(tpc)) & finite
        idx = np.flatnonzero(tmask).astype(np.int64)
        if idx.size < 2:
            continue
        pts = np.column_stack((x[idx], y[idx], z[idx])).astype(np.float64)
        pairs = cKDTree(pts).query_pairs(float(cfg.spatial_contact_radius_cm), output_type="ndarray")
        if pairs.size == 0:
            continue

        t0_int = np.round(hit_timestamps[idx]).astype(np.int32)
        pair_hits: dict[tuple[int, int], set[int]] = {}
        for a, b in pairs:
            ga = int(idx[int(a)])
            gb = int(idx[int(b)])
            if bool(locked[ga]) and bool(locked[gb]):
                continue
            ta = int(t0_int[int(a)])
            tb = int(t0_int[int(b)])
            if abs(ta - tb) <= float(cfg.spatial_different_t0_ticks):
                continue
            key = (min(ta, tb), max(ta, tb))
            s = pair_hits.setdefault(key, set())
            if not bool(locked[ga]):
                s.add(ga)
            if not bool(locked[gb]):
                s.add(gb)

        for (ta, tb), hit_set in pair_hits.items():
            pair_idx = np.asarray(sorted(hit_set), dtype=np.int64)
            if pair_idx.size < int(cfg.spatial_min_pair_hits):
                continue
            e = float(np.sum(energy[pair_idx]))
            if e < float(cfg.spatial_min_pair_energy_mev):
                continue
            rows.append(
                {
                    "TPCid": int(tpc),
                    "t0_a": int(ta),
                    "t0_b": int(tb),
                    "n_hits": int(pair_idx.size),
                    "energy_mev": e,
                    "labels": _label_summary(pair_idx, labels_global),
                    "hit_indices": pair_idx,
                    "severity": float(e * pair_idx.size),
                }
            )

    rows.sort(key=lambda r: (-float(r["severity"]), -float(r["energy_mev"]), int(r["TPCid"])))
    return rows[: int(cfg.spatial_max_pairs)]


def _restore_mixed_pair_spatially(
    row: dict[str, Any],
    *,
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    locked_hit_mask: np.ndarray | None,
    cfg: Phase25Config,
) -> dict[str, Any] | None:
    tpc = int(row["TPCid"])
    t0_a = int(row["t0_a"])
    t0_b = int(row["t0_b"])
    tpc_mask = hit_tpc_ids == tpc
    finite = _finite_assigned(hit_timestamps)
    locked = np.zeros(hit_timestamps.shape[0], dtype=bool) if locked_hit_mask is None else np.asarray(locked_hit_mask, dtype=bool)
    mask_a = tpc_mask & finite & (np.abs(hit_timestamps - float(t0_a)) <= float(cfg.t0_match_ticks))
    mask_b = tpc_mask & finite & (np.abs(hit_timestamps - float(t0_b)) <= float(cfg.t0_match_ticks))
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
    current_is_a = np.abs(hit_timestamps[idx_pair] - float(t0_a)) <= float(cfg.t0_match_ticks)
    score_current = np.where(current_is_a, score_a, score_b)
    score_other = np.where(current_is_a, score_b, score_a)
    plausible = np.minimum(score_a, score_b) <= float(cfg.spatial_max_accept_score)
    competitive = score_other <= (score_current + float(cfg.spatial_rescan_pool_margin))
    candidate_local = np.flatnonzero(plausible | competitive).astype(np.int64)
    if candidate_local.size == 0:
        return None

    candidate_idx = idx_pair[candidate_local].astype(np.int64)
    candidate_idx = candidate_idx[~locked[candidate_idx]]
    if candidate_idx.size == 0:
        return None
    candidate_pts = np.column_stack((x[candidate_idx], y[candidate_idx], z[candidate_idx])).astype(np.float64)
    comps = _connected_components(candidate_pts, cfg.spatial_smooth_component_radius_cm)

    pair_score_a = {int(g): float(s) for g, s in zip(idx_pair, score_a)}
    pair_score_b = {int(g): float(s) for g, s in zip(idx_pair, score_b)}
    new_t0_by_hit: dict[int, int] = {}
    move_rows: list[dict[str, Any]] = []

    for comp_id, comp_local in enumerate(comps, start=1):
        comp_idx = candidate_idx[comp_local].astype(np.int64)
        if comp_idx.size == 0:
            continue
        s_a = np.asarray([pair_score_a[int(i)] for i in comp_idx], dtype=np.float64)
        s_b = np.asarray([pair_score_b[int(i)] for i in comp_idx], dtype=np.float64)
        cur_a = np.abs(hit_timestamps[comp_idx] - float(t0_a)) <= float(cfg.t0_match_ticks)
        cur_b = np.abs(hit_timestamps[comp_idx] - float(t0_b)) <= float(cfg.t0_match_ticks)
        s_a_inertial = s_a.copy()
        s_b_inertial = s_b.copy()
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
        if not np.any(moved):
            continue
        moved_idx = comp_idx[moved]
        for g, nt0 in zip(moved_idx, targets[moved]):
            new_t0_by_hit[int(g)] = int(nt0)
        for old_t0, new_t0 in sorted(set((int(o), int(n)) for o, n in zip(old_vals[moved], targets[moved]))):
            sel = moved_idx[(old_vals[moved] == old_t0) & (targets[moved] == new_t0)]
            move_rows.append(
                {
                    "component_id": int(comp_id),
                    "old_t0": int(old_t0),
                    "new_t0": int(new_t0),
                    "n_hits": int(sel.size),
                    "energy_mev": float(np.sum(energy[sel])),
                    "labels": _label_summary(sel, labels_global),
                    "hit_indices": sel.astype(np.int64),
                }
            )

    if len(new_t0_by_hit) < int(cfg.spatial_min_moved_hits):
        return None

    moved_idx = np.asarray(sorted(new_t0_by_hit), dtype=np.int64)
    return {
        "TPCid": int(tpc),
        "t0_a": int(t0_a),
        "t0_b": int(t0_b),
        "moved_idx": moved_idx,
        "new_t0": np.asarray([new_t0_by_hit[int(i)] for i in moved_idx], dtype=np.float32),
        "move_rows": move_rows,
        "n_moved": int(moved_idx.size),
        "energy_moved_mev": float(np.sum(energy[moved_idx])),
    }


def _predict_family_images_batch(
    family_specs: list[tuple[int, int]],
    *,
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    model: Any,
    template: np.ndarray,
    cfg: Phase25Config,
) -> dict[tuple[int, int], np.ndarray]:
    family_specs = [(int(t), int(t0)) for t, t0 in family_specs]
    template = np.asarray(template, dtype=np.float32)
    out = {
        spec: np.zeros((120, template.shape[-1]), dtype=np.float32)
        for spec in family_specs
    }
    if not family_specs:
        return out

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    es: list[np.ndarray] = []
    tids: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    label_to_spec: dict[int, tuple[int, int]] = {}

    for family_id, (tpc, t0) in enumerate(family_specs):
        mask = (
            (hit_tpc_ids == int(tpc))
            & _finite_assigned(hit_timestamps)
            & (np.abs(hit_timestamps - float(t0)) <= float(cfg.t0_match_ticks))
        )
        idx = np.flatnonzero(mask).astype(np.int64)
        if idx.size == 0:
            continue
        if cfg.max_family_prediction_hits is not None and idx.size > int(cfg.max_family_prediction_hits):
            idx = idx[: int(cfg.max_family_prediction_hits)]
        artificial_label = 10_000_000 + int(family_id)
        label_to_spec[artificial_label] = (int(tpc), int(t0))
        xs.append(x[idx])
        ys.append(y[idx])
        zs.append(z[idx])
        es.append(energy[idx])
        tids.append(np.full(idx.size, int(tpc), dtype=np.int64))
        labs.append(np.full(idx.size, artificial_label, dtype=np.int64))

    if not xs:
        return out

    maps_4d, group_cls, group_tpcs = group_voxelize_pairs(
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(zs),
        np.concatenate(es),
        np.concatenate(tids),
        np.concatenate(labs),
        include_noise=False,
    )
    if maps_4d.shape[0] == 0:
        return out

    amps = predict_phi(
        maps_4d,
        model,
        np.asarray(group_tpcs, dtype=np.int64),
        target_scale=float(cfg.target_scale),
        batch_size=int(cfg.prediction_batch_size),
        raw_clip=cfg.raw_clip,
        min_prediction_threshold=float(cfg.min_prediction_threshold),
        device_policy=str(cfg.device_policy),
    )
    for g, cls in enumerate(np.asarray(group_cls, dtype=np.int64)):
        spec = label_to_spec.get(int(cls))
        if spec is None:
            continue
        out[spec] = (amps[g, :, None].astype(np.float32) * template[None, :]).astype(np.float32)
    return out


def _exact_update_affected_families(
    *,
    base_image: np.ndarray,
    old_hit_timestamps: np.ndarray,
    new_hit_timestamps: np.ndarray,
    affected_specs: set[tuple[int, int]],
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    model: Any,
    template: np.ndarray,
    cfg: Phase25Config,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    specs = sorted((int(t), int(t0)) for t, t0 in affected_specs)
    if not specs:
        return base_image, []

    old_imgs = _predict_family_images_batch(
        specs,
        hit_timestamps=old_hit_timestamps,
        hit_tpc_ids=hit_tpc_ids,
        x=x,
        y=y,
        z=z,
        energy=energy,
        model=model,
        template=template,
        cfg=cfg,
    )
    new_imgs = _predict_family_images_batch(
        specs,
        hit_timestamps=new_hit_timestamps,
        hit_tpc_ids=hit_tpc_ids,
        x=x,
        y=y,
        z=z,
        energy=energy,
        model=model,
        template=template,
        cfg=cfg,
    )
    out = np.asarray(base_image, dtype=np.float32).copy()
    records: list[dict[str, Any]] = []
    nt = out.shape[-1]
    for tpc, t0 in specs:
        old_shifted = _shift_image(old_imgs[(tpc, t0)], t0, nt=nt)
        new_shifted = _shift_image(new_imgs[(tpc, t0)], t0, nt=nt)
        delta = new_shifted - old_shifted
        out[int(tpc)] = np.clip(out[int(tpc)] + delta, 0.0, float(cfg.adc_clip))
        records.append(
            {
                "TPCid": int(tpc),
                "t0": int(t0),
                "old_sum": float(np.sum(old_shifted)),
                "new_sum": float(np.sum(new_shifted)),
                "delta_sum": float(np.sum(delta)),
            }
        )
    return out, records


def _loss_for_affected_specs(
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    affected_specs: set[tuple[int, int]],
    saturated_channel_cache: Any | None,
    cfg: Phase25Config,
) -> float:
    if not affected_specs:
        return 0.0

    actual = np.asarray(full_light_waveform, dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std, dtype=np.float32), 1e-6)
    model = np.asarray(base_image, dtype=np.float32)
    by_tpc: dict[int, list[int]] = {}
    for tpc, t0 in affected_specs:
        by_tpc.setdefault(int(tpc), []).append(int(t0))

    total = 0.0
    for tpc, t0s in by_tpc.items():
        if tpc < 0 or tpc >= model.shape[0]:
            continue
        keep = _keep_channel_indices(int(tpc), actual, saturated_channel_cache)
        if keep.size == 0:
            continue
        tmask = _window_mask(model.shape[-1], sorted(set(int(v) for v in t0s)), cfg)
        total += _weighted_loss(
            model[int(tpc), keep],
            actual[int(tpc), keep],
            std[int(tpc), keep],
            tmask,
            cfg.overflow_weight,
        )
    return float(total)


def _scan_light_overflows(
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    t0_candidates: dict[int, Iterable[Any]],
    hit_tpc_ids: np.ndarray,
    saturated_channel_cache: Any | None,
    allowed_tpcs: set[int],
    cfg: Phase25Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual = np.asarray(full_light_waveform, dtype=np.float32)
    model = np.asarray(base_image, dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std, dtype=np.float32), 1e-6)

    for tpc in sorted(allowed_tpcs):
        if int(tpc) not in t0_candidates:
            continue
        keep = _keep_channel_indices(int(tpc), actual, saturated_channel_cache)
        if keep.size == 0:
            continue
        a = actual[int(tpc), keep]
        m = model[int(tpc), keep]
        s = std[int(tpc), keep]
        t0s = _merge_close_t0s(t0_candidates.get(int(tpc), []), cfg.t0_merge_ticks)
        for t0 in t0s:
            tick = int(t0) + int(cfg.pulse_peak_tick)
            if tick < 0 or tick >= a.shape[1]:
                continue
            lo = max(0, tick - int(cfg.half_window_ticks))
            hi = min(a.shape[1], tick + int(cfg.half_window_ticks) + 1)
            threshold = np.maximum(float(cfg.light_overflow_sigma) * s[:, lo:hi], float(cfg.light_overflow_abs_adc))
            residual = m[:, lo:hi] - a[:, lo:hi]
            overflow = (residual > threshold) & (m[:, lo:hi] > float(cfg.light_model_activity_adc))
            peak_threshold = np.maximum(float(cfg.light_overflow_sigma) * s[:, tick], float(cfg.light_overflow_abs_adc))
            peak_residual = m[:, tick] - a[:, tick]
            peak_overflow = (peak_residual > peak_threshold) & (m[:, tick] > float(cfg.light_model_activity_adc))
            deficit = a[:, lo:hi] - m[:, lo:hi]
            peak_deficit = a[:, tick] - m[:, tick]
            n_of_ch = int(np.count_nonzero(np.any(overflow, axis=1)))
            if n_of_ch < int(cfg.light_min_overflow_channels):
                continue
            win_of = float(np.sum(np.clip(residual[overflow], 0, None)))
            pk_of = float(np.sum(np.clip(peak_residual[peak_overflow], 0, None)))
            win_def = float(np.sum(np.clip(deficit, 0, None)))
            pk_def = float(np.sum(np.clip(peak_deficit, 0, None)))
            rows.append(
                {
                    "TPCid": int(tpc),
                    "t0": int(t0),
                    "peak_tick": int(tick),
                    "overflow_channels": n_of_ch,
                    "peak_overflow_channels": int(np.count_nonzero(peak_overflow)),
                    "deficit_channels": int(np.count_nonzero(np.any(deficit > threshold, axis=1))),
                    "window_overflow": win_of,
                    "window_deficit": win_def,
                    "peak_overflow": pk_of,
                    "peak_deficit": pk_def,
                    "severity": float(pk_of / max(n_of_ch, 1) + 0.002 * win_of),
                }
            )

    rows.sort(key=lambda r: (-float(r["severity"]), -float(r["window_overflow"])))
    return rows[: int(cfg.light_overflow_rows_per_pass)]


def _build_light_components(
    *,
    TPCid: int,
    t0: int,
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    locked_hit_mask: np.ndarray | None,
    cfg: Phase25Config,
) -> list[dict[str, Any]]:
    mask = (
        (hit_tpc_ids == int(TPCid))
        & _finite_assigned(hit_timestamps)
        & (np.abs(hit_timestamps - float(t0)) <= float(cfg.t0_match_ticks))
    )
    if locked_hit_mask is not None:
        mask &= ~np.asarray(locked_hit_mask, dtype=bool)
    idx = np.flatnonzero(mask).astype(np.int64)
    if idx.size == 0:
        return []
    pts = np.column_stack((x[idx], y[idx], z[idx])).astype(np.float64)
    comps = _connected_components(pts, cfg.light_component_radius_cm)
    rows: list[dict[str, Any]] = []
    for comp_id, comp_local in enumerate(comps, start=1):
        gidx = idx[comp_local].astype(np.int64)
        e_sum = float(np.sum(energy[gidx]))
        if gidx.size < int(cfg.light_min_component_hits):
            continue
        if e_sum < float(cfg.light_min_component_energy_mev):
            continue

        labels_in = labels_global[gidx].astype(np.int64)
        valid_labels = labels_in[labels_in >= 0]
        if valid_labels.size == 0:
            parent_label = -1
            parent_fraction = 0.0
        else:
            vals, counts = np.unique(valid_labels, return_counts=True)
            best = int(np.argmax(counts))
            parent_label = int(vals[best])
            parent_fraction = float(counts[best] / max(int(gidx.size), 1))
        geom = _pca_geometry(gidx, x=x, y=y, z=z, energy=energy)

        rows.append(
            {
                "TPCid": int(TPCid),
                "component_id": int(comp_id),
                "current_t0": int(t0),
                "hit_indices": gidx,
                "n_hits": int(gidx.size),
                "energy_mev": e_sum,
                "parent_label": int(parent_label),
                "parent_fraction": float(parent_fraction),
                "labels": _label_summary(gidx, labels_global),
                "centroid": geom["centroid"],
                "direction": geom["direction"],
                "linearity": float(geom["linearity"]),
            }
        )
    rows.sort(key=lambda r: (-float(r["energy_mev"]), -int(r["n_hits"])))
    return rows[: int(cfg.light_max_components_per_t0)]


def _light_deficit_summary_kept(
    *,
    t0: int,
    actual_kept: np.ndarray,
    model_kept: np.ndarray,
    std_kept: np.ndarray,
    cfg: Phase25Config,
) -> dict[str, Any] | None:
    actual = np.asarray(actual_kept, dtype=np.float32)
    model = np.asarray(model_kept, dtype=np.float32)
    std = np.maximum(np.asarray(std_kept, dtype=np.float32), 1e-6)

    tick = int(t0) + int(cfg.pulse_peak_tick)
    if tick < 0 or tick >= actual.shape[1]:
        return None

    lo = max(0, tick - int(cfg.half_window_ticks))
    hi = min(actual.shape[1], tick + int(cfg.half_window_ticks) + 1)

    deficit = np.clip(actual[:, lo:hi] - model[:, lo:hi], 0.0, None)
    overflow = np.clip(model[:, lo:hi] - actual[:, lo:hi], 0.0, None)
    threshold = np.maximum(
        float(cfg.light_overflow_sigma) * std[:, lo:hi],
        float(cfg.light_overflow_abs_adc),
    )

    return {
        "t0": int(t0),
        "peak_tick": int(tick),
        "window_lo": int(lo),
        "window_hi": int(hi),
        "deficit_sum": float(np.sum(deficit)),
        "overflow_sum": float(np.sum(overflow)),
        "n_deficit_channels": int(np.count_nonzero(np.any(deficit > threshold, axis=1))),
        "n_overflow_channels": int(np.count_nonzero(np.any(overflow > threshold, axis=1))),
    }


def _choose_destination_rows(
    *,
    TPCid: int,
    source_t0: int,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    t0_candidates: dict[int, Iterable[Any]],
    keep_channels: np.ndarray,
    cfg: Phase25Config,
) -> list[dict[str, Any]]:
    actual_kept = np.asarray(full_light_waveform[int(TPCid), keep_channels], dtype=np.float32)
    model_kept = np.asarray(base_image[int(TPCid), keep_channels], dtype=np.float32)
    std_kept = np.maximum(np.asarray(full_light_std[int(TPCid), keep_channels], dtype=np.float32), 1e-6)
    candidates = [
        int(t)
        for t in _merge_close_t0s(t0_candidates.get(int(TPCid), []), cfg.t0_merge_ticks)
        if abs(int(t) - int(source_t0)) > int(cfg.t0_merge_ticks)
    ]
    rows = []
    for t0 in candidates:
        summary = _light_deficit_summary_kept(
            t0=int(t0),
            actual_kept=actual_kept,
            model_kept=model_kept,
            std_kept=std_kept,
            cfg=cfg,
        )
        if summary is None:
            continue
        if float(summary["deficit_sum"]) < float(cfg.light_min_dest_deficit_sum):
            continue
        rows.append(summary)
    rows.sort(key=lambda r: (-float(r["deficit_sum"]), int(r["t0"])))
    return rows[: int(cfg.light_max_dest_t0s)]


def _component_proxy_image(
    *,
    component: dict[str, Any],
    image_maps: dict[Any, Any] | None,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    energy: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if image_maps is None:
        return None, "missing_imageMaps"

    parent_label = int(component.get("parent_label", -1))
    tpc = int(component["TPCid"])
    if parent_label < 0:
        return None, "no_parent_label"

    key = (parent_label, tpc)
    if key not in image_maps:
        return None, f"missing_imageMap_{key}"

    parent_img = np.asarray(image_maps[key], dtype=np.float32)
    if parent_img.ndim != 2:
        return None, f"bad_image_shape_{parent_img.shape}"

    comp_idx = np.asarray(component["hit_indices"], dtype=np.int64)
    parent_mask = (np.asarray(labels_global, dtype=np.int64) == parent_label) & (
        np.asarray(hit_tpc_ids, dtype=np.int32) == tpc
    )
    parent_energy = float(np.sum(np.asarray(energy, dtype=np.float64)[parent_mask]))
    if parent_energy <= 1e-9:
        return None, "zero_parent_energy"

    comp_energy = float(np.sum(np.asarray(energy, dtype=np.float64)[comp_idx]))
    scale = comp_energy / parent_energy
    return (float(scale) * parent_img).astype(np.float32), f"scaled_parent_{parent_label}_scale_{scale:.4f}"


def _light_spatial_prior(
    component: dict[str, Any],
    dest_t0: int,
    all_components_by_t0: dict[int, list[dict[str, Any]]],
    cfg: Phase25Config,
) -> float:
    if not bool(cfg.light_use_spatial_prior):
        return 0.0

    dest_components = all_components_by_t0.get(int(dest_t0), [])
    if not dest_components:
        return 0.0

    c0 = np.asarray(component["centroid"], dtype=np.float64)
    d0 = np.asarray(component["direction"], dtype=np.float64)
    best: float | None = None

    for other in dest_components:
        c1 = np.asarray(other["centroid"], dtype=np.float64)
        d1 = np.asarray(other["direction"], dtype=np.float64)
        gap = float(np.linalg.norm(c0 - c1))
        angle_pen = 1.0 - abs(float(np.dot(d0, d1)))
        score = min(gap / 30.0, 4.0) + min(angle_pen / 0.30, 4.0)
        if best is None or score < best:
            best = float(score)

    return float(best if best is not None else 0.0)


def _try_light_repair_row(
    row: dict[str, Any],
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    image_maps: dict[Any, Any] | None,
    t0_candidates: dict[int, Iterable[Any]],
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    saturated_channel_cache: Any | None,
    locked_hit_mask: np.ndarray | None,
    cfg: Phase25Config,
) -> dict[str, Any] | None:
    tpc = int(row["TPCid"])
    source_t0 = int(row["t0"])

    keep = _keep_channel_indices(tpc, np.asarray(full_light_waveform, dtype=np.float32), saturated_channel_cache)
    if keep.size == 0:
        return None

    actual_kept = np.asarray(full_light_waveform[tpc, keep], dtype=np.float32)
    std_kept = np.maximum(np.asarray(full_light_std[tpc, keep], dtype=np.float32), 1e-6)
    model_kept = np.asarray(base_image[tpc, keep], dtype=np.float32)
    tpc_model = np.asarray(base_image[tpc], dtype=np.float32)
    n_ticks = int(tpc_model.shape[-1])

    candidate_dest_t0s = [
        int(t)
        for t in _merge_close_t0s(t0_candidates.get(int(tpc), []), cfg.t0_merge_ticks)
        if abs(int(t) - int(source_t0)) > int(cfg.t0_merge_ticks)
    ]
    if not candidate_dest_t0s:
        return None

    all_components_by_t0: dict[int, list[dict[str, Any]]] = {}
    for t0 in [source_t0] + candidate_dest_t0s:
        all_components_by_t0[int(t0)] = _build_light_components(
            TPCid=tpc,
            t0=int(t0),
            hit_timestamps=hit_timestamps,
            hit_tpc_ids=hit_tpc_ids,
            labels_global=labels_global,
            x=x,
            y=y,
            z=z,
            energy=energy,
            locked_hit_mask=locked_hit_mask,
            cfg=cfg,
        )

    donor_components = all_components_by_t0.get(int(source_t0), [])
    if not donor_components:
        return None

    dest_rows = _choose_destination_rows(
        TPCid=tpc,
        source_t0=source_t0,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        t0_candidates=t0_candidates,
        keep_channels=keep,
        cfg=cfg,
    )
    if not dest_rows:
        return None

    focus_t0s = [int(source_t0)] + [int(r["t0"]) for r in dest_rows]
    tmask = _window_mask(n_ticks, focus_t0s, cfg)
    before_loss = _weighted_loss(model_kept, actual_kept, std_kept, tmask, cfg.overflow_weight)

    old_summary_before = _light_deficit_summary_kept(
        t0=source_t0,
        actual_kept=actual_kept,
        model_kept=model_kept,
        std_kept=std_kept,
        cfg=cfg,
    )
    old_overflow_before = (
        float(old_summary_before["overflow_sum"])
        if old_summary_before is not None
        else float(row.get("window_overflow", 0.0))
    )

    trials: list[dict[str, Any]] = []

    for component in donor_components:
        proxy, image_status = _component_proxy_image(
            component=component,
            image_maps=image_maps,
            labels_global=labels_global,
            hit_tpc_ids=hit_tpc_ids,
            energy=energy,
        )
        if proxy is None:
            continue

        old_shift = _shift_image(proxy, source_t0, nt=n_ticks)
        for dest in dest_rows:
            dest_t0 = int(dest["t0"])
            new_shift = _shift_image(proxy, dest_t0, nt=n_ticks)
            delta = new_shift - old_shift
            trial_model_kept = np.clip(model_kept + delta[keep], 0.0, float(cfg.adc_clip))
            after_loss = _weighted_loss(trial_model_kept, actual_kept, std_kept, tmask, cfg.overflow_weight)
            loss_improvement = before_loss - after_loss

            old_after = _light_deficit_summary_kept(
                t0=source_t0,
                actual_kept=actual_kept,
                model_kept=trial_model_kept,
                std_kept=std_kept,
                cfg=cfg,
            )
            dest_after = _light_deficit_summary_kept(
                t0=dest_t0,
                actual_kept=actual_kept,
                model_kept=trial_model_kept,
                std_kept=std_kept,
                cfg=cfg,
            )

            old_overflow_after = float(old_after["overflow_sum"]) if old_after is not None else np.inf
            dest_deficit_after = float(dest_after["deficit_sum"]) if dest_after is not None else np.inf
            dest_overflow_after = float(dest_after["overflow_sum"]) if dest_after is not None else np.inf

            old_overflow_reduction = old_overflow_before - old_overflow_after
            dest_deficit_reduction = float(dest["deficit_sum"]) - dest_deficit_after
            dest_new_overflow = max(dest_overflow_after - float(dest["overflow_sum"]), 0.0)

            spatial_prior = _light_spatial_prior(component, dest_t0, all_components_by_t0, cfg)
            adjusted_gain = (
                float(loss_improvement)
                + float(old_overflow_reduction)
                + float(dest_deficit_reduction)
                - float(cfg.light_spatial_prior_weight)
                * float(spatial_prior)
                * max(float(component["energy_mev"]), 1.0)
            )

            energy_ok = _move_energy_within_limit(float(component["energy_mev"]), cfg)
            reject_reason = "move_energy_exceeds_limit" if not energy_ok else ""
            accept_like = (
                energy_ok
                and loss_improvement >= float(cfg.light_min_loss_improvement)
                and old_overflow_reduction >= float(cfg.light_min_old_overflow_reduction)
                and dest_deficit_reduction >= float(cfg.light_min_dest_deficit_reduction)
                and dest_new_overflow
                <= float(cfg.light_max_dest_new_overflow_frac) * max(float(dest_deficit_reduction), 1.0)
            )

            trials.append({
                "TPCid": int(tpc),
                "old_t0": int(source_t0),
                "new_t0": int(dest_t0),
                "component_id": int(component["component_id"]),
                "hit_indices": np.asarray(component["hit_indices"], dtype=np.int64),
                "n_hits": int(component["n_hits"]),
                "energy_mev": float(component["energy_mev"]),
                "max_move_energy_per_tpc_mev": (
                    None if cfg.max_move_energy_per_tpc_mev is None else float(cfg.max_move_energy_per_tpc_mev)
                ),
                "parent_label": int(component["parent_label"]),
                "parent_fraction": float(component["parent_fraction"]),
                "labels": str(component["labels"]),
                "image_status": str(image_status),
                "before_loss": float(before_loss),
                "after_loss": float(after_loss),
                "loss_improvement": float(loss_improvement),
                "old_overflow_before": float(old_overflow_before),
                "old_overflow_after": float(old_overflow_after),
                "old_overflow_reduction": float(old_overflow_reduction),
                "dest_deficit_before": float(dest["deficit_sum"]),
                "dest_deficit_after": float(dest_deficit_after),
                "dest_deficit_reduction": float(dest_deficit_reduction),
                "dest_overflow_before": float(dest["overflow_sum"]),
                "dest_overflow_after": float(dest_overflow_after),
                "dest_new_overflow": float(dest_new_overflow),
                "spatial_prior": float(spatial_prior),
                "adjusted_gain": float(adjusted_gain),
                "accept_like": bool(accept_like),
                "accepted": bool(accept_like),
                "reject_reason": reject_reason,
            })

    if not trials:
        return None

    trials.sort(
        key=lambda r: (
            not bool(r["accept_like"]),
            -float(r["adjusted_gain"]),
            -float(r["loss_improvement"]),
            -float(r["old_overflow_reduction"]),
            -float(r["dest_deficit_reduction"]),
            -float(r["energy_mev"]),
        )
    )
    best = trials[0]

    return best


def run_phase25_amendment(
    *,
    base_image: np.ndarray,
    hit_timestamps: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    t0_candidates: dict[int, Iterable[Any]],
    image_maps: dict[Any, Any] | None,
    model: Any,
    waveform_template: np.ndarray,
    label_info: Any | None = None,
    saturated_channel_cache: Any | None = None,
    config: Phase25Config | None = None,
) -> dict[str, Any]:
    """Run Phase 2.5 and return amended arrays plus compact diagnostics."""
    cfg = config or Phase25Config()
    t_start = time.time()

    base_out = np.asarray(base_image, dtype=np.float32).copy()
    hit_ts_out = np.asarray(hit_timestamps, dtype=np.float32).copy()

    labels_arr = np.asarray(labels_global, dtype=np.int64)
    tpc_arr = np.asarray(hit_tpc_ids, dtype=np.int32)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    e_arr = np.asarray(energy, dtype=np.float64)

    all_tpcs = set(int(v) for v in np.unique(tpc_arr))
    shower_tpcs = infer_shower_tpcs(
        labels_global=labels_arr,
        hit_tpc_ids=tpc_arr,
        label_info=label_info,
        shower_keywords=cfg.shower_keywords,
    ) if cfg.skip_shower_tpcs else set()
    allowed_tpcs = all_tpcs - shower_tpcs
    if cfg.freeze_long_track_hits:
        locked_long_track_hit_mask, long_track_lock_records = infer_long_track_hit_mask(
            labels_global=labels_arr,
            hit_tpc_ids=tpc_arr,
            label_info=label_info,
            track_keywords=cfg.long_track_keywords,
            exclude_keywords=cfg.long_track_exclude_keywords,
            min_tpcs=cfg.long_track_min_tpcs,
            return_records=True,
        )
    else:
        locked_long_track_hit_mask = np.zeros(labels_arr.shape[0], dtype=bool)
        long_track_lock_records = []

    spatial_moves: list[dict[str, Any]] = []
    spatial_trials: list[dict[str, Any]] = []
    light_moves: list[dict[str, Any]] = []
    light_trials: list[dict[str, Any]] = []
    family_update_records: list[dict[str, Any]] = []

    if cfg.verbose:
        lock_text = (
            f" | locked_long_track_hits={int(np.count_nonzero(locked_long_track_hit_mask))}"
            if bool(cfg.freeze_long_track_hits)
            else ""
        )
        print(
            f"Phase2.5 amendment | TPCs={len(all_tpcs)} | "
            f"skip_shower={len(shower_tpcs)} | "
            f"spatial={cfg.enable_spatial} | light={cfg.enable_light}"
            f"{lock_text}"
        )

    if cfg.enable_spatial:
        rows = _detect_mixed_t0_pairs(
            hit_timestamps=hit_ts_out,
            hit_tpc_ids=tpc_arr,
            labels_global=labels_arr,
            x=x_arr,
            y=y_arr,
            z=z_arr,
            energy=e_arr,
            allowed_tpcs=allowed_tpcs,
            locked_hit_mask=locked_long_track_hit_mask,
            cfg=cfg,
        )
        for row in rows:
            move = _restore_mixed_pair_spatially(
                row,
                hit_timestamps=hit_ts_out,
                hit_tpc_ids=tpc_arr,
                labels_global=labels_arr,
                x=x_arr,
                y=y_arr,
                z=z_arr,
                energy=e_arr,
                locked_hit_mask=locked_long_track_hit_mask,
                cfg=cfg,
            )
            if move is None:
                continue
            move_energy = float(move.get("energy_moved_mev", 0.0))
            moved_idx = np.asarray(move["moved_idx"], dtype=np.int64)
            old_vals = np.round(hit_ts_out[moved_idx]).astype(np.int32)
            affected: set[tuple[int, int]] = set()
            for old_t0, new_t0 in zip(old_vals, move["new_t0"].astype(np.int32)):
                affected.add((int(move["TPCid"]), int(old_t0)))
                affected.add((int(move["TPCid"]), int(new_t0)))

            if not _move_energy_within_limit(move_energy, cfg):
                trial = dict(move)
                trial.update(
                    {
                        "old_t0": old_vals.astype(np.int32),
                        "affected_specs": sorted((int(t), int(t0)) for t, t0 in affected),
                        "before_loss": np.nan,
                        "after_loss": np.nan,
                        "loss_delta": np.nan,
                        "max_loss_increase": float(cfg.spatial_max_loss_increase),
                        "max_move_energy_per_tpc_mev": (
                            None
                            if cfg.max_move_energy_per_tpc_mev is None
                            else float(cfg.max_move_energy_per_tpc_mev)
                        ),
                        "accepted": False,
                        "reject_reason": "move_energy_exceeds_limit",
                    }
                )
                spatial_trials.append(trial)
                continue

            before_ts = hit_ts_out.copy()
            before_base = base_out.copy()
            trial_ts = hit_ts_out.copy()
            trial_ts[moved_idx] = move["new_t0"]

            before_loss = _loss_for_affected_specs(
                base_image=before_base,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                affected_specs=affected,
                saturated_channel_cache=saturated_channel_cache,
                cfg=cfg,
            )
            trial_base, records = _exact_update_affected_families(
                base_image=before_base,
                old_hit_timestamps=before_ts,
                new_hit_timestamps=trial_ts,
                affected_specs=affected,
                hit_tpc_ids=tpc_arr,
                x=x_arr,
                y=y_arr,
                z=z_arr,
                energy=e_arr,
                model=model,
                template=waveform_template,
                cfg=cfg,
            )
            after_loss = _loss_for_affected_specs(
                base_image=trial_base,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                affected_specs=affected,
                saturated_channel_cache=saturated_channel_cache,
                cfg=cfg,
            )
            loss_delta = float(after_loss - before_loss)

            trial = dict(move)
            trial.update(
                {
                    "old_t0": old_vals.astype(np.int32),
                    "affected_specs": sorted((int(t), int(t0)) for t, t0 in affected),
                    "before_loss": float(before_loss),
                    "after_loss": float(after_loss),
                    "loss_delta": float(loss_delta),
                    "max_loss_increase": float(cfg.spatial_max_loss_increase),
                    "max_move_energy_per_tpc_mev": (
                        None
                        if cfg.max_move_energy_per_tpc_mev is None
                        else float(cfg.max_move_energy_per_tpc_mev)
                    ),
                    "accepted": True,
                    "reject_reason": "",
                }
            )
            spatial_trials.append(trial)

            hit_ts_out = trial_ts
            base_out = trial_base
            family_update_records.extend(records)
            spatial_moves.append(trial)

    if cfg.enable_light:
        moves_by_tpc: dict[int, int] = {}
        for _ in range(int(cfg.light_max_total_moves)):
            rows = _scan_light_overflows(
                base_image=base_out,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                t0_candidates=t0_candidates,
                hit_tpc_ids=tpc_arr,
                saturated_channel_cache=saturated_channel_cache,
                allowed_tpcs=allowed_tpcs,
                cfg=cfg,
            )
            if not rows:
                break
            accepted_this_pass = False
            for row in rows:
                tpc = int(row["TPCid"])
                if moves_by_tpc.get(tpc, 0) >= int(cfg.light_max_moves_per_tpc):
                    continue
                trial = _try_light_repair_row(
                    row,
                    base_image=base_out,
                    full_light_waveform=full_light_waveform,
                    full_light_std=full_light_std,
                    image_maps=image_maps,
                    t0_candidates=t0_candidates,
                    hit_timestamps=hit_ts_out,
                    hit_tpc_ids=tpc_arr,
                    labels_global=labels_arr,
                    x=x_arr,
                    y=y_arr,
                    z=z_arr,
                    energy=e_arr,
                    saturated_channel_cache=saturated_channel_cache,
                    locked_hit_mask=None,
                    cfg=cfg,
                )
                if trial is None:
                    continue
                trial = dict(trial)
                trial["overflow_row"] = dict(row)
                light_trials.append(trial)
                if not bool(trial.get("accepted", False)):
                    continue
                before_ts = hit_ts_out.copy()
                before_base = base_out.copy()
                moved = np.asarray(trial["hit_indices"], dtype=np.int64)
                old_t0 = int(trial["old_t0"])
                new_t0 = int(trial["new_t0"])
                hit_ts_out[moved] = np.float32(new_t0)
                affected = {(tpc, old_t0), (tpc, new_t0)}
                base_out, records = _exact_update_affected_families(
                    base_image=before_base,
                    old_hit_timestamps=before_ts,
                    new_hit_timestamps=hit_ts_out,
                    affected_specs=affected,
                    hit_tpc_ids=tpc_arr,
                    x=x_arr,
                    y=y_arr,
                    z=z_arr,
                    energy=e_arr,
                    model=model,
                    template=waveform_template,
                    cfg=cfg,
                )
                family_update_records.extend(records)
                light_moves.append(trial)
                moves_by_tpc[tpc] = moves_by_tpc.get(tpc, 0) + 1
                accepted_this_pass = True
                break
            if not accepted_this_pass:
                break

    elapsed = time.time() - t_start
    if cfg.verbose:
        moved_spatial = int(sum(m["n_moved"] for m in spatial_moves))
        moved_light = int(sum(m["n_hits"] for m in light_moves))
        print(
            "Phase2.5 done | "
            f"spatial_moves={len(spatial_moves)} ({moved_spatial} hits) | "
            f"light_moves={len(light_moves)} ({moved_light} hits) | "
            f"family_updates={len(family_update_records)} | elapsed={elapsed:.1f}s"
        )

    return {
        "baseImage": base_out.astype(np.float32),
        "hit_timestamps": hit_ts_out.astype(np.float32),
        "spatial_moves": spatial_moves,
        "spatial_trials": spatial_trials,
        "light_moves": light_moves,
        "light_trials": light_trials,
        "family_update_records": family_update_records,
        "skipped_shower_tpcs": sorted(int(v) for v in shower_tpcs),
        "locked_long_track_hits": int(np.count_nonzero(locked_long_track_hit_mask)),
        "long_track_lock_records": long_track_lock_records,
        "allowed_tpcs": sorted(int(v) for v in allowed_tpcs),
        "elapsed_s": float(elapsed),
        "config": cfg,
    }


def run_phase25_amendment_from_namespace(
    namespace: dict[str, Any],
    *,
    config: Phase25Config | None = None,
    commit: bool = False,
) -> dict[str, Any]:
    """Notebook-friendly wrapper using the standard v_testing globals."""
    cfg = config or Phase25Config()
    full_std = namespace.get("fullLightStd")
    if full_std is None:
        full_std = namespace.get("fullLightStd_phase2")
    if full_std is None:
        full_std = namespace.get("fullLightStd_phase1")
    if full_std is None:
        full_std = np.ones_like(np.asarray(_get(namespace, "fullLightWaveform"), dtype=np.float32))

    target_scale = cfg.target_scale
    first_stage_config = namespace.get("FIRST_STAGE_CONFIG")
    if first_stage_config is not None:
        try:
            target_scale = float(first_stage_config.prediction.target_scale)
        except Exception:
            pass
    cfg = Phase25Config(**{**cfg.__dict__, "target_scale": float(target_scale)})

    result = run_phase25_amendment(
        base_image=_get(namespace, "baseImage"),
        hit_timestamps=_get(namespace, "hit_timestamps"),
        full_light_waveform=_get(namespace, "fullLightWaveform"),
        full_light_std=full_std,
        labels_global=_get(namespace, "labels_global"),
        hit_tpc_ids=_get(namespace, "hitTPCid"),
        x=_get(namespace, "xset"),
        y=_get(namespace, "yset"),
        z=_get(namespace, "zset"),
        energy=_get(namespace, "Eset"),
        t0_candidates=_get(namespace, "t0Candidates"),
        image_maps=namespace.get("imageMaps"),
        model=_get(namespace, "model"),
        waveform_template=_get(namespace, "wvfm_tmpl"),
        label_info=namespace.get("label_info"),
        saturated_channel_cache=namespace.get("saturated_channel_cache"),
        config=cfg,
    )

    if commit:
        namespace["baseImage"] = result["baseImage"]
        namespace["hit_timestamps"] = result["hit_timestamps"]
        namespace["phase25_result"] = result
    return result
