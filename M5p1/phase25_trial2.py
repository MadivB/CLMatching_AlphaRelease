"""Trial-2 amendment helpers for compact notebook testing.

This module keeps the trial notebook small while reusing the existing
``phase25_amendment`` implementation for the expensive spatial repair and exact
GPU family-image updates.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

try:
    from . import phase25_amendment as p25
    from .flash_cluster_table import (
        ensure_flash_cluster_flags,
        flash_cluster_table_rows,
        mark_flash_cluster_assignment,
        source_t0s_from_received_flash_table,
    )
    from .truth_plotting import extract_event_hit_energy_and_truth_t0
except Exception:  # pragma: no cover - direct notebook import fallback
    import phase25_amendment as p25
    from flash_cluster_table import (
        ensure_flash_cluster_flags,
        flash_cluster_table_rows,
        mark_flash_cluster_assignment,
        source_t0s_from_received_flash_table,
    )
    from truth_plotting import extract_event_hit_energy_and_truth_t0


@dataclass
class Trial2Config:
    """Configuration for the trial-2 compact amendment."""

    verbose: bool = True
    commit: bool = True

    # Large-cluster flash-grid correction.
    enable_large_flash_grid_correction: bool = True
    large_cluster_min_energy_mev: float = 50.0
    large_grid_offsets_ticks: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    large_min_loss_improvement: float = 0.0
    large_max_clusters: int | None = None

    # Spatial repair.
    enable_spatial: bool = True
    skip_shower_tpcs: bool = True
    max_move_energy_per_tpc_mev: float | None = 80.0
    spatial_t0_match_ticks: int = 10
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
    spatial_max_loss_increase: float = 1.0e4

    # Light repair.
    enable_light: bool = True
    light_skip_shower_tpcs: bool = True
    light_source_exact_associated_t0s: bool = True
    light_t0_match_ticks: int = 5
    light_overflow_sigma: float = 5.0
    light_overflow_abs_adc: float = 400.0
    light_model_activity_adc: float = 400.0
    light_min_overflow_channels: int = 6
    light_source_channel_drop_strict_gt: int = 12
    light_max_dest_new_channels_abs: float = 8.0
    light_max_dest_new_channels_frac_of_src_red: float = 0.75
    light_min_dloss_per_mev: float = 3.0e7
    light_partial_move_frac_max: float = 0.35
    light_partial_source_remain_frac_max: float = 0.50
    light_max_source_deficit_increase: float = 3.0e5
    light_max_source_deficit_increase_per_mev: float = 4000.0
    light_flash_cluster_match_ticks: float = 5.0
    light_max_total_moves: int = 24
    light_max_moves_per_tpc: int = 3
    light_use_physical_chi2: bool = False
    phys_min_source_ofch_reduction: int = 8
    phys_min_dchi2_improvement: float = 5.0e2
    phys_min_dchi2_per_mev: float = 1.0e1
    phys_std_floor: float = 1.0e-6
    light_veto_multitpc_track: bool = True
    light_veto_track_min_tpcs: int = 4
    light_veto_override_min_candidates: int = 3
    light_use_rescue_branches: bool = False
    light_rescue_high_e_min_e: float = 100.0
    light_rescue_high_e_src_red_ge: int = 9
    light_rescue_high_e_max_src_remain: float = 0.85
    light_rescue_high_e_min_move_frac: float = 0.12
    light_rescue_high_e_min_dloss_per_mev: float = 1.0e6
    light_rescue_high_e_clear_src_remain: float = 0.15
    light_rescue_high_e_clear_dst_frac: float = 4.50
    light_rescue_high_e_dst_abs: float = 32.0
    light_rescue_high_e_dst_frac: float = 2.20
    light_rescue_src_clear_min_e: float = 15.0
    light_rescue_src_clear_src_red_ge: int = 14
    light_rescue_src_clear_max_src_remain: float = 0.25
    light_rescue_src_clear_dst_abs: float = 4.0
    light_rescue_src_clear_dst_frac: float = 0.30
    light_rescue_oldok_partial_min_e: float = 25.0
    light_rescue_oldok_partial_src_red_ge: int = 6
    light_rescue_oldok_partial_dst_max: int = 2
    light_rescue_oldok_partial_max_src_remain: float = 0.85
    light_rescue_oldok_partial_min_move_frac: float = 0.12
    light_rescue_oldok_partial_min_dloss_per_mev: float = 5.0e7

    # Exact family prediction.
    prediction_batch_size: int = 8
    device_policy: str = "auto"

    # Truth diagnostics.
    truth_tolerance_ticks: float = 25.0
    min_net_truth_energy_mev: float = 0.10
    print_top_rows: int = 80

    extra_metadata: dict[str, Any] = field(default_factory=dict)


def _get(namespace: dict[str, Any], name: str) -> Any:
    if name not in namespace:
        raise KeyError(f"trial2 requires `{name}` in the notebook namespace.")
    return namespace[name]


def _as_tpc_dict(values: Any) -> dict[int, list[Any]]:
    if values is None:
        return {}
    if isinstance(values, dict):
        return {int(k): list(v) for k, v in values.items()}
    return {int(i): list(v) for i, v in enumerate(values)}


def _candidate_values_for_tpc(t0_candidates: Any, tpc: int) -> list[float]:
    try:
        values = t0_candidates[int(tpc)]
    except Exception:
        return []
    out = []
    for value in values:
        try:
            val = float(value)
        except Exception:
            continue
        if np.isfinite(val) and int(round(val)) != 0:
            out.append(float(val))
    return out


def _set_candidate_values_for_tpc(t0_candidates: Any, tpc: int, values: Iterable[float]) -> None:
    cleaned = sorted(
        float(v)
        for v in values
        if v is not None and np.isfinite(float(v)) and int(round(float(v))) != 0
    )
    if isinstance(t0_candidates, dict):
        t0_candidates[int(tpc)] = cleaned
    else:
        t0_candidates[int(tpc)] = cleaned


def _merge_close_float_t0s(values: Iterable[Any], merge_ticks: float) -> list[float]:
    vals = sorted(
        float(v)
        for v in values
        if v is not None and np.isfinite(float(v)) and int(round(float(v))) != 0
    )
    if not vals:
        return []

    groups: list[list[float]] = [[vals[0]]]
    for value in vals[1:]:
        if abs(float(value) - float(groups[-1][-1])) <= float(merge_ticks):
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return [float(np.median(group)) for group in groups]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return float(np.median(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else 0.0
    vals = vals[good]
    w = w[good]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cutoff = 0.5 * float(np.sum(w))
    return float(vals[min(int(np.searchsorted(np.cumsum(w), cutoff, side="left")), vals.size - 1)])


def _label_summary(indices: np.ndarray, labels_global: np.ndarray, max_items: int = 6) -> str:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return "[]"
    labels = np.asarray(labels_global, dtype=np.int64)[idx]
    labels = labels[labels >= 0]
    if labels.size == 0:
        return "[]"
    vals, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    return "[" + ", ".join(f"{int(vals[i])}:{int(counts[i])}" for i in order[:max_items]) + "]"


def _canonicalize_candidate_t0(
    t0_candidates: Any,
    tpc: int,
    t0: float,
    *,
    merge_ticks: float,
) -> None:
    values = [
        value
        for value in _candidate_values_for_tpc(t0_candidates, int(tpc))
        if abs(float(value) - float(t0)) > float(merge_ticks)
    ]
    values.append(float(t0))
    _set_candidate_values_for_tpc(t0_candidates, int(tpc), _merge_close_float_t0s(values, merge_ticks))


def _canonicalize_flash_table_from_assignments(
    t0_candidates: Any,
    *,
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    energy: np.ndarray,
    allowed_tpcs: Iterable[int],
    match_ticks: float,
) -> tuple[Any, list[dict[str, Any]]]:
    """Merge nearby flash t0s and mark which flash entries have assigned charge.

    Only existing flash-table entries are considered. If an entry has assigned
    charge within ``match_ticks``, nearby flash entries are canonicalized to the
    assigned charge family's weighted-median t0. This keeps close aliases such as
    30/31/32/34 from being scanned as independent light-excess sources.
    """
    hit_ts = np.asarray(hit_timestamps, dtype=np.float64)
    hit_tpc = np.asarray(hit_tpc_ids, dtype=np.int32)
    labels = np.asarray(labels_global, dtype=np.int64)
    e = np.asarray(energy, dtype=np.float64)
    finite = np.isfinite(hit_ts) & (hit_ts >= 0) & (labels >= 0)

    rows: list[dict[str, Any]] = []
    for tpc in sorted(int(v) for v in allowed_tpcs):
        raw_values = _candidate_values_for_tpc(t0_candidates, int(tpc))
        if not raw_values:
            continue

        flash_values = _merge_close_float_t0s(raw_values, float(match_ticks))
        canonical_values = []
        tpc_mask = finite & (hit_tpc == int(tpc))

        for flash_t0 in flash_values:
            assoc_idx = np.flatnonzero(tpc_mask & (np.abs(hit_ts - float(flash_t0)) <= float(match_ticks))).astype(np.int64)
            associated = assoc_idx.size > 0
            canonical_t0 = float(flash_t0)
            if associated:
                canonical_t0 = _weighted_median(hit_ts[assoc_idx], np.clip(e[assoc_idx], 1e-9, None))

            canonical_values.append(float(canonical_t0))
            rows.append(
                {
                    "TPCid": int(tpc),
                    "flash_t0_before": float(flash_t0),
                    "t0": float(canonical_t0),
                    "associated": bool(associated),
                    "n_hits": int(assoc_idx.size),
                    "energy_mev": float(np.sum(e[assoc_idx])) if assoc_idx.size else 0.0,
                    "labels": _label_summary(assoc_idx, labels),
                }
            )

        _set_candidate_values_for_tpc(
            t0_candidates,
            int(tpc),
            _merge_close_float_t0s(canonical_values, float(match_ticks)),
        )

    return t0_candidates, rows


def _weighted_loss_window(
    model_tpc: np.ndarray,
    actual_tpc: np.ndarray,
    std_tpc: np.ndarray,
    t0_values: Iterable[float],
    cfg: p25.Phase25Config,
) -> float:
    tmask = p25._window_mask(model_tpc.shape[-1], [int(round(float(v))) for v in t0_values], cfg)
    std = np.maximum(np.asarray(std_tpc, dtype=np.float32), 1e-6)
    return p25._weighted_loss(
        np.asarray(model_tpc, dtype=np.float32),
        np.asarray(actual_tpc, dtype=np.float32),
        std,
        tmask,
        cfg.overflow_weight,
    )


def _shift_image_fractional(image: np.ndarray, shift_ticks: float, *, nt: int) -> np.ndarray:
    """Shift an unshifted image by a possibly fractional t0 using linear interpolation."""
    img = np.asarray(image, dtype=np.float32)
    out = np.zeros((img.shape[0], int(nt)), dtype=np.float32)
    src_x = np.arange(img.shape[1], dtype=np.float32)
    dst_x = np.arange(int(nt), dtype=np.float32) - float(shift_ticks)
    for ch in range(img.shape[0]):
        out[ch] = np.interp(dst_x, src_x, img[ch], left=0.0, right=0.0).astype(np.float32)
    return out


def run_large_cluster_flash_grid_correction_from_namespace(
    namespace: dict[str, Any],
    *,
    config: Trial2Config,
    commit: bool = True,
) -> dict[str, Any]:
    """Fine-correct large non-track/shower clusters around existing flash t0s.

    The correction is deliberately local: candidates are existing flash-table t0s
    plus the small grid in ``config.large_grid_offsets_ticks``.
    """
    t_start = time.time()
    base = np.asarray(_get(namespace, "baseImage"), dtype=np.float32).copy()
    hit_ts = np.asarray(_get(namespace, "hit_timestamps"), dtype=np.float32).copy()
    actual = np.asarray(_get(namespace, "fullLightWaveform"), dtype=np.float32)
    std = np.maximum(np.asarray(namespace.get("fullLightStd", namespace.get("fullLightStd_phase2", np.ones_like(actual))), dtype=np.float32), 1e-6)
    labels = np.asarray(_get(namespace, "labels_global"), dtype=np.int64)
    tpcs = np.asarray(_get(namespace, "hitTPCid"), dtype=np.int32)
    energy = np.asarray(_get(namespace, "Eset"), dtype=np.float64)
    image_maps = _get(namespace, "imageMaps")
    label_info = namespace.get("label_info", {})
    cluster_labels = [int(v) for v in namespace.get("cluster_labels", [])]
    track_shower = set(int(v) for v in namespace.get("track_shower_labels", []))
    t0_candidates = _get(namespace, "t0Candidates")
    flash_cluster_received = ensure_flash_cluster_flags(
        t0_candidates,
        namespace.get("flash_cluster_received_by_tpc"),
        max_t0=800.0,
    )
    flash_cluster_canonicalization_rows: list[dict[str, Any]] = []

    raw_flash = namespace.get("v8_1_flash_seed_t0s_by_tpc")
    if raw_flash is None:
        raw_flash = namespace.get("raw_v8_1_flash_seed_t0s_by_tpc")
    flash_by_tpc = _as_tpc_dict(raw_flash if raw_flash is not None else t0_candidates)

    cfg25 = _make_phase25_config(config, enable_light=False, enable_spatial=False)
    rows: list[dict[str, Any]] = []

    candidates = []
    for cid in cluster_labels:
        if cid in track_shower:
            continue
        info_type = str(label_info.get(int(cid), {}).get("type", "cluster")).lower()
        if "track" in info_type or "shower" in info_type:
            continue
        cmask = labels == int(cid)
        e = float(np.sum(energy[cmask]))
        if e < float(config.large_cluster_min_energy_mev):
            continue
        candidates.append((cid, e))

    candidates.sort(key=lambda x: (-x[1], x[0]))
    if config.large_max_clusters is not None:
        candidates = candidates[: int(config.large_max_clusters)]

    for cid, cluster_energy in candidates:
        cid = int(cid)
        for tpc in sorted(int(v) for v in np.unique(tpcs[labels == cid])):
            key = (cid, tpc)
            if key not in image_maps:
                continue
            hit_idx = np.flatnonzero((labels == cid) & (tpcs == tpc)).astype(np.int64)
            if hit_idx.size == 0:
                continue
            current_vals = hit_ts[hit_idx]
            finite = np.isfinite(current_vals) & (current_vals >= 0)
            if not np.any(finite):
                continue
            current_t0 = float(np.median(current_vals[finite]))
            flash_t0s = _candidate_values_for_tpc(t0_candidates, int(tpc))
            if not flash_t0s:
                flash_t0s = [float(v) for v in flash_by_tpc.get(int(tpc), []) if np.isfinite(float(v))]
            if not flash_t0s:
                continue

            cluster_img = np.asarray(image_maps[key], dtype=np.float32)
            nt = int(base.shape[-1])
            old_shifted = _shift_image_fractional(cluster_img, current_t0, nt=nt)
            base_without = np.clip(base[tpc] - old_shifted, 0.0, None)
            before_loss = _weighted_loss_window(base[tpc], actual[tpc], std[tpc], [current_t0], cfg25)

            trial_rows = []
            for flash_t0 in flash_t0s:
                for offset in config.large_grid_offsets_ticks:
                    cand_t0 = float(flash_t0) + float(offset)
                    new_shifted = _shift_image_fractional(cluster_img, cand_t0, nt=nt)
                    trial_model = np.clip(base_without + new_shifted, 0.0, float(cfg25.adc_clip))
                    loss = _weighted_loss_window(trial_model, actual[tpc], std[tpc], [current_t0, cand_t0], cfg25)
                    trial_rows.append((float(loss), float(cand_t0), float(flash_t0), float(offset), trial_model))
            if not trial_rows:
                continue
            trial_rows.sort(key=lambda r: (r[0], abs(r[1] - current_t0)))
            best_loss, best_t0, best_flash, best_offset, best_model = trial_rows[0]
            improvement = float(before_loss - best_loss)
            accepted = improvement >= float(config.large_min_loss_improvement) and abs(best_t0 - current_t0) > 1e-3
            row = {
                "clusterid": int(cid),
                "TPCid": int(tpc),
                "n_hits": int(hit_idx.size),
                "energy_mev": float(np.sum(energy[hit_idx])),
                "old_t0": float(current_t0),
                "new_t0": float(best_t0),
                "flash_t0": float(best_flash),
                "offset_ticks": float(best_offset),
                "before_loss": float(before_loss),
                "after_loss": float(best_loss),
                "loss_improvement": float(improvement),
                "accepted": bool(accepted),
            }
            rows.append(row)
            if accepted:
                base[tpc] = np.asarray(best_model, dtype=np.float32)
                hit_ts[hit_idx] = np.float32(best_t0)
                t0_candidates, flash_cluster_received, flash_row = mark_flash_cluster_assignment(
                    t0_candidates,
                    flash_cluster_received,
                    tpc=int(tpc),
                    t0=float(best_t0),
                    resolution_ticks=float(config.light_flash_cluster_match_ticks),
                    max_t0=800.0,
                    prefer_existing_true=False,
                    clusterid=int(cid),
                    stage="trial2_large_flash_grid",
                )
                flash_cluster_canonicalization_rows.append(flash_row)
                flash_by_tpc[int(tpc)] = list(_candidate_values_for_tpc(t0_candidates, int(tpc)))

    if commit:
        namespace["baseImage"] = base.astype(np.float32)
        namespace["hit_timestamps"] = hit_ts.astype(np.float32)
        namespace["t0Candidates"] = t0_candidates
        namespace["flash_cluster_received_by_tpc"] = flash_cluster_received

    return {
        "baseImage": base.astype(np.float32),
        "hit_timestamps": hit_ts.astype(np.float32),
        "t0Candidates": copy.deepcopy(t0_candidates),
        "flash_cluster_received_by_tpc": [[bool(v) for v in row] for row in flash_cluster_received],
        "flash_cluster_canonicalization_rows": [dict(row) for row in flash_cluster_canonicalization_rows],
        "rows": rows,
        "accepted_rows": [r for r in rows if r.get("accepted")],
        "elapsed_s": float(time.time() - t_start),
    }


def _make_phase25_config(
    config: Trial2Config,
    *,
    enable_spatial: bool | None = None,
    enable_light: bool | None = None,
) -> p25.Phase25Config:
    spatial_enabled = bool(config.enable_spatial if enable_spatial is None else enable_spatial)
    light_enabled = bool(config.enable_light if enable_light is None else enable_light)
    t0_match_ticks = int(config.spatial_t0_match_ticks if spatial_enabled and not light_enabled else config.light_t0_match_ticks)

    return p25.Phase25Config(
        verbose=bool(config.verbose),
        skip_shower_tpcs=bool(config.skip_shower_tpcs),
        max_move_energy_per_tpc_mev=config.max_move_energy_per_tpc_mev,
        t0_match_ticks=int(t0_match_ticks),
        t0_merge_ticks=0 if bool(config.light_source_exact_associated_t0s) else 5,
        prediction_batch_size=int(config.prediction_batch_size),
        device_policy=str(config.device_policy),
        enable_spatial=bool(spatial_enabled),
        spatial_contact_radius_cm=float(config.spatial_contact_radius_cm),
        spatial_different_t0_ticks=float(config.spatial_different_t0_ticks),
        spatial_max_pairs=int(config.spatial_max_pairs),
        spatial_min_pair_hits=int(config.spatial_min_pair_hits),
        spatial_min_pair_energy_mev=float(config.spatial_min_pair_energy_mev),
        spatial_component_radius_cm=float(config.spatial_component_radius_cm),
        spatial_smooth_component_radius_cm=float(config.spatial_smooth_component_radius_cm),
        spatial_min_model_hits=int(config.spatial_min_model_hits),
        spatial_min_model_energy_mev=float(config.spatial_min_model_energy_mev),
        spatial_max_models_per_t0=int(config.spatial_max_models_per_t0),
        spatial_trim_model_quantile=float(config.spatial_trim_model_quantile),
        spatial_trim_iterations=int(config.spatial_trim_iterations),
        spatial_axis_width_floor_cm=float(config.spatial_axis_width_floor_cm),
        spatial_nearest_scale_cm=float(config.spatial_nearest_scale_cm),
        spatial_endpoint_gap_scale_cm=float(config.spatial_endpoint_gap_scale_cm),
        spatial_endpoint_margin_cm=float(config.spatial_endpoint_margin_cm),
        spatial_max_accept_score=float(config.spatial_max_accept_score),
        spatial_rescan_pool_margin=float(config.spatial_rescan_pool_margin),
        spatial_component_strong_margin=float(config.spatial_component_strong_margin),
        spatial_move_margin=float(config.spatial_move_margin),
        spatial_keep_inertia=float(config.spatial_keep_inertia),
        spatial_min_moved_hits=int(config.spatial_min_moved_hits),
        spatial_max_loss_increase=float(config.spatial_max_loss_increase),
        enable_light=bool(light_enabled),
        light_overflow_sigma=float(config.light_overflow_sigma),
        light_overflow_abs_adc=float(config.light_overflow_abs_adc),
        light_model_activity_adc=float(config.light_model_activity_adc),
        light_min_overflow_channels=int(config.light_min_overflow_channels),
        light_max_total_moves=int(config.light_max_total_moves),
        light_max_moves_per_tpc=int(config.light_max_moves_per_tpc),
    )


def _associated_source_t0s_by_tpc(
    *,
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray | None = None,
    energy: np.ndarray | None = None,
    t0_candidates: Any | None = None,
    allowed_tpcs: Iterable[int],
    match_ticks: float = 5.0,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    out: dict[int, list[int]] = {}
    hit_ts = np.asarray(hit_timestamps, dtype=np.float64)
    tpcs = np.asarray(hit_tpc_ids, dtype=np.int32)
    labels = None if labels_global is None else np.asarray(labels_global, dtype=np.int64)
    e = np.ones_like(hit_ts, dtype=np.float64) if energy is None else np.asarray(energy, dtype=np.float64)
    finite = np.isfinite(hit_ts) & (hit_ts >= 0)
    if labels is not None:
        finite &= labels >= 0
    rows: list[dict[str, Any]] = []

    for tpc in sorted(int(v) for v in allowed_tpcs):
        if t0_candidates is None:
            vals = np.round(hit_ts[(tpcs == int(tpc)) & finite]).astype(np.int32)
            vals = vals[vals != 0]
            if vals.size:
                out[int(tpc)] = sorted(int(v) for v in np.unique(vals))
            continue

        source_vals = []
        for flash_t0 in _candidate_values_for_tpc(t0_candidates, int(tpc)):
            idx = np.flatnonzero(
                (tpcs == int(tpc))
                & finite
                & (np.abs(hit_ts - float(flash_t0)) <= float(match_ticks))
            ).astype(np.int64)
            associated = idx.size > 0
            rows.append(
                {
                    "TPCid": int(tpc),
                    "t0": float(flash_t0),
                    "source_t0": int(round(float(flash_t0))),
                    "associated": bool(associated),
                    "n_hits": int(idx.size),
                    "energy_mev": float(np.sum(e[idx])) if idx.size else 0.0,
                    "labels": _label_summary(idx, labels if labels is not None else np.full(hit_ts.shape, -1)),
                }
            )
            if associated:
                source_t0 = int(round(float(flash_t0)))
                if source_t0 != 0:
                    source_vals.append(source_t0)
        if source_vals:
            out[int(tpc)] = sorted(int(v) for v in np.unique(np.asarray(source_vals, dtype=np.int32)))
    return out, rows


def _light_source_detail_for_t0(
    *,
    hit_timestamps: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    energy: np.ndarray,
    tpc: int,
    t0: int,
    match_ticks: float,
) -> dict[str, Any]:
    hit_ts = np.asarray(hit_timestamps, dtype=np.float64)
    tpcs = np.asarray(hit_tpc_ids, dtype=np.int32)
    labels = np.asarray(labels_global, dtype=np.int64)
    e = np.asarray(energy, dtype=np.float64)
    finite = np.isfinite(hit_ts) & (hit_ts >= 0) & (labels >= 0)
    idx = np.flatnonzero(
        finite
        & (tpcs == int(tpc))
        & (np.abs(hit_ts - float(t0)) <= float(match_ticks))
    ).astype(np.int64)
    return {
        "charge_hits": int(idx.size),
        "charge_energy_mev": float(np.sum(e[idx])) if idx.size else 0.0,
        "labels": _label_summary(idx, labels),
    }


def _scan_light_overflows_source_only(
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    source_t0s_by_tpc: dict[int, list[int]],
    saturated_channel_cache: Any | None,
    allowed_tpcs: set[int],
    cfg: p25.Phase25Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual = np.asarray(full_light_waveform, dtype=np.float32)
    model = np.asarray(base_image, dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std, dtype=np.float32), 1e-6)

    for tpc in sorted(allowed_tpcs):
        source_t0s = source_t0s_by_tpc.get(int(tpc), [])
        if not source_t0s:
            continue
        keep = p25._keep_channel_indices(int(tpc), actual, saturated_channel_cache)
        if keep.size == 0:
            continue
        a = actual[int(tpc), keep]
        m = model[int(tpc), keep]
        s = std[int(tpc), keep]
        for t0 in sorted(set(int(v) for v in source_t0s)):
            tick = int(t0) + int(cfg.pulse_peak_tick)
            if tick < 0 or tick >= a.shape[1]:
                continue
            lo = max(0, tick - int(cfg.half_window_ticks))
            hi = min(a.shape[1], tick + int(cfg.half_window_ticks) + 1)
            threshold = np.maximum(float(cfg.light_overflow_sigma) * s[:, lo:hi], float(cfg.light_overflow_abs_adc))
            residual = m[:, lo:hi] - a[:, lo:hi]
            overflow = (residual > threshold) & (m[:, lo:hi] > float(cfg.light_model_activity_adc))
            n_of_ch = int(np.count_nonzero(np.any(overflow, axis=1)))
            if n_of_ch < int(cfg.light_min_overflow_channels):
                continue
            peak_threshold = np.maximum(float(cfg.light_overflow_sigma) * s[:, tick], float(cfg.light_overflow_abs_adc))
            peak_residual = m[:, tick] - a[:, tick]
            peak_overflow = (peak_residual > peak_threshold) & (m[:, tick] > float(cfg.light_model_activity_adc))
            deficit = a[:, lo:hi] - m[:, lo:hi]
            peak_deficit = a[:, tick] - m[:, tick]
            win_of = float(np.sum(np.clip(residual[overflow], 0, None)))
            pk_of = float(np.sum(np.clip(peak_residual[peak_overflow], 0, None)))
            rows.append(
                {
                    "TPCid": int(tpc),
                    "t0": int(t0),
                    "peak_tick": int(tick),
                    "overflow_channels": int(n_of_ch),
                    "peak_overflow_channels": int(np.count_nonzero(peak_overflow)),
                    "deficit_channels": int(np.count_nonzero(np.any(deficit > threshold, axis=1))),
                    "window_overflow": win_of,
                    "window_deficit": float(np.sum(np.clip(deficit, 0, None))),
                    "peak_overflow": pk_of,
                    "peak_deficit": float(np.sum(np.clip(peak_deficit, 0, None))),
                    "severity": float(pk_of / max(n_of_ch, 1) + 0.002 * win_of),
                }
            )
    rows.sort(
        key=lambda r: (
            -float(r["severity"]),
            -float(r["window_overflow"]),
            -int(r["overflow_channels"]),
            int(r["TPCid"]),
            int(r["t0"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["source_rank"] = int(rank)
    return rows[: int(cfg.light_overflow_rows_per_pass)]


def _overflow_summary(
    *,
    t0: int,
    model_tpc: np.ndarray,
    actual_tpc: np.ndarray,
    std_tpc: np.ndarray,
    keep: np.ndarray,
    cfg: p25.Phase25Config,
) -> dict[str, Any]:
    tick = int(t0) + int(cfg.pulse_peak_tick)
    if tick < 0 or tick >= actual_tpc.shape[1]:
        return {"ofch": 0, "overflow_sum": 0.0, "deficit_sum": 0.0}
    lo = max(0, tick - int(cfg.half_window_ticks))
    hi = min(actual_tpc.shape[1], tick + int(cfg.half_window_ticks) + 1)
    actual = np.asarray(actual_tpc[keep, lo:hi], dtype=np.float32)
    model = np.asarray(model_tpc[keep, lo:hi], dtype=np.float32)
    std = np.maximum(np.asarray(std_tpc[keep, lo:hi], dtype=np.float32), 1e-6)
    diff = model - actual
    threshold = np.maximum(float(cfg.light_overflow_sigma) * std, float(cfg.light_overflow_abs_adc))
    overflow = (diff > threshold) & (model > float(cfg.light_model_activity_adc))
    return {
        "ofch": int(np.count_nonzero(np.any(overflow, axis=1))),
        "overflow_sum": float(np.sum(np.where(overflow, np.clip(diff, 0.0, None), 0.0))),
        "deficit_sum": float(np.sum(np.clip(-diff, 0.0, None))),
    }


def _physical_chi2_window_loss(
    *,
    model_tpc: np.ndarray,
    actual_tpc: np.ndarray,
    std_tpc: np.ndarray,
    keep: np.ndarray,
    t0_values: Iterable[int | float],
    cfg: p25.Phase25Config,
    std_floor: float,
) -> tuple[float, int]:
    n_ticks = int(actual_tpc.shape[1])
    tick_mask = np.zeros(n_ticks, dtype=bool)

    for t0 in t0_values:
        tick = int(round(float(t0))) + int(cfg.pulse_peak_tick)
        if tick < 0 or tick >= n_ticks:
            continue
        lo = max(0, tick - int(cfg.half_window_ticks))
        hi = min(n_ticks, tick + int(cfg.half_window_ticks) + 1)
        if hi > lo:
            tick_mask[lo:hi] = True

    if not np.any(tick_mask) or len(keep) == 0:
        return 0.0, 0

    pred = np.asarray(model_tpc[keep][:, tick_mask], dtype=np.float64)
    actual = np.asarray(actual_tpc[keep][:, tick_mask], dtype=np.float64)
    std = np.maximum(np.asarray(std_tpc[keep][:, tick_mask], dtype=np.float64), float(std_floor))
    chi2 = float(np.sum(((pred - actual) / std) ** 2))
    return chi2, int(pred.size)


def _label_info_text_local(label_info: Any | None, label: int) -> str:
    if label_info is None:
        return ""
    info = None
    try:
        if isinstance(label_info, dict):
            info = label_info.get(int(label), label_info.get(str(int(label)), None))
        else:
            info = label_info[int(label)]
    except Exception:
        info = None
    if info is None:
        return ""
    if isinstance(info, dict):
        return " ".join(str(v) for v in info.values()).lower()
    return str(info).lower()


def _light_multitpc_track_veto(
    trial: dict[str, Any],
    *,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    energy: np.ndarray,
    label_info: Any | None,
    min_tpcs: int,
) -> dict[str, Any]:
    idx = np.asarray(trial.get("hit_indices", []), dtype=np.int64)
    labels = np.asarray(labels_global, dtype=np.int64)
    tpcs = np.asarray(hit_tpc_ids, dtype=np.int32)
    e = np.asarray(energy, dtype=np.float64)

    if idx.size == 0:
        return {
            "track_veto": False,
            "track_veto_label": -1,
            "track_veto_label_energy_mev": 0.0,
            "track_veto_label_tpcs": [],
            "track_veto_n_tpcs": 0,
            "track_veto_label_info": "",
            "track_veto_track_like": False,
        }

    good = (idx >= 0) & (idx < labels.shape[0])
    idx = idx[good]
    if idx.size == 0:
        return {
            "track_veto": False,
            "track_veto_label": -1,
            "track_veto_label_energy_mev": 0.0,
            "track_veto_label_tpcs": [],
            "track_veto_n_tpcs": 0,
            "track_veto_label_info": "",
            "track_veto_track_like": False,
        }

    moved_labels = labels[idx]
    valid = moved_labels >= 0
    if not np.any(valid):
        return {
            "track_veto": False,
            "track_veto_label": -1,
            "track_veto_label_energy_mev": 0.0,
            "track_veto_label_tpcs": [],
            "track_veto_n_tpcs": 0,
            "track_veto_label_info": "",
            "track_veto_track_like": False,
        }

    idx_valid = idx[valid]
    moved_labels = moved_labels[valid]
    vals = np.unique(moved_labels)
    label_energy = []
    label_hits = []
    for label in vals:
        m = moved_labels == int(label)
        hit_idx = idx_valid[m]
        label_energy.append(float(np.sum(e[hit_idx][np.isfinite(e[hit_idx])])))
        label_hits.append(int(hit_idx.size))
    order = np.lexsort((-np.asarray(label_hits), -np.asarray(label_energy)))
    majority_label = int(vals[int(order[0])])
    majority_energy = float(label_energy[int(order[0])])

    full_idx = np.flatnonzero(labels == majority_label).astype(np.int64)
    full_tpcs = tpcs[full_idx]
    label_tpcs = sorted(int(v) for v in np.unique(full_tpcs[full_tpcs >= 0]))
    info_text = _label_info_text_local(label_info, majority_label)
    metadata_shower_like = "shower" in info_text
    # Tracks and non-shower multi-TPC labels are risky for light-only moves.
    track_like = not bool(metadata_shower_like)
    veto = bool(track_like and len(label_tpcs) >= int(min_tpcs))

    return {
        "track_veto": bool(veto),
        "track_veto_label": int(majority_label),
        "track_veto_label_energy_mev": float(majority_energy),
        "track_veto_label_tpcs": label_tpcs,
        "track_veto_n_tpcs": int(len(label_tpcs)),
        "track_veto_label_info": str(info_text),
        "track_veto_track_like": bool(track_like),
    }


def _evaluate_trial2_light_gate(
    trial: dict[str, Any],
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    saturated_channel_cache: Any | None,
    config: Trial2Config,
    cfg25: p25.Phase25Config,
) -> dict[str, Any]:
    tpc = int(trial["TPCid"])
    old_t0 = int(trial["old_t0"])
    new_t0 = int(trial["new_t0"])
    keep = p25._keep_channel_indices(tpc, np.asarray(full_light_waveform, dtype=np.float32), saturated_channel_cache)
    actual = np.asarray(full_light_waveform[tpc], dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std[tpc], dtype=np.float32), 1e-6)
    before_model = np.asarray(base_image[tpc], dtype=np.float32)
    delta = np.asarray(trial.get("delta", np.zeros_like(before_model)), dtype=np.float32)
    after_model = np.clip(before_model + delta, 0.0, float(cfg25.adc_clip))

    before = _overflow_summary(t0=old_t0, model_tpc=before_model, actual_tpc=actual, std_tpc=std, keep=keep, cfg=cfg25)
    after = _overflow_summary(t0=old_t0, model_tpc=after_model, actual_tpc=actual, std_tpc=std, keep=keep, cfg=cfg25)
    dest_before = _overflow_summary(t0=new_t0, model_tpc=before_model, actual_tpc=actual, std_tpc=std, keep=keep, cfg=cfg25)
    dest_after = _overflow_summary(t0=new_t0, model_tpc=after_model, actual_tpc=actual, std_tpc=std, keep=keep, cfg=cfg25)

    src_red = int(before["ofch"] - after["ofch"])
    dst_new = int(max(dest_after["ofch"] - dest_before["ofch"], 0))
    src_def_inc = float(after["deficit_sum"] - before["deficit_sum"])
    comp_e = float(trial.get("energy_mev", 0.0))
    overflow_row = trial.get("overflow_row", {})
    q_energy = float(overflow_row.get("charge_energy_mev", comp_e))
    move_frac = comp_e / max(q_energy, comp_e, 1.0e-6)
    src_remain_frac = float(after["ofch"]) / max(float(before["ofch"]), 1.0)
    loss_improvement = float(trial.get("loss_improvement", -np.inf))
    loss_per_mev = loss_improvement / max(comp_e, 1.0e-6)
    old_ok = bool(trial.get("accept_like", trial.get("accepted", False)))

    if bool(config.light_use_physical_chi2):
        chi2_before, dof = _physical_chi2_window_loss(
            model_tpc=before_model,
            actual_tpc=actual,
            std_tpc=std,
            keep=keep,
            t0_values=[old_t0, new_t0],
            cfg=cfg25,
            std_floor=float(config.phys_std_floor),
        )
        chi2_after, _ = _physical_chi2_window_loss(
            model_tpc=after_model,
            actual_tpc=actual,
            std_tpc=std,
            keep=keep,
            t0_values=[old_t0, new_t0],
            cfg=cfg25,
            std_floor=float(config.phys_std_floor),
        )
        dchi2 = float(chi2_before - chi2_after)
        dchi2_per_mev = dchi2 / max(comp_e, 1.0e-6)
        dchi2_per_dof = dchi2 / max(float(dof), 1.0)

        src_ch_ok = src_red >= int(config.phys_min_source_ofch_reduction)
        dchi2_ok = dchi2 >= float(config.phys_min_dchi2_improvement)
        dchi2_per_e_ok = dchi2_per_mev >= float(config.phys_min_dchi2_per_mev)
        accepted = bool(src_ch_ok and dchi2_ok and dchi2_per_e_ok)

        return {
            "old_ok": bool(old_ok),
            "channel_ok": bool(src_ch_ok),
            "dst_ch_ok": True,
            "src_def_ok": True,
            "dchi2_ok": bool(dchi2_ok),
            "dchi2_per_e_ok": bool(dchi2_per_e_ok),
            "accepted": bool(accepted),
            "accepted_by": "physicalChi2" if accepted else "reject",
            "src_of0": int(before["ofch"]),
            "src_of1": int(after["ofch"]),
            "src_red": int(src_red),
            "dst_new": int(dst_new),
            "src_remain_frac": float(src_remain_frac),
            "move_frac": float(move_frac),
            "loss_per_mev": float(loss_per_mev),
            "src_def_inc": float(src_def_inc),
            "src_def_limit": np.nan,
            "q_energy_mev": float(q_energy),
            "chi2_before": float(chi2_before),
            "chi2_after": float(chi2_after),
            "dchi2": float(dchi2),
            "dchi2_per_mev": float(dchi2_per_mev),
            "dchi2_per_dof": float(dchi2_per_dof),
            "dof": int(dof),
            "old_loss_improvement": float(loss_improvement),
        }

    src_def_limit = (
        float(config.light_max_source_deficit_increase)
        + float(config.light_max_source_deficit_increase_per_mev) * max(comp_e, 0.0)
    )
    base_dst_limit = max(
        float(config.light_max_dest_new_channels_abs),
        float(config.light_max_dest_new_channels_frac_of_src_red) * max(float(src_red), 0.0),
    )
    channel_ok = src_red > int(config.light_source_channel_drop_strict_gt)
    dst_ch_ok = float(dst_new) <= float(base_dst_limit)
    src_def_ok = src_def_inc <= src_def_limit
    loss_per_mev_ok = loss_per_mev >= float(config.light_min_dloss_per_mev)
    partial_source_ok = not (
        move_frac < float(config.light_partial_move_frac_max)
        and src_remain_frac > float(config.light_partial_source_remain_frac_max)
    )
    if not bool(config.light_use_rescue_branches):
        accepted = bool(old_ok and channel_ok and src_def_ok)
        accepted_by = "sourceGate" if accepted else "reject"
        base_flags = [("oldOK", old_ok), ("srcCh", channel_ok), ("srcDef", src_def_ok)]
        high_e_dst_limit = np.nan
        src_clear_dst_limit = np.nan
        high_e_oldfail_ok = False
        source_clear_ok = False
        oldok_partial_ok = False
    else:
        base_flags = [
            ("srcCh", channel_ok),
            ("dstCh", dst_ch_ok),
            ("srcDef", src_def_ok),
            ("dL/E", loss_per_mev_ok),
            ("partialSrc", partial_source_ok),
        ]
        base_ok = bool(all(ok for _, ok in base_flags))

        if src_remain_frac <= float(config.light_rescue_high_e_clear_src_remain):
            high_e_dst_limit = float(config.light_rescue_high_e_clear_dst_frac) * max(float(src_red), 0.0)
        else:
            high_e_dst_limit = max(
                float(config.light_rescue_high_e_dst_abs),
                float(config.light_rescue_high_e_dst_frac) * max(float(src_red), 0.0),
            )

        high_e_oldfail_ok = (
            (not old_ok)
            and bool(src_def_ok)
            and comp_e >= float(config.light_rescue_high_e_min_e)
            and src_red >= int(config.light_rescue_high_e_src_red_ge)
            and src_remain_frac <= float(config.light_rescue_high_e_max_src_remain)
            and move_frac >= float(config.light_rescue_high_e_min_move_frac)
            and loss_per_mev >= float(config.light_rescue_high_e_min_dloss_per_mev)
            and float(dst_new) <= float(high_e_dst_limit)
        )

        src_clear_dst_limit = max(
            float(config.light_rescue_src_clear_dst_abs),
            float(config.light_rescue_src_clear_dst_frac) * max(float(src_red), 0.0),
        )
        source_clear_ok = (
            bool(src_def_ok)
            and comp_e >= float(config.light_rescue_src_clear_min_e)
            and src_red >= int(config.light_rescue_src_clear_src_red_ge)
            and src_remain_frac <= float(config.light_rescue_src_clear_max_src_remain)
            and float(dst_new) <= float(src_clear_dst_limit)
        )

        oldok_partial_ok = (
            old_ok
            and bool(src_def_ok)
            and comp_e >= float(config.light_rescue_oldok_partial_min_e)
            and src_red >= int(config.light_rescue_oldok_partial_src_red_ge)
            and dst_new <= int(config.light_rescue_oldok_partial_dst_max)
            and src_remain_frac <= float(config.light_rescue_oldok_partial_max_src_remain)
            and move_frac >= float(config.light_rescue_oldok_partial_min_move_frac)
            and loss_per_mev >= float(config.light_rescue_oldok_partial_min_dloss_per_mev)
        )

        if base_ok:
            accepted_by = "base"
        elif high_e_oldfail_ok:
            accepted_by = "highE"
        elif source_clear_ok:
            accepted_by = "srcClear"
        elif oldok_partial_ok:
            accepted_by = "oldOKpartial"
        else:
            accepted_by = "reject"
        accepted = bool(accepted_by != "reject")

    return {
        "old_ok": bool(old_ok),
        "channel_ok": bool(channel_ok),
        "dst_ch_ok": bool(dst_ch_ok),
        "src_def_ok": bool(src_def_ok),
        "loss_per_mev_ok": bool(loss_per_mev_ok),
        "partial_source_ok": bool(partial_source_ok),
        "accepted": bool(accepted),
        "accepted_by": str(accepted_by),
        "base_flags": list(base_flags),
        "src_of0": int(before["ofch"]),
        "src_of1": int(after["ofch"]),
        "src_red": int(src_red),
        "dst_new": int(dst_new),
        "base_dst_limit": float(base_dst_limit),
        "high_e_dst_limit": float(high_e_dst_limit),
        "src_clear_dst_limit": float(src_clear_dst_limit),
        "src_remain_frac": float(src_remain_frac),
        "move_frac": float(move_frac),
        "loss_per_mev": float(loss_per_mev),
        "src_def_inc": float(src_def_inc),
        "src_def_limit": float(src_def_limit),
        "q_energy_mev": float(q_energy),
        "high_e_oldfail_ok": bool(high_e_oldfail_ok),
        "source_clear_ok": bool(source_clear_ok),
        "oldok_partial_ok": bool(oldok_partial_ok),
    }


def _apply_light_track_veto_if_needed(
    trial: dict[str, Any],
    accepted: bool,
    *,
    cfg: Trial2Config,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    energy: np.ndarray,
    label_info: Any | None,
    override_track_labels: set[int] | None = None,
    override_group_sizes: dict[int, int] | None = None,
) -> bool:
    if not bool(accepted):
        trial["track_veto_ok"] = True
        trial["track_veto_overridden"] = False
        return False
    if not bool(cfg.light_veto_multitpc_track):
        trial["track_veto_ok"] = True
        trial["track_veto_overridden"] = False
        return True

    veto = _light_multitpc_track_veto(
        trial,
        labels_global=labels_global,
        hit_tpc_ids=hit_tpc_ids,
        energy=energy,
        label_info=label_info,
        min_tpcs=int(cfg.light_veto_track_min_tpcs),
    )
    trial.update(veto)
    if bool(veto.get("track_veto", False)):
        veto_label = int(veto.get("track_veto_label", -1))
        if override_track_labels is not None and veto_label in override_track_labels:
            trial["accepted_by_before_track_veto"] = str(trial.get("accepted_by", ""))
            trial["track_veto_ok"] = True
            trial["track_veto_overridden"] = True
            trial["track_veto_override_group_size"] = int((override_group_sizes or {}).get(veto_label, 0))
            return True
        trial["accepted_by_before_track_veto"] = str(trial.get("accepted_by", ""))
        trial["accepted_by"] = "reject"
        trial["track_veto_ok"] = False
        trial["track_veto_overridden"] = False
        return False

    trial["track_veto_ok"] = True
    trial["track_veto_overridden"] = False
    return True


def _light_track_veto_override_groups(
    trials: list[dict[str, Any]],
    *,
    min_candidates: int,
) -> tuple[set[int], dict[int, int]]:
    """Find long-track labels with enough otherwise accepted light proposals."""
    counts: dict[int, int] = {}
    for trial in trials:
        if not bool(trial.get("track_veto", False)):
            continue
        if bool(trial.get("track_veto_ok", True)):
            continue
        before_veto = str(trial.get("accepted_by_before_track_veto", ""))
        if not before_veto or before_veto == "reject":
            continue
        label = int(trial.get("track_veto_label", -1))
        if label < 0:
            continue
        counts[label] = counts.get(label, 0) + 1

    threshold = max(int(min_candidates), 1)
    override_labels = {int(label) for label, count in counts.items() if int(count) >= threshold}
    return override_labels, counts


def run_trial2_phase25_from_namespace(
    namespace: dict[str, Any],
    *,
    config: Trial2Config | None = None,
    commit: bool | None = None,
) -> dict[str, Any]:
    """Run trial2 large-grid correction, spatial repair, then configured light repair."""
    cfg = config or Trial2Config()
    do_commit = cfg.commit if commit is None else bool(commit)
    t_start = time.time()

    before_ts = np.asarray(_get(namespace, "hit_timestamps"), dtype=np.float32).copy()
    before_base = np.asarray(_get(namespace, "baseImage"), dtype=np.float32).copy()
    before_t0_candidates = copy.deepcopy(_get(namespace, "t0Candidates"))
    before_flash_cluster_received = copy.deepcopy(namespace.get("flash_cluster_received_by_tpc", None))

    large_result = {"rows": [], "accepted_rows": [], "elapsed_s": 0.0}
    if cfg.enable_large_flash_grid_correction:
        large_result = run_large_cluster_flash_grid_correction_from_namespace(namespace, config=cfg, commit=True)

    cfg25_spatial = _make_phase25_config(cfg, enable_spatial=cfg.enable_spatial, enable_light=False)
    spatial_result = p25.run_phase25_amendment_from_namespace(namespace, config=cfg25_spatial, commit=True)

    base_out = np.asarray(spatial_result["baseImage"], dtype=np.float32).copy()
    hit_ts_out = np.asarray(spatial_result["hit_timestamps"], dtype=np.float32).copy()
    labels = np.asarray(_get(namespace, "labels_global"), dtype=np.int64)
    label_info = namespace.get("label_info", None)
    tpcs = np.asarray(_get(namespace, "hitTPCid"), dtype=np.int32)
    x = np.asarray(_get(namespace, "xset"), dtype=np.float64)
    y = np.asarray(_get(namespace, "yset"), dtype=np.float64)
    z = np.asarray(_get(namespace, "zset"), dtype=np.float64)
    energy = np.asarray(_get(namespace, "Eset"), dtype=np.float64)
    full_light = np.asarray(_get(namespace, "fullLightWaveform"), dtype=np.float32)
    full_std = np.asarray(namespace.get("fullLightStd", namespace.get("fullLightStd_phase2", np.ones_like(full_light))), dtype=np.float32)
    t0_candidates = _get(namespace, "t0Candidates")
    cfg25_light = _make_phase25_config(cfg, enable_spatial=False, enable_light=True)
    if bool(cfg.light_use_physical_chi2):
        # Match the physical-loss audit cell: make p25 permissive and let the
        # dChi2 gate below make the final decision.
        cfg25_light.max_move_energy_per_tpc_mev = None
        cfg25_light.light_min_loss_improvement = -1.0e30
        cfg25_light.light_min_old_overflow_reduction = -1.0e30
        cfg25_light.light_min_dest_deficit_reduction = -1.0e30
        cfg25_light.light_max_dest_new_overflow_frac = 1.0e30
        cfg25_light.light_overflow_rows_per_pass = 1_000_000_000
    all_hit_tpcs = sorted(set(int(v) for v in np.unique(tpcs)))
    spatial_allowed_tpcs = sorted(set(int(v) for v in spatial_result.get("allowed_tpcs", all_hit_tpcs)))
    light_allowed_tpcs = spatial_allowed_tpcs if bool(cfg.light_skip_shower_tpcs) else all_hit_tpcs

    flash_cluster_received = namespace.get("flash_cluster_received_by_tpc", None)
    if flash_cluster_received is not None:
        flash_cluster_received = ensure_flash_cluster_flags(t0_candidates, flash_cluster_received, max_t0=800.0)
        flash_canonicalization_rows = list(large_result.get("flash_cluster_canonicalization_rows", []))
    else:
        # Fallback for older notebooks that do not yet carry the explicit
        # received-cluster flags. New trial2 notebooks should use the flagged path.
        t0_candidates, flash_canonicalization_rows = _canonicalize_flash_table_from_assignments(
            t0_candidates,
            hit_timestamps=hit_ts_out,
            hit_tpc_ids=tpcs,
            labels_global=labels,
            energy=energy,
            allowed_tpcs=light_allowed_tpcs,
            match_ticks=float(cfg.light_flash_cluster_match_ticks),
        )
    namespace["t0Candidates"] = t0_candidates
    if flash_cluster_received is not None:
        namespace["flash_cluster_received_by_tpc"] = flash_cluster_received
    t0_candidates_dict = _as_tpc_dict(t0_candidates)

    light_moves: list[dict[str, Any]] = []
    light_trials: list[dict[str, Any]] = []
    light_source_rows: list[dict[str, Any]] = []
    family_update_records: list[dict[str, Any]] = list(spatial_result.get("family_update_records", []))
    if flash_cluster_received is not None:
        source_t0s_by_tpc, flash_cluster_association_rows = source_t0s_from_received_flash_table(
            t0_candidates,
            flash_cluster_received,
            allowed_tpcs=light_allowed_tpcs,
            min_sep_ticks=float(cfg.light_flash_cluster_match_ticks),
        )
    else:
        source_t0s_by_tpc, flash_cluster_association_rows = _associated_source_t0s_by_tpc(
            hit_timestamps=hit_ts_out,
            hit_tpc_ids=tpcs,
            labels_global=labels,
            energy=energy,
            t0_candidates=t0_candidates_dict,
            allowed_tpcs=light_allowed_tpcs,
            match_ticks=float(cfg.light_flash_cluster_match_ticks),
        )

    moves_by_tpc: dict[int, int] = {}
    light_track_veto_override_labels: set[int] = set()
    light_track_veto_override_counts: dict[int, int] = {}
    if cfg.enable_light:
        if bool(cfg.light_use_physical_chi2) or bool(cfg.light_use_rescue_branches):
            track_veto_candidates: list[dict[str, Any]] = []
            rows = _scan_light_overflows_source_only(
                base_image=base_out,
                full_light_waveform=full_light,
                full_light_std=full_std,
                source_t0s_by_tpc=source_t0s_by_tpc,
                saturated_channel_cache=namespace.get("saturated_channel_cache"),
                allowed_tpcs=set(int(v) for v in light_allowed_tpcs),
                cfg=cfg25_light,
            )
            light_source_rows = [dict(row) for row in rows]
            for row in rows:
                row.update(
                    _light_source_detail_for_t0(
                        hit_timestamps=hit_ts_out,
                        hit_tpc_ids=tpcs,
                        labels_global=labels,
                        energy=energy,
                        tpc=int(row["TPCid"]),
                        t0=int(row["t0"]),
                        match_ticks=float(cfg.light_t0_match_ticks),
                    )
                )
            for source_rank, row in enumerate(rows, start=1):
                tpc = int(row["TPCid"])
                row = dict(row)
                row["source_rank"] = int(row.get("source_rank", source_rank))
                if len(light_moves) >= int(cfg.light_max_total_moves):
                    break
                if moves_by_tpc.get(tpc, 0) >= int(cfg.light_max_moves_per_tpc):
                    continue
                trial = p25._try_light_repair_row(
                    row,
                    base_image=base_out,
                    full_light_waveform=full_light,
                    full_light_std=full_std,
                    image_maps=namespace.get("imageMaps"),
                    t0_candidates=t0_candidates_dict,
                    hit_timestamps=hit_ts_out,
                    hit_tpc_ids=tpcs,
                    labels_global=labels,
                    x=x,
                    y=y,
                    z=z,
                    energy=energy,
                    saturated_channel_cache=namespace.get("saturated_channel_cache"),
                    locked_hit_mask=None,
                    cfg=cfg25_light,
                )
                if trial is None:
                    light_trials.append(
                        {
                            "source_rank": int(row["source_rank"]),
                            "overflow_row": dict(row),
                            "accepted": False,
                            "accepted_by": "no_trial",
                            "no_trial": True,
                            "reason": "no_trial",
                            "TPCid": int(tpc),
                            "old_t0": int(row["t0"]),
                            "new_t0": -1,
                            "component_id": -1,
                            "hit_indices": np.asarray([], dtype=np.int64),
                            "n_hits": 0,
                            "energy_mev": 0.0,
                            "q_energy_mev": float(row.get("charge_energy_mev", 0.0)),
                            "src_of0": int(row.get("overflow_channels", 0)),
                            "src_of1": int(row.get("overflow_channels", 0)),
                            "src_red": 0,
                            "dst_new": 0,
                            "src_def_inc": 0.0,
                            "src_def_limit": np.nan,
                            "old_ok": False,
                            "channel_ok": False,
                            "src_def_ok": False,
                            "loss_improvement": 0.0,
                            "loss_per_mev": 0.0,
                            "dchi2": 0.0,
                            "dchi2_per_mev": 0.0,
                            "dchi2_per_dof": 0.0,
                        }
                    )
                    continue
                trial = dict(trial)
                trial["source_rank"] = int(row["source_rank"])
                trial["overflow_row"] = dict(row)
                proxy, image_status = p25._component_proxy_image(
                    component=trial,
                    image_maps=namespace.get("imageMaps"),
                    labels_global=labels,
                    hit_tpc_ids=tpcs,
                    energy=energy,
                )
                trial["image_status"] = str(image_status)
                if proxy is None:
                    trial.update(
                        {
                            "accepted": False,
                            "accepted_by": "no_trial",
                            "no_trial": True,
                            "reason": str(image_status),
                            "src_of0": int(row.get("overflow_channels", 0)),
                            "src_of1": int(row.get("overflow_channels", 0)),
                            "src_red": 0,
                            "dst_new": 0,
                        }
                    )
                    light_trials.append(trial)
                    continue
                old_shift = p25._shift_image(proxy, int(trial["old_t0"]), nt=base_out.shape[-1])
                new_shift = p25._shift_image(proxy, int(trial["new_t0"]), nt=base_out.shape[-1])
                trial["delta"] = (new_shift - old_shift).astype(np.float32)
                gate = _evaluate_trial2_light_gate(
                    trial,
                    base_image=base_out,
                    full_light_waveform=full_light,
                    full_light_std=full_std,
                    saturated_channel_cache=namespace.get("saturated_channel_cache"),
                    config=cfg,
                    cfg25=cfg25_light,
                )
                accepted = bool(gate["accepted"])
                trial.update(gate)
                accepted = _apply_light_track_veto_if_needed(
                    trial,
                    accepted,
                    cfg=cfg,
                    labels_global=labels,
                    hit_tpc_ids=tpcs,
                    energy=energy,
                    label_info=label_info,
                )
                trial["accepted"] = bool(accepted)
                light_trials.append(trial)
                if (
                    (not accepted)
                    and bool(trial.get("track_veto", False))
                    and (not bool(trial.get("track_veto_ok", True)))
                    and str(trial.get("accepted_by_before_track_veto", "")) not in {"", "reject"}
                ):
                    track_veto_candidates.append(trial)
                if not accepted:
                    continue

                before_ts_pass = hit_ts_out.copy()
                before_base_pass = base_out.copy()
                moved = np.asarray(trial["hit_indices"], dtype=np.int64)
                old_t0 = int(trial["old_t0"])
                new_t0 = int(trial["new_t0"])
                hit_ts_out[moved] = np.float32(new_t0)

                affected = {(tpc, old_t0), (tpc, new_t0)}
                base_out, records = p25._exact_update_affected_families(
                    base_image=before_base_pass,
                    old_hit_timestamps=before_ts_pass,
                    new_hit_timestamps=hit_ts_out,
                    affected_specs=affected,
                    hit_tpc_ids=tpcs,
                    x=x,
                    y=y,
                    z=z,
                    energy=energy,
                    model=_get(namespace, "model"),
                    template=_get(namespace, "wvfm_tmpl"),
                    cfg=cfg25_light,
                )
                family_update_records.extend(records)
                light_moves.append(trial)
                moves_by_tpc[tpc] = moves_by_tpc.get(tpc, 0) + 1

            light_track_veto_override_labels, light_track_veto_override_counts = _light_track_veto_override_groups(
                track_veto_candidates,
                min_candidates=int(cfg.light_veto_override_min_candidates),
            )

            if light_track_veto_override_labels:
                for trial in sorted(track_veto_candidates, key=lambda item: int(item.get("source_rank", 999999))):
                    veto_label = int(trial.get("track_veto_label", -1))
                    if veto_label not in light_track_veto_override_labels:
                        continue
                    tpc = int(trial.get("TPCid", -1))
                    trial["track_veto_override_eligible"] = True
                    trial["track_veto_override_group_size"] = int(light_track_veto_override_counts.get(veto_label, 0))

                    if len(light_moves) >= int(cfg.light_max_total_moves):
                        trial["track_veto_override_skipped"] = "total_move_limit"
                        continue
                    if moves_by_tpc.get(tpc, 0) >= int(cfg.light_max_moves_per_tpc):
                        trial["track_veto_override_skipped"] = "per_tpc_move_limit"
                        continue

                    gate = _evaluate_trial2_light_gate(
                        trial,
                        base_image=base_out,
                        full_light_waveform=full_light,
                        full_light_std=full_std,
                        saturated_channel_cache=namespace.get("saturated_channel_cache"),
                        config=cfg,
                        cfg25=cfg25_light,
                    )
                    accepted = bool(gate["accepted"])
                    trial.update(gate)
                    accepted = _apply_light_track_veto_if_needed(
                        trial,
                        accepted,
                        cfg=cfg,
                        labels_global=labels,
                        hit_tpc_ids=tpcs,
                        energy=energy,
                        label_info=label_info,
                        override_track_labels=light_track_veto_override_labels,
                        override_group_sizes=light_track_veto_override_counts,
                    )
                    trial["accepted"] = bool(accepted)
                    if not accepted:
                        trial["track_veto_override_skipped"] = str(trial.get("accepted_by", "reject"))
                        continue

                    before_ts_pass = hit_ts_out.copy()
                    before_base_pass = base_out.copy()
                    moved = np.asarray(trial["hit_indices"], dtype=np.int64)
                    old_t0 = int(trial["old_t0"])
                    new_t0 = int(trial["new_t0"])
                    hit_ts_out[moved] = np.float32(new_t0)

                    affected = {(tpc, old_t0), (tpc, new_t0)}
                    base_out, records = p25._exact_update_affected_families(
                        base_image=before_base_pass,
                        old_hit_timestamps=before_ts_pass,
                        new_hit_timestamps=hit_ts_out,
                        affected_specs=affected,
                        hit_tpc_ids=tpcs,
                        x=x,
                        y=y,
                        z=z,
                        energy=energy,
                        model=_get(namespace, "model"),
                        template=_get(namespace, "wvfm_tmpl"),
                        cfg=cfg25_light,
                    )
                    family_update_records.extend(records)
                    light_moves.append(trial)
                    moves_by_tpc[tpc] = moves_by_tpc.get(tpc, 0) + 1
        else:
            for _ in range(int(cfg.light_max_total_moves)):
                rows = _scan_light_overflows_source_only(
                    base_image=base_out,
                    full_light_waveform=full_light,
                    full_light_std=full_std,
                    source_t0s_by_tpc=source_t0s_by_tpc,
                    saturated_channel_cache=namespace.get("saturated_channel_cache"),
                    allowed_tpcs=set(int(v) for v in light_allowed_tpcs),
                    cfg=cfg25_light,
                )
                if not rows:
                    break
                accepted_this_pass = False
                for row in rows:
                    tpc = int(row["TPCid"])
                    row = dict(row)
                    row.update(
                        _light_source_detail_for_t0(
                            hit_timestamps=hit_ts_out,
                            hit_tpc_ids=tpcs,
                            labels_global=labels,
                            energy=energy,
                            tpc=tpc,
                            t0=int(row["t0"]),
                            match_ticks=float(cfg.light_t0_match_ticks),
                        )
                    )
                    if moves_by_tpc.get(tpc, 0) >= int(cfg.light_max_moves_per_tpc):
                        continue
                    trial = p25._try_light_repair_row(
                        row,
                        base_image=base_out,
                        full_light_waveform=full_light,
                        full_light_std=full_std,
                        image_maps=namespace.get("imageMaps"),
                        t0_candidates=t0_candidates_dict,
                        hit_timestamps=hit_ts_out,
                        hit_tpc_ids=tpcs,
                        labels_global=labels,
                        x=x,
                        y=y,
                        z=z,
                        energy=energy,
                        saturated_channel_cache=namespace.get("saturated_channel_cache"),
                        locked_hit_mask=None,
                        cfg=cfg25_light,
                    )
                    if trial is None:
                        continue
                    trial = dict(trial)
                    trial["overflow_row"] = dict(row)
                    proxy, image_status = p25._component_proxy_image(
                        component=trial,
                        image_maps=namespace.get("imageMaps"),
                        labels_global=labels,
                        hit_tpc_ids=tpcs,
                        energy=energy,
                    )
                    trial["image_status"] = str(image_status)
                    if proxy is not None:
                        old_shift = p25._shift_image(proxy, int(trial["old_t0"]), nt=base_out.shape[-1])
                        new_shift = p25._shift_image(proxy, int(trial["new_t0"]), nt=base_out.shape[-1])
                        trial["delta"] = (new_shift - old_shift).astype(np.float32)
                    gate = _evaluate_trial2_light_gate(
                        trial,
                        base_image=base_out,
                        full_light_waveform=full_light,
                        full_light_std=full_std,
                        saturated_channel_cache=namespace.get("saturated_channel_cache"),
                        config=cfg,
                        cfg25=cfg25_light,
                    )
                    accepted = bool(gate["accepted"])
                    trial.update(gate)
                    accepted = _apply_light_track_veto_if_needed(
                        trial,
                        accepted,
                        cfg=cfg,
                        labels_global=labels,
                        hit_tpc_ids=tpcs,
                        energy=energy,
                        label_info=label_info,
                    )
                    trial["accepted"] = bool(accepted)
                    light_trials.append(trial)
                    if not accepted:
                        continue

                    before_ts_pass = hit_ts_out.copy()
                    before_base_pass = base_out.copy()
                    moved = np.asarray(trial["hit_indices"], dtype=np.int64)
                    old_t0 = int(trial["old_t0"])
                    new_t0 = int(trial["new_t0"])
                    hit_ts_out[moved] = np.float32(new_t0)
                    affected = {(tpc, old_t0), (tpc, new_t0)}
                    base_out, records = p25._exact_update_affected_families(
                        base_image=before_base_pass,
                        old_hit_timestamps=before_ts_pass,
                        new_hit_timestamps=hit_ts_out,
                        affected_specs=affected,
                        hit_tpc_ids=tpcs,
                        x=x,
                        y=y,
                        z=z,
                        energy=energy,
                        model=_get(namespace, "model"),
                        template=_get(namespace, "wvfm_tmpl"),
                        cfg=cfg25_light,
                    )
                    family_update_records.extend(records)
                    light_moves.append(trial)
                    moves_by_tpc[tpc] = moves_by_tpc.get(tpc, 0) + 1
                    if flash_cluster_received is not None:
                        t0_candidates, flash_cluster_received, flash_row = mark_flash_cluster_assignment(
                            t0_candidates,
                            flash_cluster_received,
                            tpc=int(tpc),
                            t0=float(new_t0),
                            resolution_ticks=float(cfg.light_flash_cluster_match_ticks),
                            max_t0=800.0,
                            prefer_existing_true=False,
                            clusterid=int(trial.get("component_id", -1)),
                            stage="trial2_light_repair",
                        )
                        flash_canonicalization_rows.append(flash_row)
                        namespace["flash_cluster_received_by_tpc"] = flash_cluster_received
                    else:
                        _canonicalize_candidate_t0(
                            t0_candidates,
                            int(tpc),
                            float(new_t0),
                            merge_ticks=float(cfg.light_flash_cluster_match_ticks),
                        )
                    namespace["t0Candidates"] = t0_candidates
                    t0_candidates_dict = _as_tpc_dict(t0_candidates)
                    if flash_cluster_received is not None:
                        source_t0s_by_tpc, flash_cluster_association_rows = source_t0s_from_received_flash_table(
                            t0_candidates,
                            flash_cluster_received,
                            allowed_tpcs=light_allowed_tpcs,
                            min_sep_ticks=float(cfg.light_flash_cluster_match_ticks),
                        )
                    else:
                        source_t0s_by_tpc, flash_cluster_association_rows = _associated_source_t0s_by_tpc(
                            hit_timestamps=hit_ts_out,
                            hit_tpc_ids=tpcs,
                            labels_global=labels,
                            energy=energy,
                            t0_candidates=t0_candidates_dict,
                            allowed_tpcs=light_allowed_tpcs,
                            match_ticks=float(cfg.light_flash_cluster_match_ticks),
                        )
                    accepted_this_pass = True
                    break
                if not accepted_this_pass:
                    break

    result = {
        "baseImage": base_out.astype(np.float32),
        "hit_timestamps": hit_ts_out.astype(np.float32),
        "hit_timestamps_before": before_ts,
        "baseImage_before": before_base,
        "large_flash_grid": large_result,
        "spatial_moves": list(spatial_result.get("spatial_moves", [])),
        "spatial_trials": list(spatial_result.get("spatial_trials", [])),
        "light_moves": light_moves,
        "light_trials": light_trials,
        "light_source_rows": light_source_rows,
        "light_track_veto_override_labels": sorted(int(v) for v in light_track_veto_override_labels),
        "light_track_veto_override_counts": {int(k): int(v) for k, v in light_track_veto_override_counts.items()},
        "family_update_records": family_update_records,
        "skipped_shower_tpcs": list(spatial_result.get("skipped_shower_tpcs", [])),
        "allowed_tpcs": spatial_allowed_tpcs,
        "spatial_allowed_tpcs": spatial_allowed_tpcs,
        "light_allowed_tpcs": light_allowed_tpcs,
        "source_t0s_by_tpc": source_t0s_by_tpc,
        "t0Candidates": copy.deepcopy(t0_candidates),
        "flash_cluster_received_by_tpc": copy.deepcopy(flash_cluster_received),
        "flash_cluster_table_rows": flash_cluster_table_rows(t0_candidates, flash_cluster_received)
        if flash_cluster_received is not None
        else [],
        "flash_canonicalization_rows": flash_canonicalization_rows,
        "flash_cluster_associations": flash_cluster_association_rows,
        "config": cfg,
        "phase25_config": cfg25_light,
        "elapsed_s": float(time.time() - t_start),
    }

    if do_commit:
        namespace["baseImage"] = result["baseImage"]
        namespace["hit_timestamps"] = result["hit_timestamps"]
        namespace["t0Candidates"] = result["t0Candidates"]
        if result["flash_cluster_received_by_tpc"] is not None:
            namespace["flash_cluster_received_by_tpc"] = result["flash_cluster_received_by_tpc"]
        namespace["phase25_trial2_result"] = result
    else:
        # Restore namespace if the large/spatial helper committed intermediate state.
        namespace["baseImage"] = before_base
        namespace["hit_timestamps"] = before_ts
        namespace["t0Candidates"] = before_t0_candidates
        if before_flash_cluster_received is None:
            namespace.pop("flash_cluster_received_by_tpc", None)
        else:
            namespace["flash_cluster_received_by_tpc"] = before_flash_cluster_received

    return result


def _truth_arrays_from_namespace(namespace: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n_hits = len(np.asarray(_get(namespace, "hit_timestamps")))
    if "event_truth" in namespace and isinstance(namespace["event_truth"], dict):
        et = namespace["event_truth"]
        if "truth_t0" in et and len(et["truth_t0"]) == n_hits:
            return np.asarray(et["truth_t0"], dtype=np.float64), np.asarray(et["hit_energy"], dtype=np.float64)

    hits_ref = namespace.get("hits_ref", namespace.get("hit_refs"))
    if hits_ref is None:
        raise KeyError("Need hits_ref or hit_refs to extract truth.")
    ev_id = namespace.get("ev_id", namespace.get("event_id", namespace.get("eventid")))
    event_truth = extract_event_hit_energy_and_truth_t0(
        _get(namespace, "h5"),
        hits_ref,
        event_id=int(ev_id),
        hits_full=namespace.get("hits_full"),
        convert_to_matching_ticks=True,
    )
    namespace["event_truth"] = event_truth
    return np.asarray(event_truth["truth_t0"], dtype=np.float64), np.asarray(event_truth["hit_energy"], dtype=np.float64)


def _classify_move_hits(
    *,
    idx: np.ndarray,
    old_t0: np.ndarray,
    new_t0: np.ndarray,
    truth_t0: np.ndarray,
    energy: np.ndarray,
    tolerance_ticks: float,
) -> dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return {"n": 0, "energy": 0.0, "net_correct_energy": 0.0, "categories": {}}
    truth = truth_t0[idx]
    e = energy[idx]
    valid = np.isfinite(truth) & np.isfinite(e) & (e > 0)
    old_correct = valid & (np.abs(truth - old_t0) <= float(tolerance_ticks))
    new_correct = valid & (np.abs(truth - new_t0) <= float(tolerance_ticks))
    categories = {
        "wrong -> correct": valid & (~old_correct) & new_correct,
        "correct -> wrong": valid & old_correct & (~new_correct),
        "correct -> correct": valid & old_correct & new_correct,
        "wrong -> wrong": valid & (~old_correct) & (~new_correct),
    }
    return {
        "n": int(np.count_nonzero(valid)),
        "energy": float(np.sum(e[valid])),
        "net_correct_energy": float(np.sum(e[new_correct]) - np.sum(e[old_correct])),
        "categories": {k: (int(np.count_nonzero(v)), float(np.sum(e[v]))) for k, v in categories.items()},
    }


def print_stage_truth_summary(
    namespace: dict[str, Any],
    result: dict[str, Any],
    *,
    stage: str,
    print_top: int | None = None,
) -> list[dict[str, Any]]:
    """Print a compact truth table for spatial or light moves."""
    cfg: Trial2Config = result.get("config", Trial2Config())
    truth_t0, hit_energy = _truth_arrays_from_namespace(namespace)
    old_reco = np.asarray(result["hit_timestamps_before"], dtype=np.float64)
    new_reco = np.asarray(result["hit_timestamps"], dtype=np.float64)
    labels = np.asarray(_get(namespace, "labels_global"), dtype=np.int64)
    tpcs = np.asarray(_get(namespace, "hitTPCid"), dtype=np.int32)
    moves = list(result.get(f"{stage}_moves", []))
    rows: list[dict[str, Any]] = []

    for move_id, move in enumerate(moves, start=1):
        if stage == "spatial":
            idx = np.asarray(move.get("moved_idx", []), dtype=np.int64)
        else:
            idx = np.asarray(move.get("hit_indices", []), dtype=np.int64)
        if idx.size == 0:
            continue
        old_vals = old_reco[idx]
        new_vals = new_reco[idx]
        stats = _classify_move_hits(
            idx=idx,
            old_t0=old_vals,
            new_t0=new_vals,
            truth_t0=truth_t0,
            energy=hit_energy,
            tolerance_ticks=float(cfg.truth_tolerance_ticks),
        )
        cat_e = {k: v[1] for k, v in stats["categories"].items()}
        dom_old = int(round(float(np.nanmedian(old_vals)))) if old_vals.size else -1
        dom_new = int(round(float(np.nanmedian(new_vals)))) if new_vals.size else -1
        vals, counts = np.unique(labels[idx], return_counts=True)
        order = np.argsort(counts)[::-1]
        label_text = "[" + ", ".join(f"{int(vals[i])}:{int(counts[i])}" for i in order[:4]) + "]"
        rows.append(
            {
                "stage": stage,
                "move": int(move_id),
                "TPCid": int(move.get("TPCid", int(tpcs[idx[0]]))),
                "old": int(dom_old),
                "new": int(dom_new),
                "hits": int(idx.size),
                "energy": float(np.sum(hit_energy[idx][np.isfinite(hit_energy[idx])])),
                "netE": float(stats["net_correct_energy"]),
                "WtoC_E": float(cat_e.get("wrong -> correct", 0.0)),
                "CtoW_E": float(cat_e.get("correct -> wrong", 0.0)),
                "WW_E": float(cat_e.get("wrong -> wrong", 0.0)),
                "labels": label_text,
                "accepted": bool(move.get("accepted", True)),
                "src_red": int(move.get("src_red", -1)),
                "loss": float(move.get("loss_improvement", -move.get("loss_delta", np.nan))),
            }
        )

    rows.sort(key=lambda r: (-abs(float(r["netE"])), int(r["move"])))
    top = int(print_top if print_top is not None else cfg.print_top_rows)
    total_e = sum(float(r["energy"]) for r in rows)
    net_e = sum(float(r["netE"]) for r in rows)
    print(f"Trial2 {stage} moved-hit truth summary")
    print(f"  moves          : {len(rows)}")
    print(f"  moved energy   : {total_e:.4f} MeV")
    print(f"  net correct E  : {net_e:+.4f} MeV")
    print()
    header = (
        f"{'move':>4} {'TPC':>4} {'old':>6} {'new':>6} {'hits':>6} "
        f"{'E[MeV]':>9} {'netE':>9} {'W->C':>9} {'C->W':>9} {'WW':>9} "
        f"{'srcRed':>6} {'loss':>11} {'labels':>24}"
    )
    print(header)
    print("-" * len(header))
    for r in rows[:top]:
        print(
            f"{r['move']:4d} {r['TPCid']:4d} {r['old']:6d} {r['new']:6d} "
            f"{r['hits']:6d} {r['energy']:9.3f} {r['netE']:9.3f} "
            f"{r['WtoC_E']:9.3f} {r['CtoW_E']:9.3f} {r['WW_E']:9.3f} "
            f"{r['src_red']:6d} {r['loss']:11.2e} {r['labels']:>24}"
        )
    return rows


def _trial2_light_reason(trial: dict[str, Any]) -> str:
    if bool(trial.get("no_trial", False)):
        return str(trial.get("reason", "no_trial"))
    if bool(trial.get("track_veto", False)):
        if bool(trial.get("track_veto_overridden", False)) and bool(trial.get("accepted", False)):
            size = int(trial.get("track_veto_override_group_size", 0))
            return f"accepted:trackOverride{size}"
        label = int(trial.get("track_veto_label", -1))
        n_tpcs = int(trial.get("track_veto_n_tpcs", 0))
        return f"rejected:track{n_tpcs}TPC(label{label})"
    accepted_by = str(trial.get("accepted_by", ""))
    if accepted_by == "physicalChi2":
        return "accepted"
    if "dchi2" in trial and accepted_by == "reject":
        failed = []
        if not bool(trial.get("channel_ok", False)):
            failed.append("srcCh")
        if not bool(trial.get("dchi2_ok", False)):
            failed.append("dChi2")
        if not bool(trial.get("dchi2_per_e_ok", False)):
            failed.append("dChi2/E")
        return "accepted" if not failed else "rejected:" + "+".join(failed)
    if accepted_by and accepted_by not in {"reject", "sourceGate"}:
        return "accepted" if accepted_by == "base" and bool(trial.get("old_ok", False)) else f"accepted:{accepted_by}"
    if "base_flags" in trial and accepted_by == "reject":
        failed = [name for name, ok in trial.get("base_flags", []) if not bool(ok)]
        out = "rejected:" + "+".join(failed) if failed else "rejected:noBranch"
        if not bool(trial.get("old_ok", False)):
            out += "+oldFail"
        return out
    failed = []
    if not bool(trial.get("old_ok", False)):
        failed.append("oldOK")
    if not bool(trial.get("channel_ok", False)):
        failed.append("srcCh")
    if not bool(trial.get("src_def_ok", False)):
        failed.append("srcDef")
    return "accepted" if not failed else "rejected:" + "+".join(failed)


def print_trial2_light_acceptance_table(
    namespace: dict[str, Any],
    result: dict[str, Any],
    *,
    print_top: int | None = None,
) -> list[dict[str, Any]]:
    """Print all Trial2 light proposals with the actual source-gated decisions."""
    cfg: Trial2Config = result.get("config", Trial2Config())
    truth_t0, hit_energy = _truth_arrays_from_namespace(namespace)
    labels = np.asarray(_get(namespace, "labels_global"), dtype=np.int64)
    trials = list(result.get("light_trials", []))
    rows: list[dict[str, Any]] = []

    for trial_id, trial in enumerate(trials, start=1):
        idx = np.asarray(trial.get("hit_indices", []), dtype=np.int64)
        old_t0 = int(trial.get("old_t0", -1))
        new_t0 = int(trial.get("new_t0", -1))
        accepted = bool(trial.get("accepted", False))

        if idx.size:
            old_vals = np.full(idx.shape, float(old_t0), dtype=np.float64)
            new_vals = np.full(idx.shape, float(new_t0), dtype=np.float64)
            stats = _classify_move_hits(
                idx=idx,
                old_t0=old_vals,
                new_t0=new_vals,
                truth_t0=truth_t0,
                energy=hit_energy,
                tolerance_ticks=float(cfg.truth_tolerance_ticks),
            )
            vals, counts = np.unique(labels[idx], return_counts=True)
            order = np.argsort(counts)[::-1]
            label_text = "[" + ", ".join(f"{int(vals[i])}:{int(counts[i])}" for i in order[:4]) + "]"
        else:
            stats = {"energy": 0.0, "net_correct_energy": np.nan, "categories": {}}
            label_text = "[]"

        net_truth_e = float(stats.get("net_correct_energy", np.nan))
        if np.isfinite(net_truth_e):
            truth_should_accept = net_truth_e > float(cfg.min_net_truth_energy_mev)
            truth_should_reject = net_truth_e < -float(cfg.min_net_truth_energy_mev)
            if accepted and truth_should_accept:
                decision = "Correctly accepted"
            elif accepted and truth_should_reject:
                decision = "Incorrectly accepted"
            elif (not accepted) and truth_should_reject:
                decision = "Correctly rejected"
            elif (not accepted) and truth_should_accept:
                decision = "Incorrectly rejected"
            else:
                decision = "Neutral"
        else:
            decision = "Accepted" if accepted else "Rejected"

        rows.append(
            {
                "trial": int(trial_id),
                "source_rank": int(trial.get("source_rank", trial_id)),
                "decision": str(decision),
                "accepted": bool(accepted),
                "reason": _trial2_light_reason(trial),
                "accepted_by": str(trial.get("accepted_by", "reject")),
                "TPCid": int(trial.get("TPCid", -1)),
                "old_t0": int(old_t0),
                "new_t0": int(new_t0),
                "component_id": int(trial.get("component_id", -1)),
                "n_hits": int(idx.size),
                "energy_mev": float(trial.get("energy_mev", stats.get("energy", 0.0))),
                "q_energy_mev": float(trial.get("q_energy_mev", trial.get("overflow_row", {}).get("charge_energy_mev", 0.0))),
                "src_of0": int(trial.get("src_of0", 0)),
                "src_of1": int(trial.get("src_of1", 0)),
                "src_red": int(trial.get("src_red", 0)),
                "dst_new": int(trial.get("dst_new", 0)),
                "dst_ch_ok": bool(trial.get("dst_ch_ok", False)),
                "base_dst_limit": float(trial.get("base_dst_limit", np.nan)),
                "src_remain_frac": float(trial.get("src_remain_frac", np.nan)),
                "move_frac": float(trial.get("move_frac", np.nan)),
                "src_def_inc": float(trial.get("src_def_inc", np.nan)),
                "src_def_limit": float(trial.get("src_def_limit", np.nan)),
                "old_ok": bool(trial.get("old_ok", False)),
                "channel_ok": bool(trial.get("channel_ok", False)),
                "src_def_ok": bool(trial.get("src_def_ok", False)),
                "dchi2_ok": bool(trial.get("dchi2_ok", False)),
                "dchi2_per_e_ok": bool(trial.get("dchi2_per_e_ok", False)),
                "loss_per_mev": float(trial.get("loss_per_mev", np.nan)),
                "loss_improvement": float(trial.get("loss_improvement", np.nan)),
                "chi2_before": float(trial.get("chi2_before", np.nan)),
                "chi2_after": float(trial.get("chi2_after", np.nan)),
                "dchi2": float(trial.get("dchi2", np.nan)),
                "dchi2_per_mev": float(trial.get("dchi2_per_mev", np.nan)),
                "dchi2_per_dof": float(trial.get("dchi2_per_dof", np.nan)),
                "dof": int(trial.get("dof", 0)),
                "track_veto": bool(trial.get("track_veto", False)),
                "track_veto_overridden": bool(trial.get("track_veto_overridden", False)),
                "track_veto_override_group_size": int(trial.get("track_veto_override_group_size", 0)),
                "track_veto_label": int(trial.get("track_veto_label", -1)),
                "track_veto_n_tpcs": int(trial.get("track_veto_n_tpcs", 0)),
                "track_veto_label_energy_mev": float(trial.get("track_veto_label_energy_mev", 0.0)),
                "track_veto_label_tpcs": list(trial.get("track_veto_label_tpcs", [])),
                "net_truth_E": net_truth_e,
                "labels": label_text,
            }
        )

    def _priority(row: dict[str, Any]) -> tuple[Any, ...]:
        order = {
            "Incorrectly accepted": 0,
            "Incorrectly rejected": 1,
            "Correctly accepted": 2,
            "Correctly rejected": 3,
            "Accepted": 4,
            "Rejected": 5,
            "Neutral": 6,
        }.get(str(row.get("decision", "")), 9)
        truth_e = float(row.get("net_truth_E", 0.0))
        if not np.isfinite(truth_e):
            truth_e = 0.0
        return (order, -abs(truth_e), int(row.get("source_rank", row.get("trial", 999999))))

    printable = sorted(rows, key=_priority)
    top = int(print_top if print_top is not None else cfg.print_top_rows)

    accepted_rows = [r for r in rows if r.get("accepted")]
    if bool(cfg.light_use_physical_chi2):
        print("Physical-loss light repair audit")
        print(f"  source rows found        : {len(result.get('light_source_rows', rows))}")
        print(f"  rows evaluated           : {len(rows)}")
    elif bool(cfg.light_use_rescue_branches):
        print("Trial2 v2-rescue light acceptance audit")
        print(f"  trials evaluated         : {len(rows)}")
    else:
        print("Trial2 source-gated light acceptance audit")
        print(f"  trials evaluated         : {len(rows)}")
    print(f"  accepted repairs         : {len(accepted_rows)}")
    if bool(cfg.light_use_physical_chi2):
        print(f"  source channel rule      : srcRed >= {cfg.phys_min_source_ofch_reduction}")
        print(f"  physical loss rule       : dChi2 >= {cfg.phys_min_dchi2_improvement:.2e}")
        print(f"  physical loss/E rule     : dChi2/E >= {cfg.phys_min_dchi2_per_mev:.2e}")
        half_window = int(getattr(result.get("phase25_config", None), "half_window_ticks", 18))
        print(f"  chi2 window              : union of old/new t0 windows, +/- {half_window} ticks")
        if bool(cfg.light_veto_multitpc_track):
            print(
                f"  track veto               : reject dominant moved label spanning >= {cfg.light_veto_track_min_tpcs} TPCs"
            )
            print(
                "  track veto override      : allow a vetoed long-track label when "
                f">= {cfg.light_veto_override_min_candidates} otherwise accepted proposals exist"
            )
            print("  accept uses              : srcRed, dChi2, dChi2/E, track veto")
        else:
            print("  accept uses only         : srcRed, dChi2, dChi2/E")
    elif bool(cfg.light_use_rescue_branches):
        print(f"  source channel rule      : base srcRed > {cfg.light_source_channel_drop_strict_gt}")
        print(
            "  destination rule         : base dstNew <= max("
            f"{cfg.light_max_dest_new_channels_abs:.0f}, "
            f"{cfg.light_max_dest_new_channels_frac_of_src_red:.2f}*srcRed)"
        )
        print(f"  loss density rule        : base dLoss/E >= {cfg.light_min_dloss_per_mev:.2e}")
        print(
            "  rescue branches          : highE oldFail, source-clear, oldOK-partial "
            "(same as v_test_trial2.ipynb v2-rescue)"
        )
        print(f"  light TPC scan           : {'spatial-allowed only' if cfg.light_skip_shower_tpcs else 'all hit TPCs'}")
        print("  physical dChi2 gate      : disabled")
    else:
        print(f"  source channel rule      : srcRed > {cfg.light_source_channel_drop_strict_gt}")
        print(
            "  source deficit rule      : "
            f"srcDefInc <= {cfg.light_max_source_deficit_increase:.2e} "
            f"+ {cfg.light_max_source_deficit_increase_per_mev:.2e}*E"
        )
        print("  old p25 rule             : oldOK must be True")
        print("  accept uses only         : oldOK, srcCh, srcDef")

    if any(np.isfinite(float(r.get("net_truth_E", np.nan))) for r in rows):
        print(f"  correctly accepted       : {sum(1 for r in rows if r.get('decision') == 'Correctly accepted')}")
        print(f"  incorrectly accepted     : {sum(1 for r in rows if r.get('decision') == 'Incorrectly accepted')}")
        print(f"  correctly rejected       : {sum(1 for r in rows if r.get('decision') == 'Correctly rejected')}")
        print(f"  incorrectly rejected     : {sum(1 for r in rows if r.get('decision') == 'Incorrectly rejected')}")
    print()

    if bool(cfg.light_use_physical_chi2):
        header = (
            f"{'decision':>22} {'reason':>22} {'rank':>4} {'TPC':>4} "
            f"{'old':>5} {'new':>5} {'comp':>5} {'hits':>6} "
            f"{'E':>8} {'src0':>5} {'src1':>5} {'red':>5} {'dst':>5} "
            f"{'dChi2':>12} {'dChi2/E':>11} {'dChi2/dof':>11} {'trkLbl':>6} {'trkTPC':>6} {'truthE':>9}"
        )
    elif bool(cfg.light_use_rescue_branches):
        header = (
            f"{'decision':>22} {'reason':>24} {'branch':>12} {'rank':>5} {'TPC':>4} "
            f"{'old':>5} {'new':>5} {'comp':>5} {'hits':>6} "
            f"{'E':>8} {'qE':>8} {'src0':>5} {'src1':>5} {'red':>5} {'dst':>5} "
            f"{'dstLim':>7} {'srcRem':>7} {'mvFrac':>7} {'oldOK':>5} "
            f"{'srcOK':>5} {'dstOK':>5} {'defOK':>5} {'dL/E':>10} {'truthE':>9} {'labels':>22}"
        )
    else:
        header = (
            f"{'decision':>22} {'reason':>24} {'trial':>5} {'TPC':>4} "
            f"{'old':>5} {'new':>5} {'comp':>5} {'hits':>6} "
            f"{'E':>8} {'src0':>5} {'src1':>5} {'red':>5} {'dst':>5} "
            f"{'srcDef':>10} {'defLim':>10} {'oldOK':>5} {'srcOK':>5} "
            f"{'defOK':>5} {'loss':>10} {'truthE':>9} {'labels':>22}"
        )
    print(header)
    print("-" * len(header))

    for row in printable[:top]:
        if bool(cfg.light_use_physical_chi2):
            print(
                f"{str(row.get('decision', '')):>22} "
                f"{str(row.get('reason', ''))[:22]:>22} "
                f"{int(row.get('source_rank', row.get('trial', -1))):4d} "
                f"{int(row.get('TPCid', -1)):4d} "
                f"{int(row.get('old_t0', -1)):5d} "
                f"{int(row.get('new_t0', -1)):5d} "
                f"{int(row.get('component_id', -1)):5d} "
                f"{int(row.get('n_hits', 0)):6d} "
                f"{float(row.get('energy_mev', 0.0)):8.2f} "
                f"{int(row.get('src_of0', 0)):5d} "
                f"{int(row.get('src_of1', 0)):5d} "
                f"{int(row.get('src_red', 0)):5d} "
                f"{int(row.get('dst_new', 0)):5d} "
                f"{float(row.get('dchi2', 0.0)):12.2e} "
                f"{float(row.get('dchi2_per_mev', 0.0)):11.2e} "
                f"{float(row.get('dchi2_per_dof', 0.0)):11.2e} "
                f"{int(row.get('track_veto_label', -1)):6d} "
                f"{int(row.get('track_veto_n_tpcs', 0)):6d} "
                f"{float(row.get('net_truth_E', np.nan)):9.2f}"
            )
        elif bool(cfg.light_use_rescue_branches):
            print(
                f"{str(row.get('decision', '')):>22} "
                f"{str(row.get('reason', ''))[:24]:>24} "
                f"{str(row.get('accepted_by', ''))[:12]:>12} "
                f"{int(row.get('source_rank', row.get('trial', -1))):5d} "
                f"{int(row.get('TPCid', -1)):4d} "
                f"{int(row.get('old_t0', -1)):5d} "
                f"{int(row.get('new_t0', -1)):5d} "
                f"{int(row.get('component_id', -1)):5d} "
                f"{int(row.get('n_hits', 0)):6d} "
                f"{float(row.get('energy_mev', 0.0)):8.2f} "
                f"{float(row.get('q_energy_mev', 0.0)):8.2f} "
                f"{int(row.get('src_of0', 0)):5d} "
                f"{int(row.get('src_of1', 0)):5d} "
                f"{int(row.get('src_red', 0)):5d} "
                f"{int(row.get('dst_new', 0)):5d} "
                f"{float(row.get('base_dst_limit', 0.0)):7.1f} "
                f"{float(row.get('src_remain_frac', 0.0)):7.2f} "
                f"{float(row.get('move_frac', 0.0)):7.2f} "
                f"{str(bool(row.get('old_ok', False))):>5} "
                f"{str(bool(row.get('channel_ok', False))):>5} "
                f"{str(bool(row.get('dst_ch_ok', False))):>5} "
                f"{str(bool(row.get('src_def_ok', False))):>5} "
                f"{float(row.get('loss_per_mev', 0.0)):10.2e} "
                f"{float(row.get('net_truth_E', np.nan)):9.2f} "
                f"{str(row.get('labels', '[]'))[:22]:>22}"
            )
        else:
            print(
                f"{str(row.get('decision', '')):>22} "
                f"{str(row.get('reason', ''))[:24]:>24} "
                f"{int(row.get('trial', -1)):5d} "
                f"{int(row.get('TPCid', -1)):4d} "
                f"{int(row.get('old_t0', -1)):5d} "
                f"{int(row.get('new_t0', -1)):5d} "
                f"{int(row.get('component_id', -1)):5d} "
                f"{int(row.get('n_hits', 0)):6d} "
                f"{float(row.get('energy_mev', 0.0)):8.2f} "
                f"{int(row.get('src_of0', 0)):5d} "
                f"{int(row.get('src_of1', 0)):5d} "
                f"{int(row.get('src_red', 0)):5d} "
                f"{int(row.get('dst_new', 0)):5d} "
                f"{float(row.get('src_def_inc', 0.0)):10.2e} "
                f"{float(row.get('src_def_limit', 0.0)):10.2e} "
                f"{str(bool(row.get('old_ok', False))):>5} "
                f"{str(bool(row.get('channel_ok', False))):>5} "
                f"{str(bool(row.get('src_def_ok', False))):>5} "
                f"{float(row.get('loss_improvement', 0.0)):10.2e} "
                f"{float(row.get('net_truth_E', np.nan)):9.2f} "
                f"{str(row.get('labels', '[]'))[:22]:>22}"
            )

    return rows


def print_trial2_summary(result: dict[str, Any]) -> None:
    large_rows = result.get("large_flash_grid", {}).get("accepted_rows", [])
    print("Trial2 amendment summary")
    print(f"  large flash-grid corrections : {len(large_rows)}")
    print(f"  skipped shower TPCs          : {result.get('skipped_shower_tpcs', [])}")
    print(f"  spatial accepted moves       : {len(result.get('spatial_moves', []))}")
    print(f"  spatial trials               : {len(result.get('spatial_trials', []))}")
    print(f"  light accepted moves         : {len(result.get('light_moves', []))}")
    print(f"  light trials                 : {len(result.get('light_trials', []))}")
    print(f"  family updates               : {len(result.get('family_update_records', []))}")
    print(f"  elapsed                      : {float(result.get('elapsed_s', np.nan)):.1f}s")
