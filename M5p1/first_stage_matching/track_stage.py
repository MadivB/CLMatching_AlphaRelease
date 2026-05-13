from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FirstStageConfig
from .paths import configure_paths
from .prediction import ModelBundle, PredictionBundle

configure_paths()

from pulse_shapes import timeinterpolation
from fast_unit_track_scan import unit_likelihood_curve_with_base_v1
from v10_3_support import stack_cluster_fit_inputs_v10_3
from v13_2_track_rescan import run_track_second_pass_rescan_v10_3
from v10_7_focused_track_swap import (
    build_track_swap_candidates_from_hits_v10_7_focused,
    run_track_overlap_swap_rescue_v10_7_focused,
)
from v13_4_track_t0_correction import run_track_t0_fine_correction_v13_4
from v8_leftover_absorption import (
    prepare_leftover_absorption_state_v8,
    try_absorb_leftovers_for_parent_v8,
)
try:
    from ..flash_cluster_table import (
        ensure_flash_cluster_flags,
        flash_cluster_table_rows,
        mark_flash_cluster_assignment,
        rebuild_flash_cluster_flags_from_assignments,
    )
except Exception:  # pragma: no cover - direct notebook import fallback
    from flash_cluster_table import (
        ensure_flash_cluster_flags,
        flash_cluster_table_rows,
        mark_flash_cluster_assignment,
        rebuild_flash_cluster_flags_from_assignments,
    )


@dataclass(slots=True)
class TrackStageResult:
    hit_t0: np.ndarray
    hit_t0_export: np.ndarray
    base_image: np.ndarray
    t0_candidates_by_tpc: list[list[int]]
    raw_flash_table_by_tpc: dict[int, list[int]]
    modified_flash_table_by_tpc: dict[int, list[int]]
    flash_cluster_received_by_tpc: list[list[bool]]
    flash_cluster_table_rows: list[dict[str, Any]]
    flash_cluster_canonicalization_log: list[dict[str, Any]]
    flash_table_amendment_log: list[dict[str, Any]]
    flash_seed_stats: dict[str, Any]
    assignment_info: dict[tuple[int, int], dict[str, Any]]
    track_channel_diagnostics: list[dict[str, Any]]
    cluster_full_scan_loss_dict: dict[int, dict[str, Any]]
    track_second_pass_log: list[dict[str, Any]]
    track_second_pass_stats: dict[str, Any]
    track_swap_log: list[dict[str, Any]]
    track_swap_stats: dict[str, Any]
    track_t0_fine_correction_log: list[dict[str, Any]]
    track_t0_fine_correction_stats: dict[str, Any]
    labels_global_with_leftovers: np.ndarray
    absorbed_hit_parent: np.ndarray
    leftover_absorption_log: list[dict[str, Any]]
    leftover_absorption_stats: dict[str, Any]
    leftover_state: dict[str, Any] | None
    leftover_prep_stats: dict[str, Any]
    stage_timings: dict[str, float]


def fast_stack_images(image_maps: dict[tuple[int, int], np.ndarray], cluster_id: int, tpcids: np.ndarray) -> np.ndarray:
    images = [np.asarray(image_maps[(int(cluster_id), int(tpc))], dtype=np.float32) for tpc in np.asarray(tpcids, dtype=int)]
    stacked = np.stack(images, axis=0)
    return stacked.reshape(-1, stacked.shape[-1])


def image_updater(new_image: np.ndarray, base_image: np.ndarray, sorted_tpcs: np.ndarray, *, num_channels: int = 120) -> np.ndarray:
    sorted_tpcs = np.asarray(sorted_tpcs, dtype=int)
    new_r = np.asarray(new_image, dtype=np.float32).reshape(sorted_tpcs.size, int(num_channels), -1)
    base_image[sorted_tpcs] += new_r
    return base_image


def append_candidate_t0(candidate_list: list[int], t0: float, *, min_sep: int = 2, max_t0: int = 800) -> bool:
    value = int(np.clip(np.rint(float(t0)), 0, int(max_t0)))
    for existing in candidate_list:
        if abs(int(existing) - value) <= int(min_sep):
            return False
    candidate_list.append(value)
    candidate_list.sort()
    return True


def _dominant_label_t0(hit_timestamps: np.ndarray, hit_indices: np.ndarray | None, labels_global: np.ndarray, label: int) -> tuple[int | None, float]:
    if hit_indices is None:
        values = np.asarray(hit_timestamps[np.asarray(labels_global, dtype=np.int32) == int(label)], dtype=np.float32)
    else:
        values = np.asarray(hit_timestamps[np.asarray(hit_indices, dtype=np.int64)], dtype=np.float32)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return None, 0.0
    rounded = np.asarray(np.rint(values), dtype=np.int32)
    unique, counts = np.unique(rounded, return_counts=True)
    best_idx = int(np.argmax(counts))
    return int(unique[best_idx]), float(counts[best_idx] / max(int(values.size), 1))


def _collect_final_backbone_t0_rows(
    *,
    track_shower_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    label_info: dict[int, dict[str, Any]],
    label_energy_sums: np.ndarray,
    label_hit_indices: dict[int, np.ndarray],
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    max_charge_tpc: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in track_shower_labels:
        t0, assigned_fraction = _dominant_label_t0(
            hit_timestamps=hit_timestamps,
            hit_indices=label_hit_indices.get(int(label)),
            labels_global=labels_global,
            label=int(label),
        )
        if t0 is None:
            continue
        energy = float(label_energy_sums[int(label)]) if int(label) < int(label_energy_sums.size) else 0.0
        label_type = str(label_info.get(int(label), {}).get("type", "track")).lower()
        for tpc in sorted(int(tpc) for tpc in cluster_to_tpcs.get(int(label), []) if int(tpc) < int(max_charge_tpc)):
            rows.append(
                {
                    "clusterid": int(label),
                    "tpc": int(tpc),
                    "t0": int(t0),
                    "energy": float(energy),
                    "label_type": str(label_type),
                    "assigned_fraction": float(assigned_fraction),
                }
            )
    return rows


def _amend_flash_table_with_track_t0s(
    *,
    raw_flash_table_by_tpc: dict[int, list[int]],
    final_backbone_rows: list[dict[str, Any]],
    max_charge_tpc: int,
    amend_window_ticks: int,
    max_t0: int,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    amended: dict[int, list[int]] = {
        int(tpc): sorted(set(int(v) for v in values if np.isfinite(float(v))))
        for tpc, values in (raw_flash_table_by_tpc or {}).items()
        if 0 <= int(tpc) < int(max_charge_tpc)
    }
    log: list[dict[str, Any]] = []

    backbone_rows = [
        dict(row)
        for row in final_backbone_rows
        if str(row.get("label_type", "track")).lower() in {"track", "shower"}
    ]
    backbone_rows = sorted(
        backbone_rows,
        key=lambda row: (-float(row.get("energy", 0.0)), int(row.get("clusterid", -1)), int(row.get("tpc", -1))),
    )

    for row in backbone_rows:
        tpc = int(row["tpc"])
        track_t0 = int(np.clip(int(row["t0"]), 0, int(max_t0)))
        current = list(amended.get(tpc, []))
        if len(current) == 0:
            continue

        removed = [int(v) for v in current if abs(int(v) - track_t0) <= int(amend_window_ticks)]
        if len(removed) == 0:
            continue

        survivors = [int(v) for v in current if abs(int(v) - track_t0) > int(amend_window_ticks)]
        if track_t0 not in survivors:
            survivors.append(track_t0)
        amended[tpc] = sorted(set(int(v) for v in survivors))

        log.append(
            {
                "clusterid": int(row["clusterid"]),
                "tpc": int(tpc),
                "track_t0": int(track_t0),
                "label_type": str(row.get("label_type", "track")),
                "removed_flash_t0s": sorted(set(removed)),
                "amend_window_ticks": int(amend_window_ticks),
                "energy": float(row.get("energy", 0.0)),
            }
        )

    amended = {
        int(tpc): [int(v) for v in sorted(set(values))]
        for tpc, values in sorted(amended.items())
        if len(values) > 0
    }
    return amended, log


def _rebuild_phase1_t0_candidates(
    *,
    final_backbone_rows: list[dict[str, Any]],
    amended_flash_table_by_tpc: dict[int, list[int]],
    max_charge_tpc: int,
    min_sep: int,
    max_t0: int,
) -> list[list[int]]:
    candidates: list[list[int]] = [[] for _ in range(int(max_charge_tpc))]
    for tpc, values in sorted((amended_flash_table_by_tpc or {}).items()):
        if 0 <= int(tpc) < int(max_charge_tpc):
            for t0 in values:
                append_candidate_t0(candidates[int(tpc)], int(t0), min_sep=int(min_sep), max_t0=int(max_t0))

    # Keep all final track/shower placements as valid Phase-2/3 candidate t0s.
    # This avoids stale pre-rescan/swap/fine-correction values.
    for row in sorted(final_backbone_rows, key=lambda r: (-float(r.get("energy", 0.0)), int(r.get("clusterid", -1)))):
        tpc = int(row["tpc"])
        if 0 <= tpc < int(max_charge_tpc):
            append_candidate_t0(candidates[tpc], int(row["t0"]), min_sep=int(min_sep), max_t0=int(max_t0))
    return candidates


def exact_likelihood_curve_with_base(
    predicted: np.ndarray,
    base: np.ndarray,
    actual: np.ndarray,
    error_metric: np.ndarray,
    *,
    search_range: int,
    adc_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=np.float32)
    base_ = np.asarray(base, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    err = np.maximum(np.asarray(error_metric, dtype=np.float32), 1e-6)
    shifts = np.arange(int(search_range) + 1, dtype=np.int32)
    errors = np.empty(shifts.size, dtype=np.float32)
    n_ticks = float(pred.size)
    for idx, t0 in enumerate(shifts):
        shifted = np.zeros_like(pred)
        if int(t0) > 0:
            shifted[:, int(t0) :] = pred[:, : -int(t0)]
        else:
            shifted[:] = pred
        model = np.clip(shifted + base_, None, float(adc_clip))
        errors[idx] = float(np.sum((model - act) ** 2 / err) / n_ticks)
    return shifts, errors


def correlation_unit_likelihood_curve_with_base(
    predicted: np.ndarray,
    base: np.ndarray,
    actual: np.ndarray,
    *,
    search_range: int,
    engine: str = "fft",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast phase-1 scan for unit std.

    It minimizes mean((shift(pred) + base - actual)^2) without ADC clipping.
    The saturation veto usually removes clipped channels before this stage.
    """
    return unit_likelihood_curve_with_base_v1(
        predicted,
        base,
        actual,
        search_range=int(search_range),
        engine=str(engine),
    )


def refine_t0_from_local_peak(
    *,
    raw_t0: int,
    actual_fit: np.ndarray,
    base_fit: np.ndarray,
    t0_resolution: int,
    waveform_len: int,
) -> int:
    new_t0 = int(raw_t0)
    expected_peak = 105 + new_t0
    signal_1d = np.sum(np.clip(np.asarray(actual_fit, dtype=np.float32) - np.asarray(base_fit, dtype=np.float32), 0, None), axis=0)
    start = max(0, expected_peak - int(t0_resolution))
    stop = min(int(waveform_len), expected_peak + int(t0_resolution) + 1)
    if stop > start:
        actual_peak = start + int(np.argmax(signal_1d[start:stop]))
        new_t0 += int(actual_peak - expected_peak)
    return int(np.clip(new_t0, 0, 800))


def run_track_shower_stage(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    io_group: np.ndarray,
    hit_tpc_id: np.ndarray,
    labels_global: np.ndarray,
    split_index: int,
    label_info: dict[int, dict[str, Any]],
    prediction: PredictionBundle,
    models: ModelBundle | None = None,
    flash_seed_t0s_by_tpc: dict[int, list[int]] | None = None,
    flash_seed_stats: dict[str, Any] | None = None,
    config: FirstStageConfig | None = None,
    verbose: bool = True,
) -> TrackStageResult:
    config = FirstStageConfig() if config is None else config
    t_total = time.perf_counter()
    t_leftover_prep = 0.0
    t_initial_loop = 0.0
    t_initial_fit_stack = 0.0
    t_initial_t0_scan = 0.0
    t_initial_apply = 0.0
    t_leftover_absorb = 0.0
    t_rescan = 0.0
    t_swap = 0.0
    t_fine = 0.0

    image_maps = prediction.image_maps
    cluster_to_tpcs = prediction.cluster_to_tpcs
    full_light_waveform = np.asarray(prediction.full_light_waveform, dtype=np.float32)
    labels_global_arr = np.asarray(labels_global, dtype=np.int32)
    energy_arr = np.asarray(energy, dtype=np.float32)

    track_shower_labels = list(range(int(split_index)))
    max_charge_tpc = int(full_light_waveform.shape[0])
    base_image = np.zeros_like(full_light_waveform, dtype=np.float32)
    t0_candidates: list[list[int]] = [[] for _ in range(max_charge_tpc)]
    flash_cluster_received_by_tpc = ensure_flash_cluster_flags(t0_candidates, max_t0=config.track_stage.search_range)
    flash_cluster_canonicalization_log: list[dict[str, Any]] = []
    raw_flash_table_by_tpc = {
        int(tpc): [int(v) for v in sorted(set(int(x) for x in values))]
        for tpc, values in sorted((flash_seed_t0s_by_tpc or {}).items())
        if 0 <= int(tpc) < int(max_charge_tpc) and len(values) > 0
    }
    flash_seed_stats = {} if flash_seed_stats is None else dict(flash_seed_stats)
    hit_timestamps = np.full(labels_global_arr.shape[0], np.nan, dtype=np.float32)
    assignment_info: dict[tuple[int, int], dict[str, Any]] = {}
    cluster_full_scan_loss_dict: dict[int, dict[str, Any]] = {}
    track_channel_diagnostics: list[dict[str, Any]] = []

    labels_global_with_leftovers = labels_global_arr.copy()
    absorbed_hit_parent = np.full(labels_global_arr.shape[0], -1, dtype=np.int32)
    leftover_absorption_log: list[dict[str, Any]] = []
    leftover_absorption_stats: dict[str, Any] = {}
    leftover_state: dict[str, Any] | None = None
    leftover_prep_stats: dict[str, Any] = {}

    valid_labels = labels_global_arr >= 0
    if bool(np.any(valid_labels)):
        max_label_id = int(np.max(labels_global_arr[valid_labels]))
        label_energy_sums = np.bincount(
            labels_global_arr[valid_labels],
            weights=np.asarray(energy_arr[valid_labels], dtype=np.float64),
            minlength=max_label_id + 1,
        ).astype(np.float64)
        valid_indices = np.flatnonzero(valid_labels).astype(np.int64)
        label_values = labels_global_arr[valid_indices]
        label_order = np.argsort(label_values, kind="stable")
        valid_indices_sorted = valid_indices[label_order]
        labels_sorted = label_values[label_order]
        split_points = np.flatnonzero(np.diff(labels_sorted)) + 1
        label_chunks = np.split(valid_indices_sorted, split_points)
        label_keys = np.split(labels_sorted, split_points)
        label_hit_indices = {
            int(keys[0]): np.asarray(indices, dtype=np.int64)
            for keys, indices in zip(label_keys, label_chunks)
            if int(keys[0]) >= 0
        }
    else:
        label_energy_sums = np.asarray([], dtype=np.float64)
        label_hit_indices: dict[int, np.ndarray] = {}

    if config.track_stage.enable_inline_leftover_absorption:
        t0 = time.perf_counter()
        if models is None:
            raise RuntimeError(
                "TrackStageConfig.enable_inline_leftover_absorption requires a ModelBundle. "
                "Pass models=... to run_track_shower_stage."
            )
        label_energies_all = {
            int(label): float(label_energy_sums[int(label)])
            for label in range(int(label_energy_sums.size))
            if float(label_energy_sums[int(label)]) > 0.0
        }
        leftover_state, leftover_prep_stats = prepare_leftover_absorption_state_v8(
            x=x,
            y=y,
            z=z,
            E=energy,
            hit_tpc_ids=hit_tpc_id,
            labels_global=labels_global_arr,
            cluster_to_tpcs=cluster_to_tpcs,
            label_info=label_info,
            label_energies=label_energies_all,
            expand_frac=config.track_stage.leftover_noise_expand_frac,
            capacity_fraction_mev=config.track_stage.leftover_capacity_fraction_mev,
            shower_absorb_max_hits=config.track_stage.leftover_shower_absorb_max_hits,
            huge_cluster_energy_mev=config.track_stage.leftover_huge_cluster_energy_mev,
            huge_cluster_absorb_max_hits=config.track_stage.leftover_huge_cluster_absorb_max_hits,
        )
        t_leftover_prep = time.perf_counter() - t0
        if verbose:
            print(
                "Inline leftover candidates: "
                f"parents={leftover_prep_stats.get('n_parents_with_candidates', 0)} | "
                f"noise hits={leftover_prep_stats.get('n_noise_hits', 0)} | "
                f"entries={leftover_prep_stats.get('n_candidate_entries', 0)}"
            )

    t0_initial = time.perf_counter()
    for clusterid in track_shower_labels:
        sorted_tpcs = np.sort(
            np.asarray(
                [int(tpc) for tpc in cluster_to_tpcs.get(int(clusterid), []) if int(tpc) < max_charge_tpc],
                dtype=np.int32,
            )
        )
        if sorted_tpcs.size == 0:
            continue
        if any((int(clusterid), int(tpc)) not in image_maps for tpc in sorted_tpcs):
            continue

        cluster_energy = float(label_energy_sums[int(clusterid)]) if int(clusterid) < int(label_energy_sums.size) else 0.0
        label_type = str(label_info.get(int(clusterid), {}).get("type", "track")).lower()
        use_support_mask = label_type == "track"

        t_fit_stack_start = time.perf_counter()
        full_image = fast_stack_images(image_maps, int(clusterid), sorted_tpcs)
        try:
            fit_image, fit_base, fit_actual, fit_std, fit_support_meta = stack_cluster_fit_inputs_v10_3(
                clusterid=int(clusterid),
                tpcids=sorted_tpcs,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=prediction.full_light_std_phase1,
                channel_support_cache=prediction.cluster_channel_support_cache,
                use_support_mask=use_support_mask,
                saturated_channel_cache=prediction.saturated_channel_cache,
            )
        except ValueError:
            t_initial_fit_stack += time.perf_counter() - t_fit_stack_start
            continue
        t_initial_fit_stack += time.perf_counter() - t_fit_stack_start

        t_scan_start = time.perf_counter()
        if str(config.track_stage.scan_mode).lower() in {"correlation", "unit", "unit_correlation", "fast_unit"}:
            shifts, errors = correlation_unit_likelihood_curve_with_base(
                fit_image,
                fit_base,
                fit_actual,
                search_range=config.track_stage.search_range,
                engine=config.track_stage.unit_scan_engine,
            )
        elif str(config.track_stage.scan_mode).lower() == "exact":
            shifts, errors = exact_likelihood_curve_with_base(
                fit_image,
                fit_base,
                fit_actual,
                fit_std,
                search_range=config.track_stage.search_range,
                adc_clip=config.track_stage.adc_clip,
            )
        else:
            raise ValueError("TrackStageConfig.scan_mode must be 'correlation', 'unit', 'fast_unit', or 'exact'.")
        t_initial_t0_scan += time.perf_counter() - t_scan_start

        raw_t0 = int(shifts[int(np.argmin(errors))])
        new_t0 = refine_t0_from_local_peak(
            raw_t0=raw_t0,
            actual_fit=fit_actual,
            base_fit=fit_base,
            t0_resolution=config.track_stage.t0_resolution,
            waveform_len=config.track_stage.waveform_len,
        )
        n_fit_channels = int(fit_image.shape[0])
        fit_meta_by_tpc = {int(item["tpcid"]): dict(item) for item in fit_support_meta}

        if config.track_stage.collect_scan_losses:
            cluster_full_scan_loss_dict[int(clusterid)] = {
                "clusterid": int(clusterid),
                "stage": "track",
                "mode": f"track_scan_{config.track_stage.scan_mode}",
                "tpcs": sorted_tpcs.tolist(),
                "energy": float(cluster_energy),
                "assigned": True,
                "scan_performed": True,
                "best_t0": int(new_t0),
                "best_t0_scan": int(raw_t0),
                "t0_grid": np.asarray(shifts, dtype=np.int32).copy(),
                "loss_curve": np.asarray(errors, dtype=np.float32).copy(),
                "n_fit_channels": int(n_fit_channels),
                "fit_support_meta": [dict(item) for item in fit_support_meta],
            }

        if verbose and config.track_stage.print_track_assignments:
            print(
                f"  Label {clusterid:4d} | type={label_type:<6} | "
                f"TPCs={sorted_tpcs.tolist()} | t0={new_t0} | fit_ch={n_fit_channels}"
            )

        t_apply_start = time.perf_counter()
        cluster_hit_indices = label_hit_indices.get(int(clusterid))
        if cluster_hit_indices is not None:
            hit_timestamps[cluster_hit_indices] = np.float32(new_t0)
        else:
            hit_timestamps[labels_global_arr == int(clusterid)] = np.float32(new_t0)

        for tpc in sorted_tpcs:
            t0_candidates, flash_cluster_received_by_tpc, flash_row = mark_flash_cluster_assignment(
                t0_candidates,
                flash_cluster_received_by_tpc,
                tpc=int(tpc),
                t0=float(new_t0),
                resolution_ticks=float(config.track_stage.t0_resolution),
                max_t0=float(config.track_stage.search_range),
                prefer_existing_true=False,
                clusterid=int(clusterid),
                stage=f"track_scan_{config.track_stage.scan_mode}",
            )
            flash_cluster_canonicalization_log.append(flash_row)
            fit_meta = dict(fit_meta_by_tpc.get(int(tpc), {}))
            assignment_info[(int(clusterid), int(tpc))] = {
                "stage": "track",
                "mode": f"track_scan_{config.track_stage.scan_mode}",
                "t0": float(new_t0),
                "energy": float(cluster_energy),
                "assigned": True,
                "backbone_type": str(label_type),
                "n_fit_channels": int(fit_meta.get("n_fit_channels", full_light_waveform.shape[1])),
                "channel_selection_spec": list(fit_meta.get("channel_selection_spec", [])),
                "selected_fraction": float(fit_meta.get("selected_fraction", 1.0)),
                "dominant_fraction": float(fit_meta.get("dominant_fraction", 1.0)),
                "support_mode": str(fit_meta.get("support_mode", "all_channels")),
            }

        shifted_full = timeinterpolation(full_image, shift=float(new_t0), baseline=0).astype(np.float32)
        base_image = image_updater(shifted_full, base_image, sorted_tpcs)
        base_image = np.clip(base_image, None, config.track_stage.adc_clip)
        t_initial_apply += time.perf_counter() - t_apply_start

        if label_type == "track":
            track_channel_diagnostics.append(
                {
                    "clusterid": int(clusterid),
                    "energy": float(cluster_energy),
                    "n_fit_channels": int(n_fit_channels),
                    "n_tpcs": int(sorted_tpcs.size),
                    "tpcs": sorted_tpcs.tolist(),
                    "fit_support_meta": [dict(item) for item in fit_support_meta],
                }
            )

        if config.track_stage.enable_inline_leftover_absorption and leftover_state is not None:
            t0_absorb = time.perf_counter()
            absorb_log = try_absorb_leftovers_for_parent_v8(
                parent=int(clusterid),
                absorption_state=leftover_state,
                x=x,
                y=y,
                z=z,
                E=energy,
                hit_tpc_ids=hit_tpc_id,
                labels_global=labels_global_arr,
                labels_with_leftovers=labels_global_with_leftovers,
                absorbed_hit_parent=absorbed_hit_parent,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=prediction.full_light_std_phase1,
                hit_timestamps=hit_timestamps,
                assignment_info=assignment_info,
                model=models.light_model,
                template=models.waveform_template,
                channel_support_cache=prediction.cluster_channel_support_cache,
                target_scale=config.prediction.target_scale,
                batch_size=config.track_stage.leftover_noisy_batch_size,
                raw_clip=config.prediction.raw_clip,
                min_prediction_threshold=config.track_stage.leftover_noisy_min_prediction_threshold,
                device_policy=config.track_stage.leftover_noisy_device_policy,
                support_light_fraction=config.support.light_fraction,
                support_max_gap=config.support.max_gap,
                improvement_eps=config.track_stage.leftover_absorption_improvement_eps,
                adc_clip=config.track_stage.adc_clip,
                saturated_channel_cache=prediction.saturated_channel_cache,
            )
            t_leftover_absorb += time.perf_counter() - t0_absorb
            if absorb_log is not None:
                leftover_absorption_log.append(dict(absorb_log))
    t_initial_loop = time.perf_counter() - t0_initial

    track_second_pass_log: list[dict[str, Any]] = []
    track_second_pass_stats: dict[str, Any] = {"n_tracks_rescanned": 0, "n_tracks_changed_t0": 0}
    if config.track_stage.enable_second_pass_rescan:
        t0 = time.perf_counter()
        (
            base_image,
            hit_timestamps,
            t0_candidates,
            assignment_info,
            track_second_pass_scan_updates,
            track_second_pass_log,
            track_second_pass_stats,
        ) = run_track_second_pass_rescan_v10_3(
            track_labels=track_shower_labels,
            cluster_to_tpcs=cluster_to_tpcs,
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=prediction.full_light_std_phase1,
            labels_global=labels_global_arr,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            label_info=label_info,
            channel_support_cache=prediction.cluster_channel_support_cache,
            cluster_full_scan_loss_dict=cluster_full_scan_loss_dict,
            absorbed_hit_parent=absorbed_hit_parent,
            search_range=config.track_stage.search_range,
            adc_clip=config.track_stage.adc_clip,
            t0_resolution=config.track_stage.t0_resolution,
            waveform_len=config.track_stage.waveform_len,
            collect_scan_losses=config.track_stage.collect_scan_losses,
            energies=energy,
            hit_tpc_ids=hit_tpc_id,
            min_shower_energy_mev=config.track_stage.track_guard_min_shower_energy_mev,
            min_clean_t0_separation_ticks=config.track_stage.track_guard_min_t0_separation_ticks,
            clean_worsen_tolerance_norm=config.track_stage.track_guard_clean_worsen_tolerance_norm,
            saturated_channel_cache=prediction.saturated_channel_cache,
        )
        if config.track_stage.collect_scan_losses:
            cluster_full_scan_loss_dict.update(track_second_pass_scan_updates)
        t0_candidates, flash_cluster_received_by_tpc, flash_rows = rebuild_flash_cluster_flags_from_assignments(
            t0_candidates,
            assignment_info,
            resolution_ticks=float(config.track_stage.t0_resolution),
            max_t0=float(config.track_stage.search_range),
            initial_flags=flash_cluster_received_by_tpc,
            prefer_existing_true=False,
        )
        flash_cluster_canonicalization_log.extend(flash_rows)
        t_rescan = time.perf_counter() - t0

    track_swap_log: list[dict[str, Any]] = []
    track_swap_stats: dict[str, Any] = {
        "candidate_pairs": 0,
        "candidate_stats": {},
        "accepted_swaps": 0,
        "swapped_clusters": [],
        "swap_log": [],
    }
    if config.track_stage.enable_overlap_swap:
        t0 = time.perf_counter()
        track_swap_candidate_rows, track_swap_candidate_stats = build_track_swap_candidates_from_hits_v10_7_focused(
            track_labels=track_shower_labels,
            labels_global=labels_global_arr,
            hit_tpc_ids=hit_tpc_id,
            x=x,
            y=y,
            z=z,
            energies=energy,
            label_info=label_info,
            min_energy_ratio=config.track_stage.swap_min_energy_ratio,
            min_shared_tpcs=config.track_stage.swap_min_shared_tpcs,
            max_tpc_sym_diff=config.track_stage.swap_max_tpc_sym_diff,
            max_angle_deg=config.track_stage.swap_max_angle_deg,
            max_shared_yz_dist_cm=config.track_stage.swap_max_shared_yz_dist_cm,
            max_endpoint_yz_dist_cm=config.track_stage.swap_max_endpoint_yz_dist_cm,
        )
        n_swap_candidates = int(len(track_swap_candidate_rows))
        if len(track_swap_candidate_rows) > 0:
            (
                base_image,
                hit_timestamps,
                t0_candidates,
                assignment_info,
                track_swap_log,
                track_swap_stats,
            ) = run_track_overlap_swap_rescue_v10_7_focused(
                track_labels=track_shower_labels,
                cluster_to_tpcs=cluster_to_tpcs,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=prediction.full_light_std_phase1,
                labels_global=labels_global_arr,
                hit_tpc_ids=hit_tpc_id,
                x=x,
                y=y,
                z=z,
                energies=energy,
                hit_timestamps=hit_timestamps,
                t0_candidates=t0_candidates,
                assignment_info=assignment_info,
                label_info=label_info,
                channel_support_cache=prediction.cluster_channel_support_cache,
                protected_track_shower_timestamps=np.full(labels_global_arr.shape[0], np.nan, dtype=np.float32),
                absorbed_hit_parent=absorbed_hit_parent,
                min_energy_ratio=config.track_stage.swap_min_energy_ratio,
                min_shared_tpcs=config.track_stage.swap_min_shared_tpcs,
                max_tpc_sym_diff=config.track_stage.swap_max_tpc_sym_diff,
                max_angle_deg=config.track_stage.swap_max_angle_deg,
                max_shared_yz_dist_cm=config.track_stage.swap_max_shared_yz_dist_cm,
                max_endpoint_yz_dist_cm=config.track_stage.swap_max_endpoint_yz_dist_cm,
                min_t0_separation_ticks=config.track_stage.swap_min_t0_separation_ticks,
                max_passes=config.track_stage.swap_max_passes,
                improvement_eps=config.track_stage.swap_improvement_eps,
                adc_clip=config.track_stage.adc_clip,
                search_range=config.track_stage.search_range,
                lock_swapped_clusters=config.track_stage.swap_lock_swapped_clusters,
            )
            t0_candidates, flash_cluster_received_by_tpc, flash_rows = rebuild_flash_cluster_flags_from_assignments(
                t0_candidates,
                assignment_info,
                resolution_ticks=float(config.track_stage.t0_resolution),
                max_t0=float(config.track_stage.search_range),
                initial_flags=flash_cluster_received_by_tpc,
                prefer_existing_true=False,
            )
            flash_cluster_canonicalization_log.extend(flash_rows)
        track_swap_stats["candidate_pairs"] = int(n_swap_candidates)
        track_swap_stats["candidate_stats"] = dict(track_swap_candidate_stats or {})
        t_swap = time.perf_counter() - t0

    track_t0_fine_correction_log: list[dict[str, Any]] = []
    track_t0_fine_correction_stats: dict[str, Any] = {"n_tracks_scanned": 0, "n_tracks_changed": 0}
    if config.track_stage.enable_fine_correction:
        t0 = time.perf_counter()
        (
            base_image,
            hit_timestamps,
            t0_candidates,
            assignment_info,
            track_t0_fine_correction_log,
            track_t0_fine_correction_stats,
        ) = run_track_t0_fine_correction_v13_4(
            track_labels=track_shower_labels,
            cluster_to_tpcs=cluster_to_tpcs,
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            labels_global=labels_global_arr,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            label_info=label_info,
            channel_support_cache=prediction.cluster_channel_support_cache,
            saturated_channel_cache=prediction.saturated_channel_cache,
            max_charge_tpc=max_charge_tpc,
            adc_clip=config.track_stage.adc_clip,
            grid_offsets=config.track_stage.fine_grid_offsets,
            improvement_eps=config.track_stage.fine_improvement_eps,
            verbose=False,
        )
        t0_candidates, flash_cluster_received_by_tpc, flash_rows = rebuild_flash_cluster_flags_from_assignments(
            t0_candidates,
            assignment_info,
            resolution_ticks=float(config.track_stage.t0_resolution),
            max_t0=float(config.track_stage.search_range),
            initial_flags=flash_cluster_received_by_tpc,
            prefer_existing_true=False,
        )
        flash_cluster_canonicalization_log.extend(flash_rows)
        t_fine = time.perf_counter() - t0

    final_backbone_t0_rows = _collect_final_backbone_t0_rows(
        track_shower_labels=track_shower_labels,
        cluster_to_tpcs=cluster_to_tpcs,
        label_info=label_info,
        label_energy_sums=label_energy_sums,
        label_hit_indices=label_hit_indices,
        labels_global=labels_global_arr,
        hit_timestamps=hit_timestamps,
        max_charge_tpc=max_charge_tpc,
    )
    modified_flash_table_by_tpc, flash_table_amendment_log = _amend_flash_table_with_track_t0s(
        raw_flash_table_by_tpc=raw_flash_table_by_tpc,
        final_backbone_rows=final_backbone_t0_rows,
        max_charge_tpc=max_charge_tpc,
        amend_window_ticks=config.track_stage.flash_amend_window_ticks,
        max_t0=config.track_stage.search_range,
    )
    t0_candidates = _rebuild_phase1_t0_candidates(
        final_backbone_rows=final_backbone_t0_rows,
        amended_flash_table_by_tpc=modified_flash_table_by_tpc,
        max_charge_tpc=max_charge_tpc,
        min_sep=max(int(config.track_stage.flash_candidate_min_sep_ticks), int(config.track_stage.t0_resolution)),
        max_t0=config.track_stage.search_range,
    )
    t0_candidates, flash_cluster_received_by_tpc, flash_rows = rebuild_flash_cluster_flags_from_assignments(
        t0_candidates,
        assignment_info,
        resolution_ticks=float(config.track_stage.t0_resolution),
        max_t0=float(config.track_stage.search_range),
        initial_flags=None,
        prefer_existing_true=False,
    )
    flash_cluster_canonicalization_log.extend(flash_rows)
    flash_rows_out = flash_cluster_table_rows(t0_candidates, flash_cluster_received_by_tpc)
    modified_flash_table_by_tpc = {
        int(tpc): [int(round(float(v))) if abs(float(v) - round(float(v))) < 1e-6 else float(v) for v in values]
        for tpc, values in enumerate(t0_candidates)
        if len(values) > 0
    }

    hit_t0_export = np.where(np.isfinite(hit_timestamps), hit_timestamps, -1.0).astype(np.float32)
    leftover_absorption_stats = {
        "n_absorption_records": int(len(leftover_absorption_log)),
        "n_absorbed_hits": int(np.count_nonzero(absorbed_hit_parent >= 0)),
        "accepted_parents": [int(item["clusterid"]) for item in leftover_absorption_log],
    }
    flash_seed_stats.update(
        {
            "n_raw_flash_tpcs": int(len(raw_flash_table_by_tpc)),
            "n_raw_flash_t0s": int(sum(len(v) for v in raw_flash_table_by_tpc.values())),
            "n_amended_flash_tpcs": int(len(modified_flash_table_by_tpc)),
            "n_amended_flash_t0s": int(sum(len(v) for v in modified_flash_table_by_tpc.values())),
            "n_flash_amendments": int(len(flash_table_amendment_log)),
            "flash_amend_window_ticks": int(config.track_stage.flash_amend_window_ticks),
            "n_cluster_received_flash_t0s": int(sum(sum(bool(v) for v in values) for values in flash_cluster_received_by_tpc)),
        }
    )
    stage_timings = {
        "leftover_prep_s": float(t_leftover_prep),
        "initial_track_scan_s": float(t_initial_loop),
        "initial_fit_stack_s": float(t_initial_fit_stack),
        "initial_t0_scan_s": float(t_initial_t0_scan),
        "initial_apply_s": float(t_initial_apply),
        "inline_leftover_absorption_s": float(t_leftover_absorb),
        "track_rescan_s": float(t_rescan),
        "track_swap_s": float(t_swap),
        "fine_correction_s": float(t_fine),
        "total_s": float(time.perf_counter() - t_total),
    }

    return TrackStageResult(
        hit_t0=hit_timestamps,
        hit_t0_export=hit_t0_export,
        base_image=np.asarray(base_image, dtype=np.float32),
        t0_candidates_by_tpc=t0_candidates,
        raw_flash_table_by_tpc={
            int(tpc): [int(v) for v in values]
            for tpc, values in sorted(raw_flash_table_by_tpc.items())
        },
        modified_flash_table_by_tpc=modified_flash_table_by_tpc,
        flash_cluster_received_by_tpc=[
            [bool(v) for v in values]
            for values in flash_cluster_received_by_tpc
        ],
        flash_cluster_table_rows=[dict(row) for row in flash_rows_out],
        flash_cluster_canonicalization_log=[dict(row) for row in flash_cluster_canonicalization_log],
        flash_table_amendment_log=[dict(item) for item in flash_table_amendment_log],
        flash_seed_stats=dict(flash_seed_stats),
        assignment_info=assignment_info,
        track_channel_diagnostics=track_channel_diagnostics,
        cluster_full_scan_loss_dict=cluster_full_scan_loss_dict,
        track_second_pass_log=[dict(item) for item in track_second_pass_log],
        track_second_pass_stats=dict(track_second_pass_stats),
        track_swap_log=[dict(item) for item in track_swap_log],
        track_swap_stats=dict(track_swap_stats),
        track_t0_fine_correction_log=[dict(item) for item in track_t0_fine_correction_log],
        track_t0_fine_correction_stats=dict(track_t0_fine_correction_stats),
        labels_global_with_leftovers=np.asarray(labels_global_with_leftovers, dtype=np.int32),
        absorbed_hit_parent=np.asarray(absorbed_hit_parent, dtype=np.int32),
        leftover_absorption_log=[dict(item) for item in leftover_absorption_log],
        leftover_absorption_stats=dict(leftover_absorption_stats),
        leftover_state=leftover_state,
        leftover_prep_stats=dict(leftover_prep_stats),
        stage_timings=dict(stage_timings),
    )


__all__ = [
    "TrackStageResult",
    "run_track_shower_stage",
    "correlation_unit_likelihood_curve_with_base",
    "exact_likelihood_curve_with_base",
]
