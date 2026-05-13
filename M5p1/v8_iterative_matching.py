from __future__ import annotations

from typing import Any

import numpy as np

try:
    from v3_2_global_matching import (
        _assign_cluster_at_t0,
        _build_scan_loss_entry,
        _full_scan_assign,
        _loss_matrix_single_tpc,
        _mark_cluster_unassigned,
        _scan_best_shift_multi,
        _shift_block,
        append_candidate_t0,
        compute_error_metric,
    )
    from v4_hierarchical_matching import rebalance_step4_clusters_v4
    from v8_leftover_absorption import try_absorb_leftovers_for_parent_v8
    from v12_saturation_mask import materialize_support_entry_v12
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v3_2_global_matching import (
        _assign_cluster_at_t0,
        _build_scan_loss_entry,
        _full_scan_assign,
        _loss_matrix_single_tpc,
        _mark_cluster_unassigned,
        _scan_best_shift_multi,
        _shift_block,
        append_candidate_t0,
        compute_error_metric,
    )
    from M5p1.v4_hierarchical_matching import rebalance_step4_clusters_v4
    from M5p1.v8_leftover_absorption import try_absorb_leftovers_for_parent_v8
    from M5p1.v12_saturation_mask import materialize_support_entry_v12


def _maybe_absorb_leftovers_after_assignment(
    *,
    clusterid: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    leftover_absorption_context: dict[str, Any] | None,
    adc_clip: float,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if leftover_absorption_context is None:
        return None

    labels_with_leftovers = leftover_absorption_context.get("labels_with_leftovers")
    if labels_with_leftovers is None:
        return None

    log = try_absorb_leftovers_for_parent_v8(
        parent=int(clusterid),
        absorption_state=leftover_absorption_context.get("state"),
        x=np.asarray(leftover_absorption_context["x"], dtype=np.float32),
        y=np.asarray(leftover_absorption_context["y"], dtype=np.float32),
        z=np.asarray(leftover_absorption_context["z"], dtype=np.float32),
        E=np.asarray(leftover_absorption_context["E"], dtype=np.float32),
        hit_tpc_ids=np.asarray(leftover_absorption_context["hit_tpc_ids"], dtype=np.int32),
        labels_global=np.asarray(labels_global, dtype=int),
        labels_with_leftovers=np.asarray(labels_with_leftovers, dtype=int),
        absorbed_hit_parent=leftover_absorption_context.get("absorbed_hit_parent"),
        image_maps=image_maps,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        hit_timestamps=hit_timestamps,
        assignment_info=assignment_info,
        model=leftover_absorption_context["model"],
        template=leftover_absorption_context.get("template"),
        channel_support_cache=channel_support_cache,
        target_scale=float(leftover_absorption_context.get("target_scale", 1e-3)),
        batch_size=int(leftover_absorption_context.get("batch_size", 4)),
        raw_clip=tuple(leftover_absorption_context.get("raw_clip", (0.0, float(adc_clip)))),
        min_prediction_threshold=leftover_absorption_context.get("min_prediction_threshold", 100.0),
        device_policy=str(leftover_absorption_context.get("device_policy", "auto")),
        support_light_fraction=float(leftover_absorption_context.get("support_light_fraction", 0.90)),
        support_max_gap=int(leftover_absorption_context.get("support_max_gap", 2)),
        improvement_eps=float(leftover_absorption_context.get("improvement_eps", 0.0)),
        adc_clip=float(adc_clip),
        saturated_channel_cache=saturated_channel_cache,
    )
    if log is not None:
        leftover_absorption_context.setdefault("absorption_log", []).append(log)
    return log


def _full_scan_primary_clusters(
    *,
    cluster_ids: list[int],
    active_cluster_tpcs: dict[int, np.ndarray],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energies: dict[int, float],
    search_range: int,
    adc_clip: float,
    collect_scan_losses: bool,
    assignment_improvement_eps: float,
    stage_name: str,
    accepted_mode: str,
    rejected_mode: str,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    leftover_absorption_context: dict[str, Any] | None,
    saturated_channel_cache: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], set[int]]:
    logs: list[dict[str, Any]] = []
    scan_updates: dict[int, dict[str, Any]] = {}
    assigned_clusters: set[int] = set()

    ordered = sorted(
        [int(cid) for cid in cluster_ids],
        key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
        reverse=True,
    )

    for clusterid in ordered:
        tpcs = active_cluster_tpcs.get(int(clusterid))
        if tpcs is None or len(tpcs) == 0:
            continue
        cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
        accepted, log, scan_entry = _full_scan_assign(
            int(clusterid),
            np.asarray(tpcs, dtype=int),
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energy=cluster_energy,
            stage=str(stage_name),
            accepted_mode=str(accepted_mode),
            forced_mode=f"{accepted_mode}_forced",
            rejected_mode=str(rejected_mode),
            assignment_improvement_eps=float(assignment_improvement_eps),
            search_range=int(search_range),
            adc_clip=float(adc_clip),
            t0_resolution=5,
            pulse_peak_tick=105,
            collect_scan_losses=collect_scan_losses,
            channel_support_cache=channel_support_cache,
            channel_saturation_cache=saturated_channel_cache,
        )
        logs.append(log)
        if scan_entry is not None:
            scan_updates[int(clusterid)] = scan_entry
        if accepted:
            assigned_clusters.add(int(clusterid))
            _maybe_absorb_leftovers_after_assignment(
                clusterid=int(clusterid),
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                channel_support_cache=channel_support_cache,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                assignment_info=assignment_info,
                leftover_absorption_context=leftover_absorption_context,
                adc_clip=float(adc_clip),
                saturated_channel_cache=saturated_channel_cache,
            )

    return logs, scan_updates, assigned_clusters


def _normalize_delta_row(
    delta_row: np.ndarray,
    *,
    cluster_energy: float,
    anchor_energy: float,
) -> np.ndarray:
    row = np.asarray(delta_row, dtype=np.float32)
    scale = float(np.max(np.abs(row)))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    safe_anchor = max(float(anchor_energy), 1e-6)
    safe_cluster = max(float(cluster_energy), 1e-6)
    energy_factor = float(np.sqrt(safe_cluster / safe_anchor))
    energy_factor = float(np.clip(energy_factor, 0.25, 1.0))
    return (row / float(scale)) * float(energy_factor)


def _push_back_align_t0(
    *,
    cluster_wave: np.ndarray,
    actual_block: np.ndarray,
    base_block: np.ndarray,
    objective_t0: int,
    max_backtrack_ticks: int,
    search_range: int,
) -> int:
    if int(max_backtrack_ticks) <= 0:
        return int(np.clip(objective_t0, 0, int(search_range)))

    cluster_profile = np.sum(np.asarray(cluster_wave, dtype=np.float32), axis=0)
    residual_profile = np.sum(
        np.clip(
            np.asarray(actual_block, dtype=np.float32) - np.asarray(base_block, dtype=np.float32),
            0.0,
            None,
        ),
        axis=0,
    )
    if cluster_profile.size == 0 or residual_profile.size == 0:
        return int(np.clip(objective_t0, 0, int(search_range)))

    cluster_peak_tick = int(np.argmax(cluster_profile))
    expected_peak = int(objective_t0) + int(cluster_peak_tick)
    s_start = max(0, expected_peak - int(max_backtrack_ticks))
    s_end = min(residual_profile.shape[0], expected_peak + 1)
    if s_end <= s_start:
        return int(np.clip(objective_t0, 0, int(search_range)))

    local_peak = int(np.argmax(residual_profile[s_start:s_end]))
    aligned_peak = s_start + local_peak
    aligned_t0 = int(objective_t0) + int(aligned_peak - expected_peak)
    return int(np.clip(aligned_t0, 0, int(search_range)))


def _score_single_tpc_cluster_t0(
    *,
    cluster_wave: np.ndarray,
    base_block: np.ndarray,
    actual_block: np.ndarray,
    error_block: np.ndarray,
    t0: int,
    adc_clip: float,
) -> float:
    shifted = _shift_block(np.asarray(cluster_wave, dtype=np.float32)[None, :, :], int(t0))[0]
    candidate_model = np.clip(np.asarray(base_block, dtype=np.float32) + shifted, None, float(adc_clip))
    return float(
        compute_error_metric(
            candidate_model,
            np.asarray(actual_block, dtype=np.float32),
            np.asarray(error_block, dtype=np.float32),
        )
    )


def _get_single_tpc_support(
    *,
    clusterid: int,
    tpcid: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    full_wave = np.asarray(image_maps[(int(clusterid), int(tpcid))], dtype=np.float32)
    entry = None if channel_support_cache is None else channel_support_cache.get((int(clusterid), int(tpcid)))
    support_entry = materialize_support_entry_v12(
        waveform=full_wave,
        tpcid=int(tpcid),
        base_entry=entry,
        saturated_channel_cache=saturated_channel_cache,
    )
    channel_indices = np.asarray(support_entry["channel_indices"], dtype=np.int32)
    selected_wave = np.asarray(support_entry["selected_waveform"], dtype=np.float32)

    support_meta = {
        "channel_indices": channel_indices,
        "channel_labels": list(support_entry.get("channel_labels", [])),
        "channel_selection_spec": list(support_entry.get("channel_selection_spec", [])),
        "selected_fraction": float(support_entry.get("selected_fraction", 1.0)),
        "dominant_fraction": float(support_entry.get("dominant_fraction", 1.0)),
        "support_mode": str(support_entry.get("support_mode", "masked_channels")),
        "n_channels_preclip": int(support_entry.get("n_channels_preclip", channel_indices.size)),
        "n_saturated_channels_dropped": int(support_entry.get("n_saturated_channels_dropped", 0)),
        "dropped_channel_labels": list(support_entry.get("dropped_channel_labels", [])),
        "saturation_fallback_to_all_unsaturated": bool(
            support_entry.get("saturation_fallback_to_all_unsaturated", False)
        ),
    }
    return full_wave, selected_wave, channel_indices, support_meta


def _extract_selected_tpc_blocks(
    *,
    tpcid: int,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.asarray(channel_indices, dtype=np.int32)
    return (
        np.asarray(base_image[int(tpcid), idx, :], dtype=np.float32),
        np.asarray(full_light_waveform[int(tpcid), idx, :], dtype=np.float32),
        np.asarray(full_light_std[int(tpcid), idx, :], dtype=np.float32),
    )


def _scan_cluster_on_current_base(
    *,
    clusterid: int,
    tpcid: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    saturated_channel_cache: dict[int, np.ndarray] | None,
    search_range: int,
    adc_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int, float, np.ndarray, float]:
    full_wave, selected_wave, channel_indices, support_meta = _get_single_tpc_support(
        clusterid=int(clusterid),
        tpcid=int(tpcid),
        image_maps=image_maps,
        channel_support_cache=channel_support_cache,
        saturated_channel_cache=saturated_channel_cache,
    )
    cluster_block = full_wave[None, :, :]
    selected_block = selected_wave[None, :, :]
    base_selected, actual_selected, error_selected = _extract_selected_tpc_blocks(
        tpcid=int(tpcid),
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        channel_indices=channel_indices,
    )
    current_error = compute_error_metric(base_selected, actual_selected, error_selected)
    best_t0, best_score, loss_curve = _scan_best_shift_multi(
        selected_block,
        base_selected[None, :, :],
        actual_selected[None, :, :],
        error_selected[None, :, :],
        search_range=int(search_range),
        adc_clip=float(adc_clip),
        return_curve=True,
    )
    return (
        cluster_block,
        selected_wave,
        channel_indices,
        support_meta,
        int(best_t0),
        float(best_score),
        np.asarray(loss_curve, dtype=np.float32),
        float(current_error),
    )


def _masked_loss_matrix_single_tpc(
    *,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    saturated_channel_cache: dict[int, np.ndarray] | None,
    tpcid: int,
    clusters: np.ndarray,
    placed_mask: np.ndarray,
    t0_candidates: list[int],
    adc_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    remaining_clusters = np.asarray(clusters[~placed_mask], dtype=int)
    n_remaining = int(len(remaining_clusters))
    n_candidates = int(len(t0_candidates))

    loss_matrix = np.full((n_remaining, n_candidates), np.inf, dtype=np.float32)
    current_errors = np.full(n_remaining, np.inf, dtype=np.float32)
    if n_remaining == 0 or n_candidates == 0:
        return loss_matrix, current_errors, remaining_clusters

    for row_idx, clusterid in enumerate(remaining_clusters):
        _, selected_wave, channel_indices, _ = _get_single_tpc_support(
            clusterid=int(clusterid),
            tpcid=int(tpcid),
            image_maps=image_maps,
            channel_support_cache=channel_support_cache,
            saturated_channel_cache=saturated_channel_cache,
        )
        base_selected, actual_selected, error_selected = _extract_selected_tpc_blocks(
            tpcid=int(tpcid),
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_indices=channel_indices,
        )
        current_errors[row_idx] = float(compute_error_metric(base_selected, actual_selected, error_selected))

        selected_block = np.asarray(selected_wave, dtype=np.float32)[None, :, :]
        for col_idx, t0 in enumerate(t0_candidates):
            shifted = _shift_block(selected_block, int(t0))[0]
            candidate_model = np.clip(base_selected + shifted, None, float(adc_clip))
            loss_matrix[row_idx, col_idx] = float(
                compute_error_metric(candidate_model, actual_selected, error_selected)
            )

    return loss_matrix, current_errors, remaining_clusters


def _try_t0_expansion_for_cluster(
    *,
    clusterid: int,
    tpcid: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    saturated_channel_cache: dict[int, np.ndarray] | None,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energy: float,
    search_range: int,
    adc_clip: float,
    collect_scan_losses: bool,
    full_scan_assign_eps: float,
    backward_peak_align_ticks: int,
    leftover_absorption_context: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[int, dict[str, Any]]]:
    scan_updates: dict[int, dict[str, Any]] = {}
    cluster_block, selected_wave, channel_indices, support_meta, best_t0, best_score, loss_curve, current_error = _scan_cluster_on_current_base(
        clusterid=int(clusterid),
        tpcid=int(tpcid),
        image_maps=image_maps,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        channel_support_cache=channel_support_cache,
        saturated_channel_cache=saturated_channel_cache,
        search_range=int(search_range),
        adc_clip=float(adc_clip),
    )
    base_tpc, actual_tpc, error_tpc = _extract_selected_tpc_blocks(
        tpcid=int(tpcid),
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        channel_indices=channel_indices,
    )

    aligned_t0 = _push_back_align_t0(
        cluster_wave=selected_wave,
        actual_block=actual_tpc,
        base_block=base_tpc,
        objective_t0=int(best_t0),
        max_backtrack_ticks=int(backward_peak_align_ticks),
        search_range=int(search_range),
    )
    aligned_score = float(best_score)
    if int(aligned_t0) != int(best_t0):
        aligned_score = _score_single_tpc_cluster_t0(
            cluster_wave=selected_wave,
            base_block=base_tpc,
            actual_block=actual_tpc,
            error_block=error_tpc,
            t0=int(aligned_t0),
            adc_clip=float(adc_clip),
        )

    improvement = float(current_error - float(aligned_score))
    appended = append_candidate_t0(t0_candidates[int(tpcid)], int(aligned_t0), max_t0=int(search_range))

    if collect_scan_losses:
        scan_updates[int(clusterid)] = _build_scan_loss_entry(
            clusterid=int(clusterid),
            stage="iterative_t0_expansion",
            mode="iterative_t0_expansion",
            tpcs=[int(tpcid)],
            energy=float(cluster_energy),
            best_t0=int(aligned_t0),
            assigned=False,
            search_range=int(search_range),
            loss_curve=loss_curve,
            best_t0_scan=int(best_t0),
        )

    if appended:
        assignment_info[(int(clusterid), int(tpcid))] = {
            "stage": "iterative_t0_expansion",
            "mode": "iterative_t0_expansion_added",
            "t0": np.nan,
            "energy": float(cluster_energy),
            "assigned": False,
            "suggested_t0": float(aligned_t0),
            "objective_t0": float(best_t0),
            "error_after": float(aligned_score),
            "improvement": float(improvement),
            "n_fit_channels": int(channel_indices.size),
        }
        return (
            "expanded",
            {
                "clusterid": int(clusterid),
                "tpcs": [int(tpcid)],
                "energy": float(cluster_energy),
                "assigned": False,
                "mode": "iterative_t0_expansion_added",
                "label": "iterative_t0_expansion",
                "t0": int(aligned_t0),
                "objective_t0": int(best_t0),
                "improvement": float(improvement),
                "n_fit_channels": int(channel_indices.size),
            },
            scan_updates,
        )

    if improvement > float(full_scan_assign_eps):
        _assign_cluster_at_t0(
            int(clusterid),
            np.asarray([int(tpcid)], dtype=int),
            int(aligned_t0),
            cluster_block=cluster_block,
            base_image=base_image,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energy=float(cluster_energy),
            mode="iterative_full_scan_assign",
            stage="iterative_full_scan_assign",
            error_after=float(aligned_score),
            improvement=float(improvement),
            adc_clip=float(adc_clip),
            max_t0=int(search_range),
        )
        assignment_info[(int(clusterid), int(tpcid))]["objective_t0"] = float(best_t0)
        assignment_info[(int(clusterid), int(tpcid))]["n_fit_channels"] = int(channel_indices.size)
        _maybe_absorb_leftovers_after_assignment(
            clusterid=int(clusterid),
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_support_cache=channel_support_cache,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            assignment_info=assignment_info,
            leftover_absorption_context=leftover_absorption_context,
            adc_clip=float(adc_clip),
            saturated_channel_cache=saturated_channel_cache,
        )
        return (
            "assigned",
            {
                "clusterid": int(clusterid),
                "tpcs": [int(tpcid)],
                "energy": float(cluster_energy),
                "assigned": True,
                "mode": "iterative_full_scan_assign",
                "label": "iterative_t0_expansion",
                "t0": int(aligned_t0),
                "objective_t0": int(best_t0),
                "improvement": float(improvement),
                "n_fit_channels": int(channel_indices.size),
            },
            scan_updates,
        )

    _mark_cluster_unassigned(
        int(clusterid),
        np.asarray([int(tpcid)], dtype=int),
        labels_global=labels_global,
        hit_timestamps=hit_timestamps,
        assignment_info=assignment_info,
        unassigned_by_tpc=unassigned_by_tpc,
        cluster_energy=float(cluster_energy),
        mode="iterative_t0_expansion_unassigned",
        stage="iterative_t0_expansion",
        candidate_t0=int(aligned_t0),
        candidate_error=float(aligned_score),
        improvement=float(improvement),
    )
    assignment_info[(int(clusterid), int(tpcid))]["objective_t0"] = float(best_t0)
    assignment_info[(int(clusterid), int(tpcid))]["n_fit_channels"] = int(channel_indices.size)
    return (
        "unassigned",
        {
            "clusterid": int(clusterid),
            "tpcs": [int(tpcid)],
            "energy": float(cluster_energy),
            "assigned": False,
            "mode": "iterative_t0_expansion_unassigned",
            "label": "iterative_t0_expansion",
            "t0": int(aligned_t0),
            "objective_t0": int(best_t0),
            "improvement": float(improvement),
            "n_fit_channels": int(channel_indices.size),
        },
        scan_updates,
    )


def _iterative_band_matrix_assign_tpc(
    *,
    tpcid: int,
    cluster_ids: list[int],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    saturated_channel_cache: dict[int, np.ndarray] | None,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energies: dict[int, float],
    band_fraction: float,
    positive_row_margin: float,
    matrix_worsen_tolerance_norm: float,
    search_range: int,
    adc_clip: float,
    collect_scan_losses: bool,
    full_scan_assign_eps: float,
    backward_peak_align_ticks: int,
    leftover_absorption_context: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    scan_updates: dict[int, dict[str, Any]] = {}
    assigned_clusters: list[int] = []
    matrix_assigned_clusters: list[int] = []
    expanded_clusters: list[int] = []
    full_scan_assigned_clusters: list[int] = []
    full_scan_unassigned_clusters: list[int] = []
    band_anchor_sequence: list[int] = []

    remaining: set[int] = {int(cid) for cid in cluster_ids}
    exhausted_expansion: set[int] = set()

    actual_tpc = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
    error_tpc = np.asarray(full_light_std[int(tpcid)], dtype=np.float32)

    while remaining:
        anchor_cluster = max(
            remaining,
            key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
        )
        anchor_energy = float(cluster_energies.get(int(anchor_cluster), 0.0))
        band_anchor_sequence.append(int(anchor_cluster))

        band_clusters = sorted(
            [
                int(cid)
                for cid in remaining
                if float(cluster_energies.get(int(cid), 0.0)) >= float(band_fraction) * float(anchor_energy)
            ],
            key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
            reverse=True,
        )

        candidate_grid = sorted(int(t0) for t0 in t0_candidates[int(tpcid)])
        if len(candidate_grid) == 0:
            action, log, new_scan_updates = _try_t0_expansion_for_cluster(
                clusterid=int(anchor_cluster),
                tpcid=int(tpcid),
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                channel_support_cache=channel_support_cache,
                saturated_channel_cache=saturated_channel_cache,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                t0_candidates=t0_candidates,
                assignment_info=assignment_info,
                unassigned_by_tpc=unassigned_by_tpc,
                cluster_energy=float(anchor_energy),
                search_range=int(search_range),
                adc_clip=float(adc_clip),
                collect_scan_losses=collect_scan_losses,
                full_scan_assign_eps=float(full_scan_assign_eps),
                backward_peak_align_ticks=int(backward_peak_align_ticks),
                leftover_absorption_context=leftover_absorption_context,
            )
            logs.append(log)
            scan_updates.update(new_scan_updates)
            if action == "expanded":
                expanded_clusters.append(int(anchor_cluster))
                continue
            remaining.remove(int(anchor_cluster))
            if action == "assigned":
                assigned_clusters.append(int(anchor_cluster))
                full_scan_assigned_clusters.append(int(anchor_cluster))
            else:
                full_scan_unassigned_clusters.append(int(anchor_cluster))
            continue

        loss_matrix, current_errors, band_now = _masked_loss_matrix_single_tpc(
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_support_cache=channel_support_cache,
            saturated_channel_cache=saturated_channel_cache,
            tpcid=int(tpcid),
            clusters=np.asarray(band_clusters, dtype=int),
            placed_mask=np.zeros(len(band_clusters), dtype=bool),
            t0_candidates=candidate_grid,
            adc_clip=float(adc_clip),
        )
        if loss_matrix.size == 0 or len(band_now) == 0:
            break

        delta_matrix = np.asarray(loss_matrix, dtype=np.float32) - np.asarray(current_errors, dtype=np.float32)[:, None]
        norm_matrix = np.zeros_like(delta_matrix, dtype=np.float32)
        row_mins = np.min(delta_matrix, axis=1)

        for row_idx, cid in enumerate(band_now):
            norm_matrix[row_idx] = _normalize_delta_row(
                delta_matrix[row_idx],
                cluster_energy=float(cluster_energies.get(int(cid), 0.0)),
                anchor_energy=float(anchor_energy),
            )

        expandable_rows = [
            int(cid)
            for row_idx, cid in enumerate(band_now)
            if float(row_mins[row_idx]) > float(positive_row_margin) and int(cid) not in exhausted_expansion
        ]
        if expandable_rows:
            expand_cluster = max(
                expandable_rows,
                key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
            )
            action, log, new_scan_updates = _try_t0_expansion_for_cluster(
                clusterid=int(expand_cluster),
                tpcid=int(tpcid),
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                channel_support_cache=channel_support_cache,
                saturated_channel_cache=saturated_channel_cache,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                t0_candidates=t0_candidates,
                assignment_info=assignment_info,
                unassigned_by_tpc=unassigned_by_tpc,
                cluster_energy=float(cluster_energies.get(int(expand_cluster), 0.0)),
                search_range=int(search_range),
                adc_clip=float(adc_clip),
                collect_scan_losses=collect_scan_losses,
                full_scan_assign_eps=float(full_scan_assign_eps),
                backward_peak_align_ticks=int(backward_peak_align_ticks),
                leftover_absorption_context=leftover_absorption_context,
            )
            logs.append(log)
            scan_updates.update(new_scan_updates)
            if action == "expanded":
                expanded_clusters.append(int(expand_cluster))
                continue
            exhausted_expansion.add(int(expand_cluster))
            remaining.remove(int(expand_cluster))
            if action == "assigned":
                assigned_clusters.append(int(expand_cluster))
                full_scan_assigned_clusters.append(int(expand_cluster))
            else:
                full_scan_unassigned_clusters.append(int(expand_cluster))
            continue

        best_flat = int(np.argmin(norm_matrix))
        best_cluster_idx = int(best_flat // norm_matrix.shape[1])
        best_t0_idx = int(best_flat % norm_matrix.shape[1])
        clusterid = int(band_now[best_cluster_idx])
        opt_t0 = int(candidate_grid[best_t0_idx])
        best_norm = float(norm_matrix[best_cluster_idx, best_t0_idx])

        if best_norm > float(matrix_worsen_tolerance_norm):
            action, log, new_scan_updates = _try_t0_expansion_for_cluster(
                clusterid=int(clusterid),
                tpcid=int(tpcid),
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                channel_support_cache=channel_support_cache,
                saturated_channel_cache=saturated_channel_cache,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                t0_candidates=t0_candidates,
                assignment_info=assignment_info,
                unassigned_by_tpc=unassigned_by_tpc,
                cluster_energy=float(cluster_energies.get(int(clusterid), 0.0)),
                search_range=int(search_range),
                adc_clip=float(adc_clip),
                collect_scan_losses=collect_scan_losses,
                full_scan_assign_eps=float(full_scan_assign_eps),
                backward_peak_align_ticks=int(backward_peak_align_ticks),
                leftover_absorption_context=leftover_absorption_context,
            )
            logs.append(log)
            scan_updates.update(new_scan_updates)
            if action == "expanded":
                expanded_clusters.append(int(clusterid))
                continue
            exhausted_expansion.add(int(clusterid))
            remaining.remove(int(clusterid))
            if action == "assigned":
                assigned_clusters.append(int(clusterid))
                full_scan_assigned_clusters.append(int(clusterid))
            else:
                full_scan_unassigned_clusters.append(int(clusterid))
            continue

        cluster_wave_full, selected_wave, channel_indices, _ = _get_single_tpc_support(
            clusterid=int(clusterid),
            tpcid=int(tpcid),
            image_maps=image_maps,
            channel_support_cache=channel_support_cache,
            saturated_channel_cache=saturated_channel_cache,
        )
        cluster_block = np.asarray(cluster_wave_full, dtype=np.float32)[None, :, :]
        base_tpc, actual_sel, error_sel = _extract_selected_tpc_blocks(
            tpcid=int(tpcid),
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_indices=channel_indices,
        )
        current_error = float(np.asarray(current_errors, dtype=np.float32)[best_cluster_idx])
        aligned_t0 = _push_back_align_t0(
            cluster_wave=selected_wave,
            actual_block=actual_sel,
            base_block=base_tpc,
            objective_t0=int(opt_t0),
            max_backtrack_ticks=int(backward_peak_align_ticks),
            search_range=int(search_range),
        )
        shifted = _shift_block(np.asarray(selected_wave, dtype=np.float32)[None, :, :], int(aligned_t0))[0]
        candidate_model = np.clip(base_tpc + shifted, None, float(adc_clip))
        candidate_error = compute_error_metric(candidate_model, actual_sel, error_sel)
        raw_delta = float(candidate_error - float(current_error))

        _assign_cluster_at_t0(
            int(clusterid),
            np.asarray([int(tpcid)], dtype=int),
            int(aligned_t0),
            cluster_block=cluster_block,
            base_image=base_image,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energy=float(cluster_energies.get(int(clusterid), 0.0)),
            mode="iterative_matrix",
            stage="iterative_matrix",
            error_after=float(candidate_error),
            improvement=float(-raw_delta),
            adc_clip=float(adc_clip),
            max_t0=int(search_range),
        )
        assignment_info[(int(clusterid), int(tpcid))]["objective_t0"] = float(opt_t0)
        assignment_info[(int(clusterid), int(tpcid))]["n_fit_channels"] = int(channel_indices.size)
        _maybe_absorb_leftovers_after_assignment(
            clusterid=int(clusterid),
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_support_cache=channel_support_cache,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            assignment_info=assignment_info,
            leftover_absorption_context=leftover_absorption_context,
            adc_clip=float(adc_clip),
            saturated_channel_cache=saturated_channel_cache,
        )
        log = {
            "clusterid": int(clusterid),
            "tpcs": [int(tpcid)],
            "energy": float(cluster_energies.get(int(clusterid), 0.0)),
            "assigned": True,
            "mode": "iterative_matrix",
            "label": "iterative_matrix",
            "t0": int(aligned_t0),
            "objective_t0": int(opt_t0),
            "improvement": float(-raw_delta),
            "normalized_score": float(best_norm),
            "anchor_cluster": int(anchor_cluster),
            "n_fit_channels": int(channel_indices.size),
        }
        logs.append(log)
        if collect_scan_losses and int(clusterid) not in scan_updates:
            scan_updates[int(clusterid)] = _build_scan_loss_entry(
                clusterid=int(clusterid),
                stage="iterative_matrix",
                mode="iterative_matrix",
                tpcs=[int(tpcid)],
                energy=float(cluster_energies.get(int(clusterid), 0.0)),
                best_t0=int(aligned_t0),
                assigned=True,
                search_range=int(search_range),
                loss_curve=None,
                best_t0_scan=None,
            )
        remaining.remove(int(clusterid))
        assigned_clusters.append(int(clusterid))
        matrix_assigned_clusters.append(int(clusterid))

    stats = {
        "assigned_clusters": sorted(set(int(cid) for cid in assigned_clusters)),
        "matrix_assigned_clusters": sorted(set(int(cid) for cid in matrix_assigned_clusters)),
        "expanded_clusters": sorted(set(int(cid) for cid in expanded_clusters)),
        "full_scan_assigned_clusters": sorted(set(int(cid) for cid in full_scan_assigned_clusters)),
        "full_scan_unassigned_clusters": sorted(set(int(cid) for cid in full_scan_unassigned_clusters)),
        "band_anchor_sequence": [int(cid) for cid in band_anchor_sequence],
    }
    return logs, scan_updates, stats


def assign_small_clusters_v8(
    cluster_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energies: dict[int, float],
    *,
    large_cluster_energy_mev: float = 50.0,
    minimum_iterative_energy_mev: float = 0.0,
    energy_band_fraction: float = 0.20,
    positive_row_margin: float = 1e-4,
    matrix_worsen_tolerance_norm: float = 0.15,
    full_scan_assign_eps: float = 0.0,
    search_range: int = 800,
    adc_clip: float = 60780.0,
    collect_scan_losses: bool = False,
    assignment_improvement_eps: float = 0.0,
    backward_peak_align_ticks: int = 5,
    leftover_absorption_context: dict[str, Any] | None = None,
    saturated_channel_cache: dict[int, np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    dict[int, list[int]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    assignment_log: list[dict[str, Any]] = []
    scan_loss_dict: dict[int, dict[str, Any]] = {}

    active_cluster_tpcs: dict[int, np.ndarray] = {}
    for clusterid in cluster_labels:
        tpcs = sorted(
            {
                int(tpc)
                for tpc in cluster_to_tpcs.get(int(clusterid), [])
                if int(tpc) < int(base_image.shape[0]) and (int(clusterid), int(tpc)) in image_maps
            }
        )
        if tpcs:
            active_cluster_tpcs[int(clusterid)] = np.asarray(tpcs, dtype=int)

    primary_clusters: list[int] = []
    iterative_single_tpc: dict[int, list[int]] = {}
    pruned_iterative_clusters: list[int] = []

    for clusterid, tpcs in active_cluster_tpcs.items():
        cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
        if len(tpcs) > 1 or cluster_energy > float(large_cluster_energy_mev):
            primary_clusters.append(int(clusterid))
            continue
        if cluster_energy < float(minimum_iterative_energy_mev):
            pruned_iterative_clusters.append(int(clusterid))
            _mark_cluster_unassigned(
                int(clusterid),
                np.asarray(tpcs, dtype=int),
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                assignment_info=assignment_info,
                unassigned_by_tpc=unassigned_by_tpc,
                cluster_energy=float(cluster_energy),
                mode="below_iterative_energy_threshold",
                stage="iterative_preselection",
            )
            if collect_scan_losses:
                scan_loss_dict[int(clusterid)] = _build_scan_loss_entry(
                    clusterid=int(clusterid),
                    stage="iterative_preselection",
                    mode="below_iterative_energy_threshold",
                    tpcs=[int(tpc) for tpc in tpcs],
                    energy=float(cluster_energy),
                    best_t0=None,
                    assigned=False,
                    search_range=int(search_range),
                    loss_curve=None,
                    best_t0_scan=None,
                )
            assignment_log.append(
                {
                    "clusterid": int(clusterid),
                    "tpcs": [int(tpc) for tpc in tpcs],
                    "energy": float(cluster_energy),
                    "assigned": False,
                    "mode": "below_iterative_energy_threshold",
                    "label": "iterative_preselection",
                    "t0": -1,
                    "improvement": 0.0,
                }
            )
            continue
        if len(tpcs) == 1:
            iterative_single_tpc.setdefault(int(tpcs[0]), []).append(int(clusterid))

    primary_logs, primary_scan_updates, primary_assigned = _full_scan_primary_clusters(
        cluster_ids=primary_clusters,
        active_cluster_tpcs=active_cluster_tpcs,
        image_maps=image_maps,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        labels_global=labels_global,
        hit_timestamps=hit_timestamps,
        t0_candidates=t0_candidates,
        assignment_info=assignment_info,
        unassigned_by_tpc=unassigned_by_tpc,
        cluster_energies=cluster_energies,
        search_range=int(search_range),
        adc_clip=float(adc_clip),
        collect_scan_losses=collect_scan_losses,
        assignment_improvement_eps=float(assignment_improvement_eps),
        stage_name="primary_full_scan",
        accepted_mode="primary_full_scan",
        rejected_mode="primary_full_scan_unassigned",
        channel_support_cache=channel_support_cache,
        leftover_absorption_context=leftover_absorption_context,
        saturated_channel_cache=saturated_channel_cache,
    )
    assignment_log.extend(primary_logs)
    scan_loss_dict.update(primary_scan_updates)

    iterative_logs: list[dict[str, Any]] = []
    iterative_scan_updates: dict[int, dict[str, Any]] = {}
    iterative_assigned_clusters: list[int] = []
    iterative_matrix_clusters: list[int] = []
    iterative_expanded_clusters: list[int] = []
    iterative_full_scan_assigned: list[int] = []
    iterative_full_scan_unassigned: list[int] = []
    band_anchor_sequence_by_tpc: dict[int, list[int]] = {}

    for tpcid in sorted(iterative_single_tpc):
        cluster_ids_here = sorted(
            iterative_single_tpc[int(tpcid)],
            key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
            reverse=True,
        )
        logs, scan_updates, stats = _iterative_band_matrix_assign_tpc(
            tpcid=int(tpcid),
            cluster_ids=cluster_ids_here,
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_support_cache=channel_support_cache,
            saturated_channel_cache=saturated_channel_cache,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energies=cluster_energies,
            band_fraction=float(energy_band_fraction),
            positive_row_margin=float(positive_row_margin),
            matrix_worsen_tolerance_norm=float(matrix_worsen_tolerance_norm),
            search_range=int(search_range),
            adc_clip=float(adc_clip),
            collect_scan_losses=collect_scan_losses,
            full_scan_assign_eps=float(full_scan_assign_eps),
            backward_peak_align_ticks=int(backward_peak_align_ticks),
            leftover_absorption_context=leftover_absorption_context,
        )
        iterative_logs.extend(logs)
        iterative_scan_updates.update(scan_updates)
        iterative_assigned_clusters.extend(int(cid) for cid in stats.get("assigned_clusters", []))
        iterative_matrix_clusters.extend(int(cid) for cid in stats.get("matrix_assigned_clusters", []))
        iterative_expanded_clusters.extend(int(cid) for cid in stats.get("expanded_clusters", []))
        iterative_full_scan_assigned.extend(int(cid) for cid in stats.get("full_scan_assigned_clusters", []))
        iterative_full_scan_unassigned.extend(int(cid) for cid in stats.get("full_scan_unassigned_clusters", []))
        band_anchor_sequence_by_tpc[int(tpcid)] = [int(cid) for cid in stats.get("band_anchor_sequence", [])]

    assignment_log.extend(iterative_logs)
    scan_loss_dict.update(iterative_scan_updates)

    stage_stats = {
        "primary_full_scan_clusters": sorted(set(int(cid) for cid in primary_clusters)),
        "primary_full_scan_assigned": sorted(set(int(cid) for cid in primary_assigned)),
        "iterative_assigned_clusters": sorted(set(int(cid) for cid in iterative_assigned_clusters)),
        "iterative_matrix_clusters": sorted(set(int(cid) for cid in iterative_matrix_clusters)),
        "iterative_expanded_clusters": sorted(set(int(cid) for cid in iterative_expanded_clusters)),
        "iterative_full_scan_assigned": sorted(set(int(cid) for cid in iterative_full_scan_assigned)),
        "iterative_full_scan_unassigned": sorted(set(int(cid) for cid in iterative_full_scan_unassigned)),
        "pruned_iterative_clusters": sorted(set(int(cid) for cid in pruned_iterative_clusters)),
        "step4_clusters": sorted(set(int(cid) for cid in iterative_assigned_clusters)),
        "step4_assigned_clusters": sorted(set(int(cid) for cid in iterative_assigned_clusters)),
        "energy_band_fraction": float(energy_band_fraction),
        "large_cluster_energy_mev": float(large_cluster_energy_mev),
        "minimum_iterative_energy_mev": float(minimum_iterative_energy_mev),
        "positive_row_margin": float(positive_row_margin),
        "matrix_worsen_tolerance_norm": float(matrix_worsen_tolerance_norm),
        "backward_peak_align_ticks": int(backward_peak_align_ticks),
        "channel_support_entries": int(0 if channel_support_cache is None else len(channel_support_cache)),
        "band_anchor_sequence_by_tpc": {
            int(tpc): [int(cid) for cid in sequence]
            for tpc, sequence in sorted(band_anchor_sequence_by_tpc.items())
        },
    }

    return (
        base_image,
        hit_timestamps,
        t0_candidates,
        assignment_info,
        unassigned_by_tpc,
        assignment_log,
        scan_loss_dict,
        stage_stats,
    )


__all__ = [
    "assign_small_clusters_v8",
    "rebalance_step4_clusters_v4",
]
