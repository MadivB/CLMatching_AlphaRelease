from __future__ import annotations

import importlib
from typing import Any
import warnings

import numpy as np

try:
    from v3_2_global_matching import _build_scan_loss_entry, _mark_cluster_unassigned
    from v8_iterative_matching import _full_scan_primary_clusters, _iterative_band_matrix_assign_tpc
    from v8_1_flash_seeding import merge_flash_t0_candidates_after_primary
    from flash_cluster_table import (
        flash_cluster_table_rows,
        rebuild_flash_cluster_flags_from_assignments,
    )
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v3_2_global_matching import _build_scan_loss_entry, _mark_cluster_unassigned
    from M5p1.v8_iterative_matching import _full_scan_primary_clusters, _iterative_band_matrix_assign_tpc
    from M5p1.v8_1_flash_seeding import merge_flash_t0_candidates_after_primary
    from M5p1.flash_cluster_table import (
        flash_cluster_table_rows,
        rebuild_flash_cluster_flags_from_assignments,
    )


def _replace_nested_bool_list(target: list[list[bool]], values: list[list[bool]]) -> None:
    target[:] = [[bool(v) for v in row] for row in values]


def _cluster_image_tpcs(
    *,
    clusterid: int,
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    max_tpcs: int,
) -> list[int]:
    return sorted(
        {
            int(tpc)
            for tpc in cluster_to_tpcs.get(int(clusterid), [])
            if 0 <= int(tpc) < int(max_tpcs) and (int(clusterid), int(tpc)) in image_maps
        }
    )


def _partition_cluster_population(
    *,
    cluster_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    cluster_energies: dict[int, float],
    large_cluster_energy_mev: float,
    minimum_iterative_energy_mev: float,
    max_tpcs: int,
) -> tuple[dict[int, np.ndarray], list[int], dict[int, list[int]], list[int]]:
    active_cluster_tpcs: dict[int, np.ndarray] = {}
    primary_clusters: list[int] = []
    iterative_single_tpc: dict[int, list[int]] = {}
    pruned_iterative_clusters: list[int] = []

    for clusterid in cluster_labels:
        tpcs = _cluster_image_tpcs(
            clusterid=int(clusterid),
            cluster_to_tpcs=cluster_to_tpcs,
            image_maps=image_maps,
            max_tpcs=int(max_tpcs),
        )
        if len(tpcs) == 0:
            continue

        active_cluster_tpcs[int(clusterid)] = np.asarray(tpcs, dtype=int)
        cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))

        if len(tpcs) > 1 or cluster_energy > float(large_cluster_energy_mev):
            primary_clusters.append(int(clusterid))
            continue

        if cluster_energy < float(minimum_iterative_energy_mev):
            pruned_iterative_clusters.append(int(clusterid))
            continue

        if len(tpcs) == 1:
            iterative_single_tpc.setdefault(int(tpcs[0]), []).append(int(clusterid))

    return active_cluster_tpcs, primary_clusters, iterative_single_tpc, pruned_iterative_clusters


def run_large_cluster_scan_phase_v11(
    *,
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
    large_cluster_energy_mev: float = 50.0,
    minimum_iterative_energy_mev: float = 0.5,
    search_range: int = 800,
    adc_clip: float = 60780.0,
    collect_scan_losses: bool = False,
    assignment_improvement_eps: float = 0.0,
    flash_seed_t0s_by_tpc: dict[int, list[int]] | None = None,
    flash_seed_t0_resolution: int = 5,
    flash_cluster_received_by_tpc: list[list[bool]] | None = None,
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
    dict[int, np.ndarray],
    dict[int, list[int]],
    list[int],
]:
    (
        active_cluster_tpcs,
        primary_clusters,
        iterative_single_tpc,
        pruned_iterative_clusters,
    ) = _partition_cluster_population(
        cluster_labels=cluster_labels,
        cluster_to_tpcs=cluster_to_tpcs,
        image_maps=image_maps,
        cluster_energies=cluster_energies,
        large_cluster_energy_mev=float(large_cluster_energy_mev),
        minimum_iterative_energy_mev=float(minimum_iterative_energy_mev),
        max_tpcs=int(base_image.shape[0]),
    )

    logs, scan_updates, assigned_clusters = _full_scan_primary_clusters(
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
        collect_scan_losses=bool(collect_scan_losses),
        assignment_improvement_eps=float(assignment_improvement_eps),
        stage_name="primary_full_scan",
        accepted_mode="primary_full_scan",
        rejected_mode="primary_full_scan_unassigned",
        channel_support_cache=channel_support_cache,
        leftover_absorption_context=leftover_absorption_context,
        saturated_channel_cache=saturated_channel_cache,
    )

    t0_candidates, merged_flash_t0s_by_tpc, flash_merge_stats = merge_flash_t0_candidates_after_primary(
        t0_candidates=t0_candidates,
        flash_seed_t0s_by_tpc=flash_seed_t0s_by_tpc,
        candidate_min_sep=int(flash_seed_t0_resolution),
        max_t0=int(search_range),
    )
    flash_cluster_canonicalization_rows: list[dict[str, Any]] = []
    if flash_cluster_received_by_tpc is not None:
        t0_candidates, rebuilt_flags, flash_rows = rebuild_flash_cluster_flags_from_assignments(
            t0_candidates,
            assignment_info,
            resolution_ticks=float(flash_seed_t0_resolution),
            max_t0=float(search_range),
            initial_flags=flash_cluster_received_by_tpc,
            prefer_existing_true=False,
        )
        _replace_nested_bool_list(flash_cluster_received_by_tpc, rebuilt_flags)
        flash_cluster_canonicalization_rows.extend(flash_rows)

    stage_stats = {
        "primary_full_scan_clusters": sorted(set(int(cid) for cid in primary_clusters)),
        "primary_full_scan_assigned": sorted(set(int(cid) for cid in assigned_clusters)),
        "large_cluster_energy_mev": float(large_cluster_energy_mev),
        "minimum_iterative_energy_mev": float(minimum_iterative_energy_mev),
        "flash_seed_t0s_input_by_tpc": {
            int(tpc): [int(v) for v in values]
            for tpc, values in sorted((flash_seed_t0s_by_tpc or {}).items())
        },
        "flash_seed_t0s_merged_by_tpc": {
            int(tpc): [int(v) for v in values]
            for tpc, values in sorted(merged_flash_t0s_by_tpc.items())
        },
        "flash_merge_stats": dict(flash_merge_stats),
        "flash_cluster_table_rows": flash_cluster_table_rows(t0_candidates, flash_cluster_received_by_tpc)
        if flash_cluster_received_by_tpc is not None
        else [],
        "flash_cluster_canonicalization_rows": [dict(row) for row in flash_cluster_canonicalization_rows],
        "n_cluster_received_flash_t0s": int(
            sum(sum(bool(v) for v in row) for row in flash_cluster_received_by_tpc)
        )
        if flash_cluster_received_by_tpc is not None
        else 0,
    }

    return (
        base_image,
        hit_timestamps,
        t0_candidates,
        assignment_info,
        unassigned_by_tpc,
        logs,
        scan_updates,
        stage_stats,
        active_cluster_tpcs,
        iterative_single_tpc,
        pruned_iterative_clusters,
    )


def run_small_cluster_matrix_phase_v11(
    *,
    active_cluster_tpcs: dict[int, np.ndarray],
    iterative_single_tpc: dict[int, list[int]],
    pruned_iterative_clusters: list[int],
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
    energy_band_fraction: float = 0.20,
    positive_row_margin: float = -1e-4,
    matrix_worsen_tolerance_norm: float = 0.15,
    search_range: int = 800,
    adc_clip: float = 60780.0,
    collect_scan_losses: bool = False,
    full_scan_assign_eps: float = -1.0,
    backward_peak_align_ticks: int = 5,
    leftover_absorption_context: dict[str, Any] | None = None,
    saturated_channel_cache: dict[int, np.ndarray] | None = None,
    shower_rescue_context: dict[str, Any] | None = None,
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
    shower_rescue_logs: list[dict[str, Any]] = []
    shower_rescue_summary: dict[int, dict[str, Any]] = {}
    shower_rescue_tpcs: set[int] = set()

    if shower_rescue_context is not None and bool(shower_rescue_context.get("enabled", True)):
        shower_rescue_module_name = str(shower_rescue_context.get("module", "shower_rescue_release_v1_1"))
        srv = importlib.import_module(shower_rescue_module_name)
        namespace = shower_rescue_context.get("namespace")
        if namespace is None:
            namespace = {}
            shower_rescue_context["namespace"] = namespace

        # Bind the current mutable phase state into the rescue namespace.  This is
        # important because the rescue path intentionally redoes non-track shower
        # TPCs using the latest post-phase-2 track/base image state.
        namespace["baseImage"] = base_image
        namespace["hit_timestamps"] = hit_timestamps
        namespace["t0Candidates"] = t0_candidates
        namespace["assignment_info"] = assignment_info
        namespace["imageMaps"] = image_maps
        namespace["fullLightWaveform"] = full_light_waveform
        namespace["fullLightStd"] = full_light_std
        namespace["saturated_channel_cache"] = saturated_channel_cache

        explicit_tpcs = shower_rescue_context.get("target_tpcs")
        if explicit_tpcs is not None:
            candidate_rescue_tpcs = [int(v) for v in np.asarray(explicit_tpcs, dtype=np.int32).tolist()]
        else:
            candidate_rescue_tpcs = srv.find_shower_tpcs_in_event(
                namespace=namespace,
                min_shower_hits=int(shower_rescue_context.get("min_shower_hits", 1)),
                min_shower_energy_mev=float(shower_rescue_context.get("min_shower_energy_mev", 0.0)),
            )

        if bool(shower_rescue_context.get("restrict_to_iterative_tpcs", False)):
            iterative_tpc_set = {int(v) for v in iterative_single_tpc}
            candidate_rescue_tpcs = [int(tpc) for tpc in candidate_rescue_tpcs if int(tpc) in iterative_tpc_set]

        rescue_kwargs = dict(shower_rescue_context.get("rescue_kwargs", {}))
        fail_open = bool(shower_rescue_context.get("fail_open", False))

        for tpcid in sorted(set(int(v) for v in candidate_rescue_tpcs)):
            try:
                result = srv.run_release_1_1_shower_rescue_for_tpc(
                    int(tpcid),
                    namespace=namespace,
                    base_image=base_image,
                    hit_timestamps=hit_timestamps,
                    t0_candidates=t0_candidates,
                    assignment_info=assignment_info,
                    image_maps=image_maps,
                    hit_tpc_ids=namespace.get("hitTPCid"),
                    full_light_waveform=full_light_waveform,
                    full_light_std=full_light_std,
                    adc_clip=float(adc_clip),
                    cluster_energies=cluster_energies,
                    **rescue_kwargs,
                )
            except Exception as exc:
                if not fail_open:
                    raise
                shower_rescue_summary[int(tpcid)] = {
                    "status": "failed_open",
                    "error": str(exc),
                }
                continue

            final_result = result["final_result"]
            base_image[int(tpcid)] = np.asarray(final_result["pred_total_full"], dtype=np.float32)
            if final_result.get("hit_timestamps_final_global") is not None:
                hit_timestamps[:] = np.asarray(final_result["hit_timestamps_final_global"], dtype=hit_timestamps.dtype)
            else:
                hit_tpc_ids = namespace.get("hitTPCid")
                if hit_tpc_ids is not None:
                    tpc_idx = np.flatnonzero(np.asarray(hit_tpc_ids, dtype=np.int32) == int(tpcid))
                    hit_timestamps[tpc_idx] = np.asarray(final_result["t_tpc_final"], dtype=hit_timestamps.dtype)

            for log_row in result.get("assignment_logs", []):
                cid = int(log_row["clusterid"])
                t0 = int(log_row["t0"])
                assignment_info[(cid, int(tpcid))] = {
                    "clusterid": int(cid),
                    "tpcid": int(tpcid),
                    "t0": int(t0),
                    "mode": "shower_rescue_v1_1",
                    "stage": "shower_rescue_v1_1",
                    "energy": float(log_row.get("energy", cluster_energies.get(int(cid), 0.0))),
                    "energy_fraction": float(log_row.get("energy_fraction", 1.0)),
                    "split_t0s": list(log_row.get("split_t0s", [])),
                }
                if int(tpcid) in unassigned_by_tpc:
                    unassigned_by_tpc[int(tpcid)] = [
                        int(v) for v in unassigned_by_tpc[int(tpcid)] if int(v) != int(cid)
                    ]
                assignment_log.append(dict(log_row))

            shower_rescue_tpcs.add(int(tpcid))
            shower_rescue_logs.extend(dict(row) for row in result.get("assignment_logs", []))
            shower_rescue_summary[int(tpcid)] = {
                "status": "applied",
                "detected_shower_t0": int(result.get("detected_shower_t0", -1)),
                "n_fill_assignments": int(len(result.get("fill_result", {}).get("accepted_rows", []))),
                "n_shower_promotions": int(len(result.get("shower_result", {}).get("accepted_rows", []))),
                "n_nonshower_steps": int(
                    len([st for st in result.get("nonshower_states", []) if st.get("selected_row") is not None])
                ),
                "n_absorbed_hits": int(len(result.get("absorb_result", {}).get("absorbed_rows", []))),
                "n_final_contribution_rows": int(len(final_result.get("contribution_rows", []))),
            }

        namespace["baseImage"] = base_image
        namespace["hit_timestamps"] = hit_timestamps
        namespace["assignment_info"] = assignment_info

    for clusterid in sorted(set(int(cid) for cid in pruned_iterative_clusters)):
        tpcs = active_cluster_tpcs.get(int(clusterid), np.asarray([], dtype=int))
        if len(tpcs) == 1 and int(tpcs[0]) in shower_rescue_tpcs:
            continue
        cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
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

    iterative_logs: list[dict[str, Any]] = []
    iterative_scan_updates: dict[int, dict[str, Any]] = {}
    iterative_assigned_clusters: list[int] = []
    iterative_matrix_clusters: list[int] = []
    iterative_expanded_clusters: list[int] = []
    iterative_full_scan_assigned: list[int] = []
    iterative_full_scan_unassigned: list[int] = []
    band_anchor_sequence_by_tpc: dict[int, list[int]] = {}

    for tpcid in sorted(iterative_single_tpc):
        if int(tpcid) in shower_rescue_tpcs:
            continue
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
            collect_scan_losses=bool(collect_scan_losses),
            full_scan_assign_eps=float(full_scan_assign_eps),
            backward_peak_align_ticks=int(backward_peak_align_ticks),
            leftover_absorption_context=leftover_absorption_context,
            saturated_channel_cache=saturated_channel_cache,
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
    shower_rescue_assigned_clusters = sorted(
        set(int(row["clusterid"]) for row in shower_rescue_logs if row.get("assigned", False))
    )

    stage_stats = {
        "iterative_assigned_clusters": sorted(set(int(cid) for cid in iterative_assigned_clusters)),
        "iterative_matrix_clusters": sorted(set(int(cid) for cid in iterative_matrix_clusters)),
        "iterative_expanded_clusters": sorted(set(int(cid) for cid in iterative_expanded_clusters)),
        "iterative_full_scan_assigned": sorted(set(int(cid) for cid in iterative_full_scan_assigned)),
        "iterative_full_scan_unassigned": sorted(set(int(cid) for cid in iterative_full_scan_unassigned)),
        "pruned_iterative_clusters": sorted(set(int(cid) for cid in pruned_iterative_clusters)),
        "step4_clusters": sorted(set(int(cid) for cid in iterative_assigned_clusters) | set(shower_rescue_assigned_clusters)),
        "step4_assigned_clusters": sorted(set(int(cid) for cid in iterative_assigned_clusters) | set(shower_rescue_assigned_clusters)),
        "energy_band_fraction": float(energy_band_fraction),
        "positive_row_margin": float(positive_row_margin),
        "matrix_worsen_tolerance_norm": float(matrix_worsen_tolerance_norm),
        "backward_peak_align_ticks": int(backward_peak_align_ticks),
        "shower_rescue_tpcs": sorted(int(v) for v in shower_rescue_tpcs),
        "shower_rescue_summary": {
            int(tpc): dict(row)
            for tpc, row in sorted(shower_rescue_summary.items())
        },
        "shower_rescue_assigned_clusters": shower_rescue_assigned_clusters,
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


def snapshot_backbone_hits_v11(
    *,
    hit_timestamps: np.ndarray,
    labels_global: np.ndarray,
    v8_absorbed_hit_parent: np.ndarray | None = None,
    track_shower_labels: list[int] | None = None,
) -> dict[str, Any]:
    decided_mask = np.isfinite(np.asarray(hit_timestamps, dtype=np.float32))
    decided_indices = np.flatnonzero(decided_mask)
    absorbed_parent = None
    if v8_absorbed_hit_parent is not None:
        absorbed_parent = np.asarray(v8_absorbed_hit_parent[decided_mask], dtype=np.int32).copy()
    return {
        "mask": np.asarray(decided_mask, dtype=bool).copy(),
        "indices": np.asarray(decided_indices, dtype=np.int64).copy(),
        "expected_t0": np.asarray(hit_timestamps[decided_mask], dtype=np.float32).copy(),
        "labels": np.asarray(labels_global[decided_mask], dtype=np.int32).copy(),
        "absorbed_parent": absorbed_parent,
        "track_shower_labels": [] if track_shower_labels is None else [int(v) for v in track_shower_labels],
        "n_hits": int(np.count_nonzero(decided_mask)),
    }


def verify_backbone_hits_unchanged_v11(
    snapshot: dict[str, Any],
    *,
    hit_timestamps: np.ndarray,
    stage_name: str,
    atol: float = 1e-4,
    max_examples: int = 8,
) -> dict[str, Any]:
    mask = np.asarray(snapshot["mask"], dtype=bool)
    expected = np.asarray(snapshot["expected_t0"], dtype=np.float32)
    indices = np.asarray(snapshot["indices"], dtype=np.int64)
    labels = np.asarray(snapshot["labels"], dtype=np.int32)
    current = np.asarray(hit_timestamps[mask], dtype=np.float32)

    mismatch = (~np.isfinite(current)) | (np.abs(current - expected) > float(atol))
    n_changed = int(np.count_nonzero(mismatch))
    n_total = int(mask.sum())

    if n_changed > 0:
        warning_msg = (
            f"WARNING: {n_changed}/{n_total} track/shower-phase decided hits changed after {stage_name}."
        )
        warnings.warn(warning_msg)
        print(warning_msg)
        print(f"{'hit_idx':>8} {'label':>8} {'expected':>12} {'current':>12}")
        print("-" * 46)
        bad_idx = np.flatnonzero(mismatch)[: int(max_examples)]
        for local_idx in bad_idx.tolist():
            print(
                f"{int(indices[local_idx]):8d} "
                f"{int(labels[local_idx]):8d} "
                f"{float(expected[local_idx]):12.4f} "
                f"{float(current[local_idx]):12.4f}"
            )
    else:
        print(
            f"Backbone integrity OK after {stage_name}: "
            f"{n_total}/{n_total} decided hits unchanged."
        )

    return {
        "stage_name": str(stage_name),
        "n_total": int(n_total),
        "n_changed": int(n_changed),
        "changed_indices": np.asarray(indices[mismatch], dtype=np.int64).copy(),
    }


__all__ = [
    "run_large_cluster_scan_phase_v11",
    "run_small_cluster_matrix_phase_v11",
    "snapshot_backbone_hits_v11",
    "verify_backbone_hits_unchanged_v11",
]
