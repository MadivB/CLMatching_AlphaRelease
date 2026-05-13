from __future__ import annotations

from typing import Any

import numpy as np

try:
    from v3_2_global_matching import (
        _build_scan_loss_entry,
        _mark_cluster_unassigned,
        _shift_block,
        append_candidate_t0,
        compute_error_metric,
    )
    from v4_hierarchical_matching import rebalance_step4_clusters_v4
    from v8_iterative_matching import _full_scan_primary_clusters, _iterative_band_matrix_assign_tpc
    from v8_1_flash_seeding import merge_flash_t0_candidates_after_primary
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v3_2_global_matching import (
        _build_scan_loss_entry,
        _mark_cluster_unassigned,
        _shift_block,
        append_candidate_t0,
        compute_error_metric,
    )
    from M5p1.v4_hierarchical_matching import rebalance_step4_clusters_v4
    from M5p1.v8_iterative_matching import _full_scan_primary_clusters, _iterative_band_matrix_assign_tpc
    from M5p1.v8_1_flash_seeding import merge_flash_t0_candidates_after_primary


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


def _cluster_assigned_t0(
    *,
    clusterid: int,
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    max_tpcs: int,
) -> int | None:
    values: list[int] = []
    for tpcid in _cluster_image_tpcs(
        clusterid=int(clusterid),
        cluster_to_tpcs=cluster_to_tpcs,
        image_maps=image_maps,
        max_tpcs=int(max_tpcs),
    ):
        info = assignment_info.get((int(clusterid), int(tpcid)), {})
        if not bool(info.get("assigned", False)):
            continue
        t0 = info.get("t0", np.nan)
        if np.isfinite(t0):
            values.append(int(round(float(t0))))

    if len(values) == 0:
        return None

    unique_vals = sorted(set(values))
    if len(unique_vals) != 1:
        return None
    return int(unique_vals[0])


def _is_primary_swap_eligible(
    *,
    clusterid: int,
    tpcid: int,
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    cluster_energies: dict[int, float],
    large_cluster_energy_mev: float,
) -> bool:
    info = assignment_info.get((int(clusterid), int(tpcid)), {})
    if not bool(info.get("assigned", False)):
        return False

    t0 = info.get("t0", np.nan)
    if not np.isfinite(t0):
        return False

    if str(info.get("stage", "")) == "track":
        return True

    return float(cluster_energies.get(int(clusterid), 0.0)) > float(large_cluster_energy_mev)


def _score_primary_swap_pair(
    *,
    cluster_a: int,
    cluster_b: int,
    t0_a: int,
    t0_b: int,
    union_tpcs: list[int],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    adc_clip: float,
) -> tuple[float, float]:
    current_score = 0.0
    swapped_score = 0.0

    for tpcid in union_tpcs:
        current_block = np.asarray(base_image[int(tpcid)], dtype=np.float32)
        actual_block = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
        error_block = np.asarray(full_light_std[int(tpcid)], dtype=np.float32)

        current_score += float(compute_error_metric(current_block, actual_block, error_block))

        rebuilt_block = current_block.copy()
        if (int(cluster_a), int(tpcid)) in image_maps:
            wave_a = np.asarray(image_maps[(int(cluster_a), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block - _shift_block(wave_a[None, :, :], int(t0_a))[0],
                0.0,
                None,
            )
        if (int(cluster_b), int(tpcid)) in image_maps:
            wave_b = np.asarray(image_maps[(int(cluster_b), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block - _shift_block(wave_b[None, :, :], int(t0_b))[0],
                0.0,
                None,
            )

        swapped_block = rebuilt_block
        if (int(cluster_a), int(tpcid)) in image_maps:
            wave_a = np.asarray(image_maps[(int(cluster_a), int(tpcid))], dtype=np.float32)
            swapped_block = np.clip(
                swapped_block + _shift_block(wave_a[None, :, :], int(t0_b))[0],
                None,
                float(adc_clip),
            )
        if (int(cluster_b), int(tpcid)) in image_maps:
            wave_b = np.asarray(image_maps[(int(cluster_b), int(tpcid))], dtype=np.float32)
            swapped_block = np.clip(
                swapped_block + _shift_block(wave_b[None, :, :], int(t0_a))[0],
                None,
                float(adc_clip),
            )

        swapped_score += float(compute_error_metric(swapped_block, actual_block, error_block))

    return float(current_score), float(swapped_score)


def _apply_primary_swap_pair(
    *,
    cluster_a: int,
    cluster_b: int,
    t0_a: int,
    t0_b: int,
    union_tpcs: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    protected_track_shower_timestamps: np.ndarray | None,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    adc_clip: float,
    improvement: float,
    trigger_tpcid: int,
    search_range: int,
) -> None:
    for tpcid in union_tpcs:
        rebuilt_block = np.asarray(base_image[int(tpcid)], dtype=np.float32).copy()
        if (int(cluster_a), int(tpcid)) in image_maps:
            wave_a = np.asarray(image_maps[(int(cluster_a), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block - _shift_block(wave_a[None, :, :], int(t0_a))[0],
                0.0,
                None,
            )
        if (int(cluster_b), int(tpcid)) in image_maps:
            wave_b = np.asarray(image_maps[(int(cluster_b), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block - _shift_block(wave_b[None, :, :], int(t0_b))[0],
                0.0,
                None,
            )
        if (int(cluster_a), int(tpcid)) in image_maps:
            wave_a = np.asarray(image_maps[(int(cluster_a), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block + _shift_block(wave_a[None, :, :], int(t0_b))[0],
                None,
                float(adc_clip),
            )
        if (int(cluster_b), int(tpcid)) in image_maps:
            wave_b = np.asarray(image_maps[(int(cluster_b), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block + _shift_block(wave_b[None, :, :], int(t0_a))[0],
                None,
                float(adc_clip),
            )
        base_image[int(tpcid)] = rebuilt_block

    mask_a = np.asarray(labels_global == int(cluster_a), dtype=bool)
    mask_b = np.asarray(labels_global == int(cluster_b), dtype=bool)
    hit_timestamps[mask_a] = float(t0_b)
    hit_timestamps[mask_b] = float(t0_a)

    if protected_track_shower_timestamps is not None:
        info_a = assignment_info.get((int(cluster_a), int(trigger_tpcid)), {})
        info_b = assignment_info.get((int(cluster_b), int(trigger_tpcid)), {})
        if str(info_a.get("stage", "")) == "track":
            protected_track_shower_timestamps[mask_a] = float(t0_b)
        if str(info_b.get("stage", "")) == "track":
            protected_track_shower_timestamps[mask_b] = float(t0_a)

    for clusterid, new_t0, partner, old_t0 in (
        (int(cluster_a), int(t0_b), int(cluster_b), int(t0_a)),
        (int(cluster_b), int(t0_a), int(cluster_a), int(t0_b)),
    ):
        cluster_tpcs = _cluster_image_tpcs(
            clusterid=int(clusterid),
            cluster_to_tpcs=cluster_to_tpcs,
            image_maps=image_maps,
            max_tpcs=int(base_image.shape[0]),
        )
        for tpcid in cluster_tpcs:
            append_candidate_t0(t0_candidates[int(tpcid)], int(new_t0), max_t0=int(search_range))
            info = dict(assignment_info.get((int(clusterid), int(tpcid)), {}))
            info["t0"] = float(new_t0)
            info["swap_partner"] = int(partner)
            info["swap_trigger_tpcid"] = int(trigger_tpcid)
            info["swap_old_t0"] = float(old_t0)
            info["swap_new_t0"] = float(new_t0)
            info["swap_improvement"] = float(improvement)
            info["swap_stage"] = "primary_swap"
            info["mode"] = f"{info.get('mode', 'assigned')}_swapped"
            assignment_info[(int(clusterid), int(tpcid))] = info


def _run_primary_t0_swap_stage(
    *,
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    protected_track_shower_timestamps: np.ndarray | None,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    cluster_energies: dict[int, float],
    large_cluster_energy_mev: float,
    adc_clip: float,
    search_range: int,
    energy_similarity_fraction: float,
    max_passes_per_tpc: int,
    improvement_eps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    stats_by_tpc: dict[int, dict[str, Any]] = {}
    swapped_clusters: set[int] = set()
    total_pairs_considered = 0

    for tpcid in range(int(base_image.shape[0])):
        passes_done = 0
        accepted_here = 0
        considered_here = 0

        while passes_done < int(max_passes_per_tpc):
            eligible_clusters = sorted(
                {
                    int(clusterid)
                    for (clusterid, info_tpc), info in assignment_info.items()
                    if int(info_tpc) == int(tpcid)
                    and _is_primary_swap_eligible(
                        clusterid=int(clusterid),
                        tpcid=int(tpcid),
                        assignment_info=assignment_info,
                        cluster_energies=cluster_energies,
                        large_cluster_energy_mev=float(large_cluster_energy_mev),
                    )
                }
            )
            if len(eligible_clusters) < 2:
                break

            best_swap: dict[str, Any] | None = None

            for idx_a in range(len(eligible_clusters)):
                cluster_a = int(eligible_clusters[idx_a])
                energy_a = float(cluster_energies.get(int(cluster_a), 0.0))
                t0_a = _cluster_assigned_t0(
                    clusterid=int(cluster_a),
                    cluster_to_tpcs=cluster_to_tpcs,
                    image_maps=image_maps,
                    assignment_info=assignment_info,
                    max_tpcs=int(base_image.shape[0]),
                )
                if t0_a is None:
                    continue

                for idx_b in range(idx_a + 1, len(eligible_clusters)):
                    cluster_b = int(eligible_clusters[idx_b])
                    energy_b = float(cluster_energies.get(int(cluster_b), 0.0))
                    rel_diff = abs(float(energy_a) - float(energy_b)) / max(max(float(energy_a), float(energy_b)), 1e-6)
                    if rel_diff > float(energy_similarity_fraction):
                        continue

                    t0_b = _cluster_assigned_t0(
                        clusterid=int(cluster_b),
                        cluster_to_tpcs=cluster_to_tpcs,
                        image_maps=image_maps,
                        assignment_info=assignment_info,
                        max_tpcs=int(base_image.shape[0]),
                    )
                    if t0_b is None or int(t0_a) == int(t0_b):
                        continue

                    union_tpcs = sorted(
                        set(
                            _cluster_image_tpcs(
                                clusterid=int(cluster_a),
                                cluster_to_tpcs=cluster_to_tpcs,
                                image_maps=image_maps,
                                max_tpcs=int(base_image.shape[0]),
                            )
                        )
                        | set(
                            _cluster_image_tpcs(
                                clusterid=int(cluster_b),
                                cluster_to_tpcs=cluster_to_tpcs,
                                image_maps=image_maps,
                                max_tpcs=int(base_image.shape[0]),
                            )
                        )
                    )
                    if len(union_tpcs) == 0:
                        continue

                    current_score, swapped_score = _score_primary_swap_pair(
                        cluster_a=int(cluster_a),
                        cluster_b=int(cluster_b),
                        t0_a=int(t0_a),
                        t0_b=int(t0_b),
                        union_tpcs=union_tpcs,
                        image_maps=image_maps,
                        base_image=base_image,
                        full_light_waveform=full_light_waveform,
                        full_light_std=full_light_std,
                        adc_clip=float(adc_clip),
                    )
                    considered_here += 1
                    total_pairs_considered += 1
                    improvement = float(current_score - swapped_score)
                    if improvement <= float(improvement_eps):
                        continue

                    if best_swap is None or improvement > float(best_swap["improvement"]):
                        best_swap = {
                            "tpcid": int(tpcid),
                            "cluster_a": int(cluster_a),
                            "cluster_b": int(cluster_b),
                            "energy_a": float(energy_a),
                            "energy_b": float(energy_b),
                            "t0_a_old": int(t0_a),
                            "t0_b_old": int(t0_b),
                            "t0_a_new": int(t0_b),
                            "t0_b_new": int(t0_a),
                            "relative_energy_diff": float(rel_diff),
                            "current_score": float(current_score),
                            "swapped_score": float(swapped_score),
                            "improvement": float(improvement),
                            "union_tpcs": [int(v) for v in union_tpcs],
                        }

            if best_swap is None:
                break

            _apply_primary_swap_pair(
                cluster_a=int(best_swap["cluster_a"]),
                cluster_b=int(best_swap["cluster_b"]),
                t0_a=int(best_swap["t0_a_old"]),
                t0_b=int(best_swap["t0_b_old"]),
                union_tpcs=[int(v) for v in best_swap["union_tpcs"]],
                cluster_to_tpcs=cluster_to_tpcs,
                image_maps=image_maps,
                base_image=base_image,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                protected_track_shower_timestamps=protected_track_shower_timestamps,
                t0_candidates=t0_candidates,
                assignment_info=assignment_info,
                adc_clip=float(adc_clip),
                improvement=float(best_swap["improvement"]),
                trigger_tpcid=int(tpcid),
                search_range=int(search_range),
            )
            logs.append(dict(best_swap))
            swapped_clusters.add(int(best_swap["cluster_a"]))
            swapped_clusters.add(int(best_swap["cluster_b"]))
            accepted_here += 1
            passes_done += 1

        stats_by_tpc[int(tpcid)] = {
            "pairs_considered": int(considered_here),
            "accepted_swaps": int(accepted_here),
            "max_passes": int(max_passes_per_tpc),
        }

    stats = {
        "pairs_considered": int(total_pairs_considered),
        "accepted_swaps": int(len(logs)),
        "swapped_clusters": sorted(int(cid) for cid in swapped_clusters),
        "stats_by_tpc": {
            int(tpc): dict(values)
            for tpc, values in sorted(stats_by_tpc.items())
            if int(values.get("pairs_considered", 0)) > 0 or int(values.get("accepted_swaps", 0)) > 0
        },
        "swap_log": [dict(item) for item in logs],
    }
    return logs, stats


def assign_small_clusters_v8_intermediate_test(
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
    flash_seed_t0s_by_tpc: dict[int, list[int]] | None = None,
    flash_seed_t0_resolution: int = 5,
    primary_swap_enable: bool = True,
    primary_swap_energy_fraction: float = 0.10,
    primary_swap_max_passes_per_tpc: int = 4,
    primary_swap_improvement_eps: float = 0.0,
    protected_track_shower_timestamps: np.ndarray | None = None,
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
    )
    assignment_log.extend(primary_logs)
    scan_loss_dict.update(primary_scan_updates)

    primary_swap_logs: list[dict[str, Any]] = []
    primary_swap_stats: dict[str, Any] = {
        "pairs_considered": 0,
        "accepted_swaps": 0,
        "swapped_clusters": [],
        "stats_by_tpc": {},
        "swap_log": [],
    }
    if bool(primary_swap_enable):
        primary_swap_logs, primary_swap_stats = _run_primary_t0_swap_stage(
            cluster_to_tpcs=cluster_to_tpcs,
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            protected_track_shower_timestamps=protected_track_shower_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            cluster_energies=cluster_energies,
            large_cluster_energy_mev=float(large_cluster_energy_mev),
            adc_clip=float(adc_clip),
            search_range=int(search_range),
            energy_similarity_fraction=float(primary_swap_energy_fraction),
            max_passes_per_tpc=int(primary_swap_max_passes_per_tpc),
            improvement_eps=float(primary_swap_improvement_eps),
        )

    t0_candidates, merged_flash_t0s_by_tpc, flash_merge_stats = merge_flash_t0_candidates_after_primary(
        t0_candidates=t0_candidates,
        flash_seed_t0s_by_tpc=flash_seed_t0s_by_tpc,
        candidate_min_sep=int(flash_seed_t0_resolution),
        max_t0=int(search_range),
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
        "primary_swap_enabled": bool(primary_swap_enable),
        "primary_swap_energy_fraction": float(primary_swap_energy_fraction),
        "primary_swap_max_passes_per_tpc": int(primary_swap_max_passes_per_tpc),
        "primary_swap_improvement_eps": float(primary_swap_improvement_eps),
        "primary_swap_logs": [dict(item) for item in primary_swap_logs],
        "primary_swap_stats": dict(primary_swap_stats),
        "band_anchor_sequence_by_tpc": {
            int(tpc): [int(cid) for cid in sequence]
            for tpc, sequence in sorted(band_anchor_sequence_by_tpc.items())
        },
        "flash_seed_t0s_input_by_tpc": {
            int(tpc): [int(v) for v in values]
            for tpc, values in sorted((flash_seed_t0s_by_tpc or {}).items())
        },
        "flash_seed_t0s_merged_by_tpc": {
            int(tpc): [int(v) for v in values]
            for tpc, values in sorted(merged_flash_t0s_by_tpc.items())
        },
        "flash_merge_stats": dict(flash_merge_stats),
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
    "assign_small_clusters_v8_intermediate_test",
    "rebalance_step4_clusters_v4",
    "assign_small_clusters_v10_1",
]


assign_small_clusters_v10_1 = assign_small_clusters_v8_intermediate_test
