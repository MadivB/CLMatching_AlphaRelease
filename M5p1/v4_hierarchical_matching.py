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

try:
    from v3_3_global_matching import _collect_tpc_cluster_cache, _residual_peaks_for_tpc
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v3_3_global_matching import _collect_tpc_cluster_cache, _residual_peaks_for_tpc


def _top_separated_scan_minima(
    loss_curve: np.ndarray,
    *,
    max_candidates: int,
    min_sep: int,
    max_t0: int,
    extra_t0s: list[int] | None = None,
) -> list[int]:
    curve = np.asarray(loss_curve, dtype=np.float32)
    if curve.size == 0:
        return []

    chosen: list[int] = []
    order = np.argsort(curve)
    for idx in order:
        t0 = int(np.clip(int(idx), 0, int(max_t0)))
        if any(abs(int(t0) - int(prev)) < int(min_sep) for prev in chosen):
            continue
        chosen.append(int(t0))
        if len(chosen) >= int(max_candidates):
            break

    if extra_t0s is not None:
        for t0 in extra_t0s:
            t0 = int(np.clip(np.rint(t0), 0, int(max_t0)))
            if any(abs(int(t0) - int(prev)) < int(max(1, min_sep // 2)) for prev in chosen):
                continue
            chosen.append(int(t0))

    return sorted(set(int(t0) for t0 in chosen))


def _dedupe_states(states: list[dict[str, Any]], beam_width: int) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    for state in states:
        key = tuple(sorted((int(cid), int(t0)) for cid, t0 in state["assignments"].items()))
        prev = best_by_key.get(key)
        if prev is None or float(state["score"]) < float(prev["score"]):
            best_by_key[key] = state
    kept = sorted(best_by_key.values(), key=lambda item: float(item["score"]))
    return kept[: max(1, int(beam_width))]


def _local_refine_collective_state(
    assignments: dict[int, int],
    *,
    cluster_images: dict[int, np.ndarray],
    candidate_map: dict[int, list[int]],
    base_seed: np.ndarray,
    actual_tpc: np.ndarray,
    error_tpc: np.ndarray,
    adc_clip: float,
    max_iterations: int,
    relax_eps: float,
) -> tuple[dict[int, int], np.ndarray, float, list[dict[str, Any]]]:
    assignments = {int(cid): int(t0) for cid, t0 in assignments.items()}
    model = np.asarray(base_seed, dtype=np.float32).copy()
    for clusterid, t0 in assignments.items():
        shifted = _shift_block(cluster_images[int(clusterid)][None, :, :], int(t0))[0]
        model = np.clip(model + shifted, None, adc_clip)
    score = compute_error_metric(model, actual_tpc, error_tpc)

    relax_log: list[dict[str, Any]] = []
    order = sorted(assignments, key=int)
    for _ in range(int(max_iterations)):
        changed = False
        for clusterid in order:
            old_t0 = int(assignments[int(clusterid)])
            cluster_image = np.asarray(cluster_images[int(clusterid)], dtype=np.float32)
            old_shifted = _shift_block(cluster_image[None, :, :], int(old_t0))[0]
            base_without = np.clip(model - old_shifted, 0.0, None)

            best_t0 = int(old_t0)
            best_model = model
            best_score = float(score)
            for t0 in candidate_map.get(int(clusterid), []):
                shifted = _shift_block(cluster_image[None, :, :], int(t0))[0]
                candidate_model = np.clip(base_without + shifted, None, adc_clip)
                candidate_score = compute_error_metric(candidate_model, actual_tpc, error_tpc)
                if float(candidate_score) + float(relax_eps) < float(best_score):
                    best_t0 = int(t0)
                    best_model = candidate_model
                    best_score = float(candidate_score)

            if int(best_t0) != int(old_t0):
                assignments[int(clusterid)] = int(best_t0)
                model = np.asarray(best_model, dtype=np.float32)
                score = float(best_score)
                changed = True
                relax_log.append(
                    {
                        "clusterid": int(clusterid),
                        "old_t0": int(old_t0),
                        "new_t0": int(best_t0),
                        "score_after": float(score),
                    }
                )
        if not changed:
            break

    return assignments, model.astype(np.float32), float(score), relax_log


def _collective_large_tpc_assign(
    *,
    tpcid: int,
    cluster_ids: list[int],
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
    scan_top_k: int,
    scan_min_sep: int,
    beam_width: int,
    relax_iterations: int,
    relax_eps: float,
    collect_scan_losses: bool,
    debug_enabled: bool = False,
    max_saved_beam_states: int = 8,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    scan_updates: dict[int, dict[str, Any]] = {}

    if len(cluster_ids) == 0:
        return logs, scan_updates, {
            "assigned_clusters": [],
            "ordering_used": [],
            "candidate_counts": {},
            "final_score": None,
            "relaxation_log": [],
        }

    actual_tpc = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
    error_tpc = np.asarray(full_light_std[int(tpcid)], dtype=np.float32)
    base_seed = np.asarray(base_image[int(tpcid)], dtype=np.float32).copy()
    initial_score = compute_error_metric(base_seed, actual_tpc, error_tpc)

    cluster_images: dict[int, np.ndarray] = {}
    candidate_map: dict[int, list[int]] = {}
    scan_minimum_t0: dict[int, int] = {}
    individual_gains: dict[int, float] = {}
    profile_strengths: dict[int, float] = {}
    debug_info: dict[str, Any] = {
        "tpcid": int(tpcid),
        "initial_score": float(initial_score),
        "cluster_ids": [int(cid) for cid in cluster_ids],
        "scan_minimum_t0": {},
        "candidate_t0s": {},
        "individual_gains": {},
        "profile_strengths": {},
        "global_shared_candidates": [],
        "orderings": [],
        "beam_history": [],
        "final_assignments": {},
        "final_score": None,
        "relaxation_log": [],
    }

    for clusterid in cluster_ids:
        cluster_image = np.asarray(image_maps[(int(clusterid), int(tpcid))], dtype=np.float32)
        cluster_images[int(clusterid)] = cluster_image
        cluster_block = cluster_image[None, :, :]
        best_t0, best_score, loss_curve = _scan_best_shift_multi(
            cluster_block,
            base_seed[None, :, :],
            actual_tpc[None, :, :],
            error_tpc[None, :, :],
            search_range=search_range,
            adc_clip=adc_clip,
            return_curve=True,
        )
        extra_t0s = [int(t0) for t0 in t0_candidates[int(tpcid)]]
        candidate_t0s = _top_separated_scan_minima(
            loss_curve if loss_curve is not None else np.asarray([], dtype=np.float32),
            max_candidates=scan_top_k,
            min_sep=scan_min_sep,
            max_t0=search_range,
            extra_t0s=extra_t0s,
        )
        if int(best_t0) not in candidate_t0s:
            candidate_t0s.append(int(best_t0))
            candidate_t0s = sorted(set(int(t0) for t0 in candidate_t0s))
        candidate_map[int(clusterid)] = candidate_t0s
        scan_minimum_t0[int(clusterid)] = int(best_t0)
        individual_gains[int(clusterid)] = float(initial_score - float(best_score))
        profile_strengths[int(clusterid)] = float(np.sum(np.clip(cluster_image, 0.0, None)))
        debug_info["scan_minimum_t0"][int(clusterid)] = int(best_t0)
        debug_info["candidate_t0s"][int(clusterid)] = [int(t0) for t0 in candidate_t0s]
        debug_info["individual_gains"][int(clusterid)] = float(individual_gains[int(clusterid)])
        debug_info["profile_strengths"][int(clusterid)] = float(profile_strengths[int(clusterid)])

        if collect_scan_losses:
            scan_updates[int(clusterid)] = _build_scan_loss_entry(
                clusterid=int(clusterid),
                stage="collective_large_scan",
                mode="collective_large_scan",
                tpcs=[int(tpcid)],
                energy=float(cluster_energies.get(int(clusterid), 0.0)),
                best_t0=int(best_t0),
                assigned=False,
                search_range=search_range,
                loss_curve=loss_curve,
                best_t0_scan=int(best_t0),
            )

    shared_candidates = sorted(
        {
            int(t0)
            for t0 in t0_candidates[int(tpcid)]
        }
        | {
            int(t0)
            for values in candidate_map.values()
            for t0 in values
        }
        | {
            int(t0)
            for t0 in scan_minimum_t0.values()
        }
    )
    for clusterid in cluster_ids:
        augmented = sorted(
            {
                int(t0)
                for t0 in candidate_map.get(int(clusterid), [])
            }
            | {
                int(t0)
                for t0 in shared_candidates
            }
        )
        candidate_map[int(clusterid)] = augmented
        debug_info["candidate_t0s"][int(clusterid)] = [int(t0) for t0 in augmented]
    debug_info["global_shared_candidates"] = [int(t0) for t0 in shared_candidates]

    orderings: list[list[int]] = []
    orderings.append(
        sorted(
            cluster_ids,
            key=lambda cid: (float(individual_gains.get(int(cid), 0.0)), float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
            reverse=True,
        )
    )
    orderings.append(
        sorted(
            cluster_ids,
            key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), float(individual_gains.get(int(cid), 0.0)), -int(cid)),
            reverse=True,
        )
    )
    orderings.append(
        sorted(
            cluster_ids,
            key=lambda cid: (float(profile_strengths.get(int(cid), 0.0)), float(individual_gains.get(int(cid), 0.0)), -int(cid)),
            reverse=True,
        )
    )
    orderings.append(
        sorted(
            cluster_ids,
            key=lambda cid: (int(scan_minimum_t0.get(int(cid), 0)), -float(cluster_energies.get(int(cid), 0.0)), int(cid)),
        )
    )
    orderings.append(
        sorted(
            cluster_ids,
            key=lambda cid: (-int(scan_minimum_t0.get(int(cid), 0)), -float(cluster_energies.get(int(cid), 0.0)), int(cid)),
        )
    )

    unique_orderings: list[list[int]] = []
    seen_orders: set[tuple[int, ...]] = set()
    for ordering in orderings:
        key = tuple(int(cid) for cid in ordering)
        if key in seen_orders:
            continue
        seen_orders.add(key)
        unique_orderings.append([int(cid) for cid in ordering])

    best_state: dict[str, Any] | None = None
    chosen_ordering: list[int] = []
    best_relax_log: list[dict[str, Any]] = []

    for ordering in unique_orderings:
        debug_info["orderings"].append([int(cid) for cid in ordering])
        beam = [
            {
                "assignments": {},
                "model": base_seed.copy(),
                "score": float(initial_score),
            }
        ]
        for clusterid in ordering:
            expanded: list[dict[str, Any]] = []
            cluster_image = cluster_images[int(clusterid)]
            for state in beam:
                model = np.asarray(state["model"], dtype=np.float32)
                for t0 in candidate_map.get(int(clusterid), []):
                    shifted = _shift_block(cluster_image[None, :, :], int(t0))[0]
                    candidate_model = np.clip(model + shifted, None, adc_clip)
                    candidate_score = compute_error_metric(candidate_model, actual_tpc, error_tpc)
                    assignments = dict(state["assignments"])
                    assignments[int(clusterid)] = int(t0)
                    expanded.append(
                        {
                            "assignments": assignments,
                            "model": candidate_model.astype(np.float32),
                            "score": float(candidate_score),
                        }
                    )
            beam = _dedupe_states(expanded, beam_width)
            if debug_enabled:
                debug_info["beam_history"].append(
                    {
                        "ordering": [int(cid) for cid in ordering],
                        "inserted_cluster": int(clusterid),
                        "top_states": [
                            {
                                "score": float(state["score"]),
                                "assignments": {
                                    int(cid): int(t0) for cid, t0 in sorted(state["assignments"].items())
                                },
                            }
                            for state in beam[: max(1, int(max_saved_beam_states))]
                        ],
                    }
                )

        for state in beam:
            refined_assignments, refined_model, refined_score, relax_log = _local_refine_collective_state(
                state["assignments"],
                cluster_images=cluster_images,
                candidate_map=candidate_map,
                base_seed=base_seed,
                actual_tpc=actual_tpc,
                error_tpc=error_tpc,
                adc_clip=adc_clip,
                max_iterations=relax_iterations,
                relax_eps=relax_eps,
            )
            if best_state is None or float(refined_score) < float(best_state["score"]):
                best_state = {
                    "assignments": refined_assignments,
                    "model": refined_model.astype(np.float32),
                    "score": float(refined_score),
                }
                chosen_ordering = [int(cid) for cid in ordering]
                best_relax_log = relax_log

    if best_state is None:
        return logs, scan_updates, {
            "assigned_clusters": [],
            "ordering_used": [],
            "candidate_counts": {int(cid): len(candidate_map.get(int(cid), [])) for cid in cluster_ids},
            "final_score": None,
            "relaxation_log": [],
        }

    base_image[int(tpcid)] = np.asarray(best_state["model"], dtype=np.float32)
    final_score = float(best_state["score"])
    final_assignments = {int(cid): int(t0) for cid, t0 in best_state["assignments"].items()}

    for clusterid in cluster_ids:
        final_t0 = int(final_assignments[int(clusterid)])
        hit_timestamps[labels_global == int(clusterid)] = float(final_t0)
        append_candidate_t0(t0_candidates[int(tpcid)], int(final_t0), max_t0=search_range)
        cluster_image = cluster_images[int(clusterid)]
        shifted = _shift_block(cluster_image[None, :, :], int(final_t0))[0]
        base_without = np.clip(np.asarray(best_state["model"], dtype=np.float32) - shifted, 0.0, None)
        score_without = compute_error_metric(base_without, actual_tpc, error_tpc)
        improvement = float(score_without - final_score)

        mode = "collective_large_tpc"
        if any(int(item["clusterid"]) == int(clusterid) for item in best_relax_log):
            mode = "collective_large_tpc_relaxed"

        assignment_info[(int(clusterid), int(tpcid))] = {
            "stage": "collective_large_tpc",
            "mode": str(mode),
            "t0": float(final_t0),
            "energy": float(cluster_energies.get(int(clusterid), 0.0)),
            "assigned": True,
            "error_after": float(final_score),
            "improvement": float(improvement),
            "ordering_signature": [int(cid) for cid in chosen_ordering],
        }
        if int(clusterid) in unassigned_by_tpc[int(tpcid)]:
            unassigned_by_tpc[int(tpcid)] = [
                cid for cid in unassigned_by_tpc[int(tpcid)] if int(cid) != int(clusterid)
            ]

        logs.append(
            {
                "clusterid": int(clusterid),
                "tpcs": [int(tpcid)],
                "energy": float(cluster_energies.get(int(clusterid), 0.0)),
                "assigned": True,
                "mode": str(mode),
                "label": "collective_large_tpc",
                "t0": int(final_t0),
                "raw_t0": int(scan_minimum_t0[int(clusterid)]),
                "improvement": float(improvement),
            }
        )

        if collect_scan_losses and int(clusterid) in scan_updates:
            scan_updates[int(clusterid)]["assigned"] = True
            scan_updates[int(clusterid)]["best_t0"] = int(final_t0)
            scan_updates[int(clusterid)]["mode"] = str(mode)

    debug_info["final_assignments"] = {int(cid): int(t0) for cid, t0 in sorted(final_assignments.items())}
    debug_info["final_score"] = float(final_score)
    debug_info["relaxation_log"] = best_relax_log

    return logs, scan_updates, {
        "assigned_clusters": [int(cid) for cid in cluster_ids],
        "ordering_used": [int(cid) for cid in chosen_ordering],
        "candidate_counts": {int(cid): len(candidate_map.get(int(cid), [])) for cid in cluster_ids},
        "final_score": float(final_score),
        "relaxation_log": best_relax_log,
        "debug": debug_info,
    }


def _assign_remaining_clusters_error_matrix(
    *,
    tpcid: int,
    cluster_ids: list[int],
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
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[int], list[int], list[int]]:
    logs: list[dict[str, Any]] = []
    scan_updates: dict[int, dict[str, Any]] = {}
    assigned_clusters: list[int] = []
    nonimproving_clusters: list[int] = []
    seed_scan_clusters: list[int] = []

    if len(cluster_ids) == 0:
        return logs, scan_updates, assigned_clusters, nonimproving_clusters, seed_scan_clusters

    candidate_grid = sorted(int(t0) for t0 in t0_candidates[int(tpcid)])
    remaining_clusters = np.asarray([int(cid) for cid in cluster_ids], dtype=int)

    if len(candidate_grid) == 0:
        seed_cluster = max(
            remaining_clusters.tolist(),
            key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
        )
        cluster_energy = float(cluster_energies.get(int(seed_cluster), 0.0))
        accepted, log, scan_entry = _full_scan_assign(
            int(seed_cluster),
            np.asarray([int(tpcid)], dtype=int),
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
            stage="error_matrix_seed_scan",
            accepted_mode="error_matrix_seed_scan",
            forced_mode="error_matrix_seed_scan_forced",
            rejected_mode="error_matrix_seed_scan_unassigned",
            assignment_improvement_eps=0.0,
            search_range=search_range,
            adc_clip=adc_clip,
            t0_resolution=5,
            pulse_peak_tick=105,
            collect_scan_losses=collect_scan_losses,
        )
        logs.append(log)
        if scan_entry is not None:
            scan_updates[int(seed_cluster)] = scan_entry
        if accepted:
            assigned_clusters.append(int(seed_cluster))
            seed_scan_clusters.append(int(seed_cluster))
            candidate_grid = sorted(int(t0) for t0 in t0_candidates[int(tpcid)])
            remaining_clusters = np.asarray(
                [int(cid) for cid in remaining_clusters if int(cid) != int(seed_cluster)],
                dtype=int,
            )
        else:
            candidate_grid = []

    if len(remaining_clusters) == 0 or len(candidate_grid) == 0:
        return logs, scan_updates, assigned_clusters, nonimproving_clusters, seed_scan_clusters

    placed_mask = np.zeros(len(remaining_clusters), dtype=bool)
    actual_tpc = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
    error_tpc = np.asarray(full_light_std[int(tpcid)], dtype=np.float32)
    current_error = compute_error_metric(np.asarray(base_image[int(tpcid)], dtype=np.float32), actual_tpc, error_tpc)

    while True:
        loss_matrix, remaining_now = _loss_matrix_single_tpc(
            image_maps,
            actual_tpc,
            np.asarray(base_image[int(tpcid)], dtype=np.float32),
            error_tpc,
            tpcid=int(tpcid),
            clusters=remaining_clusters,
            placed_mask=placed_mask,
            t0_candidates=candidate_grid,
            adc_clip=adc_clip,
        )
        if loss_matrix.size == 0 or len(remaining_now) == 0:
            break

        best_flat = int(np.argmin(loss_matrix))
        best_cluster_idx = int(best_flat // loss_matrix.shape[1])
        best_t0_idx = int(best_flat % loss_matrix.shape[1])
        clusterid = int(remaining_now[best_cluster_idx])
        opt_t0 = int(candidate_grid[best_t0_idx])
        orig_idx = int(np.where(remaining_clusters == clusterid)[0][0])
        placed_mask[orig_idx] = True

        cluster_block = np.asarray(image_maps[(int(clusterid), int(tpcid))], dtype=np.float32)[None, :, :]
        shifted = _shift_block(cluster_block, int(opt_t0))[0]
        base_tpc = np.asarray(base_image[int(tpcid)], dtype=np.float32)
        candidate_model = np.clip(base_tpc + shifted, None, adc_clip)
        candidate_error = compute_error_metric(candidate_model, actual_tpc, error_tpc)
        improvement = float(current_error - candidate_error)
        mode = "error_matrix"
        if float(improvement) <= 0.0:
            mode = "error_matrix_nonimproving"
            nonimproving_clusters.append(int(clusterid))

        _assign_cluster_at_t0(
            int(clusterid),
            np.asarray([int(tpcid)], dtype=int),
            int(opt_t0),
            cluster_block=cluster_block,
            base_image=base_image,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energy=float(cluster_energies.get(int(clusterid), 0.0)),
            mode=str(mode),
            stage="error_matrix",
            error_after=float(candidate_error),
            improvement=float(improvement),
            adc_clip=adc_clip,
            max_t0=search_range,
        )
        logs.append(
            {
                "clusterid": int(clusterid),
                "tpcs": [int(tpcid)],
                "energy": float(cluster_energies.get(int(clusterid), 0.0)),
                "assigned": True,
                "mode": str(mode),
                "label": "error_matrix",
                "t0": int(opt_t0),
                "improvement": float(improvement),
            }
        )
        if collect_scan_losses:
            scan_updates[int(clusterid)] = _build_scan_loss_entry(
                clusterid=int(clusterid),
                stage="error_matrix",
                mode=str(mode),
                tpcs=[int(tpcid)],
                energy=float(cluster_energies.get(int(clusterid), 0.0)),
                best_t0=int(opt_t0),
                assigned=True,
                search_range=search_range,
                loss_curve=None,
                best_t0_scan=None,
            )

        assigned_clusters.append(int(clusterid))
        current_error = float(candidate_error)

    return logs, scan_updates, assigned_clusters, nonimproving_clusters, seed_scan_clusters


def assign_small_clusters_v4(
    cluster_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
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
    *,
    large_cluster_energy_mev: float = 10.0,
    minimum_error_matrix_energy_mev: float = 0.0,
    collective_scan_top_k: int = 4,
    collective_scan_min_sep: int = 6,
    collective_beam_width: int = 6,
    collective_relax_iterations: int = 3,
    collective_relax_eps: float = 2e-4,
    search_range: int = 800,
    adc_clip: float = 60780.0,
    collect_scan_losses: bool = False,
    debug_tpcs: list[int] | None = None,
    max_saved_beam_states: int = 8,
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

    multi_tpc_clusters = sorted(
        [cid for cid, tpcs in active_cluster_tpcs.items() if len(tpcs) > 1],
        key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
        reverse=True,
    )
    single_tpc_by_tpc: dict[int, list[int]] = {}
    for clusterid, tpcs in active_cluster_tpcs.items():
        if len(tpcs) == 1:
            single_tpc_by_tpc.setdefault(int(tpcs[0]), []).append(int(clusterid))

    placed_clusters: set[int] = set()
    collective_large_clusters: list[int] = []
    error_matrix_clusters: list[int] = []
    error_matrix_nonimproving_clusters: list[int] = []
    error_matrix_seed_scans: list[int] = []
    pruned_error_matrix_clusters: list[int] = []
    collective_relax_log: list[dict[str, Any]] = []
    collective_tpc_scores: dict[int, float] = {}
    collective_orderings: dict[int, list[int]] = {}
    collective_debug: dict[int, dict[str, Any]] = {}
    debug_tpc_set = {int(tpc) for tpc in (debug_tpcs or [])}

    for clusterid in multi_tpc_clusters:
        tpcs = active_cluster_tpcs[int(clusterid)]
        cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
        accepted, log, scan_entry = _full_scan_assign(
            int(clusterid),
            tpcs,
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
            stage="multi_tpc_full_scan",
            accepted_mode="multi_tpc_full_scan",
            forced_mode="multi_tpc_full_scan_forced",
            rejected_mode="multi_tpc_unassigned",
            assignment_improvement_eps=0.0,
            search_range=search_range,
            adc_clip=adc_clip,
            t0_resolution=5,
            pulse_peak_tick=105,
            collect_scan_losses=collect_scan_losses,
        )
        assignment_log.append(log)
        if scan_entry is not None:
            scan_loss_dict[int(clusterid)] = scan_entry
        if accepted:
            placed_clusters.add(int(clusterid))

    for tpcid in sorted(single_tpc_by_tpc):
        clusters_here = sorted(
            single_tpc_by_tpc[int(tpcid)],
            key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
            reverse=True,
        )

        large_clusters = [
            int(cid)
            for cid in clusters_here
            if float(cluster_energies.get(int(cid), 0.0)) > float(large_cluster_energy_mev)
        ]
        remaining_clusters = [
            int(cid)
            for cid in clusters_here
            if int(cid) not in large_clusters
        ]

        if len(large_clusters) > 0:
            logs, scan_updates, collective_stats = _collective_large_tpc_assign(
                tpcid=int(tpcid),
                cluster_ids=[int(cid) for cid in large_clusters],
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
                search_range=search_range,
                adc_clip=adc_clip,
                scan_top_k=collective_scan_top_k,
                scan_min_sep=collective_scan_min_sep,
                beam_width=collective_beam_width,
                relax_iterations=collective_relax_iterations,
                relax_eps=collective_relax_eps,
                collect_scan_losses=collect_scan_losses,
                debug_enabled=(int(tpcid) in debug_tpc_set),
                max_saved_beam_states=max_saved_beam_states,
            )
            assignment_log.extend(logs)
            scan_loss_dict.update(scan_updates)
            placed_clusters.update(int(cid) for cid in collective_stats.get("assigned_clusters", []))
            collective_large_clusters.extend(int(cid) for cid in collective_stats.get("assigned_clusters", []))
            collective_relax_log.extend(collective_stats.get("relaxation_log", []))
            if collective_stats.get("final_score") is not None:
                collective_tpc_scores[int(tpcid)] = float(collective_stats["final_score"])
            collective_orderings[int(tpcid)] = [int(cid) for cid in collective_stats.get("ordering_used", [])]
            if int(tpcid) in debug_tpc_set and collective_stats.get("debug") is not None:
                collective_debug[int(tpcid)] = collective_stats["debug"]

        if float(minimum_error_matrix_energy_mev) > 0.0:
            next_remaining: list[int] = []
            for clusterid in remaining_clusters:
                cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
                if cluster_energy < float(minimum_error_matrix_energy_mev):
                    pruned_error_matrix_clusters.append(int(clusterid))
                    _mark_cluster_unassigned(
                        int(clusterid),
                        np.asarray([int(tpcid)], dtype=int),
                        labels_global=labels_global,
                        hit_timestamps=hit_timestamps,
                        assignment_info=assignment_info,
                        unassigned_by_tpc=unassigned_by_tpc,
                        cluster_energy=cluster_energy,
                        mode="below_error_matrix_energy_threshold",
                        stage="error_matrix_preselection",
                    )
                    assignment_log.append(
                        {
                            "clusterid": int(clusterid),
                            "tpcs": [int(tpcid)],
                            "energy": float(cluster_energy),
                            "assigned": False,
                            "mode": "below_error_matrix_energy_threshold",
                            "label": "error_matrix_preselection",
                            "t0": -1,
                            "improvement": 0.0,
                        }
                    )
                    if collect_scan_losses:
                        scan_loss_dict[int(clusterid)] = _build_scan_loss_entry(
                            clusterid=int(clusterid),
                            stage="error_matrix_preselection",
                            mode="below_error_matrix_energy_threshold",
                            tpcs=[int(tpcid)],
                            energy=float(cluster_energy),
                            best_t0=None,
                            assigned=False,
                            search_range=search_range,
                            loss_curve=None,
                            best_t0_scan=None,
                        )
                else:
                    next_remaining.append(int(clusterid))
            remaining_clusters = next_remaining

        logs, scan_updates, assigned_step4, nonimproving_step4, seed_scans = _assign_remaining_clusters_error_matrix(
            tpcid=int(tpcid),
            cluster_ids=[int(cid) for cid in remaining_clusters],
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
            search_range=search_range,
            adc_clip=adc_clip,
            collect_scan_losses=collect_scan_losses,
        )
        assignment_log.extend(logs)
        scan_loss_dict.update(scan_updates)
        placed_clusters.update(int(cid) for cid in assigned_step4)
        error_matrix_clusters.extend(int(cid) for cid in assigned_step4)
        error_matrix_nonimproving_clusters.extend(int(cid) for cid in nonimproving_step4)
        error_matrix_seed_scans.extend(int(cid) for cid in seed_scans)

    stage_stats = {
        "multi_tpc_clusters": [int(cid) for cid in multi_tpc_clusters],
        "collective_large_clusters": sorted(set(int(cid) for cid in collective_large_clusters)),
        "error_matrix_clusters": sorted(set(int(cid) for cid in error_matrix_clusters)),
        "error_matrix_nonimproving_clusters": sorted(set(int(cid) for cid in error_matrix_nonimproving_clusters)),
        "error_matrix_seed_scans": sorted(set(int(cid) for cid in error_matrix_seed_scans)),
        "pruned_error_matrix_clusters": sorted(set(int(cid) for cid in pruned_error_matrix_clusters)),
        "step4_clusters": sorted(set(int(cid) for cid in error_matrix_clusters)),
        "step4_assigned_clusters": sorted(set(int(cid) for cid in error_matrix_clusters)),
        "collective_relaxation_log": collective_relax_log,
        "collective_relaxed_clusters": sorted(set(int(item["clusterid"]) for item in collective_relax_log)),
        "collective_tpc_scores": {int(tpc): float(score) for tpc, score in sorted(collective_tpc_scores.items())},
        "collective_orderings": {int(tpc): [int(cid) for cid in order] for tpc, order in sorted(collective_orderings.items())},
        "collective_debug": collective_debug,
        "large_cluster_energy_mev": float(large_cluster_energy_mev),
        "minimum_error_matrix_energy_mev": float(minimum_error_matrix_energy_mev),
        "collective_scan_top_k": int(collective_scan_top_k),
        "collective_scan_min_sep": int(collective_scan_min_sep),
        "collective_beam_width": int(collective_beam_width),
        "collective_relax_iterations": int(collective_relax_iterations),
        "collective_relax_eps": float(collective_relax_eps),
        "debug_tpcs": sorted(int(tpc) for tpc in debug_tpc_set),
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


def rebalance_step4_clusters_v4(
    cluster_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    cluster_energies: dict[int, float],
    *,
    max_iterations: int = 6,
    max_moves_per_cluster: int = 2,
    max_peaks_per_sign: int = 2,
    min_peak_sep: int = 12,
    peak_window: int = 20,
    min_peak_fraction: float = 0.20,
    smooth_width: int = 11,
    min_move_ticks: int = 4,
    min_improvement: float = 1e-4,
    search_range: int | None = None,
    adc_clip: float = 60780.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if search_range is None:
        search_range = int(full_light_waveform.shape[-1] - 1)

    rebalance_log: list[dict[str, Any]] = []
    move_counts: dict[int, int] = {}
    visited_t0s: dict[int, set[int]] = {}
    active_tpcs = sorted(
        {
            int(tpc)
            for clusterid in cluster_labels
            for tpc in cluster_to_tpcs.get(int(clusterid), [])
            if int(tpc) < int(base_image.shape[0]) and (int(clusterid), int(tpc)) in image_maps
        }
    )

    for clusterid in cluster_labels:
        cluster_mask = labels_global == int(clusterid)
        current_t0s = hit_timestamps[cluster_mask]
        current_t0s = current_t0s[np.isfinite(current_t0s)]
        if current_t0s.size == 0:
            continue
        visited_t0s[int(clusterid)] = {int(np.rint(float(current_t0s[0])))}

    iterations_run = 0
    tpcs_with_candidates: set[int] = set()
    moved_clusters: list[int] = []

    for iteration in range(int(max_iterations)):
        iterations_run = iteration + 1
        best_move: dict[str, Any] | None = None
        best_move_meta: dict[str, Any] | None = None

        for tpcid in active_tpcs:
            movable_clusters = []
            for clusterid in cluster_labels:
                if int(tpcid) not in cluster_to_tpcs.get(int(clusterid), []):
                    continue
                info = assignment_info.get((int(clusterid), int(tpcid)))
                if info is None or not info.get("assigned", False):
                    continue
                if move_counts.get(int(clusterid), 0) >= int(max_moves_per_cluster):
                    continue
                movable_clusters.append(int(clusterid))

            if len(movable_clusters) == 0:
                continue

            actual_tpc = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
            predicted_tpc = np.asarray(base_image[int(tpcid)], dtype=np.float32)
            _, _, deficit_peaks, excess_peaks = _residual_peaks_for_tpc(
                actual_tpc,
                predicted_tpc,
                smooth_width=smooth_width,
                max_peaks_per_sign=max_peaks_per_sign,
                min_peak_sep=min_peak_sep,
                min_peak_fraction=min_peak_fraction,
            )
            cluster_cache = _collect_tpc_cluster_cache(
                int(tpcid),
                movable_clusters,
                image_maps=image_maps,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
            )
            if len(cluster_cache) == 0:
                continue

            candidate_clusters: set[int] = set()
            for excess_peak in excess_peaks:
                for clusterid, cache in cluster_cache.items():
                    if abs(int(cache["peak_tick"]) - int(excess_peak)) <= int(peak_window):
                        candidate_clusters.add(int(clusterid))
            for clusterid in movable_clusters:
                mode = str(assignment_info.get((int(clusterid), int(tpcid)), {}).get("mode", ""))
                if "nonimproving" in mode:
                    candidate_clusters.add(int(clusterid))

            if len(candidate_clusters) == 0 and len(excess_peaks) == 0:
                # If the residual finder sees no obvious excess peak, still try the most
                # weakly supported assigned cluster on this TPC.
                weakest = sorted(
                    movable_clusters,
                    key=lambda cid: (
                        float(cluster_energies.get(int(cid), 0.0)),
                        float(cluster_cache.get(int(cid), {}).get("peak_value", 0.0)),
                    ),
                )
                if len(weakest) > 0:
                    candidate_clusters.add(int(weakest[0]))

            if len(candidate_clusters) == 0:
                continue

            tpcs_with_candidates.add(int(tpcid))
            for clusterid in sorted(candidate_clusters):
                cluster_mask = labels_global == int(clusterid)
                current_t0s = hit_timestamps[cluster_mask]
                current_t0s = current_t0s[np.isfinite(current_t0s)]
                if current_t0s.size == 0:
                    continue
                old_t0 = int(np.rint(float(current_t0s[0])))

                tpcs = sorted(
                    int(tpc)
                    for tpc in cluster_to_tpcs.get(int(clusterid), [])
                    if int(tpc) < int(base_image.shape[0]) and (int(clusterid), int(tpc)) in image_maps
                )
                if len(tpcs) == 0:
                    continue

                tpcs_arr = np.asarray(tpcs, dtype=int)
                cluster_block = np.stack(
                    [np.asarray(image_maps[(int(clusterid), int(tpc))], dtype=np.float32) for tpc in tpcs_arr],
                    axis=0,
                )
                base_block = np.asarray(base_image[tpcs_arr], dtype=np.float32)
                actual_block = np.asarray(full_light_waveform[tpcs_arr], dtype=np.float32)
                error_block = np.asarray(full_light_std[tpcs_arr], dtype=np.float32)
                old_shifted = _shift_block(cluster_block, int(old_t0))
                base_without = np.clip(base_block - old_shifted, 0.0, None)
                current_score = compute_error_metric(base_block, actual_block, error_block)

                best_t0, best_score, _ = _scan_best_shift_multi(
                    cluster_block,
                    base_without,
                    actual_block,
                    error_block,
                    search_range=search_range,
                    adc_clip=adc_clip,
                    return_curve=False,
                )
                if abs(int(best_t0) - int(old_t0)) < int(min_move_ticks):
                    continue
                if int(best_t0) in visited_t0s.get(int(clusterid), set()):
                    continue

                improvement = float(current_score - float(best_score))
                if improvement <= float(min_improvement):
                    continue

                candidate_model = np.clip(base_without + _shift_block(cluster_block, int(best_t0)), None, adc_clip)
                if best_move is None or float(improvement) > float(best_move["improvement"]):
                    best_move = {
                        "clusterid": int(clusterid),
                        "old_t0": int(old_t0),
                        "new_t0": int(best_t0),
                        "tpcs": [int(tpc) for tpc in tpcs],
                        "candidate_model": np.asarray(candidate_model, dtype=np.float32),
                        "current_score": float(current_score),
                        "candidate_score": float(best_score),
                        "improvement": float(improvement),
                    }
                    best_move_meta = {
                        "source_tpc": int(tpcid),
                        "cluster_energy": float(cluster_energies.get(int(clusterid), 0.0)),
                        "old_peak_tick": int(cluster_cache.get(int(clusterid), {}).get("peak_tick", -1)),
                    }

        if best_move is None or best_move_meta is None:
            break

        clusterid = int(best_move["clusterid"])
        new_t0 = int(best_move["new_t0"])
        old_t0 = int(best_move["old_t0"])
        tpcs_arr = np.asarray(best_move["tpcs"], dtype=int)
        base_image[tpcs_arr] = np.asarray(best_move["candidate_model"], dtype=np.float32)
        hit_timestamps[labels_global == int(clusterid)] = float(new_t0)
        move_counts[int(clusterid)] = int(move_counts.get(int(clusterid), 0)) + 1
        moved_clusters.append(int(clusterid))
        visited_t0s.setdefault(int(clusterid), set()).add(int(new_t0))

        for tpc in tpcs_arr:
            append_candidate_t0(t0_candidates[int(tpc)], int(new_t0), max_t0=int(search_range))
            old_info = assignment_info.get((int(clusterid), int(tpc)), {})
            assignment_info[(int(clusterid), int(tpc))] = {
                **old_info,
                "stage": "step4_residual_rebalance",
                "mode": "step4_residual_rebalance_move",
                "t0": float(new_t0),
                "assigned": True,
                "energy": float(cluster_energies.get(int(clusterid), 0.0)),
                "error_after": float(best_move["candidate_score"]),
                "improvement": float(best_move["improvement"]),
                "moved_from_t0": float(old_t0),
                "rebalance_iteration": int(iteration),
                "rebalance_source_tpc": int(best_move_meta["source_tpc"]),
            }

        rebalance_log.append(
            {
                "clusterid": int(clusterid),
                "tpcs": [int(tpc) for tpc in tpcs_arr.tolist()],
                "old_t0": int(old_t0),
                "new_t0": int(new_t0),
                "improvement": float(best_move["improvement"]),
                "score_before": float(best_move["current_score"]),
                "score_after": float(best_move["candidate_score"]),
                "source_tpc": int(best_move_meta["source_tpc"]),
                "cluster_energy": float(best_move_meta["cluster_energy"]),
                "old_peak_tick": int(best_move_meta["old_peak_tick"]),
            }
        )

    stats = {
        "iterations_run": int(iterations_run),
        "moves_accepted": int(len(rebalance_log)),
        "moved_clusters": sorted(set(int(cid) for cid in moved_clusters)),
        "move_counts": {int(cid): int(count) for cid, count in sorted(move_counts.items())},
        "tpcs_with_candidates": sorted(int(tpc) for tpc in tpcs_with_candidates),
        "max_iterations": int(max_iterations),
        "max_moves_per_cluster": int(max_moves_per_cluster),
        "min_improvement": float(min_improvement),
        "search_range": int(search_range),
        "stopped_because": "no_improving_move" if len(rebalance_log) < int(max_iterations) else "max_iterations",
    }
    return base_image, hit_timestamps, t0_candidates, assignment_info, rebalance_log, stats
