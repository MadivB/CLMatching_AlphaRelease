from __future__ import annotations

from typing import Any

import numpy as np

try:
    from v3_2_global_matching import (
        assign_small_clusters_v32,
        append_candidate_t0,
        compute_error_metric,
        _shift_block,
        _stack_cluster_images,
        _stack_tpcs,
    )
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v3_2_global_matching import (
        assign_small_clusters_v32,
        append_candidate_t0,
        compute_error_metric,
        _shift_block,
        _stack_cluster_images,
        _stack_tpcs,
    )


def _smooth_curve(curve: np.ndarray, width: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    width = max(1, int(width))
    if width <= 1 or curve.size == 0:
        return curve
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(curve, kernel, mode="same")


def _top_unique_peaks(
    curve: np.ndarray,
    *,
    max_peaks: int,
    min_sep: int,
    min_height: float,
) -> list[int]:
    curve = np.asarray(curve, dtype=np.float32)
    if curve.size == 0 or max_peaks <= 0:
        return []

    order = np.argsort(curve)[::-1]
    peaks: list[int] = []
    for idx in order:
        val = float(curve[int(idx)])
        if val < float(min_height):
            break
        if any(abs(int(idx) - int(prev)) < int(min_sep) for prev in peaks):
            continue
        peaks.append(int(idx))
        if len(peaks) >= int(max_peaks):
            break
    return peaks


def _residual_peaks_for_tpc(
    actual_tpc: np.ndarray,
    predicted_tpc: np.ndarray,
    *,
    smooth_width: int,
    max_peaks_per_sign: int,
    min_peak_sep: int,
    min_peak_fraction: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    residual = np.asarray(actual_tpc, dtype=np.float32) - np.asarray(predicted_tpc, dtype=np.float32)
    deficit = np.clip(residual, 0.0, None).sum(axis=0)
    excess = np.clip(-residual, 0.0, None).sum(axis=0)

    deficit = _smooth_curve(deficit, smooth_width)
    excess = _smooth_curve(excess, smooth_width)

    deficit_floor = max(
        float(deficit.mean() + deficit.std()),
        float(min_peak_fraction) * float(deficit.max()) if deficit.size else 0.0,
    )
    excess_floor = max(
        float(excess.mean() + excess.std()),
        float(min_peak_fraction) * float(excess.max()) if excess.size else 0.0,
    )

    deficit_peaks = _top_unique_peaks(
        deficit,
        max_peaks=max_peaks_per_sign,
        min_sep=min_peak_sep,
        min_height=deficit_floor,
    )
    excess_peaks = _top_unique_peaks(
        excess,
        max_peaks=max_peaks_per_sign,
        min_sep=min_peak_sep,
        min_height=excess_floor,
    )
    return deficit, excess, deficit_peaks, excess_peaks


def _current_cluster_t0(clusterid: int, labels_global: np.ndarray, hit_timestamps: np.ndarray) -> int | None:
    cluster_mask = labels_global == int(clusterid)
    if not np.any(cluster_mask):
        return None
    finite_t0s = hit_timestamps[cluster_mask]
    finite_t0s = finite_t0s[np.isfinite(finite_t0s)]
    if finite_t0s.size == 0:
        return None
    return int(np.rint(float(finite_t0s[0])))


def _cluster_shifted_tpc_image(
    image_maps: dict[tuple[int, int], np.ndarray],
    clusterid: int,
    tpcid: int,
    t0: int,
) -> np.ndarray:
    raw = np.asarray(image_maps[(int(clusterid), int(tpcid))], dtype=np.float32)[None, :, :]
    return _shift_block(raw, int(t0))[0]


def _collect_tpc_cluster_cache(
    tpcid: int,
    candidate_clusters: list[int],
    *,
    image_maps: dict[tuple[int, int], np.ndarray],
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    for clusterid in candidate_clusters:
        current_t0 = _current_cluster_t0(int(clusterid), labels_global, hit_timestamps)
        if current_t0 is None:
            continue
        shifted_image = _cluster_shifted_tpc_image(image_maps, int(clusterid), int(tpcid), int(current_t0))
        profile = shifted_image.sum(axis=0)
        if profile.size == 0 or float(profile.max()) <= 0.0:
            continue
        peak_tick = int(np.argmax(profile))
        cache[int(clusterid)] = {
            "t0": int(current_t0),
            "profile": profile,
            "peak_tick": peak_tick,
            "peak_value": float(profile[peak_tick]),
            "image": shifted_image,
        }
    return cache


def _evaluate_cluster_move(
    clusterid: int,
    new_t0: int,
    *,
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    adc_clip: float,
) -> dict[str, Any] | None:
    old_t0 = _current_cluster_t0(int(clusterid), labels_global, hit_timestamps)
    if old_t0 is None:
        return None
    if int(new_t0) == int(old_t0):
        return None

    tpcs = sorted(
        int(tpc)
        for tpc in cluster_to_tpcs.get(int(clusterid), [])
        if int(tpc) < int(base_image.shape[0]) and (int(clusterid), int(tpc)) in image_maps
    )
    if len(tpcs) == 0:
        return None

    tpcs_arr = np.asarray(tpcs, dtype=int)
    cluster_block = _stack_cluster_images(image_maps, int(clusterid), tpcs_arr)
    base_block = _stack_tpcs(base_image, tpcs_arr)
    actual_block = _stack_tpcs(full_light_waveform, tpcs_arr)
    error_block = _stack_tpcs(full_light_std, tpcs_arr)

    old_shifted = _shift_block(cluster_block, int(old_t0))
    base_without = np.clip(base_block - old_shifted, 0.0, None)

    current_score = compute_error_metric(base_block, actual_block, error_block)
    new_shifted = _shift_block(cluster_block, int(new_t0))
    candidate_model = np.clip(base_without + new_shifted, None, adc_clip)
    candidate_score = compute_error_metric(candidate_model, actual_block, error_block)

    return {
        "clusterid": int(clusterid),
        "old_t0": int(old_t0),
        "new_t0": int(new_t0),
        "tpcs": tpcs,
        "cluster_block": cluster_block,
        "candidate_model": candidate_model,
        "current_score": float(current_score),
        "candidate_score": float(candidate_score),
        "improvement": float(current_score - candidate_score),
    }


def rebalance_assigned_clusters_v33(
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
    max_iterations: int = 8,
    max_moves_per_cluster: int = 2,
    max_peaks_per_sign: int = 2,
    max_clusters_per_excess: int = 3,
    min_peak_sep: int = 12,
    peak_window: int = 20,
    min_peak_fraction: float = 0.20,
    smooth_width: int = 11,
    min_move_ticks: int = 4,
    min_improvement: float = 1e-4,
    max_movable_energy_mev: float | None = None,
    adc_clip: float = 60780.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
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
        current_t0 = _current_cluster_t0(int(clusterid), labels_global, hit_timestamps)
        if current_t0 is None:
            continue
        visited_t0s[int(clusterid)] = {int(current_t0)}

    iterations_run = 0
    tpcs_with_proposals: set[int] = set()
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
                if (int(clusterid), int(tpcid)) not in assignment_info:
                    continue
                info = assignment_info[(int(clusterid), int(tpcid))]
                if not info.get("assigned", False):
                    continue
                if max_movable_energy_mev is not None and float(cluster_energies.get(int(clusterid), 0.0)) > float(max_movable_energy_mev):
                    continue
                if move_counts.get(int(clusterid), 0) >= int(max_moves_per_cluster):
                    continue
                movable_clusters.append(int(clusterid))

            if len(movable_clusters) == 0:
                continue

            actual_tpc = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
            predicted_tpc = np.asarray(base_image[int(tpcid)], dtype=np.float32)
            deficit, excess, deficit_peaks, excess_peaks = _residual_peaks_for_tpc(
                actual_tpc,
                predicted_tpc,
                smooth_width=smooth_width,
                max_peaks_per_sign=max_peaks_per_sign,
                min_peak_sep=min_peak_sep,
                min_peak_fraction=min_peak_fraction,
            )
            if len(deficit_peaks) == 0 or len(excess_peaks) == 0:
                continue

            cluster_cache = _collect_tpc_cluster_cache(
                int(tpcid),
                movable_clusters,
                image_maps=image_maps,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
            )
            if len(cluster_cache) == 0:
                continue

            tpcs_with_proposals.add(int(tpcid))
            tested_pairs: set[tuple[int, int]] = set()

            for excess_peak in excess_peaks:
                scored_clusters = []
                for clusterid, cache in cluster_cache.items():
                    peak_tick = int(cache["peak_tick"])
                    if abs(int(peak_tick) - int(excess_peak)) > int(peak_window):
                        continue
                    local_support = float(cache["profile"][int(excess_peak)])
                    if local_support <= 0.0:
                        continue
                    scored_clusters.append((local_support, int(clusterid)))

                scored_clusters.sort(reverse=True)
                for _, clusterid in scored_clusters[: int(max_clusters_per_excess)]:
                    cache = cluster_cache[int(clusterid)]
                    old_t0 = int(cache["t0"])
                    peak_tick = int(cache["peak_tick"])
                    for deficit_peak in deficit_peaks:
                        new_t0 = int(old_t0 + int(deficit_peak) - int(peak_tick))
                        if abs(int(new_t0) - int(old_t0)) < int(min_move_ticks):
                            continue
                        if new_t0 < 0 or new_t0 >= int(full_light_waveform.shape[-1]):
                            continue
                        if int(new_t0) in visited_t0s.get(int(clusterid), set()):
                            continue
                        pair = (int(clusterid), int(new_t0))
                        if pair in tested_pairs:
                            continue
                        tested_pairs.add(pair)

                        move_eval = _evaluate_cluster_move(
                            int(clusterid),
                            int(new_t0),
                            cluster_to_tpcs=cluster_to_tpcs,
                            image_maps=image_maps,
                            base_image=base_image,
                            full_light_waveform=full_light_waveform,
                            full_light_std=full_light_std,
                            labels_global=labels_global,
                            hit_timestamps=hit_timestamps,
                            adc_clip=adc_clip,
                        )
                        if move_eval is None:
                            continue
                        improvement = float(move_eval["improvement"])
                        if improvement <= float(min_improvement):
                            continue
                        if best_move is None or improvement > float(best_move["improvement"]):
                            best_move = move_eval
                            best_move_meta = {
                                "source_tpc": int(tpcid),
                                "excess_peak": int(excess_peak),
                                "deficit_peak": int(deficit_peak),
                                "excess_height": float(excess[int(excess_peak)]),
                                "deficit_height": float(deficit[int(deficit_peak)]),
                                "cluster_peak_tick": int(peak_tick),
                                "cluster_energy": float(cluster_energies.get(int(clusterid), 0.0)),
                            }

        if best_move is None or best_move_meta is None:
            break

        clusterid = int(best_move["clusterid"])
        new_t0 = int(best_move["new_t0"])
        old_t0 = int(best_move["old_t0"])
        tpcs = np.asarray(best_move["tpcs"], dtype=int)
        base_image[tpcs] = np.asarray(best_move["candidate_model"], dtype=np.float32)
        hit_timestamps[labels_global == int(clusterid)] = float(new_t0)
        moved_clusters.append(int(clusterid))
        move_counts[int(clusterid)] = int(move_counts.get(int(clusterid), 0)) + 1
        visited_t0s.setdefault(int(clusterid), set()).add(int(new_t0))

        for tpc in tpcs:
            append_candidate_t0(t0_candidates[int(tpc)], int(new_t0), max_t0=int(full_light_waveform.shape[-1] - 1))
            old_info = assignment_info.get((int(clusterid), int(tpc)), {})
            assignment_info[(int(clusterid), int(tpc))] = {
                **old_info,
                "stage": "residual_rebalance",
                "mode": "residual_rebalance_move",
                "t0": float(new_t0),
                "assigned": True,
                "energy": float(cluster_energies.get(int(clusterid), 0.0)),
                "error_after": float(best_move["candidate_score"]),
                "improvement": float(best_move["improvement"]),
                "moved_from_t0": float(old_t0),
                "rebalance_iteration": int(iteration),
                "rebalance_source_tpc": int(best_move_meta["source_tpc"]),
                "rebalance_excess_peak": int(best_move_meta["excess_peak"]),
                "rebalance_deficit_peak": int(best_move_meta["deficit_peak"]),
            }

        rebalance_log.append(
            {
                "clusterid": int(clusterid),
                "tpcs": [int(tpc) for tpc in tpcs.tolist()],
                "old_t0": int(old_t0),
                "new_t0": int(new_t0),
                "improvement": float(best_move["improvement"]),
                "score_before": float(best_move["current_score"]),
                "score_after": float(best_move["candidate_score"]),
                "source_tpc": int(best_move_meta["source_tpc"]),
                "excess_peak": int(best_move_meta["excess_peak"]),
                "deficit_peak": int(best_move_meta["deficit_peak"]),
                "excess_height": float(best_move_meta["excess_height"]),
                "deficit_height": float(best_move_meta["deficit_height"]),
                "cluster_peak_tick": int(best_move_meta["cluster_peak_tick"]),
                "cluster_energy": float(best_move_meta["cluster_energy"]),
            }
        )

    stats = {
        "iterations_run": int(iterations_run),
        "moves_accepted": int(len(rebalance_log)),
        "moved_clusters": sorted(set(int(cid) for cid in moved_clusters)),
        "move_counts": {int(cid): int(count) for cid, count in sorted(move_counts.items())},
        "tpcs_with_proposals": sorted(int(tpc) for tpc in tpcs_with_proposals),
        "max_iterations": int(max_iterations),
        "max_moves_per_cluster": int(max_moves_per_cluster),
        "max_peaks_per_sign": int(max_peaks_per_sign),
        "max_clusters_per_excess": int(max_clusters_per_excess),
        "min_improvement": float(min_improvement),
        "stopped_because": "no_improving_move" if len(rebalance_log) < int(max_iterations) else "max_iterations",
    }
    return base_image, hit_timestamps, t0_candidates, assignment_info, rebalance_log, stats
