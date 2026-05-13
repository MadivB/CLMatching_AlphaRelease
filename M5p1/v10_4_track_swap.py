from __future__ import annotations

from typing import Any

import numpy as np

try:
    from v3_2_global_matching import _shift_block, append_candidate_t0, compute_error_metric
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v3_2_global_matching import _shift_block, append_candidate_t0, compute_error_metric


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

    return int(np.median(np.asarray(values, dtype=np.int32)))


def _track_geometry_by_tpc(
    *,
    clusterid: int,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any] | None:
    mask = np.asarray(labels_global, dtype=np.int32) == int(clusterid)
    if not np.any(mask):
        return None

    x_sel = np.asarray(x[mask], dtype=np.float64)
    y_sel = np.asarray(y[mask], dtype=np.float64)
    z_sel = np.asarray(z[mask], dtype=np.float64)
    e_sel = np.asarray(energies[mask], dtype=np.float64)
    tpc_sel = np.asarray(hit_tpc_ids[mask], dtype=np.int32)
    points = np.column_stack([x_sel, y_sel, z_sel])
    if points.shape[0] < 2:
        return None

    center = np.mean(points, axis=0)
    centered = points - center[None, :]
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = np.asarray(vh[0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    direction = direction / norm

    proj = centered @ direction
    endpoint_a = center + direction * float(np.min(proj))
    endpoint_b = center + direction * float(np.max(proj))
    length_cm = float(np.max(proj) - np.min(proj))

    centroids_by_tpc: dict[int, np.ndarray] = {}
    yz_centroids_by_tpc: dict[int, np.ndarray] = {}
    for tpcid in np.unique(tpc_sel):
        mask_tpc = tpc_sel == int(tpcid)
        pts_tpc = points[mask_tpc]
        centroid_tpc = np.mean(pts_tpc, axis=0)
        centroids_by_tpc[int(tpcid)] = np.asarray(centroid_tpc, dtype=np.float64)
        yz_centroids_by_tpc[int(tpcid)] = np.asarray(centroid_tpc[1:], dtype=np.float64)

    return {
        "clusterid": int(clusterid),
        "energy": float(np.sum(e_sel)),
        "n_hits": int(points.shape[0]),
        "tpcs": _cluster_image_tpcs(
            clusterid=int(clusterid),
            cluster_to_tpcs=cluster_to_tpcs,
            image_maps=image_maps,
            max_tpcs=int(np.max(hit_tpc_ids)) + 1 if hit_tpc_ids.size > 0 else 0,
        ),
        "center": np.asarray(center, dtype=np.float64),
        "direction": np.asarray(direction, dtype=np.float64),
        "endpoint_a": np.asarray(endpoint_a, dtype=np.float64),
        "endpoint_b": np.asarray(endpoint_b, dtype=np.float64),
        "length_cm": float(length_cm),
        "centroids_by_tpc": {int(k): np.asarray(v, dtype=np.float64) for k, v in centroids_by_tpc.items()},
        "yz_centroids_by_tpc": {int(k): np.asarray(v, dtype=np.float64) for k, v in yz_centroids_by_tpc.items()},
    }


def _min_endpoint_yz_distance_cm(geom_a: dict[str, Any], geom_b: dict[str, Any]) -> float:
    endpoints_a = [
        np.asarray(geom_a["endpoint_a"], dtype=np.float64)[1:],
        np.asarray(geom_a["endpoint_b"], dtype=np.float64)[1:],
    ]
    endpoints_b = [
        np.asarray(geom_b["endpoint_a"], dtype=np.float64)[1:],
        np.asarray(geom_b["endpoint_b"], dtype=np.float64)[1:],
    ]
    return float(min(np.linalg.norm(a - b) for a in endpoints_a for b in endpoints_b))


def _pair_support_channels(
    *,
    cluster_a: int,
    cluster_b: int,
    tpcid: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
) -> np.ndarray:
    n_channels = int(np.asarray(image_maps[(int(cluster_a), int(tpcid))] if (int(cluster_a), int(tpcid)) in image_maps else image_maps[(int(cluster_b), int(tpcid))]).shape[0])
    if channel_support_cache is None:
        return np.arange(n_channels, dtype=np.int32)

    selected: set[int] = set()
    for clusterid in (int(cluster_a), int(cluster_b)):
        entry = dict(channel_support_cache.get((int(clusterid), int(tpcid)), {}))
        indices = np.asarray(entry.get("channel_indices", []), dtype=np.int32)
        selected.update(int(v) for v in indices.tolist())

    if len(selected) == 0:
        return np.arange(n_channels, dtype=np.int32)
    return np.asarray(sorted(selected), dtype=np.int32)


def _score_track_swap_pair(
    *,
    cluster_a: int,
    cluster_b: int,
    t0_a: int,
    t0_b: int,
    union_tpcs: list[int],
    image_maps: dict[tuple[int, int], np.ndarray],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    adc_clip: float,
) -> tuple[float, float, dict[int, int]]:
    current_score = 0.0
    swapped_score = 0.0
    fit_channels_by_tpc: dict[int, int] = {}

    for tpcid in union_tpcs:
        channel_indices = _pair_support_channels(
            cluster_a=int(cluster_a),
            cluster_b=int(cluster_b),
            tpcid=int(tpcid),
            image_maps=image_maps,
            channel_support_cache=channel_support_cache,
        )
        fit_channels_by_tpc[int(tpcid)] = int(channel_indices.size)

        current_block = np.asarray(base_image[int(tpcid), channel_indices], dtype=np.float32)
        actual_block = np.asarray(full_light_waveform[int(tpcid), channel_indices], dtype=np.float32)
        error_block = np.asarray(full_light_std[int(tpcid), channel_indices], dtype=np.float32)

        current_score += float(compute_error_metric(current_block, actual_block, error_block))

        rebuilt_block = np.asarray(current_block, dtype=np.float32).copy()
        if (int(cluster_a), int(tpcid)) in image_maps:
            wave_a = np.asarray(image_maps[(int(cluster_a), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block - _shift_block(wave_a[None, channel_indices, :], int(t0_a))[0],
                0.0,
                None,
            )
        if (int(cluster_b), int(tpcid)) in image_maps:
            wave_b = np.asarray(image_maps[(int(cluster_b), int(tpcid))], dtype=np.float32)
            rebuilt_block = np.clip(
                rebuilt_block - _shift_block(wave_b[None, channel_indices, :], int(t0_b))[0],
                0.0,
                None,
            )

        swapped_block = np.asarray(rebuilt_block, dtype=np.float32)
        if (int(cluster_a), int(tpcid)) in image_maps:
            wave_a = np.asarray(image_maps[(int(cluster_a), int(tpcid))], dtype=np.float32)
            swapped_block = np.clip(
                swapped_block + _shift_block(wave_a[None, channel_indices, :], int(t0_b))[0],
                None,
                float(adc_clip),
            )
        if (int(cluster_b), int(tpcid)) in image_maps:
            wave_b = np.asarray(image_maps[(int(cluster_b), int(tpcid))], dtype=np.float32)
            swapped_block = np.clip(
                swapped_block + _shift_block(wave_b[None, channel_indices, :], int(t0_a))[0],
                None,
                float(adc_clip),
            )

        swapped_score += float(compute_error_metric(swapped_block, actual_block, error_block))

    return float(current_score), float(swapped_score), fit_channels_by_tpc


def _apply_track_swap_pair(
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
    absorbed_hit_parent: np.ndarray | None,
    trigger_pair_rank: int,
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

    mask_a = np.asarray(labels_global, dtype=np.int32) == int(cluster_a)
    mask_b = np.asarray(labels_global, dtype=np.int32) == int(cluster_b)
    hit_timestamps[mask_a] = float(t0_b)
    hit_timestamps[mask_b] = float(t0_a)

    if absorbed_hit_parent is not None:
        absorbed = np.asarray(absorbed_hit_parent, dtype=np.int32)
        hit_timestamps[absorbed == int(cluster_a)] = float(t0_b)
        hit_timestamps[absorbed == int(cluster_b)] = float(t0_a)

    if protected_track_shower_timestamps is not None:
        protected_track_shower_timestamps[mask_a] = float(t0_b)
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
            info.update(
                {
                    "t0": float(new_t0),
                    "swap_partner": int(partner),
                    "swap_old_t0": float(old_t0),
                    "swap_new_t0": float(new_t0),
                    "swap_improvement": float(improvement),
                    "swap_stage": "track_swap_rescue",
                    "track_swap_rank": int(trigger_pair_rank),
                    "mode": "track_swap_rescue",
                }
            )
            assignment_info[(int(clusterid), int(tpcid))] = info


def run_track_swap_rescue_v10_4(
    *,
    track_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    label_info: dict[int, dict[str, Any]],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    protected_track_shower_timestamps: np.ndarray | None = None,
    absorbed_hit_parent: np.ndarray | None = None,
    max_relative_energy_diff: float = 0.30,
    max_angle_deg: float = 14.0,
    min_shared_tpcs: int = 1,
    min_tpc_overlap_fraction: float = 0.50,
    max_shared_yz_dist_cm: float = 20.0,
    max_endpoint_yz_dist_cm: float = 24.0,
    min_t0_separation_ticks: int = 8,
    max_passes: int = 6,
    improvement_eps: float = 0.0,
    adc_clip: float = 60780.0,
    search_range: int = 800,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cos_angle_threshold = float(np.cos(np.deg2rad(float(max_angle_deg))))
    track_ids = [
        int(clusterid)
        for clusterid in track_labels
        if str(label_info.get(int(clusterid), {}).get("type", "track")).lower() == "track"
    ]
    geometry: dict[int, dict[str, Any]] = {}
    for clusterid in track_ids:
        geom = _track_geometry_by_tpc(
            clusterid=int(clusterid),
            labels_global=labels_global,
            hit_tpc_ids=hit_tpc_ids,
            x=x,
            y=y,
            z=z,
            energies=energies,
            cluster_to_tpcs=cluster_to_tpcs,
            image_maps=image_maps,
        )
        if geom is not None:
            geometry[int(clusterid)] = geom

    logs: list[dict[str, Any]] = []
    considered_pairs: list[dict[str, Any]] = []
    n_pairs_considered = 0
    n_swapped_clusters = 0

    for pass_idx in range(int(max_passes)):
        best_swap: dict[str, Any] | None = None

        for idx_a in range(len(track_ids)):
            cluster_a = int(track_ids[idx_a])
            geom_a = geometry.get(int(cluster_a))
            if geom_a is None:
                continue
            t0_a = _cluster_assigned_t0(
                clusterid=int(cluster_a),
                cluster_to_tpcs=cluster_to_tpcs,
                image_maps=image_maps,
                assignment_info=assignment_info,
                max_tpcs=int(base_image.shape[0]),
            )
            if t0_a is None:
                continue

            for idx_b in range(idx_a + 1, len(track_ids)):
                cluster_b = int(track_ids[idx_b])
                geom_b = geometry.get(int(cluster_b))
                if geom_b is None:
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
                if abs(int(t0_a) - int(t0_b)) < int(min_t0_separation_ticks):
                    continue

                energy_a = float(geom_a["energy"])
                energy_b = float(geom_b["energy"])
                rel_energy_diff = abs(energy_a - energy_b) / max(max(energy_a, energy_b), 1e-6)
                if rel_energy_diff > float(max_relative_energy_diff):
                    continue

                tpcs_a = set(int(v) for v in geom_a["tpcs"])
                tpcs_b = set(int(v) for v in geom_b["tpcs"])
                shared_tpcs = sorted(tpcs_a & tpcs_b)
                if len(shared_tpcs) < int(min_shared_tpcs):
                    continue
                overlap_fraction = len(shared_tpcs) / max(min(len(tpcs_a), len(tpcs_b)), 1)
                if overlap_fraction < float(min_tpc_overlap_fraction):
                    continue

                direction_a = np.asarray(geom_a["direction"], dtype=np.float64)
                direction_b = np.asarray(geom_b["direction"], dtype=np.float64)
                direction_cosine = float(abs(np.dot(direction_a, direction_b)))
                if direction_cosine < float(cos_angle_threshold):
                    continue

                shared_yz_distances = []
                for tpcid in shared_tpcs:
                    yz_a = np.asarray(geom_a["yz_centroids_by_tpc"].get(int(tpcid)), dtype=np.float64)
                    yz_b = np.asarray(geom_b["yz_centroids_by_tpc"].get(int(tpcid)), dtype=np.float64)
                    if yz_a.size == 2 and yz_b.size == 2:
                        shared_yz_distances.append(float(np.linalg.norm(yz_a - yz_b)))
                if len(shared_yz_distances) == 0:
                    continue
                shared_yz_dist_cm = float(np.mean(shared_yz_distances))
                if shared_yz_dist_cm > float(max_shared_yz_dist_cm):
                    continue

                endpoint_yz_dist_cm = _min_endpoint_yz_distance_cm(geom_a, geom_b)
                if endpoint_yz_dist_cm > float(max_endpoint_yz_dist_cm):
                    continue

                union_tpcs = sorted(tpcs_a | tpcs_b)
                current_score, swapped_score, fit_channels_by_tpc = _score_track_swap_pair(
                    cluster_a=int(cluster_a),
                    cluster_b=int(cluster_b),
                    t0_a=int(t0_a),
                    t0_b=int(t0_b),
                    union_tpcs=union_tpcs,
                    image_maps=image_maps,
                    channel_support_cache=channel_support_cache,
                    base_image=base_image,
                    full_light_waveform=full_light_waveform,
                    full_light_std=full_light_std,
                    adc_clip=float(adc_clip),
                )
                improvement = float(current_score - swapped_score)
                n_pairs_considered += 1
                considered_pairs.append(
                    {
                        "cluster_a": int(cluster_a),
                        "cluster_b": int(cluster_b),
                        "t0_a": int(t0_a),
                        "t0_b": int(t0_b),
                        "shared_tpcs": [int(v) for v in shared_tpcs],
                        "overlap_fraction": float(overlap_fraction),
                        "rel_energy_diff": float(rel_energy_diff),
                        "direction_cosine": float(direction_cosine),
                        "shared_yz_dist_cm": float(shared_yz_dist_cm),
                        "endpoint_yz_dist_cm": float(endpoint_yz_dist_cm),
                        "current_score": float(current_score),
                        "swapped_score": float(swapped_score),
                        "improvement": float(improvement),
                    }
                )
                if improvement <= float(improvement_eps):
                    continue

                if best_swap is None or improvement > float(best_swap["improvement"]):
                    best_swap = {
                        "pass_index": int(pass_idx),
                        "cluster_a": int(cluster_a),
                        "cluster_b": int(cluster_b),
                        "energy_a": float(energy_a),
                        "energy_b": float(energy_b),
                        "t0_a_old": int(t0_a),
                        "t0_b_old": int(t0_b),
                        "t0_a_new": int(t0_b),
                        "t0_b_new": int(t0_a),
                        "shared_tpcs": [int(v) for v in shared_tpcs],
                        "union_tpcs": [int(v) for v in union_tpcs],
                        "overlap_fraction": float(overlap_fraction),
                        "relative_energy_diff": float(rel_energy_diff),
                        "direction_cosine": float(direction_cosine),
                        "shared_yz_dist_cm": float(shared_yz_dist_cm),
                        "endpoint_yz_dist_cm": float(endpoint_yz_dist_cm),
                        "current_score": float(current_score),
                        "swapped_score": float(swapped_score),
                        "improvement": float(improvement),
                        "fit_channels_by_tpc": {int(k): int(v) for k, v in fit_channels_by_tpc.items()},
                        "n_fit_channels_total": int(sum(int(v) for v in fit_channels_by_tpc.values())),
                    }

        if best_swap is None:
            break

        _apply_track_swap_pair(
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
            absorbed_hit_parent=absorbed_hit_parent,
            trigger_pair_rank=int(pass_idx),
            search_range=int(search_range),
        )
        logs.append(dict(best_swap))
        n_swapped_clusters += 2

    stats = {
        "pairs_considered": int(n_pairs_considered),
        "accepted_swaps": int(len(logs)),
        "swapped_clusters": sorted(
            {
                int(item["cluster_a"])
                for item in logs
            }
            | {
                int(item["cluster_b"])
                for item in logs
            }
        ),
        "max_relative_energy_diff": float(max_relative_energy_diff),
        "max_angle_deg": float(max_angle_deg),
        "min_shared_tpcs": int(min_shared_tpcs),
        "min_tpc_overlap_fraction": float(min_tpc_overlap_fraction),
        "max_shared_yz_dist_cm": float(max_shared_yz_dist_cm),
        "max_endpoint_yz_dist_cm": float(max_endpoint_yz_dist_cm),
        "min_t0_separation_ticks": int(min_t0_separation_ticks),
        "max_passes": int(max_passes),
        "improvement_eps": float(improvement_eps),
        "considered_pairs": [dict(item) for item in considered_pairs],
        "swap_log": [dict(item) for item in logs],
    }
    return base_image, hit_timestamps, t0_candidates, assignment_info, logs, stats


__all__ = [
    "run_track_swap_rescue_v10_4",
]
