from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

try:
    from v10_4_track_swap import (
        _apply_track_swap_pair,
        _cluster_assigned_t0,
        _min_endpoint_yz_distance_cm,
        _score_track_swap_pair,
    )
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v10_4_track_swap import (
        _apply_track_swap_pair,
        _cluster_assigned_t0,
        _min_endpoint_yz_distance_cm,
        _score_track_swap_pair,
    )


def _track_geometry_from_hits(
    *,
    clusterid: int,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
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

    yz_centroids_by_tpc: dict[int, np.ndarray] = {}
    for tpcid in np.unique(tpc_sel):
        mask_tpc = tpc_sel == int(tpcid)
        centroid_tpc = np.mean(points[mask_tpc], axis=0)
        yz_centroids_by_tpc[int(tpcid)] = np.asarray(centroid_tpc[1:], dtype=np.float64)

    return {
        "clusterid": int(clusterid),
        "energy": float(np.sum(e_sel)),
        "n_hits": int(points.shape[0]),
        "tpcs": sorted(int(v) for v in np.unique(tpc_sel)),
        "center": np.asarray(center, dtype=np.float64),
        "direction": np.asarray(direction, dtype=np.float64),
        "endpoint_a": np.asarray(endpoint_a, dtype=np.float64),
        "endpoint_b": np.asarray(endpoint_b, dtype=np.float64),
        "yz_centroids_by_tpc": {int(k): np.asarray(v, dtype=np.float64) for k, v in yz_centroids_by_tpc.items()},
    }


def _candidate_priority(row: dict[str, Any]) -> float:
    return float(
        5.0 * float(len(row["shared_tpcs"]))
        - 1.5 * float(row["tpc_sym_diff_count"])
        + 3.0 * float(row["energy_ratio"])
        + 4.0 * float(row["direction_cosine"])
        - 0.08 * float(row["shared_yz_dist_cm"])
        - 0.04 * float(row["endpoint_yz_dist_cm"])
    )


def build_track_swap_candidates_from_hits_v10_7_focused(
    *,
    track_labels: list[int],
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    min_energy_ratio: float = 0.50,
    min_shared_tpcs: int = 1,
    max_tpc_sym_diff: int = 3,
    max_angle_deg: float = 20.0,
    max_shared_yz_dist_cm: float = 35.0,
    max_endpoint_yz_dist_cm: float = 45.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cos_angle_threshold = float(np.cos(np.deg2rad(float(max_angle_deg))))

    geometry: dict[int, dict[str, Any]] = {}
    tpc_to_tracks: dict[int, list[int]] = {}
    skipped_non_tracks: list[int] = []
    skipped_missing_geometry: list[int] = []

    for clusterid in sorted(int(v) for v in track_labels):
        if str(label_info.get(int(clusterid), {}).get("type", "track")).lower() != "track":
            skipped_non_tracks.append(int(clusterid))
            continue

        geom = _track_geometry_from_hits(
            clusterid=int(clusterid),
            labels_global=labels_global,
            hit_tpc_ids=hit_tpc_ids,
            x=x,
            y=y,
            z=z,
            energies=energies,
        )
        if geom is None:
            skipped_missing_geometry.append(int(clusterid))
            continue

        geometry[int(clusterid)] = geom
        for tpcid in geom["tpcs"]:
            tpc_to_tracks.setdefault(int(tpcid), []).append(int(clusterid))

    pair_keys: set[tuple[int, int]] = set()
    for track_ids in tpc_to_tracks.values():
        unique_ids = sorted(set(int(v) for v in track_ids))
        if len(unique_ids) < 2:
            continue
        for cluster_a, cluster_b in combinations(unique_ids, 2):
            pair_keys.add((int(cluster_a), int(cluster_b)))

    candidates: list[dict[str, Any]] = []
    reject_counts = {
        "energy_ratio": 0,
        "shared_tpcs": 0,
        "tpc_sym_diff": 0,
        "direction": 0,
        "shared_yz": 0,
        "endpoint_yz": 0,
    }

    for cluster_a, cluster_b in sorted(pair_keys):
        geom_a = geometry.get(int(cluster_a))
        geom_b = geometry.get(int(cluster_b))
        if geom_a is None or geom_b is None:
            continue

        energy_a = float(geom_a["energy"])
        energy_b = float(geom_b["energy"])
        energy_ratio = min(energy_a, energy_b) / max(max(energy_a, energy_b), 1e-6)
        if energy_ratio < float(min_energy_ratio):
            reject_counts["energy_ratio"] += 1
            continue

        tpcs_a = set(int(v) for v in geom_a["tpcs"])
        tpcs_b = set(int(v) for v in geom_b["tpcs"])
        shared_tpcs = sorted(tpcs_a & tpcs_b)
        if len(shared_tpcs) < int(min_shared_tpcs):
            reject_counts["shared_tpcs"] += 1
            continue

        sym_diff_tpcs = sorted(tpcs_a ^ tpcs_b)
        if len(sym_diff_tpcs) > int(max_tpc_sym_diff):
            reject_counts["tpc_sym_diff"] += 1
            continue

        direction_a = np.asarray(geom_a["direction"], dtype=np.float64)
        direction_b = np.asarray(geom_b["direction"], dtype=np.float64)
        direction_cosine = float(abs(np.dot(direction_a, direction_b)))
        if direction_cosine < float(cos_angle_threshold):
            reject_counts["direction"] += 1
            continue

        shared_yz_distances = []
        for tpcid in shared_tpcs:
            yz_a = np.asarray(geom_a["yz_centroids_by_tpc"].get(int(tpcid)), dtype=np.float64)
            yz_b = np.asarray(geom_b["yz_centroids_by_tpc"].get(int(tpcid)), dtype=np.float64)
            if yz_a.size == 2 and yz_b.size == 2:
                shared_yz_distances.append(float(np.linalg.norm(yz_a - yz_b)))
        if len(shared_yz_distances) == 0:
            reject_counts["shared_yz"] += 1
            continue

        shared_yz_dist_cm = float(np.mean(shared_yz_distances))
        if shared_yz_dist_cm > float(max_shared_yz_dist_cm):
            reject_counts["shared_yz"] += 1
            continue

        endpoint_yz_dist_cm = _min_endpoint_yz_distance_cm(geom_a, geom_b)
        if endpoint_yz_dist_cm > float(max_endpoint_yz_dist_cm):
            reject_counts["endpoint_yz"] += 1
            continue

        row = {
            "cluster_a": int(cluster_a),
            "cluster_b": int(cluster_b),
            "energy_a": float(energy_a),
            "energy_b": float(energy_b),
            "energy_ratio": float(energy_ratio),
            "relative_energy_diff": float(1.0 - energy_ratio),
            "tpcs_a": sorted(int(v) for v in tpcs_a),
            "tpcs_b": sorted(int(v) for v in tpcs_b),
            "shared_tpcs": [int(v) for v in shared_tpcs],
            "union_tpcs": sorted(int(v) for v in (tpcs_a | tpcs_b)),
            "sym_diff_tpcs": [int(v) for v in sym_diff_tpcs],
            "tpc_sym_diff_count": int(len(sym_diff_tpcs)),
            "tpc_count_gap": int(abs(len(tpcs_a) - len(tpcs_b))),
            "direction_cosine": float(direction_cosine),
            "shared_yz_dist_cm": float(shared_yz_dist_cm),
            "endpoint_yz_dist_cm": float(endpoint_yz_dist_cm),
            "n_shared_tpcs": int(len(shared_tpcs)),
        }
        row["candidate_priority"] = float(_candidate_priority(row))
        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            -float(row["candidate_priority"]),
            -int(row["n_shared_tpcs"]),
            int(row["tpc_sym_diff_count"]),
            -float(row["energy_ratio"]),
            -float(row["direction_cosine"]),
            float(row["shared_yz_dist_cm"]),
            float(row["endpoint_yz_dist_cm"]),
            int(row["cluster_a"]),
            int(row["cluster_b"]),
        )
    )

    stats = {
        "n_tracks_with_geometry": int(len(geometry)),
        "n_tpc_buckets": int(len(tpc_to_tracks)),
        "n_bucket_pairs": int(len(pair_keys)),
        "n_candidates": int(len(candidates)),
        "min_energy_ratio": float(min_energy_ratio),
        "min_shared_tpcs": int(min_shared_tpcs),
        "max_tpc_sym_diff": int(max_tpc_sym_diff),
        "max_angle_deg": float(max_angle_deg),
        "max_shared_yz_dist_cm": float(max_shared_yz_dist_cm),
        "max_endpoint_yz_dist_cm": float(max_endpoint_yz_dist_cm),
        "reject_counts": {str(k): int(v) for k, v in reject_counts.items()},
        "skipped_non_tracks": [int(v) for v in skipped_non_tracks],
        "skipped_missing_geometry": [int(v) for v in skipped_missing_geometry],
    }
    return candidates, stats


def run_track_overlap_swap_rescue_v10_7_focused(
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
    min_energy_ratio: float = 0.50,
    min_shared_tpcs: int = 1,
    max_tpc_sym_diff: int = 3,
    max_angle_deg: float = 20.0,
    max_shared_yz_dist_cm: float = 35.0,
    max_endpoint_yz_dist_cm: float = 45.0,
    min_t0_separation_ticks: int = 8,
    max_passes: int = 8,
    improvement_eps: float = 0.0,
    adc_clip: float = 60780.0,
    search_range: int = 800,
    lock_swapped_clusters: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    candidate_rows, candidate_stats = build_track_swap_candidates_from_hits_v10_7_focused(
        track_labels=track_labels,
        labels_global=labels_global,
        hit_tpc_ids=hit_tpc_ids,
        x=x,
        y=y,
        z=z,
        energies=energies,
        label_info=label_info,
        min_energy_ratio=float(min_energy_ratio),
        min_shared_tpcs=int(min_shared_tpcs),
        max_tpc_sym_diff=int(max_tpc_sym_diff),
        max_angle_deg=float(max_angle_deg),
        max_shared_yz_dist_cm=float(max_shared_yz_dist_cm),
        max_endpoint_yz_dist_cm=float(max_endpoint_yz_dist_cm),
    )

    logs: list[dict[str, Any]] = []
    considered_pairs_by_pass: list[dict[str, Any]] = []
    swapped_clusters: set[int] = set()

    for pass_idx in range(int(max_passes)):
        best_swap: dict[str, Any] | None = None
        considered_this_pass = 0

        for pair_rank, row in enumerate(candidate_rows):
            cluster_a = int(row["cluster_a"])
            cluster_b = int(row["cluster_b"])
            if bool(lock_swapped_clusters) and (
                int(cluster_a) in swapped_clusters or int(cluster_b) in swapped_clusters
            ):
                continue

            t0_a = _cluster_assigned_t0(
                clusterid=int(cluster_a),
                cluster_to_tpcs=cluster_to_tpcs,
                image_maps=image_maps,
                assignment_info=assignment_info,
                max_tpcs=int(base_image.shape[0]),
            )
            t0_b = _cluster_assigned_t0(
                clusterid=int(cluster_b),
                cluster_to_tpcs=cluster_to_tpcs,
                image_maps=image_maps,
                assignment_info=assignment_info,
                max_tpcs=int(base_image.shape[0]),
            )
            if t0_a is None or t0_b is None or int(t0_a) == int(t0_b):
                continue
            if abs(int(t0_a) - int(t0_b)) < int(min_t0_separation_ticks):
                continue

            union_tpcs = [
                int(tpcid)
                for tpcid in row["union_tpcs"]
                if 0 <= int(tpcid) < int(base_image.shape[0])
                and (
                    (int(cluster_a), int(tpcid)) in image_maps
                    or (int(cluster_b), int(tpcid)) in image_maps
                )
            ]
            if len(union_tpcs) == 0:
                continue

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
            considered_this_pass += 1

            if improvement <= float(improvement_eps):
                continue

            if best_swap is None or improvement > float(best_swap["improvement"]):
                best_swap = {
                    "pass_index": int(pass_idx),
                    "pair_rank": int(pair_rank),
                    "cluster_a": int(cluster_a),
                    "cluster_b": int(cluster_b),
                    "t0_a_old": int(t0_a),
                    "t0_b_old": int(t0_b),
                    "t0_a_new": int(t0_b),
                    "t0_b_new": int(t0_a),
                    "current_score": float(current_score),
                    "swapped_score": float(swapped_score),
                    "improvement": float(improvement),
                    "fit_channels_by_tpc": {int(k): int(v) for k, v in fit_channels_by_tpc.items()},
                    "n_fit_channels_total": int(sum(int(v) for v in fit_channels_by_tpc.values())),
                    "union_tpcs": [int(v) for v in union_tpcs],
                    **{str(k): v for k, v in row.items() if str(k) not in {"cluster_a", "cluster_b", "union_tpcs"}},
                }

        considered_pairs_by_pass.append(
            {
                "pass_index": int(pass_idx),
                "pairs_scored": int(considered_this_pass),
            }
        )

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
            trigger_pair_rank=int(best_swap["pair_rank"]),
            search_range=int(search_range),
        )
        logs.append(dict(best_swap))
        swapped_clusters.update({int(best_swap["cluster_a"]), int(best_swap["cluster_b"])})

    stats = {
        **{str(k): v for k, v in candidate_stats.items()},
        "pairs_scored_by_pass": [dict(item) for item in considered_pairs_by_pass],
        "accepted_swaps": int(len(logs)),
        "swapped_clusters": sorted(int(v) for v in swapped_clusters),
        "min_t0_separation_ticks": int(min_t0_separation_ticks),
        "max_passes": int(max_passes),
        "improvement_eps": float(improvement_eps),
        "lock_swapped_clusters": bool(lock_swapped_clusters),
        "swap_log": [dict(item) for item in logs],
    }
    return base_image, hit_timestamps, t0_candidates, assignment_info, logs, stats


__all__ = [
    "build_track_swap_candidates_from_hits_v10_7_focused",
    "run_track_overlap_swap_rescue_v10_7_focused",
]
