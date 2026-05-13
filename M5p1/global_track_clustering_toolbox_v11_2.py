from __future__ import annotations

from typing import Any
from pathlib import Path
import sys
import importlib.util

import numpy as np
from sklearn.cluster import DBSCAN

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))


def _import_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_base_toolbox = _import_sibling(
    "_m5p1_global_track_clustering_toolbox_v11_2_sibling",
    "global_track_clustering_toolbox.py",
)
_global_track_clustering = _import_sibling(
    "_m5p1_global_track_clustering_v11_2_sibling",
    "global_track_clustering.py",
)

_build_tpc_segments_toolbox = _base_toolbox._build_tpc_segments_toolbox
_match_segments_across_tpcs_toolbox = _base_toolbox._match_segments_across_tpcs_toolbox
_fit_line_metrics = _base_toolbox._fit_line_metrics
_line_distances = _base_toolbox._line_distances
plot_selected_tpcs_clustering = _base_toolbox.plot_selected_tpcs_clustering

_assign_vertex_ids = _global_track_clustering._assign_vertex_ids
_cluster_track_endpoints = _global_track_clustering._cluster_track_endpoints
_tpc_id_from_io = _global_track_clustering._tpc_id_from_io


def _leftover_dbscan_membership(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    leftover_indices: np.ndarray,
    *,
    eps: float = 4.0,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    leftover_indices = np.asarray(leftover_indices, dtype=int)
    if leftover_indices.size == 0:
        return leftover_indices, np.zeros((0,), dtype=int)

    pts = np.column_stack([x[leftover_indices], y[leftover_indices], z[leftover_indices]])
    if pts.shape[0] < int(min_samples):
        return leftover_indices, np.full(pts.shape[0], -1, dtype=int)

    db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(pts)
    return leftover_indices, np.asarray(db.labels_, dtype=int)


def _build_track_local_models(
    global_tracks: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    lam: float,
    radius_scale: float,
    min_base_radius_cm: float,
    endpoint_margin_cm: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    models_by_tpc: dict[int, list[dict[str, Any]]] = {}
    track_stats: dict[int, dict[str, Any]] = {}

    for gid, gt in enumerate(global_tracks):
        per_tpc_hits: dict[int, list[np.ndarray]] = {}
        for seg in gt.get("segments", []):
            per_tpc_hits.setdefault(int(seg["tpc"]), []).append(np.asarray(seg["hits"], dtype=int))

        n_track_hits = int(np.asarray(gt["hit_indices"], dtype=int).size)
        track_stats[int(gid)] = {
            "n_hits_before_absorption": int(n_track_hits),
            "n_absorbed_noise_hits": 0,
            "tpc_models": {},
        }

        for tpc, parts in per_tpc_hits.items():
            hit_idx = np.unique(np.concatenate(parts))
            if hit_idx.size < 2:
                continue

            pts = np.column_stack([x[hit_idx], y[hit_idx], z[hit_idx]]).astype(np.float64)
            metrics = _fit_line_metrics(pts)
            proj = (pts - metrics["point"]) @ metrics["direction"]
            d = _line_distances(pts, metrics["point"], metrics["direction"])
            q90 = float(np.percentile(d, 90.0)) if d.size > 0 else 0.0
            base_radius = max(float(lam), float(min_base_radius_cm), float(q90))
            absorb_radius = float(radius_scale) * float(base_radius)
            margin = max(float(endpoint_margin_cm), float(absorb_radius))

            model = {
                "gid": int(gid),
                "tpc": int(tpc),
                "point": np.asarray(metrics["point"], dtype=np.float64),
                "direction": np.asarray(metrics["direction"], dtype=np.float64),
                "tmin": float(np.min(proj)),
                "tmax": float(np.max(proj)),
                "radius": float(absorb_radius),
                "margin": float(margin),
                "n_hits": int(hit_idx.size),
            }
            models_by_tpc.setdefault(int(tpc), []).append(model)
            track_stats[int(gid)]["tpc_models"][int(tpc)] = {
                "n_hits": int(hit_idx.size),
                "base_radius_cm": float(base_radius),
                "absorb_radius_cm": float(absorb_radius),
                "endpoint_margin_cm": float(margin),
            }

    return models_by_tpc, track_stats


def _fast_absorb_noise_into_tracks(
    global_tracks: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    io_group: np.ndarray,
    noise_indices: np.ndarray,
    *,
    lam: float,
    radius_scale: float = 1.5,
    min_base_radius_cm: float = 1.2,
    endpoint_margin_cm: float = 4.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    noise_indices = np.asarray(noise_indices, dtype=int)
    if len(global_tracks) == 0 or noise_indices.size == 0:
        return global_tracks, {
            "enabled": True,
            "n_noise_hits_considered": int(noise_indices.size),
            "n_absorbed_noise_hits": 0,
            "absorbed_hit_indices": np.zeros((0,), dtype=int),
            "absorbed_by_track": {},
            "track_stats": {},
            "tpc_stats": {},
        }

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    io_group = np.asarray(io_group, dtype=int)
    hit_tpc_ids = _tpc_id_from_io(io_group)

    models_by_tpc, track_stats = _build_track_local_models(
        global_tracks,
        x,
        y,
        z,
        lam=float(lam),
        radius_scale=float(radius_scale),
        min_base_radius_cm=float(min_base_radius_cm),
        endpoint_margin_cm=float(endpoint_margin_cm),
    )

    assigned_track = np.full(noise_indices.size, -1, dtype=int)
    assigned_score = np.full(noise_indices.size, np.inf, dtype=np.float64)
    tpc_stats: dict[int, dict[str, Any]] = {}

    for tpc in sorted(int(v) for v in np.unique(hit_tpc_ids[noise_indices])):
        local_models = models_by_tpc.get(int(tpc), [])
        if len(local_models) == 0:
            continue

        noise_mask = hit_tpc_ids[noise_indices] == int(tpc)
        local_pos = np.flatnonzero(noise_mask)
        if local_pos.size == 0:
            continue

        hit_idx = noise_indices[local_pos]
        pts = np.column_stack([x[hit_idx], y[hit_idx], z[hit_idx]]).astype(np.float64)
        best_local_score = np.full(local_pos.size, np.inf, dtype=np.float64)
        best_local_gid = np.full(local_pos.size, -1, dtype=int)

        for model in local_models:
            dif = pts - model["point"]
            proj = dif @ model["direction"]
            perp2 = np.sum(dif * dif, axis=1) - proj**2
            perp = np.sqrt(np.maximum(perp2, 0.0))
            ok = (
                (perp <= float(model["radius"]))
                & (proj >= float(model["tmin"]) - float(model["margin"]))
                & (proj <= float(model["tmax"]) + float(model["margin"]))
            )
            if not np.any(ok):
                continue

            score = perp / max(float(model["radius"]), 1e-6)
            update = ok & (score < best_local_score)
            if np.any(update):
                best_local_score[update] = score[update]
                best_local_gid[update] = int(model["gid"])

        chosen = best_local_gid >= 0
        assigned_track[local_pos[chosen]] = best_local_gid[chosen]
        assigned_score[local_pos[chosen]] = best_local_score[chosen]
        tpc_stats[int(tpc)] = {
            "n_noise_hits": int(local_pos.size),
            "n_track_models": int(len(local_models)),
            "n_absorbed_hits": int(np.count_nonzero(chosen)),
        }

    absorbed_mask = assigned_track >= 0
    absorbed_hit_indices = noise_indices[absorbed_mask]
    absorbed_by_track: dict[int, int] = {}

    if absorbed_hit_indices.size > 0:
        for gid in sorted(int(v) for v in np.unique(assigned_track[absorbed_mask])):
            extra_idx = noise_indices[assigned_track == int(gid)]
            if extra_idx.size == 0:
                continue
            global_tracks[int(gid)]["hit_indices"] = np.unique(
                np.concatenate(
                    [
                        np.asarray(global_tracks[int(gid)]["hit_indices"], dtype=int),
                        np.asarray(extra_idx, dtype=int),
                    ]
                )
            )
            pts_all = np.column_stack(
                [
                    x[global_tracks[int(gid)]["hit_indices"]],
                    y[global_tracks[int(gid)]["hit_indices"]],
                    z[global_tracks[int(gid)]["hit_indices"]],
                ]
            ).astype(np.float64)
            metrics = _fit_line_metrics(pts_all)
            global_tracks[int(gid)]["point"] = np.asarray(metrics["point"], dtype=np.float64)
            global_tracks[int(gid)]["direction"] = np.asarray(metrics["direction"], dtype=np.float64)
            global_tracks[int(gid)]["endpoints"] = np.asarray(metrics["endpoints"], dtype=np.float64)
            absorbed_by_track[int(gid)] = int(extra_idx.size)
            track_stats[int(gid)]["n_absorbed_noise_hits"] = int(extra_idx.size)

    return global_tracks, {
        "enabled": True,
        "n_noise_hits_considered": int(noise_indices.size),
        "n_absorbed_noise_hits": int(absorbed_hit_indices.size),
        "absorbed_hit_indices": np.asarray(absorbed_hit_indices, dtype=int),
        "absorbed_by_track": {int(k): int(v) for k, v in sorted(absorbed_by_track.items())},
        "track_stats": track_stats,
        "tpc_stats": tpc_stats,
        "assigned_score_summary": {
            "min": float(np.min(assigned_score[absorbed_mask])) if np.any(absorbed_mask) else np.nan,
            "median": float(np.median(assigned_score[absorbed_mask])) if np.any(absorbed_mask) else np.nan,
            "max": float(np.max(assigned_score[absorbed_mask])) if np.any(absorbed_mask) else np.nan,
        },
    }


def build_global_labels_toolbox(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    io_group: np.ndarray,
    *,
    lam: float = 1.5,
    rss_threshold: float = 1.5e6,
    iters: int = 800,
    min_inliers: int = 35,
    k_for_scale: int = 8,
    attach_multiplier: float = 1.3,
    seed: int = 0,
    min_length_cm: float = 30.0,
    n_tpcs: int = 70,
    match_dist_tol: float = 4.0,
    match_angle_deg: float = 10.0,
    match_endpoint_dist_tol: float = 25.0,
    match_endpoint_weight: float = 0.45,
    match_angle_weight: float = 0.35,
    match_quality_weight: float = 0.15,
    match_max_tpc_gap: int | None = None,
    vertex_eps: float = 10.0,
    vertex_min_samples: int = 3,
    min_tracks_for_shower: int = 3,
    split_track_components: bool = True,
    split_radius_cm: float = 4.0,
    split_min_component_hits: int = 20,
    promote_line_like_leftovers: bool = True,
    rescue_dbscan_eps: float = 4.0,
    rescue_dbscan_min_samples: int = 3,
    rescue_min_hits: int = 15,
    rescue_min_length_cm: float = 25.0,
    rescue_min_linearity: float = 0.92,
    rescue_max_transverse_rms: float = 3.5,
    track_noise_absorption_enable: bool = True,
    track_noise_absorb_radius_scale: float = 1.5,
    track_noise_absorb_min_base_radius_cm: float = 1.2,
    track_noise_absorb_endpoint_margin_cm: float = 4.0,
    leftover_dbscan_eps: float = 4.0,
    leftover_dbscan_min_samples: int = 3,
    return_label_info: bool = False,
    return_debug_info: bool = False,
) -> Any:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    io_group = np.asarray(io_group, dtype=int)
    n_hits = int(len(x))
    if not (len(y) == n_hits and len(z) == n_hits and len(io_group) == n_hits):
        raise ValueError("x, y, z, io_group must have the same length")

    segments, segment_debug = _build_tpc_segments_toolbox(
        x,
        y,
        z,
        io_group,
        lam=float(lam),
        rss_threshold=float(rss_threshold),
        iters=int(iters),
        min_inliers=int(min_inliers),
        k_for_scale=int(k_for_scale),
        attach_multiplier=float(attach_multiplier),
        seed=int(seed),
        min_length_cm=float(min_length_cm),
        n_tpcs=int(n_tpcs),
        split_track_components=bool(split_track_components),
        split_radius_cm=float(split_radius_cm),
        split_min_component_hits=int(split_min_component_hits),
        promote_line_like_leftovers=bool(promote_line_like_leftovers),
        rescue_dbscan_eps=float(rescue_dbscan_eps),
        rescue_dbscan_min_samples=int(rescue_dbscan_min_samples),
        rescue_min_hits=int(rescue_min_hits),
        rescue_min_length_cm=float(rescue_min_length_cm),
        rescue_min_linearity=float(rescue_min_linearity),
        rescue_max_transverse_rms=float(rescue_max_transverse_rms),
    )

    if len(segments) == 0:
        labels_global = np.full(n_hits, -1, dtype=int)
        outputs: list[Any] = [labels_global, 0]
        if return_label_info:
            outputs.append({})
        if return_debug_info:
            outputs.append(
                {
                    "segments": [],
                    "segment_debug": segment_debug,
                    "matching_debug": {"candidate_edges": [], "accepted_edges": []},
                    "global_tracks": [],
                    "track_noise_absorption": {"enabled": bool(track_noise_absorption_enable)},
                }
            )
        return tuple(outputs) if len(outputs) > 1 else labels_global

    global_tracks, matching_debug = _match_segments_across_tpcs_toolbox(
        segments,
        x,
        y,
        z,
        dist_tol=float(match_dist_tol),
        angle_tol_deg=float(match_angle_deg),
        endpoint_dist_tol=float(match_endpoint_dist_tol),
        endpoint_weight=float(match_endpoint_weight),
        angle_weight=float(match_angle_weight),
        quality_weight=float(match_quality_weight),
        max_tpc_gap=None if match_max_tpc_gap is None else int(match_max_tpc_gap),
    )

    if len(global_tracks) == 0:
        labels_global = np.full(n_hits, -1, dtype=int)
        for gid, seg in enumerate(segments):
            labels_global[np.asarray(seg["hits"], dtype=int)] = int(gid)
        label_info = {
            int(gid): {
                "type": "track",
                "tpcs": [int(seg["tpc"])],
                "n_hits": int(len(seg["hits"])),
                "n_tracks": 1,
                "track_ids": [int(gid)],
                "backbone": True,
                "segment_sources": [str(seg["source"])],
                "n_rescue_segments": 1 if str(seg["source"]) == "rescue_cluster" else 0,
            }
            for gid, seg in enumerate(segments)
        }
        outputs = [labels_global, len(segments)]
        if return_label_info:
            outputs.append(label_info)
        if return_debug_info:
            outputs.append(
                {
                    "segments": [dict(seg) for seg in segments],
                    "segment_debug": segment_debug,
                    "matching_debug": matching_debug,
                    "global_tracks": [dict(gt) for gt in global_tracks],
                    "track_noise_absorption": {"enabled": bool(track_noise_absorption_enable)},
                }
            )
        return tuple(outputs) if len(outputs) > 1 else labels_global

    provisional_track_labels = np.full(n_hits, -1, dtype=int)
    for gid, gt in enumerate(global_tracks):
        provisional_track_labels[np.asarray(gt["hit_indices"], dtype=int)] = int(gid)

    leftover_indices = np.flatnonzero(provisional_track_labels == -1)
    leftover_indices_db, leftover_db_labels = _leftover_dbscan_membership(
        x,
        y,
        z,
        leftover_indices,
        eps=float(leftover_dbscan_eps),
        min_samples=int(leftover_dbscan_min_samples),
    )
    noise_indices = leftover_indices_db[leftover_db_labels < 0]

    if bool(track_noise_absorption_enable):
        global_tracks, track_noise_debug = _fast_absorb_noise_into_tracks(
            global_tracks,
            x,
            y,
            z,
            io_group,
            noise_indices,
            lam=float(lam),
            radius_scale=float(track_noise_absorb_radius_scale),
            min_base_radius_cm=float(track_noise_absorb_min_base_radius_cm),
            endpoint_margin_cm=float(track_noise_absorb_endpoint_margin_cm),
        )
    else:
        track_noise_debug = {
            "enabled": False,
            "n_noise_hits_considered": int(noise_indices.size),
            "n_absorbed_noise_hits": 0,
            "absorbed_hit_indices": np.zeros((0,), dtype=int),
            "absorbed_by_track": {},
            "track_stats": {},
            "tpc_stats": {},
        }

    clusters, cluster_tracks, endpoint_xyz, endpoint_meta, ep_labels = _cluster_track_endpoints(
        global_tracks,
        eps=float(vertex_eps),
        min_samples=int(vertex_min_samples),
    )
    track_to_vertex, vertex_meta = _assign_vertex_ids(
        global_tracks,
        cluster_tracks,
        int(min_tracks_for_shower),
    )

    labels_global = np.full(n_hits, -1, dtype=int)
    unique_vids = sorted(set(int(vid) for vid in track_to_vertex.values()))
    vid_map = {int(old): int(new) for new, old in enumerate(unique_vids)}
    split_index = int(len(unique_vids))
    label_info: dict[int, dict[str, Any]] = {}

    for gid, gt in enumerate(global_tracks):
        vid = track_to_vertex.get(int(gid))
        if vid is None:
            continue
        labels_global[np.asarray(gt["hit_indices"], dtype=int)] = int(vid_map[int(vid)])

    for old_vid in unique_vids:
        meta = vertex_meta.get(int(old_vid), {})
        track_ids = [int(gid) for gid in meta.get("track_ids", [])]
        if len(track_ids) > 0:
            hit_indices = np.unique(np.concatenate([global_tracks[gid]["hit_indices"] for gid in track_ids]))
            tpcs = sorted(
                {
                    int(seg["tpc"])
                    for gid in track_ids
                    for seg in global_tracks[gid]["segments"]
                }
            )
            segment_sources = [
                str(seg["source"])
                for gid in track_ids
                for seg in global_tracks[gid]["segments"]
            ]
        else:
            hit_indices = np.zeros((0,), dtype=int)
            tpcs = []
            segment_sources = []
        label_info[int(vid_map[int(old_vid)])] = {
            "type": str(meta.get("type", "track")),
            "tpcs": tpcs,
            "n_hits": int(hit_indices.size),
            "n_tracks": int(meta.get("n_tracks", 1)),
            "track_ids": track_ids,
            "cluster_id": meta.get("cluster_id"),
            "backbone": True,
            "segment_sources": list(segment_sources),
            "n_rescue_segments": int(sum(src == "rescue_cluster" for src in segment_sources)),
        }

    max_vid = int(split_index - 1)
    for cid in sorted(int(v) for v in np.unique(leftover_db_labels) if int(v) >= 0):
        max_vid += 1
        global_hit_idx = leftover_indices_db[leftover_db_labels == int(cid)]
        labels_global[global_hit_idx] = int(max_vid)
        label_info[int(max_vid)] = {
            "type": "cluster",
            "tpcs": sorted(int(v) for v in np.unique(_tpc_id_from_io(io_group[global_hit_idx]))),
            "n_hits": int(len(global_hit_idx)),
            "n_tracks": 0,
            "track_ids": [],
            "cluster_id": None,
            "backbone": False,
            "segment_sources": [],
            "n_rescue_segments": 0,
        }

    outputs: list[Any] = [labels_global, split_index]
    if return_label_info:
        outputs.append(label_info)
    if return_debug_info:
        outputs.append(
            {
                "segments": [dict(seg) for seg in segments],
                "segment_debug": segment_debug,
                "matching_debug": matching_debug,
                "global_tracks": [dict(gt) for gt in global_tracks],
                "endpoint_xyz": np.asarray(endpoint_xyz, dtype=np.float64),
                "endpoint_meta": list(endpoint_meta),
                "endpoint_labels": np.asarray(ep_labels, dtype=int),
                "clusters": {int(k): list(v) for k, v in clusters.items()},
                "cluster_tracks": {int(k): [int(vv) for vv in vals] for k, vals in cluster_tracks.items()},
                "leftover_dbscan": {
                    "n_leftover_hits": int(leftover_indices_db.size),
                    "n_clustered_hits": int(np.count_nonzero(leftover_db_labels >= 0)),
                    "n_noise_hits": int(np.count_nonzero(leftover_db_labels < 0)),
                },
                "track_noise_absorption": track_noise_debug,
            }
        )
    return tuple(outputs) if len(outputs) > 1 else labels_global


__all__ = [
    "build_global_labels_toolbox",
    "plot_selected_tpcs_clustering",
]
