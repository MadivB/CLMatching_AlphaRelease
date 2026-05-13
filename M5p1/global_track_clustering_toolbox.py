from __future__ import annotations

from typing import Any
from pathlib import Path
import sys
import importlib.util

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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


_track_fit_ransac = _import_sibling("_m5p1_track_fit_ransac_toolbox_sibling", "track_fit_ransac.py")
_global_track_clustering = _import_sibling("_m5p1_global_track_clustering_toolbox_sibling", "global_track_clustering.py")

fit_tracks_labels = _track_fit_ransac.fit_tracks_labels
_assign_vertex_ids = _global_track_clustering._assign_vertex_ids
_cluster_track_endpoints = _global_track_clustering._cluster_track_endpoints
_extend_segment_to_active_box = _global_track_clustering._extend_segment_to_active_box
_segment_segment_distance = _global_track_clustering._segment_segment_distance
_tpc_id_from_io = _global_track_clustering._tpc_id_from_io
_angle_diff_deg = _global_track_clustering._angle_diff_deg


_PALETTE = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Light24
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Bold
)


def _line_distances(points: np.ndarray, p0: np.ndarray, v: np.ndarray) -> np.ndarray:
    dif = np.asarray(points, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.linalg.norm(dif, axis=1)
    v = v / nv
    dot_prod = np.dot(dif, v)
    sq_dist = np.sum(dif**2, axis=1) - dot_prod**2
    return np.sqrt(np.maximum(sq_dist, 0.0))


def _refit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    points = np.asarray(points, dtype=np.float64)
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    v = vt[0]
    v /= np.linalg.norm(v) + 1e-12
    t = (points - c) @ v
    p_a = c + t.min() * v
    p_b = c + t.max() * v
    return c, v, (p_a, p_b)


def _fit_line_metrics(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    c, v, (p_a, p_b) = _refit_line(points)
    singular_vals = np.linalg.svd(points - c, full_matrices=False, compute_uv=False)
    var = singular_vals**2
    linearity = float(var[0] / max(np.sum(var), 1e-12)) if singular_vals.size > 0 else 1.0
    transverse_rms = float(np.sqrt(np.mean(_line_distances(points, c, v) ** 2))) if len(points) > 0 else np.nan
    return {
        "point": c,
        "direction": v,
        "endpoints": (p_a, p_b),
        "length_cm": float(np.linalg.norm(p_b - p_a)),
        "linearity": float(linearity),
        "transverse_rms": float(transverse_rms),
    }


def _components_by_radius(points: np.ndarray, radius_cm: float) -> list[np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    n_points = int(points.shape[0])
    if n_points == 0:
        return []
    if n_points == 1:
        return [np.array([0], dtype=int)]

    parent = np.arange(n_points, dtype=int)
    radius2 = float(radius_cm) ** 2

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Local quadratic scan is acceptable because this only runs inside one label.
    for i in range(n_points):
        dif = points[i + 1 :] - points[i]
        if dif.size == 0:
            continue
        d2 = np.sum(dif * dif, axis=1)
        close = np.flatnonzero(d2 <= radius2)
        for off in close:
            union(i, int(i + 1 + off))

    groups: dict[int, list[int]] = {}
    for idx in range(n_points):
        groups.setdefault(find(idx), []).append(int(idx))

    components = [np.asarray(indices, dtype=int) for indices in groups.values()]
    components.sort(key=lambda arr: (-int(arr.size), int(arr[0])))
    return components


def _best_endpoint_pair(endpoints_a: np.ndarray, endpoints_b: np.ndarray) -> tuple[float, int, int]:
    best_dist = np.inf
    best_pair = (0, 0)
    pts_a = np.asarray(endpoints_a, dtype=np.float64)
    pts_b = np.asarray(endpoints_b, dtype=np.float64)
    for end_a in (0, 1):
        for end_b in (0, 1):
            dist = float(np.linalg.norm(pts_a[end_a] - pts_b[end_b]))
            if dist < best_dist:
                best_dist = dist
                best_pair = (int(end_a), int(end_b))
    return float(best_dist), int(best_pair[0]), int(best_pair[1])


def _segment_dict(
    *,
    tpc: int,
    hit_indices: np.ndarray,
    local_id: int,
    source: str,
    source_track_id: int | None,
    source_cluster_id: int | None,
) -> dict[str, Any]:
    metrics = _fit_line_metrics(np.asarray(hit_indices["points"], dtype=np.float64))
    return {
        "tpc": int(tpc),
        "local_id": int(local_id),
        "hits": np.asarray(hit_indices["indices"], dtype=int),
        "point": np.asarray(metrics["point"], dtype=np.float64),
        "direction": np.asarray(metrics["direction"], dtype=np.float64),
        "endpoints": np.asarray(metrics["endpoints"], dtype=np.float64),
        "n_hits": int(len(hit_indices["indices"])),
        "length_cm": float(metrics["length_cm"]),
        "linearity": float(metrics["linearity"]),
        "transverse_rms": float(metrics["transverse_rms"]),
        "source": str(source),
        "source_track_id": None if source_track_id is None else int(source_track_id),
        "source_cluster_id": None if source_cluster_id is None else int(source_cluster_id),
    }


def _build_tpc_segments_toolbox(
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    io_group = np.asarray(io_group, dtype=int)
    tpc_ids = _tpc_id_from_io(io_group)

    segments: list[dict[str, Any]] = []
    debug_tpcs: dict[int, dict[str, Any]] = {}

    for tpc in range(int(n_tpcs)):
        mask = tpc_ids == int(tpc)
        if not np.any(mask):
            continue

        idx_global = np.flatnonzero(mask)
        pts_tpc = np.column_stack([x[mask], y[mask], z[mask]]).astype(np.float64)

        labels_tpc, _ = fit_tracks_labels(
            pts_tpc[:, 0],
            pts_tpc[:, 1],
            pts_tpc[:, 2],
            lam=float(lam),
            rss_threshold=float(rss_threshold),
            iters=int(iters),
            min_inliers=int(min_inliers),
            k_for_scale=int(k_for_scale),
            attach_multiplier=float(attach_multiplier),
            seed=int(seed),
            min_length_cm=float(min_length_cm),
        )

        accepted_mask = np.zeros(len(idx_global), dtype=bool)
        track_segments_added = 0
        rescue_segments_added = 0
        dropped_track_fragments = 0

        local_track_ids = sorted(int(v) for v in np.unique(labels_tpc) if int(v) >= 0)
        next_local_id = max(local_track_ids, default=-1) + 1

        for track_id in local_track_ids:
            local_idx = np.flatnonzero(labels_tpc == int(track_id))
            if local_idx.size == 0:
                continue

            components = [np.asarray(local_idx, dtype=int)]
            if bool(split_track_components):
                comp_local = _components_by_radius(pts_tpc[local_idx], radius_cm=float(split_radius_cm))
                if len(comp_local) > 1:
                    components = [local_idx[sub_idx] for sub_idx in comp_local]

            kept_components = [comp for comp in components if int(comp.size) >= int(split_min_component_hits)]
            if len(kept_components) == 0:
                dropped_track_fragments += int(local_idx.size)
                continue

            for comp_id, comp_idx in enumerate(kept_components):
                accepted_mask[comp_idx] = True
                payload = {
                    "indices": idx_global[comp_idx],
                    "points": pts_tpc[comp_idx],
                }
                segments.append(
                    _segment_dict(
                        tpc=int(tpc),
                        hit_indices=payload,
                        local_id=int(next_local_id),
                        source="track_component" if len(kept_components) > 1 else "track",
                        source_track_id=int(track_id),
                        source_cluster_id=None,
                    )
                )
                next_local_id += 1
                track_segments_added += 1

            if len(kept_components) < len(components):
                kept_hits = int(sum(comp.size for comp in kept_components))
                dropped_track_fragments += int(local_idx.size - kept_hits)

        leftover_indices = np.flatnonzero(~accepted_mask)
        rescue_debug: list[dict[str, Any]] = []

        if bool(promote_line_like_leftovers) and leftover_indices.size >= int(rescue_dbscan_min_samples):
            db = DBSCAN(eps=float(rescue_dbscan_eps), min_samples=int(rescue_dbscan_min_samples)).fit(pts_tpc[leftover_indices])
            for cid in sorted(int(v) for v in np.unique(db.labels_) if int(v) >= 0):
                cluster_local = leftover_indices[np.flatnonzero(db.labels_ == int(cid))]
                if cluster_local.size < int(rescue_min_hits):
                    continue

                metrics = _fit_line_metrics(pts_tpc[cluster_local])
                if metrics["length_cm"] < float(rescue_min_length_cm):
                    continue
                if metrics["linearity"] < float(rescue_min_linearity):
                    continue
                if metrics["transverse_rms"] > float(rescue_max_transverse_rms):
                    continue

                payload = {
                    "indices": idx_global[cluster_local],
                    "points": pts_tpc[cluster_local],
                }
                segments.append(
                    _segment_dict(
                        tpc=int(tpc),
                        hit_indices=payload,
                        local_id=int(next_local_id),
                        source="rescue_cluster",
                        source_track_id=None,
                        source_cluster_id=int(cid),
                    )
                )
                rescue_segments_added += 1
                rescue_debug.append(
                    {
                        "cluster_id": int(cid),
                        "n_hits": int(cluster_local.size),
                        "length_cm": float(metrics["length_cm"]),
                        "linearity": float(metrics["linearity"]),
                        "transverse_rms": float(metrics["transverse_rms"]),
                    }
                )
                next_local_id += 1

        debug_tpcs[int(tpc)] = {
            "n_hits": int(idx_global.size),
            "n_initial_track_ids": int(len(local_track_ids)),
            "n_track_segments_added": int(track_segments_added),
            "n_rescue_segments_added": int(rescue_segments_added),
            "n_leftover_hits_after_track_split": int(leftover_indices.size),
            "n_dropped_track_fragment_hits": int(dropped_track_fragments),
            "rescue_segments": rescue_debug,
        }

    return segments, {
        "tpc_stats": debug_tpcs,
        "n_segments": int(len(segments)),
        "n_rescue_segments": int(sum(1 for seg in segments if seg["source"] == "rescue_cluster")),
    }


def _match_segments_across_tpcs_toolbox(
    segments: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    dist_tol: float = 4.0,
    angle_tol_deg: float = 10.0,
    endpoint_dist_tol: float = 25.0,
    endpoint_weight: float = 0.45,
    angle_weight: float = 0.35,
    quality_weight: float = 0.15,
    max_tpc_gap: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(segments) == 0:
        return [], {
            "candidate_edges": [],
            "accepted_edges": [],
            "n_candidate_edges": 0,
            "n_accepted_edges": 0,
        }

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    bounds = (
        float(np.min(x)),
        float(np.max(x)),
        float(np.min(y)),
        float(np.max(y)),
        float(np.min(z)),
        float(np.max(z)),
    )

    ext_endpoints = []
    for seg in segments:
        p1, p2 = np.asarray(seg["endpoints"], dtype=np.float64)
        ext_endpoints.append(_extend_segment_to_active_box(p1, p2, bounds))

    candidate_edges: list[dict[str, Any]] = []
    best_by_side: dict[tuple[int, int], dict[str, Any]] = {}

    for i in range(len(segments)):
        seg_i = segments[i]
        for j in range(i + 1, len(segments)):
            seg_j = segments[j]
            if int(seg_i["tpc"]) == int(seg_j["tpc"]):
                continue
            if max_tpc_gap is not None and abs(int(seg_i["tpc"]) - int(seg_j["tpc"])) > int(max_tpc_gap):
                continue

            angle_deg = float(_angle_diff_deg(seg_i["direction"], seg_j["direction"]))
            if angle_deg > float(angle_tol_deg):
                continue

            d_seg = float(
                _segment_segment_distance(
                    ext_endpoints[i][0],
                    ext_endpoints[i][1],
                    ext_endpoints[j][0],
                    ext_endpoints[j][1],
                )
            )
            if d_seg > float(dist_tol):
                continue

            endpoint_dist, end_i, end_j = _best_endpoint_pair(seg_i["endpoints"], seg_j["endpoints"])
            if endpoint_dist > float(endpoint_dist_tol):
                continue

            score = (
                float(d_seg) / max(float(dist_tol), 1e-6)
                + float(endpoint_weight) * (float(endpoint_dist) / max(float(endpoint_dist_tol), 1e-6))
                + float(angle_weight) * (float(angle_deg) / max(float(angle_tol_deg), 1e-6))
                - float(quality_weight) * min(float(seg_i["linearity"]), float(seg_j["linearity"]))
            )

            cand = {
                "i": int(i),
                "j": int(j),
                "tpc_i": int(seg_i["tpc"]),
                "tpc_j": int(seg_j["tpc"]),
                "end_i": int(end_i),
                "end_j": int(end_j),
                "score": float(score),
                "angle_deg": float(angle_deg),
                "segment_dist": float(d_seg),
                "endpoint_dist": float(endpoint_dist),
                "source_i": str(seg_i["source"]),
                "source_j": str(seg_j["source"]),
            }
            candidate_edges.append(cand)

            side_i = (int(i), int(end_i))
            side_j = (int(j), int(end_j))
            if side_i not in best_by_side or float(score) < float(best_by_side[side_i]["score"]):
                best_by_side[side_i] = cand
            if side_j not in best_by_side or float(score) < float(best_by_side[side_j]["score"]):
                best_by_side[side_j] = cand

    accepted_edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for cand in candidate_edges:
        side_i = (int(cand["i"]), int(cand["end_i"]))
        side_j = (int(cand["j"]), int(cand["end_j"]))
        if best_by_side.get(side_i) is not cand:
            continue
        if best_by_side.get(side_j) is not cand:
            continue

        pair_key = (min(int(cand["i"]), int(cand["j"])), max(int(cand["i"]), int(cand["j"])))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        accepted_edges.append(dict(cand))

    parent = np.arange(len(segments), dtype=int)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in accepted_edges:
        union(int(edge["i"]), int(edge["j"]))

    groups: dict[int, list[int]] = {}
    for i in range(len(segments)):
        groups.setdefault(find(i), []).append(int(i))

    global_tracks: list[dict[str, Any]] = []
    for members in groups.values():
        all_hits = np.unique(np.concatenate([segments[m]["hits"] for m in members]))
        pts = np.column_stack([x[all_hits], y[all_hits], z[all_hits]]).astype(np.float64)
        metrics = _fit_line_metrics(pts)
        global_tracks.append(
            {
                "segments": [segments[m] for m in members],
                "segment_indices": [int(m) for m in members],
                "hit_indices": np.asarray(all_hits, dtype=int),
                "point": np.asarray(metrics["point"], dtype=np.float64),
                "direction": np.asarray(metrics["direction"], dtype=np.float64),
                "endpoints": np.asarray(metrics["endpoints"], dtype=np.float64),
            }
        )

    debug = {
        "candidate_edges": [dict(item) for item in candidate_edges],
        "accepted_edges": [dict(item) for item in accepted_edges],
        "n_candidate_edges": int(len(candidate_edges)),
        "n_accepted_edges": int(len(accepted_edges)),
    }
    return global_tracks, debug


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
                }
            )
        return tuple(outputs) if len(outputs) > 1 else labels_global

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

    leftover_mask = labels_global == -1
    if np.any(leftover_mask):
        leftover_indices = np.where(leftover_mask)[0]
        pts = np.column_stack([x[leftover_mask], y[leftover_mask], z[leftover_mask]])
        db = DBSCAN(eps=4.0, min_samples=3).fit(pts)
        max_vid = int(split_index - 1)
        for cid in sorted(int(v) for v in np.unique(db.labels_) if int(v) >= 0):
            max_vid += 1
            idx_in_leftover = np.flatnonzero(db.labels_ == int(cid))
            global_hit_idx = leftover_indices[idx_in_leftover]
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

    outputs = [labels_global, split_index]
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
            }
        )
    return tuple(outputs) if len(outputs) > 1 else labels_global


def plot_selected_tpcs_clustering(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels: np.ndarray,
    tpc_ids: list[int] | tuple[int, ...] | np.ndarray,
    *,
    energies: np.ndarray | None = None,
    title: str = "Selected TPC clustering",
    save_path: str | None = None,
    show_noise: bool = True,
    marker_size: float = 3.5,
) -> go.Figure:
    tpc_ids = [int(tpc) for tpc in tpc_ids]
    if len(tpc_ids) == 0:
        raise ValueError("tpc_ids must contain at least one TPC id")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=int)
    if energies is not None:
        energies = np.asarray(energies, dtype=np.float64)

    mask = np.isin(hit_tpc_ids, np.asarray(tpc_ids, dtype=np.int64))
    if not np.any(mask):
        raise RuntimeError(f"No hits found in requested TPCs: {tpc_ids}")

    x_sel = x[mask]
    y_sel = y[mask]
    z_sel = z[mask]
    labels_sel = labels[mask]
    tpc_sel = hit_tpc_ids[mask]
    energy_sel = None if energies is None else energies[mask]

    unique_labels = sorted(int(v) for v in np.unique(labels_sel) if int(v) >= 0)
    if energy_sel is None:
        ordering = unique_labels
        label_energy = {int(lab): float(np.count_nonzero(labels_sel == int(lab))) for lab in unique_labels}
    else:
        label_energy = {
            int(lab): float(np.nansum(energy_sel[labels_sel == int(lab)]))
            for lab in unique_labels
        }
        ordering = sorted(unique_labels, key=lambda lab: (-label_energy[int(lab)], int(lab)))

    fig = go.Figure()
    for idx, lab in enumerate(ordering):
        label_mask = labels_sel == int(lab)
        color = _PALETTE[idx % len(_PALETTE)]
        hover_text = [
            f"label={int(lab)}<br>TPC={int(tpc)}"
            + (f"<br>E={float(e):.2f} MeV" if energy_sel is not None else "")
            for tpc, e in zip(
                tpc_sel[label_mask],
                np.zeros(int(np.count_nonzero(label_mask))) if energy_sel is None else energy_sel[label_mask],
            )
        ]
        name = f"Label {int(lab)}"
        if energy_sel is not None:
            name += f" (E = {label_energy[int(lab)]:.2f} MeV)"
        fig.add_trace(
            go.Scatter3d(
                x=z_sel[label_mask],
                y=y_sel[label_mask],
                z=x_sel[label_mask],
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=0.85, line=dict(width=0)),
                text=hover_text,
                hoverinfo="text+x+y+z",
                name=name,
            )
        )

    if bool(show_noise):
        noise_mask = labels_sel < 0
        if np.any(noise_mask):
            fig.add_trace(
                go.Scatter3d(
                    x=z_sel[noise_mask],
                    y=y_sel[noise_mask],
                    z=x_sel[noise_mask],
                    mode="markers",
                    marker=dict(size=max(marker_size - 0.5, 1.5), color="gray", opacity=0.35, line=dict(width=0)),
                    name=f"Noise ({int(np.count_nonzero(noise_mask))} hits)",
                )
            )

    fig.update_layout(
        scene=dict(xaxis_title="z", yaxis_title="y", zaxis_title="x"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"{title} | TPCs={tpc_ids}",
        showlegend=True,
    )

    if save_path is not None:
        fig.write_html(save_path)
    fig.show()
    return fig


def summarize_labels_on_selected_tpcs(
    labels: np.ndarray,
    hit_tpc_ids: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    tpc_ids: list[int] | tuple[int, ...] | np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=int)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int64)
    tpc_ids = [int(tpc) for tpc in tpc_ids]
    mask = np.isin(hit_tpc_ids, np.asarray(tpc_ids, dtype=np.int64))
    labels_here = sorted(int(v) for v in np.unique(labels[mask]) if int(v) >= 0)

    rows = []
    for lab in labels_here:
        info = dict(label_info.get(int(lab), {}))
        rows.append(
            {
                "label": int(lab),
                "type": str(info.get("type", "unknown")),
                "tpcs": [int(v) for v in info.get("tpcs", [])],
                "n_hits": int(info.get("n_hits", 0)),
                "n_tracks": int(info.get("n_tracks", 0)),
                "n_rescue_segments": int(info.get("n_rescue_segments", 0)),
                "segment_sources": list(info.get("segment_sources", [])),
            }
        )
    return rows


__all__ = [
    "build_global_labels_toolbox",
    "plot_selected_tpcs_clustering",
    "summarize_labels_on_selected_tpcs",
]
