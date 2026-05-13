from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
NEWML_DIR = ROOT_DIR / "NewMLSection"
if str(NEWML_DIR) not in sys.path:
    sys.path.insert(0, str(NEWML_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(1, str(THIS_DIR))

from ML_NDfull_perceiver import process_clusters_to_imageMaps
from global_track_clustering_toolbox_v11_2 import build_global_labels_toolbox
from pulse_shapes import timeinterpolation
from run_v11_vertex_eval import extract_truth_vertex_and_t0_for_hits
from v10_3_support import (
    build_cluster_channel_support_cache_v10_3,
    get_cluster_tpc_fit_block_v10_3,
    stack_cluster_fit_inputs_v10_3,
)
from v11_plotting_purpose_eval import (
    ADC_CLIP,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_FILE,
    DEFAULT_PULSE_PATH,
    T0_RESOLUTION,
    WVFM_LEN,
    _load_event_inputs,
    load_v11_plotting_resources,
    max_likelihood_curve_with_base,
)
from v13_2_track_rescan import _fit_track_t0_with_shower_guard_v13_2
from v12_saturation_mask import build_saturated_channel_cache_v12


@dataclass
class MicroFragment:
    fragment_id: int
    original_label: int
    hit_indices_local: np.ndarray
    energy_mev: float
    n_hits: int
    centroid: np.ndarray
    direction: np.ndarray
    endpoint_a: np.ndarray
    endpoint_b: np.ndarray
    start_point: np.ndarray
    end_point: np.ndarray
    linearity: float
    truth_vertex_id: int
    truth_t0: float
    truth_purity: float


def _smooth_1d(values: np.ndarray, width: int) -> np.ndarray:
    width = max(int(width), 1)
    if width <= 1:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def _find_local_maxima(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return np.asarray([], dtype=np.int32)
    core = (arr[1:-1] >= arr[:-2]) & (arr[1:-1] > arr[2:])
    return np.flatnonzero(core).astype(np.int32) + 1


def _select_separated_peaks(values: np.ndarray, *, min_peak_fraction: float, min_sep: int) -> list[int]:
    arr = np.asarray(values, dtype=np.float64)
    peaks = _find_local_maxima(arr)
    if peaks.size == 0:
        return []
    peak_values = arr[peaks]
    keep_mask = peak_values >= float(min_peak_fraction) * float(np.max(arr))
    peaks = peaks[keep_mask]
    if peaks.size == 0:
        return []
    ordered = sorted(peaks.tolist(), key=lambda idx: (-float(arr[idx]), int(idx)))
    chosen: list[int] = []
    for idx in ordered:
        if all(abs(int(idx) - int(prev)) > int(min_sep) for prev in chosen):
            chosen.append(int(idx))
    chosen.sort()
    return chosen


def _refine_peak_indices_on_raw(
    raw_values: np.ndarray,
    coarse_peaks: list[int] | np.ndarray,
    *,
    refine_radius: int,
    min_sep: int,
) -> list[int]:
    arr = np.asarray(raw_values, dtype=np.float64)
    if arr.size == 0:
        return []
    peaks = [int(v) for v in np.asarray(coarse_peaks, dtype=np.int32).tolist()]
    if len(peaks) == 0:
        return []

    refined: list[int] = []
    for peak in peaks:
        lo = max(0, int(peak) - int(refine_radius))
        hi = min(arr.size, int(peak) + int(refine_radius) + 1)
        if hi <= lo:
            continue
        local_idx = int(lo + np.argmax(arr[lo:hi]))
        refined.append(int(local_idx))

    if len(refined) == 0:
        return []

    ordered = sorted(set(refined), key=lambda idx: (-float(arr[int(idx)]), int(idx)))
    chosen: list[int] = []
    for idx in ordered:
        if all(abs(int(idx) - int(prev)) > int(min_sep) for prev in chosen):
            chosen.append(int(idx))
    chosen.sort()
    return chosen


def _normalize_direction(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def _linearity_and_endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    if pts.shape[0] == 1:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return center, direction, pts[0].copy(), pts[0].copy(), 1.0
    centered = pts - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    direction = _normalize_direction(vh[0])
    proj = centered @ direction
    endpoint_a = center + direction * float(np.min(proj))
    endpoint_b = center + direction * float(np.max(proj))
    var = singular * singular
    linearity = float(var[0] / max(float(np.sum(var)), 1e-12))
    return center, direction, endpoint_a, endpoint_b, linearity


def _endpoint_local_spread(points: np.ndarray, endpoint: np.ndarray, radius_cm: float) -> float:
    dif = np.asarray(points, dtype=np.float64) - np.asarray(endpoint, dtype=np.float64)[None, :]
    dist = np.linalg.norm(dif, axis=1)
    local = dif[dist <= float(radius_cm)]
    if local.shape[0] == 0:
        local = dif[np.argsort(dist)[: min(8, dif.shape[0])]]
    if local.shape[0] == 0:
        return 0.0
    centered = local - np.mean(local, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _choose_start_endpoint(points: np.ndarray, endpoint_a: np.ndarray, endpoint_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread_a = _endpoint_local_spread(points, endpoint_a, radius_cm=16.0)
    spread_b = _endpoint_local_spread(points, endpoint_b, radius_cm=16.0)
    if spread_a <= spread_b:
        return np.asarray(endpoint_a, dtype=np.float64), np.asarray(endpoint_b, dtype=np.float64)
    return np.asarray(endpoint_b, dtype=np.float64), np.asarray(endpoint_a, dtype=np.float64)


def _project_fragment_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    centroid, direction, endpoint_a, endpoint_b, linearity = _linearity_and_endpoints(points)
    start_point, end_point = _choose_start_endpoint(points, endpoint_a, endpoint_b)
    direction = _normalize_direction(end_point - start_point)
    return centroid, direction, endpoint_a, endpoint_b, start_point, end_point, linearity


def _line_point_distance(point: np.ndarray, anchor: np.ndarray, direction: np.ndarray) -> float:
    vec = np.asarray(point, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    direc = _normalize_direction(direction)
    proj = float(np.dot(vec, direc))
    perp = vec - proj * direc
    return float(np.linalg.norm(perp))


def _estimate_apex(fragments: list[MicroFragment], fragment_ids: list[int]) -> np.ndarray:
    if len(fragment_ids) == 0:
        return np.zeros(3, dtype=np.float64)
    lines: list[tuple[np.ndarray, np.ndarray, float]] = []
    for frag in fragments:
        if int(frag.fragment_id) not in set(int(v) for v in fragment_ids):
            continue
        weight = max(float(frag.energy_mev), 1e-6) * max(float(frag.linearity), 0.35)
        lines.append((np.asarray(frag.start_point, dtype=np.float64), np.asarray(frag.direction, dtype=np.float64), weight))
    if len(lines) == 0:
        return np.mean([np.asarray(f.centroid, dtype=np.float64) for f in fragments], axis=0)
    if len(lines) == 1:
        return np.asarray(lines[0][0], dtype=np.float64)
    mat = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for point, direction, weight in lines:
        direction = _normalize_direction(direction)
        proj = eye - np.outer(direction, direction)
        mat += float(weight) * proj
        rhs += float(weight) * proj @ np.asarray(point, dtype=np.float64)
    reg = 1e-6 * eye
    try:
        apex = np.linalg.solve(mat + reg, rhs)
    except np.linalg.LinAlgError:
        apex = np.average(np.asarray([line[0] for line in lines], dtype=np.float64), axis=0, weights=np.asarray([line[2] for line in lines], dtype=np.float64))
    return np.asarray(apex, dtype=np.float64)


def _fragment_truth_summary(
    *,
    truth_vertex_id: np.ndarray,
    truth_t0: np.ndarray,
    energies: np.ndarray,
    hit_mask: np.ndarray,
) -> tuple[int, float, float]:
    local_vertex = np.asarray(truth_vertex_id[hit_mask], dtype=np.int64)
    local_t0 = np.asarray(truth_t0[hit_mask], dtype=np.float64)
    local_energy = np.asarray(energies[hit_mask], dtype=np.float64)
    valid = (local_vertex >= 0) & np.isfinite(local_t0) & (local_energy > 0.0)
    if not np.any(valid):
        return -1, np.nan, np.nan
    uniq = np.unique(local_vertex[valid])
    energies_by_vertex = np.asarray([np.sum(local_energy[valid][local_vertex[valid] == int(vid)]) for vid in uniq], dtype=np.float64)
    dom_vid = int(uniq[int(np.argmax(energies_by_vertex))])
    dom_mask = valid & (local_vertex == int(dom_vid))
    purity = float(np.sum(local_energy[dom_mask]) / np.sum(local_energy[valid]))
    return int(dom_vid), float(np.median(local_t0[dom_mask])), float(purity)


def _cluster_points_dbscan(points: np.ndarray, eps_cm: float, min_samples: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.asarray([], dtype=np.int32)
    if pts.shape[0] < int(min_samples):
        return np.zeros(pts.shape[0], dtype=np.int32)
    db_labels = DBSCAN(eps=float(eps_cm), min_samples=int(min_samples)).fit_predict(pts).astype(np.int32)
    if np.all(db_labels < 0):
        return np.zeros(pts.shape[0], dtype=np.int32)
    if np.any(db_labels < 0):
        valid_labels = sorted(int(v) for v in np.unique(db_labels) if int(v) >= 0)
        valid_centroids = {
            int(lbl): np.mean(pts[db_labels == int(lbl)], axis=0)
            for lbl in valid_labels
        }
        for idx in np.flatnonzero(db_labels < 0).tolist():
            point = pts[int(idx)]
            nearest = min(
                valid_labels,
                key=lambda lbl: float(np.linalg.norm(point - valid_centroids[int(lbl)])),
            )
            db_labels[int(idx)] = int(nearest)
    return np.asarray(db_labels, dtype=np.int32)


def _radius_connected_components(points: np.ndarray, radius_cm: float) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return []
    if pts.shape[0] == 1:
        return [np.asarray([0], dtype=np.int32)]
    nbrs = NearestNeighbors(radius=float(radius_cm), algorithm="ball_tree").fit(pts)
    neighbors = nbrs.radius_neighbors(return_distance=False)
    seen = np.zeros(pts.shape[0], dtype=bool)
    comps: list[np.ndarray] = []
    for seed in range(pts.shape[0]):
        if bool(seen[seed]):
            continue
        stack = [int(seed)]
        seen[seed] = True
        comp: list[int] = []
        while stack:
            idx = int(stack.pop())
            comp.append(int(idx))
            for nxt in neighbors[idx].tolist():
                nxt = int(nxt)
                if not bool(seen[nxt]):
                    seen[nxt] = True
                    stack.append(int(nxt))
        comps.append(np.asarray(sorted(comp), dtype=np.int32))
    comps.sort(key=lambda arr: int(arr.size), reverse=True)
    return comps


def _local_hit_miss_to_apex(points: np.ndarray, apex: np.ndarray, n_neighbors: int = 12) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    apex = np.asarray(apex, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    if pts.shape[0] == 1:
        return np.asarray([float(np.linalg.norm(pts[0] - apex))], dtype=np.float64)
    n_neighbors = max(3, min(int(n_neighbors), int(pts.shape[0])))
    nbrs = NearestNeighbors(n_neighbors=int(n_neighbors), algorithm="ball_tree").fit(pts)
    _, indices = nbrs.kneighbors(pts)
    miss = np.zeros(pts.shape[0], dtype=np.float64)
    for i in range(pts.shape[0]):
        neigh = pts[np.asarray(indices[i], dtype=np.int32)]
        center = np.mean(neigh, axis=0)
        centered = neigh - center
        if neigh.shape[0] == 1:
            direction = _normalize_direction(center - apex)
        else:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = _normalize_direction(vh[0])
        ray = center - apex
        if float(np.dot(direction, ray)) < 0.0:
            direction = -direction
        miss[i] = _line_point_distance(apex, center, direction)
    return np.asarray(miss, dtype=np.float64)


def _split_component_by_apex_miss(
    *,
    points: np.ndarray,
    energies: np.ndarray,
    apex: np.ndarray,
    min_center_gap_cm: float = 8.0,
    min_side_energy_mev: float = 12.0,
    connected_radius_cm: float = 6.0,
) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    en = np.asarray(energies, dtype=np.float64)
    if pts.shape[0] < 24:
        return [np.arange(pts.shape[0], dtype=np.int32)]

    miss = _local_hit_miss_to_apex(pts, apex, n_neighbors=12)
    if miss.size < 24:
        return [np.arange(pts.shape[0], dtype=np.int32)]

    c0 = float(np.percentile(miss, 25.0))
    c1 = float(np.percentile(miss, 75.0))
    if abs(c1 - c0) < 1e-6:
        return [np.arange(pts.shape[0], dtype=np.int32)]
    for _ in range(12):
        assign0 = np.abs(miss - c0) <= np.abs(miss - c1)
        assign1 = ~assign0
        if int(np.count_nonzero(assign0)) == 0 or int(np.count_nonzero(assign1)) == 0:
            return [np.arange(pts.shape[0], dtype=np.int32)]
        new_c0 = float(np.mean(miss[assign0]))
        new_c1 = float(np.mean(miss[assign1]))
        if abs(new_c0 - c0) < 1e-4 and abs(new_c1 - c1) < 1e-4:
            c0, c1 = new_c0, new_c1
            break
        c0, c1 = new_c0, new_c1
    low_center, high_center = sorted([float(c0), float(c1)])
    if float(high_center - low_center) < float(min_center_gap_cm):
        return [np.arange(pts.shape[0], dtype=np.int32)]

    threshold = 0.5 * (float(low_center) + float(high_center))
    low_mask = miss <= threshold
    high_mask = miss > threshold
    if float(np.sum(en[low_mask])) < float(min_side_energy_mev) or float(np.sum(en[high_mask])) < float(min_side_energy_mev):
        return [np.arange(pts.shape[0], dtype=np.int32)]

    parts: list[np.ndarray] = []
    for mask in [low_mask, high_mask]:
        side_idx = np.flatnonzero(mask).astype(np.int32)
        if side_idx.size == 0:
            continue
        side_pts = pts[side_idx]
        for comp in _radius_connected_components(side_pts, radius_cm=float(connected_radius_cm)):
            if comp.size == 0:
                continue
            parts.append(np.asarray(side_idx[np.asarray(comp, dtype=np.int32)], dtype=np.int32))
    if len(parts) <= 1:
        return [np.arange(pts.shape[0], dtype=np.int32)]
    return parts


def _estimate_provisional_shower_apex(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    target_tpc: int,
) -> np.ndarray | None:
    local_mask = (np.asarray(hit_tpc_ids, dtype=np.int32) == int(target_tpc)) & (np.asarray(labels_global, dtype=np.int32) >= 0)
    if not np.any(local_mask):
        return None
    local_labels = sorted(int(v) for v in np.unique(np.asarray(labels_global, dtype=np.int32)[local_mask]))
    if len(local_labels) == 0:
        return None
    energy_by_label = {
        int(lbl): float(np.sum(np.asarray(energies, dtype=np.float64)[local_mask & (np.asarray(labels_global, dtype=np.int32) == int(lbl))]))
        for lbl in local_labels
    }
    shower_labels = [
        int(lbl)
        for lbl in local_labels
        if str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() == "shower"
    ]
    if len(shower_labels) > 0:
        seed_label = max(shower_labels, key=lambda lbl: float(energy_by_label[int(lbl)]))
    else:
        nontrack_labels = [
            int(lbl)
            for lbl in local_labels
            if str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() != "track"
        ]
        if len(nontrack_labels) == 0:
            return None
        seed_label = max(nontrack_labels, key=lambda lbl: float(energy_by_label[int(lbl)]))

    mask = local_mask & (np.asarray(labels_global, dtype=np.int32) == int(seed_label))
    pts = np.column_stack((
        np.asarray(x, dtype=np.float64)[mask],
        np.asarray(y, dtype=np.float64)[mask],
        np.asarray(z, dtype=np.float64)[mask],
    ))
    if pts.shape[0] == 0:
        return None
    _, _, _, _, start_point, _, _ = _project_fragment_geometry(pts)
    return np.asarray(start_point, dtype=np.float64)


def _estimate_provisional_shower_seed_label(
    *,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    target_tpc: int,
) -> tuple[int | None, dict[int, float]]:
    local_mask = (np.asarray(hit_tpc_ids, dtype=np.int32) == int(target_tpc)) & (np.asarray(labels_global, dtype=np.int32) >= 0)
    if not np.any(local_mask):
        return None, {}
    local_labels = sorted(int(v) for v in np.unique(np.asarray(labels_global, dtype=np.int32)[local_mask]))
    if len(local_labels) == 0:
        return None, {}
    energy_by_label = {
        int(lbl): float(np.sum(np.asarray(energies, dtype=np.float64)[local_mask & (np.asarray(labels_global, dtype=np.int32) == int(lbl))]))
        for lbl in local_labels
    }
    shower_labels = [
        int(lbl)
        for lbl in local_labels
        if str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() == "shower"
    ]
    if len(shower_labels) > 0:
        return int(max(shower_labels, key=lambda lbl: float(energy_by_label[int(lbl)]))), energy_by_label
    nontrack_labels = [
        int(lbl)
        for lbl in local_labels
        if str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() != "track"
    ]
    if len(nontrack_labels) == 0:
        return None, energy_by_label
    return int(max(nontrack_labels, key=lambda lbl: float(energy_by_label[int(lbl)]))), energy_by_label


def _project_label_geometry_for_tpcs(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label: int,
    tpcs: list[int] | tuple[int, ...] | np.ndarray,
) -> dict[str, Any] | None:
    tpc_set = set(int(v) for v in np.asarray(tpcs, dtype=np.int32).tolist())
    mask = np.isin(np.asarray(hit_tpc_ids, dtype=np.int32), np.asarray(sorted(tpc_set), dtype=np.int32)) & (np.asarray(labels_global, dtype=np.int32) == int(label))
    if not np.any(mask):
        return None
    pts = np.column_stack((
        np.asarray(x, dtype=np.float64)[mask],
        np.asarray(y, dtype=np.float64)[mask],
        np.asarray(z, dtype=np.float64)[mask],
    ))
    centroid, direction, endpoint_a, endpoint_b, start_point, end_point, linearity = _project_fragment_geometry(pts)
    return {
        "label": int(label),
        "tpcs": sorted(int(v) for v in np.unique(np.asarray(hit_tpc_ids, dtype=np.int32)[mask]) if int(v) >= 0),
        "n_hits": int(pts.shape[0]),
        "energy_mev": float(np.sum(np.asarray(energies, dtype=np.float64)[mask])),
        "centroid": np.asarray(centroid, dtype=np.float64),
        "direction": np.asarray(direction, dtype=np.float64),
        "start_point": np.asarray(start_point, dtype=np.float64),
        "end_point": np.asarray(end_point, dtype=np.float64),
        "linearity": float(linearity),
        "endpoint_a": np.asarray(endpoint_a, dtype=np.float64),
        "endpoint_b": np.asarray(endpoint_b, dtype=np.float64),
        "points": np.asarray(pts, dtype=np.float64),
    }


def _score_apex_candidate_against_local_nontrack(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    target_tpc: int,
    apex: np.ndarray,
    min_label_energy_mev: float = 15.0,
) -> tuple[float, list[dict[str, float]]]:
    apex = np.asarray(apex, dtype=np.float64)
    local_mask = (np.asarray(hit_tpc_ids, dtype=np.int32) == int(target_tpc)) & (np.asarray(labels_global, dtype=np.int32) >= 0)
    if not np.any(local_mask):
        return np.inf, []
    local_labels = [
        int(lbl)
        for lbl in sorted(int(v) for v in np.unique(np.asarray(labels_global, dtype=np.int32)[local_mask]))
        if str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() != "track"
    ]
    rows: list[dict[str, float]] = []
    total_weight = 0.0
    total_score = 0.0
    for label in local_labels:
        label_mask = local_mask & (np.asarray(labels_global, dtype=np.int32) == int(label))
        label_energy = float(np.sum(np.asarray(energies, dtype=np.float64)[label_mask]))
        if label_energy < float(min_label_energy_mev):
            continue
        pts = np.column_stack((
            np.asarray(x, dtype=np.float64)[label_mask],
            np.asarray(y, dtype=np.float64)[label_mask],
            np.asarray(z, dtype=np.float64)[label_mask],
        ))
        if pts.shape[0] == 0:
            continue
        centroid, direction, _, _, start_point, _, linearity = _project_fragment_geometry(pts)
        miss = float(_line_point_distance(apex, start_point, direction))
        ray = np.asarray(centroid, dtype=np.float64) - apex
        ray_norm = float(np.linalg.norm(ray))
        angle_pen = 0.0 if ray_norm <= 1e-8 else 1.0 - abs(float(np.dot(_normalize_direction(direction), ray / ray_norm)))
        label_score = min(miss / 28.0, 6.0) + 0.70 * min(angle_pen / 0.35, 6.0)
        weight = float(label_energy) * max(float(linearity), 0.35)
        rows.append(
            {
                "label": float(label),
                "energy_mev": float(label_energy),
                "linearity": float(linearity),
                "miss_cm": float(miss),
                "angle_penalty": float(angle_pen),
                "score": float(label_score),
                "weight": float(weight),
            }
        )
        total_weight += float(weight)
        total_score += float(weight) * float(label_score)
    if total_weight <= 0.0:
        return np.inf, rows
    return float(total_score / total_weight), rows


def _estimate_neighbor_aware_shower_apex(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    target_tpc: int,
    neighbor_tpc_gap: int = 1,
    min_neighbor_energy_mev: float = 40.0,
    min_label_shift_cm: float = 20.0,
    min_score_improvement_frac: float = 0.08,
    aligned_shared_label_min_energy_mev: float = 35.0,
    aligned_shared_label_min_hits: int = 40,
    align_cos_min: float = 0.78,
    align_line_dist_max_cm: float = 45.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    target_tpc = int(target_tpc)
    local_apex = _estimate_provisional_shower_apex(
        x=x,
        y=y,
        z=z,
        energies=energies,
        hit_tpc_ids=hit_tpc_ids,
        labels_global=labels_global,
        label_info=label_info,
        target_tpc=int(target_tpc),
    )
    seed_label, energy_by_label = _estimate_provisional_shower_seed_label(
        energies=energies,
        hit_tpc_ids=hit_tpc_ids,
        labels_global=labels_global,
        label_info=label_info,
        target_tpc=int(target_tpc),
    )
    debug: dict[str, Any] = {
        "mode": "local",
        "seed_label": None if seed_label is None else int(seed_label),
        "local_apex_xyz": None if local_apex is None else [float(v) for v in np.asarray(local_apex, dtype=np.float64).tolist()],
        "selected_apex_xyz": None if local_apex is None else [float(v) for v in np.asarray(local_apex, dtype=np.float64).tolist()],
        "selected_labels": [] if seed_label is None else [int(seed_label)],
        "selected_tpcs": [int(target_tpc)],
        "local_score": None,
        "propagated_score": None,
    }
    if local_apex is None or seed_label is None:
        return local_apex, debug

    local_geom = _project_label_geometry_for_tpcs(
        x=x,
        y=y,
        z=z,
        energies=energies,
        hit_tpc_ids=hit_tpc_ids,
        labels_global=labels_global,
        label=int(seed_label),
        tpcs=[int(target_tpc)],
    )
    if local_geom is None:
        return local_apex, debug

    seed_tpcs_all = sorted(int(v) for v in np.unique(np.asarray(hit_tpc_ids, dtype=np.int32)[np.asarray(labels_global, dtype=np.int32) == int(seed_label)]) if int(v) >= 0)
    candidate_tpcs = sorted(
        int(v)
        for v in seed_tpcs_all
        if abs(int(v) - int(target_tpc)) <= int(neighbor_tpc_gap)
    )
    if int(target_tpc) not in set(candidate_tpcs):
        candidate_tpcs.append(int(target_tpc))
        candidate_tpcs.sort()

    shared_neighbor_tpcs = [int(v) for v in candidate_tpcs if int(v) != int(target_tpc)]
    if len(shared_neighbor_tpcs) == 0:
        return local_apex, debug

    neighbor_energy = float(
        np.sum(
            np.asarray(energies, dtype=np.float64)[
                np.isin(np.asarray(hit_tpc_ids, dtype=np.int32), np.asarray(shared_neighbor_tpcs, dtype=np.int32))
                & (np.asarray(labels_global, dtype=np.int32) == int(seed_label))
            ]
        )
    )
    if neighbor_energy < float(min_neighbor_energy_mev):
        return local_apex, debug

    seed_chain_geom = _project_label_geometry_for_tpcs(
        x=x,
        y=y,
        z=z,
        energies=energies,
        hit_tpc_ids=hit_tpc_ids,
        labels_global=labels_global,
        label=int(seed_label),
        tpcs=candidate_tpcs,
    )
    if seed_chain_geom is None:
        return local_apex, debug

    selected_labels = [int(seed_label)]
    selected_points = [np.asarray(seed_chain_geom["points"], dtype=np.float64)]
    base_direction = _normalize_direction(np.asarray(seed_chain_geom["direction"], dtype=np.float64))
    base_start = np.asarray(seed_chain_geom["start_point"], dtype=np.float64)

    local_mask = (np.asarray(hit_tpc_ids, dtype=np.int32) == int(target_tpc)) & (np.asarray(labels_global, dtype=np.int32) >= 0)
    local_nontrack_labels = [
        int(lbl)
        for lbl in sorted(int(v) for v in np.unique(np.asarray(labels_global, dtype=np.int32)[local_mask]))
        if int(lbl) != int(seed_label) and str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() != "track"
    ]
    for label in local_nontrack_labels:
        local_energy = float(energy_by_label.get(int(label), 0.0))
        if local_energy < float(aligned_shared_label_min_energy_mev):
            continue
        label_tpcs = sorted(int(v) for v in np.unique(np.asarray(hit_tpc_ids, dtype=np.int32)[np.asarray(labels_global, dtype=np.int32) == int(label)]) if int(v) >= 0)
        if int(target_tpc) not in set(label_tpcs):
            continue
        if not any(abs(int(v) - int(target_tpc)) <= int(neighbor_tpc_gap) and int(v) != int(target_tpc) for v in label_tpcs):
            continue
        geom = _project_label_geometry_for_tpcs(
            x=x,
            y=y,
            z=z,
            energies=energies,
            hit_tpc_ids=hit_tpc_ids,
            labels_global=labels_global,
            label=int(label),
            tpcs=[int(v) for v in candidate_tpcs if int(v) in set(label_tpcs)],
        )
        if geom is None or int(geom["n_hits"]) < int(aligned_shared_label_min_hits):
            continue
        align = abs(float(np.dot(_normalize_direction(np.asarray(geom["direction"], dtype=np.float64)), base_direction)))
        line_dist = float(_line_point_distance(np.asarray(geom["start_point"], dtype=np.float64), base_start, base_direction))
        if align < float(align_cos_min) or line_dist > float(align_line_dist_max_cm):
            continue
        selected_labels.append(int(label))
        selected_points.append(np.asarray(geom["points"], dtype=np.float64))

    propagated_points = np.concatenate(selected_points, axis=0) if len(selected_points) > 1 else np.asarray(selected_points[0], dtype=np.float64)
    _, _, _, _, propagated_apex, _, _ = _project_fragment_geometry(propagated_points)

    local_score, local_rows = _score_apex_candidate_against_local_nontrack(
        x=x,
        y=y,
        z=z,
        energies=energies,
        hit_tpc_ids=hit_tpc_ids,
        labels_global=labels_global,
        label_info=label_info,
        target_tpc=int(target_tpc),
        apex=np.asarray(local_apex, dtype=np.float64),
    )
    propagated_score, propagated_rows = _score_apex_candidate_against_local_nontrack(
        x=x,
        y=y,
        z=z,
        energies=energies,
        hit_tpc_ids=hit_tpc_ids,
        labels_global=labels_global,
        label_info=label_info,
        target_tpc=int(target_tpc),
        apex=np.asarray(propagated_apex, dtype=np.float64),
    )
    apex_shift_cm = float(np.linalg.norm(np.asarray(propagated_apex, dtype=np.float64) - np.asarray(local_apex, dtype=np.float64)))
    improve = (
        np.isfinite(local_score)
        and np.isfinite(propagated_score)
        and float(apex_shift_cm) >= float(min_label_shift_cm)
        and float(propagated_score) <= (1.0 - float(min_score_improvement_frac)) * float(local_score)
    )

    debug.update(
        {
            "candidate_tpcs": [int(v) for v in candidate_tpcs],
            "shared_neighbor_tpcs": [int(v) for v in shared_neighbor_tpcs],
            "neighbor_energy_mev": float(neighbor_energy),
            "selected_labels": [int(v) for v in selected_labels],
            "seed_chain_direction": [float(v) for v in np.asarray(base_direction, dtype=np.float64).tolist()],
            "seed_chain_start_xyz": [float(v) for v in np.asarray(base_start, dtype=np.float64).tolist()],
            "propagated_apex_xyz": [float(v) for v in np.asarray(propagated_apex, dtype=np.float64).tolist()],
            "local_score": None if not np.isfinite(local_score) else float(local_score),
            "propagated_score": None if not np.isfinite(propagated_score) else float(propagated_score),
            "apex_shift_cm": float(apex_shift_cm),
            "local_label_rows": local_rows,
            "propagated_label_rows": propagated_rows,
        }
    )
    if bool(improve):
        debug["mode"] = "neighbor_propagated"
        debug["selected_apex_xyz"] = [float(v) for v in np.asarray(propagated_apex, dtype=np.float64).tolist()]
        debug["selected_tpcs"] = [int(v) for v in candidate_tpcs]
        return np.asarray(propagated_apex, dtype=np.float64), debug

    return np.asarray(local_apex, dtype=np.float64), debug


def split_nontrack_hits_into_microfragments(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    truth_vertex_id: np.ndarray,
    truth_t0: np.ndarray,
    target_tpc: int,
    dbscan_eps_cm: float = 6.0,
    dbscan_min_samples: int = 4,
    min_fragment_energy_mev: float = 0.25,
    provisional_apex: np.ndarray | None = None,
    fine_dbscan_eps_cm: float | None = None,
    fine_dbscan_min_samples: int | None = None,
    large_label_energy_mev: float = 80.0,
    miss_split_energy_mev: float = 80.0,
) -> tuple[list[MicroFragment], np.ndarray]:
    target_tpc = int(target_tpc)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    labels_global = np.asarray(labels_global, dtype=np.int32)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    truth_vertex_id = np.asarray(truth_vertex_id, dtype=np.int64)
    truth_t0 = np.asarray(truth_t0, dtype=np.float64)

    local_mask = (hit_tpc_ids == int(target_tpc)) & (labels_global >= 0)
    nontrack_labels = sorted(
        {
            int(label)
            for label in np.unique(labels_global[local_mask])
            if str(label_info.get(int(label), {}).get("type", "cluster")).lower() != "track"
        }
    )

    fragment_ids_local = np.full(labels_global.shape[0], -1, dtype=np.int32)
    fragments: list[MicroFragment] = []
    next_fragment_id = 0

    for original_label in nontrack_labels:
        mask = local_mask & (labels_global == int(original_label))
        if not np.any(mask):
            continue
        local_hit_indices = np.flatnonzero(mask).astype(np.int32)
        pts = np.column_stack((x[local_hit_indices], y[local_hit_indices], z[local_hit_indices]))
        label_energy_mev = float(np.sum(energies[local_hit_indices]))
        use_fine = (
            provisional_apex is not None
            and fine_dbscan_eps_cm is not None
            and fine_dbscan_min_samples is not None
            and float(label_energy_mev) >= float(large_label_energy_mev)
        )
        base_eps = float(fine_dbscan_eps_cm) if bool(use_fine) else float(dbscan_eps_cm)
        base_min_samples = int(fine_dbscan_min_samples) if bool(use_fine) else int(dbscan_min_samples)
        db_labels = _cluster_points_dbscan(pts, eps_cm=float(base_eps), min_samples=int(base_min_samples))

        part_hit_lists: list[np.ndarray] = []
        for part in sorted(int(v) for v in np.unique(db_labels) if int(v) >= 0):
            part_mask_local = db_labels == int(part)
            part_hit_indices = np.asarray(local_hit_indices[part_mask_local], dtype=np.int32)
            if part_hit_indices.size == 0:
                continue
            if (
                provisional_apex is not None
                and float(np.sum(energies[part_hit_indices])) >= float(miss_split_energy_mev)
            ):
                part_pts = np.column_stack((x[part_hit_indices], y[part_hit_indices], z[part_hit_indices]))
                split_parts = _split_component_by_apex_miss(
                    points=part_pts,
                    energies=np.asarray(energies[part_hit_indices], dtype=np.float64),
                    apex=np.asarray(provisional_apex, dtype=np.float64),
                    min_center_gap_cm=8.0,
                    min_side_energy_mev=12.0,
                    connected_radius_cm=6.0,
                )
                if len(split_parts) > 1:
                    for local_idx in split_parts:
                        refined_indices = np.asarray(part_hit_indices[np.asarray(local_idx, dtype=np.int32)], dtype=np.int32)
                        if refined_indices.size > 0:
                            part_hit_lists.append(refined_indices)
                    continue
            part_hit_lists.append(np.asarray(part_hit_indices, dtype=np.int32))

        for part_hit_indices in part_hit_lists:
            energy_mev = float(np.sum(energies[part_hit_indices]))
            if energy_mev < float(min_fragment_energy_mev):
                continue
            part_pts = np.column_stack((x[part_hit_indices], y[part_hit_indices], z[part_hit_indices]))
            centroid, direction, endpoint_a, endpoint_b, start_point, end_point, linearity = _project_fragment_geometry(part_pts)
            dom_vid, dom_t0, truth_purity = _fragment_truth_summary(
                truth_vertex_id=truth_vertex_id,
                truth_t0=truth_t0,
                energies=energies,
                hit_mask=np.isin(np.arange(labels_global.shape[0]), part_hit_indices),
            )
            frag = MicroFragment(
                fragment_id=int(next_fragment_id),
                original_label=int(original_label),
                hit_indices_local=np.asarray(part_hit_indices, dtype=np.int32),
                energy_mev=float(energy_mev),
                n_hits=int(part_hit_indices.size),
                centroid=np.asarray(centroid, dtype=np.float64),
                direction=np.asarray(direction, dtype=np.float64),
                endpoint_a=np.asarray(endpoint_a, dtype=np.float64),
                endpoint_b=np.asarray(endpoint_b, dtype=np.float64),
                start_point=np.asarray(start_point, dtype=np.float64),
                end_point=np.asarray(end_point, dtype=np.float64),
                linearity=float(linearity),
                truth_vertex_id=int(dom_vid),
                truth_t0=float(dom_t0),
                truth_purity=float(truth_purity),
            )
            fragments.append(frag)
            fragment_ids_local[part_hit_indices] = int(next_fragment_id)
            next_fragment_id += 1

    return fragments, fragment_ids_local


def _build_selected_prediction_inputs(
    *,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    target_tpc: int,
    fragments: list[MicroFragment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]], int, list[int], list[int]]:
    target_tpc = int(target_tpc)
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)

    local_mask = hit_tpc_ids == int(target_tpc)
    track_labels = sorted(
        {
            int(label)
            for label in np.unique(labels_global[local_mask])
            if int(label) >= 0 and str(label_info.get(int(label), {}).get("type", "cluster")).lower() == "track"
        }
    )
    track_hit_mask = np.isin(labels_global, np.asarray(track_labels, dtype=np.int32))

    microfrag_offset = int(np.max(labels_global[labels_global >= 0])) + 1000 if np.any(labels_global >= 0) else 1000
    fragment_label_by_id = {int(f.fragment_id): int(microfrag_offset + int(f.fragment_id)) for f in fragments}

    selected_mask = np.asarray(track_hit_mask | local_mask, dtype=bool)
    selected_indices = np.flatnonzero(selected_mask).astype(np.int32)
    selected_labels = np.asarray(labels_global[selected_mask], dtype=np.int32)

    if fragments:
        selected_indices_map = {int(idx): pos for pos, idx in enumerate(selected_indices.tolist())}
        for fragment in fragments:
            new_label = int(fragment_label_by_id[int(fragment.fragment_id)])
            for hit_idx in fragment.hit_indices_local.tolist():
                pos = selected_indices_map.get(int(hit_idx))
                if pos is not None:
                    selected_labels[int(pos)] = int(new_label)

    label_info_selected: dict[int, dict[str, Any]] = {}
    for label in track_labels:
        label_info_selected[int(label)] = dict(label_info.get(int(label), {}))
    for fragment in fragments:
        new_label = int(fragment_label_by_id[int(fragment.fragment_id)])
        label_info_selected[int(new_label)] = {
            "type": "cluster",
            "original_label": int(fragment.original_label),
            "fragment_id": int(fragment.fragment_id),
            "linearity": float(fragment.linearity),
            "energy_mev": float(fragment.energy_mev),
        }

    return (
        np.asarray(x[selected_mask], dtype=np.float64),
        np.asarray(y[selected_mask], dtype=np.float64),
        np.asarray(z[selected_mask], dtype=np.float64),
        np.asarray(energies[selected_mask], dtype=np.float64),
        np.asarray(hit_tpc_ids[selected_mask], dtype=np.int32),
        np.asarray(selected_labels, dtype=np.int32),
        label_info_selected,
        int(microfrag_offset),
        [int(v) for v in track_labels],
        [int(fragment_label_by_id[int(f.fragment_id)]) for f in fragments],
    )


def _fit_track_base_on_selected_labels(
    *,
    track_labels: list[int],
    image_maps: dict[tuple[int, int], np.ndarray],
    cluster_to_tpcs: dict[int, list[int]],
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    saturated_channel_cache: dict[str, Any] | None,
    labels_global_selected: np.ndarray | None = None,
    hit_tpc_ids_selected: np.ndarray | None = None,
    energies_selected: np.ndarray | None = None,
    label_info_selected: dict[int, dict[str, Any]] | None = None,
    hit_timestamps_hint: np.ndarray | None = None,
    labels_global_full: np.ndarray | None = None,
    min_shower_energy_mev: float = 80.0,
    min_clean_t0_separation_ticks: int = 8,
    clean_worsen_tolerance_norm: float = 0.08,
) -> tuple[np.ndarray, dict[int, int], dict[int, list[int]]]:
    base_image = np.zeros_like(full_light_waveform, dtype=np.float32)
    track_t0_by_label: dict[int, int] = {}
    fit_channels_by_label: dict[int, list[int]] = {}

    for clusterid in track_labels:
        tpcs = np.asarray(sorted(int(v) for v in cluster_to_tpcs.get(int(clusterid), [])), dtype=np.int32)
        if tpcs.size == 0:
            continue
        tpc_image_full = np.concatenate([np.asarray(image_maps[(int(clusterid), int(tpcid))], dtype=np.float32) for tpcid in tpcs.tolist()], axis=0)
        tpc_image_fit, tpc_base_fit, tpc_actual_fit, tpc_std_fit, fit_support_meta = stack_cluster_fit_inputs_v10_3(
            clusterid=int(clusterid),
            tpcids=tpcs,
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_support_cache=channel_support_cache,
            use_support_mask=True,
            saturated_channel_cache=saturated_channel_cache,
        )

        hinted_t0 = None
        if hit_timestamps_hint is not None and labels_global_full is not None:
            hint_mask = (np.asarray(labels_global_full, dtype=np.int32) == int(clusterid)) & np.isfinite(np.asarray(hit_timestamps_hint, dtype=np.float64))
            if np.any(hint_mask):
                hinted_t0 = int(np.clip(np.rint(np.median(np.asarray(hit_timestamps_hint, dtype=np.float64)[hint_mask])), 0, 800))

        decision = None
        if (
            labels_global_selected is not None
            and hit_tpc_ids_selected is not None
            and energies_selected is not None
            and label_info_selected is not None
        ):
            decision = _fit_track_t0_with_shower_guard_v13_2(
                clusterid=int(clusterid),
                sorted_tpcs=tpcs,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                labels_global=np.asarray(labels_global_selected, dtype=np.int32),
                hit_tpc_ids=np.asarray(hit_tpc_ids_selected, dtype=np.int32),
                energies=np.asarray(energies_selected, dtype=np.float64),
                label_info=label_info_selected,
                channel_support_cache=channel_support_cache,
                search_range=800,
                adc_clip=ADC_CLIP,
                t0_resolution=T0_RESOLUTION,
                waveform_len=WVFM_LEN,
                min_shower_energy_mev=float(min_shower_energy_mev),
                min_clean_t0_separation_ticks=int(min_clean_t0_separation_ticks),
                clean_worsen_tolerance_norm=float(clean_worsen_tolerance_norm),
                saturated_channel_cache=saturated_channel_cache,
            )
        if hinted_t0 is not None:
            newt0 = int(hinted_t0)
            if (
                decision is not None
                and str(decision.get("mode", "")) == "shower_guard_clean_tpcs"
                and abs(int(hinted_t0) - int(decision["selected_t0"])) >= int(min_clean_t0_separation_ticks)
            ):
                newt0 = int(decision["selected_t0"])
                fit_support_meta = [dict(item) for item in decision["selected_fit_support_meta"]]
        elif decision is not None:
            newt0 = int(decision["selected_t0"])
            fit_support_meta = [dict(item) for item in decision["selected_fit_support_meta"]]
        else:
            shifts, errors = max_likelihood_curve_with_base(
                predicted=tpc_image_fit,
                base=tpc_base_fit,
                actual=tpc_actual_fit,
                error_metric=tpc_std_fit,
                search_range=800,
            )
            raw_t0 = int(shifts[int(np.argmin(errors))])
            newt0 = int(raw_t0)
            expected_peak = 105 + int(newt0)
            signal_1d = np.sum(np.clip(tpc_actual_fit - tpc_base_fit, 0.0, None), axis=0)
            s_start = max(0, expected_peak - T0_RESOLUTION)
            s_end = min(WVFM_LEN, expected_peak + T0_RESOLUTION + 1)
            if s_end > s_start:
                actual_peak = int(s_start + np.argmax(signal_1d[s_start:s_end]))
                newt0 += int(actual_peak - expected_peak)
        track_t0_by_label[int(clusterid)] = int(newt0)
        fit_channels_by_label[int(clusterid)] = [int(item["n_fit_channels"]) for item in fit_support_meta]
        shifted = timeinterpolation(tpc_image_full, shift=float(newt0), baseline=0.0).astype(np.float32)
        offset = 0
        for tpcid in tpcs.tolist():
            block = np.asarray(image_maps[(int(clusterid), int(tpcid))], dtype=np.float32)
            nchan = int(block.shape[0])
            base_image[int(tpcid)] += shifted[offset:offset + nchan]
            offset += nchan
        base_image = np.clip(base_image, None, ADC_CLIP)

    return np.asarray(base_image, dtype=np.float32), track_t0_by_label, fit_channels_by_label


def _residual_profile_after_tracks(
    *,
    target_tpc: int,
    full_light_waveform: np.ndarray,
    track_base_image: np.ndarray,
    saturated_channel_cache: dict[str, Any] | None,
    apply_saturation_veto: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if saturated_channel_cache is None or not bool(apply_saturation_veto):
        keep_idx = np.arange(full_light_waveform.shape[1], dtype=np.int32)
    else:
        keep_idx = np.asarray(saturated_channel_cache["allowed_channel_indices_by_tpc"][int(target_tpc)], dtype=np.int32)
    actual = np.asarray(full_light_waveform[int(target_tpc), keep_idx], dtype=np.float32)
    track_base = np.asarray(track_base_image[int(target_tpc), keep_idx], dtype=np.float32)
    residual = np.sum(np.clip(actual - track_base, 0.0, None), axis=0)
    return keep_idx, np.asarray(residual, dtype=np.float64)


def _candidate_t0s_from_residual_profile(
    residual_profile: np.ndarray,
    *,
    pulse_peak_tick: int = 105,
    smooth_width: int = 11,
    min_peak_fraction: float = 0.06,
    min_sep_ticks: int = 10,
    refine_radius_ticks: int = 5,
) -> list[int]:
    smoothed = _smooth_1d(residual_profile, width=int(smooth_width))
    coarse_peaks = _select_separated_peaks(
        smoothed,
        min_peak_fraction=float(min_peak_fraction),
        min_sep=int(min_sep_ticks),
    )
    peaks = _refine_peak_indices_on_raw(
        residual_profile,
        coarse_peaks,
        refine_radius=int(refine_radius_ticks),
        min_sep=int(min_sep_ticks),
    )
    if len(peaks) == 0:
        peaks = [int(v) for v in coarse_peaks]
    t0s = []
    for peak in peaks:
        t0 = int(np.clip(int(peak) - int(pulse_peak_tick), 0, 800))
        if all(abs(int(t0) - int(prev)) > int(min_sep_ticks) for prev in t0s):
            t0s.append(int(t0))
    return sorted(int(v) for v in t0s)


def _scan_fragment_losses_on_candidates(
    *,
    fragment_label: int,
    target_tpc: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    track_base_image: np.ndarray,
    candidate_t0s: list[int],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    saturated_channel_cache: dict[str, Any] | None,
) -> tuple[dict[int, float], int, float]:
    fit_block = get_cluster_tpc_fit_block_v10_3(
        clusterid=int(fragment_label),
        tpcid=int(target_tpc),
        image_maps=image_maps,
        base_image=track_base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        channel_support_cache=channel_support_cache,
        use_support_mask=True,
        saturated_channel_cache=saturated_channel_cache,
    )
    if int(fit_block["channel_indices"].size) == 0:
        raise ValueError(
            f"Fragment label {int(fragment_label)} on TPC {int(target_tpc)} has no usable fit channels."
        )
    pred = np.asarray(fit_block["selected_wave"], dtype=np.float32)
    actual = np.asarray(fit_block["actual_block"], dtype=np.float32)
    std = np.asarray(fit_block["std_block"], dtype=np.float32)
    base = np.asarray(fit_block["base_block"], dtype=np.float32)
    scores: dict[int, float] = {}
    best_t0 = None
    best_score = None
    for t0 in candidate_t0s:
        shifted = timeinterpolation(pred, shift=float(t0), baseline=0.0).astype(np.float32)
        model = np.clip(base + shifted, None, ADC_CLIP)
        score = float(np.mean((model - actual) ** 2 / np.maximum(std, 1e-6)))
        scores[int(t0)] = float(score)
        if best_score is None or score < best_score:
            best_score = float(score)
            best_t0 = int(t0)
    if best_t0 is None:
        best_t0 = 0
        best_score = np.inf
    return scores, int(best_t0), float(best_score)


def _seed_families(
    *,
    fragments: list[MicroFragment],
    fragment_label_by_id: dict[int, int],
    fragment_best_t0: dict[int, int],
    seed_min_energy_mev: float,
    family_time_merge_ticks: int,
) -> list[dict[str, Any]]:
    seed_frags = [
        frag
        for frag in fragments
        if float(frag.energy_mev) >= float(seed_min_energy_mev)
    ]
    ordered = sorted(seed_frags, key=lambda frag: (-float(frag.energy_mev), int(frag.fragment_id)))
    families: list[dict[str, Any]] = []
    for frag in ordered:
        t0 = int(fragment_best_t0[int(frag.fragment_id)])
        placed = False
        for family in families:
            if abs(int(family["t0"]) - int(t0)) <= int(family_time_merge_ticks):
                family["fragment_ids"].append(int(frag.fragment_id))
                placed = True
                break
        if not placed:
            families.append(
                {
                    "seed_fragment_id": int(frag.fragment_id),
                    "fragment_ids": [int(frag.fragment_id)],
                    "t0": int(t0),
                    "apex": np.asarray(frag.start_point, dtype=np.float64),
                }
            )
    if not families and fragments:
        frag = max(fragments, key=lambda item: float(item.energy_mev))
        families.append(
            {
                "seed_fragment_id": int(frag.fragment_id),
                "fragment_ids": [int(frag.fragment_id)],
                "t0": int(fragment_best_t0[int(frag.fragment_id)]),
                "apex": np.asarray(frag.start_point, dtype=np.float64),
            }
        )
    return families


def _seed_families_from_residual_candidates(
    *,
    fragments: list[MicroFragment],
    candidate_t0s: list[int],
    fragment_loss_scores: dict[int, dict[int, float]],
    seed_min_energy_mev: float,
    seed_max_time_delta: float = 0.40,
) -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    used_seed_fragments: set[int] = set()

    for candidate_t0 in candidate_t0s:
        best_fragment = None
        best_score = None
        for fragment in fragments:
            if int(fragment.fragment_id) in used_seed_fragments:
                continue
            if float(fragment.energy_mev) < float(seed_min_energy_mev):
                continue
            score_map = fragment_loss_scores.get(int(fragment.fragment_id), {})
            if int(candidate_t0) not in score_map:
                continue
            losses = np.asarray([float(v) for v in score_map.values()], dtype=np.float64)
            best_loss = float(np.min(losses))
            worst_loss = float(np.max(losses))
            denom = max(worst_loss - best_loss, 1e-6)
            time_delta = (float(score_map[int(candidate_t0)]) - best_loss) / denom
            candidate_score = float(time_delta) - 0.03 * float(fragment.energy_mev)
            if best_score is None or candidate_score < best_score:
                best_score = float(candidate_score)
                best_fragment = (fragment, float(time_delta))

        if best_fragment is None:
            continue

        fragment, time_delta = best_fragment
        if float(time_delta) > float(seed_max_time_delta):
            continue

        families.append(
            {
                "seed_fragment_id": int(fragment.fragment_id),
                "fragment_ids": [int(fragment.fragment_id)],
                "t0": int(candidate_t0),
                "apex": np.asarray(fragment.start_point, dtype=np.float64),
            }
        )
        used_seed_fragments.add(int(fragment.fragment_id))

    return families


def _fragment_family_score(
    *,
    fragment: MicroFragment,
    family_t0: int,
    family_apex: np.ndarray,
    loss_scores: dict[int, float],
    w_time: float,
    w_miss: float,
    w_angle: float,
) -> float:
    losses = np.asarray([float(v) for v in loss_scores.values()], dtype=np.float64)
    best_loss = float(np.min(losses))
    worst_loss = float(np.max(losses))
    denom = max(worst_loss - best_loss, 1e-6)
    time_loss = float(loss_scores[int(family_t0)])
    time_delta = (time_loss - best_loss) / denom

    apex = np.asarray(family_apex, dtype=np.float64)
    miss = _line_point_distance(apex, fragment.start_point, fragment.direction)
    ray = np.asarray(fragment.centroid, dtype=np.float64) - apex
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm <= 1e-8:
        angle_pen = 0.0
    else:
        angle_pen = 1.0 - abs(float(np.dot(_normalize_direction(fragment.direction), ray / ray_norm)))

    space_weight = 0.35 + 0.65 * float(np.clip(fragment.linearity, 0.0, 1.0))
    score = (
        float(w_time) * time_delta
        + space_weight * float(w_miss) * min(miss / 18.0, 5.0)
        + space_weight * float(w_angle) * min(angle_pen / 0.25, 5.0)
    )
    return float(score)


def _refine_family_t0s(
    *,
    families: list[dict[str, Any]],
    fragments: list[MicroFragment],
    fragment_label_by_id: dict[int, int],
    image_maps: dict[tuple[int, int], np.ndarray],
    target_tpc: int,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    track_base_image: np.ndarray,
    keep_idx: np.ndarray,
    search_half_window: int = 8,
    n_iters: int = 2,
) -> None:
    frag_map = {int(f.fragment_id): f for f in fragments}
    actual = np.asarray(full_light_waveform[int(target_tpc), keep_idx], dtype=np.float32)
    std = np.asarray(full_light_std[int(target_tpc), keep_idx], dtype=np.float32)

    family_blocks: dict[int, np.ndarray] = {}
    for family_idx, family in enumerate(families):
        total = np.zeros((keep_idx.size, WVFM_LEN), dtype=np.float32)
        for fragment_id in family["fragment_ids"]:
            label = int(fragment_label_by_id[int(fragment_id)])
            total += np.asarray(image_maps[(int(label), int(target_tpc))], dtype=np.float32)[keep_idx]
        family_blocks[int(family_idx)] = total

    for _ in range(int(n_iters)):
        for family_idx, family in enumerate(families):
            other_base = np.asarray(track_base_image[int(target_tpc), keep_idx], dtype=np.float32).copy()
            for other_idx, other_family in enumerate(families):
                if int(other_idx) == int(family_idx):
                    continue
                other_base += timeinterpolation(
                    family_blocks[int(other_idx)],
                    shift=float(other_family["t0"]),
                    baseline=0.0,
                ).astype(np.float32)
            current_t0 = int(family["t0"])
            grid = [
                int(np.clip(current_t0 + delta, 0, 800))
                for delta in range(-int(search_half_window), int(search_half_window) + 1)
            ]
            grid = sorted(set(grid))
            best_t0 = current_t0
            best_score = None
            for t0 in grid:
                model = np.clip(
                    other_base + timeinterpolation(family_blocks[int(family_idx)], shift=float(t0), baseline=0.0).astype(np.float32),
                    None,
                    ADC_CLIP,
                )
                score = float(np.mean((model - actual) ** 2 / np.maximum(std, 1e-6)))
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_t0 = int(t0)
            family["t0"] = int(best_t0)

        for family in families:
            family["apex"] = _estimate_apex(fragments, family["fragment_ids"])


def _score_shifted_block(
    *,
    block: np.ndarray,
    t0: int,
    base: np.ndarray,
    actual: np.ndarray,
    std: np.ndarray,
) -> float:
    model = np.clip(
        np.asarray(base, dtype=np.float32)
        + timeinterpolation(np.asarray(block, dtype=np.float32), shift=float(t0), baseline=0.0).astype(np.float32),
        None,
        ADC_CLIP,
    )
    return float(np.mean((model - actual) ** 2 / np.maximum(std, 1e-6)))


def _peel_secondary_families_from_dominant(
    *,
    families: list[dict[str, Any]],
    fragments: list[MicroFragment],
    fragment_label_by_id: dict[int, int],
    image_maps: dict[tuple[int, int], np.ndarray],
    target_tpc: int,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    track_base_image: np.ndarray,
    keep_idx: np.ndarray,
    family_time_merge_ticks: int,
) -> list[dict[str, Any]]:
    if len(families) == 0:
        return families

    families = [dict(family) for family in families]
    dominant_idx = int(np.argmax([len(family["fragment_ids"]) for family in families]))
    dominant = families[dominant_idx]
    dominant_t0 = int(dominant["t0"])
    dominant_apex = np.asarray(dominant["apex"], dtype=np.float64)

    family_blocks: dict[int, np.ndarray] = {}
    for family_idx, family in enumerate(families):
        block = np.zeros((keep_idx.size, WVFM_LEN), dtype=np.float32)
        for fragment_id in family["fragment_ids"]:
            label = int(fragment_label_by_id[int(fragment_id)])
            block += np.asarray(image_maps[(int(label), int(target_tpc))], dtype=np.float32)[keep_idx]
        family_blocks[int(family_idx)] = block

    actual = np.asarray(full_light_waveform[int(target_tpc), keep_idx], dtype=np.float32)
    std = np.asarray(full_light_std[int(target_tpc), keep_idx], dtype=np.float32)
    dominant_model = np.clip(
        np.asarray(track_base_image[int(target_tpc), keep_idx], dtype=np.float32)
        + timeinterpolation(family_blocks[int(dominant_idx)], shift=float(dominant_t0), baseline=0.0).astype(np.float32),
        None,
        ADC_CLIP,
    )
    residual_after_dominant = np.sum(np.clip(actual - dominant_model, 0.0, None), axis=0)
    secondary_t0s = [
        int(v)
        for v in _candidate_t0s_from_residual_profile(
            residual_after_dominant,
            pulse_peak_tick=105,
            smooth_width=11,
            min_peak_fraction=0.08,
            min_sep_ticks=int(family_time_merge_ticks),
        )
        if abs(int(v) - int(dominant_t0)) > int(family_time_merge_ticks)
    ]
    if len(secondary_t0s) == 0:
        return families

    dominant_frag_ids = [int(v) for v in dominant["fragment_ids"]]
    moved_by_t0: dict[int, list[int]] = {}
    dominant_block = np.asarray(family_blocks[int(dominant_idx)], dtype=np.float32)
    for fragment_id in dominant_frag_ids:
        frag_obj = next(item for item in fragments if int(item.fragment_id) == int(fragment_id))
        dominant_miss = _line_point_distance(dominant_apex, frag_obj.start_point, frag_obj.direction)
        dominant_ray = np.asarray(frag_obj.centroid, dtype=np.float64) - dominant_apex
        dominant_ray_norm = float(np.linalg.norm(dominant_ray))
        if dominant_ray_norm <= 1e-8:
            dominant_angle_pen = 0.0
        else:
            dominant_angle_pen = 1.0 - abs(float(np.dot(_normalize_direction(frag_obj.direction), dominant_ray / dominant_ray_norm)))
        if float(dominant_miss) < 12.0 and float(dominant_angle_pen) < 0.30:
            continue

        label = int(fragment_label_by_id[int(fragment_id)])
        frag_block = np.asarray(image_maps[(int(label), int(target_tpc))], dtype=np.float32)[keep_idx]
        base_without_fragment = np.clip(
            np.asarray(track_base_image[int(target_tpc), keep_idx], dtype=np.float32)
            + timeinterpolation(dominant_block - frag_block, shift=float(dominant_t0), baseline=0.0).astype(np.float32),
            None,
            ADC_CLIP,
        )
        stay_score = _score_shifted_block(
            block=frag_block,
            t0=int(dominant_t0),
            base=base_without_fragment,
            actual=actual,
            std=std,
        )
        best_secondary_t0 = None
        best_secondary_score = None
        for cand_t0 in secondary_t0s:
            cand_score = _score_shifted_block(
                block=frag_block,
                t0=int(cand_t0),
                base=base_without_fragment,
                actual=actual,
                std=std,
            )
            if best_secondary_score is None or cand_score < best_secondary_score:
                best_secondary_score = float(cand_score)
                best_secondary_t0 = int(cand_t0)
        if best_secondary_t0 is None:
            continue
        if float(best_secondary_score) + 1e-4 < float(stay_score):
            moved_by_t0.setdefault(int(best_secondary_t0), []).append(int(fragment_id))

    new_families: list[dict[str, Any]] = []
    moved_fragment_ids = sorted({int(fid) for ids in moved_by_t0.values() for fid in ids})
    for cand_t0, fragment_ids in sorted(moved_by_t0.items()):
        if len(fragment_ids) == 0:
            continue
        new_families.append(
            {
                "seed_fragment_id": int(fragment_ids[0]),
                "fragment_ids": [int(v) for v in fragment_ids],
                "t0": int(cand_t0),
                "apex": _estimate_apex(fragments, [int(v) for v in fragment_ids]),
            }
        )

    if len(new_families) == 0:
        return families

    families[dominant_idx]["fragment_ids"] = [
        int(fid) for fid in families[dominant_idx]["fragment_ids"] if int(fid) not in set(moved_fragment_ids)
    ]
    families[dominant_idx]["apex"] = _estimate_apex(fragments, families[dominant_idx]["fragment_ids"])
    families.extend(new_families)
    return [family for family in families if len(family["fragment_ids"]) > 0]


def run_local_tpc_shower_specialist(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    truth_vertex_id: np.ndarray,
    truth_t0: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    model: Any,
    template: np.ndarray,
    target_tpc: int,
    dbscan_eps_cm: float = 6.0,
    dbscan_min_samples: int = 4,
    seed_min_energy_mev: float = 3.0,
    family_time_merge_ticks: int = 10,
    peak_smooth_width: int = 11,
    peak_min_fraction: float = 0.06,
    w_time: float = 1.0,
    w_miss: float = 1.0,
    w_angle: float = 1.0,
    saturated_channel_cache: dict[str, Any] | None = None,
    provisional_apex_override: np.ndarray | None = None,
    hit_timestamps_hint: np.ndarray | None = None,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    labels_global = np.asarray(labels_global, dtype=np.int32)
    truth_vertex_id = np.asarray(truth_vertex_id, dtype=np.int64)
    truth_t0 = np.asarray(truth_t0, dtype=np.float64)

    provisional_apex_debug: dict[str, Any] | None = None
    provisional_apex = None if provisional_apex_override is None else np.asarray(provisional_apex_override, dtype=np.float64)
    if provisional_apex is None:
        provisional_apex, provisional_apex_debug = _estimate_neighbor_aware_shower_apex(
            x=x,
            y=y,
            z=z,
            energies=energies,
            hit_tpc_ids=hit_tpc_ids,
            labels_global=labels_global,
            label_info=label_info,
            target_tpc=int(target_tpc),
        )
    if provisional_apex is None:
        provisional_apex = _estimate_provisional_shower_apex(
            x=x,
            y=y,
            z=z,
            energies=energies,
            hit_tpc_ids=hit_tpc_ids,
            labels_global=labels_global,
            label_info=label_info,
            target_tpc=int(target_tpc),
        )

    fragments, fragment_ids_local = split_nontrack_hits_into_microfragments(
        x=x,
        y=y,
        z=z,
        energies=energies,
        hit_tpc_ids=hit_tpc_ids,
        labels_global=labels_global,
        label_info=label_info,
        truth_vertex_id=truth_vertex_id,
        truth_t0=truth_t0,
        target_tpc=int(target_tpc),
        dbscan_eps_cm=float(dbscan_eps_cm),
        dbscan_min_samples=int(dbscan_min_samples),
        provisional_apex=None if provisional_apex is None else np.asarray(provisional_apex, dtype=np.float64),
        fine_dbscan_eps_cm=1.5,
        fine_dbscan_min_samples=2,
        large_label_energy_mev=80.0,
        miss_split_energy_mev=80.0,
    )
    if len(fragments) == 0:
        raise RuntimeError(f"No non-track fragments found on TPC {int(target_tpc)}.")

    (
        x_sel,
        y_sel,
        z_sel,
        e_sel,
        tpc_sel,
        labels_sel,
        label_info_sel,
        microfrag_offset,
        track_labels,
        microfrag_labels,
    ) = _build_selected_prediction_inputs(
        labels_global=labels_global,
        label_info=label_info,
        hit_tpc_ids=hit_tpc_ids,
        x=x,
        y=y,
        z=z,
        energies=energies,
        target_tpc=int(target_tpc),
        fragments=fragments,
    )

    image_maps, _ = process_clusters_to_imageMaps(
        x_sel,
        y_sel,
        z_sel,
        e_sel,
        tpc_sel,
        labels_sel,
        model=model,
        template=template,
    )
    cluster_to_tpcs: dict[int, list[int]] = {}
    for (clusterid, tpcid) in image_maps.keys():
        cluster_to_tpcs.setdefault(int(clusterid), []).append(int(tpcid))

    split_index = int(microfrag_offset)
    channel_support_cache, _ = build_cluster_channel_support_cache_v10_3(
        image_maps,
        cluster_to_tpcs,
        label_info_sel,
        split_index=int(split_index),
        light_fraction=0.90,
        max_gap=2,
    )

    track_base_image, track_t0_by_label, fit_channels_by_label = _fit_track_base_on_selected_labels(
        track_labels=track_labels,
        image_maps=image_maps,
        cluster_to_tpcs=cluster_to_tpcs,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        channel_support_cache=channel_support_cache,
        saturated_channel_cache=saturated_channel_cache,
        labels_global_selected=np.asarray(labels_sel, dtype=np.int32),
        hit_tpc_ids_selected=np.asarray(tpc_sel, dtype=np.int32),
        energies_selected=np.asarray(e_sel, dtype=np.float64),
        label_info_selected=label_info_sel,
        hit_timestamps_hint=None if hit_timestamps_hint is None else np.asarray(hit_timestamps_hint, dtype=np.float64),
        labels_global_full=np.asarray(labels_global, dtype=np.int32),
    )

    keep_idx, residual_profile = _residual_profile_after_tracks(
        target_tpc=int(target_tpc),
        full_light_waveform=full_light_waveform,
        track_base_image=track_base_image,
        saturated_channel_cache=saturated_channel_cache,
        apply_saturation_veto=True,
    )
    _, residual_profile_full = _residual_profile_after_tracks(
        target_tpc=int(target_tpc),
        full_light_waveform=full_light_waveform,
        track_base_image=track_base_image,
        saturated_channel_cache=saturated_channel_cache,
        apply_saturation_veto=False,
    )
    candidate_t0s = _candidate_t0s_from_residual_profile(
        residual_profile,
        pulse_peak_tick=105,
        smooth_width=int(peak_smooth_width),
        min_peak_fraction=float(peak_min_fraction),
        min_sep_ticks=int(family_time_merge_ticks),
    )
    if len(candidate_t0s) == 0:
        candidate_t0s = [0]

    fragment_label_by_id = {int(f.fragment_id): int(microfrag_offset + int(f.fragment_id)) for f in fragments}
    fragment_loss_scores: dict[int, dict[int, float]] = {}
    fragment_best_t0: dict[int, int] = {}
    fragment_best_score: dict[int, float] = {}
    for fragment in fragments:
        scores, best_t0, best_score = _scan_fragment_losses_on_candidates(
            fragment_label=int(fragment_label_by_id[int(fragment.fragment_id)]),
            target_tpc=int(target_tpc),
            image_maps=image_maps,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            track_base_image=track_base_image,
            candidate_t0s=candidate_t0s,
            channel_support_cache=channel_support_cache,
            saturated_channel_cache=saturated_channel_cache,
        )
        fragment_loss_scores[int(fragment.fragment_id)] = dict(scores)
        fragment_best_t0[int(fragment.fragment_id)] = int(best_t0)
        fragment_best_score[int(fragment.fragment_id)] = float(best_score)

    families = _seed_families_from_residual_candidates(
        fragments=fragments,
        candidate_t0s=candidate_t0s,
        fragment_loss_scores=fragment_loss_scores,
        seed_min_energy_mev=float(seed_min_energy_mev),
        seed_max_time_delta=0.45,
    )
    if len(families) == 0:
        families = _seed_families(
        fragments=fragments,
        fragment_label_by_id=fragment_label_by_id,
        fragment_best_t0=fragment_best_t0,
        seed_min_energy_mev=float(seed_min_energy_mev),
        family_time_merge_ticks=int(family_time_merge_ticks),
        )
    for family in families:
        family["apex"] = _estimate_apex(fragments, family["fragment_ids"])

    family_assignments: dict[int, int] = {}
    for _ in range(2):
        family_assignments = {}
        for family_idx, family in enumerate(families):
            seed_fragment_id = int(family["seed_fragment_id"])
            family["fragment_ids"] = [int(seed_fragment_id)]
            family_assignments[int(seed_fragment_id)] = int(family_idx)

        ordered_fragments = sorted(fragments, key=lambda frag: (-float(frag.energy_mev), int(frag.fragment_id)))
        for fragment in ordered_fragments:
            if int(fragment.fragment_id) in family_assignments:
                continue
            best_family_idx = None
            best_score = None
            for family_idx, family in enumerate(families):
                family_t0 = int(family["t0"])
                if int(family_t0) not in fragment_loss_scores[int(fragment.fragment_id)]:
                    continue
                score = _fragment_family_score(
                    fragment=fragment,
                    family_t0=int(family_t0),
                    family_apex=np.asarray(family["apex"], dtype=np.float64),
                    loss_scores=fragment_loss_scores[int(fragment.fragment_id)],
                    w_time=float(w_time),
                    w_miss=float(w_miss),
                    w_angle=float(w_angle),
                )
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_family_idx = int(family_idx)
            if best_family_idx is None:
                continue
            family_assignments[int(fragment.fragment_id)] = int(best_family_idx)
            families[int(best_family_idx)]["fragment_ids"].append(int(fragment.fragment_id))

        families = [family for family in families if len(family["fragment_ids"]) > 0]
        for family in families:
            family["apex"] = _estimate_apex(fragments, family["fragment_ids"])

    _refine_family_t0s(
        families=families,
        fragments=fragments,
        fragment_label_by_id=fragment_label_by_id,
        image_maps=image_maps,
        target_tpc=int(target_tpc),
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        track_base_image=track_base_image,
        keep_idx=keep_idx,
    )
    families = _peel_secondary_families_from_dominant(
        families=families,
        fragments=fragments,
        fragment_label_by_id=fragment_label_by_id,
        image_maps=image_maps,
        target_tpc=int(target_tpc),
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        track_base_image=track_base_image,
        keep_idx=keep_idx,
        family_time_merge_ticks=int(family_time_merge_ticks),
    )
    _refine_family_t0s(
        families=families,
        fragments=fragments,
        fragment_label_by_id=fragment_label_by_id,
        image_maps=image_maps,
        target_tpc=int(target_tpc),
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        track_base_image=track_base_image,
        keep_idx=keep_idx,
    )

    hit_family = np.full(labels_global.shape[0], -1, dtype=np.int32)
    for family_idx, family in enumerate(families):
        for fragment_id in family["fragment_ids"]:
            frag = next(item for item in fragments if int(item.fragment_id) == int(fragment_id))
            hit_family[frag.hit_indices_local] = int(family_idx)

    family_rows: list[dict[str, Any]] = []
    local_nontrack_mask = (hit_tpc_ids == int(target_tpc)) & (fragment_ids_local >= 0)
    unique_truth_vertices = sorted(int(v) for v in np.unique(truth_vertex_id[local_nontrack_mask]) if int(v) >= 0)
    truth_rows: list[dict[str, Any]] = []

    for family_idx, family in enumerate(families):
        fam_mask = hit_family == int(family_idx)
        if not np.any(fam_mask):
            continue
        local_vertex = truth_vertex_id[fam_mask]
        local_t0 = truth_t0[fam_mask]
        local_energy = energies[fam_mask]
        valid = (local_vertex >= 0) & np.isfinite(local_t0) & (local_energy > 0.0)
        dominant_vertex = -1
        dominant_t0 = np.nan
        purity = np.nan
        if np.any(valid):
            uniq = np.unique(local_vertex[valid])
            energy_by_vertex = np.asarray([np.sum(local_energy[valid][local_vertex[valid] == int(vid)]) for vid in uniq], dtype=np.float64)
            dominant_vertex = int(uniq[int(np.argmax(energy_by_vertex))])
            dom_mask = valid & (local_vertex == int(dominant_vertex))
            dominant_t0 = float(np.median(local_t0[dom_mask]))
            purity = float(np.sum(local_energy[dom_mask]) / np.sum(local_energy[valid]))
        family_rows.append(
            {
                "family_id": int(family_idx),
                "t0": int(family["t0"]),
                "n_fragments": int(len(family["fragment_ids"])),
                "n_hits": int(np.count_nonzero(fam_mask)),
                "energy_mev": float(np.sum(energies[fam_mask])),
                "dominant_truth_vertex": int(dominant_vertex),
                "dominant_truth_t0": None if not np.isfinite(dominant_t0) else float(dominant_t0),
                "purity": None if not np.isfinite(purity) else float(purity),
                "fragment_ids": [int(v) for v in family["fragment_ids"]],
                "apex_xyz": [float(v) for v in np.asarray(family["apex"], dtype=np.float64).tolist()],
            }
        )

    for truth_vertex in unique_truth_vertices:
        truth_mask = local_nontrack_mask & (truth_vertex_id == int(truth_vertex))
        truth_energy = float(np.sum(energies[truth_mask]))
        truth_t0_med = float(np.median(truth_t0[truth_mask])) if np.any(np.isfinite(truth_t0[truth_mask])) else np.nan
        best_completeness = 0.0
        best_family = None
        for family_idx in range(len(families)):
            fam_mask = hit_family == int(family_idx)
            overlap_energy = float(np.sum(energies[fam_mask & truth_mask]))
            completeness = overlap_energy / max(truth_energy, 1e-12)
            if completeness > best_completeness:
                best_completeness = completeness
                best_family = int(family_idx)
        truth_rows.append(
            {
                "truth_vertex_id": int(truth_vertex),
                "truth_t0": None if not np.isfinite(truth_t0_med) else float(truth_t0_med),
                "energy_mev": float(truth_energy),
                "best_family_id": None if best_family is None else int(best_family),
                "best_family_completeness": float(best_completeness),
            }
        )

    return {
        "target_tpc": int(target_tpc),
        "n_fragments": int(len(fragments)),
        "n_families": int(len(families)),
        "candidate_t0s": [int(v) for v in candidate_t0s],
        "track_labels": [int(v) for v in track_labels],
        "track_t0_by_label": {int(k): int(v) for k, v in track_t0_by_label.items()},
        "fragment_ids_local": np.asarray(fragment_ids_local, dtype=np.int32),
        "hit_family": np.asarray(hit_family, dtype=np.int32),
        "fragments": fragments,
        "families": families,
        "family_rows": family_rows,
        "truth_rows": truth_rows,
        "residual_profile": np.asarray(residual_profile, dtype=np.float64),
        "residual_profile_full": np.asarray(residual_profile_full, dtype=np.float64),
        "keep_channel_indices": np.asarray(keep_idx, dtype=np.int32),
        "track_base_image": np.asarray(track_base_image, dtype=np.float32),
        "fragment_label_by_id": {int(k): int(v) for k, v in fragment_label_by_id.items()},
        "fragment_loss_scores": {
            int(fid): {int(k): float(v) for k, v in score_map.items()}
            for fid, score_map in fragment_loss_scores.items()
        },
        "fragment_best_t0": {int(k): int(v) for k, v in fragment_best_t0.items()},
        "fragment_best_score": {int(k): float(v) for k, v in fragment_best_score.items()},
        "provisional_apex_xyz": None if provisional_apex is None else [float(v) for v in np.asarray(provisional_apex, dtype=np.float64).tolist()],
        "provisional_apex_debug": provisional_apex_debug,
        "fragment_best_t0_counts": {
            int(t0): int(np.count_nonzero(np.asarray(list(fragment_best_t0.values()), dtype=np.int32) == int(t0)))
            for t0 in sorted(set(int(v) for v in fragment_best_t0.values()))
        },
        "image_maps": image_maps,
        "cluster_to_tpcs": cluster_to_tpcs,
        "fit_channels_by_label": {int(k): [int(vv) for vv in vals] for k, vals in fit_channels_by_label.items()},
    }


def plot_tpc_family_assignment_html(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    hit_family: np.ndarray,
    target_tpc: int,
    title: str,
    save_path: Path,
) -> None:
    mask_tpc = np.asarray(hit_tpc_ids == int(target_tpc), dtype=bool)
    nontrack_mask = mask_tpc & np.isin(
        labels_global,
        np.asarray(
            [
                int(label)
                for label in np.unique(labels_global[mask_tpc])
                if int(label) >= 0 and str(label_info.get(int(label), {}).get("type", "cluster")).lower() != "track"
            ],
            dtype=np.int32,
        ),
    )
    track_mask = mask_tpc & ~nontrack_mask

    colors = [
        "#2E7D32",
        "#1565C0",
        "#EF6C00",
        "#8E24AA",
        "#00897B",
        "#C62828",
        "#6D4C41",
        "#3949AB",
    ]
    fig = go.Figure()
    if np.any(track_mask):
        fig.add_trace(
            go.Scatter3d(
                x=np.asarray(z[track_mask], dtype=np.float64),
                y=np.asarray(y[track_mask], dtype=np.float64),
                z=np.asarray(x[track_mask], dtype=np.float64),
                mode="markers",
                marker=dict(size=2.0, color="lightgray", opacity=0.30),
                name="Fixed tracks",
            )
        )
    for family_id in sorted(int(v) for v in np.unique(hit_family[nontrack_mask]) if int(v) >= 0):
        mask = nontrack_mask & (hit_family == int(family_id))
        fig.add_trace(
            go.Scatter3d(
                x=np.asarray(z[mask], dtype=np.float64),
                y=np.asarray(y[mask], dtype=np.float64),
                z=np.asarray(x[mask], dtype=np.float64),
                mode="markers",
                marker=dict(size=3.0, color=colors[int(family_id) % len(colors)], opacity=0.85),
                name=f"Family {int(family_id)}",
            )
        )
    fig.update_layout(
        title=str(title),
        scene=dict(xaxis_title="z", yaxis_title="y", zaxis_title="x"),
        margin=dict(l=0, r=0, b=0, t=36),
        showlegend=True,
    )
    fig.write_html(str(save_path))


def _run_case(
    *,
    eventid: int,
    tpcid: int,
    data_file: str,
    checkpoint_path: str,
    pulse_path: str,
    outdir: Path,
    dbscan_eps_cm: float,
    dbscan_min_samples: int,
) -> dict[str, Any]:
    resources = load_v11_plotting_resources(
        data_file=str(data_file),
        checkpoint_path=str(checkpoint_path),
        pulse_path=str(pulse_path),
    )
    try:
        event = _load_event_inputs(resources, int(eventid))
        truth = extract_truth_vertex_and_t0_for_hits(
            resources.h5,
            event["hit_refs"],
            convert_to_matching_ticks=True,
        )

        labels_global, split_index, label_info, _ = build_global_labels_toolbox(
            event["xset"],
            event["yset"],
            event["zset"],
            event["io_group"],
            lam=1.2,
            rss_threshold=1.5e6,
            iters=800,
            min_inliers=35,
            k_for_scale=8,
            attach_multiplier=1.15,
            seed=0,
            min_length_cm=30.0,
            n_tpcs=70,
            match_dist_tol=5.0,
            match_angle_deg=12.0,
            match_endpoint_dist_tol=40.0,
            match_endpoint_weight=0.45,
            match_angle_weight=0.35,
            match_quality_weight=0.15,
            match_max_tpc_gap=None,
            vertex_eps=10.0,
            vertex_min_samples=3,
            min_tracks_for_shower=3,
            split_track_components=True,
            split_radius_cm=4.0,
            split_min_component_hits=20,
            promote_line_like_leftovers=True,
            rescue_dbscan_eps=4.0,
            rescue_dbscan_min_samples=3,
            rescue_min_hits=15,
            rescue_min_length_cm=20.0,
            rescue_min_linearity=0.88,
            rescue_max_transverse_rms=5.0,
            track_noise_absorption_enable=True,
            track_noise_absorb_radius_scale=1.5,
            track_noise_absorb_min_base_radius_cm=1.2,
            track_noise_absorb_endpoint_margin_cm=4.0,
            leftover_dbscan_eps=4.0,
            leftover_dbscan_min_samples=3,
            return_label_info=True,
            return_debug_info=True,
        )

        saturated_channel_cache, _ = build_saturated_channel_cache_v12(
            event["fullLightWaveform"],
            clip_threshold=60700.0,
            max_clip_ticks=6,
        )

        best_result = None
        best_score = None
        best_params = None
        eps_grid = sorted(set([3.0, 4.0, 5.0, 6.0, 7.0, float(dbscan_eps_cm)]))
        for eps in eps_grid:
            if float(eps) <= 0.0:
                continue
            for min_samples in sorted(set([max(int(dbscan_min_samples) - 1, 3), int(dbscan_min_samples), int(dbscan_min_samples) + 1])):
                result = run_local_tpc_shower_specialist(
                    x=event["xset"],
                    y=event["yset"],
                    z=event["zset"],
                    energies=event["Eset"],
                    hit_tpc_ids=event["hitTPCid"],
                    labels_global=labels_global,
                    label_info=label_info,
                    truth_vertex_id=np.asarray(truth["vertex_id"], dtype=np.int64),
                    truth_t0=np.asarray(truth["truth_t0"], dtype=np.float64),
                    full_light_waveform=event["fullLightWaveform"],
                    full_light_std=event["fullLightStd"],
                    model=resources.model,
                    template=resources.wvfm_tmpl,
                    target_tpc=int(tpcid),
                    dbscan_eps_cm=float(eps),
                    dbscan_min_samples=int(min_samples),
                    seed_min_energy_mev=3.0,
                    family_time_merge_ticks=10,
                    peak_smooth_width=11,
                    peak_min_fraction=0.06,
                    w_time=1.0,
                    w_miss=1.0,
                    w_angle=1.0,
                    saturated_channel_cache=saturated_channel_cache,
                )
                weighted_score = 0.0
                for row in result["truth_rows"]:
                    if float(row["energy_mev"]) < 5.0:
                        continue
                    weighted_score += float(row["energy_mev"]) * float(row["best_family_completeness"])
                for row in result["family_rows"]:
                    if float(row["energy_mev"]) < 5.0:
                        continue
                    if row["purity"] is not None:
                        weighted_score += float(row["energy_mev"]) * float(row["purity"])
                if best_score is None or weighted_score > best_score:
                    best_score = float(weighted_score)
                    best_result = result
                    best_params = {"dbscan_eps_cm": float(eps), "dbscan_min_samples": int(min_samples)}

        if best_result is None:
            raise RuntimeError("No specialist result was produced.")

        stem = f"event{int(eventid):03d}_tpc{int(tpcid):02d}_shower_specialist"
        outdir.mkdir(parents=True, exist_ok=True)
        html_path = outdir / f"{stem}.html"
        plot_tpc_family_assignment_html(
            x=event["xset"],
            y=event["yset"],
            z=event["zset"],
            hit_tpc_ids=event["hitTPCid"],
            labels_global=labels_global,
            label_info=label_info,
            hit_family=np.asarray(best_result["hit_family"], dtype=np.int32),
            target_tpc=int(tpcid),
            title=f"TPC local shower specialist | event {int(eventid)} | TPC {int(tpcid)}",
            save_path=html_path,
        )
        summary = {
            "eventid": int(eventid),
            "tpcid": int(tpcid),
            "best_params": dict(best_params),
            "candidate_t0s": [int(v) for v in best_result["candidate_t0s"]],
            "track_labels": [int(v) for v in best_result["track_labels"]],
            "track_t0_by_label": {int(k): int(v) for k, v in best_result["track_t0_by_label"].items()},
            "fragment_best_t0_counts": {
                int(k): int(v) for k, v in best_result["fragment_best_t0_counts"].items()
            },
            "family_rows": best_result["family_rows"],
            "truth_rows": best_result["truth_rows"],
            "n_fragments": int(best_result["n_fragments"]),
            "n_families": int(best_result["n_families"]),
            "html_path": str(html_path),
        }
        summary_path = outdir / f"{stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        try:
            resources.h5.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Local TPC shower specialist using joint space-plus-time assignment.")
    parser.add_argument("--eventid", type=int, default=1)
    parser.add_argument("--tpcid", type=int, default=8)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--pulse-path", default=DEFAULT_PULSE_PATH)
    parser.add_argument("--outdir", default=str(Path("M5p1") / "tpc_shower_specialist_examples"))
    parser.add_argument("--dbscan-eps-cm", type=float, default=6.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=4)
    args = parser.parse_args()

    summary = _run_case(
        eventid=int(args.eventid),
        tpcid=int(args.tpcid),
        data_file=str(args.data_file),
        checkpoint_path=str(args.checkpoint_path),
        pulse_path=str(args.pulse_path),
        outdir=Path(args.outdir),
        dbscan_eps_cm=float(args.dbscan_eps_cm),
        dbscan_min_samples=int(args.dbscan_min_samples),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
