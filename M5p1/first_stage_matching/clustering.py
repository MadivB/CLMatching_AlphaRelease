from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ClusteringConfig
from .paths import M5P1_DIR, configure_paths, import_from_path

configure_paths()


@dataclass(slots=True)
class ClusteringResult:
    labels_global: np.ndarray
    split_index: int
    label_info: dict[int, dict[str, Any]]
    debug: dict[str, Any]
    track_shower_labels: list[int]
    cluster_labels: list[int]
    n_noise: int
    n_labeled: int
    n_labels: int
    backbone_type_counts: dict[str, int]


def load_track_clustering_toolbox():
    return import_from_path(
        "ndqlmatching_v12_clustering_toolbox_runtime",
        M5P1_DIR / "global_track_clustering_toolbox_v11_2.py",
    )


def run_global_track_clustering(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    io_group: np.ndarray,
    config: ClusteringConfig | None = None,
) -> ClusteringResult:
    config = ClusteringConfig() if config is None else config
    toolbox = load_track_clustering_toolbox()
    labels_global, split_index, label_info, debug = toolbox.build_global_labels_toolbox(
        x,
        y,
        z,
        io_group,
        lam=config.lam,
        rss_threshold=config.rss_threshold,
        iters=config.iters,
        min_inliers=config.min_inliers,
        k_for_scale=config.k_for_scale,
        attach_multiplier=config.attach_multiplier,
        seed=config.seed,
        min_length_cm=config.min_length_cm,
        n_tpcs=config.n_tpcs,
        match_dist_tol=config.match_dist_tol,
        match_angle_deg=config.match_angle_deg,
        match_endpoint_dist_tol=config.match_endpoint_dist_tol,
        match_endpoint_weight=config.match_endpoint_weight,
        match_angle_weight=config.match_angle_weight,
        match_quality_weight=config.match_quality_weight,
        match_max_tpc_gap=config.match_max_tpc_gap,
        vertex_eps=config.vertex_eps,
        vertex_min_samples=config.vertex_min_samples,
        min_tracks_for_shower=config.min_tracks_for_shower,
        split_track_components=config.split_track_components,
        split_radius_cm=config.split_radius_cm,
        split_min_component_hits=config.split_min_component_hits,
        promote_line_like_leftovers=config.promote_line_like_leftovers,
        rescue_dbscan_eps=config.rescue_dbscan_eps,
        rescue_dbscan_min_samples=config.rescue_dbscan_min_samples,
        rescue_min_hits=config.rescue_min_hits,
        rescue_min_length_cm=config.rescue_min_length_cm,
        rescue_min_linearity=config.rescue_min_linearity,
        rescue_max_transverse_rms=config.rescue_max_transverse_rms,
        track_noise_absorption_enable=config.track_noise_absorption_enable,
        track_noise_absorb_radius_scale=config.track_noise_absorb_radius_scale,
        track_noise_absorb_min_base_radius_cm=config.track_noise_absorb_min_base_radius_cm,
        track_noise_absorb_endpoint_margin_cm=config.track_noise_absorb_endpoint_margin_cm,
        leftover_dbscan_eps=config.leftover_dbscan_eps,
        leftover_dbscan_min_samples=config.leftover_dbscan_min_samples,
        return_label_info=True,
        return_debug_info=True,
    )

    labels_global = np.asarray(labels_global, dtype=np.int32)
    n_noise = int(np.count_nonzero(labels_global < 0))
    n_labeled = int(np.count_nonzero(labels_global >= 0))
    n_labels = int(labels_global[labels_global >= 0].max() + 1) if n_labeled else 0
    track_shower_labels = list(range(int(split_index)))
    cluster_labels = list(range(int(split_index), int(n_labels)))
    backbone_type_counts: dict[str, int] = {}
    for label in track_shower_labels:
        label_type = str(label_info.get(int(label), {}).get("type", "track"))
        backbone_type_counts[label_type] = backbone_type_counts.get(label_type, 0) + 1

    return ClusteringResult(
        labels_global=labels_global,
        split_index=int(split_index),
        label_info={int(k): dict(v) for k, v in label_info.items()},
        debug=dict(debug),
        track_shower_labels=track_shower_labels,
        cluster_labels=cluster_labels,
        n_noise=n_noise,
        n_labeled=n_labeled,
        n_labels=n_labels,
        backbone_type_counts=backbone_type_counts,
    )


__all__ = [
    "ClusteringResult",
    "load_track_clustering_toolbox",
    "run_global_track_clustering",
]
