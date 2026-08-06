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
    energy: np.ndarray | None = None,
) -> ClusteringResult:
    """``energy`` (per-hit MeV) is only consumed by the opt-in intersection
    refinement; the vanilla path never reads it."""
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

    if config.intersection_refine_enable:
        if energy is None:
            raise ValueError(
                "ClusteringConfig.intersection_refine_enable=True requires "
                "energy= to be passed to run_global_track_clustering."
            )
        from .intersection_refine import refine_intersections

        refined, ir_stats = refine_intersections(
            labels_global=np.asarray(labels_global),
            split_index=int(split_index),
            label_info=label_info,
            debug=debug,
            x=x,
            y=y,
            z=z,
            io_group=io_group,
            energy=energy,
            config=config,
        )
        labels_global = refined
        debug = dict(debug)
        debug["intersection_refinement"] = ir_stats
        # Refresh membership-derived diagnostics of touched labels only
        # ("type" and all other keys stay untouched; entries never deleted).
        touched = {int(l) for rec in ir_stats.get("pairs", []) for l in rec["labels"]}
        if touched:
            tpc_ids = (np.asarray(io_group, dtype=np.int64) - 1) // 2
            labels_arr = np.asarray(labels_global)
            for lab in sorted(touched):
                if lab not in label_info:
                    continue
                members = np.flatnonzero(labels_arr == lab)
                label_info[lab]["n_hits"] = int(members.size)
                label_info[lab]["tpcs"] = sorted(int(v) for v in np.unique(tpc_ids[members]))

    if config.vertex_merge_enable:
        from .vertex_merge import merge_vertex_tracks

        labels_global, vm_stats = merge_vertex_tracks(
            labels_global=np.asarray(labels_global),
            split_index=int(split_index),
            label_info=label_info,
            x=x,
            y=y,
            z=z,
            config=config,
        )
        debug = dict(debug)
        debug["vertex_merge"] = vm_stats
        touched = {int(l) for g in vm_stats.get("groups", []) for l in g}
        if touched:
            tpc_ids = (np.asarray(io_group, dtype=np.int64) - 1) // 2
            labels_arr = np.asarray(labels_global)
            for lab in sorted(touched):
                if lab not in label_info:
                    continue
                members = np.flatnonzero(labels_arr == lab)
                if members.size:
                    label_info[lab]["n_hits"] = int(members.size)
                    label_info[lab]["tpcs"] = sorted(int(v) for v in np.unique(tpc_ids[members]))

    if getattr(config, "shower_absorb_enable", False):
        from .shower_absorb import absorb_shower_noise

        labels_global, sa_stats = absorb_shower_noise(
            labels_global=np.asarray(labels_global),
            split_index=int(split_index),
            label_info=label_info,
            x=x,
            y=y,
            z=z,
            config=config,
        )
        debug = dict(debug)
        debug["shower_absorb"] = sa_stats
        touched = {int(r["label"]) for r in sa_stats.get("per_shower", [])}
        if touched:
            tpc_ids = (np.asarray(io_group, dtype=np.int64) - 1) // 2
            labels_arr = np.asarray(labels_global)
            for lab in sorted(touched):
                if lab not in label_info:
                    continue
                members = np.flatnonzero(labels_arr == lab)
                if members.size:
                    label_info[lab]["n_hits"] = int(members.size)
                    label_info[lab]["tpcs"] = sorted(int(v) for v in np.unique(tpc_ids[members]))

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
