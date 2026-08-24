from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import REPO_ROOT


def _resolve_perceiver_default() -> str:
    """Resolve the perceiver-weight path from paths.yaml; fall back to the
    legacy in-repo location if paths.yaml is absent or doesn't override it."""
    try:
        from .asset_resolver import resolve_asset
        r = resolve_asset("perceiver_charge_light_relation")
        if r.candidates:
            return r.path or r.candidates[0]
    except Exception:
        pass
    return str(REPO_ROOT / "NewMLSection/runs/ndfull_run_distributed/checkpoint.pt")


def _resolve_pulse_template_default() -> str:
    """Resolve the pulse template from paths.yaml; fall back to the bundled
    copy at assets/avg_pulse.npy."""
    try:
        from .asset_resolver import resolve_asset
        r = resolve_asset("pulse_template")
        if r.candidates:
            return r.path or r.candidates[0]
    except Exception:
        pass
    bundled = REPO_ROOT / "assets" / "avg_pulse.npy"
    if bundled.exists():
        return str(bundled)
    return "/global/cfs/cdirs/dune/users/yuxuan/interactLevel/clusteringStudy/dataDrivenLUTtable/MLApproach/CNNApproach/avg_pulse.npy"


def _resolve_variance_default() -> tuple[str, ...]:
    """Resolve variance checkpoint candidates from paths.yaml."""
    try:
        from .asset_resolver import resolve_asset
        r = resolve_asset("variance_prediction")
        if r.candidates:
            return tuple(r.candidates)
    except Exception:
        pass
    return (
        str(REPO_ROOT / "NewMLSection/var_prediction/runs/var_run_perceiver_aligned_0001000_v2/best_model.pt"),
        str(REPO_ROOT / "NewMLSection/var_prediction/runs/var_run_perceiver_aligned_0001000_v2/checkpoint.pt"),
    )


@dataclass(slots=True)
class ModelConfig:
    light_checkpoint: str = field(default_factory=_resolve_perceiver_default)
    pulse_template: str = field(default_factory=_resolve_pulse_template_default)
    variance_checkpoints: tuple[str, ...] = field(default_factory=_resolve_variance_default)
    allow_variance_fallback: bool = True
    device: str = "auto"


@dataclass(slots=True)
class PredictionConfig:
    image_batch_size: int = 8
    image_prediction_mode: str = "streaming"  # "streaming" avoids all-groups dense voxel tensors; "dense" is legacy.
    # Quick-test knob: when set, skip perceiver image prediction for blob
    # clusters (label >= split_index) that are single-TPC AND have total
    # E <= this many MeV -- exactly the population Phase 3 would sweep up
    # (v11_phased_matching._partition_cluster_population routes a cluster to
    # the small-cluster matrix stage unless it spans >1 TPC or E > 50 MeV).
    # Backbone track/shower labels are always predicted.  None = off
    # (vanilla, bit-identical).  Streaming mode only.
    min_cluster_energy_mev: float | None = None
    image_voxelize_device: str = "auto"  # "auto" uses GPU scatter when the light model is on CUDA.
    image_use_mixed_precision: bool = False
    image_amp_dtype: str = "bf16"
    image_store_dense_meta: bool = False
    variance_batch_size: int = 8
    target_scale: float = 1e-3
    variance_input_scale: float = 1e-3
    variance_target_scale: float = 1e-3
    variance_min_sigma_adc: float = 1.0
    raw_clip: tuple[float, float] = (0.0, 60780.0)
    min_prediction_threshold: float | None = 100.0
    device_policy: str = "auto"


@dataclass(slots=True)
class ClusteringConfig:
    lam: float = 1.2
    rss_threshold: float = 1.5e6
    iters: int = 800
    min_inliers: int = 35
    k_for_scale: int = 8
    attach_multiplier: float = 1.15
    seed: int = 0
    min_length_cm: float = 30.0
    n_tpcs: int = 70
    match_dist_tol: float = 5.0
    match_angle_deg: float = 12.0
    match_endpoint_dist_tol: float = 40.0
    match_endpoint_weight: float = 0.45
    match_angle_weight: float = 0.35
    match_quality_weight: float = 0.15
    match_max_tpc_gap: int | None = None
    vertex_eps: float = 10.0
    vertex_min_samples: int = 3
    min_tracks_for_shower: int = 3
    split_track_components: bool = True
    split_radius_cm: float = 4.0
    split_min_component_hits: int = 20
    promote_line_like_leftovers: bool = True
    rescue_dbscan_eps: float = 4.0
    rescue_dbscan_min_samples: int = 3
    rescue_min_hits: int = 15
    rescue_min_length_cm: float = 20.0
    rescue_min_linearity: float = 0.88
    rescue_max_transverse_rms: float = 5.0
    track_noise_absorption_enable: bool = True
    track_noise_absorb_radius_scale: float = 1.5
    track_noise_absorb_min_base_radius_cm: float = 1.2
    track_noise_absorb_endpoint_margin_cm: float = 4.0
    leftover_dbscan_eps: float = 4.0
    leftover_dbscan_min_samples: int = 3
    # ---- intersection refinement (opt-in; vanilla bit-identical when off).
    # Post-clustering hit-level relabel at track/cluster and track/track
    # crossings; see first_stage_matching/intersection_refine.py.  Requires
    # `energy` to be passed to run_global_track_clustering when enabled.
    intersection_refine_enable: bool = False
    ir_bin_cm: float = 2.0                       # axial density bin width
    ir_min_baseline_bins: int = 5                # >=10 cm clean track to trust rho0
    ir_min_baseline_mevcm: float = 0.05          # below this the "track" is junk
    ir_soft_factor: float = 1.5                  # soft cap = f * baseline in a crossing
    ir_endpoint_soft_factor: float | None = None # None = skip endpoint windows (Bragg guard); e.g. 3.0 to allow bounded shedding there
    ir_min_excess_mev: float = 3.0               # trigger dead-band
    ir_window_margin_cm: float = 3.0             # slop around the geometric contact
    ir_max_window_frac: float = 0.5              # window > half the segment => parallel/degenerate, skip
    ir_endpoint_guard_cm: float = 6.0            # Bragg rise lives in the last few cm
    ir_core_frac: float = 0.5                    # core radius = frac * r_base (never moves)
    ir_core_min_cm: float = 0.55
    ir_halo_link_cm: float = 2.8                 # shed hits must touch the blob (stage-2.5 radius)
    ir_theta_min_deg: float = 10.0               # below this, crossing windows blow up
    ir_cross_dist_scale: float = 1.0
    ir_halflen_min_cm: float = 3.0
    ir_halflen_max_cm: float = 10.0
    ir_clear_margin: float = 0.35                # geometric pinning margin (stage-2.5 uses 0.08)
    ir_ratio_tol: float = 0.15                   # don't touch near-balanced crossings
    ir_donor_floor_ratio: float = 1.0            # donor keeps >= its own baseline continuation
    ir_pair_energy_cap_mev: float = 80.0         # stage-2.5 commit-gate value
    ir_recipient_gain_frac: float = 1.5          # blob gains <= f * its own energy
    ir_min_recipient_hits: int = 3
    ir_min_pool_hits: int = 4
    ir_min_track_hits: int = 10
    ir_min_track_len_cm: float = 8.0
    ir_max_pairs: int = 256                      # runaway guard (a busy 176k-hit ND event has ~180 candidates)
    ir_gap_guard_cm: float = 4.0                 # new-gap rollback threshold (matches gap-density prune)
    ir_recipient_types: tuple = ("cluster", "shower")
    # ---- vertex-track merging (opt-in, v0.2.5; vanilla bit-identical when off).
    # Merge backbone track clusters whose ENDS converge on a common origin and
    # whose end-directions extend away from it; deliberately conservative —
    # when in doubt, keep them fragmented.  first_stage_matching/vertex_merge.py.
    vertex_merge_enable: bool = False
    vm_end_radius_cm: float = 4.0                # endpoint-to-endpoint gate
    vm_line_tol_cm: float = 2.5                  # both end-lines must pass this close to the vertex
    vm_angle_min_deg: float = 15.0               # exclude collinear joins (broken tracks)
    vm_angle_max_deg: float = 165.0              # exclude back-to-back pass-throughs
    vm_min_hits: int = 15
    vm_min_length_cm: float = 4.0
    vm_end_len_cm: float = 8.0                   # local end-fit window
    vm_crossing_guard_cm: float = 3.0            # endpoint near the OTHER track's core => X, not V
    vm_min_linearity: float = 0.75               # fuzzy end fits never merge
    vm_min_local_hits: int = 6
    vm_max_group: int = 6                        # oversized vertex groups are dropped whole
    # ---- shower noise absorption (opt-in, v0.3.5): showers admit label -1
    # hits through their PCA ellipsoid; tracks keep the existing toolbox pass.
    shower_absorb_enable: bool = False
    shower_absorb_k_sigma: float = 2.2
    shower_absorb_pad_frac: float = 0.10
    shower_absorb_axis_floor_cm: float = 2.0
    shower_absorb_max_frac: float = 0.30


@dataclass(slots=True)
class SupportConfig:
    light_fraction: float = 0.90
    max_gap: int = 2
    saturation_clip_threshold: float = 60700.0
    saturation_max_clip_ticks: int = 6


@dataclass(slots=True)
class TrackStageConfig:
    adc_clip: float = 60780.0
    waveform_len: int = 1000
    t0_resolution: int = 5
    search_range: int = 895
    scan_mode: str = "correlation"  # "correlation" is fast unit-std, "exact" matches notebook loop.
    unit_scan_engine: str = "fft"  # "fft" is fastest for the fixed 1000-tick unit-std scan; "numpy" is fallback.
    print_track_assignments: bool = False
    enable_second_pass_rescan: bool = True
    enable_overlap_swap: bool = True
    enable_fine_correction: bool = True
    fine_grid_offsets: np.ndarray = field(default_factory=lambda: np.arange(-1.5, 1.5 + 1e-6, 0.5, dtype=np.float32))
    fine_improvement_eps: float = 0.0
    enable_flash_table_seeding: bool = True
    flash_max_new_per_tpc: int | None = None
    flash_tick_divisor: float = 16.0
    flash_tick_offset: float = -5.0
    flash_amend_window_ticks: int = 10
    flash_candidate_min_sep_ticks: int = 2
    track_guard_min_shower_energy_mev: float = 80.0
    track_guard_min_t0_separation_ticks: int = 8
    track_guard_clean_worsen_tolerance_norm: float = 0.08
    swap_min_energy_ratio: float = 0.50
    swap_min_shared_tpcs: int = 1
    swap_max_tpc_sym_diff: int = 3
    swap_max_angle_deg: float = 20.0
    swap_max_shared_yz_dist_cm: float = 35.0
    swap_max_endpoint_yz_dist_cm: float = 45.0
    swap_min_t0_separation_ticks: int = 8
    swap_max_passes: int = 8
    swap_improvement_eps: float = 0.0
    swap_lock_swapped_clusters: bool = True
    enable_inline_leftover_absorption: bool = False
    leftover_noise_expand_frac: float = 0.20
    leftover_capacity_fraction_mev: float = 0.20
    leftover_shower_absorb_max_hits: int = 50
    leftover_huge_cluster_energy_mev: float = 50.0
    leftover_huge_cluster_absorb_max_hits: int = 20
    leftover_absorption_improvement_eps: float = 0.0
    leftover_noisy_batch_size: int = 4
    leftover_noisy_device_policy: str = "auto"
    leftover_noisy_min_prediction_threshold: float | None = 100.0
    collect_scan_losses: bool = False


@dataclass(slots=True)
class FirstStageConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    support: SupportConfig = field(default_factory=SupportConfig)
    track_stage: TrackStageConfig = field(default_factory=TrackStageConfig)


__all__ = [
    "ModelConfig",
    "PredictionConfig",
    "ClusteringConfig",
    "SupportConfig",
    "TrackStageConfig",
    "FirstStageConfig",
]
