"""Alpha-quality batch wrapper for the v2 light-rescue pipeline.

End-to-end per-event driver: front stage + Phase 2 large-cluster scan + v2
light rescue.  Built on top of the validated v2 module
(``phase25_trial2_v2_light_rescue``) without modifying any existing module.

Designed for GPU-node use.  Loads the perceiver model once via
``load_first_stage_models`` and reuses it across all events to avoid the
per-event reload cost.

Per-event outputs (NPZ + JSON) are written under ``--out-dir``:

    <out_dir>/<file_basename>__ev<event_id>.npz
    <out_dir>/<file_basename>__ev<event_id>.json

The NPZ holds the ``hit_timestamps`` before/after rescue, plus a small
collection of arrays useful for offline diff.  The JSON holds the per-move
audit + summary metrics.

CLI example::

    python -m M5p1.phase25_trial2_valpha_batch \\
        --files "<glob>" --max-files 10 --max-events-per-file 0 \\
        --out-dir /pscratch/sd/y/yuxuan/light_rescue_test/valpha_runs

The ``--max-events-per-file 0`` value means "all events in the file".
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob as glob_mod
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Path setup -- mirror what the notebook's cell 2 does so our imports work
# whether we run as a module or as a script.


def _configure_paths() -> str:
    here = Path(__file__).resolve()
    base_dir = here.parent.parent  # .../M5p1ReleaseVersion
    candidates = [
        str(base_dir),
        str(base_dir / "M5p1"),
        str(base_dir / "NewMLSection"),
        str(base_dir.parent / "2x2CLMatching"),
    ]
    for path in candidates:
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)
    return str(base_dir)


_BASE_DIR = _configure_paths()


# Imports below depend on the path set above.
from M5p1.first_stage_matching import (  # noqa: E402
    FirstStageConfig,
    load_first_stage_models,
    run_first_stage_charge_light_matching,
)
from M5p1.flash_cluster_table import (  # noqa: E402
    ensure_flash_cluster_flags,
    flash_cluster_table_rows,
    rebuild_flash_cluster_flags_from_assignments,
)
from M5p1 import phase25_trial2_v2_light_rescue as v2  # noqa: E402
from v11_phased_matching import (  # noqa: E402
    run_large_cluster_scan_phase_v11,
    snapshot_backbone_hits_v11,
    verify_backbone_hits_unchanged_v11,
)

# Constants from the notebook.
N_TPCS = 70
WVFM_LEN = 1000
ADC_CLIP = 60780.0
T0_RESOLUTION = 5
V11_LARGE_CLUSTER_ENERGY_MEV = 50.0
V11_MINIMUM_ITERATIVE_ENERGY_MEV = 0.5
V11_PRIMARY_ASSIGNMENT_IMPROVEMENT_EPS = 0.0
RETURN_FULL_SCAN_LOSSES = False


# ---------------------------------------------------------------------------
# Config builders


def build_default_first_stage_config() -> FirstStageConfig:
    fs = FirstStageConfig()
    fs.prediction.image_prediction_mode = "streaming"
    fs.prediction.image_batch_size = 8
    fs.prediction.image_voxelize_device = "auto"
    fs.prediction.image_use_mixed_precision = False
    fs.prediction.image_store_dense_meta = False
    fs.track_stage.scan_mode = "correlation"
    fs.track_stage.unit_scan_engine = "fft"
    fs.track_stage.collect_scan_losses = True
    fs.track_stage.print_track_assignments = False
    fs.track_stage.enable_second_pass_rescan = True
    fs.track_stage.enable_overlap_swap = True
    fs.track_stage.enable_fine_correction = True
    fs.track_stage.enable_inline_leftover_absorption = True
    fs.track_stage.flash_candidate_min_sep_ticks = T0_RESOLUTION
    fs.track_stage.flash_amend_window_ticks = T0_RESOLUTION
    return fs


def build_default_v2_config(
    *,
    n_outer_passes: int = 1,
    light_max_total_moves: int = 24,
    light_max_moves_per_tpc: int = 3,
    device_policy: str = "auto",
    verbose: bool = False,
) -> v2.Trial2V2LightConfig:
    """Mirror the notebook cell-15 defaults; multi-pass is opt-in."""
    return v2.Trial2V2LightConfig(
        verbose=bool(verbose),
        commit=True,
        # Large-cluster flash-grid + spatial.
        enable_large_flash_grid_correction=True,
        large_cluster_min_energy_mev=50.0,
        large_grid_offsets_ticks=(-1.0, -0.5, 0.0, 0.5, 1.0),
        large_min_loss_improvement=0.0,
        enable_spatial=True,
        skip_shower_tpcs=True,
        max_move_energy_per_tpc_mev=80.0,
        spatial_t0_match_ticks=10,
        spatial_different_t0_ticks=10.0,
        spatial_contact_radius_cm=4.0,
        spatial_max_pairs=24,
        spatial_min_pair_hits=24,
        spatial_min_pair_energy_mev=3.0,
        spatial_component_radius_cm=2.6,
        spatial_smooth_component_radius_cm=2.8,
        spatial_min_model_hits=8,
        spatial_min_model_energy_mev=0.05,
        spatial_max_models_per_t0=16,
        spatial_trim_model_quantile=0.85,
        spatial_trim_iterations=2,
        spatial_axis_width_floor_cm=0.55,
        spatial_nearest_scale_cm=2.2,
        spatial_endpoint_gap_scale_cm=6.0,
        spatial_endpoint_margin_cm=10.0,
        spatial_max_accept_score=3.0,
        spatial_rescan_pool_margin=0.75,
        spatial_component_strong_margin=0.08,
        spatial_move_margin=0.08,
        spatial_keep_inertia=0.0,
        # V2 light rescue.  Loose overflow defaults proved best on ev0+ev6.
        enable_light=True,
        light_skip_shower_tpcs=False,
        light_source_exact_associated_t0s=True,
        light_t0_match_ticks=5,
        light_flash_cluster_match_ticks=5.0,
        light_overflow_sigma=3.0,
        light_overflow_abs_adc=400.0,
        light_model_activity_adc=400.0,
        light_min_overflow_channels=6,
        light_use_physical_chi2=True,
        light_use_rescue_branches=False,
        phys_min_source_ofch_reduction=8,
        phys_min_dchi2_improvement=5.0e2,
        phys_min_dchi2_per_mev=1.0e1,
        phys_std_floor=1.0e-6,
        light_veto_multitpc_track=True,
        light_veto_track_min_tpcs=4,
        enable_force_override=True,
        light_force_override_min_src_ofch=22,
        light_force_override_min_peak_ofch=12,
        light_force_override_min_severity=250.0,
        light_force_override_min_src_red=14,
        light_force_override_min_dchi2=5.0e3,
        light_force_override_min_dchi2_per_e=5.0e1,
        enable_reshuffle=True,
        reshuffle_min_severe_rows=2,
        reshuffle_max_clusters=3,
        reshuffle_max_t0_pool=4,
        reshuffle_severity_floor=100.0,
        reshuffle_min_dchi2=5.0e2,
        reshuffle_min_dchi2_per_e=1.0e1,
        reshuffle_max_total=6,
        n_outer_passes=int(n_outer_passes),
        later_pass_phys_min_dchi2_improvement=2.0e3,
        later_pass_phys_min_dchi2_per_mev=25.0,
        later_pass_phys_min_source_ofch_reduction=12,
        light_max_total_moves=int(light_max_total_moves),
        light_max_moves_per_tpc=int(light_max_moves_per_tpc),
        prediction_batch_size=8,
        device_policy=str(device_policy),
    )


# ---------------------------------------------------------------------------
# Per-event pipeline


def _build_namespace(fs_result, h5: h5py.File, fs_config: FirstStageConfig) -> dict[str, Any]:
    """Mirror the notebook cell-3 bindings into a dict (no globals())."""
    ns: dict[str, Any] = {}
    # Event / charge / light bindings
    ns["h5"] = h5
    ns["lightID"] = int(fs_result.event.light_id)
    ns["hit_refs"] = fs_result.event.hit_refs
    ns["hits_evt"] = h5["charge/calib_prompt_hits/data"][fs_result.event.hit_refs]
    ns["hits_full"] = h5["charge/calib_prompt_hits/data"]
    ns["hits_ref"] = h5["charge/events/ref/charge/calib_prompt_hits/ref"]
    ns["geom_map"] = fs_result.event.geom_map

    ns["xset"] = fs_result.event.x
    ns["yset"] = fs_result.event.y
    ns["zset"] = fs_result.event.z
    ns["Eset"] = fs_result.event.energy
    ns["io_group"] = fs_result.event.io_group
    ns["hitTPCid"] = fs_result.event.hit_tpc_id
    ns["Nhits"] = len(ns["xset"])
    ns["fullLightWaveform"] = fs_result.event.full_light_waveform
    max_charge_tpc = int(ns["fullLightWaveform"].shape[0])
    ns["max_charge_tpc"] = max_charge_tpc

    # Clustering bindings
    ns["labels_global"] = fs_result.clustering.labels_global
    ns["split_index"] = int(fs_result.clustering.split_index)
    ns["label_info"] = fs_result.clustering.label_info
    ns["track_shower_labels"] = list(fs_result.clustering.track_shower_labels)
    ns["cluster_labels"] = list(fs_result.clustering.cluster_labels)

    # Models / prediction
    ns["model"] = fs_result.models.light_model
    ns["wvfm_tmpl"] = fs_result.models.waveform_template
    ns["device"] = fs_result.models.device

    ns["imageMaps"] = fs_result.prediction.image_maps
    ns["cluster_to_tpcs"] = fs_result.prediction.cluster_to_tpcs
    ns["tpc_to_clusters"] = fs_result.prediction.tpc_to_clusters
    ns["cluster_channel_support_cache"] = fs_result.prediction.cluster_channel_support_cache
    ns["saturated_channel_cache"] = fs_result.prediction.saturated_channel_cache
    ns["fullLightStd_phase1"] = fs_result.prediction.full_light_std_phase1
    ns["fullLightStd_phase2"] = fs_result.prediction.full_light_std_phase2
    ns["fullLightStd"] = fs_result.prediction.full_light_std_phase2

    # Track / shower stage.  Use ``hit_t0`` (numeric, NaN for unassigned), not
    # ``hit_t0_export`` (which substitutes -1 sentinel).
    ns["baseImage"] = fs_result.track_stage.base_image.copy()
    ns["hit_timestamps"] = fs_result.track_stage.hit_t0.copy()
    ns["t0Candidates"] = [list(values) for values in fs_result.track_stage.t0_candidates_by_tpc]
    ns["assignment_info"] = fs_result.track_stage.assignment_info

    # Flash table flags
    flash_recv = [
        [bool(v) for v in row]
        for row in getattr(fs_result.track_stage, "flash_cluster_received_by_tpc", [])
    ]
    if len(flash_recv) != len(ns["t0Candidates"]):
        ns["t0Candidates"], flash_recv, _ = rebuild_flash_cluster_flags_from_assignments(
            ns["t0Candidates"],
            fs_result.track_stage.assignment_info,
            resolution_ticks=float(T0_RESOLUTION),
            max_t0=800.0,
            initial_flags=None,
            prefer_existing_true=False,
        )
    else:
        flash_recv = ensure_flash_cluster_flags(ns["t0Candidates"], flash_recv)
    ns["flash_cluster_received_by_tpc"] = flash_recv

    # Notebook initializes ``unassigned_by_tpc`` as an empty dict here -- the
    # real per-TPC unassigned lists are populated downstream by Phase 2.
    ns["unassigned_by_tpc"] = {int(tpc): [] for tpc in range(max_charge_tpc)}

    # v8 leftover bookkeeping (used by Phase 2 inline absorption context).
    ns["labels_global_v8"] = fs_result.track_stage.labels_global_with_leftovers.copy()
    ns["v8_absorbed_hit_parent"] = fs_result.track_stage.absorbed_hit_parent.copy()
    ns["v8_absorption_log"] = list(fs_result.track_stage.leftover_absorption_log)
    ns["v8_absorption_stats"] = dict(fs_result.track_stage.leftover_absorption_stats)

    # v8.1 flash seeding (Phase 2 needs this).
    ns["v8_1_flash_seed_t0s_by_tpc"] = {
        int(tpc): [int(v) for v in values]
        for tpc, values in fs_result.track_stage.modified_flash_table_by_tpc.items()
    }
    ns["v8_1_flash_seed_stats"] = dict(fs_result.track_stage.flash_seed_stats)
    ns["v8_1_flash_amendment_log"] = list(fs_result.track_stage.flash_table_amendment_log)

    # Inline leftover absorption context for Phase 2 (mirrors notebook cell 3).
    leftover_state = fs_result.track_stage.leftover_state
    leftover_inline = bool(fs_config.track_stage.enable_inline_leftover_absorption)
    v11_ctx = None
    if leftover_inline and leftover_state is not None:
        v11_ctx = {
            "state": leftover_state,
            "labels_with_leftovers": ns["labels_global_v8"],
            "absorbed_hit_parent": ns["v8_absorbed_hit_parent"],
            "x": ns["xset"],
            "y": ns["yset"],
            "z": ns["zset"],
            "E": ns["Eset"],
            "hit_tpc_ids": ns["hitTPCid"],
            "model": ns["model"],
            "template": ns["wvfm_tmpl"],
            "channel_support_cache": ns["cluster_channel_support_cache"],
            "saturated_channel_cache": ns["saturated_channel_cache"],
            "target_scale": fs_config.prediction.target_scale,
            "batch_size": int(fs_config.track_stage.leftover_noisy_batch_size),
            "raw_clip": fs_config.prediction.raw_clip,
            "min_prediction_threshold": fs_config.track_stage.leftover_noisy_min_prediction_threshold,
            "device_policy": str(fs_config.track_stage.leftover_noisy_device_policy),
            "support_light_fraction": float(fs_config.support.light_fraction),
            "support_max_gap": int(fs_config.support.max_gap),
            "improvement_eps": float(fs_config.track_stage.leftover_absorption_improvement_eps),
            "absorption_log": ns["v8_absorption_log"],
        }
    ns["v11_1_leftover_absorption_context"] = v11_ctx
    return ns


def _run_phase2(ns: dict[str, Any], *, verbose: bool = False) -> None:
    """Mirror notebook cell 9 (large-cluster scan)."""
    cluster_energies = {
        int(clid): float(np.asarray(ns["Eset"])[np.asarray(ns["labels_global"]) == clid].sum())
        for clid in ns["cluster_labels"]
    }
    ns["cluster_energies"] = cluster_energies

    (
        baseImage,
        hit_timestamps,
        t0Candidates,
        assignment_info,
        unassigned_by_tpc,
        large_cluster_assignment_log,
        large_cluster_scan_updates,
        v11_large_phase_stats,
        v11_active_cluster_tpcs,
        v11_iterative_single_tpc,
        v11_pruned_iterative_clusters,
    ) = run_large_cluster_scan_phase_v11(
        cluster_labels=ns["cluster_labels"],
        cluster_to_tpcs=ns["cluster_to_tpcs"],
        image_maps=ns["imageMaps"],
        base_image=ns["baseImage"],
        full_light_waveform=ns["fullLightWaveform"],
        full_light_std=ns["fullLightStd"],
        channel_support_cache=ns["cluster_channel_support_cache"],
        labels_global=ns["labels_global"],
        hit_timestamps=ns["hit_timestamps"],
        t0_candidates=ns["t0Candidates"],
        assignment_info=ns["assignment_info"],
        unassigned_by_tpc=ns["unassigned_by_tpc"],
        cluster_energies=cluster_energies,
        large_cluster_energy_mev=V11_LARGE_CLUSTER_ENERGY_MEV,
        minimum_iterative_energy_mev=V11_MINIMUM_ITERATIVE_ENERGY_MEV,
        search_range=800,
        adc_clip=ADC_CLIP,
        collect_scan_losses=RETURN_FULL_SCAN_LOSSES,
        assignment_improvement_eps=V11_PRIMARY_ASSIGNMENT_IMPROVEMENT_EPS,
        flash_seed_t0s_by_tpc=ns["v8_1_flash_seed_t0s_by_tpc"],
        flash_seed_t0_resolution=int(T0_RESOLUTION),
        flash_cluster_received_by_tpc=ns["flash_cluster_received_by_tpc"],
        leftover_absorption_context=ns.get("v11_1_leftover_absorption_context"),
        saturated_channel_cache=ns["saturated_channel_cache"],
    )

    ns["baseImage"] = baseImage
    ns["hit_timestamps"] = hit_timestamps
    ns["t0Candidates"] = t0Candidates
    ns["assignment_info"] = assignment_info
    ns["unassigned_by_tpc"] = unassigned_by_tpc
    ns["v11_active_cluster_tpcs"] = v11_active_cluster_tpcs
    ns["v11_iterative_single_tpc"] = v11_iterative_single_tpc
    ns["v11_pruned_iterative_clusters"] = v11_pruned_iterative_clusters
    ns["flash_cluster_received_by_tpc"] = ensure_flash_cluster_flags(
        t0Candidates, ns["flash_cluster_received_by_tpc"]
    )

    if verbose:
        n_received = sum(sum(row) for row in ns["flash_cluster_received_by_tpc"])
        n_total = sum(len(row) for row in ns["t0Candidates"])
        print(f"  phase-2 cluster flashes  : {n_received} / {n_total}")


def _summarize_v2_result(result: dict[str, Any]) -> dict[str, Any]:
    light_moves = list(result.get("light_moves", []))
    spatial_moves = list(result.get("spatial_moves", []))
    reshuffle_records = list(result.get("v2_reshuffle_records", []))
    pass_logs = list(result.get("v2_pass_logs", []))
    n_force = sum(
        1
        for m in light_moves
        if bool(m.get("track_veto_overridden", False))
        and "force_override" in str(m.get("track_veto_override_reason", ""))
    )
    n_reshuffle = sum(
        1 for m in light_moves if str(m.get("accepted_by", "")) == "reshuffle"
    )
    return {
        "n_spatial_moves": int(len(spatial_moves)),
        "n_light_moves": int(len(light_moves)),
        "n_reshuffle_moves": int(n_reshuffle),
        "n_force_overrides": int(n_force),
        "n_reshuffle_records": int(len(reshuffle_records)),
        "pass_logs": pass_logs,
        "elapsed_s": float(result.get("elapsed_s", 0.0)),
        "v2_elapsed_s": float(result.get("v2_elapsed_s", 0.0)),
        "skipped_shower_tpcs": list(result.get("skipped_shower_tpcs", [])),
    }


def _serialize_light_moves(light_moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in light_moves:
        d = {
            "TPCid": int(m.get("TPCid", -1)),
            "old_t0": int(m.get("old_t0", -1)),
            "new_t0": int(m.get("new_t0", -1)),
            "n_hits": int(m.get("n_hits", 0)),
            "energy_mev": float(m.get("energy_mev", 0.0)),
            "component_id": int(m.get("component_id", -1)),
            "parent_label": int(m.get("parent_label", -1)),
            "accepted": bool(m.get("accepted", False)),
            "accepted_by": str(m.get("accepted_by", "")),
            "src_of0": int(m.get("src_of0", 0)),
            "src_of1": int(m.get("src_of1", 0)),
            "src_red": int(m.get("src_red", 0)),
            "dst_new": int(m.get("dst_new", 0)),
            "dchi2": float(m.get("dchi2", 0.0)),
            "dchi2_per_mev": float(m.get("dchi2_per_mev", 0.0)),
            "track_veto": bool(m.get("track_veto", False)),
            "track_veto_overridden": bool(m.get("track_veto_overridden", False)),
            "track_veto_n_tpcs": int(m.get("track_veto_n_tpcs", 0)),
            "track_veto_label": int(m.get("track_veto_label", -1)),
            "outer_pass": int(m.get("outer_pass", 0)),
        }
        if "reshuffle_dchi2" in m:
            d["reshuffle_dchi2"] = float(m["reshuffle_dchi2"])
            d["reshuffle_dchi2_per_mev"] = float(m.get("reshuffle_dchi2_per_mev", 0.0))
            d["reshuffle_pool"] = list(m.get("reshuffle_pool", []))
            d["reshuffle_assignment"] = list(m.get("reshuffle_assignment", []))
        out.append(d)
    return out


@dataclass
class EventReport:
    file: str
    event_id: int
    ok: bool
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_s: float = 0.0


def process_one_event(
    *,
    data_file: str,
    event_id: int,
    h5: h5py.File,
    fs_config: FirstStageConfig,
    fs_models,
    v2_config: v2.Trial2V2LightConfig,
    out_dir: Path,
    verbose: bool = False,
    save_arrays: bool = True,
) -> EventReport:
    t_start = time.perf_counter()
    file_basename = Path(data_file).stem
    tag = f"{file_basename}__ev{int(event_id):04d}"
    npz_path = out_dir / f"{tag}.npz"
    json_path = out_dir / f"{tag}.json"

    try:
        # ---- Front stage ----
        fs_result = run_first_stage_charge_light_matching(
            h5=h5,
            event_id=int(event_id),
            config=fs_config,
            models=fs_models,
            verbose=bool(verbose),
        )

        # ---- Bind into a namespace dict ----
        ns = _build_namespace(fs_result, h5, fs_config)
        hit_ts_pre_phase2 = np.asarray(ns["hit_timestamps"], dtype=np.float32).copy()
        base_image_pre_phase2 = np.asarray(ns["baseImage"], dtype=np.float32).copy()

        # ---- Phase 1 backbone snapshot (used by Phase 2 verify, optional) ----
        backbone_snapshot = snapshot_backbone_hits_v11(
            hit_timestamps=ns["hit_timestamps"],
            labels_global=ns["labels_global"],
            v8_absorbed_hit_parent=ns.get("v8_absorbed_hit_parent"),
            track_shower_labels=ns["track_shower_labels"],
        )

        # ---- Phase 2 large-cluster scan ----
        _run_phase2(ns, verbose=verbose)
        verify_backbone_hits_unchanged_v11(
            backbone_snapshot,
            hit_timestamps=ns["hit_timestamps"],
            stage_name="Phase 2 (large-cluster scan)",
        )
        hit_ts_pre_v2 = np.asarray(ns["hit_timestamps"], dtype=np.float32).copy()

        # ---- V2 light rescue ----
        result = v2.run_trial2_v2_light_rescue_from_namespace(
            ns,
            config=v2_config,
            commit=True,
        )
        hit_ts_post_v2 = np.asarray(ns["hit_timestamps"], dtype=np.float32).copy()
        base_image_post_v2 = np.asarray(ns["baseImage"], dtype=np.float32).copy()

        summary = _summarize_v2_result(result)
        light_moves_audit = _serialize_light_moves(list(result.get("light_moves", [])))

        report = {
            "ok": True,
            "file": str(data_file),
            "file_basename": file_basename,
            "event_id": int(event_id),
            "n_hits": int(np.asarray(ns["xset"]).size),
            "summary": summary,
            "light_moves": light_moves_audit,
            "v2_pass_logs": list(result.get("v2_pass_logs", [])),
            "elapsed_s": float(time.perf_counter() - t_start),
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=1, default=float)

        if save_arrays:
            np.savez_compressed(
                npz_path,
                hit_timestamps_pre_phase2=hit_ts_pre_phase2,
                hit_timestamps_pre_v2=hit_ts_pre_v2,
                hit_timestamps_post_v2=hit_ts_post_v2,
                baseImage_delta_v2=(base_image_post_v2 - base_image_pre_phase2).astype(np.float32),
                hitTPCid=np.asarray(ns["hitTPCid"], dtype=np.int32),
            )

        return EventReport(
            file=str(data_file),
            event_id=int(event_id),
            ok=True,
            summary=summary,
            elapsed_s=report["elapsed_s"],
        )
    except Exception as exc:
        err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "file": str(data_file),
                        "event_id": int(event_id),
                        "ok": False,
                        "error": str(exc),
                        "traceback": err,
                    },
                    f,
                    indent=1,
                )
        except Exception:
            pass
        return EventReport(
            file=str(data_file),
            event_id=int(event_id),
            ok=False,
            error=err,
            elapsed_s=float(time.perf_counter() - t_start),
        )


# ---------------------------------------------------------------------------
# Batch driver


def _list_events_in_file(path: str) -> list[int]:
    with h5py.File(path, "r") as f:
        ids = np.asarray(f["charge/events/data"]["id"])
    return [int(v) for v in ids]


def run_batch(
    *,
    file_paths: list[str],
    out_dir: str,
    fs_config: FirstStageConfig | None = None,
    v2_config: v2.Trial2V2LightConfig | None = None,
    max_events_per_file: int = 0,
    skip_existing: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Process every event in ``file_paths``.

    ``max_events_per_file=0`` means "all events".  Per-event errors are
    captured in the JSON output but do not abort the batch.
    """
    fs_config = fs_config or build_default_first_stage_config()
    v2_config = v2_config or build_default_v2_config()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print(f"valpha batch: {len(file_paths)} files -> {out_dir_p}", flush=True)
    print("loading first-stage models (one-time) ...", flush=True)
    t0 = time.perf_counter()
    fs_models = load_first_stage_models(fs_config)
    print(f"  models loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    reports: list[dict[str, Any]] = []
    total_events = 0
    n_ok = 0
    n_err = 0
    t_batch = time.perf_counter()
    for fi, file_path in enumerate(file_paths):
        if not Path(file_path).exists():
            print(f"[{fi+1}/{len(file_paths)}] SKIP: missing {file_path}", flush=True)
            continue
        try:
            event_ids = _list_events_in_file(file_path)
        except Exception as exc:
            print(f"[{fi+1}/{len(file_paths)}] SKIP: failed to enumerate events in {file_path}: {exc}", flush=True)
            continue
        if int(max_events_per_file) > 0:
            event_ids = event_ids[: int(max_events_per_file)]
        print(
            f"[{fi+1}/{len(file_paths)}] {Path(file_path).name}: {len(event_ids)} events",
            flush=True,
        )
        with h5py.File(file_path, "r") as h5:
            for ev_id in event_ids:
                tag = f"{Path(file_path).stem}__ev{int(ev_id):04d}"
                json_path = out_dir_p / f"{tag}.json"
                if skip_existing and json_path.exists():
                    print(f"  ev{ev_id:04d}: SKIP (existing)", flush=True)
                    continue
                t_ev = time.perf_counter()
                report = process_one_event(
                    data_file=file_path,
                    event_id=int(ev_id),
                    h5=h5,
                    fs_config=fs_config,
                    fs_models=fs_models,
                    v2_config=v2_config,
                    out_dir=out_dir_p,
                    verbose=verbose,
                )
                total_events += 1
                if report.ok:
                    n_ok += 1
                    s = report.summary
                    print(
                        f"  ev{int(ev_id):04d}: OK light_moves={s.get('n_light_moves', 0)}"
                        f" rshfl={s.get('n_reshuffle_moves', 0)}"
                        f" forceOR={s.get('n_force_overrides', 0)}"
                        f" elapsed={time.perf_counter() - t_ev:.1f}s",
                        flush=True,
                    )
                else:
                    n_err += 1
                    short_err = (report.error or "").splitlines()[-1][:200]
                    print(
                        f"  ev{int(ev_id):04d}: ERR after {time.perf_counter() - t_ev:.1f}s: {short_err}",
                        flush=True,
                    )
                reports.append({
                    "file": report.file,
                    "event_id": report.event_id,
                    "ok": report.ok,
                    "elapsed_s": report.elapsed_s,
                    "summary": report.summary if report.ok else {"error": report.error},
                })
                # Free per-event memory between events.
                gc.collect()
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    summary_path = out_dir_p / "valpha_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_files": int(len(file_paths)),
                "n_events_attempted": int(total_events),
                "n_ok": int(n_ok),
                "n_err": int(n_err),
                "wall_s": float(time.perf_counter() - t_batch),
                "per_event": reports,
            },
            f,
            indent=1,
            default=float,
        )
    print(
        f"valpha batch complete: events={total_events} ok={n_ok} err={n_err}"
        f" wall={time.perf_counter() - t_batch:.1f}s",
        flush=True,
    )
    return {
        "n_events_attempted": total_events,
        "n_ok": n_ok,
        "n_err": n_err,
        "summary_path": str(summary_path),
    }


# ---------------------------------------------------------------------------
# CLI


def _expand_files(file_args: list[str]) -> list[str]:
    out: list[str] = []
    for arg in file_args:
        # Allow either explicit paths or glob patterns.
        matches = sorted(glob_mod.glob(arg))
        if matches:
            out.extend(matches)
        elif Path(arg).exists():
            out.append(arg)
        else:
            print(f"WARN: no match for {arg}", file=sys.stderr)
    # de-dupe while preserving order.
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="valpha batch driver: front stage + Phase 2 + v2 light rescue.",
    )
    parser.add_argument("--files", nargs="+", required=True,
                        help="Explicit paths or glob patterns (e.g. '/dir/*.hdf5').")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for per-event NPZ + JSON.")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Cap on number of files to process (0 = all).")
    parser.add_argument("--max-events-per-file", type=int, default=0,
                        help="Cap on events per file (0 = all).")
    parser.add_argument("--n-outer-passes", type=int, default=1,
                        help="V2 multi-pass count (default 1; 2-3 for hard events).")
    parser.add_argument("--max-total-moves", type=int, default=24,
                        help="V2 light_max_total_moves (default 24).")
    parser.add_argument("--max-moves-per-tpc", type=int, default=3,
                        help="V2 light_max_moves_per_tpc (default 3).")
    parser.add_argument("--device-policy", default="auto",
                        choices=("auto", "force_cuda", "force_cpu"),
                        help="V2 prediction device policy (default auto).")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-process events even if a per-event JSON exists.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-stage diagnostics.")
    parser.add_argument("--no-arrays", action="store_true",
                        help="Skip writing per-event NPZ arrays (JSON-only).")
    args = parser.parse_args(argv)

    files = _expand_files(args.files)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    if not files:
        print("ERROR: no input files.", file=sys.stderr)
        return 2

    fs_cfg = build_default_first_stage_config()
    v2_cfg = build_default_v2_config(
        n_outer_passes=int(args.n_outer_passes),
        light_max_total_moves=int(args.max_total_moves),
        light_max_moves_per_tpc=int(args.max_moves_per_tpc),
        device_policy=str(args.device_policy),
        verbose=bool(args.verbose),
    )

    # Patch process_one_event to honour --no-arrays.
    if args.no_arrays:
        global process_one_event
        _orig = process_one_event

        def _wrap(**kw):  # type: ignore
            kw["save_arrays"] = False
            return _orig(**kw)

        process_one_event = _wrap  # type: ignore

    res = run_batch(
        file_paths=files,
        out_dir=str(args.out_dir),
        fs_config=fs_cfg,
        v2_config=v2_cfg,
        max_events_per_file=int(args.max_events_per_file),
        skip_existing=not bool(args.no_skip_existing),
        verbose=bool(args.verbose),
    )
    return 0 if res["n_err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
