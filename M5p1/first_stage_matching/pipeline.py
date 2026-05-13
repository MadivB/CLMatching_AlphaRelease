from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import h5py

from .clustering import ClusteringResult, run_global_track_clustering
from .config import FirstStageConfig
from .io import EventData, load_event, open_flow_file
from .paths import configure_paths
from .prediction import ModelBundle, PredictionBundle, load_first_stage_models, predict_first_stage_images_and_std
from .track_stage import TrackStageResult, run_track_shower_stage

configure_paths()

from v8_1_flash_seeding import extract_flash_t0_candidates_from_table


@dataclass(slots=True)
class FirstStageResult:
    event: EventData
    clustering: ClusteringResult
    models: ModelBundle
    prediction: PredictionBundle
    track_stage: TrackStageResult
    timings: dict[str, float]

    @property
    def hit_t0(self):
        return self.track_stage.hit_t0_export

    @property
    def current_light_prediction(self):
        return self.track_stage.base_image

    @property
    def modified_flash_table_by_tpc(self):
        return self.track_stage.modified_flash_table_by_tpc


def run_first_stage_charge_light_matching(
    *,
    data_file: str | None = None,
    h5: h5py.File | None = None,
    event_id: int,
    config: FirstStageConfig | None = None,
    models: ModelBundle | None = None,
    verbose: bool = True,
) -> FirstStageResult:
    """
    Run only the first-stage charge-light matching.

    Output hit times use -1 for all hits that are not assigned by the
    track/shower stage. The returned base image is the current predicted light
    after track/shower placement, rescan/swap, and fine-t0 correction.
    """
    config = FirstStageConfig() if config is None else config
    t_pipeline_total = time.perf_counter()
    t_open = 0.0
    if h5 is None:
        if data_file is None:
            raise ValueError("Provide either data_file or h5.")
        t0 = time.perf_counter()
        h5 = open_flow_file(data_file)
        t_open = time.perf_counter() - t0

    t0 = time.perf_counter()
    event = load_event(h5, int(event_id))
    t_load_event = time.perf_counter() - t0
    if verbose:
        print(
            f"Loaded event {event.event_id}: hits={event.x.size} | light_id={event.light_id} "
            f"[open={t_open:.2f}s, load={t_load_event:.2f}s]"
        )

    t0 = time.perf_counter()
    if config.track_stage.enable_flash_table_seeding:
        flash_seed_t0s_by_tpc, flash_seed_stats = extract_flash_t0_candidates_from_table(
            h5_file=h5,
            eventid=int(event_id),
            search_range=config.track_stage.search_range,
            max_new_per_tpc=config.track_stage.flash_max_new_per_tpc,
            t0_resolution=config.track_stage.t0_resolution,
            flash_tick_divisor=config.track_stage.flash_tick_divisor,
            flash_tick_offset=config.track_stage.flash_tick_offset,
            charge_tpc_min=0,
            charge_tpc_max=int(event.full_light_waveform.shape[0]) - 1,
        )
    else:
        flash_seed_t0s_by_tpc = {}
        flash_seed_stats = {
            "eventid": int(event_id),
            "n_charge_tpcs_with_flash_seeds": 0,
            "n_flash_seed_t0s": 0,
            "disabled": True,
        }
    t_flash_load = time.perf_counter() - t0
    if verbose:
        print(
            "Flash table loaded in first stage: "
            f"TPCs={int(flash_seed_stats.get('n_charge_tpcs_with_flash_seeds', 0))} | "
            f"t0s={int(flash_seed_stats.get('n_flash_seed_t0s', 0))} | "
            f"amend_window=+/-{int(config.track_stage.flash_amend_window_ticks)} ticks | "
            f"{t_flash_load:.2f}s"
        )

    t0 = time.perf_counter()
    clustering = run_global_track_clustering(
        x=event.x,
        y=event.y,
        z=event.z,
        io_group=event.io_group,
        config=config.clustering,
    )
    t_clustering = time.perf_counter() - t0
    if verbose:
        print(
            f"Clustering: labels={clustering.n_labels} | "
            f"track/shower={len(clustering.track_shower_labels)} | "
            f"clusters={len(clustering.cluster_labels)} | noise={clustering.n_noise} "
            f"[{t_clustering:.2f}s]"
        )

    t_model_load = 0.0
    if models is None:
        t0 = time.perf_counter()
        models = load_first_stage_models(config)
        t_model_load = time.perf_counter() - t0
        if verbose:
            print(f"Model load inside pipeline: {t_model_load:.2f}s")

    t0 = time.perf_counter()
    prediction = predict_first_stage_images_and_std(
        x=event.x,
        y=event.y,
        z=event.z,
        energy=event.energy,
        hit_tpc_id=event.hit_tpc_id,
        labels_global=clustering.labels_global,
        split_index=clustering.split_index,
        label_info=clustering.label_info,
        full_light_waveform=event.full_light_waveform,
        models=models,
        config=config,
    )
    t_prediction = time.perf_counter() - t0
    if verbose:
        timings = prediction.image_meta.get("timings", {}) if isinstance(prediction.image_meta, dict) else {}
        pred_pipe = prediction.image_meta.get("pipeline_timings", {}) if isinstance(prediction.image_meta, dict) else {}
        timing_text = ""
        if timings:
            timing_text = (
                f" | image total={float(timings.get('total_s', 0.0)):.1f}s"
                f" (group={float(timings.get('grouping_s', 0.0)):.1f},"
                f" voxel={float(timings.get('voxelize_s', 0.0)):.1f},"
                f" model={float(timings.get('model_s', 0.0)):.1f},"
                f" mat={float(timings.get('materialize_s', 0.0)):.1f})"
            )
        print(
            f"Prediction: imageMaps={len(prediction.image_maps)} | "
            f"support entries={prediction.cluster_channel_support_summary.get('n_entries', 0)}"
            f"{timing_text} | prediction total={t_prediction:.1f}s"
        )
        if pred_pipe:
            print(
                "Prediction stage timing: "
                f"images={float(pred_pipe.get('image_prediction_s', 0.0)):.2f}s | "
                f"maps={float(pred_pipe.get('cluster_tpc_maps_s', 0.0)):.2f}s | "
                f"support={float(pred_pipe.get('support_cache_s', 0.0)):.2f}s | "
                f"saturation={float(pred_pipe.get('saturation_veto_s', 0.0)):.2f}s | "
                f"std={float(pred_pipe.get('variance_std_s', 0.0)):.2f}s | "
                f"total={float(pred_pipe.get('total_s', 0.0)):.2f}s"
            )

    t0 = time.perf_counter()
    track_stage = run_track_shower_stage(
        x=event.x,
        y=event.y,
        z=event.z,
        energy=event.energy,
        io_group=event.io_group,
        hit_tpc_id=event.hit_tpc_id,
        labels_global=clustering.labels_global,
        split_index=clustering.split_index,
        label_info=clustering.label_info,
        prediction=prediction,
        models=models,
        flash_seed_t0s_by_tpc=flash_seed_t0s_by_tpc,
        flash_seed_stats=flash_seed_stats,
        config=config,
        verbose=verbose,
    )
    t_track_stage = time.perf_counter() - t0
    if verbose:
        n_assigned = int((track_stage.hit_t0_export >= 0).sum())
        print(
            f"First-stage done: assigned hits={n_assigned}/{track_stage.hit_t0_export.size} | "
            f"t0 table TPCs={len(track_stage.modified_flash_table_by_tpc)} | "
            f"track stage={t_track_stage:.2f}s"
        )
        print(
            "Flash table amendments: "
            f"raw t0s={int(track_stage.flash_seed_stats.get('n_raw_flash_t0s', 0))} | "
            f"amended t0s={int(track_stage.flash_seed_stats.get('n_amended_flash_t0s', 0))} | "
            f"amendments={int(track_stage.flash_seed_stats.get('n_flash_amendments', 0))}"
        )
        print(
            "Track corrections: "
            f"rescan changed={int(track_stage.track_second_pass_stats.get('n_tracks_changed_t0', 0))} | "
            f"swap candidates={int(track_stage.track_swap_stats.get('candidate_pairs', 0))} | "
            f"swap accepted={int(track_stage.track_swap_stats.get('accepted_swaps', 0))} | "
            f"fine changed={int(track_stage.track_t0_fine_correction_stats.get('n_tracks_changed', 0))} | "
            f"leftover hits absorbed={int(track_stage.leftover_absorption_stats.get('n_absorbed_hits', 0))}"
        )
        tt = track_stage.stage_timings
        print(
            "Track stage timing: "
            f"leftover_prep={float(tt.get('leftover_prep_s', 0.0)):.2f}s | "
            f"initial={float(tt.get('initial_track_scan_s', 0.0)):.2f}s "
            f"(stack={float(tt.get('initial_fit_stack_s', 0.0)):.2f}, "
            f"t0={float(tt.get('initial_t0_scan_s', 0.0)):.2f}, "
            f"apply={float(tt.get('initial_apply_s', 0.0)):.2f}) | "
            f"inline_absorb={float(tt.get('inline_leftover_absorption_s', 0.0)):.2f}s | "
            f"rescan={float(tt.get('track_rescan_s', 0.0)):.2f}s | "
            f"swap={float(tt.get('track_swap_s', 0.0)):.2f}s | "
            f"fine={float(tt.get('fine_correction_s', 0.0)):.2f}s | "
            f"total={float(tt.get('total_s', 0.0)):.2f}s"
        )
        print(
            "Pipeline timing: "
            f"open={t_open:.2f}s | load_event={t_load_event:.2f}s | "
            f"flash_table={t_flash_load:.2f}s | "
            f"clustering={t_clustering:.2f}s | model_load={t_model_load:.2f}s | "
            f"prediction={t_prediction:.2f}s | track_stage={t_track_stage:.2f}s | "
            f"total={time.perf_counter() - t_pipeline_total:.2f}s"
        )

    timings_out = {
        "open_file_s": float(t_open),
        "load_event_s": float(t_load_event),
        "flash_table_s": float(t_flash_load),
        "clustering_s": float(t_clustering),
        "model_load_s": float(t_model_load),
        "prediction_s": float(t_prediction),
        "track_stage_s": float(t_track_stage),
        "total_s": float(time.perf_counter() - t_pipeline_total),
    }

    return FirstStageResult(
        event=event,
        clustering=clustering,
        models=models,
        prediction=prediction,
        track_stage=track_stage,
        timings=timings_out,
    )


__all__ = [
    "FirstStageResult",
    "run_first_stage_charge_light_matching",
]
