from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import FirstStageConfig
from .paths import configure_paths

configure_paths()

from ML_NDfull_perceiver import load_perceiver_model, process_clusters_to_imageMaps
from v10_3_support import build_cluster_channel_support_cache_v10_3
from v12_saturation_mask import build_saturated_channel_cache_v12, apply_saturation_veto_to_support_cache_v12
from var_prediction.inference_ndfl_v2 import (
    load_model as load_variance_model,
    predict as predict_variance,
    resolve_checkpoint as resolve_variance_checkpoint,
)
from .streaming_prediction import process_clusters_to_imageMaps_streaming


@dataclass(slots=True)
class ModelBundle:
    light_model: Any
    light_meta: dict[str, Any]
    waveform_template: np.ndarray
    variance_model: Any | None
    variance_meta: dict[str, Any] | None
    variance_checkpoint_path: str | None
    device: str
    variance_device: str


@dataclass(slots=True)
class PredictionBundle:
    full_light_waveform: np.ndarray
    image_maps: dict[tuple[int, int], np.ndarray]
    image_meta: dict[str, Any]
    cluster_to_tpcs: dict[int, list[int]]
    tpc_to_clusters: dict[int, list[int]]
    cluster_channel_support_cache: dict[tuple[int, int], dict[str, Any]]
    cluster_channel_support_summary: dict[str, Any]
    saturated_channel_cache: dict[str, Any]
    saturated_channel_summary: dict[str, Any]
    saturation_support_summary: dict[str, Any]
    full_light_std_phase1: np.ndarray
    full_light_std_phase2: np.ndarray


def resolve_device(device: str = "auto") -> str:
    if str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def load_first_stage_models(config: FirstStageConfig | None = None) -> ModelBundle:
    config = FirstStageConfig() if config is None else config
    device = resolve_device(config.model.device)

    light_model, light_meta = load_perceiver_model(config.model.light_checkpoint, device=device)
    waveform_template = np.load(config.model.pulse_template).astype(np.float32) / np.float32(999.0)

    variance_model = None
    variance_meta = None
    variance_checkpoint_path = None
    variance_device = device
    try:
        variance_checkpoint_path = resolve_variance_checkpoint(list(config.model.variance_checkpoints))
        variance_model, variance_meta = load_variance_model(
            variance_checkpoint_path,
            device=variance_device,
            waveform_len=config.track_stage.waveform_len,
            num_channels=120,
        )
    except FileNotFoundError:
        if not config.model.allow_variance_fallback:
            raise
    except torch.cuda.OutOfMemoryError:
        if variance_device != "cpu":
            variance_device = "cpu"
            torch.cuda.empty_cache()
            variance_checkpoint_path = resolve_variance_checkpoint(list(config.model.variance_checkpoints))
            variance_model, variance_meta = load_variance_model(
                variance_checkpoint_path,
                device=variance_device,
                waveform_len=config.track_stage.waveform_len,
                num_channels=120,
            )
        else:
            raise

    return ModelBundle(
        light_model=light_model,
        light_meta=dict(light_meta),
        waveform_template=waveform_template,
        variance_model=variance_model,
        variance_meta=None if variance_meta is None else dict(variance_meta),
        variance_checkpoint_path=variance_checkpoint_path,
        device=device,
        variance_device=variance_device,
    )


def build_cluster_tpc_maps(
    image_maps: dict[tuple[int, int], np.ndarray],
    *,
    split_index: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    cluster_to_tpcs: dict[int, list[int]] = {}
    tpc_to_clusters: dict[int, list[int]] = {}
    for cluster_id, tpc_id in image_maps.keys():
        cluster_to_tpcs.setdefault(int(cluster_id), []).append(int(tpc_id))
        if int(cluster_id) >= int(split_index):
            tpc_to_clusters.setdefault(int(tpc_id), []).append(int(cluster_id))
    for values in cluster_to_tpcs.values():
        values.sort()
    for values in tpc_to_clusters.values():
        values.sort()
    return cluster_to_tpcs, tpc_to_clusters


def predict_first_stage_images_and_std(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    hit_tpc_id: np.ndarray,
    labels_global: np.ndarray,
    split_index: int,
    label_info: dict[int, dict[str, Any]],
    full_light_waveform: np.ndarray,
    models: ModelBundle,
    config: FirstStageConfig | None = None,
) -> PredictionBundle:
    config = FirstStageConfig() if config is None else config

    t_total = time.perf_counter()
    t0 = time.perf_counter()
    if str(config.prediction.image_prediction_mode).lower() == "streaming":
        image_maps, image_meta = process_clusters_to_imageMaps_streaming(
            x,
            y,
            z,
            energy,
            hit_tpc_id,
            labels_global,
            model=models.light_model,
            target_scale=config.prediction.target_scale,
            template=models.waveform_template,
            batch_size=config.prediction.image_batch_size,
            raw_clip=config.prediction.raw_clip,
            min_prediction_threshold=config.prediction.min_prediction_threshold,
            device_policy=config.prediction.device_policy,
            voxelize_device=config.prediction.image_voxelize_device,
            use_mixed_precision=config.prediction.image_use_mixed_precision,
            amp_dtype=config.prediction.image_amp_dtype,
            store_dense_meta=config.prediction.image_store_dense_meta,
        )
    elif str(config.prediction.image_prediction_mode).lower() == "dense":
        image_maps, image_meta = process_clusters_to_imageMaps(
            x,
            y,
            z,
            energy,
            hit_tpc_id,
            labels_global,
            model=models.light_model,
            target_scale=config.prediction.target_scale,
            template=models.waveform_template,
            batch_size=config.prediction.image_batch_size,
            raw_clip=config.prediction.raw_clip,
            min_prediction_threshold=config.prediction.min_prediction_threshold,
            device_policy=config.prediction.device_policy,
        )
        image_meta = dict(image_meta)
        image_meta["streaming"] = False
    else:
        raise ValueError("PredictionConfig.image_prediction_mode must be 'streaming' or 'dense'.")
    t_image = time.perf_counter() - t0

    t0 = time.perf_counter()
    cluster_to_tpcs, tpc_to_clusters = build_cluster_tpc_maps(image_maps, split_index=split_index)
    t_maps = time.perf_counter() - t0

    t0 = time.perf_counter()
    support_cache, support_summary = build_cluster_channel_support_cache_v10_3(
        image_maps,
        cluster_to_tpcs,
        label_info,
        split_index=int(split_index),
        light_fraction=config.support.light_fraction,
        max_gap=config.support.max_gap,
    )
    t_support = time.perf_counter() - t0

    t0 = time.perf_counter()
    saturated_channel_cache, saturated_channel_summary = build_saturated_channel_cache_v12(
        full_light_waveform,
        clip_threshold=config.support.saturation_clip_threshold,
        max_clip_ticks=config.support.saturation_max_clip_ticks,
    )
    support_cache, saturation_support_summary = apply_saturation_veto_to_support_cache_v12(
        support_cache,
        image_maps=image_maps,
        saturated_channel_cache=saturated_channel_cache,
    )
    t_saturation = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_light_std_phase1 = np.ones_like(full_light_waveform, dtype=np.float32)
    if models.variance_model is None:
        full_light_std_phase2 = np.ones_like(full_light_waveform, dtype=np.float32)
    else:
        full_light_std_phase2 = predict_variance(
            models.variance_model,
            full_light_waveform,
            batch_size=config.prediction.variance_batch_size,
            input_scale=config.prediction.variance_input_scale,
            target_scale=config.prediction.variance_target_scale,
            device=models.variance_device,
            return_variance=True,
            min_sigma=config.prediction.variance_min_sigma_adc,
        ).astype(np.float32)
    t_variance = time.perf_counter() - t0

    image_meta = dict(image_meta)
    image_meta["pipeline_timings"] = {
        "image_prediction_s": float(t_image),
        "cluster_tpc_maps_s": float(t_maps),
        "support_cache_s": float(t_support),
        "saturation_veto_s": float(t_saturation),
        "variance_std_s": float(t_variance),
        "total_s": float(time.perf_counter() - t_total),
    }

    return PredictionBundle(
        full_light_waveform=np.asarray(full_light_waveform, dtype=np.float32),
        image_maps=image_maps,
        image_meta=dict(image_meta),
        cluster_to_tpcs=cluster_to_tpcs,
        tpc_to_clusters=tpc_to_clusters,
        cluster_channel_support_cache=support_cache,
        cluster_channel_support_summary=dict(support_summary),
        saturated_channel_cache=saturated_channel_cache,
        saturated_channel_summary=dict(saturated_channel_summary),
        saturation_support_summary=dict(saturation_support_summary),
        full_light_std_phase1=full_light_std_phase1,
        full_light_std_phase2=full_light_std_phase2,
    )


__all__ = [
    "ModelBundle",
    "PredictionBundle",
    "resolve_device",
    "load_first_stage_models",
    "build_cluster_tpc_maps",
    "predict_first_stage_images_and_std",
]
