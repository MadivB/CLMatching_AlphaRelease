from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import torch

from .config import FirstStageConfig
from .streaming_prediction import process_clusters_to_imageMaps_streaming


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_image_prediction_from_result(
    first_stage_result: Any,
    config: FirstStageConfig | None = None,
    *,
    batch_sizes: tuple[int, ...] = (8, 16, 24, 32),
    mixed_precision_options: tuple[bool, ...] = (False, True),
    voxelize_devices: tuple[str, ...] = ("auto",),
    amp_dtype: str = "bf16",
    keep_outputs: bool = False,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Benchmark all-cluster image prediction using an existing first-stage result.

    This reruns only the image prediction kernel path. It reuses event arrays,
    labels, and loaded models already held by ``first_stage_result``.
    """
    config = FirstStageConfig() if config is None else config
    event = first_stage_result.event
    clustering = first_stage_result.clustering
    models = first_stage_result.models

    rows: list[dict[str, Any]] = []
    kept_outputs: list[tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]] = []

    if verbose:
        print("Image prediction benchmark")
        print(
            f"{'batch':>6} {'mp':>5} {'voxdev':>8} {'total':>10} {'model':>10} "
            f"{'voxel':>10} {'mat':>10} {'groups':>8} {'maps':>8}"
        )
        print("-" * 84)

    for voxelize_device in voxelize_devices:
        for use_mp in mixed_precision_options:
            for batch_size in batch_sizes:
                gc.collect()
                _empty_cuda_cache()

                t0 = time.perf_counter()
                image_maps, meta = process_clusters_to_imageMaps_streaming(
                    event.x,
                    event.y,
                    event.z,
                    event.energy,
                    event.hit_tpc_id,
                    clustering.labels_global,
                    model=models.light_model,
                    target_scale=config.prediction.target_scale,
                    template=models.waveform_template,
                    batch_size=int(batch_size),
                    raw_clip=config.prediction.raw_clip,
                    min_prediction_threshold=config.prediction.min_prediction_threshold,
                    device_policy=config.prediction.device_policy,
                    voxelize_device=str(voxelize_device),
                    use_mixed_precision=bool(use_mp),
                    amp_dtype=str(amp_dtype),
                    store_dense_meta=False,
                )
                wall_s = float(time.perf_counter() - t0)
                timings = dict(meta.get("timings", {}))

                row = {
                    "batch_size": int(batch_size),
                    "mixed_precision": bool(use_mp),
                    "amp_dtype": str(amp_dtype),
                    "voxelize_device": str(timings.get("voxelize_device", voxelize_device)),
                    "wall_s": wall_s,
                    "total_s": float(timings.get("total_s", wall_s)),
                    "grouping_s": float(timings.get("grouping_s", 0.0)),
                    "voxelize_s": float(timings.get("voxelize_s", 0.0)),
                    "model_s": float(timings.get("model_s", 0.0)),
                    "materialize_s": float(timings.get("materialize_s", 0.0)),
                    "n_groups": int(timings.get("n_groups", len(image_maps))),
                    "n_image_maps": int(len(image_maps)),
                }
                rows.append(row)

                if verbose:
                    print(
                        f"{row['batch_size']:6d} {str(row['mixed_precision']):>5} "
                        f"{row['voxelize_device']:>8} {row['total_s']:10.2f} "
                        f"{row['model_s']:10.2f} {row['voxelize_s']:10.2f} "
                        f"{row['materialize_s']:10.2f} {row['n_groups']:8d} "
                        f"{row['n_image_maps']:8d}"
                    )

                if keep_outputs:
                    kept_outputs.append((image_maps, meta))
                else:
                    del image_maps
                    del meta
                    gc.collect()
                    _empty_cuda_cache()

    if keep_outputs:
        for row, output in zip(rows, kept_outputs):
            row["output"] = output

    rows.sort(key=lambda item: float(item["total_s"]))
    if verbose and rows:
        best = rows[0]
        print()
        print(
            "Best image setting: "
            f"batch={best['batch_size']} | "
            f"mixed_precision={best['mixed_precision']} | "
            f"voxelize_device={best['voxelize_device']} | "
            f"total={best['total_s']:.2f}s | model={best['model_s']:.2f}s"
        )

    return rows


__all__ = ["benchmark_image_prediction_from_result"]
