"""
streaming_prediction_fast.py

Drop-in replacement for process_clusters_to_imageMaps_streaming.

The single biggest lever available without changing dtypes or the model
architecture is letting cuDNN benchmark all convolution algorithms for your
specific input shapes (B x 1 x 50 x 300 x 100) and keep the fastest one.
PyTorch's default picks a conservative safe algorithm; the benchmark can find
FFT-based or Winograd kernels that are substantially faster for large spatial
dims.  On A100 this can reduce model inference time by 2-4x.

TF32 is also explicitly enabled (it is the default in recent PyTorch but
setting it here guarantees it regardless of global state).

No dtype changes, no CUDA graphs (graph capture overhead dominates for this
model), no modifications to existing files.

Warmup: one forward pass the first time a (model, batch_size) pair is seen.
This triggers cuDNN benchmarking and caches the result.  The cost (~0.5 s) is
paid once per session, not once per event.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paths import configure_paths

configure_paths()

from ML_NDfull_perceiver import DEFAULT_TPC_YAML, NUM_TARGETS, NX, NY, NZ

from .streaming_prediction import (
    GroupedVoxelHits,
    build_grouped_voxel_hits,
    voxelize_batch_cpu,
    voxelize_batch_cuda,
    _model_device,
    _sync_if_cuda,
)

# Tracks (id(model), batch_size) pairs that have already been warmed up so
# cuDNN benchmarking is paid at most once per session.
_WARMED_UP: set[tuple[int, int]] = set()


def _enable_fast_mode() -> None:
    """Enable cuDNN benchmark and TF32 — safe, fp32-only optimisations."""
    torch.backends.cudnn.benchmark = True   # search for fastest conv algorithm
    torch.backends.cudnn.allow_tf32 = True  # tensor-core fp32 for conv (A100)
    torch.backends.cuda.matmul.allow_tf32 = True  # tensor-core fp32 for matmul


@torch.inference_mode()
def _warmup(model: torch.nn.Module, batch_size: int, device: torch.device) -> None:
    """
    Run two forward passes to:
      1. Let cuDNN benchmark all convolution algorithms and cache the winner.
      2. Initialise model.pos_embedding (set lazily on first forward).
    Two passes ensure the second pass (used for timing) uses the cached result.
    """
    x = torch.zeros(batch_size, 1, NX, NY, NZ, device=device, dtype=torch.float32)
    t = torch.zeros(batch_size, device=device, dtype=torch.long)
    for _ in range(2):
        _ = model(x, t)
    torch.cuda.synchronize(device)


def process_clusters_to_imageMaps_fast(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    tpc_ids: np.ndarray,
    labels: np.ndarray,
    *,
    model: torch.nn.Module,
    target_scale: float = 1e-3,
    template: np.ndarray | None = None,
    include_noise: bool = False,
    batch_size: int = 16,
    raw_clip: tuple[float, float] | None = (0.0, 60780.0),
    min_prediction_threshold: float | None = 100.0,
    yaml_path: str | Path = DEFAULT_TPC_YAML,
    device_policy: str = "auto",
    voxelize_device: str = "auto",
    use_mixed_precision: bool = False,  # accepted for API compat; ignored
    amp_dtype: str = "bf16",            # accepted for API compat; ignored
    store_dense_meta: bool = False,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """
    Optimised drop-in for process_clusters_to_imageMaps_streaming.

    Changes vs baseline:
      - cuDNN benchmark enabled: cuDNN searches for fastest 3D-conv algorithm.
      - TF32 explicitly enabled for conv and matmul.
      - One-time warmup per (model, batch_size) to trigger the benchmark search.
      - Default batch_size=16 (halves loop iterations vs baseline 8).
        Auto-falls back to 8 on OOM.
      - use_mixed_precision is silently ignored (bf16 cuBLAS is broken on
        this Perlmutter cuBLAS version).
    """
    global _WARMED_UP

    _enable_fast_mode()

    t_total = time.perf_counter()
    t0 = time.perf_counter()
    grouped = build_grouped_voxel_hits(
        x, y, z, energy, tpc_ids, labels,
        include_noise=include_noise,
        yaml_path=yaml_path,
    )
    timing_group = time.perf_counter() - t0

    n_groups = int(grouped.group_cls.size)
    wave_len = int(template.size) if template is not None else 1000

    if n_groups == 0:
        return {}, {
            "group_cls": np.array([], dtype=int),
            "group_tpcs": np.array([], dtype=int),
            "maps_4d": np.zeros((0, 1, NX, NY, NZ), dtype=np.float32) if store_dense_meta else None,
            "amplitudes": np.zeros((0, NUM_TARGETS), dtype=np.float32),
            "waveforms": np.zeros((0, NUM_TARGETS, wave_len), dtype=np.float32) if store_dense_meta else None,
            "timings": {
                "grouping_s": float(timing_group), "voxelize_s": 0.0,
                "model_s": 0.0, "materialize_s": 0.0,
                "total_s": float(time.perf_counter() - t_total),
            },
        }

    device = _model_device(model, device_policy=device_policy)
    model = model.to(device).eval()

    use_cuda_voxel = (
        str(voxelize_device).lower() in {"cuda", "gpu"}
        or (str(voxelize_device).lower() == "auto" and device.type == "cuda")
    )
    if use_cuda_voxel and device.type != "cuda":
        use_cuda_voxel = False

    bs = max(1, int(batch_size))

    # One-time warmup: triggers cuDNN algorithm search and caches the winner.
    if device.type == "cuda":
        warmup_key = (id(model), bs)
        if warmup_key not in _WARMED_UP:
            try:
                _warmup(model, bs, device)
                _WARMED_UP.add(warmup_key)
            except torch.cuda.OutOfMemoryError:
                if bs > 8:
                    torch.cuda.empty_cache()
                    bs = 8
                    warmup_key = (id(model), bs)
                    if warmup_key not in _WARMED_UP:
                        _warmup(model, bs, device)
                        _WARMED_UP.add(warmup_key)
                else:
                    raise

    tmpl = (
        np.ones(wave_len, dtype=np.float32)
        if template is None
        else np.asarray(template, dtype=np.float32)
    )
    image_maps: dict[tuple[int, int], np.ndarray] = {}
    amplitudes = np.zeros((n_groups, NUM_TARGETS), dtype=np.float32)
    dense_maps: list | None = [] if store_dense_meta else None
    dense_waves: list | None = [] if store_dense_meta else None

    voxelize_s = 0.0
    model_s = 0.0
    materialize_s = 0.0
    t_scale = np.float32(1.0 / float(target_scale))

    for start in range(0, n_groups, bs):
        stop = min(n_groups, start + bs)

        t0 = time.perf_counter()
        if use_cuda_voxel:
            batch_voxels = voxelize_batch_cuda(grouped, start, stop, device=device)
            _sync_if_cuda(device)
        else:
            batch_voxels = voxelize_batch_cpu(grouped, start, stop)
        voxelize_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        batch_tpcs = np.asarray(grouped.group_tpcs[start:stop], dtype=np.int64)
        with torch.inference_mode():
            if isinstance(batch_voxels, torch.Tensor):
                xt = batch_voxels.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                xt = torch.from_numpy(np.asarray(batch_voxels, dtype=np.float32)).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
            tt = torch.from_numpy(batch_tpcs).to(device=device, dtype=torch.long, non_blocking=True)
            out = model(xt, tt)
            if isinstance(out, dict):
                out = out["final"]
        _sync_if_cuda(device)
        model_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        batch_amps = out.float().cpu().numpy().astype(np.float32, copy=False) * t_scale
        if raw_clip is not None:
            np.clip(batch_amps, float(raw_clip[0]), float(raw_clip[1]), out=batch_amps)
        if min_prediction_threshold is not None:
            batch_amps[batch_amps < float(min_prediction_threshold)] = 0.0

        amplitudes[start:stop] = batch_amps
        batch_wave = batch_amps[:, :, None] * tmpl[None, None, :]
        for local, group_id in enumerate(range(start, stop)):
            image_maps[
                (int(grouped.group_cls[group_id]), int(grouped.group_tpcs[group_id]))
            ] = batch_wave[local].copy()

        if store_dense_meta:
            if isinstance(batch_voxels, torch.Tensor):
                dense_maps.append(batch_voxels.detach().cpu().numpy().astype(np.float32, copy=False))  # type: ignore[union-attr]
            else:
                dense_maps.append(np.asarray(batch_voxels, dtype=np.float32))  # type: ignore[union-attr]
            dense_waves.append(np.asarray(batch_wave, dtype=np.float32))  # type: ignore[union-attr]
        materialize_s += time.perf_counter() - t0

    if store_dense_meta:
        maps_4d = np.concatenate(dense_maps, axis=0) if dense_maps else np.zeros((0, 1, NX, NY, NZ), dtype=np.float32)  # type: ignore[arg-type]
        waveforms = np.concatenate(dense_waves, axis=0) if dense_waves else np.zeros((0, NUM_TARGETS, wave_len), dtype=np.float32)  # type: ignore[arg-type]
    else:
        maps_4d = None
        waveforms = None

    return image_maps, {
        "group_cls": np.asarray(grouped.group_cls, dtype=int),
        "group_tpcs": np.asarray(grouped.group_tpcs, dtype=int),
        "maps_4d": maps_4d,
        "amplitudes": amplitudes,
        "waveforms": waveforms,
        "streaming": True,
        "timings": {
            "grouping_s": float(timing_group),
            "voxelize_s": float(voxelize_s),
            "model_s": float(model_s),
            "materialize_s": float(materialize_s),
            "total_s": float(time.perf_counter() - t_total),
            "n_groups": int(n_groups),
            "batch_size": int(bs),
            "cuda_graph": False,
            "voxelize_device": "cuda" if use_cuda_voxel else "cpu",
            "mixed_precision": False,
            "amp_dtype": "none",
        },
    }


__all__ = ["process_clusters_to_imageMaps_fast"]
