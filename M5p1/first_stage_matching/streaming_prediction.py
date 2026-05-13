from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .paths import configure_paths

configure_paths()

from ML_NDfull_perceiver import DEFAULT_TPC_YAML, NUM_TARGETS, NX, NY, NZ, load_tpc_geometries


@dataclass(slots=True)
class GroupedVoxelHits:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    energy: np.ndarray
    tpc_ids: np.ndarray
    labels: np.ndarray
    group_idx: np.ndarray
    group_cls: np.ndarray
    group_tpcs: np.ndarray
    group_offsets: np.ndarray
    hit_order: np.ndarray
    voxel_lin: np.ndarray


def _model_device(model: torch.nn.Module, device_policy: str = "auto") -> torch.device:
    if str(device_policy) == "force_cpu":
        return torch.device("cpu")
    for param in model.parameters():
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _amp_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError("amp_dtype must be 'bf16' or 'fp16'.")


def build_grouped_voxel_hits(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    tpc_ids: np.ndarray,
    labels: np.ndarray,
    *,
    include_noise: bool = False,
    restrict_clusters: set[int] | None = None,
    yaml_path: str | Path = DEFAULT_TPC_YAML,
) -> GroupedVoxelHits:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    e_arr = np.asarray(energy, dtype=np.float32)
    tpc_arr = np.asarray(tpc_ids, dtype=np.int64)
    lab_arr = np.asarray(labels, dtype=np.int64)

    valid = np.ones(lab_arr.shape[0], dtype=bool) if include_noise else (lab_arr >= 0)
    if restrict_clusters is not None:
        valid &= np.isin(lab_arr, np.asarray(sorted(int(v) for v in restrict_clusters), dtype=np.int64))

    if not np.any(valid):
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float32)
        return GroupedVoxelHits(
            x=np.zeros((0,), dtype=np.float64),
            y=np.zeros((0,), dtype=np.float64),
            z=np.zeros((0,), dtype=np.float64),
            energy=empty_f,
            tpc_ids=empty_i,
            labels=empty_i,
            group_idx=empty_i,
            group_cls=empty_i,
            group_tpcs=empty_i,
            group_offsets=np.zeros((1,), dtype=np.int64),
            hit_order=empty_i,
            voxel_lin=empty_i,
        )

    xv = x_arr[valid]
    yv = y_arr[valid]
    zv = z_arr[valid]
    ev = e_arr[valid]
    tpc = tpc_arr[valid]
    lab = lab_arr[valid]

    max_tpc = 128
    keys = lab * max_tpc + tpc
    _, first_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    n_groups = int(first_idx.size)

    # Match the existing process_clusters_to_imageMaps ordering: first time a
    # (label, tpc) pair appears in the hit stream.
    order = np.argsort(first_idx)
    rank = np.empty(n_groups, dtype=np.int64)
    rank[order] = np.arange(n_groups, dtype=np.int64)
    group_idx = rank[inv].astype(np.int64, copy=False)
    first_pos = first_idx[order]
    group_cls = lab[first_pos].astype(np.int64, copy=False)
    group_tpcs = tpc[first_pos].astype(np.int64, copy=False)

    counts = np.bincount(group_idx, minlength=n_groups).astype(np.int64, copy=False)
    group_offsets = np.empty(n_groups + 1, dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(counts, out=group_offsets[1:])
    hit_order = np.argsort(group_idx, kind="stable").astype(np.int64, copy=False)

    geom_map = load_tpc_geometries(yaml_path)
    voxel_lin = np.full(xv.shape[0], -1, dtype=np.int64)
    for tpc_val in np.unique(tpc):
        geom = geom_map.get(int(tpc_val))
        if geom is None:
            warnings.warn(f"Unknown charge TPC id {int(tpc_val)} - skipping.", RuntimeWarning)
            continue
        sel = tpc == int(tpc_val)
        ix = np.clip(((xv[sel] - geom.x_min) * geom.inv_dx).astype(np.int32), 0, NX - 1)
        iy = np.clip(((yv[sel] - geom.y_min) * geom.inv_dy).astype(np.int32), 0, NY - 1)
        iz = np.clip(((zv[sel] - geom.z_min) * geom.inv_dz).astype(np.int32), 0, NZ - 1)
        voxel_lin[sel] = (
            ix.astype(np.int64) * (NY * NZ)
            + iy.astype(np.int64) * NZ
            + iz.astype(np.int64)
        )

    return GroupedVoxelHits(
        x=xv,
        y=yv,
        z=zv,
        energy=ev,
        tpc_ids=tpc,
        labels=lab,
        group_idx=group_idx,
        group_cls=group_cls,
        group_tpcs=group_tpcs,
        group_offsets=group_offsets,
        hit_order=hit_order,
        voxel_lin=voxel_lin,
    )


def _batch_hit_indices(grouped: GroupedVoxelHits, start: int, stop: int) -> np.ndarray:
    lo = int(grouped.group_offsets[int(start)])
    hi = int(grouped.group_offsets[int(stop)])
    idx = np.asarray(grouped.hit_order[lo:hi], dtype=np.int64)
    return idx[np.asarray(grouped.voxel_lin[idx], dtype=np.int64) >= 0]


def voxelize_batch_cpu(grouped: GroupedVoxelHits, start: int, stop: int) -> np.ndarray:
    batch_size = int(stop) - int(start)
    maps = np.zeros((batch_size, 1, NX, NY, NZ), dtype=np.float32)
    idx = _batch_hit_indices(grouped, start, stop)
    if idx.size == 0:
        return maps

    local_group = np.asarray(grouped.group_idx[idx], dtype=np.int64) - int(start)
    flat_idx = local_group * (NX * NY * NZ) + np.asarray(grouped.voxel_lin[idx], dtype=np.int64)
    np.add.at(maps.reshape(-1), flat_idx, np.asarray(grouped.energy[idx], dtype=np.float32))
    return maps


def voxelize_batch_cuda(
    grouped: GroupedVoxelHits,
    start: int,
    stop: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(stop) - int(start)
    flat = torch.zeros(batch_size * NX * NY * NZ, device=device, dtype=torch.float32)
    idx = _batch_hit_indices(grouped, start, stop)
    if idx.size == 0:
        return flat.view(batch_size, 1, NX, NY, NZ)

    local_group = np.asarray(grouped.group_idx[idx], dtype=np.int64) - int(start)
    flat_idx = local_group * (NX * NY * NZ) + np.asarray(grouped.voxel_lin[idx], dtype=np.int64)
    scatter_idx = torch.from_numpy(flat_idx).to(device=device, dtype=torch.long, non_blocking=True)
    scatter_val = torch.from_numpy(np.asarray(grouped.energy[idx], dtype=np.float32)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    flat.scatter_add_(0, scatter_idx, scatter_val)
    return flat.view(batch_size, 1, NX, NY, NZ)


@torch.inference_mode()
def predict_amplitudes_from_voxels(
    voxels: np.ndarray | torch.Tensor,
    model: torch.nn.Module,
    tpc_ids: np.ndarray,
    *,
    target_scale: float = 1e-3,
    raw_clip: tuple[float, float] | None = (0.0, 60780.0),
    min_prediction_threshold: float | None = 100.0,
    device: torch.device | None = None,
    use_mixed_precision: bool = False,
    amp_dtype: str = "bf16",
) -> np.ndarray:
    if device is None:
        device = _model_device(model)

    if isinstance(voxels, torch.Tensor):
        xt = voxels.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        xt = torch.from_numpy(np.asarray(voxels, dtype=np.float32)).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    tt = torch.from_numpy(np.asarray(tpc_ids, dtype=np.int64)).to(device=device, dtype=torch.long, non_blocking=True)

    autocast_enabled = bool(use_mixed_precision) and device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=_amp_dtype(amp_dtype), enabled=autocast_enabled):
        out = model(xt, tt)
        if isinstance(out, dict):
            out = out["final"]

    amps = out.detach().float().cpu().numpy().astype(np.float32, copy=False)
    amps *= np.float32(1.0 / float(target_scale))
    if raw_clip is not None:
        np.clip(amps, float(raw_clip[0]), float(raw_clip[1]), out=amps)
    if min_prediction_threshold is not None:
        amps[amps < float(min_prediction_threshold)] = 0.0
    return amps


def process_clusters_to_imageMaps_streaming(
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
    batch_size: int = 8,
    raw_clip: tuple[float, float] | None = (0.0, 60780.0),
    min_prediction_threshold: float | None = 100.0,
    yaml_path: str | Path = DEFAULT_TPC_YAML,
    device_policy: str = "auto",
    voxelize_device: str = "auto",
    use_mixed_precision: bool = False,
    amp_dtype: str = "bf16",
    store_dense_meta: bool = False,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """
    Streaming equivalent of process_clusters_to_imageMaps.

    The dense 3D voxel tensor is built one inference batch at a time, so peak
    memory is O(batch_size) groups instead of O(all groups). The returned
    imageMaps are still full (120, waveform_len) arrays for every group.
    """
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    grouped = build_grouped_voxel_hits(
        x,
        y,
        z,
        energy,
        tpc_ids,
        labels,
        include_noise=include_noise,
        yaml_path=yaml_path,
    )
    timing_group = time.perf_counter() - t0

    n_groups = int(grouped.group_cls.size)
    wave_len = int(template.size) if template is not None else 1000
    if n_groups == 0:
        empty = {
            "group_cls": np.array([], dtype=int),
            "group_tpcs": np.array([], dtype=int),
            "maps_4d": np.zeros((0, 1, NX, NY, NZ), dtype=np.float32) if store_dense_meta else None,
            "amplitudes": np.zeros((0, NUM_TARGETS), dtype=np.float32),
            "waveforms": np.zeros((0, NUM_TARGETS, wave_len), dtype=np.float32) if store_dense_meta else None,
            "timings": {
                "grouping_s": float(timing_group),
                "voxelize_s": 0.0,
                "model_s": 0.0,
                "materialize_s": 0.0,
                "total_s": float(time.perf_counter() - t_total),
            },
        }
        return {}, empty

    device = _model_device(model, device_policy=device_policy)
    model = model.to(device).eval()
    use_cuda_voxel = (
        str(voxelize_device).lower() in {"cuda", "gpu"}
        or (str(voxelize_device).lower() == "auto" and device.type == "cuda")
    )
    if use_cuda_voxel and device.type != "cuda":
        use_cuda_voxel = False

    tmpl = np.ones(wave_len, dtype=np.float32) if template is None else np.asarray(template, dtype=np.float32)
    image_maps: dict[tuple[int, int], np.ndarray] = {}
    amplitudes = np.zeros((n_groups, NUM_TARGETS), dtype=np.float32)
    dense_maps = [] if store_dense_meta else None
    dense_waves = [] if store_dense_meta else None

    voxelize_s = 0.0
    model_s = 0.0
    materialize_s = 0.0
    bs = max(1, int(batch_size))

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
        batch_amps = predict_amplitudes_from_voxels(
            batch_voxels,
            model,
            batch_tpcs,
            target_scale=target_scale,
            raw_clip=raw_clip,
            min_prediction_threshold=min_prediction_threshold,
            device=device,
            use_mixed_precision=use_mixed_precision,
            amp_dtype=amp_dtype,
        )
        _sync_if_cuda(device)
        model_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        amplitudes[start:stop] = batch_amps
        batch_wave = batch_amps[:, :, None] * tmpl[None, None, :]
        for local, group_id in enumerate(range(start, stop)):
            image_maps[(int(grouped.group_cls[group_id]), int(grouped.group_tpcs[group_id]))] = np.asarray(
                batch_wave[local],
                dtype=np.float32,
            ).copy()
        if store_dense_meta:
            if isinstance(batch_voxels, torch.Tensor):
                dense_maps.append(batch_voxels.detach().cpu().numpy().astype(np.float32, copy=False))
            else:
                dense_maps.append(np.asarray(batch_voxels, dtype=np.float32))
            dense_waves.append(np.asarray(batch_wave, dtype=np.float32))
        materialize_s += time.perf_counter() - t0

    if store_dense_meta:
        maps_4d = np.concatenate(dense_maps, axis=0) if dense_maps else np.zeros((0, 1, NX, NY, NZ), dtype=np.float32)
        waveforms = np.concatenate(dense_waves, axis=0) if dense_waves else np.zeros((0, NUM_TARGETS, wave_len), dtype=np.float32)
    else:
        maps_4d = None
        waveforms = None

    meta = {
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
            "voxelize_device": "cuda" if use_cuda_voxel else "cpu",
            "mixed_precision": bool(use_mixed_precision),
            "amp_dtype": str(amp_dtype),
        },
    }
    return image_maps, meta


__all__ = [
    "GroupedVoxelHits",
    "build_grouped_voxel_hits",
    "voxelize_batch_cpu",
    "voxelize_batch_cuda",
    "predict_amplitudes_from_voxels",
    "process_clusters_to_imageMaps_streaming",
]
