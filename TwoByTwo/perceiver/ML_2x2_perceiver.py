"""
ML_2x2_perceiver.py
===================
Inference toolbox for 2x2-demonstrator 48-channel light prediction using the
HybridPerceiver3D architecture (shared definition in perceiver3d.py).

Counterpart to ML_NDfull_perceiver.py with identical public API, so code
written against the ND-LAr toolbox ports to the 2x2 by swapping the import.

Geometry (matches prep_2x2.py exactly)
--------------------------------------
  Voxel grid 32×128×64 (x, y, z) per TPC, 1 cm bins:
    x : per-TPC 32-cm drift ranges (_X_RANGE_BY_TPC)
    y : [-64, 64) cm
    z : [0, 64) cm for TPCs {0,1,4,5}, [-64, 0) cm for TPCs {2,3,6,7}
  48 light channels per TPC, ordered (side 0, y_rel 0..23) then
  (side 1, y_rel 0..23).

  NOTE on TPC ids: in the 2x2, hits are assigned to a TPC via
  io_group - 1 == TPCid where TPCid is the SAME id used on the light side
  (see prep_2x2.py). There is no even/odd swap like in ND-LAr full —
  light_tpc_to_charge_tpc here is the identity, kept only for API parity.

Public API
----------
  load_2x2_model(pt_path, ...)          → (model, meta)
                                          (alias: load_perceiver_model)
  predict_phi(maps_4d, model, tpc_ids)  → (G, 48) raw-ADC amplitudes
  group_voxelize_pairs(x, y, z, E, tpc_ids, labels)
                                        → (maps_4d [G×1×32×128×64], cls, tpcs)
  process_clusters_to_imageMaps(...)    → imageMaps[(cluster_id, tpc_id)] → (48, L)
  build_image_map_dict(wave, group_cls, group_tpcs)

Example
-------
  import ML_2x2_perceiver as ml2

  model, meta = ml2.load_2x2_model("runs/2x2_run/best_model.pt")
  imageMaps, info = ml2.process_clusters_to_imageMaps(
      x, y, z, E, tpc_ids, labels, model=model
  )
  wvfm = imageMaps[(cluster_id, tpc_id)]   # (48, 1000)
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from perceiver3d import (
    HybridPerceiver3D, load_checkpoint_state, build_model_from_state,
)

# ─────────────────────────────────────────────────────────────────────────────
# Geometry (must match prep_2x2.py)
# ─────────────────────────────────────────────────────────────────────────────

NX, NY, NZ  = 32, 128, 64
NUM_TARGETS = 48

_X_RANGE_BY_TPC = {
    0: ( 32.5,  64.5), 2: ( 32.5,  64.5),
    1: (  2.5,  34.5), 3: (  2.5,  34.5),
    4: (-34.5,  -2.5), 6: (-34.5,  -2.5),
    5: (-64.5, -32.5), 7: (-64.5, -32.5),
}
_POSZ_TPCS = {0, 1, 4, 5}
_NEGZ_TPCS = {2, 3, 6, 7}


def _bin_y(y: np.ndarray) -> np.ndarray:
    idx = np.floor(y + 64.0).astype(np.int32)  # [-64,64) -> [0..127]
    return np.clip(idx, 0, NY - 1)


def _bin_z(z: np.ndarray, tpc_id: int) -> np.ndarray:
    if tpc_id in _POSZ_TPCS:
        idx = np.floor(z).astype(np.int32)           # [0,64) -> [0..63]
    elif tpc_id in _NEGZ_TPCS:
        idx = np.floor(z + 64.0).astype(np.int32)    # [-64,0) -> [0..63]
    else:
        raise ValueError(f"Unknown TPC id: {tpc_id}")
    return np.clip(idx, 0, NZ - 1)


def _bin_x(x: np.ndarray, tpc_id: int) -> np.ndarray:
    x_lo, _ = _X_RANGE_BY_TPC[tpc_id]
    idx = np.floor(x - x_lo).astype(np.int32)
    return np.clip(idx, 0, NX - 1)


def group_voxelize_pairs(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_ids: np.ndarray, labels: np.ndarray,
    *, include_noise: bool = False,
    restrict_clusters=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized voxelization per (cluster_id, tpc_id) pair.
    Returns (maps_4d [G,1,NX,NY,NZ], group_cls, group_tpcs).
    tpc_ids are 2x2 TPC ids (0–7, = io_group - 1).
    """
    x  = np.asarray(x, np.float64); y  = np.asarray(y, np.float64)
    z  = np.asarray(z, np.float64); E  = np.asarray(E, np.float32)
    tpc_ids = np.asarray(tpc_ids, np.int64); labels = np.asarray(labels, np.int64)
    n = x.shape[0]

    valid = (labels >= 0) if not include_noise else np.ones(n, bool)
    if restrict_clusters is not None:
        valid &= np.isin(labels, np.asarray(list(restrict_clusters), np.int64))
    if not np.any(valid):
        return (np.zeros((0, 1, NX, NY, NZ), np.float32),
                np.zeros((0,), int), np.zeros((0,), int))

    xv = x[valid]; yv = y[valid]; zv = z[valid]; Ev = E[valid]
    lab = labels[valid]; tpc = tpc_ids[valid]

    MAX_TPC = 16
    keys = lab * MAX_TPC + tpc
    uniq_keys, first_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    G = int(uniq_keys.size)
    order = np.argsort(first_idx)
    rank = np.empty(G, dtype=np.int64); rank[order] = np.arange(G, dtype=np.int64)
    g = rank[inv]
    pos = first_idx[order]
    group_cls  = lab[pos].astype(int)
    group_tpcs = tpc[pos].astype(int)

    maps_flat = np.zeros(G * NX * NY * NZ, np.float32)
    V = NX * NY * NZ

    for tpc_val in np.unique(tpc):
        t = int(tpc_val)
        if t not in _X_RANGE_BY_TPC:
            warnings.warn(f"Unknown 2x2 TPC id {t} — skipping.", RuntimeWarning)
            continue
        sel = (tpc == tpc_val)
        xs = xv[sel]; ys = yv[sel]; zs = zv[sel]; Es = Ev[sel]; gs = g[sel]
        ix = _bin_x(xs, t); iy = _bin_y(ys); iz = _bin_z(zs, t)
        lin = ix.astype(np.int64) * (NY * NZ) + iy.astype(np.int64) * NZ + iz.astype(np.int64)
        np.add.at(maps_flat, gs.astype(np.int64) * V + lin, Es)

    maps_4d = maps_flat.reshape(G, NX, NY, NZ)[:, None, :, :, :]
    return maps_4d, group_cls, group_tpcs


def light_tpc_to_charge_tpc(light_tpc_id: int) -> int:
    """Identity in the 2x2 (kept for API parity with ML_NDfull_perceiver)."""
    return light_tpc_id

def charge_tpc_to_light_tpc(charge_tpc_id: int) -> int:
    return charge_tpc_id


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_2x2_model(
    pt_path: str,
    *,
    base_channels: Optional[int] = None,
    embed_dim: Optional[int] = None,
    num_tpcs: Optional[int] = None,
    num_targets: Optional[int] = None,
    num_decoder_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    dropout: float = 0.0,           # 0 at inference
    device: Optional[str] = None,
    target_scale: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a trained 2x2 HybridPerceiver3D checkpoint (train_perceiver.py
    --detector 2x2). Architecture hyper-parameters are read from the stored
    config or inferred from tensor shapes; pass explicit values only to
    override.

    Returns (model.eval(), meta).
    """
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))

    state, ck_meta = load_checkpoint_state(pt_path)
    model, arch = build_model_from_state(
        state,
        config=ck_meta.get("config"),
        dropout=dropout,
        base_channels=base_channels,
        embed_dim=embed_dim,
        num_tpcs=num_tpcs,
        num_targets=num_targets,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
    )
    model = model.to(dev).eval()

    meta: Dict[str, Any] = {
        "out_dim":      arch["num_targets"],
        "target_scale": target_scale,
        "device":       str(dev),
        "arch":         arch,
        "checkpoint":   ck_meta,
    }
    return model, meta


# API-parity alias: code written against ML_NDfull_perceiver can swap the
# import and keep calling load_perceiver_model.
load_perceiver_model = load_2x2_model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _is_cuda_oom(exc: Exception) -> bool:
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError): return True
    except Exception: pass
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


@torch.no_grad()
def predict_phi(
    maps_4d: np.ndarray,            # (G, 1, NX, NY, NZ)
    model: nn.Module,
    tpc_ids: np.ndarray,            # (G,) int, 2x2 TPC ids (0–7)
    *,
    target_scale: float = 1e-3,
    batch_size: int = 16,           # tiny model + small grid → larger batches
    raw_clip: Optional[Tuple[float, float]] = None,
    min_prediction_threshold: Optional[float] = 100.0,
    device_policy: str = "auto",
) -> np.ndarray:
    """
    Batched inference → (G, 48) float32 in raw ADC units.

    raw_clip defaults to None (no clipping); pass e.g. (0.0, 65535.0) to
    clamp to the ADC range of your light readout.
    """
    G = maps_4d.shape[0]
    if G == 0:
        return np.zeros((0, NUM_TARGETS), np.float32)

    x = np.asarray(maps_4d, np.float32)
    tids = np.asarray(tpc_ids, np.int64)

    if device_policy == "force_cpu":
        cur_dev = torch.device("cpu")
    else:
        for p in model.parameters():
            cur_dev = p.device; break
        else:
            cur_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(cur_dev).eval()
    preds = np.zeros((G, NUM_TARGETS), np.float32)

    for start in range(0, G, batch_size):
        sl = slice(start, start + batch_size)
        while True:
            try:
                xt = torch.from_numpy(x[sl]).to(cur_dev, torch.float32, non_blocking=True)
                tt = torch.from_numpy(tids[sl]).to(cur_dev, torch.long)
                out = model(xt, tt)
                if isinstance(out, dict): out = out["final"]
                preds[sl] = out.detach().cpu().float().numpy()
                break
            except Exception as e:
                if cur_dev.type == "cuda" and device_policy == "auto" and _is_cuda_oom(e):
                    warnings.warn("predict_phi: CUDA OOM — falling back to CPU.", RuntimeWarning)
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                    cur_dev = torch.device("cpu"); model = model.to(cur_dev).eval()
                    continue
                raise

    preds_raw = preds * (1.0 / target_scale)
    if raw_clip is not None:
        np.clip(preds_raw, raw_clip[0], raw_clip[1], out=preds_raw)
    if min_prediction_threshold is not None:
        preds_raw[preds_raw < float(min_prediction_threshold)] = 0.0
    return preds_raw


# ─────────────────────────────────────────────────────────────────────────────
# Packaging
# ─────────────────────────────────────────────────────────────────────────────

def build_image_map_dict(
    wave: np.ndarray,           # (G, K, L)
    group_cls: np.ndarray,
    group_tpcs: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Map (cluster_id, tpc_id) → (48, L) waveform."""
    return {
        (int(c), int(t)): wave[g]
        for g, (c, t) in enumerate(zip(group_cls, group_tpcs))
    }


def process_clusters_to_imageMaps(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_ids: np.ndarray, labels: np.ndarray,
    *,
    model: nn.Module,
    target_scale: float = 1e-3,
    template: Optional[np.ndarray] = None,
    include_noise: bool = False,
    batch_size: int = 16,
    raw_clip: Optional[Tuple[float, float]] = None,
    min_prediction_threshold: Optional[float] = 100.0,
    device_policy: str = "auto",
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, Any]]:
    """
    Full pipeline: hits → imageMaps[(cluster_id, tpc_id)] → (48, L).

    Parameters
    ----------
    x, y, z, E    : hit arrays in cm / MeV (2x2 module coordinates)
    tpc_ids       : 2x2 TPC ids (0–7, = io_group - 1) per hit
    labels        : cluster IDs per hit (-1 = noise, excluded by default)
    model         : loaded HybridPerceiver3D (from load_2x2_model)
    template      : 1D pulse-shape array. None → flat ones(1000).
    """
    maps_4d, group_cls, group_tpcs = group_voxelize_pairs(
        x, y, z, E, tpc_ids, labels, include_noise=include_noise,
    )
    L = int(template.size) if template is not None else 1000

    if maps_4d.shape[0] == 0:
        empty = {
            "group_cls":  np.array([], int), "group_tpcs": np.array([], int),
            "maps_4d":    np.zeros((0, 1, NX, NY, NZ), np.float32),
            "amplitudes": np.zeros((0, NUM_TARGETS),    np.float32),
            "waveforms":  np.zeros((0, NUM_TARGETS, L), np.float32),
        }
        return {}, empty

    amps = predict_phi(
        maps_4d, model, group_tpcs,
        target_scale=target_scale, batch_size=batch_size,
        raw_clip=raw_clip,
        min_prediction_threshold=min_prediction_threshold,
        device_policy=device_policy,
    )

    tmpl = np.ones(1000, np.float32) if template is None else np.asarray(template, np.float32)
    wave = amps[:, :, None] * tmpl[None, None, :]    # (G, 48, L)

    imageMaps = build_image_map_dict(wave, group_cls, group_tpcs)
    meta = {
        "group_cls":  group_cls, "group_tpcs": group_tpcs,
        "maps_4d":    maps_4d,   "amplitudes": amps, "waveforms": wave,
    }
    return imageMaps, meta


# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "NX", "NY", "NZ", "NUM_TARGETS",
    "group_voxelize_pairs",
    "HybridPerceiver3D", "load_2x2_model", "load_perceiver_model",
    "predict_phi",
    "build_image_map_dict", "process_clusters_to_imageMaps",
    "light_tpc_to_charge_tpc", "charge_tpc_to_light_tpc",
]
