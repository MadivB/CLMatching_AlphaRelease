"""
ML_NDfull.py
============
Importable inference toolbox for NDLAr-full 120-channel light prediction.
Mirrors the API of ML_3DCNN_v3_1.py (used for the 2x2 detector) but adapted
for ND-full geometry and the LatentCNN architecture.

Key differences from ML_3DCNN_v3_1.py (2x2):
  - Geometry    : 70 TPCs (0–69), 50×300×100 voxels per TPC in physical UNITS
                  Bounds loaded from tpc_boundaries.yaml (not hardcoded)
  - Outputs     : 120 light channels per TPC (vs. 48 for 2x2)
  - Model       : LatentCNN (3D CNN → latent z → channel decoder)
  - TPC IDs     : always use CHARGE TPC IDs (even-odd swapped from light TPC IDs)

Public API
----------
  load_latent_model(pt_path, ...)
      Load a trained LatentCNN checkpoint; returns (model, meta).

  predict_phi(maps_4d, model, tpc_ids, ...)
      Batched prediction → (G, 120) amplitude array in raw units.

  group_voxelize_pairs(x, y, z, E, tpc_ids, labels, ...)
      Vectorized voxelization per (cluster_id, charge_tpc_id) pair.
      Returns (maps_4d [G×1×50×300×100], group_cls, group_tpcs).

  process_clusters_to_imageMaps(x, y, z, E, tpc_ids, labels, model, ...)
      One-call pipeline:
        voxelize → predict → expand to (K, L) waveform images
        Returns imageMaps[(cluster_id, charge_tpc_id)] → np.ndarray (120, L)
        and meta dict.

  build_image_map_dict(wave, group_cls, group_tpcs)
      Map arrays to the {(cluster_id, tpc_id): (K, L)} dict.

Geometry
--------
  TPC boundaries are read from DEFAULT_TPC_YAML (can be overridden).
  Call load_tpc_geometries(yaml_path) to get {tpc_id: TPCGeometry}.
  The module-level _TPC_GEOM cache is populated lazily on first use.

Example usage
-------------
  import ML_NDfull as mlnd

  model, meta = mlnd.load_latent_model("runs/latent_run/best_model.pt")

  imageMaps, info = mlnd.process_clusters_to_imageMaps(
      x, y, z, E, tpc_ids, labels, model=model
  )
  wvfm_120x1000 = imageMaps[(cluster_id, charge_tpc_id)]  # (120, 1000)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any, Iterable

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

NX, NY, NZ = 50, 300, 100          # voxel grid dimensions (same as training)
NUM_TARGETS = 120                   # light channels per TPC

DEFAULT_TPC_YAML = Path(
    "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/samplePreparation/tpc_boundaries.yaml"
)


@dataclass(frozen=True)
class TPCGeometry:
    tpc_id: int
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float
    inv_dx: float; inv_dy: float; inv_dz: float   # 1/(bin width)


def load_tpc_geometries(path: str | Path = DEFAULT_TPC_YAML) -> Dict[int, TPCGeometry]:
    """
    Load per-TPC boundaries from the YAML file and compute voxel bin sizes.
    Each TPC has its own x/y/z extent in physical units (mm or cm).
    The voxel grid is always 50×300×100 regardless of the physical extent.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"TPC boundaries YAML not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f)

    out: Dict[int, TPCGeometry] = {}
    for k, v in raw.items():
        t = int(k)
        xlo, xhi = float(v["x_min"]), float(v["x_max"])
        ylo, yhi = float(v["y_min"]), float(v["y_max"])
        zlo, zhi = float(v["z_min"]), float(v["z_max"])
        inv_dx = NX / (xhi - xlo)
        inv_dy = NY / (yhi - ylo)
        inv_dz = NZ / (zhi - zlo)
        out[t] = TPCGeometry(
            tpc_id=t,
            x_min=xlo, x_max=xhi,
            y_min=ylo, y_max=yhi,
            z_min=zlo, z_max=zhi,
            inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        )
    return out


# Module-level cache (populated lazily)
_TPC_GEOM_CACHE: Optional[Dict[int, TPCGeometry]] = None

def _get_geom_cache(yaml_path: str | Path = DEFAULT_TPC_YAML) -> Dict[int, TPCGeometry]:
    global _TPC_GEOM_CACHE
    if _TPC_GEOM_CACHE is None:
        _TPC_GEOM_CACHE = load_tpc_geometries(yaml_path)
    return _TPC_GEOM_CACHE


def voxelize_hits_for_tpc(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_id: int,
    geom: Optional[TPCGeometry] = None,
    yaml_path: str | Path = DEFAULT_TPC_YAML,
) -> np.ndarray:
    """
    Voxelize (x, y, z, E) hits for a SINGLE TPC into a (NX, NY, NZ) grid.
    Hits outside the TPC boundary are silently clipped to the boundary voxel.
    Returns float32 array of shape (NX, NY, NZ).
    """
    if geom is None:
        geom = _get_geom_cache(yaml_path)[tpc_id]

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    E = np.asarray(E, dtype=np.float32)

    ix = np.clip(((x - geom.x_min) * geom.inv_dx).astype(np.int32), 0, NX - 1)
    iy = np.clip(((y - geom.y_min) * geom.inv_dy).astype(np.int32), 0, NY - 1)
    iz = np.clip(((z - geom.z_min) * geom.inv_dz).astype(np.int32), 0, NZ - 1)

    lin = ix.astype(np.int64) * (NY * NZ) + iy.astype(np.int64) * NZ + iz.astype(np.int64)
    acc = np.bincount(lin, weights=E, minlength=NX * NY * NZ).astype(np.float32)
    return acc.reshape(NX, NY, NZ)


def group_voxelize_pairs(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_ids: np.ndarray,
    labels: np.ndarray,
    *,
    include_noise: bool = False,
    restrict_clusters: Optional[Iterable[int]] = None,
    yaml_path: str | Path = DEFAULT_TPC_YAML,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one voxel map (1, NX, NY, NZ) for every (cluster_id, charge_tpc_id)
    pair present in the data.

    Parameters
    ----------
    x, y, z, E  : 1D arrays of hit positions and deposited energy
    tpc_ids      : 1D int array — CHARGE TPC IDs per hit (0..69 for ND-full)
    labels       : 1D int array — cluster/track assignment per hit
                   (-1 = noise, excluded by default)
    include_noise: include hits with label == -1 (as their own cluster=-1)
    restrict_clusters : if given, only include these cluster IDs

    Returns
    -------
    maps_4d     : (G, 1, NX, NY, NZ) float32
    group_cls   : (G,) int  — cluster ID for each group
    group_tpcs  : (G,) int  — charge TPC ID for each group
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    E = np.asarray(E, dtype=np.float32)
    tpc_ids = np.asarray(tpc_ids, dtype=np.int64)
    labels  = np.asarray(labels,  dtype=np.int64)

    n = x.shape[0]
    if not (y.shape[0] == z.shape[0] == E.shape[0] == tpc_ids.shape[0] == labels.shape[0] == n):
        raise ValueError("x, y, z, E, tpc_ids, labels must be same-length 1D arrays.")

    valid = (labels >= 0) if not include_noise else np.ones(n, dtype=bool)
    if restrict_clusters is not None:
        rc = np.asarray(list(restrict_clusters), dtype=np.int64)
        valid &= np.isin(labels, rc)

    if not np.any(valid):
        return (
            np.zeros((0, 1, NX, NY, NZ), dtype=np.float32),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    xv = x[valid]; yv = y[valid]; zv = z[valid]; Ev = E[valid]
    lab = labels[valid];  tpc = tpc_ids[valid]

    # Build unique (cluster_id, tpc_id) keys — use a large stride so keys are unique
    MAX_TPC = 128
    keys = lab * MAX_TPC + tpc
    uniq_keys, first_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    G = int(uniq_keys.size)

    # Preserve insertion order (first occurrence)
    order_by_first = np.argsort(first_idx)
    rank = np.empty(G, dtype=np.int64)
    rank[order_by_first] = np.arange(G, dtype=np.int64)
    g = rank[inv]   # (N_valid,) — group index per hit

    pos = first_idx[order_by_first]
    group_cls  = lab[pos].astype(int)
    group_tpcs = tpc[pos].astype(int)

    # Load geometry
    geom_map = _get_geom_cache(yaml_path)

    # Vectorised voxelization across all groups simultaneously
    maps_flat = np.zeros(G * NX * NY * NZ, dtype=np.float32)
    V = NX * NY * NZ

    for tpc_val in np.unique(tpc):
        sel = (tpc == tpc_val)
        geom = geom_map.get(int(tpc_val))
        if geom is None:
            warnings.warn(f"Unknown charge TPC id {int(tpc_val)} — skipping hits.", RuntimeWarning)
            continue

        xs = xv[sel]; ys = yv[sel]; zs = zv[sel]; Es = Ev[sel]; gs = g[sel]

        ix = np.clip(((xs - geom.x_min) * geom.inv_dx).astype(np.int32), 0, NX - 1)
        iy = np.clip(((ys - geom.y_min) * geom.inv_dy).astype(np.int32), 0, NY - 1)
        iz = np.clip(((zs - geom.z_min) * geom.inv_dz).astype(np.int32), 0, NZ - 1)

        lin_local  = ix.astype(np.int64)*(NY*NZ) + iy.astype(np.int64)*NZ + iz.astype(np.int64)
        lin_global = gs.astype(np.int64) * V + lin_local
        np.add.at(maps_flat, lin_global, Es)

    maps_4d = maps_flat.reshape(G, NX, NY, NZ)[:, None, :, :, :]   # (G,1,NX,NY,NZ)
    return maps_4d, group_cls, group_tpcs


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture (matches latentMLTrain.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnAct(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=1):
        super().__init__()
        gn = min(8, c_out)
        while c_out % gn != 0: gn -= 1
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(gn, c_out), nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class _LightChannelDecoder(nn.Module):
    def __init__(self, num_channels: int, latent_dim: int,
                 channel_emb_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_channels = num_channels
        self.channel_emb  = nn.Embedding(num_channels, channel_emb_dim)
        mlp_in = latent_dim + channel_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.register_buffer("ch_idx", torch.arange(num_channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0); K = self.num_channels
        e    = self.channel_emb(self.ch_idx)           # (K, emb_dim)
        z_ex = z.unsqueeze(1).expand(B, K, -1)        # (B, K, latent)
        e_ex = e.unsqueeze(0).expand(B, K, -1)        # (B, K, emb)
        inp  = torch.cat([z_ex, e_ex], dim=-1)        # (B, K, latent+emb)
        return self.mlp(inp.view(B * K, -1)).view(B, K)


class LatentCNN(nn.Module):
    """
    Charge encoder (3D CNN, 50×300×100 input) + light channel decoder.
    Architecture matches latentMLTrain.py exactly so checkpoints load cleanly.
    """
    def __init__(
        self,
        c_in: int = 1,
        num_targets: int = NUM_TARGETS,
        num_tpcs: int = 72,
        base_ch: int = 32,
        tpc_embed_dim: int = 16,
        dropout: float = 0.0,          # 0 at inference
        latent_dim: int = 32,
        channel_emb_dim: int = 32,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.stem  = nn.Sequential(_ConvBnAct(c_in, base_ch),
                                   _ConvBnAct(base_ch, base_ch))
        self.down1 = _ConvBnAct(base_ch,     base_ch * 2, stride=2)
        self.down2 = _ConvBnAct(base_ch * 2, base_ch * 4, stride=2)
        self.extra = _ConvBnAct(base_ch * 4, base_ch * 4)

        feat_dim = base_ch * 4
        self.tpc_emb = nn.Embedding(num_tpcs, tpc_embed_dim)
        proj_in = feat_dim + tpc_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(proj_in, proj_in), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_in, latent_dim),
        )
        self.decoder = _LightChannelDecoder(
            num_channels=num_targets, latent_dim=latent_dim,
            channel_emb_dim=channel_emb_dim, hidden_dim=decoder_hidden,
            dropout=dropout,
        )
        self.latent_dim = latent_dim

    @torch.no_grad()
    def encode(self, x: torch.Tensor, tpc_ids: torch.Tensor) -> torch.Tensor:
        x = self.stem(x); x = self.down1(x); x = self.down2(x); x = self.extra(x)
        x = x.mean(dim=[2, 3, 4])           # Global Average Pool → (B, feat_dim)
        e = self.tpc_emb(tpc_ids)
        h = torch.cat([x.float(), e], dim=-1)
        return self.proj(h)                  # (B, latent_dim)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, tpc_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x, tpc_ids))   # (B, 120)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_latent_model(
    pt_path: str,
    *,
    base_ch: int = 32,
    latent_dim: int = 32,
    channel_emb_dim: int = 32,
    decoder_hidden: int = 128,
    num_tpcs: int = 72,
    num_targets: int = NUM_TARGETS,
    device: Optional[str] = None,
    target_scale: float = 1e-3,        # kept in meta for callers
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a trained LatentCNN checkpoint from latentMLTrain.py.

    Returns
    -------
    (model.eval(), meta)
    meta = {
        "out_dim"       : 120,
        "latent_dim"    : 32,
        "target_scale"  : 1e-3,
        "checkpoint"    : {epoch, latent_dim, ...} (raw ckpt keys if present),
    }

    Tips
    ----
    - If train used --base-ch 32 --latent-dim 32 (defaults), no extra args needed.
    - To run on CPU (no GPU): pass device="cpu".
    """
    dev = torch.device(device if device is not None
                       else ("cuda" if torch.cuda.is_available() else "cpu"))

    ck = torch.load(pt_path, map_location="cpu")
    # Accept both a raw state_dict and a checkpoint dict with "model" key
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        ck_meta = {k: v for k, v in ck.items() if k != "model"}
        # Allow checkpoint to override latent_dim and channel_emb_dim
        if "latent_dim"      in ck: latent_dim      = int(ck["latent_dim"])
        if "channel_emb_dim" in ck: channel_emb_dim = int(ck["channel_emb_dim"])
    else:
        state = ck
        ck_meta = {}

    # Strip DDP 'module.' prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    model = LatentCNN(
        c_in=1, num_targets=num_targets, num_tpcs=num_tpcs,
        base_ch=base_ch, tpc_embed_dim=16, dropout=0.0,
        latent_dim=latent_dim, channel_emb_dim=channel_emb_dim,
        decoder_hidden=decoder_hidden,
    )
    incompat = model.load_state_dict(state, strict=True)
    model = model.to(dev).eval()

    meta: Dict[str, Any] = {
        "out_dim":      num_targets,
        "latent_dim":   latent_dim,
        "target_scale": target_scale,
        "device":       str(dev),
        "checkpoint":   ck_meta,
    }
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _is_cuda_oom(exc: Exception) -> bool:
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError): return True
    except Exception: pass
    msg = str(exc).lower()
    return ("cuda" in msg and "out of memory" in msg)


@torch.no_grad()
def predict_phi(
    maps_4d: np.ndarray,            # (G, 1, NX, NY, NZ) float32
    model: nn.Module,
    tpc_ids: np.ndarray,            # (G,) int — charge TPC IDs
    *,
    target_scale: float = 1e-3,
    batch_size: int = 16,
    raw_clip: Optional[Tuple[float, float]] = (0.0, 60780.0),
    device_policy: str = "auto",    # "auto" | "force_cpu"
) -> np.ndarray:
    """
    Run batched LatentCNN inference on pre-voxelized maps.

    Returns
    -------
    preds_raw : (G, 120) float32 array in RAW amplitude units (ADC counts).
    """
    G = maps_4d.shape[0]
    if G == 0:
        return np.zeros((0, NUM_TARGETS), dtype=np.float32)

    # Normalise input: per-sample max-normalisation (matches training __getitem__)
    x = np.asarray(maps_4d, dtype=np.float32)           # (G,1,NX,NY,NZ)
    for i in range(G):
        vmax = x[i].max()
        # if vmax > 1e-6: x[i] /= vmax

    tids = np.asarray(tpc_ids, dtype=np.int64)

    # Determine device
    if device_policy == "force_cpu":
        cur_dev = torch.device("cpu")
    else:
        for p in model.parameters():
            cur_dev = p.device; break
        else:
            cur_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(cur_dev).eval()
    preds_scaled = np.zeros((G, NUM_TARGETS), dtype=np.float32)

    for start in range(0, G, batch_size):
        sl = slice(start, start + batch_size)
        while True:
            try:
                xt = torch.from_numpy(x[sl]).to(cur_dev, dtype=torch.float32,
                                                non_blocking=(cur_dev.type=="cuda"))
                tt = torch.from_numpy(tids[sl]).to(cur_dev, dtype=torch.long)
                y  = model(xt, tt)          # (batch, 120)
                preds_scaled[sl] = y.detach().cpu().float().numpy()
                break
            except Exception as e:
                if cur_dev.type == "cuda" and device_policy == "auto" and _is_cuda_oom(e):
                    warnings.warn("predict_phi: CUDA OOM — falling back to CPU.", RuntimeWarning)
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                    cur_dev = torch.device("cpu"); model = model.to(cur_dev).eval()
                    continue
                raise

    # Convert from scaled units to raw units
    preds_raw = preds_scaled * (1.0 / target_scale)
    if raw_clip is not None:
        np.clip(preds_raw, raw_clip[0], raw_clip[1], out=preds_raw)
    return preds_raw


# ─────────────────────────────────────────────────────────────────────────────
# Waveform construction & packaging
# ─────────────────────────────────────────────────────────────────────────────

def build_image_map_dict(
    wave: np.ndarray,           # (G, K, L)
    group_cls: np.ndarray,      # (G,)
    group_tpcs: np.ndarray,     # (G,)
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Map (cluster_id, charge_tpc_id) → (K, L) waveform array.

    Usage:
        wvfm = imageMaps[(cluster_id, charge_tpc_id)]  # (120, 1000)
    """
    return {
        (int(c), int(t)): wave[g]
        for g, (c, t) in enumerate(zip(group_cls, group_tpcs))
    }


def process_clusters_to_imageMaps(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_ids: np.ndarray,
    labels: np.ndarray,
    *,
    model: nn.Module,
    target_scale: float = 1e-3,
    template: Optional[np.ndarray] = None,
    include_noise: bool = False,
    batch_size: int = 16,
    raw_clip: Tuple[float, float] = (0.0, 60780.0),
    yaml_path: str | Path = DEFAULT_TPC_YAML,
    device_policy: str = "auto",
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, Any]]:
    """
    Full pipeline from raw hits → waveform image maps.

    Steps
    -----
    1. Voxelize hits per (cluster_id, charge_tpc_id) pair
    2. Predict 120-channel peak amplitudes with LatentCNN
    3. Expand amplitudes to waveforms: amplitude × template  (K, L)
    4. Return dict keyed by (cluster_id, charge_tpc_id)

    Parameters
    ----------
    x, y, z, E    : hit arrays (physical units matching tpc_boundaries.yaml)
    tpc_ids       : CHARGE TPC IDs per hit (0–69 for ND-full)
    labels        : cluster/track IDs per hit (-1 = noise, excluded by default)
    model         : loaded LatentCNN (from load_latent_model)
    target_scale  : must match what was used during training (default 1e-3)
    template      : 1D array (L,) — pulse shape template.
                    If None, defaults to ones(1000) so output is a flat image.
                    Pass your actual SiPM pulse shape here for realistic waveforms.
    raw_clip      : (lo, hi) clamp for amplitude in raw ADC units
    yaml_path     : path to tpc_boundaries.yaml

    Returns
    -------
    imageMaps : Dict[(cluster_id, charge_tpc_id), np.ndarray (120, L)]
    meta      : dict with intermediate tensors for diagnostics:
                  "group_cls"  : (G,)
                  "group_tpcs" : (G,)
                  "maps_4d"    : (G, 1, NX, NY, NZ)
                  "amplitudes" : (G, 120) in raw ADC units
                  "waveforms"  : (G, 120, L)
    """
    maps_4d, group_cls, group_tpcs = group_voxelize_pairs(
        x, y, z, E, tpc_ids, labels,
        include_noise=include_noise, yaml_path=yaml_path,
    )

    L = int(template.size) if template is not None else 1000

    if maps_4d.shape[0] == 0:
        empty = {
            "group_cls":  np.array([], dtype=int),
            "group_tpcs": np.array([], dtype=int),
            "maps_4d":    np.zeros((0, 1, NX, NY, NZ), dtype=np.float32),
            "amplitudes": np.zeros((0, NUM_TARGETS),   dtype=np.float32),
            "waveforms":  np.zeros((0, NUM_TARGETS, L), dtype=np.float32),
        }
        return {}, empty

    # Predict amplitudes (G, 120) in raw ADC units
    amps = predict_phi(
        maps_4d, model, group_tpcs,
        target_scale=target_scale, batch_size=batch_size,
        raw_clip=raw_clip, device_policy=device_policy,
    )

    # Build waveforms: (G, 120, L) = amps (G,120,1) * template (1,1,L)
    if template is None:
        tmpl = np.ones(1000, dtype=np.float32)
    else:
        tmpl = np.asarray(template, dtype=np.float32)
        assert tmpl.ndim == 1, "template must be 1D"
    wave = amps[:, :, None] * tmpl[None, None, :]   # (G, 120, L)

    imageMaps = build_image_map_dict(wave, group_cls, group_tpcs)

    meta = {
        "group_cls":  group_cls,    # (G,)
        "group_tpcs": group_tpcs,   # (G,)
        "maps_4d":    maps_4d,      # (G, 1, NX, NY, NZ)
        "amplitudes": amps,         # (G, 120)
        "waveforms":  wave,         # (G, 120, L)
    }
    return imageMaps, meta


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: light TPC → charge TPC ID conversion (even-odd swap)
# ─────────────────────────────────────────────────────────────────────────────

def light_tpc_to_charge_tpc(light_tpc_id: int) -> int:
    """
    Convert a light-side TPC ID to the corresponding charge TPC ID.
    Convention used in compile_samples_to_zarr.py:
      even light → charge = light + 1
      odd  light → charge = light - 1
    """
    if light_tpc_id % 2 == 0:
        return light_tpc_id + 1
    else:
        return light_tpc_id - 1


def charge_tpc_to_light_tpc(charge_tpc_id: int) -> int:
    """Inverse of light_tpc_to_charge_tpc (same formula by symmetry)."""
    return light_tpc_to_charge_tpc(charge_tpc_id)


# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Geometry
    "NX", "NY", "NZ", "NUM_TARGETS", "DEFAULT_TPC_YAML",
    "TPCGeometry", "load_tpc_geometries",
    # Voxelization
    "voxelize_hits_for_tpc", "group_voxelize_pairs",
    # Model
    "LatentCNN", "load_latent_model",
    # Inference
    "predict_phi",
    # Pipeline
    "build_image_map_dict", "process_clusters_to_imageMaps",
    # TPC ID conversion
    "light_tpc_to_charge_tpc", "charge_tpc_to_light_tpc",
]
