"""
ML_NDfull_perceiver.py
======================
Inference toolbox for NDLAr-full 120-channel light prediction using the
HybridPerceiver3D architecture from train_ndfull.py.

Counterpart to ML_NDfull_latent.py (LatentCNN).

Architecture: 3D CNN stem with SE blocks → 3D sinusoidal positional encoding
→ TPC FiLM conditioning → Transformer decoder (120 learned queries) →
independent regression heads per target.

Public API (identical to ML_NDfull_latent.py for drop-in interoperability)
---------------------------------------------------------------------------
  load_perceiver_model(pt_path, ...)
      Load a trained HybridPerceiver3D checkpoint → (model, meta)

  predict_phi(maps_4d, model, tpc_ids, ...)
      Batched prediction → (G, 120) amplitude array in raw ADC units

  group_voxelize_pairs(x, y, z, E, tpc_ids, labels, ...)
      Vectorized voxelization per (cluster_id, charge_tpc_id) pair
      Returns (maps_4d [G×1×50×300×100], group_cls, group_tpcs)

  process_clusters_to_imageMaps(x, y, z, E, tpc_ids, labels, model, ...)
      Full pipeline → imageMaps[(cluster_id, charge_tpc_id)] → (120, L)

  build_image_map_dict(wave, group_cls, group_tpcs)

Geometry
--------
  TPC boundaries loaded from tpc_boundaries.yaml; 50×300×100 voxels per TPC.

Example
-------
  import ML_NDfull_perceiver as mlp

  model, meta = mlp.load_perceiver_model("runs/ndfull_run_distributed/best_model.pt")

  imageMaps, info = mlp.process_clusters_to_imageMaps(
      x, y, z, E, tpc_ids, labels, model=model
  )
  wvfm = imageMaps[(cluster_id, charge_tpc_id)]   # (120, 1000)
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterable

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Geometry (shared with ML_NDfull_latent.py)
# ─────────────────────────────────────────────────────────────────────────────

NX, NY, NZ   = 50, 300, 100
NUM_TARGETS  = 120

DEFAULT_TPC_YAML = Path(
    "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/samplePreparation/tpc_boundaries.yaml"
)


@dataclass(frozen=True)
class TPCGeometry:
    tpc_id: int
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float
    inv_dx: float; inv_dy: float; inv_dz: float


def load_tpc_geometries(path: str | Path = DEFAULT_TPC_YAML) -> Dict[int, TPCGeometry]:
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
        out[t] = TPCGeometry(
            tpc_id=t, x_min=xlo, x_max=xhi,
            y_min=ylo, y_max=yhi, z_min=zlo, z_max=zhi,
            inv_dx=NX / (xhi - xlo),
            inv_dy=NY / (yhi - ylo),
            inv_dz=NZ / (zhi - zlo),
        )
    return out


_TPC_GEOM_CACHE: Optional[Dict[int, TPCGeometry]] = None

def _get_geom_cache(yaml_path=DEFAULT_TPC_YAML) -> Dict[int, TPCGeometry]:
    global _TPC_GEOM_CACHE
    if _TPC_GEOM_CACHE is None:
        _TPC_GEOM_CACHE = load_tpc_geometries(yaml_path)
    return _TPC_GEOM_CACHE


def group_voxelize_pairs(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_ids: np.ndarray, labels: np.ndarray,
    *, include_noise: bool = False,
    restrict_clusters=None,
    yaml_path=DEFAULT_TPC_YAML,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized voxelization. Returns (maps_4d [G,1,NX,NY,NZ], group_cls, group_tpcs).
    tpc_ids must be CHARGE TPC IDs (0–69).
    """
    x  = np.asarray(x, np.float64); y  = np.asarray(y, np.float64)
    z  = np.asarray(z, np.float64); E  = np.asarray(E, np.float32)
    tpc_ids = np.asarray(tpc_ids, np.int64); labels = np.asarray(labels, np.int64)
    n = x.shape[0]

    valid = (labels >= 0) if not include_noise else np.ones(n, bool)
    if restrict_clusters is not None:
        valid &= np.isin(labels, np.asarray(list(restrict_clusters), np.int64))
    if not np.any(valid):
        return (np.zeros((0,1,NX,NY,NZ),np.float32),
                np.zeros((0,),int), np.zeros((0,),int))

    xv=x[valid]; yv=y[valid]; zv=z[valid]; Ev=E[valid]
    lab=labels[valid]; tpc=tpc_ids[valid]

    MAX_TPC = 128
    keys = lab * MAX_TPC + tpc
    uniq_keys, first_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    G = int(uniq_keys.size)
    order = np.argsort(first_idx)
    rank = np.empty(G, dtype=np.int64); rank[order] = np.arange(G, dtype=np.int64)
    g = rank[inv]
    pos = first_idx[order]
    group_cls  = lab[pos].astype(int)
    group_tpcs = tpc[pos].astype(int)

    geom_map = _get_geom_cache(yaml_path)
    maps_flat = np.zeros(G * NX * NY * NZ, np.float32)
    V = NX * NY * NZ

    for tpc_val in np.unique(tpc):
        geom = geom_map.get(int(tpc_val))
        if geom is None:
            warnings.warn(f"Unknown charge TPC id {int(tpc_val)} — skipping.", RuntimeWarning)
            continue
        sel = (tpc == tpc_val)
        xs=xv[sel]; ys=yv[sel]; zs=zv[sel]; Es=Ev[sel]; gs=g[sel]
        ix = np.clip(((xs-geom.x_min)*geom.inv_dx).astype(np.int32), 0, NX-1)
        iy = np.clip(((ys-geom.y_min)*geom.inv_dy).astype(np.int32), 0, NY-1)
        iz = np.clip(((zs-geom.z_min)*geom.inv_dz).astype(np.int32), 0, NZ-1)
        lin = ix.astype(np.int64)*(NY*NZ) + iy.astype(np.int64)*NZ + iz.astype(np.int64)
        np.add.at(maps_flat, gs.astype(np.int64)*V + lin, Es)

    maps_4d = maps_flat.reshape(G, NX, NY, NZ)[:, None, :, :, :]
    return maps_4d, group_cls, group_tpcs


def light_tpc_to_charge_tpc(light_tpc_id: int) -> int:
    return light_tpc_id + 1 if light_tpc_id % 2 == 0 else light_tpc_id - 1

def charge_tpc_to_light_tpc(charge_tpc_id: int) -> int:
    return light_tpc_to_charge_tpc(charge_tpc_id)


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture (matches train_ndfull.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from torch import amp as _amp
    def _autocast(enabled: bool): return _amp.autocast("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _ac
    def _autocast(enabled: bool): return _ac(enabled=enabled)


class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.avg_pool(x).reshape(b, c)
        with _autocast(enabled=False):
            y = self.fc(y.float())
        return x * y.reshape(b, c, 1, 1, 1).expand_as(x)


class ResBlock3D_SE(nn.Module):
    def __init__(self, ch, hidden=None, gn_groups=8):
        super().__init__()
        h = hidden or ch
        self.conv1 = nn.Conv3d(ch, h, 3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(min(gn_groups, h), h)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(h, ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(min(gn_groups, ch), ch)
        self.se    = SEBlock3D(ch)
    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.se(self.gn2(self.conv2(y)))
        return self.act(x + y)


class DownBlock3D_SE(nn.Module):
    def __init__(self, c_in, c_out, stride=(2,2,2), gn_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.gn   = nn.GroupNorm(min(gn_groups, c_out), c_out)
        self.act  = nn.SiLU(inplace=True)
        self.res  = ResBlock3D_SE(c_out, gn_groups=gn_groups)
    def forward(self, x):
        return self.res(self.act(self.gn(self.conv(x))))


def _create_3d_sinusoidal_pos(D, H, W, dim):
    dz = (dim // 3) // 2 * 2
    dy = (dim // 3) // 2 * 2
    dx = dim - dz - dy

    def _pe1d(n, d):
        pos = torch.arange(n).unsqueeze(-1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.) / max(1, d)))
        pe  = torch.zeros(n, d)
        if d > 0:
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
        return pe

    pez = _pe1d(D, dz).view(D,1,1,dz).expand(D,H,W,dz)
    pey = _pe1d(H, dy).view(1,H,1,dy).expand(D,H,W,dy)
    pex = _pe1d(W, dx).view(1,1,W,dx).expand(D,H,W,dx)
    pe  = torch.cat([pez, pey, pex], dim=-1)   # (D,H,W,dim)
    return pe.view(1, -1, dim)                  # (1, D*H*W, dim)


class PerceiverDecoder(nn.Module):
    def __init__(self, embed_dim, num_targets=120, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.target_queries = nn.Parameter(torch.randn(1, num_targets, embed_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder    = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
    def forward(self, memory):
        B = memory.size(0)
        q = self.target_queries.expand(B, -1, -1)
        return self.final_norm(self.decoder(q, memory))


class HybridPerceiver3D(nn.Module):
    """
    Architecture from train_ndfull.py. Checkpoint loads cleanly from
    runs/ndfull_run_distributed/best_model.pt.
    """
    def __init__(self, c_in=1, base_channels=64, embed_dim=512,
                 num_targets=120, tpc_embed_dim=32, num_tpcs=72,
                 num_decoder_layers=4, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(c_in, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels), nn.SiLU(inplace=True),
            ResBlock3D_SE(base_channels),
        )
        self.down1 = DownBlock3D_SE(base_channels, base_channels * 2)
        self.down2 = DownBlock3D_SE(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock3D_SE(base_channels * 4, embed_dim)

        self.embed_dim    = embed_dim
        self.pos_embedding = None           # built lazily on first forward

        self.tpc_emb        = nn.Embedding(num_tpcs, tpc_embed_dim)
        self.tpc_film_gamma = nn.Linear(tpc_embed_dim, embed_dim)
        self.tpc_film_beta  = nn.Linear(tpc_embed_dim, embed_dim)

        self.decoder = PerceiverDecoder(
            embed_dim=embed_dim, num_targets=num_targets,
            num_layers=num_decoder_layers, num_heads=8, dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x, tpc_ids):
        x = self.stem(x); x = self.down1(x); x = self.down2(x); x = self.down3(x)
        B, C, D, H, W = x.size()

        if self.pos_embedding is None or self.pos_embedding.shape[1] != D*H*W:
            pe = _create_3d_sinusoidal_pos(D, H, W, self.embed_dim).to(x.device)
            self.pos_embedding = pe

        x = x.view(B, C, -1).transpose(1, 2)    # (B, D*H*W, embed_dim)

        # TPC FiLM conditioning
        with _autocast(enabled=False):
            e     = self.tpc_emb(tpc_ids)
            gamma = self.tpc_film_gamma(e).unsqueeze(1)
            beta  = self.tpc_film_beta(e).unsqueeze(1)

        x = x * (1.0 + gamma) + beta
        x = x + self.pos_embedding

        target_features = self.decoder(x)             # (B, 120, embed_dim)
        final = self.regressor(target_features).squeeze(-1)   # (B, 120)
        return {"final": final}


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_perceiver_model(
    pt_path: str,
    *,
    base_channels: int = 64,
    embed_dim: int = 512,
    num_tpcs: int = 72,
    num_targets: int = NUM_TARGETS,
    num_decoder_layers: int = 4,
    dropout: float = 0.0,           # 0 at inference
    device: Optional[str] = None,
    target_scale: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a trained HybridPerceiver3D checkpoint from train_ndfull.py.

    Returns (model.eval(), meta).
    meta = {"out_dim": 120, "target_scale": 1e-3, "device": "...", ...}
    """
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))

    ck = torch.load(pt_path, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        state   = ck["model"]
        ck_meta = {k: v for k, v in ck.items() if k != "model"}
    else:
        state   = ck
        ck_meta = {}

    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    model = HybridPerceiver3D(
        c_in=1, base_channels=base_channels, embed_dim=embed_dim,
        num_targets=num_targets, num_tpcs=num_tpcs,
        num_decoder_layers=num_decoder_layers, dropout=dropout,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(dev).eval()

    meta: Dict[str, Any] = {
        "out_dim":      num_targets,
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
    return "cuda" in msg and "out of memory" in msg


@torch.no_grad()
def predict_phi(
    maps_4d: np.ndarray,            # (G, 1, NX, NY, NZ)
    model: nn.Module,
    tpc_ids: np.ndarray,            # (G,) int, charge TPC IDs
    *,
    target_scale: float = 1e-3,
    batch_size: int = 4,            # Perceiver is heavier; default smaller
    raw_clip: Optional[Tuple[float, float]] = (0.0, 60780.0),
    min_prediction_threshold: Optional[float] = 100.0,
    device_policy: str = "auto",
) -> np.ndarray:
    """
    Batched inference → (G, 120) float32 in raw ADC units.
    """
    G = maps_4d.shape[0]
    if G == 0:
        return np.zeros((0, NUM_TARGETS), np.float32)

    x = np.asarray(maps_4d, np.float32)
    for i in range(G):                          # per-sample max-norm (matches training)
        vmax = x[i].max()
        # if vmax > 1e-6: x[i] /= vmax

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
                # model returns {"final": tensor} or raw tensor — handle both
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
    """Map (cluster_id, charge_tpc_id) → (120, L) waveform."""
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
    batch_size: int = 4,
    raw_clip: Tuple[float, float] = (0.0, 60780.0),
    min_prediction_threshold: Optional[float] = 100.0,
    yaml_path=DEFAULT_TPC_YAML,
    device_policy: str = "auto",
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, Any]]:
    """
    Full pipeline: hits → imageMaps[(cluster_id, charge_tpc_id)] → (120, L).

    Parameters
    ----------
    x, y, z, E    : hit arrays in physical units matching tpc_boundaries.yaml
    tpc_ids       : CHARGE TPC IDs (0–69) per hit
    labels        : cluster IDs per hit (-1 = noise, excluded by default)
    model         : loaded HybridPerceiver3D (from load_perceiver_model)
    template      : 1D pulse-shape array. None → flat ones(1000).
    min_prediction_threshold
                  : amplitudes below this raw-ADC threshold are suppressed to 0.
                    Set to None to disable thresholding entirely.
    """
    maps_4d, group_cls, group_tpcs = group_voxelize_pairs(
        x, y, z, E, tpc_ids, labels,
        include_noise=include_noise, yaml_path=yaml_path,
    )
    L = int(template.size) if template is not None else 1000

    if maps_4d.shape[0] == 0:
        empty = {
            "group_cls":  np.array([], int), "group_tpcs": np.array([], int),
            "maps_4d":    np.zeros((0,1,NX,NY,NZ), np.float32),
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
    wave = amps[:, :, None] * tmpl[None, None, :]    # (G, 120, L)

    imageMaps = build_image_map_dict(wave, group_cls, group_tpcs)
    meta = {
        "group_cls":  group_cls, "group_tpcs": group_tpcs,
        "maps_4d":    maps_4d,   "amplitudes": amps, "waveforms": wave,
    }
    return imageMaps, meta


# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "NX", "NY", "NZ", "NUM_TARGETS", "DEFAULT_TPC_YAML",
    "TPCGeometry", "load_tpc_geometries",
    "group_voxelize_pairs",
    "HybridPerceiver3D", "load_perceiver_model",
    "predict_phi",
    "build_image_map_dict", "process_clusters_to_imageMaps",
    "light_tpc_to_charge_tpc", "charge_tpc_to_light_tpc",
]
