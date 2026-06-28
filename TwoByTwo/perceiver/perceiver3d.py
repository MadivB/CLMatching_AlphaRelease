"""
perceiver3d.py
==============
Shared HybridPerceiver3D architecture for charge → light amplitude prediction.

This is the single source of truth for the model used by:
  - train_perceiver.py        (training, ND-LAr full and 2x2)
  - ML_NDfull_perceiver.py    (ND-LAr inference toolbox)
  - ML_2x2_perceiver.py       (2x2 inference toolbox)

Architecture
------------
  1. 3D ResNet stem + Squeeze-and-Excitation (SE) blocks.
  2. Three stride-2 down-sampling stages (channel progression
     base → 2·base → 4·base → embed_dim).
  3. Spatial flattening + deterministic 3D sinusoidal positional encoding
     (built lazily from the feature-map shape, so any voxel grid works).
  4. TPC FiLM conditioning (per-TPC learned gamma/beta on the tokens).
  5. Transformer decoder (Pre-LN): one learned query per light channel,
     alternating self-attention (inter-target communication) and
     cross-attention (looking at the 3D feature tokens).
  6. Shared regression head applied to each target token.

Size presets
------------
  "ndlar"        compact ND-LAr full model   (~3.9 M params, default)
  "ndlar-large"  original ND-LAr full model  (~40.8 M params; matches the
                 historical train_ndfull.py checkpoints, e.g.
                 runs/ndfull_run_distributed/best_model.pt)
  "2x2"          2x2 demonstrator model      (~0.9 M params)

Checkpoint compatibility
------------------------
`arch_from_state_dict` recovers every architecture hyper-parameter except
`num_heads` directly from tensor shapes, so checkpoints trained at any size
(including the legacy 512-dim ND model) load without manual configuration.
`num_heads` does not affect tensor shapes; training stores it in the
checkpoint "config" dict, and `resolve_arch` falls back to a rule that
matches all historical models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ---------------- AMP shim ----------------
try:
    from torch import amp as _amp
    def _autocast(enabled: bool): return _amp.autocast("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _ac
    def _autocast(enabled: bool): return _ac(enabled=enabled)


# ---------------- Size presets ----------------
PRESETS: Dict[str, Dict[str, Any]] = {
    # Compact ND-LAr full model. embed_dim 512→128 and base_channels 64→32
    # relative to the original; see README ("How big should the model be?").
    "ndlar": dict(
        c_in=1, base_channels=32, embed_dim=128, num_targets=120,
        tpc_embed_dim=32, num_tpcs=72, num_decoder_layers=4, num_heads=8,
    ),
    # Original train_ndfull.py architecture — kept for loading/retraining
    # legacy checkpoints and for capacity-comparison studies.
    "ndlar-large": dict(
        c_in=1, base_channels=64, embed_dim=512, num_targets=120,
        tpc_embed_dim=32, num_tpcs=72, num_decoder_layers=4, num_heads=8,
    ),
    # 2x2 demonstrator: 48 light channels per TPC, 8 TPCs, 32×128×64 voxel
    # grid. The problem is far smaller than ND-LAr full, hence the heavy cut.
    "2x2": dict(
        c_in=1, base_channels=16, embed_dim=64, num_targets=48,
        tpc_embed_dim=8, num_tpcs=8, num_decoder_layers=3, num_heads=4,
    ),
}


# ---------------- Building blocks ----------------
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
        # Force float32 for SE linear layers to avoid cuBLAS errors in some
        # Torch/CUDA environments.
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
    def __init__(self, c_in, c_out, stride=(2, 2, 2), gn_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.gn   = nn.GroupNorm(min(gn_groups, c_out), c_out)
        self.act  = nn.SiLU(inplace=True)
        self.res  = ResBlock3D_SE(c_out, gn_groups=gn_groups)

    def forward(self, x):
        return self.res(self.act(self.gn(self.conv(x))))


def create_3d_sinusoidal_pos(D, H, W, dim):
    """Deterministic 3D sinusoidal positional encoding, (1, D*H*W, dim)."""
    dim_z = (dim // 3) // 2 * 2
    dim_y = (dim // 3) // 2 * 2
    dim_x = dim - dim_z - dim_y

    def _pe1d(n, d):
        pos = torch.arange(n).unsqueeze(-1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / max(1, d)))
        pe  = torch.zeros(n, d)
        if d > 0:
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
        return pe

    pe_z = _pe1d(D, dim_z).view(D, 1, 1, dim_z).expand(D, H, W, dim_z)
    pe_y = _pe1d(H, dim_y).view(1, H, 1, dim_y).expand(D, H, W, dim_y)
    pe_x = _pe1d(W, dim_x).view(1, 1, W, dim_x).expand(D, H, W, dim_x)
    pe   = torch.cat([pe_z, pe_y, pe_x], dim=-1)   # (D, H, W, dim)
    return pe.view(1, -1, dim)                      # (1, D*H*W, dim)


class PerceiverDecoder(nn.Module):
    def __init__(self, embed_dim, num_targets=120, num_layers=4, num_heads=8,
                 dropout=0.1, dim_feedforward=None):
        super().__init__()
        self.target_queries = nn.Parameter(torch.randn(1, num_targets, embed_dim))
        nn.init.normal_(self.target_queries, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder    = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, memory):
        B = memory.size(0)
        # Run the TransformerDecoder in fp32. Under fp16/bf16 autocast the
        # attention's batched-GEMM path hits CUBLAS_STATUS_INVALID_VALUE for
        # these shapes; the decoder is tiny relative to the 3D-conv encoder,
        # so keeping it in fp32 costs almost nothing.
        with _autocast(enabled=False):
            memory32 = memory.float()
            q = self.target_queries.expand(B, -1, -1).contiguous()
            return self.final_norm(self.decoder(q, memory32))


class HybridPerceiver3D(nn.Module):
    """3D CNN+SE encoder → positional encoding → TPC FiLM → Perceiver decoder."""

    def __init__(self, c_in=1, base_channels=32, embed_dim=128, num_targets=120,
                 tpc_embed_dim=32, num_tpcs=72, num_decoder_layers=4,
                 num_heads=8, dropout=0.1, dim_feedforward=None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(c_in, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(inplace=True),
            ResBlock3D_SE(base_channels),
        )
        self.down1 = DownBlock3D_SE(base_channels, base_channels * 2)
        self.down2 = DownBlock3D_SE(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock3D_SE(base_channels * 4, embed_dim)

        self.embed_dim     = embed_dim
        self.pos_embedding = None           # built lazily on first forward

        self.tpc_emb        = nn.Embedding(num_tpcs, tpc_embed_dim)
        self.tpc_film_gamma = nn.Linear(tpc_embed_dim, embed_dim)
        self.tpc_film_beta  = nn.Linear(tpc_embed_dim, embed_dim)

        self.decoder = PerceiverDecoder(
            embed_dim=embed_dim, num_targets=num_targets,
            num_layers=num_decoder_layers, num_heads=num_heads,
            dropout=dropout, dim_feedforward=dim_feedforward,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x, tpc_ids):
        x = self.stem(x); x = self.down1(x); x = self.down2(x); x = self.down3(x)
        B, C, D, H, W = x.size()

        if self.pos_embedding is None or self.pos_embedding.shape[1] != D * H * W:
            pe = create_3d_sinusoidal_pos(D, H, W, self.embed_dim).to(x.device)
            self.pos_embedding = pe

        x = x.view(B, C, -1).transpose(1, 2)    # (B, D*H*W, embed_dim)

        # TPC FiLM conditioning (fp32: tiny linear layers, same cuBLAS caveat)
        with _autocast(enabled=False):
            e     = self.tpc_emb(tpc_ids)
            gamma = self.tpc_film_gamma(e).unsqueeze(1)
            beta  = self.tpc_film_beta(e).unsqueeze(1)

        x = x * (1.0 + gamma) + beta
        x = x + self.pos_embedding

        target_features = self.decoder(x)                       # (B, K, embed)
        final = self.regressor(target_features).squeeze(-1)    # (B, K)
        return {"final": final}


# ---------------- Helpers ----------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_checkpoint_state(pt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a checkpoint file → (state_dict, meta).

    Handles both raw state dicts (legacy best_model.pt) and full checkpoints
    of the form {"model": state, "config": {...}, ...}.
    """
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        meta  = {k: v for k, v in ck.items() if k != "model"}
    else:
        state = ck
        meta  = {}
    return strip_module_prefix(state), meta


def arch_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Recover architecture hyper-parameters from tensor shapes.

    Everything except num_heads is uniquely determined:
      stem.0.weight                  (base, c_in, 3, 3, 3)
      decoder.target_queries         (1, num_targets, embed_dim)
      tpc_emb.weight                 (num_tpcs, tpc_embed_dim)
      decoder.decoder.layers.{i}.*   → num_decoder_layers
      decoder...layers.0.linear1.weight  (dim_feedforward, embed_dim)
    """
    state = strip_module_prefix(state)
    stem_w  = state["stem.0.weight"]
    queries = state["decoder.target_queries"]
    tpc_w   = state["tpc_emb.weight"]

    layer_ids = set()
    for k in state:
        if k.startswith("decoder.decoder.layers."):
            layer_ids.add(int(k.split(".")[3]))
    if not layer_ids:
        raise ValueError("State dict has no decoder layers — not a HybridPerceiver3D checkpoint?")

    return dict(
        c_in=int(stem_w.shape[1]),
        base_channels=int(stem_w.shape[0]),
        embed_dim=int(queries.shape[2]),
        num_targets=int(queries.shape[1]),
        tpc_embed_dim=int(tpc_w.shape[1]),
        num_tpcs=int(tpc_w.shape[0]),
        num_decoder_layers=max(layer_ids) + 1,
        dim_feedforward=int(state["decoder.decoder.layers.0.linear1.weight"].shape[0]),
        num_heads=None,   # not recoverable from shapes; see resolve_arch
    )


def _default_num_heads(embed_dim: int) -> int:
    # Matches all historical models: 8 heads at embed_dim ≥ 128, else 4.
    if embed_dim >= 128 and embed_dim % 8 == 0:
        return 8
    return 4


def resolve_arch(state: Dict[str, torch.Tensor],
                 config: Optional[Dict[str, Any]] = None,
                 **overrides: Any) -> Dict[str, Any]:
    """
    Merge shape-inferred hyper-parameters with a stored config and explicit
    overrides (priority: overrides > config > inferred > num_heads rule).
    Override values of None are ignored.
    """
    arch = arch_from_state_dict(state)
    for src in (config or {}), overrides:
        for k, v in src.items():
            if k in arch and v is not None:
                arch[k] = v
    if arch.get("num_heads") is None:
        arch["num_heads"] = _default_num_heads(arch["embed_dim"])
    return arch


def build_model_from_state(state: Dict[str, torch.Tensor],
                           config: Optional[Dict[str, Any]] = None,
                           dropout: float = 0.0,
                           **overrides: Any) -> Tuple[HybridPerceiver3D, Dict[str, Any]]:
    """Construct a HybridPerceiver3D matching `state` and load the weights."""
    state = strip_module_prefix(state)
    arch  = resolve_arch(state, config, **overrides)
    model = HybridPerceiver3D(dropout=dropout, **arch)
    model.load_state_dict(state, strict=True)
    return model, arch


__all__ = [
    "PRESETS",
    "SEBlock3D", "ResBlock3D_SE", "DownBlock3D_SE",
    "create_3d_sinusoidal_pos", "PerceiverDecoder", "HybridPerceiver3D",
    "count_parameters", "strip_module_prefix", "load_checkpoint_state",
    "arch_from_state_dict", "resolve_arch", "build_model_from_state",
]
