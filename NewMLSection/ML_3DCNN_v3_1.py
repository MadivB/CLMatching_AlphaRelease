"""
ML_3DCNN.py (v3)
----------------
Importable PyTorch models and utilities for ND LAr light prediction.

v3 changes to match the new training model layout (train_cnnx6_phased_ddp.py):
- Exact stem/down/trunk structure with dict outputs {"final","aux"}
- Widened post-CNN widths via `width_mult_after_cnn` for trunk/heads/aux
- HeadMLP with two hidden layers (no softplus clamp; clamp handled by output_cap_scaled)
- Supports 48-output single-bank (heads.*) and type-split (headsA.* / headsB.*)
- Keeps aux_head for strict state_dict loading (aux can be ignored at inference)
- Backward-compatible loader supports legacy 12-output checkpoints

Public contents:
- ResBlock3D, DownBlock3D, MLPBlock, HeadMLP
- Model48Single, Model48TypeSplit, CNN12Compat (legacy 12-out)
- load_cnn_model_flex: auto-detects and loads 12/48-output checkpoints
- predict_phi, predict_phi_simple: batched and single-sample inference helpers
- voxelize_xyzE_by_tpc, group_voxelize, group_voxelize_pairs
- predict_multi_phi_to_waveforms, build_image_map_dict, process_clusters_to_imageMaps
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List, Sequence, Iterable
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Blocks -----------------------------

class ResBlock3D(nn.Module):
    def __init__(self, ch: int, hidden: Optional[int] = None, gn_groups: int = 8):
        super().__init__()
        h = hidden or ch
        self.conv1 = nn.Conv3d(ch, h, 3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=min(gn_groups, h), num_channels=h)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(h, ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(x + y)


class DownBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: Tuple[int,int,int] = (2,2,2), gn_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.gn   = nn.GroupNorm(num_groups=min(gn_groups, c_out), num_channels=c_out)
        self.act  = nn.SiLU(inplace=True)
        self.res  = ResBlock3D(c_out, gn_groups=gn_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn(self.conv(x)))
        return self.res(x)


class MLPBlock(nn.Module):
    """Residual MLP block (LayerNorm outside is handled by the trunk)."""
    def __init__(self, dim: int, hidden: int, drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln  = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.fc1(x)))
        y = self.drop(self.fc2(y))
        return self.ln(x + y)

# ----------------------------- Head -----------------------------

class HeadMLP(nn.Module):
    """Matches training: Linear->SiLU->Dropout->Linear->SiLU->Dropout->Linear(6)."""
    def __init__(self, dim: int, hidden: int, drop: float = 0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(inplace=True), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.SiLU(inplace=True), nn.Dropout(drop),
            nn.Linear(hidden, 6)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

# -------------------------- Model Variants --------------------------

class _Backbone(nn.Module):
    """
    Shared backbone: stem + 3 downs + global pool + trunk MLP.

    Notes:
      - For base=24, pooled feature size is feat_dim = base*4 = 96 (not 192).
      - v = pooled conv features (B, feat_dim)
      - t = trunk features (B, dim)
    """
    def __init__(self, c_in: int, base: int,
                 trunk_width: int, trunk_depth: int, trunk_mult: float,
                 dropout: float, width_mult_after_cnn: float,
                 tpc_cond: str = "none", num_tpcs: int = 8, tpc_embed_dim: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(c_in, base, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, base), num_channels=base),
            nn.SiLU(inplace=True),
            ResBlock3D(base),
        )
        self.down1 = DownBlock3D(base,   base*2)
        self.down2 = DownBlock3D(base*2, base*4)
        self.down3 = DownBlock3D(base*4, base*4)
        self.pool  = nn.AdaptiveAvgPool3d((1,1,1))
        self.feat_dim = base*4  # = 96 when base=24

        feat = self.feat_dim
        dim  = int(trunk_width * float(width_mult_after_cnn))
        blocks: List[nn.Module] = [nn.LayerNorm(feat), nn.Linear(feat, dim), nn.SiLU(inplace=True), nn.Dropout(dropout)]
        for _ in range(int(trunk_depth)):
            blocks.append(MLPBlock(dim, hidden=int(dim*trunk_mult), drop=dropout))
        self.trunk = nn.Sequential(*blocks)

        self.trunk_dim = dim
        self.width_mult_after_cnn = float(width_mult_after_cnn)

        # TPC conditioning (FiLM)
        self.tpc_cond = tpc_cond
        self.num_tpcs = int(num_tpcs)
        self.tpc_embed_dim = int(tpc_embed_dim)
        if self.tpc_cond == "film":
            self.tpc_emb = nn.Embedding(self.num_tpcs, self.tpc_embed_dim)
            self.tpc_to_gamma = nn.Linear(self.tpc_embed_dim, self.feat_dim)
            self.tpc_to_beta  = nn.Linear(self.tpc_embed_dim, self.feat_dim)
        elif self.tpc_cond == "none":
            self.tpc_emb = None
        else:
            raise ValueError(f"Unknown tpc_cond: {tpc_cond}")

    def forward_backbone(self, x: torch.Tensor, tpc_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pooled conv features 'v' and trunk features 't'."""
        x = self.stem(x); x = self.down1(x); x = self.down2(x); x = self.down3(x)
        v = self.pool(x).flatten(1)   # (B, feat_dim)

        # Apply FiLM
        if (self.tpc_cond == "film") and (tpc_ids is not None):
            e = self.tpc_emb(tpc_ids)
            gamma = self.tpc_to_gamma(e)
            beta  = self.tpc_to_beta(e)
            v = v * (1.0 + gamma) + beta

        t = self.trunk(v)             # (B, dim)
        return v, t


class Model48Single(_Backbone):
    """
    48-output model with a single bank of heads: heads[i] -> 6 channels, plus aux_head (48).
    Forward returns dict: {"final": (B,48), "aux": (B,48)} with output-cap applied in scaled units.
    """
    def __init__(self, n_groups: int, c_in: int = 1, base: int = 24,
                 trunk_width: int = 256, trunk_depth: int = 4, trunk_mult: float = 1.0,
                 head_hidden: int = 256, dropout: float = 0.1,
                 output_cap_scaled: float = 0.0,
                 width_mult_after_cnn: float = 2.0,
                 tpc_cond: str = "none", num_tpcs: int = 8, tpc_embed_dim: int = 32):
        super().__init__(c_in, base, trunk_width, trunk_depth, trunk_mult, dropout, width_mult_after_cnn,
                         tpc_cond, num_tpcs, tpc_embed_dim)
        head_hidden_eff = int(head_hidden * self.width_mult_after_cnn)
        self.heads = nn.ModuleList([HeadMLP(self.trunk_dim, head_hidden_eff, dropout) for _ in range(int(n_groups))])
        aux_hidden = int(self.feat_dim * self.width_mult_after_cnn)
        self.aux_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, aux_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, 48),
        )
        self.output_cap_scaled = float(output_cap_scaled)

    def forward(self, x: torch.Tensor, tpc_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        v, t = self.forward_backbone(x, tpc_ids)
        outs = [h(t) for h in self.heads]
        final = torch.cat(outs, dim=-1)    # (B, 48)
        aux   = self.aux_head(v)           # (B, 48)
        # Remove max clipping, enforce non-negative
        final = torch.relu(final)
        aux   = torch.relu(aux)
        return {"final": final, "aux": aux}


class Model48TypeSplit(_Backbone):
    """
    48-output model with two banks of heads (A/B). For groups {0,2,4,6} use headsA, else headsB.
    Forward returns dict: {"final": (B,48), "aux": (B,48)} with output-cap applied.
    """
    def __init__(self, n_groups: int, c_in: int = 1, base: int = 24,
                 trunk_width: int = 256, trunk_depth: int = 4, trunk_mult: float = 1.0,
                 head_hidden: int = 256, dropout: float = 0.1,
                 output_cap_scaled: float = 0.0,
                 width_mult_after_cnn: float = 2.0,
                 tpc_cond: str = "none", num_tpcs: int = 8, tpc_embed_dim: int = 32):
        super().__init__(c_in, base, trunk_width, trunk_depth, trunk_mult, dropout, width_mult_after_cnn,
                         tpc_cond, num_tpcs, tpc_embed_dim)
        self.A_groups = set([0,2,4,6])
        self.B_groups = set([1,3,5,7])
        head_hidden_eff = int(head_hidden * self.width_mult_after_cnn)
        self.headsA = nn.ModuleList([HeadMLP(self.trunk_dim, head_hidden_eff, dropout) for _ in range(int(n_groups))])
        self.headsB = nn.ModuleList([HeadMLP(self.trunk_dim, head_hidden_eff, dropout) for _ in range(int(n_groups))])
        aux_hidden = int(self.feat_dim * self.width_mult_after_cnn)
        self.aux_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, aux_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, 48),
        )
        self.output_cap_scaled = float(output_cap_scaled)

    def forward(self, x: torch.Tensor, tpc_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        v, t = self.forward_backbone(x, tpc_ids)
        outs: List[torch.Tensor] = []
        for gi in range(len(self.headsA)):
            h = self.headsA[gi] if gi in self.A_groups else self.headsB[gi]
            outs.append(h(t))
        final = torch.cat(outs, dim=-1)
        aux   = self.aux_head(v)
        # Remove max clipping, enforce non-negative
        final = torch.relu(final)
        aux   = torch.relu(aux)
        return {"final": final, "aux": aux}

# -------------------------- Legacy 12-output (optional) --------------------------

def _make_head_legacy(feat: int, dropout: float = 0.1) -> nn.Sequential:
    # a simple 2-layer head; kept for older 12-out checkpoints
    return nn.Sequential(
        nn.Linear(feat, feat),
        nn.SiLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(feat, 6),
    )

class CNN12Compat(nn.Module):
    """
    Legacy 12-output model (two 6-channel heads). Forward returns dict for uniformity.
    """
    def __init__(self, c_in: int = 1, base: int = 24, dropout: float = 0.1,
                 mlp_depth: int = 3, mlp_width_mult: float = 4.0,
                 output_cap_scaled: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(c_in, base, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, base), num_channels=base),
            nn.SiLU(inplace=True),
            ResBlock3D(base),
        )
        self.down1 = DownBlock3D(base, base*2)
        self.down2 = DownBlock3D(base*2, base*4)
        self.down3 = DownBlock3D(base*4, base*4)
        self.pool  = nn.AdaptiveAvgPool3d((1,1,1))
        feat = base*4  # 96 for base=24

        width = int(feat * mlp_width_mult)
        blocks: List[nn.Module] = [nn.LayerNorm(feat), nn.Linear(feat, feat), nn.SiLU(inplace=True), nn.Dropout(dropout)]
        for _ in range(mlp_depth):
            blocks.append(MLPBlock(feat, hidden=width, drop=dropout))
        self.shared = nn.Sequential(*blocks)

        self.head0 = _make_head_legacy(feat, dropout)
        self.head1 = _make_head_legacy(feat, dropout)
        self.output_cap_scaled = float(output_cap_scaled)

    def forward(self, x: torch.Tensor, tpc_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.stem(x); x = self.down1(x); x = self.down2(x); x = self.down3(x)
        x = self.pool(x).flatten(1); x = self.shared(x)
        a = self.head0(x); b = self.head1(x)
        y = torch.cat([a, b], dim=-1)  # (B, 12)
        # Remove max clipping, enforce non-negative
        y = torch.relu(y)
        # Keep aux for API consistency (zero tensor)
        aux = torch.zeros(y.shape[0], y.shape[1], device=y.device, dtype=y.dtype)
        return {"final": y, "aux": aux}

# -------------------------- Loader (auto-detect) --------------------------

def load_cnn_model_flex(
    pt_path: str,
    *,
    # new-model knobs (must match training-time values)
    base: int = 24,
    trunk_width: int = 256,
    trunk_depth: int = 4,
    trunk_mult: float = 1.0,
    head_hidden: int = 256,
    dropout: float = 0.1,
    width_mult_after_cnn: float = 2.0,
    # output cap in RAW units (scaled inside)
    output_cap_raw: float = 60768.0,
    target_scale: float = 1e-3,
    device: Optional[str] = None,
    expected_outputs: Optional[int] = None,  # 12 or 48
    # TPC conditioning
    tpc_cond: str = "auto",
    num_tpcs: int = 8,
    tpc_embed_dim: int = 32,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a checkpoint supporting variants:
      - 12-output (head0/head1)
      - 48-output single (heads.N.*)
      - 48-output type-split (headsA.N.* and headsB.N.*)

    Returns
    -------
    (model.eval(), meta) with meta = {"out_dim": K, "variant": "...", "n_groups": ...}
    """
    device_t = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    ck = torch.load(pt_path, map_location="cpu")
    state = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck

    # Strip 'module.' if present
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Strip a possible 'backbone.' prefix (old exports)
    if any(k.startswith("backbone.") for k in state.keys()):
        state = {k.replace("backbone.", "", 1): v for k, v in state.items()}

    keys = list(state.keys())
    has_12  = any(k.startswith("head0.") or k.startswith("head1.") for k in keys)
    has_sgl = any(k.startswith("heads.") for k in keys)
    has_spl = any(k.startswith("headsA.") or k.startswith("headsB.") for k in keys)
    
    # Auto-detect TPC conditioning
    if tpc_cond == "auto":
        if any(k.startswith("tpc_emb.") for k in keys):
            tpc_cond = "film"
        else:
            tpc_cond = "none"

    out_cap_scaled = float(output_cap_raw) * float(target_scale)

    if has_12:
        model = CNN12Compat(c_in=1, base=base, dropout=dropout,
                            mlp_depth=3, mlp_width_mult=4.0,
                            output_cap_scaled=out_cap_scaled)
        variant = "12"; K = 12
    elif has_spl:
        gi_max = -1
        for k in keys:
            if k.startswith("headsA.") or k.startswith("headsB."):
                try:
                    gi = int(k.split(".", 2)[1]); gi_max = max(gi_max, gi)
                except Exception:
                    pass
        n_groups = gi_max + 1 if gi_max >= 0 else 8
        model = Model48TypeSplit(n_groups=n_groups, c_in=1, base=base,
                                 trunk_width=trunk_width, trunk_depth=trunk_depth, trunk_mult=trunk_mult,
                                 head_hidden=head_hidden, dropout=dropout,
                                 output_cap_scaled=out_cap_scaled,
                                 width_mult_after_cnn=width_mult_after_cnn,
                                 tpc_cond=tpc_cond, num_tpcs=num_tpcs, tpc_embed_dim=tpc_embed_dim)
        variant = "48_split"; K = 6 * n_groups
    elif has_sgl:
        gi_max = -1
        for k in keys:
            if k.startswith("heads."):
                try:
                    gi = int(k.split(".", 2)[1]); gi_max = max(gi_max, gi)
                except Exception:
                    pass
        n_groups = gi_max + 1 if gi_max >= 0 else 8
        model = Model48Single(n_groups=n_groups, c_in=1, base=base,
                              trunk_width=trunk_width, trunk_depth=trunk_depth, trunk_mult=trunk_mult,
                              head_hidden=head_hidden, dropout=dropout,
                              output_cap_scaled=out_cap_scaled,
                              width_mult_after_cnn=width_mult_after_cnn,
                              tpc_cond=tpc_cond, num_tpcs=num_tpcs, tpc_embed_dim=tpc_embed_dim)
        variant = "48_single"; K = 6 * n_groups
    else:
        raise RuntimeError("Could not infer model variant (no head0/head1, heads.*, or headsA/headsB keys).")

    incompat = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, "missing_keys", []))
    unexpected = list(getattr(incompat, "unexpected_keys", []))
    if missing or unexpected:
        # try strict to surface a clearer error if it truly mismatches
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            raise RuntimeError(f"State dict mismatch.\nMissing: {missing}\nUnexpected: {unexpected}\n{e}")

    if expected_outputs is not None and K != expected_outputs:
        raise ValueError(f"Checkpoint has {K} outputs but expected {expected_outputs}.")

    model = model.to(device_t).eval()
    meta: Dict[str, Any] = {"out_dim": K, "variant": variant, "n_groups": (K // 6 if K % 6 == 0 else None)}
    return model, meta

# -------------------------- Predictors --------------------------

def _is_cuda_oom(exc: Exception) -> bool:
    """Best-effort detection of CUDA OOM."""
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return ("cuda" in msg and "out of memory" in msg) or ("cuda" in msg and "oom" in msg)

@torch.no_grad()
def predict_phi(
    grids,
    model_01: nn.Module,
    model_other: Optional[nn.Module] = None,
    tpcs_for_grids: Optional[Sequence[int]] = None,
    device_01: Optional[str] = None,
    device_other: Optional[str] = None,
    input_scale: str = "none",          # "p99" | "sum" | "none"
    target_scale: float = 1e-3,        # convert back to RAW by 1/target_scale
    batch_size: int = 32,
    raw_clip: Optional[Tuple[float, float]] = None,
    expected_outputs: Optional[int] = None,  # assert K if desired (12 or 48)
    device_policy: str = "auto",       # "auto" | "force_cuda" | "force_cpu"
) -> np.ndarray:
    """
    Batched inference with GPU->CPU fallback on CUDA OOM.

    device_policy:
      - "auto"        : try preferred device (CUDA if available), on CUDA OOM -> warn + fallback to CPU
      - "force_cuda"  : force CUDA; a CUDA OOM will RAISE (no fallback)
      - "force_cpu"   : run on CPU only
    """
    # normalize input to (N,1,32,128,64)
    def _as_3d_list(obj) -> List[np.ndarray]:
        arrs: List[np.ndarray] = []
        if isinstance(obj, (list, tuple)):
            for item in obj:
                a = np.asarray(item, dtype=np.float32)
                if a.ndim == 3 and a.shape == (32,128,64):
                    arrs.append(a)
                elif a.ndim == 4 and a.shape[1:] == (32,128,64):      # (M,32,128,64)
                    arrs.extend([a[i] for i in range(a.shape[0])])
                elif a.ndim == 5 and a.shape[1] == 1 and a.shape[2:] == (32,128,64):  # (M,1,32,128,64)
                    arrs.extend([a[i,0] for i in range(a.shape[0])])
                else:
                    raise ValueError(f"Unsupported element shape in list: {a.shape}")
            return arrs
        else:
            a = np.asarray(obj, dtype=np.float32)
            if a.ndim == 3 and a.shape == (32,128,64):
                return [a]
            elif a.ndim == 4 and a.shape[1:] == (32,128,64):          # (N,32,128,64)
                return [a[i] for i in range(a.shape[0])]
            elif a.ndim == 5 and a.shape[1] == 1 and a.shape[2:] == (32,128,64):     # (N,1,32,128,64)
                return [a[i,0] for i in range(a.shape[0])]
            else:
                raise ValueError(f"Unexpected grids shape {a.shape}")

    vols = _as_3d_list(grids)
    x = np.stack(vols, axis=0).astype(np.float32)  # (N,32,128,64)
    N = x.shape[0]
    x = x[:, None, ...]                            # (N,1,32,128,64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # per-sample input scaling
    if input_scale == "p99":
        p = np.percentile(x, 99, axis=(2,3,4), keepdims=True)
        p = np.where(p > 1e-6, p, 1.0)
        x = (x / p).astype(np.float32, copy=False)
    elif input_scale == "sum":
        s = x.sum(axis=(2,3,4), keepdims=True)
        s = np.where(s > 1e-6, s, 1.0)
        x = (x / s).astype(np.float32, copy=False)
    elif input_scale in (None, "none"):
        pass
    else:
        raise ValueError(f"Unknown input_scale '{input_scale}'")

    # device helpers
    def _preferred_device_for(model: nn.Module, explicit: Optional[str]) -> torch.device:
        if device_policy == "force_cpu":
            return torch.device("cpu")
        if device_policy == "force_cuda":
            return torch.device("cuda")
        if explicit is not None:
            return torch.device(explicit)
        for p in model.parameters():
            return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _extract_final(y_scaled: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(y_scaled, dict):
            y_scaled = y_scaled.get("final", None)
            if y_scaled is None:
                raise RuntimeError("Model returned a dict without 'final' key.")
        return y_scaled

    def _run_batches_with_fallback(model: nn.Module, idxs: np.ndarray, pref_dev: torch.device,
                                   tpc_ids_all: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if idxs.size == 0:
            return None
        cur_dev = pref_dev
        if device_policy == "force_cpu":
            cur_dev = torch.device("cpu")
        elif device_policy == "force_cuda":
            cur_dev = torch.device("cuda")
        model = model.to(cur_dev).eval()

        bs = max(1, int(batch_size))
        out_scaled: Optional[np.ndarray] = None
        for start in range(0, idxs.size, bs):
            sl = idxs[start:start+bs]
            while True:
                try:
                    xt = torch.from_numpy(x[sl]).to(device=cur_dev, dtype=torch.float32,
                                                   non_blocking=(cur_dev.type == "cuda"))
                    if tpc_ids_all is not None:
                        tids = torch.from_numpy(tpc_ids_all[sl]).to(device=cur_dev, dtype=torch.long)
                        y_scaled = _extract_final(model(xt, tpc_ids=tids))
                    else:
                        y_scaled = _extract_final(model(xt))
                    y_np = y_scaled.detach().to(torch.float32).cpu().numpy()
                    if out_scaled is None:
                        K = y_np.shape[1]
                        if expected_outputs is not None and K != expected_outputs:
                            raise ValueError(f"Model produced {K} outputs; expected {expected_outputs}.")
                        out_scaled = np.zeros((idxs.size, K), dtype=np.float32)
                    out_scaled[start:start+sl.size, :] = y_np
                    break
                except Exception as e:
                    if cur_dev.type == "cuda" and device_policy == "auto" and _is_cuda_oom(e):
                        warnings.warn("predict_phi: CUDA OOM — falling back to CPU for remaining batches.", RuntimeWarning)
                        try: torch.cuda.empty_cache()
                        except Exception: pass
                        cur_dev = torch.device("cpu"); model = model.to(cur_dev).eval()
                        continue
                    raise
        return out_scaled

    # routing
    use_dual = (model_other is not None) and (tpcs_for_grids is not None)
    preds_scaled: Optional[np.ndarray] = None

    if use_dual:
        tpcs_arr = np.asarray(tpcs_for_grids, dtype=int)
        if tpcs_arr.shape[0] != N:
            raise ValueError(f"tpcs_for_grids length ({tpcs_arr.shape[0]}) != number of grids ({N}).")
        idx_01 = np.where((tpcs_arr == 0) | (tpcs_arr == 1))[0]
        idx_ot = np.setdiff1d(np.arange(N, dtype=int), idx_01, assume_unique=False)

        dev01 = _preferred_device_for(model_01, device_01)
        devOT = _preferred_device_for(model_other, device_other)

        y01_scaled = _run_batches_with_fallback(model_01, idx_01, dev01,
                                                tpc_ids_all=(tpcs_arr if tpcs_arr is not None else None))
        yot_scaled = _run_batches_with_fallback(model_other, idx_ot, devOT,
                                                tpc_ids_all=(tpcs_arr if tpcs_arr is not None else None))

        if y01_scaled is not None:
            K = y01_scaled.shape[1]
        elif yot_scaled is not None:
            K = yot_scaled.shape[1]
        else:
            raise ValueError("No samples were provided.")

        preds_scaled = np.zeros((N, K), dtype=np.float32)
        if y01_scaled is not None:
            preds_scaled[idx_01] = y01_scaled
        if yot_scaled is not None:
            if expected_outputs is not None and yot_scaled.shape[1] != expected_outputs:
                raise ValueError(f"model_other produced {yot_scaled.shape[1]} outputs; expected {expected_outputs}.")
            if y01_scaled is not None and yot_scaled.shape[1] != y01_scaled.shape[1]:
                raise ValueError("model_01 and model_other output sizes (K) do not match.")
            preds_scaled[idx_ot] = yot_scaled
    else:
        dev01 = _preferred_device_for(model_01, device_01)
        # if tpcs_for_grids is provided, pass it
        tids = np.asarray(tpcs_for_grids, dtype=int) if tpcs_for_grids is not None else None
        preds_scaled = _run_batches_with_fallback(model_01, np.arange(N, dtype=int), dev01, tpc_ids_all=tids)

    if preds_scaled is None:
        raise RuntimeError("Prediction produced no outputs.")

    # rescale to RAW units
    inv = 1.0 / float(target_scale) if target_scale not in (None, 0.0) else 1.0
    preds_raw = preds_scaled * inv
    if raw_clip is not None:
        lo, hi = raw_clip
        np.clip(preds_raw, lo, hi, out=preds_raw)
    return preds_raw

@torch.no_grad()
def predict_phi_simple(
    grid: np.ndarray,
    model: nn.Module,
    device: Optional[str] = None,
    input_scale: str = "none",          # "p99" | "sum" | "none"
    target_scale: float = 1e-3,        # convert back to RAW by 1/target_scale
    device_policy: str = "auto",       # "auto" | "force_cuda" | "force_cpu"
    tpc_id: Optional[int] = None,      # <--- Added optional TPC ID
) -> np.ndarray:
    """
    Single-sample inference with GPU->CPU fallback on CUDA OOM (policy 'auto').
    Optionally accepts a tpc_id for FiLM conditioning.
    """
    x = np.asarray(grid, dtype=np.float32)
    if x.shape != (32, 128, 64):
        raise ValueError(f"Expected grid shape (32,128,64), got {x.shape}")
    x = x[None, None, ...]  # (1,1,32,128,64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    if input_scale == "p99":
        p = np.percentile(x, 99, axis=(2,3,4), keepdims=True)
        p = np.where(p > 1e-6, p, 1.0)
        x = (x / p).astype(np.float32, copy=False)
    elif input_scale == "sum":
        s = x.sum(axis=(2,3,4), keepdims=True)
        s = np.where(s > 1e-6, s, 1.0)
        x = (x / s).astype(np.float32, copy=False)
    elif input_scale in (None, "none"):
        pass
    else:
        raise ValueError(f"Unknown input_scale '{input_scale}'")

    if device_policy == "force_cpu":
        cur_dev = torch.device("cpu")
    elif device_policy == "force_cuda":
        cur_dev = torch.device("cuda")
    else:
        cur_dev = torch.device(device) if device is not None else (
            next(model.parameters()).device
            if any(True for _ in model.parameters())
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    model = model.to(cur_dev).eval()

    def _extract_final(y_scaled):
        if isinstance(y_scaled, dict):
            y_scaled = y_scaled.get("final", None)
            if y_scaled is None:
                raise RuntimeError("Model returned a dict without 'final' key.")
        return y_scaled

    while True:
        try:
            xt = torch.from_numpy(x).to(device=cur_dev, dtype=torch.float32,
                                        non_blocking=(cur_dev.type == "cuda"))
            
            # Prepare TPC ID tensor if provided
            tids = None
            if tpc_id is not None:
                tids = torch.tensor([tpc_id], device=cur_dev, dtype=torch.long)

            y_scaled = _extract_final(model(xt, tpc_ids=tids))
            y_np = y_scaled.detach().to(torch.float32).cpu().numpy()  # (1,K)
            break
        except Exception as e:
            if cur_dev.type == "cuda" and device_policy == "auto" and _is_cuda_oom(e):
                warnings.warn("predict_phi_simple: CUDA OOM — falling back to CPU.", RuntimeWarning)
                try: torch.cuda.empty_cache()
                except Exception: pass
                cur_dev = torch.device("cpu"); model = model.to(cur_dev).eval()
                continue
            raise

    inv = 1.0 / float(target_scale) if target_scale not in (None, 0.0) else 1.0
    phi_raw = (y_np * inv)[0]  # (K,)
    return phi_raw

# -------------------------- Voxelization helpers --------------------------

# Geometry / binning parameters
NX, NY, NZ = 32, 128, 64
_X_RANGE_BY_TPC: Dict[int, Tuple[float, float]] = {
    0: ( 32.5,  64.5), 2: ( 32.5,  64.5),
    1: (  2.5,  34.5), 3: (  2.5,  34.5),
    4: (-34.5,  -2.5), 6: (-34.5,  -2.5),
    5: (-64.5, -32.5), 7: (-64.5, -32.5),
}
_POSZ_TPCS = {0, 1, 4, 5}
_NEGZ_TPCS = {2, 3, 6, 7}

def _bin_y(y: np.ndarray) -> np.ndarray:
    """y ∈ [-64,64) -> [0..127]"""
    idx = np.floor(y + 64.0).astype(np.int32)
    return np.clip(idx, 0, NY - 1)

def _bin_z(z: np.ndarray, tpc_id: int) -> np.ndarray:
    """Handle +z and -z TPC conventions exactly like prep script."""
    if tpc_id in _POSZ_TPCS:
        idx = np.floor(z).astype(np.int32)           # [0,64) -> [0..63]
    elif tpc_id in _NEGZ_TPCS:
        idx = np.floor(z + 64.0).astype(np.int32)    # [-64,0) -> [0..63]
    else:
        raise ValueError(f"Unknown TPC id: {tpc_id}")
    return np.clip(idx, 0, NZ - 1)

def _bin_x(x: np.ndarray, tpc_id: int) -> np.ndarray:
    """Map physical x to [0..31] depending on TPC’s allowed range."""
    if tpc_id not in _X_RANGE_BY_TPC:
        raise ValueError(f"Unknown TPC id: {tpc_id}")
    x_lo, _ = _X_RANGE_BY_TPC[tpc_id]
    idx = np.floor(x - x_lo).astype(np.int32)
    return np.clip(idx, 0, NX - 1)

def voxelize_xyzE_by_tpc(x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
                         tpc_ids, tpcs_order: Optional[List[int]] = None
                         ) -> Tuple[List[np.ndarray], List[int]]:  # (NX,NY,NZ)
    """
    Bin per-hit (x,y,z,E) into per-TPC voxel grids of shape (NX,NY,NZ).

    tpc_ids : int or 1D array-like
    tpcs_order : list[int] or None => controls output order
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z); E = np.asarray(E)
    n = x.shape[0]
    if not (y.shape[0] == z.shape[0] == E.shape[0] == n):
        raise ValueError("x, y, z, E must be 1D arrays of the same length.")

    if np.isscalar(tpc_ids) or isinstance(tpc_ids, (int, np.integer)):
        tpc_scalar = int(tpc_ids)
        tpc_ids_arr = np.full(n, tpc_scalar, dtype=int)
        scalar_mode = True
    else:
        tpc_ids_arr = np.asarray(tpc_ids)
        if tpc_ids_arr.shape[0] != n:
            raise ValueError("tpc_ids must be scalar or a 1D array matching x/y/z/E length.")
        scalar_mode = False

    if tpcs_order is None:
        if scalar_mode:
            tpcs_order = [int(tpc_ids_arr[0])] if n > 0 else [int(tpc_scalar)]
        else:
            _, first_idx = np.unique(tpc_ids_arr, return_index=True)
            tpcs_order = [int(tpc_ids_arr[i]) for i in np.sort(first_idx)]
    else:
        tpcs_order = list(map(int, tpcs_order))

    out_maps: List[np.ndarray] = []
    outTPCs: List[int] = []

    for tpc in tpcs_order:
        sel = (tpc_ids_arr == tpc)
        if not np.any(sel):
            out_maps.append(np.zeros((NX, NY, NZ), dtype=np.float32))
            outTPCs.append(tpc)
            continue

        xs = x[sel]; ys = y[sel]; zs = z[sel]; Es = E[sel].astype(np.float32, copy=False)
        ix = _bin_x(xs, tpc); iy = _bin_y(ys); iz = _bin_z(zs, tpc)
        lin = ix.astype(np.int64) * (NY * NZ) + iy.astype(np.int64) * NZ + iz.astype(np.int64)
        acc = np.bincount(lin, weights=Es, minlength=NX * NY * NZ).astype(np.float32, copy=False)
        grid = acc.reshape(NX, NY, NZ)

        out_maps.append(grid)
        outTPCs.append(tpc)

    return out_maps, outTPCs

# ---------------- Vectorized grouping voxelizers ----------------

def group_voxelize(x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
                   tpc_ids: np.ndarray, labels: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one (NX,NY,NZ) map for every (cluster_id, TPCid) pair present. Excludes labels == -1.
    Returns maps (G,NX,NY,NZ), group_cls (G,), group_tpcs (G,)
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z); E = np.asarray(E)
    tpc_ids = np.asarray(tpc_ids); labels = np.asarray(labels)
    n = x.shape[0]
    if not (y.shape[0] == z.shape[0] == E.shape[0] == tpc_ids.shape[0] == labels.shape[0] == n):
        raise ValueError("x, y, z, E, tpc_ids, labels must be same-length 1D arrays.")
    valid = labels >= 0
    if not np.any(valid):
        return (np.zeros((0, NX, NY, NZ), dtype=np.float32),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=int))

    x = x[valid]; y = y[valid]; z = z[valid]; E = E[valid].astype(np.float32, copy=False)
    lab = labels[valid].astype(np.int64, copy=False)
    tpc = tpc_ids[valid].astype(np.int64, copy=False)

    pairs = np.column_stack([lab, tpc]).astype(np.int64, copy=False).view([('lab', np.int64), ('tpc', np.int64)])
    uniq, inv, first_idx = np.unique(pairs, return_inverse=True, return_index=True)
    order = np.argsort(first_idx)
    old2new = np.empty(order.size, dtype=np.int64); old2new[order] = np.arange(order.size, dtype=np.int64)
    g = old2new[inv]

    group_cls  = uniq['lab'][order].astype(int, copy=False)
    group_tpcs = uniq['tpc'][order].astype(int, copy=False)
    G = group_cls.size

    x_lo = np.empty_like(x, dtype=np.float64)
    for tpc_val in np.unique(tpc):
        if int(tpc_val) not in _X_RANGE_BY_TPC:
            raise ValueError(f"Unknown TPC id: {int(tpc_val)}")
        x_lo[tpc == tpc_val] = _X_RANGE_BY_TPC[int(tpc_val)][0]
    ix = np.floor(x - x_lo).astype(np.int32); ix = np.clip(ix, 0, NX-1)

    iy = _bin_y(y).astype(np.int32)

    pos_mask = np.isin(tpc, list(_POSZ_TPCS))
    if not np.all(np.isin(tpc, list(_POSZ_TPCS) + list(_NEGZ_TPCS))):
        unknown = np.unique(tpc[~np.isin(tpc, list(_POSZ_TPCS) + list(_NEGZ_TPCS))])
        raise ValueError(f"Unknown TPC ids for z-binning: {unknown.tolist()}")
    z_shifted = np.where(pos_mask, z, z + 64.0)
    iz = np.floor(z_shifted).astype(np.int32); iz = np.clip(iz, 0, NZ-1)

    V = NX * NY * NZ
    lin = ix.astype(np.int64) * (NY * NZ) + iy.astype(np.int64) * NZ + iz.astype(np.int64)
    global_lin = g.astype(np.int64) * V + lin
    acc = np.bincount(global_lin, weights=E, minlength=int(G)*V).astype(np.float32, copy=False)
    maps = acc.reshape(G, V).reshape(G, NX, NY, NZ)

    return maps, group_cls, group_tpcs


def _select_clusters_by_E(E: np.ndarray, labels: np.ndarray, Ethreshold: float) -> np.ndarray:
    """Return sorted unique cluster ids (>=0) whose total E >= Ethreshold."""
    lab = labels.astype(np.int64, copy=False)
    mask = lab >= 0
    if not np.any(mask):
        return np.array([], dtype=int)
    labv = lab[mask]
    Ev   = E[mask].astype(np.float64, copy=False)
    max_lab = int(labv.max())
    sums = np.bincount(labv, weights=Ev, minlength=max_lab+1)
    keep = np.flatnonzero(sums >= float(Ethreshold))
    return keep.astype(int, copy=False)

def group_voxelize_pairs(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_ids: np.ndarray, labels: np.ndarray,
    *, include_noise: bool = False,
    restrict_clusters: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized: one (NX,NY,NZ) map for every (cluster_id, TPCid) pair present.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    E = np.asarray(E, dtype=np.float32)
    tpc_ids = np.asarray(tpc_ids)
    labels  = np.asarray(labels)

    n = len(x)
    if not (len(y) == len(z) == len(E) == len(tpc_ids) == len(labels) == n):
        raise ValueError("x,y,z,E,tpc_ids,labels must be same-length 1D arrays.")

    valid = (labels >= 0) if not include_noise else np.ones_like(labels, dtype=bool)
    if restrict_clusters is not None:
        rc = np.asarray(list(restrict_clusters), dtype=labels.dtype)
        valid &= np.isin(labels, rc)

    if not np.any(valid):
        return (np.zeros((0, NX, NY, NZ), np.float32),
                np.zeros((0,), int), np.zeros((0,), int))

    x = x[valid]; y = y[valid]; z = z[valid]; E = E[valid]
    lab = labels[valid].astype(np.int64, copy=False)
    tpc = tpc_ids[valid].astype(np.int64, copy=False)

    STRIDE = 16
    keys = lab * STRIDE + tpc
    uniq_keys, first_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    G = int(uniq_keys.size)

    rank_by_first = np.empty(G, dtype=np.int64)
    rank_by_first[np.argsort(first_idx)] = np.arange(G, dtype=np.int64)
    g = rank_by_first[inv]

    pos = first_idx[np.argsort(first_idx)]
    group_cls  = lab[pos].astype(int, copy=False)
    group_tpcs = tpc[pos].astype(int, copy=False)

    x_lo = np.empty_like(x, dtype=np.float64)
    for tpc_val in np.unique(tpc):
        if int(tpc_val) not in _X_RANGE_BY_TPC:
            raise ValueError(f"Unknown TPC id for x-binning: {int(tpc_val)}")
        x_lo[tpc == tpc_val] = _X_RANGE_BY_TPC[int(tpc_val)][0]
    ix = np.floor(x - x_lo).astype(np.int32); ix = np.clip(ix, 0, NX-1)

    iy = _bin_y(y).astype(np.int32)

    all_known = np.isin(np.unique(tpc), list(_POSZ_TPCS) + list(_NEGZ_TPCS))
    if not np.all(all_known):
        missing = np.unique(tpc)[~all_known].tolist()
        raise ValueError(f"Unknown TPC ids for z-binning: {missing}")
    iz = np.where(np.isin(tpc, list(_POSZ_TPCS)), np.floor(z), np.floor(z + 64.0)).astype(np.int32)
    iz = np.clip(iz, 0, NZ-1)

    V = NX * NY * NZ
    lin = ix.astype(np.int64)*(NY*NZ) + iy.astype(np.int64)*NZ + iz.astype(np.int64)
    global_lin = g.astype(np.int64) * V + lin
    acc = np.bincount(global_lin, weights=E, minlength=int(G)*V).astype(np.float32, copy=False)
    maps_4d = acc.reshape(G, NX, NY, NZ)

    return maps_4d, group_cls, group_tpcs

# ------------------------- φ -> reorder -> waveforms -------------------------

def _make_pair_perm(K: int) -> Optional[np.ndarray]:
    """
    Reorder model outputs from [0..K/2-1 | K/2..K-1] to [(0,K/2),(1,K/2),... reversed by Y)].
    For K=48 this reproduces your prior permutation.
    """
    if K % 2 != 0:
        return None
    half = K // 2
    return np.fromiter((i for y in range(half-1, -1, -1) for i in (y, half + y)), dtype=int, count=K)

def predict_multi_phi_to_waveforms(
    maps_4d: np.ndarray,               # (G, NX, NY, NZ)
    tpcs_for_maps: np.ndarray,         # (G,)
    model_01, model_other,
    *, expected_outputs: int = 48,
    input_scale: str = "p99", target_scale: float = 1e-3,
    template: Optional[np.ndarray] = None,   # (L,), if None falls back to ones(L=1000)
    use_reorder: bool = True,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1) run predict_phi (dual-model routing) to get amplitudes (G,K)
    2) optional channel reorder (K stays K)
    3) expand to waveforms (G,K,L) by broadcasting with template
    """
    amps = predict_phi(
        maps_4d, model_01=model_01, model_other=model_other,
        tpcs_for_grids=tpcs_for_maps,
        input_scale=input_scale, target_scale=target_scale,
        expected_outputs=expected_outputs, batch_size=batch_size
    ).astype(np.float32, copy=False)                     # (G,K)

    K = amps.shape[1]
    if expected_outputs is not None and K != expected_outputs:
        raise ValueError(f"Model produced K={K}, expected {expected_outputs}.")

    if use_reorder:
        perm = _make_pair_perm(K)
        if perm is not None and perm.size == K:
            amps = amps[:, perm]

    if template is None:
        L = 1000
        tmpl = np.ones((L,), dtype=np.float32)
    else:
        tmpl = np.asarray(template, dtype=np.float32)
        if tmpl.ndim != 1:
            raise ValueError("template must be 1D.")
        L = int(tmpl.size)

    wave = amps[:, :, None] * tmpl[None, None, :]        # (G,K,L)
    return amps, wave

# ------------------------- Packaging -------------------------

def build_image_map_dict(
    wave: np.ndarray,                  # (G,K,L)
    group_cls: np.ndarray,             # (G,)
    group_tpcs: np.ndarray,            # (G,)
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Map (cluster_id, tpc_id) -> waveforms[K,L] (views).
    """
    imageMaps: Dict[Tuple[int, int], np.ndarray] = {}
    for g, (c, t) in enumerate(zip(group_cls, group_tpcs)):
        imageMaps[(int(c), int(t))] = wave[g]            # view
    return imageMaps

# ===================== process_clusters_to_imageMaps =====================

def process_clusters_to_imageMaps(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, E: np.ndarray,
    tpc_ids: np.ndarray, labels: np.ndarray,
    *, 
    model_01, model_other,
    expected_outputs: int = 48,
    input_scale: str = "none", target_scale: float = 1e-3,
    template: Optional[np.ndarray] = None,
    include_noise: bool = False,
    batch_size: int = 32,
) -> Tuple[Dict[Tuple[int,int], np.ndarray], Dict[str, np.ndarray]]:
    """
    Build voxel maps for every (cluster_id, tpc_id) pair present (labels==-1 excluded by default),
    predict amplitudes with dual-model routing, reorder channels, expand to waveforms, and return:

        imageMaps[(cluster_id, tpc_id)] -> (K, L) waveforms (views into the big array)

    meta includes group arrays and the dense tensors.
    """
    maps_4d, group_cls, group_tpcs = group_voxelize_pairs(
        x, y, z, E, tpc_ids, labels,
        include_noise=include_noise, restrict_clusters=None
    )

    if maps_4d.shape[0] == 0:
        empty_wave_len = (template.size if (template is not None) else 1000)
        return {}, {
            "group_cls":   np.array([], dtype=int),
            "group_tpcs":  np.array([], dtype=int),
            "maps_4d":     np.zeros((0, NX, NY, NZ), dtype=np.float32),
            "amplitudes":  np.zeros((0, expected_outputs), dtype=np.float32),
            "waveforms":   np.zeros((0, expected_outputs, empty_wave_len), dtype=np.float32),
        }

    amps, wave = predict_multi_phi_to_waveforms(
        maps_4d, group_tpcs,
        model_01=model_01, model_other=model_other,
        expected_outputs=expected_outputs,
        input_scale=input_scale, target_scale=target_scale,
        template=template, use_reorder=True, batch_size=batch_size
    )

    imageMaps = build_image_map_dict(wave, group_cls, group_tpcs)

    meta = {
        "group_cls":  group_cls,   # (G,)
        "group_tpcs": group_tpcs,  # (G,)
        "maps_4d":    maps_4d,     # (G, NX, NY, NZ)
        "amplitudes": amps,        # (G, K)
        "waveforms":  wave,        # (G, K, L)
    }
    return imageMaps, meta

__all__ = [
    "ResBlock3D", "DownBlock3D", "MLPBlock", "HeadMLP",
    "Model48Single", "Model48TypeSplit", "CNN12Compat",
    "load_cnn_model_flex",
    "predict_phi", "predict_phi_simple",
    "voxelize_xyzE_by_tpc", "group_voxelize",
    "group_voxelize_pairs",
    "predict_multi_phi_to_waveforms",
    "build_image_map_dict",
    "process_clusters_to_imageMaps",
    "NX", "NY", "NZ",
]
