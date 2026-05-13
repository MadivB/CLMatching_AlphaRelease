#!/usr/bin/env python3
"""
train_ndfull.py

A hybrid 3D CNN + Transformer Decoder model for 120-target prediction adapted to ND-full geometries.
Architecture:
  1. Scaled 3D ResNet Stem (64 channels) + Squeeze-and-Excitation (SE) blocks.
  2. Spatial flattening + Deterministic 3D Sinusoidal Positional Encoding (built dynamically based on features shape).
  3. TPC FiLM conditioning on the 3D feature maps (covering 0-71 TPCs).
  4. Transformer Decoder (Pre-LN): 120 Learned Queries alternate between Self-Attention
     (inter-target communication) and Cross-Attention (looking at 3D features).
  5. Independent regression heads per target token.
  6. Energy-weighted Huber loss to prioritize massive charge/light deposits.
"""
from __future__ import annotations
import os, math, json, time, argparse, random, copy
from typing import List, Sequence, Tuple, Dict, Any, Optional

import numpy as np
import zarr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- AMP shim ----------------
try:
    from torch import amp as _amp
    def _autocast(enabled: bool): return _amp.autocast("cuda", enabled=enabled)
    def _GradScaler(enabled: bool): return _amp.GradScaler("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _cuda_autocast, GradScaler as _CudaGradScaler
    def _autocast(enabled: bool): return _cuda_autocast(enabled=enabled)
    def _GradScaler(enabled: bool): return _CudaGradScaler(enabled=enabled)

# ------------- Distributed Utilities -------------
def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_env_rank_world_local() -> Tuple[int,int,int]:
    r = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    w = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    lr = int(os.environ.get("LOCAL_RANK", 0))
    return r, w, lr

def is_main_process() -> bool:
    return (not ddp_is_initialized()) or dist.get_rank() == 0

def ddp_all_reduce_(t: torch.Tensor):
    if ddp_is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ---------------- Dataset ----------------
def _open_group_r(path: str) -> zarr.hierarchy.Group:
    """
    Robustly open a Zarr group in read-only mode.
    """
    try:
        return zarr.open(path, mode='r')
    except Exception:
        # Fallback for complex store setups
        store = zarr.DirectoryStore(path)
        return zarr.open_group(store=store, mode='r')

class MultiZarrDataset(Dataset):
    def __init__(self, zarr_paths: Sequence[str], input_scale="none", target_scale=1e-3, upcast_to="float32"):
        self.groups = [_open_group_r(p) for p in zarr_paths]
        self.vox = [g["voxels"] for g in self.groups]
        self.tgt = [g["targets"] for g in self.groups]
        
        self.tpcds = []
        for g in self.groups:
            # We must use 'charge_tpc_ids' for the input feature mapping in NDLAr-full zarrs
            if "charge_tpc_ids" in g: self.tpcds.append(g["charge_tpc_ids"])
            elif "tpc_ids" in g: self.tpcds.append(g["tpc_ids"])
            elif "event_ids" in g: self.tpcds.append(g["event_ids"])
            else: self.tpcds.append(None)

        self.sizes = [v.shape[0] for v in self.vox]
        self.cum = np.cumsum([0] + self.sizes).tolist()
        self.N = self.cum[-1]
        self.C = int(self.vox[0].shape[1])
        self.K = int(self.tgt[0].shape[1])
        self.input_scale = input_scale
        self.target_scale = float(target_scale)
        self.upcast_to = upcast_to

    def __len__(self): return self.N

    def _map(self, idx: int) -> Tuple[int, int]:
        if idx < 0: idx += self.N
        for i in range(len(self.sizes)):
            if idx < self.cum[i+1]: return i, idx - self.cum[i]
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        fi, li = self._map(idx)
        x = np.asarray(self.vox[fi][li], dtype=np.float32)
        y = np.asarray(self.tgt[fi][li], dtype=np.float32)
        
        tpcid = 0
        if self.tpcds[fi] is not None:
            val = self.tpcds[fi][li]
            tpcid = int(val[1] if isinstance(val, np.ndarray) and val.size > 1 else val)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.input_scale == "sum":
            s = x.sum(axis=(1,2,3), keepdims=True); s = np.where(s > 1e-6, s, 1.0)
            x = x / s
        elif self.input_scale == "p99":
            p = np.percentile(x, 99, axis=(1,2,3), keepdims=True); p = np.where(p > 1e-6, p, 1.0)
            x = x / p

        if self.upcast_to: x = x.astype(self.upcast_to, copy=False)
        y = y * self.target_scale
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(tpcid, dtype=torch.long)


class TargetsOnlyDataset(Dataset):
    """Lightweight dataset that reads ONLY targets (no voxels).
    Used for fast standardizer fitting — avoids loading 6MB voxel
    grids unnecessarily (600GB I/O on 100k samples)."""
    def __init__(self, full_dataset: MultiZarrDataset):
        # Re-use the already-open zarr handles from the full dataset
        self.tgt    = full_dataset.tgt
        self.sizes  = full_dataset.sizes
        self.cum    = full_dataset.cum
        self.N      = full_dataset.N
        self.K      = full_dataset.K
        self.scale  = full_dataset.target_scale

    def __len__(self): return self.N

    def _map(self, idx):
        if idx < 0: idx += self.N
        for i in range(len(self.sizes)):
            if idx < self.cum[i+1]: return i, idx - self.cum[i]
        raise IndexError(idx)

    def __getitem__(self, idx):
        fi, li = self._map(idx)
        y = np.asarray(self.tgt[fi][li], dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(y * self.scale)

# ---------------- Advanced Architecture ----------------
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
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)
        # Force float32 for SE linear layers to avoid cuBLAS errors in some Torch/CUDA environments
        with _autocast(enabled=False):
            y = self.fc(y.float())
        y = y.reshape(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResBlock3D_SE(nn.Module):
    def __init__(self, ch, hidden=None, gn_groups=8):
        super().__init__()
        h = hidden or ch
        self.conv1 = nn.Conv3d(ch, h, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(gn_groups, h), num_channels=h)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(h, ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)
        self.se = SEBlock3D(ch)

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.se(self.gn2(self.conv2(y)))
        return self.act(x + y)

class DownBlock3D_SE(nn.Module):
    def __init__(self, c_in, c_out, stride=(2,2,2), gn_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(gn_groups, c_out), num_channels=c_out)
        self.act = nn.SiLU(inplace=True)
        self.res = ResBlock3D_SE(c_out, gn_groups=gn_groups)

    def forward(self, x):
        x = self.act(self.gn(self.conv(x)))
        return self.res(x)

class PerceiverDecoder(nn.Module):
    def __init__(self, embed_dim, num_targets=120, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        # Expanded to 120 targets
        self.target_queries = nn.Parameter(torch.randn(1, num_targets, embed_dim))
        nn.init.normal_(self.target_queries, std=0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True  
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim) 

    def forward(self, memory):
        B = memory.size(0)
        q = self.target_queries.expand(B, -1, -1) 
        out = self.decoder(q, memory) 
        return self.final_norm(out)

def create_3d_sinusoidal_pos(D, H, W, dim):
    """
    Dynamically generates the 3D encoding based on the passed D, H, W structure.
    """
    dim_z = (dim // 3) // 2 * 2
    dim_y = (dim // 3) // 2 * 2
    dim_x = dim - dim_z - dim_y 
    
    z = torch.arange(D).unsqueeze(-1).float()
    div_term_z = torch.exp(torch.arange(0, dim_z, 2).float() * -(math.log(10000.0) / max(1, dim_z)))
    pe_z = torch.zeros(D, dim_z)
    if dim_z > 0:
        pe_z[:, 0::2] = torch.sin(z * div_term_z)
        pe_z[:, 1::2] = torch.cos(z * div_term_z)
    
    y = torch.arange(H).unsqueeze(-1).float()
    div_term_y = torch.exp(torch.arange(0, dim_y, 2).float() * -(math.log(10000.0) / max(1, dim_y)))
    pe_y = torch.zeros(H, dim_y)
    if dim_y > 0:
        pe_y[:, 0::2] = torch.sin(y * div_term_y)
        pe_y[:, 1::2] = torch.cos(y * div_term_y)
    
    x = torch.arange(W).unsqueeze(-1).float()
    div_term_x = torch.exp(torch.arange(0, dim_x, 2).float() * -(math.log(10000.0) / max(1, dim_x)))
    pe_x = torch.zeros(W, dim_x)
    if dim_x > 0:
        pe_x[:, 0::2] = torch.sin(x * div_term_x)
        pe_x[:, 1::2] = torch.cos(x * div_term_x)
    
    pe_z = pe_z.view(D, 1, 1, dim_z).expand(D, H, W, dim_z)
    pe_y = pe_y.view(1, H, 1, dim_y).expand(D, H, W, dim_y)
    pe_x = pe_x.view(1, 1, W, dim_x).expand(D, H, W, dim_x)
    
    pe = torch.cat([pe_z, pe_y, pe_x], dim=-1)
    return pe.view(1, -1, dim) 

class HybridPerceiver3D(nn.Module):
    # num_tpcs expanded to 72 to safely map up to NDLAr physical tracks
    def __init__(self, c_in=1, base_channels=64, embed_dim=512, num_targets=120, 
                 tpc_embed_dim=32, num_tpcs=72, num_decoder_layers=4, dropout=0.1):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv3d(c_in, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
            ResBlock3D_SE(base_channels),
        )
        self.down1 = DownBlock3D_SE(base_channels, base_channels * 2)
        self.down2 = DownBlock3D_SE(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock3D_SE(base_channels * 4, embed_dim) 
        
        # We drop the hardcoded pos_embedding initialization because we'll build it lazily on the first pass!
        self.embed_dim = embed_dim
        self.pos_embedding = None
        
        self.tpc_emb = nn.Embedding(num_tpcs, tpc_embed_dim)
        self.tpc_film_gamma = nn.Linear(tpc_embed_dim, embed_dim)
        self.tpc_film_beta = nn.Linear(tpc_embed_dim, embed_dim)
        
        self.decoder = PerceiverDecoder(
            embed_dim=embed_dim, num_targets=num_targets, 
            num_layers=num_decoder_layers, num_heads=8, dropout=dropout
        )
        
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x, tpc_ids):
        x = self.stem(x); x = self.down1(x); x = self.down2(x); x = self.down3(x)
        B, C, D, H, W = x.size()
        
        # Lazy initialization of pos_embedding structure
        if self.pos_embedding is None or self.pos_embedding.shape[1] != (D*H*W):
            pe = create_3d_sinusoidal_pos(D, H, W, self.embed_dim).to(x.device)
            self.pos_embedding = pe

        x = x.view(B, C, -1).transpose(1, 2) 
        
        e = self.tpc_emb(tpc_ids)             
        gamma = self.tpc_film_gamma(e).unsqueeze(1)
        beta  = self.tpc_film_beta(e).unsqueeze(1)
        x = x * (1.0 + gamma) + beta
        
        x = x + self.pos_embedding
        target_features = self.decoder(x) 
        
        final_preds = self.regressor(target_features).squeeze(-1) 
        return {"final": final_preds}

# ---------------- Masked Utilities & Stats ----------------
def masked_mean(t: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    m = mask.bool()
    if not m.any(): return None
    return t[m].mean()

class TargetStandardizer:
    def __init__(self, mean, std):
        self.mean = mean.view(1, -1)
        self.std  = torch.clamp(std.view(1, -1), min=1e-6)
    def encode(self, y): return (y - self.mean) / self.std
    def decode(self, yz): return yz * self.std + self.mean

@torch.no_grad()
def fit_target_standardizer_masked(loader, device, sentinel_scaled: float, zero_below_scaled: float):
    # Get underlying full dataset regardless of wrapping (Subset, etc)
    ds_full = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    K = ds_full.K
    # Use a lightweight targets-only loader — avoids loading 6MB voxel grids per sample
    tgt_ds     = TargetsOnlyDataset(ds_full)
    tgt_loader = DataLoader(tgt_ds, batch_size=256, shuffle=False,
                            num_workers=min(16, os.cpu_count() or 4),
                            pin_memory=True)
    s1 = torch.zeros(K, device=device); s2 = torch.zeros(K, device=device); n = torch.zeros(K, device=device)
    
    for y in tgt_loader:
        y = y.to(device)
        alive = (y != sentinel_scaled)
        y_eff = torch.where(alive & (y < zero_below_scaled), torch.zeros_like(y), y)
        s1 += (y_eff * alive).sum(dim=0)
        s2 += ((y_eff ** 2) * alive).sum(dim=0)
        n  += alive.sum(dim=0)
        
    ddp_all_reduce_(s1); ddp_all_reduce_(s2); ddp_all_reduce_(n)
    n = torch.clamp(n, min=1.0)
    mean = s1 / n
    std  = torch.sqrt(torch.clamp((s2 / n) - mean**2, min=1e-12))
    return TargetStandardizer(mean.detach(), std.detach())

@torch.no_grad()
def evaluate_distributed(model, loader, device, amp, stdzr, inv_target_scale, sentinel_scaled, zero_below_scaled):
    model.eval()
    K = loader.dataset.dataset.K if hasattr(loader.dataset, "dataset") else loader.dataset.K
    sum_err2 = torch.zeros(K, device=device); sum_abs = torch.zeros(K, device=device)
    sum_y = torch.zeros(K, device=device); sum_y2 = torch.zeros(K, device=device)
    cnt = torch.zeros(K, device=device)
    total_err2_scaled = torch.zeros(1, device=device); total_cnt = torch.zeros(1, device=device)

    for x, y, tpc in loader:
        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        valid = (y != sentinel_scaled)
        y_eff = torch.where(valid & (y < zero_below_scaled), torch.zeros_like(y), y)

        with _autocast(amp):
            pred = model(x, tpc_ids=tpc)["final"]
            p_in, y_in = (stdzr.encode(pred), stdzr.encode(y_eff)) if stdzr else (pred, y_eff)
            
            err2 = (p_in - y_in) ** 2
            total_err2_scaled += err2[valid].sum()
            total_cnt += valid.sum()

            inv = float(inv_target_scale)
            Praw, Yraw = pred * inv, y_eff * inv
            
            sum_abs += ((Praw - Yraw).abs() * valid).sum(dim=0)
            sum_err2 += (((Praw - Yraw) ** 2) * valid).sum(dim=0)
            sum_y += (Yraw * valid).sum(dim=0)
            sum_y2 += ((Yraw**2) * valid).sum(dim=0)
            cnt += valid.sum(dim=0)

    for t in (sum_err2, sum_abs, sum_y, sum_y2, cnt, total_err2_scaled, total_cnt): ddp_all_reduce_(t)

    m_cnt = torch.clamp(cnt, min=1.0)
    mse_k, mae_k = sum_err2 / m_cnt, sum_abs / m_cnt
    var_k = torch.clamp(sum_y2 / m_cnt - (sum_y / m_cnt)**2, min=1e-12)
    r2_k = 1.0 - (mse_k / var_k)

    ok = cnt > 0
    _avg = lambda v: float(v[ok].mean().item()) if ok.any() else float("nan")

    return {
        "loss": float((total_err2_scaled / torch.clamp(total_cnt, min=1.0)).item()),
        "mse": _avg(mse_k), "mae": _avg(mae_k), "r2": _avg(r2_k),
    }

# ---------------- Training ----------------
def cosine_warmup_lr(step, total_steps, base_lr, warmup_steps=0):
    if step < warmup_steps: return base_lr * (step / max(1, warmup_steps))
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))

def train_sub_epoch(
    model, loader, loader_iter, opt, scaler, device, base_lr, warmup_steps,
    total_steps, global_step, stdzr, amp, sentinel_scaled, zero_below_scaled,
    sub_epoch_size, inv_target_scale,
):
    """Run at most sub_epoch_size batches; return (avg_loss, global_step, loader_iter, tr_r2)."""
    model.train(); total_loss = 0.0; n = 0; step = global_step
    K = None
    sum_err2 = sum_y = sum_y2 = cnt = None

    for _ in range(sub_epoch_size):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x, y, tpc = batch
        lr_now = cosine_warmup_lr(step, total_steps, base_lr, warmup_steps)
        for g in opt.param_groups: g["lr"] = lr_now

        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        valid_mask = (y != sentinel_scaled)
        y_eff = torch.where(valid_mask & (y < zero_below_scaled), torch.zeros_like(y), y)

        if K is None:
            K = y.shape[1]
            sum_err2 = torch.zeros(K, device=device)
            sum_y    = torch.zeros(K, device=device)
            sum_y2   = torch.zeros(K, device=device)
            cnt      = torch.zeros(K, device=device)

        opt.zero_grad(set_to_none=True)
        with _autocast(amp):
            pred = model(x, tpc_ids=tpc)["final"]
            p_in, y_in = (stdzr.encode(pred), stdzr.encode(y_eff)) if stdzr else (pred, y_eff)

            loss_vec = F.smooth_l1_loss(p_in, y_in, reduction="none")
            weights = (y_in.abs() + 1.0)
            weighted_loss_vec = loss_vec * weights
            loss = masked_mean(weighted_loss_vec, valid_mask)

        if loss is None or not torch.isfinite(loss):
            step += 1; continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs; n += bs; step += 1

        # Accumulate train R² stats
        with torch.no_grad():
            inv = float(inv_target_scale)
            Praw = pred.detach() * inv
            Yraw = y_eff * inv
            sum_err2 += (((Praw - Yraw)**2) * valid_mask).sum(0)
            sum_y    += (Yraw * valid_mask).sum(0)
            sum_y2   += ((Yraw**2) * valid_mask).sum(0)
            cnt      += valid_mask.sum(0)

    loss_tensor = torch.tensor([total_loss, n], device=device, dtype=torch.float64)
    ddp_all_reduce_(loss_tensor)
    avg_loss = float(loss_tensor[0].item()) / max(1, int(loss_tensor[1].item()))

    # Compute train R²
    tr_r2 = float("nan")
    if K is not None:
        for t in (sum_err2, sum_y, sum_y2, cnt): ddp_all_reduce_(t)
        ok = cnt > 0; cnt_c = cnt.clamp(min=1.)
        mse_k = sum_err2 / cnt_c
        var_k = (sum_y2 / cnt_c - (sum_y / cnt_c)**2).clamp(min=1e-12)
        r2_k  = 1. - mse_k / var_k
        tr_r2 = float(r2_k[ok].mean()) if ok.any() else float("nan")

    return avg_loss, step, loader_iter, tr_r2

# ---------------- Entry ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16, help="Per GPU batch size") 
    ap.add_argument("--lr", type=float, default=3e-4) 
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--out-dir", type=str, default="./runs/train_ndfull_run")
    ap.add_argument("--dead-sentinel", type=float, default=-10000.0)
    ap.add_argument("--zero-below-raw", type=float, default=100.0)
    ap.add_argument("--target-scale", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP). Set --no-amp if cuBLAS errors occur.")
    ap.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP.")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint in out-dir if it exists.")
    ap.add_argument("--debug-limit", type=int, default=0, help="If >0, limit training iterator to this many steps initially to debug architectures")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # DDP Init
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, world_size, local_rank = get_env_rank_world_local()
    device = torch.device("cuda", local_rank)

    if is_main_process():
        os.makedirs(args.out_dir, exist_ok=True)
        print("Starting 3D Cross-Attention Perceiver Training optimized for NDLAr-full geometries!")

    # Dataset & Loaders
    zpaths = sorted([os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if d.endswith(".zarr")])
    ds_full = MultiZarrDataset(zpaths, target_scale=args.target_scale)
    n_val = max(1, int(args.val_frac * len(ds_full))); n_train = len(ds_full) - n_val
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_sampler = DistributedSampler(ds_train, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(ds_val, shuffle=False)
    
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)

    # Model definition
    model = HybridPerceiver3D(c_in=ds_full.C, num_targets=ds_full.K).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = _GradScaler(enabled=True)

    # Standardization mapping
    sentinel_scaled = args.dead_sentinel * args.target_scale
    zero_below_scaled = args.zero_below_raw * args.target_scale
    if is_main_process(): print("Fitting Target Standardizers...")
    stdzr = fit_target_standardizer_masked(train_loader, device, sentinel_scaled, zero_below_scaled)

    N_SUBEPOCHS = 5
    total_steps = args.epochs * N_SUBEPOCHS * (len(train_loader) // N_SUBEPOCHS + 1)
    inv_target_scale = 1.0 / args.target_scale

    # --- RESUME LOGIC ---
    start_sub_epoch = 0
    global_step = 0
    best_val = float('inf')

    if args.resume:
        checkpoint_path = os.path.join(args.out_dir, "checkpoint.pt")
        best_model_path = os.path.join(args.out_dir, "best_model.pt")

        if os.path.exists(checkpoint_path):
            if is_main_process(): print(f"Resuming training from full checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model.module.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            scaler.load_state_dict(ckpt["scaler"])
            start_sub_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_val = ckpt["best_val"]
        elif os.path.exists(best_model_path):
            if is_main_process(): print(f"Full checkpoint not found. Loading weights from: {best_model_path}")
            model.module.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        else:
            if is_main_process(): print("No checkpoint or best_model found to resume from. Starting fresh.")

    if args.debug_limit > 0 and is_main_process():
        print(f"!!! DEBUG LIMIT ENABLED: Stepping {args.debug_limit} iteration ONLY to verify shape tracking.")
        args.epochs += 1

    sub_epoch_size = max(1, len(train_loader) // N_SUBEPOCHS)
    total_sub_epochs = args.epochs * N_SUBEPOCHS
    loader_iter = iter(train_loader)

    # Main Loop (sub-epoch = 1/5 of full dataset)
    for sub_epoch in range(start_sub_epoch, total_sub_epochs):
        if sub_epoch % N_SUBEPOCHS == 0:
            train_sampler.set_epoch(sub_epoch // N_SUBEPOCHS)
            loader_iter = iter(train_loader)

        if args.debug_limit > 0:
            print("Running shapes test pass...")
            for idx, (bx, by, bt) in enumerate(train_loader):
                if idx >= args.debug_limit: break
                model(bx.to(device), tpc_ids=bt.to(device))
                print(f"Shapes map OK! Step {idx+1}/{args.debug_limit}")
            break

        train_loss, global_step, loader_iter, tr_r2 = train_sub_epoch(
            model, train_loader, loader_iter, opt, scaler, device,
            args.lr, args.warmup_steps, total_steps, global_step, stdzr,
            args.amp, sentinel_scaled, zero_below_scaled,
            sub_epoch_size, inv_target_scale,
        )

        vm = evaluate_distributed(model, val_loader, device, args.amp, stdzr,
                                  inv_target_scale, sentinel_scaled, zero_below_scaled)

        if is_main_process():
            real_epoch   = sub_epoch // N_SUBEPOCHS + 1
            sub_in_epoch = sub_epoch % N_SUBEPOCHS + 1
            print(f"Epoch {real_epoch:03d}.{sub_in_epoch}/5 | "
                  f"Train Loss: {train_loss:.4f} | Train R2: {tr_r2:.4f} | "
                  f"Val MSE: {vm['mse']:.4f} | Val R2: {vm['r2']:.4f}")

            torch.save({
                "epoch": sub_epoch,
                "model": model.module.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
                "best_val": best_val
            }, os.path.join(args.out_dir, "checkpoint.pt"))

            if vm['mse'] < best_val:
                best_val = vm['mse']
                torch.save(model.module.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
                print(f"  ✓ Saved new best model")

    if ddp_is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
