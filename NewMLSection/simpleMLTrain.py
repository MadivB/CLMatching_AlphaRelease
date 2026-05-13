#!/usr/bin/env python3
"""
simpleMLTrain.py

A compact 3D CNN + Global Average Pooling model for 120-target prediction
designed specifically for small datasets (~4000 samples).

Philosophy: reduce overfitting by:
  - Fewer parameters (compact 2-block CNN, no Transformer)
  - Global Average Pooling to collapse spatial dimensions
  - Higher dropout (0.3) + strong weight decay (1e-3)
  - Simple shared MLP head for all 120 targets

Architecture:
  Input: (B, C, 50, 300, 100) voxel grid
  → Stem Conv3D (C→32)
  → DownBlock (32→64, stride 2)
  → DownBlock (64→128, stride 2)
  → Global Average Pool → (B, 128)
  → TPC embedding concatenated → (B, 128 + 16)
  → MLP → (B, 120)
"""
from __future__ import annotations
import os, math, argparse, random
from typing import Sequence, Tuple, Optional

import numpy as np
import zarr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------- AMP shim ----------
try:
    from torch import amp as _amp
    def _autocast(enabled: bool): return _amp.autocast("cuda", enabled=enabled)
    def _GradScaler(enabled: bool): return _amp.GradScaler("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast_wrap, GradScaler as _CudaGS
    def _autocast(enabled: bool): return _autocast_wrap(enabled=enabled)
    def _GradScaler(enabled: bool): return _CudaGS(enabled=enabled)

# ---------- Distributed ----------
def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main():
    return (not ddp_is_initialized()) or dist.get_rank() == 0

def all_reduce_(t):
    if ddp_is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- Dataset (same as train_ndfull) ----------
def _open_group(path: str):
    try:
        return zarr.open(path, mode='r')
    except Exception:
        return zarr.open_group(zarr.DirectoryStore(path), mode='r')

class MultiZarrDataset(Dataset):
    def __init__(self, zarr_paths: Sequence[str], target_scale=1e-3):
        self.groups = [_open_group(p) for p in zarr_paths]
        self.vox = [g["voxels"] for g in self.groups]
        self.tgt = [g["targets"] for g in self.groups]
        self.tpcds = []
        for g in self.groups:
            if "charge_tpc_ids" in g: self.tpcds.append(g["charge_tpc_ids"])
            elif "tpc_ids" in g: self.tpcds.append(g["tpc_ids"])
            else: self.tpcds.append(None)
        self.sizes = [v.shape[0] for v in self.vox]
        self.cum = np.cumsum([0] + self.sizes).tolist()
        self.N = self.cum[-1]
        self.C = int(self.vox[0].shape[1])
        self.K = int(self.tgt[0].shape[1])
        self.target_scale = float(target_scale)

    def __len__(self): return self.N

    def _map(self, idx):
        if idx < 0: idx += self.N
        for i, s in enumerate(self.sizes):
            if idx < self.cum[i+1]: return i, idx - self.cum[i]
        raise IndexError(idx)

    def __getitem__(self, idx):
        fi, li = self._map(idx)
        x = np.asarray(self.vox[fi][li], dtype=np.float32)
        y = np.asarray(self.tgt[fi][li], dtype=np.float32)
        tpcid = 0
        if self.tpcds[fi] is not None:
            val = self.tpcds[fi][li]
            tpcid = int(val[1] if isinstance(val, np.ndarray) and val.size > 1 else val)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        # Simple input normalization: scale by voxel max
        vmax = x.max()
        if vmax > 1e-6: x = x / vmax
        y = y * self.target_scale
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(tpcid, dtype=torch.long)

# ---------- Model ----------
class ConvBnAct(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=1):
        super().__init__()
        # Use small group norms with min group size guard
        gn = min(8, c_out)
        while c_out % gn != 0: gn -= 1
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(gn, c_out),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Simple3DCNN(nn.Module):
    """
    Compact 3D CNN designed for sparse datasets (~4000 samples).
    Two downsampling stages + Global Average Pooling + MLP head.
    Total parameters: ~2.1M (vs ~47M for HybridPerceiver3D)
    """
    def __init__(self, c_in=1, num_targets=120, num_tpcs=72,
                 base_ch=32, tpc_embed_dim=16, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnAct(c_in, base_ch, 3, stride=1, padding=1),
            ConvBnAct(base_ch, base_ch, 3, stride=1, padding=1),
        )
        # Two strided downsampling stages
        self.down1 = ConvBnAct(base_ch, base_ch * 2, stride=2)         # /2
        self.down2 = ConvBnAct(base_ch * 2, base_ch * 4, stride=2)     # /4
        # Additional depth to increase receptive field without adding params
        self.extra = ConvBnAct(base_ch * 4, base_ch * 4, 3, padding=1)

        feat_dim = base_ch * 4  # 128

        # TPC embedding (small)
        self.tpc_emb = nn.Embedding(num_tpcs, tpc_embed_dim)

        mlp_in = feat_dim + tpc_embed_dim  # 128 + 16 = 144
        self.head = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_targets),
        )

    def forward(self, x, tpc_ids):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.extra(x)
        # Global Average Pool: (B, C, D, H, W) → (B, C)
        x = x.mean(dim=[2, 3, 4])
        # Force float32 for embedding + MLP head to avoid cuBLAS FP16 errors
        # on environments where 16-bit GEMM is broken (common local torch install issue)
        with _autocast(enabled=False):
            e = self.tpc_emb(tpc_ids)               # (B, tpc_embed_dim)
            x = torch.cat([x.float(), e], dim=-1)   # (B, 144)
            return self.head(x)                     # (B, 120)

# ---------- Targets-only dataset for fast standardizer fitting ----------
class _TargetsOnlyDS(Dataset):
    """Reads only target arrays — no voxels. Used for fast standardizer
    fitting to avoid loading 6MB voxel grids per sample unnecessarily."""
    def __init__(self, full_ds):
        self.tgt   = full_ds.tgt
        self.sizes = full_ds.sizes
        self.cum   = full_ds.cum
        self.N     = full_ds.N
        self.K     = full_ds.K
        self.scale = full_ds.target_scale
    def __len__(self): return self.N
    def _map(self, idx):
        if idx < 0: idx += self.N
        for i in range(len(self.sizes)):
            if idx < self.cum[i+1]: return i, idx - self.cum[i]
        raise IndexError(idx)
    def __getitem__(self, idx):
        fi, li = self._map(idx)
        y = np.asarray(self.tgt[fi][li], dtype=np.float32)
        return torch.from_numpy(np.nan_to_num(y, nan=0.0) * self.scale)

# ---------- Training utilities ----------
def masked_mean(t, mask):
    m = mask.bool()
    if not m.any(): return None
    return t[m].mean()

class TargetStandardizer:
    def __init__(self, mean, std):
        self.mean = mean.view(1, -1)
        self.std = std.view(1, -1).clamp(min=1e-6)
    def encode(self, y): return (y - self.mean) / self.std
    def decode(self, y): return y * self.std + self.mean

@torch.no_grad()
def fit_standardizer(loader, device, sentinel_scaled, zero_below_scaled):
    ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    K = ds.K
    s1 = torch.zeros(K, device=device)
    s2 = torch.zeros(K, device=device)
    n  = torch.zeros(K, device=device)
    # Read ONLY targets (not voxels) — avoids ~6MB I/O per sample on 100k dataset
    # Use a high batch size since target arrays are tiny (120 floats = 480 bytes each)
    tgt_loader = DataLoader(
        _TargetsOnlyDS(ds), batch_size=512, shuffle=False,
        num_workers=min(16, os.cpu_count() or 4), pin_memory=True,
    )
    for y in tgt_loader:
        y = y.to(device)
        alive = y != sentinel_scaled
        y_eff = torch.where(alive & (y < zero_below_scaled), torch.zeros_like(y), y)
        s1 += (y_eff * alive).sum(0)
        s2 += ((y_eff**2) * alive).sum(0)
        n  += alive.sum(0)
    all_reduce_(s1); all_reduce_(s2); all_reduce_(n)
    n = n.clamp(min=1.)
    mean = s1 / n
    std  = (((s2 / n) - mean**2).clamp(min=1e-12)).sqrt()
    return TargetStandardizer(mean.detach(), std.detach())

def cosine_lr(step, total_steps, base_lr, warmup):
    if step < warmup: return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))

def train_epoch(model, loader, opt, scaler, device, base_lr, warmup, total_steps,
                global_step, stdzr, amp, sentinel_scaled, zero_below_scaled):
    model.train()
    total_loss = 0.0; n = 0

    for x, y, tpc in loader:
        lr = cosine_lr(global_step, total_steps, base_lr, warmup)
        for g in opt.param_groups: g["lr"] = lr

        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        valid = y != sentinel_scaled
        y_eff = torch.where(valid & (y < zero_below_scaled), torch.zeros_like(y), y)

        opt.zero_grad(set_to_none=True)
        with _autocast(amp):
            pred = model(x, tpc_ids=tpc)
            if stdzr:
                p_in = stdzr.encode(pred)
                y_in = stdzr.encode(y_eff)
            else:
                p_in, y_in = pred, y_eff
            loss_all = F.smooth_l1_loss(p_in, y_in, reduction="none")
            loss = masked_mean(loss_all, valid)

        if loss is None or not torch.isfinite(loss):
            global_step += 1; continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs; n += bs
        global_step += 1

    lt = torch.tensor([total_loss, float(n)], device=device)
    all_reduce_(lt)
    avg = float(lt[0]) / max(1., float(lt[1]))
    return avg, global_step

@torch.no_grad()
def evaluate(model, loader, device, amp, stdzr, inv_scale, sentinel_scaled, zero_below_scaled):
    model.eval()
    ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    K = ds.K
    total_err2 = torch.zeros(1, device=device)
    total_cnt  = torch.zeros(1, device=device)
    sum_err2   = torch.zeros(K, device=device)
    sum_y      = torch.zeros(K, device=device)
    sum_y2     = torch.zeros(K, device=device)
    cnt        = torch.zeros(K, device=device)

    for x, y, tpc in loader:
        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        valid = y != sentinel_scaled
        y_eff = torch.where(valid & (y < zero_below_scaled), torch.zeros_like(y), y)

        with _autocast(amp):
            pred = model(x, tpc_ids=tpc)

        Praw = pred * inv_scale
        Yraw = y_eff * inv_scale

        if stdzr:
            p_s = stdzr.encode(pred)
            y_s = stdzr.encode(y_eff)
            total_err2 += ((p_s - y_s)**2)[valid].sum()
            total_cnt  += valid.sum()

        sum_err2 += (((Praw - Yraw)**2) * valid).sum(0)
        sum_y    += (Yraw * valid).sum(0)
        sum_y2   += ((Yraw**2) * valid).sum(0)
        cnt      += valid.sum(0)

    for t in (total_err2, total_cnt, sum_err2, sum_y, sum_y2, cnt): all_reduce_(t)

    ok = cnt > 0
    cnt_c = cnt.clamp(min=1.)
    mse_k = sum_err2 / cnt_c
    var_k = (sum_y2 / cnt_c - (sum_y / cnt_c)**2).clamp(min=1e-12)
    r2_k  = 1. - mse_k / var_k
    _avg = lambda v: float(v[ok].mean()) if ok.any() else float("nan")

    return {
        "loss": float(total_err2 / total_cnt.clamp(min=1.)),
        "mse": _avg(mse_k),
        "r2":  _avg(r2_k),
    }

# ---------- Entry ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Compact 3D CNN trainer for NDLAr-full (small dataset)")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="./runs/simple_run")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--target-scale", type=float, default=1e-3)
    ap.add_argument("--dead-sentinel", type=float, default=-10000.0)
    ap.add_argument("--zero-below-raw", type=float, default=100.0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--base-ch", type=int, default=32, help="Base channel width (fewer = less overfitting)")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)

    if is_main():
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[simpleMLTrain] base_ch={args.base_ch} dropout={args.dropout} "
              f"lr={args.lr} wd={args.weight_decay}")

    zpaths = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir) if d.endswith(".zarr")
    ])
    ds = MultiZarrDataset(zpaths, target_scale=args.target_scale)
    n_val = max(1, int(args.val_frac * len(ds)))
    n_tr  = len(ds) - n_val
    ds_tr, ds_val = random_split(ds, [n_tr, n_val],
                                 generator=torch.Generator().manual_seed(args.seed))

    tr_sampler  = DistributedSampler(ds_tr, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(ds_val, shuffle=False)
    tr_loader   = DataLoader(ds_tr, batch_size=args.batch_size, sampler=tr_sampler,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader  = DataLoader(ds_val, batch_size=args.batch_size, sampler=val_sampler,
                             num_workers=args.num_workers)

    model = Simple3DCNN(
        c_in=ds.C, num_targets=ds.K,
        base_ch=args.base_ch, dropout=args.dropout
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    if is_main():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = _GradScaler(enabled=args.amp)

    sentinel_scaled   = args.dead_sentinel * args.target_scale
    zero_below_scaled = args.zero_below_raw * args.target_scale
    inv_scale         = 1.0 / args.target_scale

    if is_main(): print("Fitting standardizers...")
    stdzr = fit_standardizer(tr_loader, device, sentinel_scaled, zero_below_scaled)

    total_steps = args.epochs * len(tr_loader)
    start_epoch = 0; global_step = 0; best_val = float("inf")

    if args.resume:
        ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location="cpu")
            model.module.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch  = ck["epoch"] + 1
            global_step  = ck["global_step"]
            best_val     = ck["best_val"]
            if is_main(): print(f"  Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        tr_sampler.set_epoch(epoch)
        tr_loss, global_step = train_epoch(
            model, tr_loader, opt, scaler, device,
            args.lr, args.warmup_steps, total_steps,
            global_step, stdzr, args.amp,
            sentinel_scaled, zero_below_scaled,
        )
        vm = evaluate(model, val_loader, device, args.amp, stdzr,
                      inv_scale, sentinel_scaled, zero_below_scaled)

        if is_main():
            print(f"Epoch {epoch+1:03d} | Train Loss: {tr_loss:.4f} | "
                  f"Val MSE: {vm['mse']:.2f} | Val R2: {vm['r2']:.4f}")

            torch.save({
                "epoch": epoch, "model": model.module.state_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "global_step": global_step, "best_val": best_val,
            }, os.path.join(args.out_dir, "checkpoint.pt"))

            if vm["mse"] < best_val:
                best_val = vm["mse"]
                torch.save(model.module.state_dict(),
                           os.path.join(args.out_dir, "best_model.pt"))
                print("  ✓ Saved new best model")

    if ddp_is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
