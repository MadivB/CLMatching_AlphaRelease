#!/usr/bin/env python3
"""
gatedMLTrain.py

Two-head Gated CNN for sparse channel prediction on NDLAr-full.

Motivation: low-energy events have most channels ~0, so a plain regression
model wastes gradient budget chasing baseline noise. This model explicitly
predicts WHICH channels are "lit" (gate head) before predicting their
amplitudes (regress head). The regression loss is masked to active channels,
focusing learning on meaningful signal.

Architecture:
  Input: (B, C, 50, 300, 100)
  → 3D CNN backbone (same as simpleMLTrain)
  → Global Average Pool → (B, feat+tpc_emb)
  → gate_head   : Linear → 120-dim sigmoid  (active channel probabilities)
  → regress_head: Linear → 120-dim          (amplitude predictions)

Loss:
  L_gate    = BCE(gate_pred, active_mask)        ← focuses on lit/unlit
  L_regress = SmoothL1(pred, target, mask=active) ← only for active channels
  L_total   = L_gate + lambda_reg * L_regress

At inference:
  final = regress_pred * (gate_pred > gate_thresh)
  dead channels get zero automatically
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
    from torch.cuda.amp import autocast as _ac, GradScaler as _GS
    def _autocast(enabled: bool): return _ac(enabled=enabled)
    def _GradScaler(enabled: bool): return _GS(enabled=enabled)

# ---------- Distributed ----------
def ddp_ok(): return dist.is_available() and dist.is_initialized()
def is_main(): return (not ddp_ok()) or dist.get_rank() == 0
def all_reduce_(t):
    if ddp_ok(): dist.all_reduce(t, op=dist.ReduceOp.SUM)
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------- Dataset (same Zarr loading as simpleMLTrain) ----------
def _open_group(path):
    try:    return zarr.open(path, mode='r')
    except: return zarr.open_group(zarr.DirectoryStore(path), mode='r')

class MultiZarrDataset(Dataset):
    def __init__(self, zarr_paths: Sequence[str], target_scale=1e-3):
        self.groups = [_open_group(p) for p in zarr_paths]
        self.vox    = [g["voxels"]  for g in self.groups]
        self.tgt    = [g["targets"] for g in self.groups]
        self.tpcds  = []
        for g in self.groups:
            if "charge_tpc_ids" in g: self.tpcds.append(g["charge_tpc_ids"])
            elif "tpc_ids"       in g: self.tpcds.append(g["tpc_ids"])
            else:                      self.tpcds.append(None)
        self.sizes = [v.shape[0] for v in self.vox]
        self.cum   = np.cumsum([0] + self.sizes).tolist()
        self.N     = self.cum[-1]
        self.C     = int(self.vox[0].shape[1])
        self.K     = int(self.tgt[0].shape[1])
        self.target_scale = float(target_scale)

    def __len__(self): return self.N

    def _map(self, idx):
        if idx < 0: idx += self.N
        for i in range(len(self.sizes)):
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
        vmax = x.max()
        if vmax > 1e-6: x = x / vmax
        return (torch.from_numpy(x),
                torch.from_numpy(y * self.target_scale),
                torch.tensor(tpcid, dtype=torch.long))


class _TargetsOnlyDS(Dataset):
    """Fast standardizer fitting: reads only targets, skips 6MB voxel grids."""
    def __init__(self, full_ds):
        self.tgt   = full_ds.tgt; self.sizes = full_ds.sizes
        self.cum   = full_ds.cum; self.N     = full_ds.N
        self.K     = full_ds.K;  self.scale = full_ds.target_scale
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

# ---------- Model ----------
class ConvBnAct(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=1):
        super().__init__()
        gn = min(8, c_out)
        while c_out % gn != 0: gn -= 1
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(gn, c_out), nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class GatedCNN(nn.Module):
    """
    Two-head 3D CNN: gate_head predicts which channels are active,
    regress_head predicts their amplitudes.

    At inference:
        gate_probs  = sigmoid(gate_head(features))     # (B, K)
        amplitudes  = regress_head(features)           # (B, K)
        final_pred  = amplitudes * (gate_probs > gate_thresh)
    """
    def __init__(self, c_in=1, num_targets=120, num_tpcs=72,
                 base_ch=32, tpc_embed_dim=16, dropout=0.3):
        super().__init__()
        # Backbone (same as Simple3DCNN)
        self.stem  = nn.Sequential(ConvBnAct(c_in, base_ch),
                                   ConvBnAct(base_ch, base_ch))
        self.down1 = ConvBnAct(base_ch,     base_ch * 2, stride=2)
        self.down2 = ConvBnAct(base_ch * 2, base_ch * 4, stride=2)
        self.extra = ConvBnAct(base_ch * 4, base_ch * 4)
        feat_dim   = base_ch * 4        # 128

        self.tpc_emb = nn.Embedding(num_tpcs, tpc_embed_dim)
        mlp_in = feat_dim + tpc_embed_dim   # 144

        # Gate head: predicts P(channel is active) per channel
        self.gate_head = nn.Sequential(
            nn.Linear(mlp_in, 256), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_targets),  # logits; sigmoid applied in loss/inference
        )

        # Regression head: predicts amplitude per channel
        self.regress_head = nn.Sequential(
            nn.Linear(mlp_in, 256), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_targets),
        )

    def _encode(self, x, tpc_ids):
        """Shared backbone: returns feature vector."""
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.extra(x)
        x = x.mean(dim=[2, 3, 4])          # Global Average Pool
        # Force float32 for Linear layers (cuBLAS guard)
        with _autocast(enabled=False):
            e = self.tpc_emb(tpc_ids)
            return torch.cat([x.float(), e], dim=-1)  # (B, 144)

    def forward(self, x, tpc_ids):
        """Returns (gate_logits, regress_pred) both shape (B, K)."""
        feat = self._encode(x, tpc_ids)
        with _autocast(enabled=False):
            gate_logits = self.gate_head(feat)
            regress     = self.regress_head(feat)
        return gate_logits, regress

    @torch.no_grad()
    def predict(self, x, tpc_ids, gate_thresh=0.5):
        """Inference: returns final amplitudes with dead channels zeroed."""
        gate_logits, regress = self.forward(x, tpc_ids)
        gate_probs = torch.sigmoid(gate_logits)
        active = (gate_probs > gate_thresh)
        return regress * active.float(), gate_probs

# ---------- Training utilities ----------
class TargetStandardizer:
    def __init__(self, mean, std):
        self.mean = mean.view(1, -1)
        self.std  = std.view(1, -1).clamp(min=1e-6)
    def encode(self, y): return (y - self.mean) / self.std
    def decode(self, y): return y * self.std  + self.mean


@torch.no_grad()
def fit_standardizer(loader, device, sentinel_scaled, zero_below_scaled):
    ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    K  = ds.K
    tgt_loader = DataLoader(_TargetsOnlyDS(ds), batch_size=512, shuffle=False,
                            num_workers=min(16, os.cpu_count() or 4), pin_memory=True)
    s1 = torch.zeros(K, device=device)
    s2 = torch.zeros(K, device=device)
    n  = torch.zeros(K, device=device)
    for y in tgt_loader:
        y = y.to(device)
        alive = (y != sentinel_scaled)
        y_eff = torch.where(alive & (y < zero_below_scaled), torch.zeros_like(y), y)
        s1 += (y_eff * alive).sum(0)
        s2 += ((y_eff**2) * alive).sum(0)
        n  += alive.sum(0)
    all_reduce_(s1); all_reduce_(s2); all_reduce_(n)
    n    = n.clamp(min=1.)
    mean = s1 / n
    std  = (((s2 / n) - mean**2).clamp(min=1e-12)).sqrt()
    return TargetStandardizer(mean.detach(), std.detach())


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup: return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))


def gated_loss(gate_logits, regress_pred, y_eff, alive, zero_below_scaled,
               stdzr, lambda_reg, gate_pos_weight):
    """
    Compute the two-component gated loss.

    active_mask: channels that have real signal above zero_below threshold
    sentinel_mask (alive): channels that are real (not dead/sentinel)

    L_gate    = BCE(gate_logits, active_mask)   over alive channels only
    L_regress = SmoothL1(regress, target)       over active channels only
    """
    # Define "lit" = alive + above zero_below
    active = alive & (y_eff >= zero_below_scaled)  # (B, K), bool

    # --- Gate loss (weighted BCE, more weight for rare positives) ---
    if alive.any():
        gt = active.float()                         # 1 = lit, 0 = dark
        weight = torch.ones_like(gate_logits)
        weight[active] = gate_pos_weight            # upweight lit channels
        bce = F.binary_cross_entropy_with_logits(
            gate_logits[alive], gt[alive],
            weight=weight[alive], reduction='mean')
    else:
        bce = gate_logits.sum() * 0.0

    # --- Regression loss (active channels only, in standardized space) ---
    if active.any():
        p = regress_pred[active]
        t = y_eff[active]
        if stdzr is not None:
            p_flat = torch.zeros_like(regress_pred)
            t_flat = torch.zeros_like(y_eff)
            p_flat[active] = p; t_flat[active] = t
            p = stdzr.encode(p_flat)[active]
            t = stdzr.encode(t_flat)[active]
        reg = F.smooth_l1_loss(p, t, reduction='mean')
    else:
        reg = regress_pred.sum() * 0.0

    return bce, reg, bce + lambda_reg * reg


def train_epoch(model, loader, opt, scaler, device, base_lr, warmup,
                total_steps, global_step, stdzr, amp,
                sentinel_scaled, zero_below_scaled, lambda_reg, gate_pos_weight):
    model.train()
    tot_loss = tot_bce = tot_reg = 0.0; n = 0

    for x, y, tpc in loader:
        lr = cosine_lr(global_step, total_steps, base_lr, warmup)
        for g in opt.param_groups: g["lr"] = lr

        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        alive = (y != sentinel_scaled)
        y_eff = torch.where(alive & (y < zero_below_scaled), torch.zeros_like(y), y)

        opt.zero_grad(set_to_none=True)
        with _autocast(amp):
            gate_logits, regress_pred = model(x, tpc_ids=tpc)
            bce, reg, loss = gated_loss(
                gate_logits, regress_pred, y_eff, alive,
                zero_below_scaled, stdzr, lambda_reg, gate_pos_weight)

        if not torch.isfinite(loss):
            global_step += 1; continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        bs = x.size(0)
        tot_loss += float(loss) * bs
        tot_bce  += float(bce)  * bs
        tot_reg  += float(reg)  * bs
        n += bs; global_step += 1

    lt = torch.tensor([tot_loss, tot_bce, tot_reg, float(n)], device=device)
    all_reduce_(lt)
    N = max(1., float(lt[3]))
    return float(lt[0])/N, float(lt[1])/N, float(lt[2])/N, global_step


@torch.no_grad()
def evaluate(model, loader, device, amp, stdzr, inv_scale,
             sentinel_scaled, zero_below_scaled, gate_thresh):
    model.eval()
    ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    K  = ds.K
    tot_r2_num = torch.zeros(1, device=device)
    tot_r2_den = torch.zeros(1, device=device)
    sum_tp = torch.zeros(1, device=device)
    sum_fp = torch.zeros(1, device=device)
    sum_fn = torch.zeros(1, device=device)
    sum_err2 = torch.zeros(K, device=device)
    sum_y    = torch.zeros(K, device=device)
    sum_y2   = torch.zeros(K, device=device)
    cnt      = torch.zeros(K, device=device)

    for x, y, tpc in loader:
        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        alive  = (y != sentinel_scaled)
        y_eff  = torch.where(alive & (y < zero_below_scaled), torch.zeros_like(y), y)
        active = alive & (y_eff >= zero_below_scaled)

        with _autocast(amp):
            gate_logits, regress_pred = model(x, tpc_ids=tpc)

        gate_probs  = torch.sigmoid(gate_logits)
        gate_binary = gate_probs > gate_thresh
        final_pred  = regress_pred * gate_binary.float()

        # Gate quality metrics (TP/FP/FN)
        sum_tp += (gate_binary & active).sum().float()
        sum_fp += (gate_binary & ~active & alive).sum().float()
        sum_fn += (~gate_binary & active).sum().float()

        # Regression quality on truly active channels
        Praw = final_pred * inv_scale
        Yraw = y_eff      * inv_scale
        sum_err2 += (((Praw - Yraw)**2) * active).sum(0)
        sum_y    += (Yraw * active).sum(0)
        sum_y2   += ((Yraw**2) * active).sum(0)
        cnt      += active.sum(0)

        # MSE in standardized space (active only)
        if stdzr is not None and active.any():
            p_s = stdzr.encode(regress_pred)
            y_s = stdzr.encode(y_eff)
            tot_r2_num += ((p_s - y_s)**2)[active].sum()
            tot_r2_den += active.sum()

    for t in (sum_tp, sum_fp, sum_fn, sum_err2, sum_y, sum_y2,
              cnt, tot_r2_num, tot_r2_den): all_reduce_(t)

    ok = cnt > 0; cnt_c = cnt.clamp(min=1.)
    mse_k  = sum_err2 / cnt_c
    var_k  = (sum_y2 / cnt_c - (sum_y / cnt_c)**2).clamp(min=1e-12)
    r2_k   = 1. - mse_k / var_k
    _avg   = lambda v: float(v[ok].mean()) if ok.any() else float("nan")

    prec = float(sum_tp / (sum_tp + sum_fp).clamp(min=1.))
    rec  = float(sum_tp / (sum_tp + sum_fn).clamp(min=1.))
    f1   = 2*prec*rec / max(prec+rec, 1e-9)

    return {
        "loss": float(tot_r2_num / tot_r2_den.clamp(min=1.)),
        "mse":  _avg(mse_k),
        "r2":   _avg(r2_k),
        "gate_precision": prec,
        "gate_recall":    rec,
        "gate_f1":        f1,
    }

# ---------- Entry ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Gated two-head CNN: detect active channels, then regress amplitudes")
    ap.add_argument("--input-dir",   required=True)
    ap.add_argument("--out-dir",     default="./runs/gated_run")
    ap.add_argument("--epochs",      type=int,   default=200)
    ap.add_argument("--batch-size",  type=int,   default=16)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--weight-decay",type=float, default=1e-3)
    ap.add_argument("--warmup-steps",type=int,   default=200)
    ap.add_argument("--num-workers", type=int,   default=16)
    ap.add_argument("--val-frac",    type=float, default=0.15)
    ap.add_argument("--target-scale",type=float, default=1e-3)
    ap.add_argument("--dead-sentinel",type=float, default=-10000.0)
    ap.add_argument("--zero-below-raw",type=float, default=100.0,
                    help="Threshold (raw units) below which a channel counts as 'dark'")
    ap.add_argument("--dropout",     type=float, default=0.3)
    ap.add_argument("--base-ch",     type=int,   default=32)
    ap.add_argument("--lambda-reg",  type=float, default=1.0,
                    help="Weight for regression loss relative to gate BCE loss")
    ap.add_argument("--gate-pos-weight", type=float, default=5.0,
                    help="Upweight for lit (positive) channels in gate BCE (rare class boost)")
    ap.add_argument("--gate-thresh", type=float, default=0.5,
                    help="Sigmoid threshold to decide channel is active at inference")
    ap.add_argument("--amp",   action="store_true",  default=False)
    ap.add_argument("--no-amp",action="store_false", dest="amp")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--seed",  type=int, default=42)
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
        print(f"[gatedMLTrain] base_ch={args.base_ch} dropout={args.dropout} "
              f"lr={args.lr} lambda_reg={args.lambda_reg} "
              f"gate_pos_weight={args.gate_pos_weight}")

    zpaths = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir) if d.endswith(".zarr")
    ])
    ds    = MultiZarrDataset(zpaths, target_scale=args.target_scale)
    n_val = max(1, int(args.val_frac * len(ds)))
    n_tr  = len(ds) - n_val
    ds_tr, ds_val = random_split(ds, [n_tr, n_val],
                                 generator=torch.Generator().manual_seed(args.seed))

    tr_sampler  = DistributedSampler(ds_tr, shuffle=True,  drop_last=True)
    val_sampler = DistributedSampler(ds_val, shuffle=False)
    tr_loader   = DataLoader(ds_tr, batch_size=args.batch_size, sampler=tr_sampler,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader  = DataLoader(ds_val, batch_size=args.batch_size, sampler=val_sampler,
                             num_workers=args.num_workers)

    model = GatedCNN(c_in=ds.C, num_targets=ds.K,
                     base_ch=args.base_ch, dropout=args.dropout).to(device)
    model = DDP(model, device_ids=[local_rank])

    if is_main():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    scaler = _GradScaler(enabled=args.amp)

    sentinel_scaled   = args.dead_sentinel * args.target_scale
    zero_below_scaled = args.zero_below_raw * args.target_scale
    inv_scale         = 1.0 / args.target_scale

    if is_main(): print("Fitting standardizers on targets...")
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
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]
            best_val    = ck["best_val"]
            if is_main(): print(f"  Resumed from epoch {start_epoch}")
        else:
            if is_main(): print("  No checkpoint found. Starting fresh.")

    for epoch in range(start_epoch, args.epochs):
        tr_sampler.set_epoch(epoch)
        tr_loss, tr_bce, tr_reg, global_step = train_epoch(
            model, tr_loader, opt, scaler, device,
            args.lr, args.warmup_steps, total_steps, global_step,
            stdzr, args.amp, sentinel_scaled, zero_below_scaled,
            args.lambda_reg, args.gate_pos_weight,
        )
        vm = evaluate(model, val_loader, device, args.amp, stdzr,
                      inv_scale, sentinel_scaled, zero_below_scaled,
                      args.gate_thresh)

        if is_main():
            print(f"Epoch {epoch+1:03d} | "
                  f"Loss: {tr_loss:.4f} (BCE:{tr_bce:.4f} Reg:{tr_reg:.4f}) | "
                  f"Val R2: {vm['r2']:.4f} | "
                  f"Gate P/R/F1: {vm['gate_precision']:.3f}/{vm['gate_recall']:.3f}/{vm['gate_f1']:.3f}")

            torch.save({
                "epoch": epoch, "model": model.module.state_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "global_step": global_step, "best_val": best_val,
            }, os.path.join(args.out_dir, "checkpoint.pt"))

            if vm["loss"] < best_val:
                best_val = vm["loss"]
                torch.save(model.module.state_dict(),
                           os.path.join(args.out_dir, "best_model.pt"))
                print("  ✓ Saved new best model")

    if ddp_ok(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
