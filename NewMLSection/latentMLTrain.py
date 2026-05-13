#!/usr/bin/env python3
"""
latentMLTrain.py

Latent-space 3D CNN with a channel-wise light decoder for NDLAr-full.

Key Idea
--------
The 120 light channels are not independent: they are samples of the same
physical light field. A plain regression head treats them as 120 separate
scalars. Instead, we force the model to:
  1. Compress the charge information into a small latent vector z (dim << 120)
  2. Decode z into amplitudes using a per-channel MLP, where each channel has
     a learnable "position embedding" that encodes its detector location.

The bottleneck (latent_dim << 120) acts as a regulariser that prevents the
model from memorising independent channel noise, and forces it to capture the
structured, low-rank nature of the light field.

Light Channel Positional Encoding
----------------------------------
The 120 channels are on two sides of the detector and are not pixelated.
Rather than hard-coding sinusoidal coordinates (which requires knowing exact
geometry), we use *learnable* per-channel embeddings. The model discovers the
geometric correlations from data. You can optionally seed these embeddings with
known (x,y,z) detector coordinates via --channel-coords-file if you have them.

Architecture
------------
    Input: (B, C, 50, 300, 100) voxel grid
    → 3D CNN stem + 2× DownBlock + Global Average Pool  → (B, feat_dim)
    → concat TPC embedding                               → (B, feat_dim + tpc_emb)
    → Linear projection                                  → z  (B, latent_dim)
    → LightChannelDecoder (broadcast z over 120 channels)
        For each channel i:
            [z ; channel_emb_i] → Linear → ReLU → Linear → a_i
    → (B, 120) amplitudes

The decoder uses a SHARED MLP (same weights for all channels) — each channel
is distinguished only by its embedding. This forces the embeddings to carry
all channel-specific information and prevents the head from overfitting.

Training
--------
  Loss = SmoothL1(pred, target) masked to alive/active channels
  Same sentinel / zero-below masking as other scripts.

Usage
-----
  torchrun --nproc_per_node=4 latentMLTrain.py \\
      --input-dir ./zarr_outputs_enhanced \\
      --out-dir ./runs/latent_run \\
      --batch-size 4 --epochs 200 \\
      --latent-dim 32 --no-amp
"""
from __future__ import annotations
import os, math, argparse, random
from typing import Sequence, Optional

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

# ---------- Dataset ----------
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
    """Fast standardizer: reads only target arrays, skips 6MB voxel grids."""
    def __init__(self, ds):
        self.tgt   = ds.tgt;   self.sizes = ds.sizes
        self.cum   = ds.cum;   self.N     = ds.N
        self.K     = ds.K;     self.scale = ds.target_scale
    def __len__(self): return self.N
    def _map(self, idx):
        if idx < 0: idx += self.N
        for i in range(len(self.sizes)):
            if idx < self.cum[i+1]: return i, idx - self.cum[i]
        raise IndexError(idx)
    def __getitem__(self, idx):
        fi, li = self._map(idx)
        y = np.asarray(self.tgt[fi][li], dtype=np.float32)
        return torch.from_numpy(np.nan_to_num(y) * self.scale)


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


class LightChannelDecoder(nn.Module):
    """
    Decodes a latent vector z into per-channel amplitudes using learnable
    channel position embeddings.

    Each channel i has a unique embedding e_i (its "address" in the light
    detector space). A shared MLP maps [z ; e_i] → amplitude_i.

    Using a SHARED MLP (not 120 separate MLPs) means:
      - Far fewer parameters (avoids overfitting)
      - The model must encode all channel-specific info in e_i
      - The decoder learns a universal translation rule: "given the global
        light field state z and this channel's geometry e_i, what amplitude?"
    """
    def __init__(self, num_channels: int, latent_dim: int,
                 channel_emb_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.num_channels   = num_channels
        self.channel_emb    = nn.Embedding(num_channels, channel_emb_dim)

        # Shared MLP: [z; e_i] → amplitude
        mlp_in = latent_dim + channel_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Pre-register channel indices buffer for efficiency
        self.register_buffer("ch_idx", torch.arange(num_channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) global latent from the CNN encoder
        Returns:
            (B, num_channels) predicted amplitudes
        """
        B = z.size(0)
        K = self.num_channels

        # Get all channel embeddings: (K, channel_emb_dim)
        e = self.channel_emb(self.ch_idx)          # (K, emb_dim)

        # Broadcast z over K channels: (B, K, latent_dim)
        z_exp = z.unsqueeze(1).expand(B, K, -1)    # (B, K, latent_dim)
        e_exp = e.unsqueeze(0).expand(B, K, -1)    # (B, K, emb_dim)

        # Concatenate and decode each channel: (B, K, latent+emb)
        inp = torch.cat([z_exp, e_exp], dim=-1)    # (B, K, latent+emb)

        # Apply shared MLP to each (sample, channel) pair
        # Flatten B×K, apply, reshape
        out = self.mlp(inp.view(B * K, -1))        # (B*K, 1)
        return out.view(B, K)                       # (B, K)


class LatentCNN(nn.Module):
    """
    Charge encoder (3D CNN) + light channel decoder.

    The CNN compresses charge voxels into a low-dimensional latent z.
    The decoder reconstructs 120 light amplitudes from z using per-channel
    positional embeddings. The latent_dim << 120 bottleneck forces the model
    to discover the low-rank structure of the light field.
    """
    def __init__(self, c_in=1, num_targets=120, num_tpcs=72,
                 base_ch=32, tpc_embed_dim=16, dropout=0.3,
                 latent_dim=32, channel_emb_dim=32, decoder_hidden=128):
        super().__init__()
        # --- Charge encoder (3D CNN backbone) ---
        self.stem  = nn.Sequential(ConvBnAct(c_in, base_ch),
                                   ConvBnAct(base_ch, base_ch))
        self.down1 = ConvBnAct(base_ch,     base_ch * 2, stride=2)
        self.down2 = ConvBnAct(base_ch * 2, base_ch * 4, stride=2)
        self.extra = ConvBnAct(base_ch * 4, base_ch * 4)

        feat_dim = base_ch * 4   # 128

        # TPC embedding (which detector module)
        self.tpc_emb = nn.Embedding(num_tpcs, tpc_embed_dim)

        # Project CNN features → latent z
        proj_in = feat_dim + tpc_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(proj_in, proj_in),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_in, latent_dim),
        )

        # --- Light channel decoder ---
        self.decoder = LightChannelDecoder(
            num_channels  = num_targets,
            latent_dim    = latent_dim,
            channel_emb_dim = channel_emb_dim,
            hidden_dim    = decoder_hidden,
            dropout       = dropout,
        )

        self.latent_dim = latent_dim

    def encode(self, x, tpc_ids):
        """Returns latent z: (B, latent_dim)."""
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.extra(x)
        x = x.mean(dim=[2, 3, 4])              # Global Average Pool
        # Force float32 for all linear layers (cuBLAS guard)
        with _autocast(enabled=False):
            e = self.tpc_emb(tpc_ids)
            h = torch.cat([x.float(), e], dim=-1)
            z = self.proj(h)
        return z

    def forward(self, x, tpc_ids):
        """Returns predicted amplitudes: (B, num_targets)."""
        z = self.encode(x, tpc_ids)
        with _autocast(enabled=False):
            pred = self.decoder(z)
        return pred

    def get_latent(self, x, tpc_ids):
        """Convenience method for analysis: returns z without decoding."""
        with torch.no_grad():
            return self.encode(x, tpc_ids)

    def load_channel_coords(self, coords: torch.Tensor):
        """
        Optionally seed channel embeddings with known detector coordinates.

        Args:
            coords: (num_channels, D) tensor of physical coordinates
                    (e.g., x/y/z positions, or side indicator + local coords)
                    Will be projected to channel_emb_dim via a linear layer.
        """
        emb_dim = self.decoder.channel_emb.embedding_dim
        K, D = coords.shape
        proj = nn.Linear(D, emb_dim, bias=False)
        with torch.no_grad():
            init = proj(coords.float())
            self.decoder.channel_emb.weight.copy_(init)
        print(f"  [LatentCNN] Seeded {K} channel embeddings from coords shape {coords.shape}")


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


def train_sub_epoch(
    model, loader, loader_iter, opt, scaler, device, base_lr, warmup,
    total_steps, global_step, stdzr, amp,
    sentinel_scaled, zero_below_scaled, sub_epoch_size, inv_scale,
):
    """Run at most sub_epoch_size batches; return (avg_loss, global_step, loader_iter, tr_r2)."""
    model.train()
    tot_loss = 0.0; n = 0
    K = None
    sum_err2 = sum_y = sum_y2 = cnt = None

    for _ in range(sub_epoch_size):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x, y, tpc = batch
        lr = cosine_lr(global_step, total_steps, base_lr, warmup)
        for g in opt.param_groups: g["lr"] = lr

        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        alive = (y != sentinel_scaled)
        y_eff = torch.where(alive & (y < zero_below_scaled), torch.zeros_like(y), y)

        if K is None:
            K = y.shape[1]
            sum_err2 = torch.zeros(K, device=device)
            sum_y    = torch.zeros(K, device=device)
            sum_y2   = torch.zeros(K, device=device)
            cnt      = torch.zeros(K, device=device)

        opt.zero_grad(set_to_none=True)
        with _autocast(amp):
            pred = model(x, tpc_ids=tpc)
            if stdzr:
                p_in = stdzr.encode(pred)
                y_in = stdzr.encode(y_eff)
            else:
                p_in, y_in = pred, y_eff
            loss_all = F.smooth_l1_loss(p_in, y_in, reduction="none")
            loss_all = loss_all * alive.float()
            loss = loss_all.sum() / alive.float().sum().clamp(min=1.)

        if not torch.isfinite(loss):
            global_step += 1; continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        bs = x.size(0)
        tot_loss += float(loss) * bs; n += bs
        global_step += 1

        # Accumulate train R² stats (in raw scale)
        with torch.no_grad():
            Praw = pred.detach() * inv_scale
            Yraw = y_eff         * inv_scale
            sum_err2 += (((Praw - Yraw)**2) * alive).sum(0)
            sum_y    += (Yraw * alive).sum(0)
            sum_y2   += ((Yraw**2) * alive).sum(0)
            cnt      += alive.sum(0)

    # Reduce loss across DDP workers
    lt = torch.tensor([tot_loss, float(n)], device=device)
    all_reduce_(lt)
    avg_loss = float(lt[0]) / max(1., float(lt[1]))

    # Compute train R²
    tr_r2 = float("nan")
    if K is not None:
        for t in (sum_err2, sum_y, sum_y2, cnt): all_reduce_(t)
        ok = cnt > 0; cnt_c = cnt.clamp(min=1.)
        mse_k = sum_err2 / cnt_c
        var_k = (sum_y2 / cnt_c - (sum_y / cnt_c)**2).clamp(min=1e-12)
        r2_k  = 1. - mse_k / var_k
        tr_r2 = float(r2_k[ok].mean()) if ok.any() else float("nan")

    return avg_loss, global_step, loader_iter, tr_r2


@torch.no_grad()
def evaluate(model, loader, device, amp, stdzr, inv_scale,
             sentinel_scaled, zero_below_scaled):
    model.eval()
    ds = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    K  = ds.K
    sum_err2 = torch.zeros(K, device=device)
    sum_y    = torch.zeros(K, device=device)
    sum_y2   = torch.zeros(K, device=device)
    cnt      = torch.zeros(K, device=device)
    tot_loss = torch.zeros(1, device=device)
    tot_n    = torch.zeros(1, device=device)

    # Track latent variance (how spread out are the z vectors?) — diagnostic
    sum_z  = torch.zeros(model.module.latent_dim if hasattr(model, "module")
                         else model.latent_dim, device=device)
    sum_z2 = torch.zeros_like(sum_z)
    nz     = torch.zeros(1, device=device)

    for x, y, tpc in loader:
        x, y, tpc = x.to(device), y.to(device), tpc.to(device)
        alive = (y != sentinel_scaled)
        y_eff = torch.where(alive & (y < zero_below_scaled), torch.zeros_like(y), y)

        with _autocast(amp):
            z    = (model.module if hasattr(model, "module") else model).encode(x, tpc)
            pred = (model.module if hasattr(model, "module") else model).decoder(z)

        Praw = pred  * inv_scale
        Yraw = y_eff * inv_scale
        sum_err2 += (((Praw - Yraw)**2) * alive).sum(0)
        sum_y    += (Yraw * alive).sum(0)
        sum_y2   += ((Yraw**2) * alive).sum(0)
        cnt      += alive.sum(0)

        if stdzr:
            p_s = stdzr.encode(pred)
            y_s = stdzr.encode(y_eff)
            tot_loss += ((p_s - y_s)**2)[alive].sum()
            tot_n    += alive.sum()

        sum_z  += z.sum(0)
        sum_z2 += (z**2).sum(0)
        nz     += x.size(0)

    for t in (sum_err2, sum_y, sum_y2, cnt, tot_loss, tot_n, sum_z, sum_z2, nz):
        all_reduce_(t)

    ok = cnt > 0; cnt_c = cnt.clamp(min=1.)
    mse_k = sum_err2 / cnt_c
    var_k = (sum_y2 / cnt_c - (sum_y / cnt_c)**2).clamp(min=1e-12)
    r2_k  = 1. - mse_k / var_k
    _avg  = lambda v: float(v[ok].mean()) if ok.any() else float("nan")

    # Latent space statistics (mean variance across latent dims)
    z_mean = sum_z / nz.clamp(min=1.)
    z_var  = (sum_z2 / nz.clamp(min=1.) - z_mean**2).clamp(min=0.)
    latent_mean_var = float(z_var.mean())

    return {
        "loss":  float(tot_loss / tot_n.clamp(min=1.)),
        "mse":   _avg(mse_k),
        "r2":    _avg(r2_k),
        "latent_var": latent_mean_var,   # should grow as model learns; collapse = bad
    }


# ---------- Entry ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Latent 3D CNN with light channel decoder for NDLAr-full")
    ap.add_argument("--input-dir",    required=True)
    ap.add_argument("--out-dir",      default="./runs/latent_run")
    ap.add_argument("--epochs",       type=int,   default=200)
    ap.add_argument("--batch-size",   type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int,   default=200)
    ap.add_argument("--num-workers",  type=int,   default=16)
    ap.add_argument("--val-frac",     type=float, default=0.15)
    ap.add_argument("--target-scale", type=float, default=1e-3)
    ap.add_argument("--dead-sentinel",  type=float, default=-10000.0)
    ap.add_argument("--zero-below-raw", type=float, default=100.0)
    ap.add_argument("--dropout",      type=float, default=0.3)
    ap.add_argument("--base-ch",      type=int,   default=32)
    # Latent space
    ap.add_argument("--latent-dim",   type=int,   default=32,
                    help="Dimension of the bottleneck latent vector z (try 16-64)")
    ap.add_argument("--channel-emb-dim", type=int, default=32,
                    help="Dimension of each light channel's positional embedding")
    ap.add_argument("--decoder-hidden",  type=int, default=128,
                    help="Hidden dim of the shared channel decoder MLP")
    ap.add_argument("--channel-coords-file", type=str, default=None,
                    help="Optional .npy file with shape (K, D) of channel coordinates "
                         "to seed the channel embeddings (e.g. x/y/z positions)")
    ap.add_argument("--amp",    action="store_true",  default=False)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed",   type=int, default=42)
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
        print(f"[latentMLTrain] latent_dim={args.latent_dim} "
              f"channel_emb_dim={args.channel_emb_dim} "
              f"decoder_hidden={args.decoder_hidden} "
              f"base_ch={args.base_ch} dropout={args.dropout}")

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

    model = LatentCNN(
        c_in=ds.C, num_targets=ds.K, num_tpcs=72,
        base_ch=args.base_ch, tpc_embed_dim=16, dropout=args.dropout,
        latent_dim=args.latent_dim,
        channel_emb_dim=args.channel_emb_dim,
        decoder_hidden=args.decoder_hidden,
    ).to(device)

    # Optionally seed channel embeddings from known detector coordinates
    if args.channel_coords_file and os.path.exists(args.channel_coords_file):
        coords = torch.from_numpy(np.load(args.channel_coords_file))
        model.load_channel_coords(coords)

    model = DDP(model, device_ids=[local_rank])

    if is_main():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,}")
        print(f"  Compression ratio: {ds.K} channels → {args.latent_dim}-dim latent "
              f"({args.latent_dim/ds.K*100:.1f}% of original)")

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    scaler = _GradScaler(enabled=args.amp)

    sentinel_scaled   = args.dead_sentinel   * args.target_scale
    zero_below_scaled = args.zero_below_raw  * args.target_scale
    inv_scale         = 1.0 / args.target_scale

    if is_main(): print("Fitting standardizers on targets...")
    stdzr = fit_standardizer(tr_loader, device, sentinel_scaled, zero_below_scaled)

    N_SUBEPOCHS = 5   # save/eval N times per full pass over the data
    total_steps = args.epochs * N_SUBEPOCHS * (len(tr_loader) // N_SUBEPOCHS + 1)
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
            if is_main(): print(f"  Resumed from sub-epoch {start_epoch}")
        else:
            if is_main(): print("  No checkpoint found. Starting fresh.")

    sub_epoch_size = max(1, len(tr_loader) // N_SUBEPOCHS)
    total_sub_epochs = args.epochs * N_SUBEPOCHS
    loader_iter = iter(tr_loader)

    for sub_epoch in range(start_epoch, total_sub_epochs):
        # Each sub-epoch covers 1/N_SUBEPOCHS of the full data.
        # Reshuffle sampler at the start of every real epoch.
        if sub_epoch % N_SUBEPOCHS == 0:
            tr_sampler.set_epoch(sub_epoch // N_SUBEPOCHS)
            loader_iter = iter(tr_loader)

        tr_loss, global_step, loader_iter, tr_r2 = train_sub_epoch(
            model, tr_loader, loader_iter, opt, scaler, device,
            args.lr, args.warmup_steps, total_steps, global_step,
            stdzr, args.amp, sentinel_scaled, zero_below_scaled,
            sub_epoch_size, inv_scale,
        )
        vm = evaluate(model, val_loader, device, args.amp, stdzr,
                      inv_scale, sentinel_scaled, zero_below_scaled)

        if is_main():
            real_epoch  = sub_epoch // N_SUBEPOCHS + 1
            sub_in_epoch = sub_epoch % N_SUBEPOCHS + 1
            print(f"Epoch {real_epoch:03d}.{sub_in_epoch}/5 | "
                  f"Train Loss: {tr_loss:.4f} | Train R2: {tr_r2:.4f} | "
                  f"Val R2: {vm['r2']:.4f} | Val MSE: {vm['mse']:.2f} | "
                  f"Latent Var: {vm['latent_var']:.4f}")

            torch.save({
                "epoch": sub_epoch, "model": model.module.state_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "global_step": global_step, "best_val": best_val,
                "latent_dim": args.latent_dim,
                "channel_emb_dim": args.channel_emb_dim,
            }, os.path.join(args.out_dir, "checkpoint.pt"))

            if vm["loss"] < best_val:
                best_val = vm["loss"]
                torch.save(model.module.state_dict(),
                           os.path.join(args.out_dir, "best_model.pt"))
                print("  ✓ Saved new best model")

    if ddp_ok(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
