#!/usr/bin/env python3

import argparse
import math
import os
import numpy as np
import zarr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import matplotlib.pyplot as plt

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

NUM_CHANNELS = 120


# ================= Dataset ================= #

class WaveformDataset(Dataset):

    def __init__(self, zarr_path, input_scale=1e-3, target_scale=1e-3):

        root = zarr.open(zarr_path, mode="r")

        self.inputs = root["inputs"]
        self.targets = root["targets"]

        self.input_scale = input_scale
        self.target_scale = target_scale

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):

        src = np.asarray(self.inputs[idx], dtype=np.float32)
        tgt = np.asarray(self.targets[idx], dtype=np.float32)

        src = torch.from_numpy(src) * self.input_scale
        tgt = torch.from_numpy(tgt) * self.target_scale

        return src, tgt


# ================= Feed Forward ================= #

class FeedForwardModule(nn.Module):

    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ================= Convolution Module ================= #

class ConvolutionModule(nn.Module):

    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.glu = nn.GLU(dim=1)

        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
        )

        self.gn = nn.GroupNorm(8, dim)

        self.act = nn.SiLU()

        self.pw2 = nn.Conv1d(dim, dim, 1)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        y = self.ln(x).transpose(1, 2).contiguous()

        y = self.pw1(y)
        y = self.glu(y)

        y = self.dw(y)
        y = self.gn(y)
        y = self.act(y)

        y = self.pw2(y)
        y = self.drop(y)

        return y.transpose(1, 2).contiguous()


# ================= Conformer Block ================= #

class ConformerBlock(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.ff1 = FeedForwardModule(dim, dropout=dropout)

        self.ln_attn = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.conv = ConvolutionModule(dim)

        self.ff2 = FeedForwardModule(dim, dropout=dropout)

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):

        x = x + 0.5 * self.ff1(x)

        ln = self.ln_attn(x)

        # Force attention to FP32 for stability
        with torch.autocast("cuda", enabled=False):

            attn_out, _ = self.attn(
                ln.float(),
                ln.float(),
                ln.float(),
                need_weights=False,
            )

        x = x + attn_out.to(x.dtype)

        x = x + self.conv(x)

        x = x + 0.5 * self.ff2(x)

        return self.ln(x)


# ================= Model ================= #

class ConformerVarPredictor(nn.Module):

    def __init__(
        self,
        num_channels=NUM_CHANNELS,
        waveform_len=1000,
        model_dim=128,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(num_channels, model_dim)

        self.register_buffer(
            "pos_enc",
            self._make_sinusoidal(waveform_len, model_dim),
            persistent=False,
        )

        self.layers = nn.ModuleList(
            [
                ConformerBlock(model_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(model_dim, num_channels)

    def _make_sinusoidal(self, length, dim):

        pe = torch.zeros(1, length, dim)

        pos = torch.arange(length).unsqueeze(1).float()

        div = torch.exp(
            torch.arange(0, dim, 2).float()
            * -(math.log(10000.0) / dim)
        )

        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)

        return pe

    def forward(self, x):

        # (B,C,T) -> (B,T,C)
        x = x.permute(0, 2, 1)

        T = x.size(1)

        h = self.input_proj(x)

        h = h + self.pos_enc[:, :T, :]

        for layer in self.layers:
            h = layer(h)

        out = self.output_proj(h)

        logvar = out.permute(0, 2, 1)

        return torch.clamp(logvar, -12, 12)


# ================= Loss ================= #

def gaussian_nll_loss(src, tgt, logvar, calib_lambda=5.0):

    sigma = torch.exp(0.5 * logvar).clamp(1e-6, 1e3)

    resid = src - tgt

    nll = logvar + (resid ** 2) / (sigma ** 2)

    loss_nll = nll.mean()

    pull = resid / sigma
    pull_std = pull.std().clamp(min=1e-6)

    loss_width = (pull_std - 1.0) ** 2

    loss = loss_nll + calib_lambda * loss_width

    return loss, loss_nll.detach().item(), pull_std.detach().item()


# ================= Validation ================= #

def calculate_true_kurtosis(data):

    if data.size == 0:
        return 0.0

    centered = data - np.mean(data)
    m2 = np.mean(centered ** 2)
    m4 = np.mean(centered ** 4)

    if m2 <= 0:
        return 0.0

    return float(m4 / (m2 ** 2))


def validate_and_demo(
    model,
    dataloader,
    epoch,
    device,
    rank,
    plot_dir,
    plot_enabled=True,
    max_batches=50,
):

    model.eval()

    pulls_low = []
    pulls_med = []
    pulls_high = []
    all_pulls = []

    example_data = None

    with torch.no_grad():

        for i, (src, tgt) in enumerate(dataloader):

            if i >= max_batches:
                break

            src = src.to(device)
            tgt = tgt.to(device)

            logvar = model(src)

            sigma = torch.exp(0.5 * logvar).clamp(1e-6, 1e3)
            resid = src - tgt
            pull = resid / (sigma + 1e-8)

            src_np = src.cpu().numpy()
            tgt_np = tgt.cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            pull_np = pull.cpu().numpy()

            flat_src = src_np.reshape(-1)
            flat_pull = pull_np.reshape(-1)

            all_pulls.append(flat_pull)

            mask_low = flat_src <= 0.6
            if mask_low.any():
                pulls_low.append(flat_pull[mask_low])

            mask_med = (flat_src > 0.6) & (flat_src <= 6.0)
            if mask_med.any():
                pulls_med.append(flat_pull[mask_med])

            mask_high = flat_src > 6.0
            if mask_high.any():
                pulls_high.append(flat_pull[mask_high])

            if plot_enabled and rank == 0 and example_data is None:
                example_data = (src_np[0], tgt_np[0], sigma_np[0])

    width_global = 999.0
    kurt_global = 999.0

    if len(all_pulls) > 0:
        data_global = np.concatenate(all_pulls)
        width_global = float(np.std(data_global))
        kurt_global = calculate_true_kurtosis(data_global)

    if rank == 0 and plot_enabled:

        os.makedirs(plot_dir, exist_ok=True)

        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3)

        x_axis = np.linspace(-5, 5, 100)
        norm_pdf = 1.0 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_axis ** 2)

        def plot_hist(ax, data_list, title_base, color):
            if len(data_list) > 0:
                d = np.concatenate(data_list)
                if len(d) > 100000:
                    d = np.random.choice(d, 100000, replace=False)

                mu = np.mean(d)
                sig = np.std(d)
                kurt = calculate_true_kurtosis(d)

                ax.hist(
                    d,
                    bins=50,
                    range=(-5, 5),
                    density=True,
                    alpha=0.6,
                    color=color,
                    label="Data",
                )
                ax.plot(x_axis, norm_pdf, "r--", label="Normal(0,1)")
                ax.set_title(
                    f"{title_base}\nMean={mu:.2f}, Width={sig:.2f}, M4={kurt:.2f}"
                )
                ax.legend(fontsize="small")
            else:
                ax.text(0.5, 0.5, "No Data", ha="center")
                ax.set_title(title_base)

        ax_low = fig.add_subplot(gs[0, 0])
        plot_hist(ax_low, pulls_low, "Low (<0.6)", "green")
        ax_med = fig.add_subplot(gs[0, 1])
        plot_hist(ax_med, pulls_med, "Med (0.6-6.0)", "blue")
        ax_high = fig.add_subplot(gs[0, 2])
        plot_hist(ax_high, pulls_high, "High (>6.0)", "red")

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/epoch_{epoch}_regimes.png")
        plt.close()

        if example_data is not None:
            src_ex, tgt_ex, sigma_ex = example_data

            fig2, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

            im0 = axes[0].imshow(
                src_ex,
                aspect="auto",
                interpolation="none",
                cmap="viridis",
            )
            axes[0].set_title("Input Waveforms")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(
                tgt_ex,
                aspect="auto",
                interpolation="none",
                cmap="viridis",
            )
            axes[1].set_title("Target Waveforms")
            plt.colorbar(im1, ax=axes[1])

            im2 = axes[2].imshow(
                sigma_ex,
                aspect="auto",
                interpolation="none",
                cmap="magma",
            )
            axes[2].set_title("Predicted Sigma")
            plt.colorbar(im2, ax=axes[2])

            plt.tight_layout()
            plt.savefig(f"{plot_dir}/epoch_{epoch}_2d_heatmaps.png")
            plt.close()

            peak_amps = np.max(np.abs(tgt_ex), axis=1)
            top_indices = np.argsort(peak_amps)[::-1][:5]

            if len(top_indices) > 0:
                fig3, axes3 = plt.subplots(
                    len(top_indices),
                    1,
                    figsize=(12, 3 * len(top_indices)),
                    sharex=True,
                )

                if len(top_indices) == 1:
                    axes3 = [axes3]

                time_axis = np.arange(tgt_ex.shape[1])

                for i, ch_idx in enumerate(top_indices):
                    ax = axes3[i]
                    wave_in = src_ex[ch_idx]
                    wave_tgt = tgt_ex[ch_idx]
                    sigma = sigma_ex[ch_idx]

                    ax.plot(
                        time_axis,
                        wave_tgt,
                        "k-",
                        linewidth=2,
                        alpha=0.8,
                        label="Target",
                    )
                    ax.plot(
                        time_axis,
                        wave_in,
                        "g-",
                        linewidth=1,
                        alpha=0.6,
                        label="Input",
                    )
                    ax.fill_between(
                        time_axis,
                        wave_tgt - sigma,
                        wave_tgt + sigma,
                        color="red",
                        alpha=0.3,
                        label="Target +/- sigma",
                    )

                    ax.set_title(f"Channel {ch_idx} (Peak: {peak_amps[ch_idx]:.3f})")
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.legend(loc="upper right")

                plt.tight_layout()
                plt.savefig(f"{plot_dir}/epoch_{epoch}_1d_waveforms.png")
                plt.close()

    return width_global, kurt_global


def save_training_checkpoint(
    out_path,
    model,
    optimizer,
    *,
    epoch,
    waveform_len,
    val_width,
    val_kurt,
    train_loss,
    train_nll,
    train_pull,
    config,
):
    payload = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": int(epoch),
        "waveform_len": int(waveform_len),
        "metrics": {
            "val_width": float(val_width),
            "val_kurt": float(val_kurt),
            "train_loss": float(train_loss),
            "train_nll": float(train_nll),
            "train_pull": float(train_pull),
        },
        "config": dict(config),
    }
    torch.save(payload, out_path)


# ================= Main ================= #

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--out-dir", default="./runs/var_run")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--calib-lambda", type=float, default=5.0)
    parser.add_argument("--demo-every", type=int, default=5)
    parser.add_argument("--demo-batches", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=1234)

    args = parser.parse_args()

    use_ddp = "LOCAL_RANK" in os.environ

    if use_ddp:

        dist.init_process_group("nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()

        torch.cuda.set_device(local_rank)

        device = torch.device("cuda", local_rank)

    else:

        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = rank == 0

    os.makedirs(args.out_dir, exist_ok=True)

    ds = WaveformDataset(args.zarr_path)

    waveform_len = ds.inputs.shape[2]

    n_val = int(0.1 * len(ds))
    n_train = len(ds) - n_val

    split_generator = torch.Generator().manual_seed(args.split_seed)
    ds_tr, ds_val = random_split(
        ds,
        [n_train, n_val],
        generator=split_generator,
    )

    sampler = DistributedSampler(ds_tr) if use_ddp else None

    loader = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = ConformerVarPredictor(
        waveform_len=waveform_len
    ).to(device)

    if use_ddp:
        # `pos_enc` is a fixed buffer, so per-forward buffer broadcasts are
        # unnecessary and can deadlock if only rank 0 runs evaluation.
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    eval_model = model.module if use_ddp else model

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    plot_dir = os.path.join(args.out_dir, "validation_plots")
    checkpoint_path = os.path.join(args.out_dir, "checkpoint.pt")
    best_model_path = os.path.join(args.out_dir, "best_model.pt")
    best_score = float("inf")

    if is_main_process:
        print("START TRAINING")

    for epoch in range(args.epochs):

        model.train()
        running_loss = 0.0
        running_nll = 0.0
        running_pull = 0.0
        batch_count = 0

        if sampler is not None:
            sampler.set_epoch(epoch)

        for src, tgt in loader:

            src = src.to(device)
            tgt = tgt.to(device)

            opt.zero_grad(set_to_none=True)

            logvar = model(src)
            loss, nll, pull_std = gaussian_nll_loss(
                src,
                tgt,
                logvar,
                args.calib_lambda,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            running_loss += loss.item()
            running_nll += nll
            running_pull += pull_std
            batch_count += 1

        train_metrics = torch.tensor(
            [running_loss, running_nll, running_pull, float(batch_count)],
            device=device,
            dtype=torch.float64,
        )

        if use_ddp:
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)

        denom = max(train_metrics[3].item(), 1.0)
        train_loss = train_metrics[0].item() / denom
        train_nll = train_metrics[1].item() / denom
        train_pull = train_metrics[2].item() / denom

        plot_enabled = (args.demo_every > 0) and (epoch % args.demo_every == 0)

        if is_main_process:
            val_width, val_kurt = validate_and_demo(
                eval_model,
                val_loader,
                epoch,
                device,
                rank=rank,
                plot_dir=plot_dir,
                plot_enabled=plot_enabled,
                max_batches=args.demo_batches,
            )
        else:
            val_width, val_kurt = 0.0, 0.0

        val_metrics = torch.tensor(
            [val_width, val_kurt],
            device=device,
            dtype=torch.float64,
        )

        if use_ddp:
            dist.broadcast(val_metrics, src=0)

        val_width = val_metrics[0].item()
        val_kurt = val_metrics[1].item()

        if is_main_process:
            ckpt_config = {
                "num_channels": int(NUM_CHANNELS),
                "waveform_len": int(waveform_len),
                "model_dim": 128,
                "num_layers": 6,
                "num_heads": 4,
                "input_scale": float(ds.input_scale),
                "target_scale": float(ds.target_scale),
                "calib_lambda": float(args.calib_lambda),
            }
            save_training_checkpoint(
                checkpoint_path,
                eval_model,
                opt,
                epoch=epoch,
                waveform_len=waveform_len,
                val_width=val_width,
                val_kurt=val_kurt,
                train_loss=train_loss,
                train_nll=train_nll,
                train_pull=train_pull,
                config=ckpt_config,
            )
            score = abs(val_width - 1.0) + 0.1 * abs(val_kurt - 3.0)
            if score < best_score:
                best_score = score
                save_training_checkpoint(
                    best_model_path,
                    eval_model,
                    opt,
                    epoch=epoch,
                    waveform_len=waveform_len,
                    val_width=val_width,
                    val_kurt=val_kurt,
                    train_loss=train_loss,
                    train_nll=train_nll,
                    train_pull=train_pull,
                    config=ckpt_config,
                )
            demo_note = " [demo]" if plot_enabled else ""
            print(
                f"Epoch {epoch}  Loss {train_loss:.4f}  "
                f"NLL {train_nll:.4f}  PullStd {train_pull:.3f}  "
                f"ValWidth {val_width:.3f}  ValKurt {val_kurt:.3f}"
                f"{demo_note}"
            )

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
