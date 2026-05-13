from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

NUM_CHANNELS = 120
DEFAULT_INPUT_SCALE = 1e-3
DEFAULT_TARGET_SCALE = 1e-3


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x).transpose(1, 2).contiguous()
        y = self.pw1(y)
        y = self.glu(y)
        y = self.dw(y)
        y = self.gn(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y.transpose(1, 2).contiguous()


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        ln = self.ln_attn(x)
        if ln.device.type == "cuda":
            with torch.autocast("cuda", enabled=False):
                attn_out, _ = self.attn(
                    ln.float(),
                    ln.float(),
                    ln.float(),
                    need_weights=False,
                )
        else:
            attn_out, _ = self.attn(ln, ln, ln, need_weights=False)
        x = x + attn_out.to(x.dtype)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class ConformerVarPredictor(nn.Module):
    def __init__(
        self,
        num_channels: int = NUM_CHANNELS,
        waveform_len: int = 1000,
        model_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_channels, model_dim)
        self.register_buffer(
            "pos_enc",
            self._make_sinusoidal(waveform_len, model_dim),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [ConformerBlock(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(model_dim, num_channels)

    @staticmethod
    def _make_sinusoidal(length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(1, length, dim)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        t_len = x.size(1)
        h = self.input_proj(x)
        h = h + self.pos_enc[:, :t_len, :]
        for layer in self.layers:
            h = layer(h)
        out = self.output_proj(h)
        logvar = out.permute(0, 2, 1)
        return torch.clamp(logvar, -12.0, 12.0)


def resolve_checkpoint(candidate_paths: list[str] | tuple[str, ...]) -> str:
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No variance checkpoint found. Checked:\n"
        + "\n".join(f"  - {path}" for path in candidate_paths)
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    clean_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("module."):
            clean_state[key[7:]] = value
        else:
            clean_state[key] = value
    return clean_state


def _infer_model_kwargs(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    num_channels = int(state_dict["input_proj.weight"].shape[1])
    model_dim = int(state_dict["input_proj.weight"].shape[0])
    waveform_len = int(state_dict["pos_enc"].shape[1]) if "pos_enc" in state_dict else 1000
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    }
    num_layers = max(layer_ids) + 1 if layer_ids else 6
    return {
        "num_channels": num_channels,
        "waveform_len": waveform_len,
        "model_dim": model_dim,
        "num_layers": num_layers,
        "num_heads": 4,
    }


def load_model(
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
    waveform_len: int | None = None,
    num_channels: int | None = None,
    model_dim: int | None = None,
    num_layers: int | None = None,
    num_heads: int = 4,
) -> tuple[ConformerVarPredictor, dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Variance checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    inferred = _infer_model_kwargs(state_dict)

    model = ConformerVarPredictor(
        num_channels=int(num_channels or inferred["num_channels"]),
        waveform_len=int(waveform_len or inferred["waveform_len"]),
        model_dim=int(model_dim or inferred["model_dim"]),
        num_layers=int(num_layers or inferred["num_layers"]),
        num_heads=int(num_heads),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    meta: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "model_kwargs": {
            "num_channels": int(num_channels or inferred["num_channels"]),
            "waveform_len": int(waveform_len or inferred["waveform_len"]),
            "model_dim": int(model_dim or inferred["model_dim"]),
            "num_layers": int(num_layers or inferred["num_layers"]),
            "num_heads": int(num_heads),
        },
        "checkpoint_type": type(checkpoint).__name__,
    }
    if isinstance(checkpoint, dict):
        for key in ("config", "metrics", "epoch"):
            if key in checkpoint:
                meta[key] = checkpoint[key]
    print(f"Variance model loaded from {checkpoint_path} to {device}")
    return model, meta


def predict(
    model: ConformerVarPredictor,
    waveforms: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 16,
    input_scale: float = DEFAULT_INPUT_SCALE,
    target_scale: float = DEFAULT_TARGET_SCALE,
    device: str | torch.device | None = None,
    return_variance: bool = True,
    min_sigma: float = 1.0,
) -> np.ndarray:
    if isinstance(waveforms, torch.Tensor):
        wave_np = waveforms.detach().cpu().numpy()
    else:
        wave_np = np.asarray(waveforms)

    single_sample = False
    if wave_np.ndim == 2:
        single_sample = True
        wave_np = wave_np[np.newaxis, ...]
    if wave_np.ndim != 3:
        raise ValueError(
            f"Expected waveform shape (N, C, T) or (C, T), got {tuple(wave_np.shape)}"
        )

    use_device = torch.device(device) if device is not None else next(model.parameters()).device
    wave_np = np.asarray(wave_np, dtype=np.float32)
    preds: list[np.ndarray] = []

    model.to(use_device)
    model.eval()
    with torch.no_grad():
        for start in range(0, wave_np.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), wave_np.shape[0])
            batch = torch.from_numpy(wave_np[start:stop]).to(use_device, dtype=torch.float32)
            batch = batch * float(input_scale)
            logvar = model(batch)
            sigma_scaled = torch.exp(0.5 * logvar).clamp(min=1e-6, max=1e6)
            sigma_raw = sigma_scaled / float(target_scale)
            sigma_raw = sigma_raw.clamp(min=float(min_sigma))
            if return_variance:
                preds.append((sigma_raw ** 2).cpu().numpy())
            else:
                preds.append(sigma_raw.cpu().numpy())

    out = np.concatenate(preds, axis=0).astype(np.float32)
    if single_sample:
        return out[0]
    return out
