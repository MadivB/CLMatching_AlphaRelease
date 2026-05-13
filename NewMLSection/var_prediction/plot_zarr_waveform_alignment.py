#!/usr/bin/env python3
"""
Plot 1D actual vs predicted waveform overlays from a waveform zarr.

The input zarr is expected to contain:
  - inputs  : actual waveforms, shape (N, C, T)
  - targets : predicted waveforms, shape (N, C, T)

By default, plots are saved under:
  <var_prediction>/figures/<zarr_name>/
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


def open_zarr_r(path: str):
    try:
        return zarr.open(path, mode="r")
    except Exception:
        return zarr.open_group(zarr.DirectoryStore(path), mode="r")


def default_out_dir(zarr_path: str) -> str:
    name = os.path.basename(os.path.normpath(zarr_path))
    if name.endswith(".zarr"):
        name = name[:-5]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "figures", name)


def sample_indices(eligible: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if eligible.size == 0:
        return eligible
    count = min(n_samples, eligible.size)
    chosen = rng.choice(eligible, size=count, replace=False)
    return np.sort(chosen.astype(np.int64))


def stratified_sample_indices(
    inputs,
    eligible: np.ndarray,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    if eligible.size == 0:
        return eligible

    peaks = np.zeros(eligible.size, dtype=np.float32)
    for start in range(0, eligible.size, batch_size):
        stop = min(start + batch_size, eligible.size)
        batch_idx = eligible[start:stop]
        batch = np.asarray(inputs[batch_idx], dtype=np.float32)
        peaks[start:stop] = np.max(batch, axis=(1, 2))

    order = np.argsort(peaks)
    ordered_idx = eligible[order]
    count = min(int(n_samples), ordered_idx.size)
    if count <= 0:
        return np.zeros(0, dtype=np.int64)

    picks = np.linspace(0, ordered_idx.size - 1, count, dtype=np.int64)
    chosen = ordered_idx[picks]
    return np.sort(np.asarray(chosen, dtype=np.int64))


def choose_channels(actual: np.ndarray, predicted: np.ndarray, n_channels: int) -> np.ndarray:
    score = np.maximum(np.max(actual, axis=1), np.max(predicted, axis=1))
    order = np.argsort(score)[::-1]
    return order[: min(n_channels, order.size)]


def find_eligible_samples(
    inputs,
    min_amplitude: float,
    batch_size: int,
    target_count: int,
    oversample_factor: int,
) -> np.ndarray:
    n_samples = inputs.shape[0]
    eligible_chunks = []
    target_pool = max(target_count * max(1, oversample_factor), target_count)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch = np.asarray(inputs[start:stop], dtype=np.float32)
        peaks = np.max(batch, axis=(1, 2))
        local = np.flatnonzero(peaks >= min_amplitude)
        if local.size > 0:
            eligible_chunks.append(local + start)
        if sum(chunk.size for chunk in eligible_chunks) >= target_pool:
            break

    if not eligible_chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(eligible_chunks).astype(np.int64)


def write_selection_summary(
    out_path: str,
    sample_indices_chosen: np.ndarray,
    selected_channels: dict[int, np.ndarray],
    inputs,
    targets,
):
    with open(out_path, "w", encoding="ascii") as f:
        for sample_idx in sample_indices_chosen:
            actual = np.asarray(inputs[int(sample_idx)], dtype=np.float32)
            predicted = np.asarray(targets[int(sample_idx)], dtype=np.float32)
            f.write(f"sample {int(sample_idx)}\n")
            for ch in selected_channels[int(sample_idx)]:
                actual_peak_tick = int(np.argmax(actual[int(ch)]))
                pred_peak_tick = int(np.argmax(predicted[int(ch)]))
                tick_delta = pred_peak_tick - actual_peak_tick
                actual_peak = float(np.max(actual[int(ch)]))
                pred_peak = float(np.max(predicted[int(ch)]))
                f.write(
                    f"  channel {int(ch):3d}  "
                    f"actual_peak={actual_peak:10.3f} @ {actual_peak_tick:4d}  "
                    f"pred_peak={pred_peak:10.3f} @ {pred_peak_tick:4d}  "
                    f"delta={tick_delta:4d}\n"
                )
            f.write("\n")


def plot_sample(
    sample_idx: int,
    actual: np.ndarray,
    predicted: np.ndarray,
    channels: np.ndarray,
    out_path: str,
):
    nrows = len(channels)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.0 * nrows), sharex=True, squeeze=False)

    for row, ch in enumerate(channels):
        ax = axes[row, 0]
        actual_wave = actual[int(ch)]
        pred_wave = predicted[int(ch)]

        actual_peak_tick = int(np.argmax(actual_wave))
        pred_peak_tick = int(np.argmax(pred_wave))
        actual_peak = float(np.max(actual_wave))
        pred_peak = float(np.max(pred_wave))

        ax.plot(actual_wave, color="black", linewidth=1.5, label="actual")
        ax.plot(pred_wave, color="tab:red", linewidth=1.2, linestyle="--", label="predicted")
        ax.axvline(actual_peak_tick, color="black", alpha=0.35, linestyle=":")
        ax.axvline(pred_peak_tick, color="tab:red", alpha=0.35, linestyle=":")
        ax.set_ylabel(f"ch {int(ch)}")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"channel {int(ch)}  "
            f"actual peak {actual_peak:.1f} @ {actual_peak_tick}  "
            f"pred peak {pred_peak:.1f} @ {pred_peak_tick}  "
            f"delta {pred_peak_tick - actual_peak_tick}"
        )
        if row == 0:
            ax.legend(loc="upper right")

    axes[-1, 0].set_xlabel("Tick")
    fig.suptitle(f"Sample {sample_idx}: actual vs predicted waveform alignment", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--zarr-path", required=True, help="Input waveform zarr")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to var_prediction/figures/<zarr_name>/.",
    )
    parser.add_argument(
        "--min-amplitude",
        type=float,
        default=400.0,
        help="Require at least this actual amplitude somewhere in the sample",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="How many samples to plot",
    )
    parser.add_argument(
        "--channels-per-sample",
        type=int,
        default=6,
        help="How many strong channels to plot for each sample",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=256,
        help="Batch size used when scanning the zarr for high-amplitude samples",
    )
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=10,
        help="Collect this many times the requested sample count before random selection",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("random", "stratified_peak"),
        default="stratified_peak",
        help="How to choose the final plotted samples from the eligible pool",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = args.out_dir or default_out_dir(args.zarr_path)
    os.makedirs(out_dir, exist_ok=True)

    root = open_zarr_r(args.zarr_path)
    if "inputs" not in root or "targets" not in root:
        raise RuntimeError(f"{args.zarr_path} must contain 'inputs' and 'targets'")

    inputs = root["inputs"]
    targets = root["targets"]
    if inputs.shape != targets.shape:
        raise RuntimeError(
            f"inputs shape {inputs.shape} does not match targets shape {targets.shape}"
        )

    eligible = find_eligible_samples(
        inputs,
        min_amplitude=args.min_amplitude,
        batch_size=max(1, args.scan_batch_size),
        target_count=max(1, args.num_samples),
        oversample_factor=max(1, args.oversample_factor),
    )
    if eligible.size == 0:
        raise RuntimeError(
            f"No samples found with actual amplitude >= {args.min_amplitude}"
        )

    rng = np.random.default_rng(args.seed)
    if args.selection_mode == "stratified_peak":
        chosen_samples = stratified_sample_indices(
            inputs,
            eligible,
            args.num_samples,
            batch_size=max(1, args.scan_batch_size),
        )
    else:
        chosen_samples = sample_indices(eligible, args.num_samples, rng)

    selected_channels: dict[int, np.ndarray] = {}
    for sample_idx in chosen_samples:
        actual = np.asarray(inputs[int(sample_idx)], dtype=np.float32)
        predicted = np.asarray(targets[int(sample_idx)], dtype=np.float32)
        channels = choose_channels(actual, predicted, args.channels_per_sample)
        selected_channels[int(sample_idx)] = channels
        out_path = os.path.join(out_dir, f"sample_{int(sample_idx):06d}.png")
        plot_sample(
            sample_idx=int(sample_idx),
            actual=actual,
            predicted=predicted,
            channels=channels,
            out_path=out_path,
        )

    write_selection_summary(
        out_path=os.path.join(out_dir, "selected_waveforms.txt"),
        sample_indices_chosen=chosen_samples,
        selected_channels=selected_channels,
        inputs=inputs,
        targets=targets,
    )

    print(f"Saved {len(chosen_samples)} sample plots to {out_dir}")
    print(f"Selection summary: {os.path.join(out_dir, 'selected_waveforms.txt')}")


if __name__ == "__main__":
    main()
