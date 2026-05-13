from __future__ import annotations

from typing import Literal

import numpy as np


def _correlation_sum_fft(residual: np.ndarray, predicted: np.ndarray, max_shift: int) -> np.ndarray:
    try:
        from scipy.signal import fftconvolve
    except Exception:
        return _correlation_sum_numpy(residual, predicted, max_shift)

    full = fftconvolve(
        np.asarray(residual, dtype=np.float32),
        np.asarray(predicted[:, ::-1], dtype=np.float32),
        mode="full",
        axes=1,
    )
    n_ticks = int(predicted.shape[1])
    return np.asarray(full[:, n_ticks - 1 : n_ticks - 1 + int(max_shift) + 1].sum(axis=0), dtype=np.float64)


def _correlation_sum_numpy(residual: np.ndarray, predicted: np.ndarray, max_shift: int) -> np.ndarray:
    n_ticks = int(predicted.shape[1])
    corr = np.zeros(int(max_shift) + 1, dtype=np.float64)
    for row in range(int(predicted.shape[0])):
        full = np.correlate(
            np.asarray(residual[row], dtype=np.float64),
            np.asarray(predicted[row], dtype=np.float64),
            mode="full",
        )
        corr += full[n_ticks - 1 : n_ticks - 1 + int(max_shift) + 1]
    return corr


def unit_likelihood_curve_with_base_v1(
    predicted: np.ndarray,
    base: np.ndarray,
    actual: np.ndarray,
    *,
    search_range: int = 800,
    engine: Literal["auto", "fft", "numpy"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast fixed-grid t0 scan for phase-1 track/shower placement.

    This computes the unit-std objective

        mean((shift(predicted, t0) + base - actual) ** 2)

    for every integer t0 in [0, search_range] without materializing each
    shifted waveform. It intentionally omits variance weighting and ADC
    clipping; phase 1 uses std=1 and saturated channels are already vetoed.
    """
    pred = np.asarray(predicted, dtype=np.float32)
    if pred.ndim != 2:
        raise ValueError(f"predicted must have shape (n_channel, n_tick); got {pred.shape!r}")
    if pred.size == 0:
        raise ValueError("predicted has no entries")

    residual = np.asarray(actual, dtype=np.float32) - np.asarray(base, dtype=np.float32)
    if residual.shape != pred.shape:
        raise ValueError(f"base/actual shape {residual.shape!r} does not match predicted {pred.shape!r}")

    n_ticks = int(pred.shape[1])
    max_shift = min(int(search_range), int(n_ticks) - 1)
    shifts = np.arange(max_shift + 1, dtype=np.int32)

    const = float(np.sum(residual * residual, dtype=np.float64))
    pred2_by_tick = np.sum(pred * pred, axis=0, dtype=np.float64)
    pred2_prefix = np.concatenate(([0.0], np.cumsum(pred2_by_tick, dtype=np.float64)))
    shifted2 = pred2_prefix[n_ticks - shifts]

    engine_key = str(engine).lower()
    if engine_key == "numpy":
        corr = _correlation_sum_numpy(residual, pred, max_shift)
    else:
        corr = _correlation_sum_fft(residual, pred, max_shift)

    errors = (const + shifted2 - 2.0 * corr) / float(pred.size)
    return shifts, np.asarray(errors, dtype=np.float32)


__all__ = ["unit_likelihood_curve_with_base_v1"]
