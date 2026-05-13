"""
pulse_shapes.py
---------------
Utilities for forming scintillation light waveforms from either:
  (1) a saved average pulse template, or
  (2) a bi-exponential analytic model;
plus a fast, vectorized time-shift (fractional tick) interpolator.

Design goals
- Vectorized across arbitrary leading dimensions (amplitude shape S).
- Minimal assumptions: template may be provided as array or loaded from disk.
- Optional GPU support: `timeinterpolation` detects CuPy arrays and stays on-GPU.
- Clean, importable module (no I/O at import time).

Public API
- pulseShapeFormation(max_amplitudes, T=1000, *, template=None, template_path=None, template_peak=1000.0) -> np.ndarray
- pulseShapeFormationWithExp(max_amplitudes, t0=105, alpha=0.2088343868968168, beta=0.01639388470224096, T=1000) -> np.ndarray
- timeinterpolation(waveforms, shift, *, baseline=-28000.0) -> ndarray
Aliases (snake_case):
- pulse_shape_from_template = pulseShapeFormation
- pulse_shape_bi_exp       = pulseShapeFormationWithExp
- time_interpolate         = timeinterpolation
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import numpy as np


# ----------------------------- helpers -----------------------------

def _load_template_from_path(path: str) -> np.ndarray:
    """
    Load a 1D template from .npy OR .npz (expects key 'avg' for .npz).
    Returns a float32 array.
    """
    if path.endswith('.npy'):
        tmpl = np.load(path)
    else:
        # treat as npz; raise KeyError if 'avg' missing
        data = np.load(path)
        if 'avg' not in data.files:
            raise KeyError(f"Template NPZ at '{path}' must contain key 'avg'")
        tmpl = data['avg']
    tmpl = np.asarray(tmpl, dtype=np.float32)
    if tmpl.ndim != 1:
        raise ValueError(f"Template must be 1D; got shape {tmpl.shape}")
    return tmpl


def _infer_template_peak(template: np.ndarray) -> float:
    """Return the positive max as a reasonable peak normalization."""
    # Use nanmax to be resilient; fall back to 1.0 if degenerate.
    peak = float(np.nanmax(template)) if template.size else 1.0
    return peak if peak > 0 else 1.0


def _get_xp(arr):
    """
    Return the array module (numpy or cupy) matching `arr`.
    Falls back to numpy if CuPy is unavailable or `arr` is a NumPy array.
    """
    try:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return cp
    except Exception:
        pass
    return np


# ----------------------- template-based formation -----------------------

def pulseShapeFormation(
    max_amplitudes,
    T: int = 1000,
    *,
    template: Optional[np.ndarray] = None,
    template_path: Optional[str] = None,
    template_peak: Optional[float] = 1000.0,
) -> np.ndarray:
    """
    Build waveforms by scaling a stored average pulse template.

    Parameters
    ----------
    max_amplitudes : array-like, shape S
        Amplitude(s) per channel / detector (e.g., (N,48) or (48,) etc.).
    T : int
        Desired waveform length; must match the template length.
    template : (T,) array, optional
        Average pulse (peak ~1000). If None, loads from `template_path`.
    template_path : str, optional
        Path to .npy (1D) OR .npz (expects key 'avg') file holding the template.
        Ignored if `template` is provided.
    template_peak : float or None
        Peak normalization of the template. If None, will be inferred as max(template).
        Default 1000.0.

    Returns
    -------
    Y : ndarray, shape S + (T,)
        Waveforms: (amplitude / template_peak) * template.
    """
    amps = np.asarray(max_amplitudes, dtype=np.float32)  # shape S

    # Load / validate template
    if template is None:
        if template_path is None:
            raise ValueError("Either `template` or `template_path` must be provided.")
        tmpl = _load_template_from_path(template_path)
    else:
        tmpl = np.asarray(template, dtype=np.float32)

    if tmpl.ndim != 1:
        raise ValueError(f"Template must be 1D of length T; got shape {tmpl.shape}")
    if tmpl.shape[0] != T:
        raise ValueError(f"Template length {tmpl.shape[0]} != T={T}")

    # Determine peak normalization
    peak = _infer_template_peak(tmpl) if (template_peak is None) else float(template_peak)

    # Broadcast template across leading dims of amps
    scale = (amps / peak)[..., None]                     # S + (1,)
    tmpl_b = tmpl.reshape((1,) * amps.ndim + (T,))       # (1,...,1,T)
    Y = scale * tmpl_b                                   # S + (T,)
    return Y


# ----------------------- bi-exponential formation -----------------------

def pulseShapeFormationWithExp(
    max_amplitudes,
    t0: float = 105,
    alpha: float = 0.2088343868968168,
    beta:  float = 0.01639388470224096,
    T: int = 1000,
) -> np.ndarray:
    """
    Vectorized waveform generator using a two-exponential model.

    Parameters
    ----------
    max_amplitudes : array-like, shape S
        Peak amplitude(s) per channel (arbitrary leading shape S).
    t0 : float
        Start tick of the pulse (can be non-integer; rounded when applied).
    alpha, beta : float
        Decay rates for the fast and slow components.
    T : int
        Number of time samples.

    Returns
    -------
    Y : ndarray, shape S + (T,)
        For each element i in S:
          Y[i, t] = 0                                  for t < t0_i
                   = 0.7*M_i*exp(-alpha*(t - t0_i))
                     + 0.3*M_i*exp(-beta *(t - t0_i))  for t >= t0_i
    """
    amps = np.asarray(max_amplitudes, dtype=np.float64)   # shape S
    S = amps.shape

    # Broadcast t0 to shape S, round and clip to [0, T-1]
    t0_b = np.asarray(t0, dtype=np.float64)
    try:
        t0_b = np.broadcast_to(t0_b, S)
    except ValueError as e:
        raise ValueError(f"`t0` must be scalar or broadcastable to shape {S}") from e
    t0_b = np.clip(np.rint(t0_b), 0, T - 1).astype(np.float64)  # keep float for broadcasting
    t0_e = t0_b[..., None]                                      # S + (1,)

    # Time grid broadcastable with S
    t = np.arange(T, dtype=np.float64).reshape((1,) * amps.ndim + (T,))  # (1,...,1,T)

    # Positive deltas + mask for t >= t0
    dt = t - t0_e                            # S + (T,)
    mask = (dt >= 0.0)
    dt_pos = np.where(mask, dt, 0.0)

    # 70/30 split (broadcasted)
    A = 0.70 * amps[..., None]               # S + (1,)
    B = 0.30 * amps[..., None]               # S + (1,)

    # Build output; mask zeros pre-t0
    Y = (A * np.exp(-alpha * dt_pos) + B * np.exp(-beta * dt_pos)) * mask
    return Y.astype(np.float32, copy=False)


# --------------------------- time interpolation ---------------------------

def timeinterpolation(
    waveforms,
    shift: float,
    *,
    baseline: float = -28000.0,
):
    """
    Shift a batch of waveforms by an *arbitrary* (possibly fractional) number of ticks
    using first‑order (linear) interpolation.

    Parameters
    ----------
    waveforms : ndarray, shape (N, T)
        Light signals whose baseline sits at `baseline`.  `N` can be 48 or any batch size.
    shift : float
        Positive values move the waveform *to the right* (later in time).
        Negative values move it to the left. Integer shifts are exact; fractional shifts
        blend neighbouring ticks with a linear weight proportional to the fractional part.
    baseline : float, default -28000.0
        Fill value for samples that fall outside the 0–(T-1) window after the shift.

    Returns
    -------
    ndarray, shape (N, T)
        Time‑shifted waveforms (same device as input; dtype preserved).
    """
    # Detect backend to keep data on CPU/GPU
    xp = _get_xp(waveforms)

    wvf = xp.asarray(waveforms)  # keep original dtype
    if wvf.ndim != 2:
        raise ValueError("`waveforms` must have shape (N, T)")

    n_chan, n_tick = wvf.shape

    # Pre‑fill with baseline (represents no light)
    out = xp.full_like(wvf, baseline)

    # For every *target* tick j (0 … T-1) we want to sample at *source*
    # position j − shift from the original waveform.
    src_idx = xp.arange(n_tick, dtype=wvf.dtype) - shift  # (T,)

    i0 = xp.floor(src_idx).astype(xp.int64)
    frac = src_idx - i0                                   # 0 ≤ frac < 1 for valid indices
    i1 = i0 + 1

    # Mask of indices that are fully inside the valid range
    valid = (i0 >= 0) & (i1 < n_tick)
    if not bool(xp.any(valid)):
        # Entirely outside → nothing but baseline
        return out

    # Slice the *valid* portions only to keep things vectorised
    i0v = i0[valid]
    i1v = i1[valid]
    fv  = frac[valid]

    # Broadcast gather for all channels: shape (N, valid_count)
    out[:, valid] = ((1.0 - fv) * wvf[:, i0v] + fv * wvf[:, i1v])

    return out


# ------------------------------ aliases ------------------------------

# snake_case convenience aliases (optional)
pulse_shape_from_template = pulseShapeFormation
pulse_shape_bi_exp       = pulseShapeFormationWithExp
time_interpolate         = timeinterpolation


__all__ = [
    "pulseShapeFormation",
    "pulseShapeFormationWithExp",
    "timeinterpolation",
    "pulse_shape_from_template",
    "pulse_shape_bi_exp",
    "time_interpolate",
]


# ---------------------- TPC extraction / pushback ----------------------

def extract_TPC_waveforms(image: np.ndarray, TPCid: int, lookup_table: Dict[Tuple[int,int,int], Tuple[int,int]]) -> np.ndarray:
    """
    Extract waveforms for a single TPC from a full (8, 64, 1000) image.
    Returns (48, 1000) in (L0, R0, L1, R1, ..., L23, R23) order.
    """
    waveforms = []
    for i in range(24):
        vert_pos = 23 - i
        adc_L, ch_L = lookup_table[(TPCid, 0, vert_pos)]
        adc_R, ch_R = lookup_table[(TPCid, 1, vert_pos)]
        waveforms.append(image[adc_L, ch_L, :])
        waveforms.append(image[adc_R, ch_R, :])
    return np.stack(waveforms).astype(np.float32)


def extract_TPC_waveforms_multi(image: np.ndarray, TPCids, lookup_table: Dict[Tuple[int,int,int], Tuple[int,int]]) -> np.ndarray:
    """
    Extract and stack waveforms for multiple TPCs from a full (8, 64, 1000) image.
    Returns array shaped (len(TPCids) * 48, 1000).
    """
    T = image.shape[-1]
    waveforms = np.empty((len(TPCids) * 48, T), dtype=image.dtype)
    for i, TPCid in enumerate(TPCids):
        for j in range(24):
            v = 23 - j
            aL, cL = lookup_table[(TPCid, 0, v)]
            aR, cR = lookup_table[(TPCid, 1, v)]
            base = i * 48 + 2 * j
            waveforms[base]     = image[aL, cL]
            waveforms[base + 1] = image[aR, cR]
    return waveforms.astype(np.float32)


def TPC_pushback_multi(patch: np.ndarray, TPCids, baseline: float, lookup_table: Dict[Tuple[int,int,int], Tuple[int,int]]) -> np.ndarray:
    """
    Map a stacked (len(TPCids) * 48, T) patch back into a sparse full (8, 64, T) image.
    Unknown channels are filled with `baseline`.
    """
    T = patch.shape[-1]
    result = np.full((8, 64, T), baseline, dtype=patch.dtype)

    n = len(TPCids)
    indices = np.empty((n * 48, 2), dtype=np.int32)  # (adc, ch) pairs
    for i, TPCid in enumerate(TPCids):
        for j in range(24):
            v = 23 - j
            aL, cL = lookup_table[(TPCid, 0, v)]
            aR, cR = lookup_table[(TPCid, 1, v)]
            base = i * 48 + 2 * j
            indices[base]     = (aL, cL)
            indices[base + 1] = (aR, cR)

    adc_idx, ch_idx = indices[:, 0], indices[:, 1]
    result[adc_idx, ch_idx] = patch  # vectorized scatter
    return result
