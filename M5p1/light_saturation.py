"""Light-readout saturation ceiling: one definition for the whole chain (v1.2.2).

The light digitiser stores int16 samples; the physical rail is ``2**15 - 1``.  The chain
works on baseline-subtracted waveforms (``raw - mean(raw[:75])`` per channel per event), so
in that unit the rail sits at ``ADC_RAIL_RAW - baseline`` and differs channel by channel and
event by event.  Before v1.2.2 every stage used the constant 60780 (a pedestal of about
-28000 baked in), which is only right for a fixed pedestal.

Everything that clips a model waveform or decides that a channel is saturated must go
through the arrays produced here:

* ``format_light_event(...)``  -> (formatted, ceiling, clipped) for one light event, in the
  chain's charge-TPC / channel order.
* ``clip_hi(adc_clip, tpc, channels)`` -> the upper bound to hand to ``np.clip`` /
  ``np.minimum`` for one TPC block of shape (n_ch, T).  Accepts the legacy scalar or the
  per-event ``(n_tpc, n_ch)`` ceiling array, so old call sites keep working.
"""
from __future__ import annotations

from typing import Any

import numpy as np

ADC_RAIL_RAW = float(2 ** 15 - 1)     # int16 digitiser rail (32767)
BASELINE_SAMPLES = 75                 # pre-trigger samples that define the dynamic baseline
LEGACY_CLIP = 60780.0                 # pre-v1.2.2 constant; fallback only when no ceiling exists
VETO_MARGIN = 80.0                    # samples within this many counts of the rail count as clipped
                                      # (legacy: 60780 - 60700)


def baseline_and_ceiling(raw_samples: np.ndarray, n_baseline: int = BASELINE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel dynamic baseline and the rail expressed in baseline-subtracted units."""
    raw = np.asarray(raw_samples)
    baseline = np.mean(raw[..., : int(n_baseline)].astype(np.float64), axis=-1)
    ceiling = ADC_RAIL_RAW - baseline
    return baseline.astype(np.float32), ceiling.astype(np.float32)


def format_light_event(
    samples_event: np.ndarray,
    tpc_to_channels: dict[int, list[tuple[int, int]]],
    n_light_tpc: int,
    light_tpc_to_charge_tpc,
    *,
    n_channels: int = 120,
    waveform_len: int = 1000,
    clipped_event: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Format one light event (n_adc, 64, T) into (n_tpc, n_channels, T) plus per-channel
    ceiling (n_tpc, n_channels) and clipped flag (n_tpc, n_channels).

    ``clipped_event`` is the FLOW ``light/wvfm/data['clipped']`` row when available
    (identical to "touches the rail at least once"); otherwise it is derived from the data.
    """
    raw = np.asarray(samples_event)
    baseline, ceiling_raw = baseline_and_ceiling(raw)
    samples_bl = (raw.astype(np.float32) - baseline[..., None]).astype(np.float32)
    if clipped_event is None:
        clipped_raw = np.any(raw >= ADC_RAIL_RAW, axis=-1)
    else:
        clipped_raw = np.asarray(clipped_event, dtype=bool)

    n_tpc = int(n_light_tpc)
    formatted = np.zeros((n_tpc, int(n_channels), int(waveform_len)), dtype=np.float32)
    # Channels that are not read out keep the legacy constant so any stray use is harmless.
    ceiling = np.full((n_tpc, int(n_channels)), LEGACY_CLIP, dtype=np.float32)
    clipped = np.zeros((n_tpc, int(n_channels)), dtype=bool)
    for light_tpc in range(n_tpc):
        channels = tpc_to_channels.get(int(light_tpc), [])
        if len(channels) != int(n_channels):
            continue
        charge_tpc = int(light_tpc_to_charge_tpc(int(light_tpc)))
        adc_idx = np.asarray([item[0] for item in channels], dtype=np.intp)
        ch_idx = np.asarray([item[1] for item in channels], dtype=np.intp)
        formatted[charge_tpc] = samples_bl[adc_idx, ch_idx, : int(waveform_len)]
        ceiling[charge_tpc] = ceiling_raw[adc_idx, ch_idx]
        clipped[charge_tpc] = clipped_raw[adc_idx, ch_idx]
    return formatted, ceiling, clipped


def legacy_ceiling(shape: tuple[int, int]) -> np.ndarray:
    """A ceiling array filled with the pre-v1.2.2 constant (for inputs without raw samples)."""
    return np.full(tuple(int(v) for v in shape), LEGACY_CLIP, dtype=np.float32)


def ceiling_from_waveform(full_light_waveform: np.ndarray) -> np.ndarray:
    """Best estimate of the ceiling when only formatted waveforms exist (old dumps, plots):
    channels that touch the rail reveal it as their maximum; the rest keep the constant."""
    wave = np.asarray(full_light_waveform, dtype=np.float32)
    mx = wave.max(axis=-1)
    return np.where(mx >= LEGACY_CLIP - VETO_MARGIN, mx, np.float32(LEGACY_CLIP)).astype(np.float32)


def is_scalar_clip(adc_clip: Any) -> bool:
    return adc_clip is None or np.isscalar(adc_clip) or np.ndim(adc_clip) == 0


def clip_hi(adc_clip: Any, tpc: int | None = None, channels: Any = None) -> Any:
    """Upper clip bound for one TPC block of shape (n_ch, T) (or (len(channels), T)).

    ``adc_clip`` may be the legacy scalar or the per-event ceiling array (n_tpc, n_ch); a
    (n_ch,) vector is accepted too.  The result broadcasts against the trailing tick axis.
    """
    if is_scalar_clip(adc_clip):
        return float(LEGACY_CLIP if adc_clip is None else adc_clip)
    arr = np.asarray(adc_clip, dtype=np.float32)
    if arr.ndim == 2:
        if tpc is None:
            raise ValueError("clip_hi: per-event ceiling needs a TPC index")
        arr = arr[int(tpc)]
    if channels is not None:
        arr = arr[np.asarray(channels)]
    return arr[..., None]


def clip_hi_at_tick(adc_clip: Any, tpc: int | None = None, channels: Any = None) -> Any:
    """Same as ``clip_hi`` but for a single-tick column of shape (n_ch,)."""
    out = clip_hi(adc_clip, tpc, channels)
    return out if np.isscalar(out) else out[..., 0]


def clip_hi_stack(adc_clip: Any, tpcs: Any) -> Any:
    """Upper bound for a stacked block (len(tpcs), n_ch, T)."""
    if is_scalar_clip(adc_clip):
        return float(LEGACY_CLIP if adc_clip is None else adc_clip)
    arr = np.asarray(adc_clip, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("clip_hi_stack expects the per-event (n_tpc, n_ch) ceiling")
    return arr[np.asarray(tpcs, dtype=np.intp)][..., None]


def clip_hi_full(adc_clip: Any) -> Any:
    """Upper bound for the full (n_tpc, n_ch, T) image."""
    if is_scalar_clip(adc_clip):
        return float(LEGACY_CLIP if adc_clip is None else adc_clip)
    return np.asarray(adc_clip, dtype=np.float32)[..., None]


def stacked_row_ceiling(adc_clip: Any, block_meta: Any) -> Any:
    """Per-row bound (n_rows, 1) for fit inputs stacked by ``stack_cluster_fit_inputs_v10_3``:
    rows are the concatenation over ``block_meta`` items (skipping ``skipped_for_fit``) of the
    channels ``channel_indices`` of TPC ``tpcid``.  Scalar in, scalar out."""
    if is_scalar_clip(adc_clip):
        return float(LEGACY_CLIP if adc_clip is None else adc_clip)
    arr = np.asarray(adc_clip, dtype=np.float32)
    rows = []
    for item in block_meta:
        if bool(item.get("skipped_for_fit", False)):
            continue
        idx = np.asarray(item["channel_indices"], dtype=np.intp)
        if idx.size == 0:
            continue
        rows.append(arr[int(item["tpcid"])][idx])
    if not rows:
        return float(np.min(arr))
    return np.concatenate(rows)[:, None]


def clip_scalar(adc_clip: Any) -> float:
    """A single number for code paths that need one (e.g. raw prediction clip): the smallest
    ceiling in the event, never below zero."""
    if is_scalar_clip(adc_clip):
        return float(LEGACY_CLIP if adc_clip is None else adc_clip)
    return float(np.min(np.asarray(adc_clip, dtype=np.float32)))


def saturated_mask(full_light_waveform: np.ndarray, adc_clip: Any = None, *,
                   margin: float = VETO_MARGIN, max_clip_ticks: int = 6) -> np.ndarray:
    """Channels whose waveform sits within ``margin`` counts of the rail for more than
    ``max_clip_ticks`` samples.  ``adc_clip``: scalar or per-TPC ceiling matching the leading
    dims of the waveform ((n_tpc, n_ch) for a full image, (n_ch,) for one TPC)."""
    wave = np.asarray(full_light_waveform, dtype=np.float32)
    if is_scalar_clip(adc_clip):
        thr = float(LEGACY_CLIP if adc_clip is None else adc_clip) - float(margin)
        return np.sum(wave > thr, axis=-1) > int(max_clip_ticks)
    thr = np.asarray(adc_clip, dtype=np.float32) - np.float32(margin)
    return np.sum(wave > thr[..., None], axis=-1) > int(max_clip_ticks)


__all__ = [
    "ADC_RAIL_RAW", "BASELINE_SAMPLES", "LEGACY_CLIP", "VETO_MARGIN",
    "baseline_and_ceiling", "format_light_event", "legacy_ceiling", "ceiling_from_waveform",
    "is_scalar_clip", "clip_hi", "clip_hi_at_tick", "clip_hi_stack", "clip_hi_full", "clip_scalar",
    "stacked_row_ceiling", "saturated_mask",
]
