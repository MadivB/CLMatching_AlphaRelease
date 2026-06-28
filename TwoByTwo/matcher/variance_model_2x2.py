"""
Learned light-variance model for the 2x2 matcher (the v3 `var_prediction`
ConformerVarPredictor2D_TPCAware).

The matcher's chi2 denominator is the per-(tpc, channel, tick) variance.  The v3
pipeline used this learned, TPC-aware, two-sided-sigma model rather than a
hand-rolled noise estimate — it captures the real per-channel / signal-dependent
/ saturation noise structure and sharpens flash discrimination.

Channel-order note: the variance net was trained on the OLD interleaved order
(channel k -> side=k%2, y_rel=23-(k//2)), whereas this package works in the
perceiver's ORDERED_KEYS order.  We therefore feed the net old-order waveforms
and permute its output back to ORDERED_KEYS.

    var_new[j] = var_old[ 2*(23 - y_rel_j) + side_j ]   with (side_j,y_rel_j)=ORDERED_KEYS[j]
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

import geometry_2x2 as geo

_VARDIR = os.environ.get(
    "VAR2X2_DIR", "/global/cfs/cdirs/dune/users/yuxuan/2x2CLMatching/var_prediction")
if _VARDIR not in sys.path:
    sys.path.insert(0, _VARDIR)

DEFAULT_VAR_CKPT = os.path.join(_VARDIR, "checkpoints/model_prob_2D_v3-test_best.pt")

# NEW(ORDERED_KEYS) -> OLD interleaved index
_NEW_TO_OLD = np.array(
    [2 * (23 - y) + s for (s, y) in geo.ORDERED_KEYS], dtype=np.int64)


class VarianceModel2x2:
    def __init__(self, checkpoint: str = DEFAULT_VAR_CKPT, *,
                 device: Optional[str] = None):
        import torch
        import inference as _inf
        self._inf = _inf
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = _inf.load_model(checkpoint, device=device,
                                     num_channels=geo.N_CHANNELS, num_tpcs=geo.N_TPCS)

    def _old_order_waveform(self, raw_sub: np.ndarray, tbl) -> np.ndarray:
        """(8,64,1000) physical -> (8,48,1000) in OLD interleaved order."""
        out = np.zeros((geo.N_TPCS, geo.N_CHANNELS, raw_sub.shape[-1]), np.float32)
        for t in range(geo.N_TPCS):
            for k in range(geo.N_CHANNELS):
                key = (t, k % 2, 23 - (k // 2))
                if key in tbl.lut:
                    adc, ch = tbl.lut[key]
                    out[t, k] = raw_sub[adc, ch]
        return out

    def predict_variance(self, raw_sub: np.ndarray, tbl, *,
                         dead_mask: Optional[np.ndarray] = None,
                         var_floor: float = 1.0, big_var: float = 1.0e12,
                         batch_size: int = 64) -> np.ndarray:
        """Return (8,48,1000) variance in ORDERED_KEYS order."""
        old_wvfm = self._old_order_waveform(raw_sub, tbl)        # (8,48,1000) OLD
        tpc_ids = np.arange(geo.N_TPCS, dtype=np.int64)
        sigma_old = self._inf.predict(self.model, old_wvfm, tpc_ids,
                                      batch_size=batch_size, input_scale=1e-3,
                                      device=self.device)
        var_old = (np.asarray(sigma_old, np.float32) * 1000.0) ** 2  # (8,48,1000)
        var_new = var_old[:, _NEW_TO_OLD, :]                          # -> ORDERED_KEYS
        var_new = np.maximum(var_new, np.float32(var_floor))
        if dead_mask is not None:
            var_new[dead_mask] = np.float32(big_var)
        return var_new.astype(np.float32)


_CACHE = {}


def load_variance_model(checkpoint: str = DEFAULT_VAR_CKPT,
                        device: Optional[str] = None) -> VarianceModel2x2:
    key = (checkpoint, device or "")
    if key not in _CACHE:
        _CACHE[key] = VarianceModel2x2(checkpoint, device=device)
    return _CACHE[key]


__all__ = ["VarianceModel2x2", "load_variance_model", "DEFAULT_VAR_CKPT"]
