"""
Thin wrapper around the 2x2 HybridPerceiver3D (``ML_2x2_perceiver``) for the
matcher.

Given charge hits (x, y, z, E, tpc id) and per-hit cluster labels it returns
``imageMaps[(cluster_id, tpc_id)] = (48, 1000)`` predicted light waveforms, in
the ORDERED_KEYS channel order the model was trained on.

Small-blob defaults
-------------------
* ``min_prediction_threshold = 0`` — the stock inference floors predictions
  below 100 ADC to zero; that erases the light of faint/low-energy blobs, which
  is exactly the population we care about in the 2x2, so we keep everything and
  let the chi2 + noise model decide.
* ``raw_clip = (0, ADC_CLIP)`` — light is non-negative and bounded by the ADC
  rail.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

import geometry_2x2 as geo

# Make the Charge2Light package importable (perceiver + checkpoints live there).
_C2L_DIR = os.environ.get(
    "C2L_DIR", "/pscratch/sd/y/yuxuan/2x2QLMatching/Charge2Light")
if _C2L_DIR not in sys.path:
    sys.path.insert(0, _C2L_DIR)

import ML_2x2_perceiver as _ml  # noqa: E402

DEFAULT_SIM_CKPT = os.path.join(_C2L_DIR, "runs/2x2_run/best_model.pt")
DEFAULT_DATA_CKPT = os.path.join(_C2L_DIR, "runs/2x2_data_run/best_model.pt")
DEFAULT_PULSE_PATH = (
    "/global/cfs/cdirs/dune/users/yuxuan/interactLevel/clusteringStudy/"
    "dataDrivenLUTtable/MLApproach/CNNApproach/avg_pulse.npy")


class LightModel2x2:
    """Loaded perceiver + pulse template, with a convenient predict method."""

    def __init__(self, checkpoint: str, *, device: Optional[str] = None,
                 pulse_path: str = DEFAULT_PULSE_PATH,
                 target_scale: float = 1e-3,
                 min_prediction_threshold: float = 0.0,
                 raw_clip: Tuple[float, float] = (0.0, geo.ADC_CLIP)):
        self.model, self.meta = _ml.load_2x2_model(
            checkpoint, device=device, target_scale=target_scale)
        # template normalised to peak ~1.0 so predicted-waveform peak == phi
        tmpl = np.load(pulse_path).astype(np.float32)
        self.template = (tmpl / 1000.0).astype(np.float32)
        self.target_scale = float(target_scale)
        self.min_prediction_threshold = float(min_prediction_threshold)
        self.raw_clip = raw_clip

    def predict_image_maps(self, x, y, z, E, tpc_ids, labels, *,
                           include_noise: bool = False,
                           batch_size: int = 16) -> Tuple[Dict, Dict]:
        """Return (imageMaps, meta).  imageMaps[(cid, tpc)] -> (48, 1000)."""
        return _ml.process_clusters_to_imageMaps(
            np.asarray(x, np.float64), np.asarray(y, np.float64),
            np.asarray(z, np.float64), np.asarray(E, np.float64),
            np.asarray(tpc_ids, np.int32), np.asarray(labels, np.int64),
            model=self.model, target_scale=self.target_scale,
            template=self.template, include_noise=include_noise,
            batch_size=batch_size, raw_clip=self.raw_clip,
            min_prediction_threshold=self.min_prediction_threshold,
            device_policy="auto")

    def predict_single_image(self, xs, ys, zs, es, tpc) -> np.ndarray:
        """Predict one (48, 1000) waveform for a bag of hits in one TPC.

        Used by the rescue / family-rebuild paths that ask for the light of an
        arbitrary hit subset (label-free).
        """
        xs = np.asarray(xs, np.float64)
        if xs.size == 0:
            return np.zeros((geo.N_CHANNELS, geo.WVFM_LEN), dtype=np.float32)
        fake_labels = np.ones(xs.size, dtype=np.int64)
        fake_tpcs = np.full(xs.size, int(tpc), dtype=np.int32)
        maps, _ = self.predict_image_maps(
            xs, np.asarray(ys, np.float64), np.asarray(zs, np.float64),
            np.asarray(es, np.float64), fake_tpcs, fake_labels, batch_size=1)
        img = maps.get((1, int(tpc)))
        if img is None:
            return np.zeros((geo.N_CHANNELS, geo.WVFM_LEN), dtype=np.float32)
        return np.asarray(img, dtype=np.float32)


def load_light_model(mode: str = "sim", *, checkpoint: Optional[str] = None,
                     device: Optional[str] = None, **kw) -> LightModel2x2:
    """Convenience loader.  ``mode`` in {'sim', 'data'} picks the default ckpt."""
    if checkpoint is None:
        checkpoint = DEFAULT_DATA_CKPT if mode == "data" else DEFAULT_SIM_CKPT
    return LightModel2x2(checkpoint, device=device, **kw)


__all__ = ["LightModel2x2", "load_light_model",
           "DEFAULT_SIM_CKPT", "DEFAULT_DATA_CKPT", "DEFAULT_PULSE_PATH"]
