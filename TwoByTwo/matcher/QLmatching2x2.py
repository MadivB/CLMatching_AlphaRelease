"""
Clean public entry point for the 2x2 charge-light matcher — the 2x2 analogue of
``QLmatchingND.py``.

    import h5py, QLmatching2x2 as ql
    h5 = h5py.File(flow_file, "r")
    matched_t0, hit_ids = ql.run(h5, eventid=5)            # one event
    matched_t0, hit_ids = ql.runMultiple(h5, eventid=[5, 14, 27])

``matched_t0`` is the per-hit reconstructed t0 in matching ticks (the shift that
places the scintillation pulse template, peak@105, onto the observed flash);
``hit_ids`` are the row indices into ``charge/calib_prompt_hits/data`` aligned
with ``matched_t0``.  Unmatched hits carry t0 = -1.

The perceiver model is loaded once and cached.  Pick the sim/data checkpoint
with ``mode`` (default "sim"); override the checkpoint via ``QL2X2_CKPT`` or the
``checkpoint`` kwarg.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Tuple

import numpy as np

import pipeline_2x2 as _pipe
import light_model_2x2 as _lm

_MODEL_CACHE: dict = {}


def _get_model(mode: str, checkpoint: str, device: str):
    key = (mode, checkpoint or "", device or "")
    m = _MODEL_CACHE.get(key)
    if m is None:
        m = _lm.load_light_model(mode, checkpoint=checkpoint, device=device)
        _MODEL_CACHE[key] = m
    return m


def run(h5: Any, *, eventid: int, mode: str = "sim",
        checkpoint: str = None, device: str = None, dead_yaml: str = "",
        verbose: bool = False, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Run the 2x2 charge-light matching pipeline on one event.

    Returns (matched_t0, hit_ids); both empty if the event has no light.
    """
    checkpoint = checkpoint or os.environ.get("QL2X2_CKPT") or None
    model = _get_model(mode, checkpoint, device)
    res = _pipe.run_pipeline_for_event(
        h5, int(eventid), light_model=model, dead_yaml=dead_yaml,
        verbose=verbose, **kw)
    if res is None:
        return np.empty(0, np.float32), np.empty(0, np.int64)
    return (np.asarray(res["hit_timestamps"], np.float32),
            np.asarray(res["hit_refs"], np.int64))


def runMultiple(h5: Any, *, eventid: Iterable[int], **kw):
    """Run on several events; returns concatenated (matched_t0, hit_ids)."""
    all_t0, all_ids = [], []
    for ev in np.atleast_1d(eventid):
        t0, ids = run(h5, eventid=int(ev), **kw)
        all_t0.append(t0)
        all_ids.append(ids)
    if not all_t0:
        return np.empty(0, np.float32), np.empty(0, np.int64)
    return np.concatenate(all_t0), np.concatenate(all_ids)


def run_full(h5: Any, *, eventid: int, **kw):
    """Like ``run`` but returns the full pipeline result dict (clusters, logs,
    base image, t0 candidates, …) for inspection / plotting."""
    mode = kw.pop("mode", "sim")
    checkpoint = kw.pop("checkpoint", None) or os.environ.get("QL2X2_CKPT") or None
    device = kw.pop("device", None)
    model = _get_model(mode, checkpoint, device)
    return _pipe.run_pipeline_for_event(h5, int(eventid), light_model=model, **kw)


def clearCaches() -> None:
    import data_2x2
    data_2x2.clear_cache()
    _MODEL_CACHE.clear()


__all__ = ["run", "runMultiple", "run_full", "clearCaches"]
