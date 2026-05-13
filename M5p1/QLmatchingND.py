from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import sys
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
M5P1_DIR = Path(__file__).resolve().parent
ML_DIR = ROOT_DIR / "NewMLSection"

for _path in (str(ML_DIR), str(M5P1_DIR)):
    if _path in sys.path:
        sys.path.remove(_path)

# Match the notebook import order so helper modules resolve consistently.
sys.path.insert(0, str(ML_DIR))
sys.path.insert(1, str(M5P1_DIR))

from ML_NDfull_perceiver import DEFAULT_TPC_YAML, load_perceiver_model, load_tpc_geometries
from v11_plotting_purpose_eval import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_PULSE_PATH,
    V11PlottingResources,
    get_formatted_light_waveforms,
    run_v11_pipeline_for_event,
)

_HITS_DSET = "calib_prompt_hits"
_FILE_CACHE: dict[str, dict[str, Any]] = {}
_SHARED_CACHE: dict[str, Any] = {
    "geom_map": None,
    "model": None,
    "wvfm_tmpl": None,
}


def _normalize_event_ids(eventid: int | Iterable[int]) -> np.ndarray:
    if np.isscalar(eventid):
        return np.asarray([int(eventid)], dtype=np.int64)
    arr = np.asarray(list(eventid), dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("eventid must contain at least one event id")
    return arr


def _get_file_key(h5: Any) -> str:
    return str(getattr(h5, "filename", f"<h5-{id(h5)}>"))


def _get_shared_objects() -> tuple[dict[int, Any], Any, np.ndarray]:
    if _SHARED_CACHE["geom_map"] is None:
        _SHARED_CACHE["geom_map"] = load_tpc_geometries(DEFAULT_TPC_YAML)

    if _SHARED_CACHE["model"] is None:
        device_env = os.environ.get("QLMATCHINGND_DEVICE", "").strip().lower()
        if device_env in {"cpu", "cuda"}:
            device = device_env
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_perceiver_model(DEFAULT_CHECKPOINT_PATH, device=device)
        _SHARED_CACHE["model"] = model

    if _SHARED_CACHE["wvfm_tmpl"] is None:
        _SHARED_CACHE["wvfm_tmpl"] = np.load(DEFAULT_PULSE_PATH).astype(np.float32) / 999.0

    return (
        _SHARED_CACHE["geom_map"],
        _SHARED_CACHE["model"],
        np.asarray(_SHARED_CACHE["wvfm_tmpl"], dtype=np.float32),
    )


def _get_file_arrays(h5: Any) -> dict[str, Any]:
    file_key = _get_file_key(h5)
    cached = _FILE_CACHE.get(file_key)
    if cached is not None:
        return cached

    hits_ref = np.asarray(h5[f"charge/events/ref/charge/{_HITS_DSET}/ref"][:], dtype=np.int64)
    charge_light_ref = np.asarray(h5["charge/events/ref/light/events/ref"][:], dtype=np.int64)
    all_formatted_wvfms = np.asarray(get_formatted_light_waveforms(h5), dtype=np.float32)

    cached = {
        "hits_ref": hits_ref,
        "charge_light_ref": charge_light_ref,
        "all_formatted_wvfms": all_formatted_wvfms,
    }
    _FILE_CACHE[file_key] = cached
    return cached


def _build_resources(h5: Any) -> V11PlottingResources:
    geom_map, model, wvfm_tmpl = _get_shared_objects()
    file_arrays = _get_file_arrays(h5)
    return V11PlottingResources(
        data_file=_get_file_key(h5),
        h5=h5,
        hits_full=h5[f"charge/{_HITS_DSET}/data"],
        hits_ref=np.asarray(file_arrays["hits_ref"], dtype=np.int64),
        charge_light_ref=np.asarray(file_arrays["charge_light_ref"], dtype=np.int64),
        geom_map=geom_map,
        all_formatted_wvfms=np.asarray(file_arrays["all_formatted_wvfms"], dtype=np.float32),
        model=model,
        wvfm_tmpl=np.asarray(wvfm_tmpl, dtype=np.float32),
    )


def clearCaches() -> None:
    _FILE_CACHE.clear()


def run(h5: Any, *, eventid: int, lam: float = 1.2, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the current ND charge-light matching pipeline on one event.

    Parameters
    ----------
    h5 : h5py.File
        Open FLOW file handle.
    eventid : int
        Charge event id to process.
    lam : float
        Track-clustering distance parameter.
    verbose : bool
        If True, print progress from the internal pipeline.

    Returns
    -------
    matched_t0 : ndarray
        Reconstructed per-hit t0 values in matching ticks.
    hit_ids : ndarray
        Hit row ids in `charge/calib_prompt_hits/data`, aligned with `matched_t0`.
    """
    resources = _build_resources(h5)
    result = run_v11_pipeline_for_event(
        resources,
        int(eventid),
        lam=float(lam),
        verbose=bool(verbose),
        enable_phase23_leftover_absorption=True,
    )
    matched_t0 = np.asarray(result["hit_timestamps"], dtype=np.float32)
    hit_ids = np.asarray(result["hit_refs"], dtype=np.int64)
    return matched_t0, hit_ids


def runMultiple(
    h5: Any,
    *,
    eventid: int | Iterable[int],
    lam: float = 1.2,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the current ND charge-light matching pipeline on one or more events.

    Parameters
    ----------
    h5 : h5py.File
        Open FLOW file handle.
    eventid : int or iterable[int]
        One event id or a list/array of event ids.
    lam : float
        Track-clustering distance parameter.
    verbose : bool
        If True, print progress from the internal pipeline.

    Returns
    -------
    matched_t0 : ndarray
        Concatenated reconstructed per-hit t0 values in matching ticks.
    hit_ids : ndarray
        Concatenated hit row ids in `charge/calib_prompt_hits/data`, aligned with `matched_t0`.
    """
    event_ids = _normalize_event_ids(eventid)
    all_t0: list[np.ndarray] = []
    all_hit_ids: list[np.ndarray] = []

    for ev_id in event_ids:
        matched_t0, hit_ids = run(
            h5,
            eventid=int(ev_id),
            lam=float(lam),
            verbose=bool(verbose),
        )
        all_t0.append(np.asarray(matched_t0, dtype=np.float32))
        all_hit_ids.append(np.asarray(hit_ids, dtype=np.int64))

    if not all_t0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    return np.concatenate(all_t0), np.concatenate(all_hit_ids)


__all__ = ["run", "runMultiple", "clearCaches"]
