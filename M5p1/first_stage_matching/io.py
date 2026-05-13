from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from .paths import configure_paths

configure_paths()

from lut import LUT
from ML_NDfull_perceiver import DEFAULT_TPC_YAML, NUM_TARGETS, light_tpc_to_charge_tpc, load_tpc_geometries


@dataclass(slots=True)
class EventData:
    h5: h5py.File
    event_id: int
    light_id: int
    hit_refs: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    energy: np.ndarray
    io_group: np.ndarray
    hit_tpc_id: np.ndarray
    full_light_waveform: np.ndarray
    geom_map: dict[int, Any]


_LIGHT_CHANNEL_ORDER_CACHE: dict[str, tuple[dict[int, list[tuple[int, int]]], int]] = {}
_TPC_GEOMETRY_CACHE: dict[str, dict[int, Any]] = {}
_EVENT_ID_ROW_CACHE: dict[str, dict[int, int]] = {}


def open_flow_file(data_file: str) -> h5py.File:
    return h5py.File(data_file, "r")


def build_sipm_lut(h5: h5py.File) -> tuple[dict[tuple[int, int, int], tuple[int, int]], int]:
    meta = h5["geometry_info/sipm_rel_pos"].attrs["meta"]
    data = h5["geometry_info/sipm_rel_pos/data"]
    sipm_rel_pos = LUT.from_array(meta, data)
    samples = h5["light/wvfm/data"]["samples"]
    n_adc = int(samples.shape[1])
    n_chan_per_adc = 64
    lut = {}
    for adc in range(n_adc):
        for ch in range(n_chan_per_adc):
            try:
                mapping = sipm_rel_pos[(adc, ch)]
            except Exception:
                continue
            if getattr(mapping, "size", None) == 0:
                continue
            tpc, side, y = mapping[0]
            tpc = int(tpc)
            side = int(side)
            y = int(y)
            if side not in (0, 1) or tpc < 0:
                continue
            lut[(tpc, side, y)] = (int(adc), int(ch))
    return lut, n_adc


def build_light_channel_order(h5: h5py.File) -> tuple[dict[int, list[tuple[int, int]]], int]:
    cache_key = str(getattr(h5, "filename", ""))
    if cache_key in _LIGHT_CHANNEL_ORDER_CACHE:
        return _LIGHT_CHANNEL_ORDER_CACHE[cache_key]

    lut, _n_adc = build_sipm_lut(h5)
    n_light_tpc = max(t for (t, _, _) in lut) + 1
    tpc_to_channels: dict[int, list[tuple[int, int]]] = {}
    for tpc in range(n_light_tpc):
        side0 = sorted(
            [(y, adc, ch) for (ltpc, side, y), (adc, ch) in lut.items() if ltpc == tpc and side == 0],
            key=lambda item: item[0],
        )
        side1 = sorted(
            [(y, adc, ch) for (ltpc, side, y), (adc, ch) in lut.items() if ltpc == tpc and side == 1],
            key=lambda item: item[0],
        )
        tpc_to_channels[int(tpc)] = [(int(adc), int(ch)) for (_y, adc, ch) in (side0 + side1)]
    result = (tpc_to_channels, n_light_tpc)
    if cache_key:
        _LIGHT_CHANNEL_ORDER_CACHE[cache_key] = result
    return result


def _load_tpc_geometries_cached(yaml_path: str) -> dict[int, Any]:
    cache_key = str(yaml_path)
    if cache_key not in _TPC_GEOMETRY_CACHE:
        _TPC_GEOMETRY_CACHE[cache_key] = load_tpc_geometries(yaml_path)
    return _TPC_GEOMETRY_CACHE[cache_key]


def _event_id_to_row(h5: h5py.File) -> dict[int, int]:
    cache_key = str(getattr(h5, "filename", ""))
    if cache_key not in _EVENT_ID_ROW_CACHE:
        event_ids = np.asarray(h5["charge/events/data"]["id"], dtype=np.int64)
        _EVENT_ID_ROW_CACHE[cache_key] = {int(event_id): int(i) for i, event_id in enumerate(event_ids)}
    return _EVENT_ID_ROW_CACHE[cache_key]


def _event_hit_refs(h5: h5py.File, event_id: int, hits_dataset: str) -> np.ndarray:
    ref_path = f"charge/events/ref/charge/{hits_dataset}/ref"
    region_path = f"charge/events/ref/charge/{hits_dataset}/ref_region"
    if region_path in h5:
        row = _event_id_to_row(h5).get(int(event_id))
        if row is not None:
            region = h5[region_path][int(row)]
            start = int(region["start"])
            stop = int(region["stop"])
            if stop > start:
                return np.asarray(h5[ref_path][start:stop, 1], dtype=np.int64)

    refs = h5[ref_path]
    return np.asarray(refs[refs[:, 0] == int(event_id), 1], dtype=np.int64)


def format_light_waveform_for_event(
    h5: h5py.File,
    light_id: int,
    *,
    n_channels: int = NUM_TARGETS,
    waveform_len: int = 1000,
) -> np.ndarray:
    """Format one light event into charge-TPC indexing: (n_charge_tpc, 120, 1000)."""
    tpc_to_channels, n_light_tpc = build_light_channel_order(h5)
    samples = h5["light/wvfm/data"]["samples"][int(light_id)]
    baseline = np.mean(samples[..., :75], axis=-1, keepdims=True)
    samples_bl = (samples - baseline).astype(np.float32)

    formatted = np.zeros((n_light_tpc, int(n_channels), int(waveform_len)), dtype=np.float32)
    for light_tpc in range(n_light_tpc):
        channels = tpc_to_channels.get(int(light_tpc), [])
        if len(channels) != int(n_channels):
            continue
        charge_tpc = int(light_tpc_to_charge_tpc(int(light_tpc)))
        adc_idx = np.asarray([item[0] for item in channels], dtype=np.intp)
        ch_idx = np.asarray([item[1] for item in channels], dtype=np.intp)
        formatted[charge_tpc] = samples_bl[adc_idx, ch_idx, : int(waveform_len)]
    return formatted


def charge_tpc_from_io_group(io_group: np.ndarray) -> np.ndarray:
    """Authoritative charge-TPC id from hit io_group.

    Reconstructed coordinates can fluctuate across TPC boundaries.  The hit
    readout group is the source of truth for assigning hits to charge TPCs.
    """
    io = np.asarray(io_group, dtype=np.int64)
    if np.any(io <= 0):
        bad = np.unique(io[io <= 0]).tolist()
        raise ValueError(f"io_group must be positive; got invalid values {bad!r}")
    return ((io - 1) // 2).astype(np.int32)


def assign_hits_to_charge_tpc(x: np.ndarray, y: np.ndarray, z: np.ndarray, geom_map: dict[int, Any]) -> np.ndarray:
    raise RuntimeError(
        "Geometry-based hit-to-TPC assignment is disabled. Use "
        "charge_tpc_from_io_group(io_group); io_group is the authoritative TPC source."
    )


def load_event(
    h5: h5py.File,
    event_id: int,
    *,
    hits_dataset: str = "calib_prompt_hits",
    yaml_path: str = DEFAULT_TPC_YAML,
) -> EventData:
    hits_full = h5[f"charge/{hits_dataset}/data"]
    hits_ref = h5[f"charge/events/ref/charge/{hits_dataset}/ref"]
    charge_light_ref = h5["charge/events/ref/light/events/ref"]
    geom_map = _load_tpc_geometries_cached(yaml_path)

    hit_refs = _event_hit_refs(h5, int(event_id), hits_dataset)
    if hit_refs.size == 0:
        raise RuntimeError(f"No charge hits found for event {int(event_id)}")

    hits_evt = hits_full[hit_refs]
    x = np.asarray(hits_evt["x"], dtype=np.float32)
    y = np.asarray(hits_evt["y"], dtype=np.float32)
    z = np.asarray(hits_evt["z"], dtype=np.float32)
    energy = np.asarray(hits_evt["E"], dtype=np.float32)
    io_group = np.asarray(hits_evt["io_group"], dtype=np.int32)
    hit_tpc_id = charge_tpc_from_io_group(io_group)

    light_refs = charge_light_ref[charge_light_ref[:, 0] == int(event_id)]
    if len(light_refs) == 0:
        raise RuntimeError(f"No light event found for charge event {int(event_id)}")
    light_id = int(light_refs[0, 1])
    full_light_waveform = format_light_waveform_for_event(h5, light_id)

    return EventData(
        h5=h5,
        event_id=int(event_id),
        light_id=int(light_id),
        hit_refs=hit_refs,
        x=x,
        y=y,
        z=z,
        energy=energy,
        io_group=io_group,
        hit_tpc_id=hit_tpc_id,
        full_light_waveform=full_light_waveform,
        geom_map=geom_map,
    )


__all__ = [
    "EventData",
    "open_flow_file",
    "build_sipm_lut",
    "build_light_channel_order",
    "format_light_waveform_for_event",
    "charge_tpc_from_io_group",
    "assign_hits_to_charge_tpc",
    "load_event",
]
