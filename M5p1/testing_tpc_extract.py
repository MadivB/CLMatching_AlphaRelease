from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .truth_plotting import extract_truth_for_selected_hits
except Exception:
    from truth_plotting import extract_truth_for_selected_hits

TRUTH_T0_CONVENTION = "(segment_t0_us * 1000 - event_start_ns) / 16"


def _as_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _segment_cache(h5) -> dict[str, np.ndarray]:
    cache = getattr(_segment_cache, "_cache", {})
    key = (str(getattr(h5, "filename", "")), "mc_truth/segments/data")
    if key in cache:
        return cache[key]

    segments = h5["mc_truth/segments/data"][:]
    seg_ids = _as_array(segments["segment_id"], np.int64)
    seg_t0 = _as_array(segments["t0"], np.float64)

    out = {
        "seg_ids_sorted": None,
        "seg_t0_sorted": None,
    }
    order = np.argsort(seg_ids)
    out["seg_ids_sorted"] = seg_ids[order]
    out["seg_t0_sorted"] = seg_t0[order]

    if "vertex_id" in segments.dtype.names:
        out["vertex_id_sorted"] = _as_array(segments["vertex_id"], np.int64)[order]
    if "event_id" in segments.dtype.names:
        out["segment_event_id_sorted"] = _as_array(segments["event_id"], np.int64)[order]

    cache[key] = out
    _segment_cache._cache = cache
    return out


def _map_best_segments(best_segment_id: np.ndarray, has_truth: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = int(best_segment_id.shape[0])
    truth_raw = np.full(n, np.nan, dtype=np.float64)
    vertex_id = np.full(n, -1, dtype=np.int64)
    segment_event_id = np.full(n, -1, dtype=np.int64)

    query = _as_array(best_segment_id[has_truth], np.int64)
    if query.size == 0:
        return {
            "truth_raw": truth_raw,
            "vertex_id": vertex_id,
            "segment_event_id": segment_event_id,
        }

    seg_ids = cache["seg_ids_sorted"]
    pos = np.searchsorted(seg_ids, query)
    in_range = pos < seg_ids.size
    matched = np.zeros(query.shape[0], dtype=bool)
    matched[in_range] = seg_ids[pos[in_range]] == query[in_range]

    truth_rows = np.flatnonzero(has_truth)
    mapped_rows = truth_rows[matched]
    mapped_pos = pos[matched]

    truth_raw[mapped_rows] = cache["seg_t0_sorted"][mapped_pos]
    if "vertex_id_sorted" in cache:
        vertex_id[mapped_rows] = cache["vertex_id_sorted"][mapped_pos]
    if "segment_event_id_sorted" in cache:
        segment_event_id[mapped_rows] = cache["segment_event_id_sorted"][mapped_pos]

    return {
        "truth_raw": truth_raw,
        "vertex_id": vertex_id,
        "segment_event_id": segment_event_id,
    }


def extract_event_truth_t0_for_prompt_hits(
    h5,
    hit_refs: np.ndarray,
    *,
    convert_to_matching_ticks: bool = True,
    strict: bool = False,
) -> dict[str, np.ndarray | float | str]:
    """
    Return truth arrays aligned to the current event's calibrated prompt-hit order.

    The default converted truth convention matches the detector light-waveform
    convention:
        (segment_t0_us * 1000 - event_start_ns) / 16
    """
    hit_refs = _as_array(hit_refs, np.int64)
    n_hits = int(hit_refs.size)

    truth_raw = np.full(n_hits, np.nan, dtype=np.float64)
    truth_t0 = np.full(n_hits, np.nan, dtype=np.float32)
    best_segment_id = np.full(n_hits, -1, dtype=np.int64)
    best_fraction = np.full(n_hits, np.nan, dtype=np.float32)
    vertex_id = np.full(n_hits, -1, dtype=np.int64)
    segment_event_id = np.full(n_hits, -1, dtype=np.int64)

    required = [
        "mc_truth/calib_prompt_hit_backtrack/data",
        "mc_truth/segments/data",
    ]
    missing = [path for path in required if path not in h5]
    if missing:
        if strict:
            raise KeyError(f"Missing truth datasets: {missing}")
        return {
            "truth_t0": truth_t0,
            "truth_t0_raw": truth_raw,
            "best_segment_id": best_segment_id,
            "best_fraction": best_fraction,
            "vertex_id": vertex_id,
            "segment_event_id": segment_event_id,
            "raw_t0_min": np.nan,
            "truth_convention": TRUTH_T0_CONVENTION if convert_to_matching_ticks else "raw mc_truth/segments/data['t0']",
            "truth_status": f"missing truth datasets: {missing}",
        }

    backtrack = h5["mc_truth/calib_prompt_hit_backtrack/data"][hit_refs]
    segment_ids = _as_array(backtrack["segment_ids"], np.int64)
    fractions = _as_array(backtrack["fraction"], np.float32)

    valid = segment_ids >= 0
    has_truth = np.any(valid, axis=1)
    safe_fractions = np.where(valid, fractions, -np.inf)
    best_local_idx = np.argmax(safe_fractions, axis=1)
    truth_rows = np.flatnonzero(has_truth)
    if truth_rows.size:
        best_segment_id[truth_rows] = segment_ids[truth_rows, best_local_idx[truth_rows]]
        best_fraction[truth_rows] = fractions[truth_rows, best_local_idx[truth_rows]]

    mapped = _map_best_segments(best_segment_id, has_truth, _segment_cache(h5))
    truth_raw = mapped["truth_raw"]
    vertex_id = mapped["vertex_id"]
    segment_event_id = mapped["segment_event_id"]

    finite = np.isfinite(truth_raw)
    raw_t0_min = float(np.nanmin(truth_raw[finite])) if np.any(finite) else np.nan

    if convert_to_matching_ticks:
        corrected_truth = extract_truth_for_selected_hits(
            h5,
            hit_refs,
            convert_t0_to_matching_ticks=True,
        )
        truth_t0 = _as_array(corrected_truth["true_t0_rel"], np.float32)
        convention = TRUTH_T0_CONVENTION
    else:
        truth_t0 = truth_raw.astype(np.float32)
        convention = "raw mc_truth/segments/data['t0']"

    return {
        "truth_t0": truth_t0,
        "truth_t0_raw": truth_raw,
        "best_segment_id": best_segment_id,
        "best_fraction": best_fraction,
        "vertex_id": vertex_id,
        "segment_event_id": segment_event_id,
        "raw_t0_min": raw_t0_min,
        "truth_convention": convention,
        "truth_status": "ok",
    }


def extract_tpc_testing_info(
    namespace: dict[str, Any],
    tpcid: int,
    *,
    save_path: str | None = None,
    include_image_maps: bool = True,
    strict_truth: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Package all current post-Phase-2 information needed to test later-stage TPC logic.

    The returned arrays are aligned to the current event hit order, then subset to
    the designated TPC. `reco_t0` is the current `hit_timestamps` state after
    Phase 1+2. `true_t0` is converted to the same matching-tick convention.
    """
    tpcid = int(tpcid)
    h5 = namespace["h5"]
    hit_refs = _as_array(namespace["hit_refs"], np.int64)
    hit_tpc_id = _as_array(namespace["hitTPCid"], np.int32)
    tpc_mask = hit_tpc_id == tpcid
    event_hit_indices = np.flatnonzero(tpc_mask).astype(np.int64)

    if event_hit_indices.size == 0:
        raise RuntimeError(f"TPC {tpcid}: no calibrated prompt hits in current event.")

    truth = extract_event_truth_t0_for_prompt_hits(
        h5,
        hit_refs,
        convert_to_matching_ticks=True,
        strict=bool(strict_truth),
    )

    hits_evt = namespace.get("hits_evt")
    if hits_evt is None:
        hits_evt = h5["charge/calib_prompt_hits/data"][hit_refs]
    hits_evt = np.asarray(hits_evt)

    labels_global = _as_array(namespace["labels_global"], np.int32)
    hit_timestamps = _as_array(namespace["hit_timestamps"], np.float32)
    energies = _as_array(namespace["Eset"], np.float32)
    x = _as_array(namespace["xset"], np.float64)
    y = _as_array(namespace["yset"], np.float64)
    z = _as_array(namespace["zset"], np.float64)
    io_group = _as_array(namespace.get("io_group", np.full(hit_tpc_id.shape, -1)), np.int32)

    true_t0 = _as_array(truth["truth_t0"], np.float32)
    true_t0_raw = _as_array(truth["truth_t0_raw"], np.float64)
    best_segment_id = _as_array(truth["best_segment_id"], np.int64)
    best_fraction = _as_array(truth["best_fraction"], np.float32)
    vertex_id = _as_array(truth["vertex_id"], np.int64)

    track_shower_labels = set(int(v) for v in namespace.get("track_shower_labels", []))
    is_track_shower = np.isin(labels_global, np.fromiter(track_shower_labels, dtype=np.int64)) if track_shower_labels else np.zeros(labels_global.shape, dtype=bool)

    hit_table = np.empty(
        event_hit_indices.size,
        dtype=[
            ("event_hit_index", "i8"),
            ("calib_prompt_hit_index", "i8"),
            ("tpc", "i4"),
            ("label", "i4"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("energy_mev", "f4"),
            ("io_group", "i4"),
            ("reco_t0", "f4"),
            ("true_t0", "f4"),
            ("true_t0_raw", "f8"),
            ("truth_segment_id", "i8"),
            ("truth_fraction", "f4"),
            ("truth_vertex_id", "i8"),
            ("is_track_shower", "?"),
            ("is_assigned", "?"),
        ],
    )
    loc = event_hit_indices
    hit_table["event_hit_index"] = loc
    hit_table["calib_prompt_hit_index"] = hit_refs[loc]
    hit_table["tpc"] = hit_tpc_id[loc]
    hit_table["label"] = labels_global[loc]
    hit_table["x"] = x[loc]
    hit_table["y"] = y[loc]
    hit_table["z"] = z[loc]
    hit_table["energy_mev"] = energies[loc]
    hit_table["io_group"] = io_group[loc]
    hit_table["reco_t0"] = hit_timestamps[loc]
    hit_table["true_t0"] = true_t0[loc]
    hit_table["true_t0_raw"] = true_t0_raw[loc]
    hit_table["truth_segment_id"] = best_segment_id[loc]
    hit_table["truth_fraction"] = best_fraction[loc]
    hit_table["truth_vertex_id"] = vertex_id[loc]
    hit_table["is_track_shower"] = is_track_shower[loc]
    hit_table["is_assigned"] = np.isfinite(hit_timestamps[loc]) & (hit_timestamps[loc] >= 0)

    base_image = _as_array(namespace["baseImage"], np.float32)
    actual = _as_array(namespace["fullLightWaveform"], np.float32)
    std = _as_array(namespace.get("fullLightStd", np.ones_like(actual)), np.float32)

    assignment_info = namespace.get("assignment_info", {})
    assignment_info_tpc = {
        (int(cid), int(tpc)): dict(info)
        for (cid, tpc), info in assignment_info.items()
        if int(tpc) == tpcid
    }

    cluster_assignment_log = [
        dict(row)
        for row in namespace.get("cluster_assignment_log", [])
        if tpcid in [int(v) for v in row.get("tpcs", [])]
    ]

    image_maps_tpc = {}
    if include_image_maps:
        for (cid, tpc), image in namespace.get("imageMaps", {}).items():
            if int(tpc) == tpcid:
                image_maps_tpc[int(cid)] = np.asarray(image, dtype=np.float32).copy()

    saturated_channel_cache = namespace.get("saturated_channel_cache")
    if isinstance(saturated_channel_cache, dict) and "veto_mask" in saturated_channel_cache:
        saturated_mask = _as_array(saturated_channel_cache["veto_mask"][tpcid], bool)
    else:
        saturated_mask = np.sum(actual[tpcid] > 60700.0, axis=1) > 6

    labels_tpc = labels_global[loc]
    unique_labels = np.unique(labels_tpc)
    cluster_energy_by_label = {
        int(label): float(np.sum(energies[loc][labels_tpc == int(label)]))
        for label in unique_labels
    }

    out = {
        "tpcid": tpcid,
        "event_id": int(namespace.get("ev_id", -1)),
        "light_id": int(namespace.get("lightID", -1)),
        "n_hits": int(event_hit_indices.size),
        "n_assigned": int(np.count_nonzero(hit_table["is_assigned"])),
        "n_truth": int(np.count_nonzero(np.isfinite(hit_table["true_t0"]))),
        "event_hit_indices": event_hit_indices,
        "calib_prompt_hit_indices": hit_refs[loc].copy(),
        "hits": hits_evt[loc].copy(),
        "hit_table": hit_table,
        "x": x[loc].copy(),
        "y": y[loc].copy(),
        "z": z[loc].copy(),
        "energy_mev": energies[loc].copy(),
        "labels_global": labels_global[loc].copy(),
        "reco_t0": hit_timestamps[loc].copy(),
        "true_t0": true_t0[loc].copy(),
        "true_t0_raw": true_t0_raw[loc].copy(),
        "truth_segment_id": best_segment_id[loc].copy(),
        "truth_fraction": best_fraction[loc].copy(),
        "truth_vertex_id": vertex_id[loc].copy(),
        "truth_event_raw_t0_min": float(truth["raw_t0_min"]),
        "truth_t0_convention": str(truth["truth_convention"]),
        "truth_status": str(truth["truth_status"]),
        "actual_light_waveform": actual[tpcid].copy(),
        "predicted_light_waveform": base_image[tpcid].copy(),
        "light_std_waveform": std[tpcid].copy(),
        "saturated_channel_mask": saturated_mask.copy(),
        "keep_channel_indices": np.flatnonzero(~saturated_mask).astype(np.int32),
        "t0_candidates": [int(v) for v in namespace.get("t0Candidates", [[]])[tpcid]],
        "raw_flash_t0s": [int(v) for v in namespace.get("raw_v8_1_flash_seed_t0s_by_tpc", {}).get(tpcid, [])],
        "amended_flash_t0s": [int(v) for v in namespace.get("v8_1_flash_seed_t0s_by_tpc", {}).get(tpcid, [])],
        "assignment_info_tpc": assignment_info_tpc,
        "cluster_assignment_log_tpc": cluster_assignment_log,
        "cluster_energy_by_label": cluster_energy_by_label,
        "image_maps_tpc": image_maps_tpc,
        "phase2_stats": dict(namespace.get("v11_stage_stats", {})),
    }

    if save_path is not None:
        import torch

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, str(path))

    if verbose:
        print(
            f"TPC {tpcid}: hits={out['n_hits']} | assigned={out['n_assigned']} | "
            f"truth={out['n_truth']} | imageMaps={len(image_maps_tpc)} | "
            f"light_shape={out['actual_light_waveform'].shape}"
        )
        print(f"truth convention: {out['truth_t0_convention']}")
        if save_path is not None:
            print(f"saved: {save_path}")

    return out


__all__ = [
    "TRUTH_T0_CONVENTION",
    "extract_event_truth_t0_for_prompt_hits",
    "extract_tpc_testing_info",
]
