from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

try:
    from plottingTools import (
        VALID_GROUP_COLORS,
        group_hits_by_time,
        plot_3d_clusters_with_t0,
    )
except ModuleNotFoundError:
    from M5p1.plottingTools import (
        VALID_GROUP_COLORS,
        group_hits_by_time,
        plot_3d_clusters_with_t0,
    )


DEFAULT_MATCHING_TICK_SCALE = 1000.0 / 16.0
DEFAULT_MATCHING_TICK_OFFSET = 0.0
MATCHING_T0_CONVENTION = "(truth_t0_us * 1000 - event_start_ns) / 16"


def _as_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _normalize_truth_time(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    finite = np.isfinite(out)
    if np.any(finite):
        out[finite] = out[finite] - np.nanmin(out[finite])
    return out


def _event_row_for_event_id(h5: Any, event_id: int) -> int | None:
    if "charge/events/data" not in h5:
        return None

    events = h5["charge/events/data"]
    event_ids = _as_array(events["id"], np.int64)
    matches = np.flatnonzero(event_ids == int(event_id))
    if matches.size:
        return int(matches[0])
    if 0 <= int(event_id) < len(events):
        return int(event_id)
    return None


def _light_event_row_for_charge_event(h5: Any, event_id: int) -> int | None:
    if "light/events/data" not in h5:
        return None

    light_data = h5["light/events/data"]
    light_id: int | None = None

    ref_path = "charge/events/ref/light/events/ref"
    if ref_path in h5:
        refs = h5[ref_path][:]
        matches = refs[refs[:, 0].astype(np.int64) == int(event_id)]
        if len(matches):
            light_id = int(matches[0, 1])

    if light_id is None:
        light_id = int(event_id)

    light_ids = _as_array(light_data["id"], np.int64)
    matches = np.flatnonzero(light_ids == int(light_id))
    if matches.size:
        return int(matches[0])
    if 0 <= int(light_id) < len(light_data):
        return int(light_id)
    return None


def _infer_event_id_from_hit_indices(h5: Any, hit_indices: np.ndarray) -> int | None:
    if hit_indices.size == 0:
        return None
    region_path = "charge/events/ref/charge/calib_prompt_hits/ref_region"
    if region_path not in h5 or "charge/events/data" not in h5:
        return None

    regions = h5[region_path][:]
    lo = int(np.min(hit_indices))
    hi = int(np.max(hit_indices))
    for event_row, region in enumerate(regions):
        start = int(region["start"])
        stop = int(region["stop"])
        if start <= lo and hi < stop:
            events = h5["charge/events/data"]
            return int(events["id"][event_row])
    return None


def _matching_time_reference_ns(h5: Any, event_id: int | None) -> tuple[float, str]:
    """
    Return the absolute ns reference for matching ticks.

    Reconstructed t0 values use the detector event-start convention. Truth t0
    in matching ticks is `(truth_ns - event_start_ns) / 16`.
    """
    if event_id is None:
        return np.nan, "missing_event_id"

    charge_row = _event_row_for_event_id(h5, int(event_id))
    if charge_row is not None:
        row = h5["charge/events/data"][charge_row]
        names = row.dtype.names or ()
        if "unix_ts" in names and "unix_ts_usec" in names:
            ref_ns = float(row["unix_ts"]) * 1.0e9 + float(row["unix_ts_usec"]) * 1000.0
            if np.isfinite(ref_ns):
                return ref_ns, "charge/events:unix_ts+unix_ts_usec"

    light_row = _light_event_row_for_charge_event(h5, int(event_id))
    if light_row is not None:
        row = h5["light/events/data"][light_row]
        names = row.dtype.names or ()
        if "utime_ms" in names:
            utime_ms = float(np.asarray(row["utime_ms"]).ravel()[0])
            ref_ns = float(utime_ms) * 1.0e6
            if np.isfinite(ref_ns):
                return ref_ns, "light/events:utime_ms"

    return np.nan, "unavailable"


def _read_rows_by_index(dataset: Any, indices: np.ndarray) -> np.ndarray:
    """
    Read HDF5 rows aligned to arbitrary indices.

    h5py fancy indexing is fastest and safest with sorted unique indices; this
    helper preserves the caller's original order.
    """
    indices = _as_array(indices, np.int64)
    if indices.size == 0:
        return dataset[indices]

    order = np.argsort(indices)
    sorted_idx = indices[order]
    unique_idx, inverse = np.unique(sorted_idx, return_inverse=True)
    unique_rows = dataset[unique_idx]
    sorted_rows = unique_rows[inverse]
    out = np.empty(sorted_rows.shape, dtype=sorted_rows.dtype)
    out[order] = sorted_rows
    return out


def _segment_lookup(h5: Any) -> dict[str, np.ndarray]:
    cache = getattr(_segment_lookup, "_cache", {})
    key = (str(getattr(h5, "filename", "")), "mc_truth/segments/data")
    if key in cache:
        return cache[key]

    segments = h5["mc_truth/segments/data"][:]
    seg_ids = _as_array(segments["segment_id"], np.int64)
    order = np.argsort(seg_ids)

    out = {
        "segment_id": seg_ids[order],
        "t": _as_array(segments["t"], np.float64)[order] if "t" in segments.dtype.names else None,
        "t0": _as_array(segments["t0"], np.float64)[order],
        "vertex_id": (
            _as_array(segments["vertex_id"], np.int64)[order]
            if "vertex_id" in segments.dtype.names
            else None
        ),
    }

    cache[key] = out
    _segment_lookup._cache = cache
    return out


def _interaction_lookup(h5: Any) -> dict[int, dict[str, float]]:
    cache = getattr(_interaction_lookup, "_cache", {})
    key = (str(getattr(h5, "filename", "")), "mc_truth/interactions/data")
    if key in cache:
        return cache[key]

    if "mc_truth/interactions/data" not in h5:
        cache[key] = {}
        _interaction_lookup._cache = cache
        return {}

    interactions = h5["mc_truth/interactions/data"][:]
    names = interactions.dtype.names or ()
    out = {}
    for row in interactions:
        if "vertex_id" not in names:
            continue
        vid = int(row["vertex_id"])
        out[vid] = {
            "x": float(row["x_vert"]) if "x_vert" in names else np.nan,
            "y": float(row["y_vert"]) if "y_vert" in names else np.nan,
            "z": float(row["z_vert"]) if "z_vert" in names else np.nan,
            "t": float(row["t_vert"]) if "t_vert" in names else np.nan,
        }

    cache[key] = out
    _interaction_lookup._cache = cache
    return out


def extract_truth_for_selected_hits(
    h5: Any,
    hit_indices: np.ndarray,
    *,
    convert_t0_to_matching_ticks: bool = False,
    matching_tick_scale: float = DEFAULT_MATCHING_TICK_SCALE,
    matching_tick_offset: float = DEFAULT_MATCHING_TICK_OFFSET,
    event_id: int | None = None,
    strict: bool = True,
) -> dict[str, np.ndarray | str | float]:
    """
    Extract per-hit MC truth for calibrated prompt hits.

    Parameters
    ----------
    h5:
        Open HDF5 file handle.
    hit_indices:
        Calibrated prompt-hit table row indices. The returned arrays are aligned
        to this order.
    convert_t0_to_matching_ticks:
        If True, `true_t0_rel` is converted using the matching convention:
        `(true_t0_us * 1000 - event_start_ns) / 16`.
    event_id:
        Charge event id used to find the detector event-start reference. If not
        provided, it is inferred when all hit indices belong to one event.
    strict:
        If False, missing truth datasets return NaN/-1 arrays instead of raising.
    """
    hit_indices = _as_array(hit_indices, np.int64)
    n_hits = int(hit_indices.shape[0])

    required = [
        "mc_truth/calib_prompt_hit_backtrack/data",
        "mc_truth/segments/data",
    ]
    missing = [path for path in required if path not in h5]
    if missing:
        if strict:
            raise KeyError(f"Missing truth datasets: {missing}")
        return _empty_truth_info(
            n_hits,
            status=f"missing truth datasets: {missing}",
            convert_t0_to_matching_ticks=convert_t0_to_matching_ticks,
            matching_tick_scale=matching_tick_scale,
            matching_tick_offset=matching_tick_offset,
            event_id=event_id,
            reference_ns=np.nan,
            reference_source="missing_truth",
        )

    backtrack = _read_rows_by_index(
        h5["mc_truth/calib_prompt_hit_backtrack/data"],
        hit_indices,
    )

    best_segment_id = np.full(n_hits, -1, dtype=np.int64)
    best_fraction = np.zeros(n_hits, dtype=np.float32)

    segment_ids = _as_array(backtrack["segment_ids"], np.int64)
    fractions = _as_array(backtrack["fraction"], np.float32)
    valid = segment_ids >= 0
    has_truth = np.any(valid, axis=1)
    safe_fraction = np.where(valid, fractions, -np.inf)
    best_local = np.argmax(safe_fraction, axis=1)
    rows = np.flatnonzero(has_truth)
    if rows.size:
        best_segment_id[rows] = segment_ids[rows, best_local[rows]]
        best_fraction[rows] = fractions[rows, best_local[rows]]

    seg = _segment_lookup(h5)
    seg_ids_sorted = seg["segment_id"]
    query = best_segment_id[has_truth]
    pos = np.searchsorted(seg_ids_sorted, query)
    in_range = pos < seg_ids_sorted.size
    matched = np.zeros(query.shape[0], dtype=bool)
    matched[in_range] = seg_ids_sorted[pos[in_range]] == query[in_range]

    true_t = np.full(n_hits, np.nan, dtype=np.float64)
    true_t0 = np.full(n_hits, np.nan, dtype=np.float64)
    vertex_id = np.full(n_hits, -1, dtype=np.int64)

    truth_rows = np.flatnonzero(has_truth)
    mapped_rows = truth_rows[matched]
    mapped_pos = pos[matched]
    if mapped_rows.size:
        if seg["t"] is not None:
            true_t[mapped_rows] = seg["t"][mapped_pos]
        true_t0[mapped_rows] = seg["t0"][mapped_pos]
        if seg["vertex_id"] is not None:
            vertex_id[mapped_rows] = seg["vertex_id"][mapped_pos]

    vertex_truth = _interaction_lookup(h5)
    vertex_x = np.full(n_hits, np.nan, dtype=np.float64)
    vertex_y = np.full(n_hits, np.nan, dtype=np.float64)
    vertex_z = np.full(n_hits, np.nan, dtype=np.float64)
    vertex_t = np.full(n_hits, np.nan, dtype=np.float64)
    for i, vid in enumerate(vertex_id):
        if int(vid) < 0:
            continue
        info = vertex_truth.get(int(vid))
        if info is None:
            continue
        vertex_x[i] = info["x"]
        vertex_y[i] = info["y"]
        vertex_z[i] = info["z"]
        vertex_t[i] = info["t"]

    inferred_event_id = int(event_id) if event_id is not None else _infer_event_id_from_hit_indices(h5, hit_indices)
    reference_ns, reference_source = _matching_time_reference_ns(h5, inferred_event_id)
    raw_t0_min = float(np.nanmin(true_t0[np.isfinite(true_t0)])) if np.any(np.isfinite(true_t0)) else np.nan
    if np.isfinite(reference_ns):
        raw_t0_reference = float(reference_ns) / 1000.0
    else:
        raw_t0_reference = raw_t0_min
        reference_source = f"{reference_source}; fallback:selected_truth_min"

    true_t_rel = _normalize_truth_time(true_t)
    true_t0_rel = np.asarray(true_t0, dtype=np.float64).copy()
    finite_t0 = np.isfinite(true_t0_rel)
    if np.isfinite(raw_t0_reference):
        true_t0_rel[finite_t0] = true_t0_rel[finite_t0] - float(raw_t0_reference)

    if convert_t0_to_matching_ticks:
        finite = np.isfinite(true_t0_rel)
        true_t0_rel = true_t0_rel.copy()
        true_t0_rel[finite] = true_t0_rel[finite] * float(matching_tick_scale) + float(matching_tick_offset)
        convention = MATCHING_T0_CONVENTION
    else:
        convention = "relative mc_truth/segments/data['t0']"

    return {
        "best_segment_id": best_segment_id,
        "best_fraction": best_fraction,
        "true_t": true_t,
        "true_t_rel": true_t_rel,
        "true_t0": true_t0,
        "true_t0_rel": true_t0_rel,
        "vertex_id": vertex_id,
        "vertex_x": vertex_x,
        "vertex_y": vertex_y,
        "vertex_z": vertex_z,
        "vertex_t": vertex_t,
        "raw_t0_min": raw_t0_min,
        "raw_t0_reference": float(raw_t0_reference) if np.isfinite(raw_t0_reference) else np.nan,
        "truth_t0_reference_ns": float(reference_ns) if np.isfinite(reference_ns) else np.nan,
        "truth_t0_reference_source": str(reference_source),
        "event_id": -1 if inferred_event_id is None else int(inferred_event_id),
        "matching_tick_scale": float(matching_tick_scale),
        "matching_tick_offset": float(matching_tick_offset),
        "truth_t0_convention": convention,
        "truth_status": "ok",
    }


def _empty_truth_info(
    n_hits: int,
    *,
    status: str,
    convert_t0_to_matching_ticks: bool,
    matching_tick_scale: float,
    matching_tick_offset: float,
    event_id: int | None,
    reference_ns: float,
    reference_source: str,
) -> dict[str, np.ndarray | str | float]:
    convention = MATCHING_T0_CONVENTION if convert_t0_to_matching_ticks else "relative mc_truth/segments/data['t0']"
    return {
        "best_segment_id": np.full(n_hits, -1, dtype=np.int64),
        "best_fraction": np.zeros(n_hits, dtype=np.float32),
        "true_t": np.full(n_hits, np.nan, dtype=np.float64),
        "true_t_rel": np.full(n_hits, np.nan, dtype=np.float64),
        "true_t0": np.full(n_hits, np.nan, dtype=np.float64),
        "true_t0_rel": np.full(n_hits, np.nan, dtype=np.float64),
        "vertex_id": np.full(n_hits, -1, dtype=np.int64),
        "vertex_x": np.full(n_hits, np.nan, dtype=np.float64),
        "vertex_y": np.full(n_hits, np.nan, dtype=np.float64),
        "vertex_z": np.full(n_hits, np.nan, dtype=np.float64),
        "vertex_t": np.full(n_hits, np.nan, dtype=np.float64),
        "raw_t0_min": np.nan,
        "raw_t0_reference": np.nan,
        "truth_t0_reference_ns": float(reference_ns) if np.isfinite(reference_ns) else np.nan,
        "truth_t0_reference_source": str(reference_source),
        "event_id": -1 if event_id is None else int(event_id),
        "truth_t0_convention": convention,
        "truth_status": status,
        "matching_tick_scale": float(matching_tick_scale),
        "matching_tick_offset": float(matching_tick_offset),
    }


def subset_truth_info(truth_info: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    mask = _as_array(mask, bool)
    subset = {}
    for key, value in truth_info.items():
        if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
            subset[key] = value[mask]
        else:
            subset[key] = value
    return subset


def convert_truth_t0_rel_to_matching_ticks(
    truth_info: dict[str, Any],
    *,
    matching_tick_scale: float = DEFAULT_MATCHING_TICK_SCALE,
    matching_tick_offset: float = DEFAULT_MATCHING_TICK_OFFSET,
    copy: bool = True,
) -> dict[str, Any]:
    out = dict(truth_info) if copy else truth_info
    vals = _as_array(out["true_t0_rel"], np.float64).copy()
    finite = np.isfinite(vals)
    vals[finite] = vals[finite] * float(matching_tick_scale) + float(matching_tick_offset)
    out["true_t0_rel"] = vals
    out["truth_t0_convention"] = MATCHING_T0_CONVENTION
    return out


def build_truth_group_labels(
    truth_info: dict[str, Any],
    *,
    mode: str = "t0",
    time_window: int | float = 5,
) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "t0":
        safe_t0 = np.where(
            np.isfinite(_as_array(truth_info["true_t0_rel"], np.float64)),
            _as_array(truth_info["true_t0_rel"], np.float64),
            -1.0,
        )
        return group_hits_by_time(safe_t0, time_window=time_window)

    if mode == "vertex":
        vertex_ids = _as_array(truth_info["vertex_id"], np.int64)
        labels = np.full(vertex_ids.shape, -1, dtype=int)
        valid_vertices = sorted(int(v) for v in np.unique(vertex_ids) if int(v) >= 0)
        remap = {vid: i for i, vid in enumerate(valid_vertices)}
        for vid, label in remap.items():
            labels[vertex_ids == int(vid)] = int(label)
        return labels

    raise ValueError("mode must be 't0' or 'vertex'")


def plot_truth_hits_like_matching(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    truth_info: dict[str, Any],
    *,
    mode: str = "t0",
    time_window: int | float = 5,
    energies: np.ndarray | None = None,
    title: str = "Truth-colored hits",
    save_path: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """
    Plot truth-colored hits in the same style as matched-t0 visualisations.

    For `mode='t0'`, this delegates to `plot_3d_clusters_with_t0` to preserve
    the old notebook style. That helper always calls `fig.show()`.
    """
    mode = str(mode).lower()
    x = _as_array(x)
    y = _as_array(y)
    z = _as_array(z)
    if energies is not None:
        energies = _as_array(energies, np.float64)

    if mode == "t0":
        truth_labels = build_truth_group_labels(truth_info, mode="t0", time_window=time_window)
        truth_t0 = _as_array(truth_info["true_t0_rel"], np.float64)
        plot_3d_clusters_with_t0(
            x=x,
            y=y,
            z=z,
            labels=truth_labels,
            t0s=truth_t0,
            energies=energies,
            title=title,
            save_path=save_path,
        )
        return truth_labels

    if mode != "vertex":
        raise ValueError("mode must be 't0' or 'vertex'")

    vertex_ids = _as_array(truth_info["vertex_id"], np.int64)
    true_t = _as_array(truth_info["true_t_rel"], np.float64)
    true_t0 = _as_array(truth_info["true_t0_rel"], np.float64)

    valid_vertices = [int(v) for v in np.unique(vertex_ids) if int(v) >= 0]
    if energies is not None:
        vertex_energy = {
            vid: float(np.nansum(energies[vertex_ids == vid]))
            for vid in valid_vertices
        }
        ordered_vertices = sorted(valid_vertices, key=lambda vid: (-vertex_energy[vid], vid))
    else:
        vertex_energy = {
            vid: float(np.count_nonzero(vertex_ids == vid))
            for vid in valid_vertices
        }
        ordered_vertices = sorted(valid_vertices)

    fig = go.Figure()
    for i, vid in enumerate(ordered_vertices):
        mask = vertex_ids == int(vid)
        color_hex, color_name = VALID_GROUP_COLORS[i % len(VALID_GROUP_COLORS)]
        hover_text = [
            f"vertex={int(v_val)}<br>true t rel={t_val:.2f}<br>true t0 rel={t0_val:.2f}"
            for t_val, t0_val, v_val in zip(true_t[mask], true_t0[mask], vertex_ids[mask])
        ]
        fig.add_trace(
            go.Scatter3d(
                x=z[mask],
                y=y[mask],
                z=x[mask],
                mode="markers",
                marker=dict(size=4, color=color_hex, opacity=0.85, line=dict(width=0)),
                text=hover_text,
                hoverinfo="text+x+y+z",
                name=f"{color_name} - Vertex {vid} (E = {vertex_energy[vid]:.2f})",
            )
        )

    noise_mask = vertex_ids < 0
    if np.any(noise_mask):
        fig.add_trace(
            go.Scatter3d(
                x=z[noise_mask],
                y=y[noise_mask],
                z=x[noise_mask],
                mode="markers",
                marker=dict(size=4, color="gray", opacity=0.75, line=dict(width=0)),
                name=f"gray - Truth unavailable ({int(np.count_nonzero(noise_mask))} hits)",
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="z", yaxis_title="y", zaxis_title="x"),
        legend=dict(title="Truth Vertex", itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
        title=title,
        showlegend=True,
    )
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return build_truth_group_labels(truth_info, mode="vertex", time_window=time_window)


def plot_truth_for_selection(
    h5: Any,
    hit_refs: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    energies: np.ndarray | None = None,
    selection_mask: np.ndarray | None = None,
    mode: str = "t0",
    time_window: int | float = 10,
    convert_t0_to_matching_ticks: bool = True,
    title: str = "Truth-colored hits",
    save_path: str | None = None,
    show: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Extract truth for an event hit selection and plot it.

    `hit_refs`, `x`, `y`, `z`, and `energies` must be aligned to the current
    event hit order. `selection_mask` subsets those arrays before plotting.
    """
    hit_refs = _as_array(hit_refs, np.int64)
    x = _as_array(x)
    y = _as_array(y)
    z = _as_array(z)
    if energies is not None:
        energies = _as_array(energies, np.float64)

    truth_info_event = extract_truth_for_selected_hits(
        h5,
        hit_refs,
        convert_t0_to_matching_ticks=convert_t0_to_matching_ticks,
        strict=strict,
    )

    if selection_mask is None:
        selection_mask = np.ones(hit_refs.shape[0], dtype=bool)
    else:
        selection_mask = _as_array(selection_mask, bool)

    truth_subset = subset_truth_info(truth_info_event, selection_mask)
    energy_subset = energies[selection_mask] if energies is not None else None

    truth_labels = plot_truth_hits_like_matching(
        x=x[selection_mask],
        y=y[selection_mask],
        z=z[selection_mask],
        truth_info=truth_subset,
        mode=mode,
        time_window=time_window,
        energies=energy_subset,
        title=title,
        save_path=save_path,
        show=show,
    )

    return {
        "truth_info_event": truth_info_event,
        "truth_subset": truth_subset,
        "truth_labels": truth_labels,
        "selection_mask": selection_mask,
        "save_path": save_path,
    }


def plot_truth_from_namespace(
    namespace: dict[str, Any],
    *,
    target_tpc: int | None = None,
    mode: str = "t0",
    time_window: int | float = 10,
    convert_t0_to_matching_ticks: bool = True,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Notebook convenience wrapper.

    Required namespace keys:
        h5, hit_refs, xset, yset, zset, Eset, hitTPCid

    Example:
        from M5p1.truth_plotting import plot_truth_from_namespace
        out = plot_truth_from_namespace(globals(), target_tpc=15)
    """
    h5 = namespace["h5"]
    hit_refs = _as_array(namespace["hit_refs"], np.int64)
    x = _as_array(namespace["xset"])
    y = _as_array(namespace["yset"])
    z = _as_array(namespace["zset"])
    energies = _as_array(namespace["Eset"], np.float64)
    hit_tpc_id = _as_array(namespace["hitTPCid"], np.int32)

    if target_tpc is None:
        selection_mask = np.ones(hit_refs.shape[0], dtype=bool)
        tpc_title = "all TPCs"
    else:
        selection_mask = hit_tpc_id == int(target_tpc)
        tpc_title = f"TPC={int(target_tpc)}"

    ev_id = namespace.get("ev_id", namespace.get("event_id", "?"))
    if title is None:
        title = f"Truth-colored hits | event {ev_id} | mode={mode} | {tpc_title}"

    return plot_truth_for_selection(
        h5,
        hit_refs,
        x,
        y,
        z,
        energies=energies,
        selection_mask=selection_mask,
        mode=mode,
        time_window=time_window,
        convert_t0_to_matching_ticks=convert_t0_to_matching_ticks,
        title=title,
        save_path=save_path,
        show=show,
        strict=strict,
    )


def extract_event_hit_energy_and_truth_t0(
    h5: Any,
    hits_ref: np.ndarray,
    event_id: int | None = None,
    *,
    eventid: int | None = None,
    hits_full: Any | None = None,
    convert_to_matching_ticks: bool = True,
    strict: bool = True,
) -> dict[str, np.ndarray | str | float]:
    """
    Extract calibrated prompt-hit refs, energies, and truth t0 for one event.

    This preserves the old notebook call pattern:

        event_truth = extract_event_hit_energy_and_truth_t0(
            h5, hits_ref, ev_id, hits_full=hits_full,
            convert_to_matching_ticks=True,
        )

    The returned `truth_t0` is aligned to the event hit order.
    """
    hits_ref = _as_array(hits_ref, np.int64)
    if event_id is None:
        if eventid is None:
            raise TypeError("Pass event_id or eventid.")
        event_id = int(eventid)
    event_id = int(event_id)

    if hits_ref.ndim != 2 or hits_ref.shape[1] < 2:
        raise ValueError("hits_ref must be a 2D array with event id in column 0 and hit ref in column 1.")

    event_hit_refs = _as_array(hits_ref[hits_ref[:, 0] == event_id, 1], np.int64)
    if event_hit_refs.size == 0:
        raise RuntimeError(f"No calibrated prompt hits found for event {event_id}")

    if hits_full is None:
        hits_full = h5["charge/calib_prompt_hits/data"]
    hits_evt = _read_rows_by_index(hits_full, event_hit_refs)
    hit_energy = _as_array(hits_evt["E"], np.float32) if "E" in hits_evt.dtype.names else np.full(event_hit_refs.size, np.nan, dtype=np.float32)

    truth = extract_truth_for_selected_hits(
        h5,
        event_hit_refs,
        convert_t0_to_matching_ticks=bool(convert_to_matching_ticks),
        event_id=int(event_id),
        strict=strict,
    )

    return {
        "hit_refs": event_hit_refs,
        "hit_energy": hit_energy,
        "truth_t0": _as_array(truth["true_t0_rel"], np.float64),
        "truth_t0_raw": _as_array(truth["true_t0"], np.float64),
        "best_segment_id": _as_array(truth["best_segment_id"], np.int64),
        "best_fraction": _as_array(truth["best_fraction"], np.float32),
        "vertex_id": _as_array(truth["vertex_id"], np.int64),
        "raw_t0_min": truth["raw_t0_min"],
        "raw_t0_reference": truth["raw_t0_reference"],
        "truth_t0_reference_ns": truth["truth_t0_reference_ns"],
        "truth_t0_reference_source": truth["truth_t0_reference_source"],
        "matching_tick_scale": truth["matching_tick_scale"],
        "matching_tick_offset": truth["matching_tick_offset"],
        "truth_t0_convention": truth["truth_t0_convention"],
        "truth_status": truth["truth_status"],
        "truth_info": truth,
    }


def _weighted_quantile_from_sorted(sorted_x: np.ndarray, cum_w: np.ndarray, q: float) -> float:
    idx = int(np.searchsorted(cum_w, float(q), side="left"))
    idx = min(idx, len(sorted_x) - 1)
    return float(sorted_x[idx])


def energy_weighted_t0_residual_summary(
    reco_t0: np.ndarray,
    truth_t0: np.ndarray,
    hit_energy: np.ndarray,
    *,
    use_ns: bool = True,
    ns_per_tick: float = 16.0,
    window_limit: float = 400.0,
) -> dict[str, Any]:
    """
    Compute event-level energy-weighted residual statistics.

    Residual convention is `truth_t0 - reco_t0`.  If `use_ns=True`, residuals
    and summary values are converted from ticks to ns.
    """
    reco = _as_array(reco_t0, np.float64)
    truth = _as_array(truth_t0, np.float64)
    energy = _as_array(hit_energy, np.float64)
    if not (len(reco) == len(truth) == len(energy)):
        raise ValueError(
            f"Length mismatch: reco={len(reco)}, truth={len(truth)}, energy={len(energy)}"
        )

    valid = np.isfinite(reco) & np.isfinite(truth) & np.isfinite(energy) & (energy > 0)
    diff_ticks = truth[valid] - reco[valid]
    weights = energy[valid]
    diff_plot = diff_ticks * float(ns_per_tick) if bool(use_ns) else diff_ticks

    if weights.size == 0 or float(np.sum(weights)) <= 0.0:
        return {
            "valid_mask": valid,
            "diff_ticks": diff_ticks,
            "diff_plot": diff_plot,
            "weights": weights,
            "total_energy": 0.0,
            "weighted_mean": np.nan,
            "weighted_median": np.nan,
            "weighted_std": np.nan,
            "weighted_p16": np.nan,
            "weighted_p84": np.nan,
            "energy_within_window": 0.0,
            "fraction_within_window": np.nan,
            "window_limit": float(window_limit),
            "units": "ns" if bool(use_ns) else "ticks",
            "n_valid_hits": 0,
        }

    w_sum = float(np.sum(weights))
    w_mean = float(np.sum(weights * diff_plot) / w_sum)
    order = np.argsort(diff_plot)
    x_sorted = diff_plot[order]
    w_sorted = weights[order]
    cw = np.cumsum(w_sorted) / w_sum
    w_median = _weighted_quantile_from_sorted(x_sorted, cw, 0.50)
    w_p16 = _weighted_quantile_from_sorted(x_sorted, cw, 0.16)
    w_p84 = _weighted_quantile_from_sorted(x_sorted, cw, 0.84)
    w_std = float(np.sqrt(np.sum(weights * (diff_plot - w_mean) ** 2) / w_sum))
    energy_within = float(np.sum(weights[np.abs(diff_plot) <= float(window_limit)]))
    frac_within = 100.0 * energy_within / max(w_sum, 1e-12)

    return {
        "valid_mask": valid,
        "diff_ticks": diff_ticks,
        "diff_plot": diff_plot,
        "weights": weights,
        "total_energy": w_sum,
        "weighted_mean": w_mean,
        "weighted_median": w_median,
        "weighted_std": w_std,
        "weighted_p16": w_p16,
        "weighted_p84": w_p84,
        "energy_within_window": energy_within,
        "fraction_within_window": float(frac_within),
        "window_limit": float(window_limit),
        "units": "ns" if bool(use_ns) else "ticks",
        "n_valid_hits": int(np.count_nonzero(valid)),
    }


def plot_energy_weighted_t0_residual_histogram(
    reco_t0: np.ndarray,
    truth_t0: np.ndarray,
    hit_energy: np.ndarray,
    *,
    event_id: int | None = None,
    use_ns: bool = True,
    ns_per_tick: float = 16.0,
    bins: np.ndarray | None = None,
    x_range: tuple[float, float] | None = None,
    window_limit: float = 400.0,
    figsize: tuple[float, float] = (10.5, 5.8),
    dpi: int = 160,
    bar_color: str = "#4C78A8",
    title: str | None = None,
    save_path: str | None = "FinallyThere.png",
    show: bool = True,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Plot the event-level energy-weighted `truth_t0 - reco_t0` histogram.

    The bar heights are normalized to total valid hit energy, so the y-axis is
    the proportion of total reconstructed event energy.
    """
    import matplotlib.pyplot as plt

    if bins is None:
        bins = np.arange(-500, 501, 10) if bool(use_ns) else np.arange(-30, 31, 1)
    if x_range is None:
        x_range = (-500, 500) if bool(use_ns) else (-30, 30)

    summary = energy_weighted_t0_residual_summary(
        reco_t0,
        truth_t0,
        hit_energy,
        use_ns=use_ns,
        ns_per_tick=ns_per_tick,
        window_limit=window_limit,
    )
    diff_plot = _as_array(summary["diff_plot"], np.float64)
    weights = _as_array(summary["weights"], np.float64)
    w_sum = float(summary["total_energy"])
    units = str(summary["units"])
    xlabel = f"Truth t0 - Reco t0 [{units}]"

    if title is None:
        suffix = "" if event_id is None else f" | Event {int(event_id)}"
        title = f"Energy-Weighted Truth t0 - Reco t0{suffix}"

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    if weights.size > 0 and w_sum > 0.0:
        weights_norm = weights / w_sum
        hist_counts, bin_edges = np.histogram(diff_plot, bins=bins, weights=weights_norm)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = bin_edges[1] - bin_edges[0]
        ax.bar(
            bin_centers,
            hist_counts,
            width=bar_width,
            color=bar_color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
        )
    else:
        hist_counts = np.zeros(len(bins) - 1, dtype=np.float64)
        bin_edges = np.asarray(bins, dtype=np.float64)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.4)
    ax.set_xlim(x_range)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Proportion of total energy", fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_text = (
        f"Total energy: {summary['total_energy']:.1f} MeV\n"
        f"Weighted mean:   {summary['weighted_mean']:.2f}\n"
        f"Weighted median: {summary['weighted_median']:.2f}\n"
        f"Weighted std:    {summary['weighted_std']:.2f}\n"
        f"Weighted 16-84%: [{summary['weighted_p16']:.2f}, {summary['weighted_p84']:.2f}]\n"
        f"Energy within ±{summary['window_limit']:.0f} {units}: {summary['fraction_within_window']:.1f}%"
    )
    ax.text(
        0.985,
        0.965,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.92),
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

    if print_summary:
        print(f"Finite weighted hits : {summary['n_valid_hits']}")
        print(f"Total energy         : {summary['total_energy']:.4f} MeV")
        print(f"Weighted mean        : {summary['weighted_mean']:.4f}")
        print(f"Weighted median      : {summary['weighted_median']:.4f}")
        print(f"Weighted std         : {summary['weighted_std']:.4f}")
        print(f"Weighted 16-84%      : [{summary['weighted_p16']:.4f}, {summary['weighted_p84']:.4f}]")
        print(
            f"Energy within ±{summary['window_limit']:.0f} {units} : "
            f"{summary['fraction_within_window']:.1f}%"
        )
        print(
            f"Energy within +/- {summary['window_limit']:.0f} {units} : "
            f"{summary['energy_within_window']:.4f} MeV"
        )
        print(f"Total weighted energy        : {summary['total_energy']:.4f} MeV")
        print(
            f"Fraction within +/- {summary['window_limit']:.0f} {units} : "
            f"{summary['fraction_within_window']:.2f}%"
        )

    return {
        **summary,
        "hist_counts": hist_counts,
        "bin_edges": bin_edges,
        "fig": fig,
        "ax": ax,
        "save_path": save_path,
    }


def plot_event_energy_weighted_t0_residual_from_hdf5(
    h5: Any,
    hits_ref: np.ndarray,
    event_id: int | None = None,
    *,
    eventid: int | None = None,
    hits_full: Any | None = None,
    reco_t0: np.ndarray,
    convert_to_matching_ticks: bool = True,
    use_ns: bool = True,
    ns_per_tick: float = 16.0,
    bins: np.ndarray | None = None,
    x_range: tuple[float, float] | None = None,
    window_limit: float = 400.0,
    save_path: str | None = "FinallyThere.png",
    show: bool = True,
    print_summary: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Extract event truth and plot the energy-weighted t0 residual histogram.

    This is the event-level demo helper for checking the full current matching
    result after all phases.
    """
    if event_id is None:
        if eventid is None:
            raise TypeError("Pass event_id or eventid.")
        event_id = int(eventid)
    event_truth = extract_event_hit_energy_and_truth_t0(
        h5,
        hits_ref,
        event_id=int(event_id),
        hits_full=hits_full,
        convert_to_matching_ticks=convert_to_matching_ticks,
        strict=strict,
    )
    result = plot_energy_weighted_t0_residual_histogram(
        reco_t0=reco_t0,
        truth_t0=event_truth["truth_t0"],
        hit_energy=event_truth["hit_energy"],
        event_id=int(event_id),
        use_ns=use_ns,
        ns_per_tick=ns_per_tick,
        bins=bins,
        x_range=x_range,
        window_limit=window_limit,
        save_path=save_path,
        show=show,
        print_summary=print_summary,
    )
    result["event_truth"] = event_truth
    return result


def plot_wrong_hits_3d(
    TPCid: int,
    truth_t0: np.ndarray,
    *,
    hit_timestamps: np.ndarray,
    hitTPCid: np.ndarray,
    xset: np.ndarray,
    yset: np.ndarray,
    zset: np.ndarray,
    Eset: np.ndarray,
    residual_cut_ticks: float = 25.0,
    save_path: str | None = None,
    marker_size: float = 3.5,
    wrong_color: str = "red",
    correct_color: str = "lightgray",
    show: bool = True,
) -> dict[str, Any]:
    """
    Plot hits in one TPC:
      - red  : |truth_t0 - reco_t0| > residual_cut_ticks
      - grey : otherwise

    All arrays must be aligned to the current event hit order.
    """
    TPCid = int(TPCid)
    truth_t0 = _as_array(truth_t0, np.float64)
    reco_t0 = _as_array(hit_timestamps, np.float64)

    if len(truth_t0) != len(reco_t0):
        raise ValueError("truth_t0 must have the same length as hit_timestamps")

    hitTPCid = _as_array(hitTPCid, np.int64)
    xset = _as_array(xset, np.float64)
    yset = _as_array(yset, np.float64)
    zset = _as_array(zset, np.float64)
    Eset = _as_array(Eset, np.float64)

    tpc_mask = hitTPCid == TPCid

    x_tpc = xset[tpc_mask]
    y_tpc = yset[tpc_mask]
    z_tpc = zset[tpc_mask]
    E_tpc = Eset[tpc_mask]
    truth_t0_tpc = truth_t0[tpc_mask]
    reco_t0_tpc = reco_t0[tpc_mask]

    valid = np.isfinite(truth_t0_tpc) & np.isfinite(reco_t0_tpc)
    residual = truth_t0_tpc - reco_t0_tpc

    wrong_mask = valid & (np.abs(residual) > float(residual_cut_ticks))
    correct_mask = valid & (~wrong_mask)
    invalid_mask = ~valid

    energy_total_valid = float(np.sum(E_tpc[valid]))
    energy_wrong = float(np.sum(E_tpc[wrong_mask]))
    energy_correct = float(np.sum(E_tpc[correct_mask]))

    frac_correct = 100.0 * energy_correct / max(energy_total_valid, 1e-12)
    frac_wrong = 100.0 * energy_wrong / max(energy_total_valid, 1e-12)

    fig = go.Figure()

    if np.any(correct_mask):
        fig.add_trace(
            go.Scatter3d(
                x=z_tpc[correct_mask],
                y=y_tpc[correct_mask],
                z=x_tpc[correct_mask],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=correct_color,
                    opacity=0.55,
                    line=dict(width=0),
                ),
                text=[
                    f"E = {e:.2f} MeV<br>truth = {tt:.2f}<br>reco = {rt:.2f}<br>residual = {rr:.2f} ticks"
                    for e, tt, rt, rr in zip(
                        E_tpc[correct_mask],
                        truth_t0_tpc[correct_mask],
                        reco_t0_tpc[correct_mask],
                        residual[correct_mask],
                    )
                ],
                hoverinfo="text+x+y+z",
                name=f"Correct ({np.count_nonzero(correct_mask)} hits, {energy_correct:.1f} MeV)",
            )
        )

    if np.any(wrong_mask):
        fig.add_trace(
            go.Scatter3d(
                x=z_tpc[wrong_mask],
                y=y_tpc[wrong_mask],
                z=x_tpc[wrong_mask],
                mode="markers",
                marker=dict(
                    size=marker_size + 0.5,
                    color=wrong_color,
                    opacity=0.9,
                    line=dict(width=0),
                ),
                text=[
                    f"E = {e:.2f} MeV<br>truth = {tt:.2f}<br>reco = {rt:.2f}<br>residual = {rr:.2f} ticks"
                    for e, tt, rt, rr in zip(
                        E_tpc[wrong_mask],
                        truth_t0_tpc[wrong_mask],
                        reco_t0_tpc[wrong_mask],
                        residual[wrong_mask],
                    )
                ],
                hoverinfo="text+x+y+z",
                name=f"Wrong ({np.count_nonzero(wrong_mask)} hits, {energy_wrong:.1f} MeV)",
            )
        )

    if np.any(invalid_mask):
        fig.add_trace(
            go.Scatter3d(
                x=z_tpc[invalid_mask],
                y=y_tpc[invalid_mask],
                z=x_tpc[invalid_mask],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="darkgray",
                    opacity=0.25,
                    line=dict(width=0),
                ),
                text=[
                    f"E = {e:.2f} MeV<br>truth/reco unavailable"
                    for e in E_tpc[invalid_mask]
                ],
                hoverinfo="text+x+y+z",
                name=f"Invalid truth/reco ({np.count_nonzero(invalid_mask)} hits)",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="z",
            yaxis_title="y",
            zaxis_title="x",
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        title=f"TPC {TPCid} | wrong hits in red | cut = {residual_cut_ticks:.1f} ticks",
        showlegend=True,
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved html: {save_path}")

    if show:
        fig.show()

    print(f"TPC {TPCid}")
    print(f"Residual cut                 : {residual_cut_ticks:.2f} ticks ({residual_cut_ticks * 16:.0f} ns)")
    print(f"Finite truth+reco hits       : {int(np.count_nonzero(valid))}")
    print(f"Wrong hits                   : {int(np.count_nonzero(wrong_mask))}")
    print(f"Correct energy               : {energy_correct:.4f} MeV")
    print(f"Wrong energy                 : {energy_wrong:.4f} MeV")
    print(f"Correct energy fraction      : {frac_correct:.2f}%")
    print(f"Wrong energy fraction        : {frac_wrong:.2f}%")

    return {
        "TPCid": TPCid,
        "residual_cut_ticks": float(residual_cut_ticks),
        "n_valid_hits": int(np.count_nonzero(valid)),
        "n_wrong_hits": int(np.count_nonzero(wrong_mask)),
        "correct_energy": float(energy_correct),
        "wrong_energy": float(energy_wrong),
        "correct_energy_fraction": float(frac_correct),
        "wrong_energy_fraction": float(frac_wrong),
        "wrong_mask_tpc": wrong_mask,
        "correct_mask_tpc": correct_mask,
        "invalid_mask_tpc": invalid_mask,
        "residual_tpc": residual,
        "fig": fig,
    }


def plot_wrong_hits_from_namespace(
    namespace: dict[str, Any],
    TPCid: int,
    truth_t0: np.ndarray,
    *,
    residual_cut_ticks: float = 25.0,
    save_path: str | None = None,
    marker_size: float = 3.5,
    wrong_color: str = "red",
    correct_color: str = "lightgray",
    show: bool = True,
) -> dict[str, Any]:
    """
    Notebook convenience wrapper around `plot_wrong_hits_3d`.
    """
    return plot_wrong_hits_3d(
        TPCid,
        truth_t0,
        hit_timestamps=namespace["hit_timestamps"],
        hitTPCid=namespace["hitTPCid"],
        xset=namespace["xset"],
        yset=namespace["yset"],
        zset=namespace["zset"],
        Eset=namespace["Eset"],
        residual_cut_ticks=residual_cut_ticks,
        save_path=save_path,
        marker_size=marker_size,
        wrong_color=wrong_color,
        correct_color=correct_color,
        show=show,
    )


__all__ = [
    "DEFAULT_MATCHING_TICK_SCALE",
    "DEFAULT_MATCHING_TICK_OFFSET",
    "MATCHING_T0_CONVENTION",
    "extract_truth_for_selected_hits",
    "subset_truth_info",
    "convert_truth_t0_rel_to_matching_ticks",
    "build_truth_group_labels",
    "plot_truth_hits_like_matching",
    "plot_truth_for_selection",
    "plot_truth_from_namespace",
    "extract_event_hit_energy_and_truth_t0",
    "energy_weighted_t0_residual_summary",
    "plot_energy_weighted_t0_residual_histogram",
    "plot_event_energy_weighted_t0_residual_from_hdf5",
    "plot_wrong_hits_3d",
    "plot_wrong_hits_from_namespace",
]
