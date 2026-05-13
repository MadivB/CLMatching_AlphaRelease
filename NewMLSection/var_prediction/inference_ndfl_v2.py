from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .inference_ndfl import (
        DEFAULT_INPUT_SCALE,
        DEFAULT_TARGET_SCALE,
        NUM_CHANNELS,
        ConformerVarPredictor,
        load_model,
        predict,
        resolve_checkpoint,
    )
except ImportError:  # pragma: no cover - direct script/notebook fallback
    from inference_ndfl import (
        DEFAULT_INPUT_SCALE,
        DEFAULT_TARGET_SCALE,
        NUM_CHANNELS,
        ConformerVarPredictor,
        load_model,
        predict,
        resolve_checkpoint,
    )


def _channel_label(channel_index: int) -> str:
    ch = int(channel_index)
    return f"L{ch}" if ch < 60 else f"R{ch - 60}"


def _continuous_groups(local_indices: list[int], *, max_gap: int) -> list[tuple[int, int]]:
    if len(local_indices) == 0:
        return []

    ordered = sorted(int(idx) for idx in local_indices)
    groups: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]

    for idx in ordered[1:]:
        if int(idx) - int(prev) <= int(max_gap) + 1:
            prev = int(idx)
            continue
        groups.append((int(start), int(prev)))
        start = int(idx)
        prev = int(idx)
    groups.append((int(start), int(prev)))
    return groups


def select_continuous_side_channels_from_waveform(
    waveform: np.ndarray,
    *,
    light_fraction: float = 0.90,
    max_gap: int = 2,
) -> dict[str, Any]:
    """
    Build the fast single-TPC support-channel selection used by v7.

    Steps
    -----
    1. Keep the smallest set of channels carrying `light_fraction` of the total
       predicted light.
    2. Split those channels into continuous groups separately on left/right,
       allowing up to `max_gap` missing channels.
    3. Keep only the strongest group on the left and the strongest group on the
       right. Gaps inside the chosen span are filled so dead channels remain in
       the fit window.
    """
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != int(NUM_CHANNELS):
        raise ValueError(f"Expected waveform shape ({int(NUM_CHANNELS)}, T), got {wave.shape}")

    frac = float(light_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("light_fraction must be in (0, 1].")

    channel_sums = np.sum(np.clip(wave, 0.0, None), axis=1)
    total_light = float(np.sum(channel_sums))
    if total_light <= 0.0:
        empty_idx = np.array([], dtype=np.int32)
        return {
            "channel_indices": empty_idx,
            "channel_labels": [],
            "channel_selection_spec": [],
            "dominant_fraction": 0.0,
            "selected_fraction": 0.0,
            "total_light_per_channel": channel_sums,
            "left_groups": [],
            "right_groups": [],
            "chosen_left_group": None,
            "chosen_right_group": None,
            "selected_waveform": wave[empty_idx],
        }

    order = np.argsort(channel_sums)[::-1]
    sorted_sums = channel_sums[order]
    cutoff = frac * total_light
    cumulative = np.cumsum(sorted_sums)
    n_selected = int(np.searchsorted(cumulative, cutoff, side="left") + 1)
    dominant_idx = sorted(int(idx) for idx in order[:n_selected].tolist())

    def _build_groups(side: str) -> list[dict[str, Any]]:
        if side == "L":
            local = [int(ch) for ch in dominant_idx if int(ch) < 60]
            offset = 0
        else:
            local = [int(ch) - 60 for ch in dominant_idx if int(ch) >= 60]
            offset = 60

        groups: list[dict[str, Any]] = []
        for lo, hi in _continuous_groups(local, max_gap=int(max_gap)):
            full_local = list(range(int(lo), int(hi) + 1))
            full_global = [int(offset + idx) for idx in full_local]
            group_light = float(np.sum(channel_sums[full_global]))
            groups.append(
                {
                    "side": str(side),
                    "lo": int(lo),
                    "hi": int(hi),
                    "n_channels": int(len(full_local)),
                    "global_indices": full_global,
                    "labels": [_channel_label(ch) for ch in full_global],
                    "group_light": float(group_light),
                    "group_fraction": float(group_light / total_light),
                }
            )
        return groups

    left_groups = _build_groups("L")
    right_groups = _build_groups("R")
    chosen_left = max(left_groups, key=lambda item: float(item["group_light"])) if left_groups else None
    chosen_right = max(right_groups, key=lambda item: float(item["group_light"])) if right_groups else None

    final_idx: list[int] = []
    if chosen_left is not None:
        final_idx.extend(int(ch) for ch in chosen_left["global_indices"])
    if chosen_right is not None:
        final_idx.extend(int(ch) for ch in chosen_right["global_indices"])
    final_idx = sorted(set(final_idx))

    if len(final_idx) == 0 and len(dominant_idx) > 0:
        final_idx = [int(dominant_idx[0])]

    final_idx_arr = np.asarray(final_idx, dtype=np.int32)
    final_fraction = float(np.sum(channel_sums[final_idx_arr]) / total_light) if final_idx_arr.size else 0.0
    dominant_fraction = float(np.sum(channel_sums[dominant_idx]) / total_light) if dominant_idx else 0.0

    selection_spec: list[tuple[str, int, int]] = []
    if chosen_left is not None:
        selection_spec.append(("L", int(chosen_left["lo"]), int(chosen_left["hi"])))
    if chosen_right is not None:
        selection_spec.append(("R", int(chosen_right["lo"]), int(chosen_right["hi"])))

    return {
        "channel_indices": final_idx_arr,
        "channel_labels": [_channel_label(ch) for ch in final_idx_arr.tolist()],
        "channel_selection_spec": selection_spec,
        "dominant_fraction": float(dominant_fraction),
        "selected_fraction": float(final_fraction),
        "total_light_per_channel": channel_sums,
        "left_groups": left_groups,
        "right_groups": right_groups,
        "chosen_left_group": chosen_left,
        "chosen_right_group": chosen_right,
        "selected_waveform": np.asarray(wave[final_idx_arr], dtype=np.float32),
    }


def build_cluster_channel_support_cache(
    image_maps: dict[tuple[int, int], np.ndarray],
    cluster_to_tpcs: dict[int, list[int]],
    *,
    light_fraction: float = 0.90,
    max_gap: int = 2,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    """
    Precompute per-(cluster, TPC) channel support for single-TPC clusters only.
    """
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    n_selected_channels: list[int] = []
    selected_fractions: list[float] = []

    for (clusterid, tpcid), waveform in image_maps.items():
        touched_tpcs = {
            int(tpc)
            for tpc in cluster_to_tpcs.get(int(clusterid), [])
            if (int(clusterid), int(tpc)) in image_maps
        }
        if len(touched_tpcs) != 1:
            continue

        info = select_continuous_side_channels_from_waveform(
            np.asarray(waveform, dtype=np.float32),
            light_fraction=float(light_fraction),
            max_gap=int(max_gap),
        )
        entry = {
            "clusterid": int(clusterid),
            "tpcid": int(tpcid),
            "channel_indices": np.asarray(info["channel_indices"], dtype=np.int32),
            "channel_labels": list(info["channel_labels"]),
            "channel_selection_spec": list(info["channel_selection_spec"]),
            "dominant_fraction": float(info["dominant_fraction"]),
            "selected_fraction": float(info["selected_fraction"]),
            "selected_waveform": np.asarray(info["selected_waveform"], dtype=np.float32),
            "left_groups": info["left_groups"],
            "right_groups": info["right_groups"],
            "chosen_left_group": info["chosen_left_group"],
            "chosen_right_group": info["chosen_right_group"],
        }
        cache[(int(clusterid), int(tpcid))] = entry
        n_selected_channels.append(int(entry["channel_indices"].size))
        selected_fractions.append(float(entry["selected_fraction"]))

    summary = {
        "n_entries": int(len(cache)),
        "light_fraction": float(light_fraction),
        "max_gap": int(max_gap),
        "mean_selected_channels": float(np.mean(n_selected_channels)) if n_selected_channels else 0.0,
        "median_selected_channels": float(np.median(n_selected_channels)) if n_selected_channels else 0.0,
        "mean_selected_fraction": float(np.mean(selected_fractions)) if selected_fractions else 0.0,
    }
    return cache, summary


__all__ = [
    "NUM_CHANNELS",
    "DEFAULT_INPUT_SCALE",
    "DEFAULT_TARGET_SCALE",
    "ConformerVarPredictor",
    "resolve_checkpoint",
    "load_model",
    "predict",
    "select_continuous_side_channels_from_waveform",
    "build_cluster_channel_support_cache",
]
