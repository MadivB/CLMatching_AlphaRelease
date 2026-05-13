from __future__ import annotations

from typing import Any

import numpy as np


def _channel_label(channel_index: int) -> str:
    channel_index = int(channel_index)
    return f"L{channel_index}" if channel_index < 60 else f"R{channel_index - 60}"


def build_saturated_channel_cache_v12(
    full_light_waveform: np.ndarray,
    *,
    clip_threshold: float = 60700.0,
    max_clip_ticks: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wave = np.asarray(full_light_waveform, dtype=np.float32)
    if wave.ndim != 3:
        raise ValueError(
            "full_light_waveform must have shape (n_tpc, n_channels, n_ticks); "
            f"got {wave.shape!r}"
        )

    clip_counts = np.sum(wave > float(clip_threshold), axis=-1, dtype=np.int32)
    veto_mask = np.asarray(clip_counts > int(max_clip_ticks), dtype=bool)
    allowed_channel_indices_by_tpc = [
        np.flatnonzero(~veto_mask[int(tpcid)]).astype(np.int32)
        for tpcid in range(wave.shape[0])
    ]

    per_tpc_veto_counts = np.sum(veto_mask, axis=1, dtype=np.int32)
    summary = {
        "clip_threshold": float(clip_threshold),
        "max_clip_ticks": int(max_clip_ticks),
        "n_tpcs": int(wave.shape[0]),
        "n_channels_per_tpc": int(wave.shape[1]),
        "n_vetoed_channels_total": int(np.sum(veto_mask)),
        "n_tpcs_with_vetoed_channels": int(np.count_nonzero(per_tpc_veto_counts)),
        "mean_vetoed_channels_per_tpc": float(np.mean(per_tpc_veto_counts)) if per_tpc_veto_counts.size else 0.0,
        "max_vetoed_channels_per_tpc": int(np.max(per_tpc_veto_counts)) if per_tpc_veto_counts.size else 0,
        "per_tpc_veto_counts": per_tpc_veto_counts.astype(np.int32),
        "vetoed_channel_labels_by_tpc": {
            int(tpcid): [_channel_label(ch) for ch in np.flatnonzero(veto_mask[int(tpcid)]).tolist()]
            for tpcid in range(wave.shape[0])
            if bool(np.any(veto_mask[int(tpcid)]))
        },
    }
    cache = {
        "clip_threshold": float(clip_threshold),
        "max_clip_ticks": int(max_clip_ticks),
        "clip_counts": clip_counts.astype(np.int32),
        "veto_mask": veto_mask,
        "allowed_channel_indices_by_tpc": allowed_channel_indices_by_tpc,
    }
    return cache, summary


def filter_channel_indices_v12(
    channel_indices: np.ndarray,
    *,
    tpcid: int,
    full_waveform: np.ndarray,
    saturated_channel_cache: dict[str, Any] | None,
    fallback_to_all_unsaturated: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    full_wave = np.asarray(full_waveform, dtype=np.float32)
    n_channels = int(full_wave.shape[0])
    idx = np.asarray(channel_indices, dtype=np.int32).reshape(-1)
    if idx.size == 0:
        idx = np.arange(n_channels, dtype=np.int32)

    if saturated_channel_cache is None:
        return idx.astype(np.int32), np.asarray([], dtype=np.int32), False

    veto_mask_all = np.asarray(saturated_channel_cache["veto_mask"][int(tpcid)], dtype=bool)
    keep_mask = ~veto_mask_all[np.asarray(idx, dtype=np.int32)]
    filtered = np.asarray(idx[keep_mask], dtype=np.int32)
    dropped = np.asarray(idx[~keep_mask], dtype=np.int32)
    fallback_used = False

    if filtered.size == 0 and bool(fallback_to_all_unsaturated):
        filtered = np.asarray(
            saturated_channel_cache["allowed_channel_indices_by_tpc"][int(tpcid)],
            dtype=np.int32,
        )
        dropped = np.asarray(
            sorted(set(int(v) for v in idx.tolist()) | set(int(v) for v in np.flatnonzero(veto_mask_all).tolist())),
            dtype=np.int32,
        )
        fallback_used = True

    return filtered, dropped, fallback_used


def materialize_support_entry_v12(
    *,
    waveform: np.ndarray,
    tpcid: int,
    base_entry: dict[str, Any] | None,
    saturated_channel_cache: dict[str, Any] | None,
    fallback_to_all_unsaturated: bool = True,
) -> dict[str, Any]:
    wave = np.asarray(waveform, dtype=np.float32)
    n_channels = int(wave.shape[0])
    entry = {} if base_entry is None else dict(base_entry)

    preclip_indices = np.asarray(entry.get("channel_indices", np.arange(n_channels)), dtype=np.int32)
    if preclip_indices.size == 0:
        preclip_indices = np.arange(n_channels, dtype=np.int32)

    final_indices, dropped_indices, fallback_used = filter_channel_indices_v12(
        preclip_indices,
        tpcid=int(tpcid),
        full_waveform=wave,
        saturated_channel_cache=saturated_channel_cache,
        fallback_to_all_unsaturated=bool(fallback_to_all_unsaturated),
    )

    if final_indices.size == 0:
        final_indices = np.arange(n_channels, dtype=np.int32)
        dropped_indices = np.asarray([], dtype=np.int32)
        fallback_used = True

    channel_labels = entry.get("channel_labels")
    if channel_labels is None or len(channel_labels) != preclip_indices.size:
        channel_labels = [_channel_label(ch) for ch in preclip_indices.tolist()]

    final_label_map = {int(ch): _channel_label(ch) for ch in final_indices.tolist()}
    dropped_label_map = {int(ch): _channel_label(ch) for ch in dropped_indices.tolist()}

    selected_waveform = np.asarray(wave[final_indices], dtype=np.float32)
    out = dict(entry)
    out.update(
        {
            "channel_indices_preclip": np.asarray(preclip_indices, dtype=np.int32),
            "channel_indices": np.asarray(final_indices, dtype=np.int32),
            "channel_labels_preclip": list(channel_labels),
            "channel_labels": [final_label_map[int(ch)] for ch in final_indices.tolist()],
            "selected_waveform": selected_waveform,
            "saturation_veto_applied": bool(saturated_channel_cache is not None),
            "n_channels_preclip": int(preclip_indices.size),
            "n_saturated_channels_dropped": int(dropped_indices.size),
            "dropped_channel_indices": np.asarray(dropped_indices, dtype=np.int32),
            "dropped_channel_labels": [dropped_label_map[int(ch)] for ch in dropped_indices.tolist()],
            "saturation_fallback_to_all_unsaturated": bool(fallback_used),
        }
    )
    return out


def apply_saturation_veto_to_support_cache_v12(
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    *,
    image_maps: dict[tuple[int, int], np.ndarray],
    saturated_channel_cache: dict[str, Any] | None,
) -> tuple[dict[tuple[int, int], dict[str, Any]] | None, dict[str, Any]]:
    if channel_support_cache is None:
        return None, {
            "n_entries": 0,
            "n_entries_with_dropped_channels": 0,
            "n_dropped_channels_total": 0,
            "mean_fit_channels_after_veto": 0.0,
            "n_entries_with_fallback": 0,
        }

    updated_cache: dict[tuple[int, int], dict[str, Any]] = {}
    n_entries_with_dropped = 0
    n_entries_with_fallback = 0
    dropped_channels_total = 0
    fit_channels_after: list[int] = []

    for key, entry in channel_support_cache.items():
        clusterid, tpcid = int(key[0]), int(key[1])
        waveform = np.asarray(image_maps[(clusterid, tpcid)], dtype=np.float32)
        new_entry = materialize_support_entry_v12(
            waveform=waveform,
            tpcid=int(tpcid),
            base_entry=entry,
            saturated_channel_cache=saturated_channel_cache,
        )
        updated_cache[(clusterid, tpcid)] = new_entry
        fit_channels_after.append(int(new_entry["channel_indices"].size))
        dropped_here = int(new_entry.get("n_saturated_channels_dropped", 0))
        dropped_channels_total += dropped_here
        if dropped_here > 0:
            n_entries_with_dropped += 1
        if bool(new_entry.get("saturation_fallback_to_all_unsaturated", False)):
            n_entries_with_fallback += 1

    summary = {
        "n_entries": int(len(updated_cache)),
        "n_entries_with_dropped_channels": int(n_entries_with_dropped),
        "n_dropped_channels_total": int(dropped_channels_total),
        "mean_fit_channels_after_veto": float(np.mean(fit_channels_after)) if fit_channels_after else 0.0,
        "n_entries_with_fallback": int(n_entries_with_fallback),
    }
    return updated_cache, summary


__all__ = [
    "apply_saturation_veto_to_support_cache_v12",
    "build_saturated_channel_cache_v12",
    "filter_channel_indices_v12",
    "materialize_support_entry_v12",
]
