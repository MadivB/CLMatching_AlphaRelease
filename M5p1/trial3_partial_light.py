from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import numpy as np


def _channel_label(channel_index: int, *, side_channels: int = 60) -> str:
    channel_index = int(channel_index)
    if channel_index < int(side_channels):
        return f"L{channel_index}"
    return f"R{channel_index - int(side_channels)}"


def _side_expanded_indices(
    selected: np.ndarray,
    *,
    side_offset: int,
    side_channels: int,
    group_size: int,
    expand_groups: int,
    side: str,
) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int32)
    selected = selected[(selected >= int(side_offset)) & (selected < int(side_offset) + int(side_channels))]
    if selected.size == 0:
        return np.asarray([], dtype=np.int32)

    local = selected - int(side_offset)
    n_groups = int(np.ceil(float(side_channels) / float(group_size)))
    group_min = int(np.min(local) // int(group_size))
    group_max = int(np.max(local) // int(group_size))

    # L and R are physical detector sides.  Extend one hardware group outward:
    # L extends toward L0; R extends toward larger R-channel indices.
    if str(side).upper() == "L":
        out_min = max(0, group_min - int(expand_groups))
        out_max = group_max
    elif str(side).upper() == "R":
        out_min = group_min
        out_max = min(n_groups - 1, group_max + int(expand_groups))
    else:
        raise ValueError(f"Unknown side {side!r}; expected 'L' or 'R'.")

    lo = int(side_offset) + out_min * int(group_size)
    hi = min(int(side_offset) + int(side_channels), int(side_offset) + (out_max + 1) * int(group_size))
    return np.arange(lo, hi, dtype=np.int32)


def expanded_addback_channel_indices(
    selected_channels: np.ndarray,
    *,
    n_channels: int = 120,
    side_channels: int = 60,
    group_size: int = 6,
    expand_groups: int = 1,
) -> np.ndarray:
    """Return the Trial3 addback channels for a 90%-continuity support mask.

    The fit/loss mask remains the existing 90% continuous support.  The addback
    image is widened to whole 6-channel hardware groups, with one extra group
    outward on each physical detector side.  This matches the requested examples:
    R1-R10 -> R0-R17, and L7-L19 -> L0-L23.
    """
    selected = np.asarray(selected_channels, dtype=np.int32).reshape(-1)
    selected = selected[(selected >= 0) & (selected < int(n_channels))]
    if selected.size == 0:
        return np.asarray([], dtype=np.int32)

    left = _side_expanded_indices(
        selected,
        side_offset=0,
        side_channels=int(side_channels),
        group_size=int(group_size),
        expand_groups=int(expand_groups),
        side="L",
    )
    right = _side_expanded_indices(
        selected,
        side_offset=int(side_channels),
        side_channels=int(side_channels),
        group_size=int(group_size),
        expand_groups=int(expand_groups),
        side="R",
    )
    return np.unique(np.concatenate([left, right]).astype(np.int32))


def _support_indices_for_addback(entry: dict[str, Any] | None, n_channels: int) -> np.ndarray:
    if not entry:
        return np.arange(int(n_channels), dtype=np.int32)

    for key in ("channel_indices_preclip", "channel_indices"):
        if key in entry:
            idx = np.asarray(entry[key], dtype=np.int32).reshape(-1)
            idx = idx[(idx >= 0) & (idx < int(n_channels))]
            if idx.size > 0:
                return idx

    return np.arange(int(n_channels), dtype=np.int32)


def build_trial3_partial_addback_image_maps(
    image_maps: dict[tuple[int, int], np.ndarray],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    *,
    group_size: int = 6,
    expand_groups: int = 1,
    side_channels: int = 60,
    keep_full_without_support: bool = True,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Mask full predicted light images to Trial3 partial addback channels.

    ``image_maps`` stay shape-compatible.  Only channels selected by the
    group-expanded addback mask keep their predicted amplitudes; all other
    channels are zeroed.  The existing support cache still controls fitting.
    """
    support = {} if channel_support_cache is None else channel_support_cache
    out: dict[tuple[int, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    n_entries_masked = 0
    n_entries_full = 0
    selected_counts: list[int] = []
    addback_counts: list[int] = []

    for key, waveform in image_maps.items():
        clusterid, tpcid = int(key[0]), int(key[1])
        wave = np.asarray(waveform, dtype=np.float32)
        if wave.ndim != 2:
            raise ValueError(f"image_maps[{key!r}] must be 2D, got {wave.shape!r}")

        n_channels = int(wave.shape[0])
        entry = support.get((clusterid, tpcid))
        if entry is None and bool(keep_full_without_support):
            out[(clusterid, tpcid)] = np.asarray(wave, dtype=np.float32).copy()
            n_entries_full += 1
            continue

        selected = _support_indices_for_addback(entry, n_channels)
        addback = expanded_addback_channel_indices(
            selected,
            n_channels=n_channels,
            side_channels=int(side_channels),
            group_size=int(group_size),
            expand_groups=int(expand_groups),
        )

        masked = np.zeros_like(wave, dtype=np.float32)
        if addback.size > 0:
            masked[addback] = wave[addback]
        out[(clusterid, tpcid)] = masked

        n_entries_masked += 1
        selected_counts.append(int(selected.size))
        addback_counts.append(int(addback.size))
        rows.append(
            {
                "clusterid": int(clusterid),
                "tpcid": int(tpcid),
                "n_fit_channels": int(selected.size),
                "n_addback_channels": int(addback.size),
                "fit_labels": [_channel_label(ch, side_channels=int(side_channels)) for ch in selected.tolist()],
                "addback_labels": [_channel_label(ch, side_channels=int(side_channels)) for ch in addback.tolist()],
                "support_mode": "" if entry is None else str(entry.get("support_mode", "")),
                "label_type": "" if entry is None else str(entry.get("label_type", "")),
            }
        )

    summary = {
        "mode": "trial3_partial_group_addback",
        "group_size": int(group_size),
        "expand_groups": int(expand_groups),
        "side_channels": int(side_channels),
        "keep_full_without_support": bool(keep_full_without_support),
        "n_image_maps": int(len(image_maps)),
        "n_entries_masked": int(n_entries_masked),
        "n_entries_kept_full_without_support": int(n_entries_full),
        "mean_fit_channels": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "mean_addback_channels": float(np.mean(addback_counts)) if addback_counts else 0.0,
        "median_fit_channels": float(np.median(selected_counts)) if selected_counts else 0.0,
        "median_addback_channels": float(np.median(addback_counts)) if addback_counts else 0.0,
        "rows": rows,
    }
    return out, summary


def install_trial3_prediction_patch(
    pipeline_module: Any,
    *,
    group_size: int = 6,
    expand_groups: int = 1,
    side_channels: int = 60,
    keep_full_without_support: bool = True,
) -> Any:
    """Patch a first-stage pipeline module to use Trial3 partial addback maps."""
    original = pipeline_module.predict_first_stage_images_and_std

    if getattr(original, "_trial3_partial_addback_original", None) is not None:
        original = original._trial3_partial_addback_original

    def wrapped_predict_first_stage_images_and_std(*args: Any, **kwargs: Any) -> Any:
        prediction = original(*args, **kwargs)
        partial_maps, summary = build_trial3_partial_addback_image_maps(
            prediction.image_maps,
            prediction.cluster_channel_support_cache,
            group_size=int(group_size),
            expand_groups=int(expand_groups),
            side_channels=int(side_channels),
            keep_full_without_support=bool(keep_full_without_support),
        )
        meta = copy.deepcopy(prediction.image_meta)
        meta["trial3_partial_addback_summary"] = summary
        support_summary = dict(prediction.cluster_channel_support_summary)
        support_summary["trial3_partial_addback"] = {
            key: value for key, value in summary.items() if key != "rows"
        }
        return replace(
            prediction,
            image_maps=partial_maps,
            image_meta=meta,
            cluster_channel_support_summary=support_summary,
        )

    wrapped_predict_first_stage_images_and_std._trial3_partial_addback_original = original
    pipeline_module.predict_first_stage_images_and_std = wrapped_predict_first_stage_images_and_std
    return original


__all__ = [
    "build_trial3_partial_addback_image_maps",
    "expanded_addback_channel_indices",
    "install_trial3_prediction_patch",
]
