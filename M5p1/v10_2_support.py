from __future__ import annotations

from typing import Any

import numpy as np

from var_prediction.inference_ndfl_v2 import select_continuous_side_channels_from_waveform
try:
    from v12_saturation_mask import materialize_support_entry_v12
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v12_saturation_mask import materialize_support_entry_v12


def build_cluster_channel_support_cache_v10_2(
    image_maps: dict[tuple[int, int], np.ndarray],
    cluster_to_tpcs: dict[int, list[int]],
    label_info: dict[int, dict[str, Any]],
    *,
    split_index: int,
    light_fraction: float = 0.90,
    max_gap: int = 2,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    """
    Build per-(cluster, TPC) channel-support masks for v10_2.

    Relative to v10_1:
    - keep the existing single-TPC behavior
    - keep the multi-TPC non-backbone behavior
    - additionally apply the same continuous light-fraction masking to multi-TPC
      track backbones
    - leave showers on the original all-channel behavior
    """
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    n_selected_channels: list[int] = []
    selected_fractions: list[float] = []
    single_tpc_entries = 0
    multi_tpc_nonbackbone_entries = 0
    multi_tpc_track_entries = 0
    skipped_multi_tpc_shower_entries = 0
    skipped_other_multi_tpc_backbone_entries = 0

    for (clusterid, tpcid), waveform in image_maps.items():
        touched_tpcs = {
            int(tpc)
            for tpc in cluster_to_tpcs.get(int(clusterid), [])
            if (int(clusterid), int(tpc)) in image_maps
        }
        n_touched_tpcs = int(len(touched_tpcs))
        if n_touched_tpcs == 0:
            continue

        label_type = str(label_info.get(int(clusterid), {}).get("type", "cluster")).lower()
        is_single_tpc = n_touched_tpcs == 1
        is_multi_tpc_nonbackbone = (n_touched_tpcs > 1) and (int(clusterid) >= int(split_index))
        is_multi_tpc_track = (n_touched_tpcs > 1) and (label_type == "track")

        if not (is_single_tpc or is_multi_tpc_nonbackbone or is_multi_tpc_track):
            if label_type == "shower":
                skipped_multi_tpc_shower_entries += 1
            else:
                skipped_other_multi_tpc_backbone_entries += 1
            continue

        info = select_continuous_side_channels_from_waveform(
            np.asarray(waveform, dtype=np.float32),
            light_fraction=float(light_fraction),
            max_gap=int(max_gap),
        )
        if is_single_tpc:
            support_mode = "single_tpc"
            single_tpc_entries += 1
        elif is_multi_tpc_nonbackbone:
            support_mode = "multi_tpc_nonbackbone"
            multi_tpc_nonbackbone_entries += 1
        else:
            support_mode = "multi_tpc_track"
            multi_tpc_track_entries += 1

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
            "support_mode": str(support_mode),
            "n_touched_tpcs": int(n_touched_tpcs),
            "label_type": str(label_type),
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
        "n_single_tpc_entries": int(single_tpc_entries),
        "n_multi_tpc_nonbackbone_entries": int(multi_tpc_nonbackbone_entries),
        "n_multi_tpc_track_entries": int(multi_tpc_track_entries),
        "skipped_multi_tpc_shower_entries": int(skipped_multi_tpc_shower_entries),
        "skipped_other_multi_tpc_backbone_entries": int(skipped_other_multi_tpc_backbone_entries),
    }
    return cache, summary


def get_cluster_tpc_fit_block_v10_2(
    *,
    clusterid: int,
    tpcid: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    use_support_mask: bool,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    full_wave = np.asarray(image_maps[(int(clusterid), int(tpcid))], dtype=np.float32)
    if not bool(use_support_mask):
        support_entry = materialize_support_entry_v12(
            waveform=full_wave,
            tpcid=int(tpcid),
            base_entry={
                "channel_indices": np.arange(full_wave.shape[0], dtype=np.int32),
                "channel_labels": [],
                "channel_selection_spec": [],
                "selected_fraction": 1.0,
                "dominant_fraction": 1.0,
                "support_mode": "all_channels",
            },
            saturated_channel_cache=saturated_channel_cache,
        )
    else:
        entry = {} if channel_support_cache is None else dict(channel_support_cache.get((int(clusterid), int(tpcid)), {}))
        support_entry = materialize_support_entry_v12(
            waveform=full_wave,
            tpcid=int(tpcid),
            base_entry=entry,
            saturated_channel_cache=saturated_channel_cache,
        )

    channel_indices = np.asarray(support_entry["channel_indices"], dtype=np.int32)
    support_meta = {
        "channel_indices": channel_indices,
        "channel_labels": list(support_entry.get("channel_labels", [])),
        "channel_selection_spec": list(support_entry.get("channel_selection_spec", [])),
        "selected_fraction": float(support_entry.get("selected_fraction", 1.0)),
        "dominant_fraction": float(support_entry.get("dominant_fraction", 1.0)),
        "support_mode": str(support_entry.get("support_mode", "masked_channels")),
        "n_channels_preclip": int(support_entry.get("n_channels_preclip", channel_indices.size)),
        "n_saturated_channels_dropped": int(support_entry.get("n_saturated_channels_dropped", 0)),
        "dropped_channel_labels": list(support_entry.get("dropped_channel_labels", [])),
        "saturation_fallback_to_all_unsaturated": bool(
            support_entry.get("saturation_fallback_to_all_unsaturated", False)
        ),
    }

    return {
        "clusterid": int(clusterid),
        "tpcid": int(tpcid),
        "full_wave": np.asarray(full_wave, dtype=np.float32),
        "selected_wave": np.asarray(support_entry["selected_waveform"], dtype=np.float32),
        "channel_indices": np.asarray(channel_indices, dtype=np.int32),
        "base_block": np.asarray(base_image[int(tpcid), channel_indices], dtype=np.float32),
        "actual_block": np.asarray(full_light_waveform[int(tpcid), channel_indices], dtype=np.float32),
        "std_block": np.asarray(full_light_std[int(tpcid), channel_indices], dtype=np.float32),
        "support_meta": dict(support_meta),
    }


def stack_cluster_fit_inputs_v10_2(
    *,
    clusterid: int,
    tpcids: np.ndarray,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    use_support_mask: bool,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    predicted_blocks: list[np.ndarray] = []
    base_blocks: list[np.ndarray] = []
    actual_blocks: list[np.ndarray] = []
    std_blocks: list[np.ndarray] = []
    block_meta: list[dict[str, Any]] = []

    for tpcid in np.asarray(tpcids, dtype=int):
        block = get_cluster_tpc_fit_block_v10_2(
            clusterid=int(clusterid),
            tpcid=int(tpcid),
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_support_cache=channel_support_cache,
            use_support_mask=bool(use_support_mask),
            saturated_channel_cache=saturated_channel_cache,
        )
        block_meta.append(
            {
                "tpcid": int(block["tpcid"]),
                "n_fit_channels": int(block["channel_indices"].size),
                "skipped_for_fit": bool(int(block["channel_indices"].size) == 0),
                **dict(block["support_meta"]),
            }
        )
        if int(block["channel_indices"].size) == 0:
            continue
        predicted_blocks.append(np.asarray(block["selected_wave"], dtype=np.float32))
        base_blocks.append(np.asarray(block["base_block"], dtype=np.float32))
        actual_blocks.append(np.asarray(block["actual_block"], dtype=np.float32))
        std_blocks.append(np.asarray(block["std_block"], dtype=np.float32))

    if not predicted_blocks:
        raise ValueError(
            f"Cluster {int(clusterid)} has no usable fit channels after the saturation veto."
        )

    return (
        np.concatenate(predicted_blocks, axis=0),
        np.concatenate(base_blocks, axis=0),
        np.concatenate(actual_blocks, axis=0),
        np.concatenate(std_blocks, axis=0),
        block_meta,
    )


__all__ = [
    "build_cluster_channel_support_cache_v10_2",
    "get_cluster_tpc_fit_block_v10_2",
    "stack_cluster_fit_inputs_v10_2",
]
