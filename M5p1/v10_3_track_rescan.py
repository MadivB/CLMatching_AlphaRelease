from __future__ import annotations

from typing import Any

import numpy as np

from pulse_shapes import timeinterpolation

from v10_3_support import stack_cluster_fit_inputs_v10_3


def max_likelihood_curve_with_base_v10_3(
    predicted: np.ndarray,
    base: np.ndarray,
    actual: np.ndarray,
    error_metric: np.ndarray,
    *,
    search_range: int = 800,
    adc_clip: float = 60780.0,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted, dtype=np.float32)
    base_ = np.asarray(base, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    err = np.asarray(error_metric, dtype=np.float32)

    n_ticks = int(pred.size)
    shifts = np.arange(int(search_range) + 1, dtype=np.int32)
    errors = np.empty(shifts.size, dtype=np.float32)

    for idx, t0 in enumerate(shifts):
        shifted = np.zeros_like(pred)
        if int(t0) > 0:
            shifted[:, int(t0):] = pred[:, :-int(t0)]
        else:
            shifted[:] = pred
        model = np.clip(shifted + base_, None, float(adc_clip))
        errors[idx] = np.sum((model - act) ** 2 / err) / float(n_ticks)

    return shifts, errors


def _append_candidate_t0_local(t0_candidates: list[list[int]], tpcid: int, t0: int) -> None:
    tpcid = int(tpcid)
    t0 = int(t0)
    bucket = t0_candidates[tpcid]
    if t0 not in bucket:
        bucket.append(int(t0))
        bucket.sort()


def _stack_full_cluster_wave(
    *,
    clusterid: int,
    tpcids: np.ndarray,
    image_maps: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    images = [np.asarray(image_maps[(int(clusterid), int(tpc))], dtype=np.float32) for tpc in np.asarray(tpcids, dtype=int)]
    return np.stack(images, axis=0).reshape(-1, images[0].shape[-1])


def run_track_second_pass_rescan_v10_3(
    *,
    track_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    label_info: dict[int, dict[str, Any]],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    cluster_full_scan_loss_dict: dict[int, dict[str, Any]] | None = None,
    absorbed_hit_parent: np.ndarray | None = None,
    search_range: int = 800,
    adc_clip: float = 60780.0,
    t0_resolution: int = 5,
    waveform_len: int = 1000,
    collect_scan_losses: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    logs: list[dict[str, Any]] = []
    scan_updates: dict[int, dict[str, Any]] = {}
    n_changed_t0 = 0

    for clusterid in [int(cid) for cid in track_labels]:
        label_type = str(label_info.get(int(clusterid), {}).get("type", "track")).lower()
        if label_type != "track":
            continue

        sorted_tpcs = np.sort(
            np.asarray(
                [
                    int(tpc)
                    for tpc in cluster_to_tpcs.get(int(clusterid), [])
                    if (int(clusterid), int(tpc)) in image_maps
                ],
                dtype=int,
            )
        )
        if sorted_tpcs.size == 0:
            continue

        current_t0_values: list[int] = []
        for tpc in sorted_tpcs.tolist():
            info = assignment_info.get((int(clusterid), int(tpc)), {})
            t0_val = info.get("t0", np.nan)
            if np.isfinite(t0_val):
                current_t0_values.append(int(round(float(t0_val))))
        if len(current_t0_values) == 0:
            continue

        old_t0 = int(np.median(np.asarray(current_t0_values, dtype=np.int32)))

        full_wave_stack = _stack_full_cluster_wave(
            clusterid=int(clusterid),
            tpcids=sorted_tpcs,
            image_maps=image_maps,
        )
        shifted_old = timeinterpolation(full_wave_stack, shift=float(old_t0), baseline=0).astype(np.float32)
        shifted_old = shifted_old.reshape(sorted_tpcs.size, full_light_waveform.shape[1], -1)
        base_image[sorted_tpcs] = np.clip(base_image[sorted_tpcs] - shifted_old, 0.0, None)

        predicted_fit, base_fit, actual_fit, std_fit, fit_support_meta = stack_cluster_fit_inputs_v10_3(
            clusterid=int(clusterid),
            tpcids=sorted_tpcs,
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            channel_support_cache=channel_support_cache,
            use_support_mask=True,
        )
        n_fit_channels = int(predicted_fit.shape[0])

        shifts, errors = max_likelihood_curve_with_base_v10_3(
            predicted_fit,
            base_fit,
            actual_fit,
            std_fit,
            search_range=int(search_range),
            adc_clip=float(adc_clip),
        )
        raw_t0 = int(shifts[int(np.argmin(errors))])
        new_t0 = int(raw_t0)

        expected_peak = 105 + int(new_t0)
        signal_1d = np.sum(np.clip(actual_fit - base_fit, 0, None), axis=0)
        s_start = max(0, expected_peak - int(t0_resolution))
        s_end = min(int(waveform_len), expected_peak + int(t0_resolution) + 1)
        if s_end > s_start:
            local_peak = int(np.argmax(signal_1d[s_start:s_end]))
            actual_peak = int(s_start + local_peak)
            new_t0 += int(actual_peak - expected_peak)
        new_t0 = int(np.clip(new_t0, 0, int(search_range)))

        old_idx = int(np.clip(old_t0, 0, errors.size - 1))
        current_score = float(errors[old_idx])
        best_score = float(np.min(errors))
        improvement = float(current_score - best_score)

        shifted_new = timeinterpolation(full_wave_stack, shift=float(new_t0), baseline=0).astype(np.float32)
        shifted_new = shifted_new.reshape(sorted_tpcs.size, full_light_waveform.shape[1], -1)
        base_image[sorted_tpcs] = np.clip(base_image[sorted_tpcs] + shifted_new, None, float(adc_clip))

        hit_timestamps[np.asarray(labels_global, dtype=int) == int(clusterid)] = float(new_t0)
        if absorbed_hit_parent is not None:
            hit_timestamps[np.asarray(absorbed_hit_parent, dtype=np.int32) == int(clusterid)] = float(new_t0)

        fit_meta_by_tpc = {int(item["tpcid"]): dict(item) for item in fit_support_meta}
        for tpc in sorted_tpcs.tolist():
            fit_meta = dict(fit_meta_by_tpc.get(int(tpc), {}))
            old_info = dict(assignment_info.get((int(clusterid), int(tpc)), {}))
            old_info.update(
                {
                    "stage": "track",
                    "mode": "track_second_pass",
                    "t0": float(new_t0),
                    "assigned": True,
                    "backbone_type": "track",
                    "n_fit_channels": int(fit_meta.get("n_fit_channels", full_light_waveform.shape[1])),
                    "channel_selection_spec": list(fit_meta.get("channel_selection_spec", [])),
                    "selected_fraction": float(fit_meta.get("selected_fraction", 1.0)),
                    "dominant_fraction": float(fit_meta.get("dominant_fraction", 1.0)),
                    "support_mode": str(fit_meta.get("support_mode", "masked_channels")),
                    "rescan_old_t0": float(old_t0),
                    "rescan_raw_t0": float(raw_t0),
                    "rescan_final_t0": float(new_t0),
                    "rescan_delta_t0": int(new_t0 - old_t0),
                    "rescan_improvement": float(improvement),
                }
            )
            assignment_info[(int(clusterid), int(tpc))] = old_info
            _append_candidate_t0_local(t0_candidates, int(tpc), int(new_t0))

        if collect_scan_losses and cluster_full_scan_loss_dict is not None:
            previous_entry = dict(cluster_full_scan_loss_dict.get(int(clusterid), {}))
            passes = {}
            if isinstance(previous_entry.get("passes"), dict):
                passes.update(previous_entry.get("passes", {}))
            preserved = dict(previous_entry)
            preserved.pop("passes", None)
            if preserved and "initial_track_scan" not in passes:
                passes["initial_track_scan"] = preserved

            second_pass_entry = {
                "clusterid": int(clusterid),
                "stage": "track",
                "mode": "track_second_pass",
                "tpcs": sorted_tpcs.tolist(),
                "energy": float(previous_entry.get("energy", np.nan)),
                "assigned": True,
                "scan_performed": True,
                "best_t0": int(new_t0),
                "best_t0_scan": int(raw_t0),
                "t0_grid": np.asarray(shifts, dtype=np.int32).copy(),
                "loss_curve": np.asarray(errors, dtype=np.float32).copy(),
                "n_fit_channels": int(n_fit_channels),
                "fit_support_meta": [dict(item) for item in fit_support_meta],
                "rescan_old_t0": int(old_t0),
                "rescan_delta_t0": int(new_t0 - old_t0),
            }
            passes["track_second_pass"] = dict(second_pass_entry)
            second_pass_entry["passes"] = passes
            scan_updates[int(clusterid)] = second_pass_entry

        if int(new_t0) != int(old_t0):
            n_changed_t0 += 1

        logs.append(
            {
                "clusterid": int(clusterid),
                "tpcs": sorted_tpcs.tolist(),
                "old_t0": int(old_t0),
                "raw_t0": int(raw_t0),
                "new_t0": int(new_t0),
                "delta_t0": int(new_t0 - old_t0),
                "current_score": float(current_score),
                "best_score": float(best_score),
                "improvement": float(improvement),
                "n_fit_channels": int(n_fit_channels),
                "fit_support_meta": [dict(item) for item in fit_support_meta],
            }
        )

    stats = {
        "n_tracks_rescanned": int(len(logs)),
        "n_tracks_changed_t0": int(n_changed_t0),
        "mean_improvement": float(np.mean([item["improvement"] for item in logs])) if logs else 0.0,
        "median_improvement": float(np.median([item["improvement"] for item in logs])) if logs else 0.0,
    }

    return (
        base_image,
        hit_timestamps,
        t0_candidates,
        assignment_info,
        scan_updates,
        logs,
        stats,
    )


__all__ = [
    "max_likelihood_curve_with_base_v10_3",
    "run_track_second_pass_rescan_v10_3",
]
