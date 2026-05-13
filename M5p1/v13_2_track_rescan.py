from __future__ import annotations

from typing import Any

import numpy as np

from pulse_shapes import timeinterpolation
from fast_unit_track_scan import unit_likelihood_curve_with_base_v1

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

    if err.size > 0 and bool(np.all(np.abs(err - np.float32(1.0)) <= np.float32(1.0e-6))):
        return unit_likelihood_curve_with_base_v1(
            pred,
            base_,
            act,
            search_range=int(search_range),
            engine="fft",
        )

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


def _refine_t0_from_local_peak(
    *,
    raw_t0: int,
    actual_fit: np.ndarray,
    base_fit: np.ndarray,
    t0_resolution: int,
    waveform_len: int,
) -> int:
    new_t0 = int(raw_t0)
    expected_peak = 105 + int(new_t0)
    signal_1d = np.sum(np.clip(np.asarray(actual_fit, dtype=np.float32) - np.asarray(base_fit, dtype=np.float32), 0, None), axis=0)
    s_start = max(0, expected_peak - int(t0_resolution))
    s_end = min(int(waveform_len), expected_peak + int(t0_resolution) + 1)
    if s_end > s_start:
        local_peak = int(np.argmax(signal_1d[s_start:s_end]))
        actual_peak = int(s_start + local_peak)
        new_t0 += int(actual_peak - expected_peak)
    return int(new_t0)


def _identify_shower_context_tpcs_v13_2(
    *,
    clusterid: int,
    sorted_tpcs: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    energies: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    min_shower_energy_mev: float,
) -> tuple[list[int], list[int], dict[int, float]]:
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    energies = np.asarray(energies, dtype=np.float64)
    shower_tpcs: list[int] = []
    clean_tpcs: list[int] = []
    shower_energy_by_tpc: dict[int, float] = {}

    for tpcid in np.asarray(sorted_tpcs, dtype=int).tolist():
        tpc_mask = hit_tpc_ids == int(tpcid)
        local_labels = sorted(int(v) for v in np.unique(labels_global[tpc_mask]) if int(v) >= 0)
        strong_shower_energy = 0.0
        for label in local_labels:
            if int(label) == int(clusterid):
                continue
            if str(label_info.get(int(label), {}).get("type", "cluster")).lower() != "shower":
                continue
            label_mask = tpc_mask & (labels_global == int(label))
            strong_shower_energy += float(np.sum(energies[label_mask]))
        shower_energy_by_tpc[int(tpcid)] = float(strong_shower_energy)
        if float(strong_shower_energy) >= float(min_shower_energy_mev):
            shower_tpcs.append(int(tpcid))
        else:
            clean_tpcs.append(int(tpcid))
    return shower_tpcs, clean_tpcs, shower_energy_by_tpc


def _fit_track_t0_with_shower_guard_v13_2(
    *,
    clusterid: int,
    sorted_tpcs: np.ndarray,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    energies: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    search_range: int,
    adc_clip: float,
    t0_resolution: int,
    waveform_len: int,
    min_shower_energy_mev: float,
    min_clean_t0_separation_ticks: int,
    clean_worsen_tolerance_norm: float,
    saturated_channel_cache: dict[str, Any] | None,
) -> dict[str, Any]:
    sorted_tpcs = np.asarray(sorted_tpcs, dtype=int)
    all_predicted_fit, all_base_fit, all_actual_fit, all_std_fit, all_fit_support_meta = stack_cluster_fit_inputs_v10_3(
        clusterid=int(clusterid),
        tpcids=sorted_tpcs,
        image_maps=image_maps,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        channel_support_cache=channel_support_cache,
        use_support_mask=True,
        saturated_channel_cache=saturated_channel_cache,
    )
    all_shifts, all_errors = max_likelihood_curve_with_base_v10_3(
        all_predicted_fit,
        all_base_fit,
        all_actual_fit,
        all_std_fit,
        search_range=int(search_range),
        adc_clip=float(adc_clip),
    )
    all_raw_t0 = int(all_shifts[int(np.argmin(all_errors))])
    all_refined_t0 = int(
        np.clip(
            _refine_t0_from_local_peak(
                raw_t0=int(all_raw_t0),
                actual_fit=all_actual_fit,
                base_fit=all_base_fit,
                t0_resolution=int(t0_resolution),
                waveform_len=int(waveform_len),
            ),
            0,
            int(search_range),
        )
    )
    all_best_score = float(np.min(all_errors))

    shower_tpcs, clean_tpcs, shower_energy_by_tpc = _identify_shower_context_tpcs_v13_2(
        clusterid=int(clusterid),
        sorted_tpcs=sorted_tpcs,
        labels_global=labels_global,
        hit_tpc_ids=hit_tpc_ids,
        energies=energies,
        label_info=label_info,
        min_shower_energy_mev=float(min_shower_energy_mev),
    )

    decision = {
        "mode": "standard_all_tpcs",
        "shower_tpcs": [int(v) for v in shower_tpcs],
        "clean_tpcs": [int(v) for v in clean_tpcs],
        "shower_energy_by_tpc": {int(k): float(v) for k, v in shower_energy_by_tpc.items()},
        "all_raw_t0": int(all_raw_t0),
        "all_refined_t0": int(all_refined_t0),
        "all_best_score": float(all_best_score),
        "all_fit_support_meta": [dict(item) for item in all_fit_support_meta],
        "selected_raw_t0": int(all_raw_t0),
        "selected_t0": int(all_refined_t0),
        "selected_fit_support_meta": [dict(item) for item in all_fit_support_meta],
        "selected_errors": np.asarray(all_errors, dtype=np.float32),
        "selected_shifts": np.asarray(all_shifts, dtype=np.int32),
        "improvement_vs_all": 0.0,
    }

    guard_active = (
        int(sorted_tpcs.size) > 1
        and len(shower_tpcs) > 0
        and len(clean_tpcs) > 0
    )
    if not bool(guard_active):
        return decision

    clean_predicted_fit, clean_base_fit, clean_actual_fit, clean_std_fit, clean_fit_support_meta = stack_cluster_fit_inputs_v10_3(
        clusterid=int(clusterid),
        tpcids=np.asarray(clean_tpcs, dtype=int),
        image_maps=image_maps,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        channel_support_cache=channel_support_cache,
        use_support_mask=True,
        saturated_channel_cache=saturated_channel_cache,
    )
    clean_shifts, clean_errors = max_likelihood_curve_with_base_v10_3(
        clean_predicted_fit,
        clean_base_fit,
        clean_actual_fit,
        clean_std_fit,
        search_range=int(search_range),
        adc_clip=float(adc_clip),
    )
    clean_raw_t0 = int(clean_shifts[int(np.argmin(clean_errors))])
    clean_refined_t0 = int(
        np.clip(
            _refine_t0_from_local_peak(
                raw_t0=int(clean_raw_t0),
                actual_fit=clean_actual_fit,
                base_fit=clean_base_fit,
                t0_resolution=int(t0_resolution),
                waveform_len=int(waveform_len),
            ),
            0,
            int(search_range),
        )
    )
    clean_best_score = float(np.min(clean_errors))
    clean_idx_all = int(np.clip(int(all_raw_t0), 0, clean_errors.size - 1))
    clean_score_at_all = float(clean_errors[clean_idx_all])
    clean_curve_span = float(np.percentile(clean_errors, 90.0) - clean_best_score)
    clean_curve_span = max(float(clean_curve_span), 1e-6)
    clean_worsen_norm = float((clean_score_at_all - clean_best_score) / clean_curve_span)

    decision.update(
        {
            "mode": "guard_considered",
            "clean_raw_t0": int(clean_raw_t0),
            "clean_refined_t0": int(clean_refined_t0),
            "clean_best_score": float(clean_best_score),
            "clean_score_at_all_raw_t0": float(clean_score_at_all),
            "clean_worsen_norm": float(clean_worsen_norm),
            "clean_fit_support_meta": [dict(item) for item in clean_fit_support_meta],
        }
    )

    if (
        abs(int(clean_refined_t0) - int(all_refined_t0)) >= int(min_clean_t0_separation_ticks)
        and float(clean_worsen_norm) > float(clean_worsen_tolerance_norm)
    ):
        decision.update(
            {
                "mode": "shower_guard_clean_tpcs",
                "selected_raw_t0": int(clean_raw_t0),
                "selected_t0": int(clean_refined_t0),
                "selected_fit_support_meta": [dict(item) for item in clean_fit_support_meta],
                "selected_errors": np.asarray(clean_errors, dtype=np.float32),
                "selected_shifts": np.asarray(clean_shifts, dtype=np.int32),
                "improvement_vs_all": float(clean_score_at_all - clean_best_score),
            }
        )

    return decision


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
    energies: np.ndarray | None = None,
    hit_tpc_ids: np.ndarray | None = None,
    min_shower_energy_mev: float = 80.0,
    min_clean_t0_separation_ticks: int = 8,
    clean_worsen_tolerance_norm: float = 0.08,
    saturated_channel_cache: dict[str, Any] | None = None,
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

        if energies is not None and hit_tpc_ids is not None:
            fit_decision = _fit_track_t0_with_shower_guard_v13_2(
                clusterid=int(clusterid),
                sorted_tpcs=sorted_tpcs,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                labels_global=labels_global,
                hit_tpc_ids=np.asarray(hit_tpc_ids, dtype=np.int32),
                energies=np.asarray(energies, dtype=np.float32),
                label_info=label_info,
                channel_support_cache=channel_support_cache,
                search_range=int(search_range),
                adc_clip=float(adc_clip),
                t0_resolution=int(t0_resolution),
                waveform_len=int(waveform_len),
                min_shower_energy_mev=float(min_shower_energy_mev),
                min_clean_t0_separation_ticks=int(min_clean_t0_separation_ticks),
                clean_worsen_tolerance_norm=float(clean_worsen_tolerance_norm),
                saturated_channel_cache=saturated_channel_cache,
            )
        else:
            predicted_fit, base_fit, actual_fit, std_fit, fit_support_meta = stack_cluster_fit_inputs_v10_3(
                clusterid=int(clusterid),
                tpcids=sorted_tpcs,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                channel_support_cache=channel_support_cache,
                use_support_mask=True,
                saturated_channel_cache=saturated_channel_cache,
            )
            shifts_, errors_ = max_likelihood_curve_with_base_v10_3(
                predicted_fit,
                base_fit,
                actual_fit,
                std_fit,
                search_range=int(search_range),
                adc_clip=float(adc_clip),
            )
            raw_t0_ = int(shifts_[int(np.argmin(errors_))])
            fit_decision = {
                "mode": "standard_all_tpcs",
                "shower_tpcs": [],
                "clean_tpcs": [int(v) for v in sorted_tpcs.tolist()],
                "all_raw_t0": int(raw_t0_),
                "all_refined_t0": int(
                    np.clip(
                        _refine_t0_from_local_peak(
                            raw_t0=int(raw_t0_),
                            actual_fit=actual_fit,
                            base_fit=base_fit,
                            t0_resolution=int(t0_resolution),
                            waveform_len=int(waveform_len),
                        ),
                        0,
                        int(search_range),
                    )
                ),
                "selected_raw_t0": int(raw_t0_),
                "selected_t0": int(
                    np.clip(
                        _refine_t0_from_local_peak(
                            raw_t0=int(raw_t0_),
                            actual_fit=actual_fit,
                            base_fit=base_fit,
                            t0_resolution=int(t0_resolution),
                            waveform_len=int(waveform_len),
                        ),
                        0,
                        int(search_range),
                    )
                ),
                "selected_fit_support_meta": [dict(item) for item in fit_support_meta],
                "selected_errors": np.asarray(errors_, dtype=np.float32),
                "selected_shifts": np.asarray(shifts_, dtype=np.int32),
            }

        shifts = np.asarray(fit_decision["selected_shifts"], dtype=np.int32)
        errors = np.asarray(fit_decision["selected_errors"], dtype=np.float32)
        raw_t0 = int(fit_decision["selected_raw_t0"])
        new_t0 = int(fit_decision["selected_t0"])
        fit_support_meta = [dict(item) for item in fit_decision["selected_fit_support_meta"]]
        n_fit_channels = int(sum(int(item.get("n_fit_channels", 0)) for item in fit_support_meta))

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
                    "rescan_guard_mode": str(fit_decision.get("mode", "standard_all_tpcs")),
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
                "rescan_guard_mode": str(fit_decision.get("mode", "standard_all_tpcs")),
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
                "guard_mode": str(fit_decision.get("mode", "standard_all_tpcs")),
                "guard_shower_tpcs": [int(v) for v in fit_decision.get("shower_tpcs", [])],
                "guard_clean_tpcs": [int(v) for v in fit_decision.get("clean_tpcs", [])],
                "guard_clean_worsen_norm": float(fit_decision.get("clean_worsen_norm", 0.0)),
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
