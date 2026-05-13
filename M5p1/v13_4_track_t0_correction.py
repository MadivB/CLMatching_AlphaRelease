from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pulse_shapes import timeinterpolation
    from v10_3_support import stack_cluster_fit_inputs_v10_3
except ModuleNotFoundError:
    from NewMLSection.pulse_shapes import timeinterpolation
    from M5p1.v10_3_support import stack_cluster_fit_inputs_v10_3


def _stack_images(image_maps: dict[tuple[int, int], np.ndarray], cluster_id: int, tpcids: np.ndarray) -> np.ndarray:
    images = [np.asarray(image_maps[(int(cluster_id), int(tpc))], dtype=np.float32) for tpc in tpcids]
    stacked = np.stack(images, axis=0)
    return stacked.reshape(-1, stacked.shape[-1])


def _shift_stacked_image(image: np.ndarray, t0: float) -> np.ndarray:
    return timeinterpolation(np.asarray(image, dtype=np.float32), shift=float(t0), baseline=0.0).astype(np.float32)


def _append_candidate_t0(candidate_list: list[int], t0: float, *, min_sep: int = 2, max_t0: int = 800) -> bool:
    rounded = int(np.clip(np.rint(float(t0)), 0, int(max_t0)))
    for existing in candidate_list:
        if abs(int(existing) - rounded) <= int(min_sep):
            return False
    candidate_list.append(rounded)
    candidate_list.sort()
    return True


def _unit_std_loss(model: np.ndarray, actual: np.ndarray) -> float:
    model = np.asarray(model, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    return float(np.mean((model - actual) ** 2))


def run_track_t0_fine_correction_v13_4(
    *,
    track_labels: list[int] | np.ndarray,
    cluster_to_tpcs: dict[int, Any],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    label_info: dict[int, dict[str, Any]],
    channel_support_cache: Any | None = None,
    saturated_channel_cache: Any | None = None,
    max_charge_tpc: int | None = None,
    adc_clip: float = 60780.0,
    grid_offsets: np.ndarray | list[float] | None = None,
    improvement_eps: float = 0.0,
    candidate_min_sep: int = 2,
    verbose: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Fine-correct already assigned track t0 values by scanning small fractional offsets.

    This intentionally uses unit waveform std. The starting t0 is the current
    flash-derived / rescan / swap value already written into hit_timestamps.
    For each track, the current track contribution is removed from base_image,
    offsets around the current t0 are tested, and the best offset is written back.
    """
    base_image = np.asarray(base_image, dtype=np.float32).copy()
    hit_timestamps = np.asarray(hit_timestamps, dtype=np.float32).copy()
    labels_global = np.asarray(labels_global, dtype=np.int32)

    if grid_offsets is None:
        grid_offsets = np.arange(-1.5, 1.5 + 1e-6, 0.5, dtype=np.float32)
    grid_offsets = np.asarray(grid_offsets, dtype=np.float32)

    if max_charge_tpc is None:
        max_charge_tpc = int(base_image.shape[0])

    unit_std = np.ones_like(full_light_waveform, dtype=np.float32)
    correction_log: list[dict[str, Any]] = []
    n_scanned = 0
    n_changed = 0

    for clusterid_raw in track_labels:
        clusterid = int(clusterid_raw)
        label_type = str(label_info.get(clusterid, {}).get("type", "track")).lower()
        if label_type != "track":
            continue

        candidate_tpcs_from_label = cluster_to_tpcs.get(clusterid, [])
        sorted_tpcs = np.sort(
            np.asarray(
                [int(tpc) for tpc in candidate_tpcs_from_label if int(tpc) < int(max_charge_tpc)],
                dtype=np.int32,
            )
        )
        if sorted_tpcs.size == 0:
            continue
        if any((clusterid, int(tpc)) not in image_maps for tpc in sorted_tpcs):
            continue

        hit_mask = labels_global == clusterid
        current_times = np.asarray(hit_timestamps[hit_mask], dtype=np.float32)
        current_times = current_times[np.isfinite(current_times)]
        if current_times.size == 0:
            continue

        old_t0 = float(np.median(current_times))

        full_unshifted = _stack_images(image_maps, clusterid, sorted_tpcs)
        old_shifted_full = _shift_stacked_image(full_unshifted, old_t0)
        old_shifted_r = old_shifted_full.reshape(sorted_tpcs.size, base_image.shape[1], base_image.shape[2])

        base_without = base_image.copy()
        base_without[sorted_tpcs] = np.clip(base_without[sorted_tpcs] - old_shifted_r, 0.0, float(adc_clip))

        selected_wave, selected_base, selected_actual, _selected_std, fit_support_meta = stack_cluster_fit_inputs_v10_3(
            clusterid=clusterid,
            tpcids=sorted_tpcs,
            image_maps=image_maps,
            base_image=base_without,
            full_light_waveform=full_light_waveform,
            full_light_std=unit_std,
            channel_support_cache=channel_support_cache,
            use_support_mask=True,
            saturated_channel_cache=saturated_channel_cache,
        )
        if selected_wave.size == 0:
            continue

        candidate_rows = []
        for offset in grid_offsets:
            candidate_t0 = float(old_t0 + float(offset))
            if candidate_t0 < 0.0 or candidate_t0 > 800.0:
                continue
            shifted_fit = _shift_stacked_image(selected_wave, candidate_t0)
            model = np.clip(shifted_fit + selected_base, None, float(adc_clip))
            loss = _unit_std_loss(model, selected_actual)
            candidate_rows.append(
                {
                    "offset": float(offset),
                    "t0": float(candidate_t0),
                    "loss": float(loss),
                }
            )

        if len(candidate_rows) == 0:
            continue

        n_scanned += 1
        losses = np.asarray([row["loss"] for row in candidate_rows], dtype=np.float64)
        abs_offsets = np.asarray([abs(row["offset"]) for row in candidate_rows], dtype=np.float64)
        order = np.lexsort((abs_offsets, losses))
        best = candidate_rows[int(order[0])]
        old_candidates = [row for row in candidate_rows if abs(float(row["offset"])) < 1e-6]
        old_loss = float(old_candidates[0]["loss"]) if len(old_candidates) > 0 else float(
            _unit_std_loss(np.clip(_shift_stacked_image(selected_wave, old_t0) + selected_base, None, float(adc_clip)), selected_actual)
        )

        best_t0 = float(best["t0"])
        best_loss = float(best["loss"])
        improvement = float(old_loss - best_loss)
        changed = bool(abs(best_t0 - old_t0) > 1e-6 and improvement > float(improvement_eps))

        if changed:
            new_shifted_full = _shift_stacked_image(full_unshifted, best_t0)
            new_shifted_r = new_shifted_full.reshape(sorted_tpcs.size, base_image.shape[1], base_image.shape[2])
            base_image[sorted_tpcs] = np.clip(base_without[sorted_tpcs] + new_shifted_r, None, float(adc_clip))
            hit_timestamps[hit_mask] = np.float32(best_t0)
            n_changed += 1

            for tpc in sorted_tpcs:
                _append_candidate_t0(
                    t0_candidates[int(tpc)],
                    best_t0,
                    min_sep=int(candidate_min_sep),
                    max_t0=800,
                )
                key = (clusterid, int(tpc))
                info = dict(assignment_info.get(key, {}))
                old_mode = str(info.get("mode", "track_scan"))
                info.update(
                    {
                        "stage": info.get("stage", "track"),
                        "mode": f"{old_mode}+fine_t0",
                        "t0": float(best_t0),
                        "assigned": True,
                        "backbone_type": "track",
                        "fine_t0_correction": True,
                        "fine_t0_old": float(old_t0),
                        "fine_t0_new": float(best_t0),
                        "fine_t0_delta": float(best_t0 - old_t0),
                        "fine_t0_old_loss": float(old_loss),
                        "fine_t0_new_loss": float(best_loss),
                        "fine_t0_improvement": float(improvement),
                        "fine_t0_grid_offsets": [float(v) for v in grid_offsets.tolist()],
                    }
                )
                assignment_info[key] = info

        row = {
            "clusterid": int(clusterid),
            "tpcs": [int(v) for v in sorted_tpcs.tolist()],
            "old_t0": float(old_t0),
            "new_t0": float(best_t0 if changed else old_t0),
            "best_grid_t0": float(best_t0),
            "delta": float((best_t0 if changed else old_t0) - old_t0),
            "old_loss": float(old_loss),
            "best_loss": float(best_loss),
            "improvement": float(improvement),
            "changed": bool(changed),
            "n_fit_channels": int(selected_wave.shape[0]),
            "fit_support_meta": [dict(item) for item in fit_support_meta],
            "grid": [dict(item) for item in candidate_rows],
        }
        correction_log.append(row)

    stats = {
        "n_tracks_scanned": int(n_scanned),
        "n_tracks_changed": int(n_changed),
        "grid_offsets": [float(v) for v in grid_offsets.tolist()],
        "improvement_eps": float(improvement_eps),
        "unit_std": True,
    }
    if len(correction_log) > 0:
        improvements = np.asarray([float(row["improvement"]) for row in correction_log], dtype=np.float64)
        stats["mean_improvement"] = float(np.mean(improvements))
        stats["median_improvement"] = float(np.median(improvements))
    else:
        stats["mean_improvement"] = 0.0
        stats["median_improvement"] = 0.0

    if verbose:
        print(
            "v13_4 track fine-t0 correction summary: "
            f"scanned={stats['n_tracks_scanned']} | changed={stats['n_tracks_changed']} | "
            f"grid={stats['grid_offsets']}"
        )

    return base_image, hit_timestamps, t0_candidates, assignment_info, correction_log, stats
