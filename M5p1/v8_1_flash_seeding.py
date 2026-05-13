from __future__ import annotations

from typing import Any

import numpy as np

try:
    from v3_2_global_matching import append_candidate_t0
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.v3_2_global_matching import append_candidate_t0


def light_tpc_to_charge_tpc_for_flash(light_tpc_id: int) -> int:
    light_tpc_id = int(light_tpc_id)
    return light_tpc_id + 1 if light_tpc_id % 2 == 0 else light_tpc_id - 1


def charge_tpc_to_light_tpc_for_flash(charge_tpc_id: int) -> int:
    charge_tpc_id = int(charge_tpc_id)
    return light_tpc_to_charge_tpc_for_flash(charge_tpc_id)


def _scalarize_flash_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr


def _extract_flash_time_summary(
    flash_row: np.void,
    available_fields: list[str],
) -> tuple[dict[str, Any], str | None, float | None]:
    out: dict[str, Any] = {}
    preferred_time_fields = [
        "t0",
        "time",
        "time_tick",
        "ts_s",
        "ts_sync",
        "tai_ns",
        "unix_ts",
        "sample_idx",
        "sample",
        "peak_time",
        "hit_time",
        "hit_time_range",
    ]

    for field in available_fields:
        low = field.lower()
        if ("time" not in low) and ("t0" not in low) and (not low.startswith("ts")) and ("sample" not in low):
            continue

        value = _scalarize_flash_value(flash_row[field])
        if field == "hit_time_range":
            arr = np.asarray(value, dtype=np.float64).ravel()
            if arr.size >= 2:
                out["hit_time_start"] = float(arr[0])
                out["hit_time_end"] = float(arr[1])
                out["hit_time_mid"] = float(0.5 * (arr[0] + arr[1]))
            elif arr.size == 1:
                out["hit_time_mid"] = float(arr[0])
        else:
            arr = np.asarray(value)
            if arr.ndim == 0:
                try:
                    out[field] = float(value)
                except Exception:
                    out[field] = value
            else:
                out[field] = arr.tolist()

    best_name: str | None = None
    best_value: float | None = None
    for key in preferred_time_fields:
        if key == "hit_time_range":
            if "hit_time_mid" in out:
                best_name = "hit_time_mid"
                best_value = float(out["hit_time_mid"])
                break
            continue
        if key in out:
            try:
                best_name = str(key)
                best_value = float(out[key])
                break
            except Exception:
                continue

    return out, best_name, best_value


def extract_flash_t0_candidates_from_table(
    *,
    h5_file: Any,
    eventid: int,
    search_range: int = 800,
    max_new_per_tpc: int | None = None,
    t0_resolution: int = 5,
    flash_tick_divisor: float = 16.0,
    flash_tick_offset: float = -5.0,
    charge_tpc_min: int = 0,
    charge_tpc_max: int = 69,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    ref_ev_flash = h5_file["light/events/ref/light/flash/ref"][...]
    flash_data = h5_file["light/flash/data"]
    flash_ids = flash_data["id"][...].astype(np.int64)
    flash_tpcs = flash_data["tpc"][...].astype(np.int64)

    event_rows = ref_ev_flash[ref_ev_flash[:, 0].astype(np.int64) == int(eventid)]
    if len(event_rows) == 0:
        return {}, {
            "eventid": int(eventid),
            "n_event_flash_refs": 0,
            "n_charge_tpcs_with_flash_seeds": 0,
            "n_flash_seed_t0s": 0,
            "max_new_per_tpc": None if max_new_per_tpc is None else int(max_new_per_tpc),
            "t0_resolution": int(t0_resolution),
            "flash_tick_divisor": float(flash_tick_divisor),
            "flash_tick_offset": float(flash_tick_offset),
        }

    event_flash_ids = event_rows[:, 1].astype(np.int64)
    available_fields = list(flash_data.dtype.names)

    mask = np.isin(flash_ids, event_flash_ids)
    matched = flash_data[mask]
    if len(matched) == 0:
        return {}, {
            "eventid": int(eventid),
            "n_event_flash_refs": int(len(event_flash_ids)),
            "n_charge_tpcs_with_flash_seeds": 0,
            "n_flash_seed_t0s": 0,
            "max_new_per_tpc": None if max_new_per_tpc is None else int(max_new_per_tpc),
            "t0_resolution": int(t0_resolution),
            "flash_tick_divisor": float(flash_tick_divisor),
            "flash_tick_offset": float(flash_tick_offset),
        }

    raw_by_charge_tpc: dict[int, list[tuple[float, int, int, str]]] = {}
    dropped_missing_time = 0
    for row in matched:
        light_tpcid = int(row["tpc"])
        charge_tpcid = int(light_tpc_to_charge_tpc_for_flash(light_tpcid))
        if charge_tpcid < int(charge_tpc_min) or charge_tpcid > int(charge_tpc_max):
            continue

        _, best_name, best_value = _extract_flash_time_summary(row, available_fields)
        if best_value is None:
            dropped_missing_time += 1
            continue

        t0_float = float(best_value) / float(flash_tick_divisor) + float(flash_tick_offset)
        t0 = int(np.clip(np.floor(t0_float + 0.5), 0, int(search_range)))
        raw_by_charge_tpc.setdefault(int(charge_tpcid), []).append(
            (float(best_value), float(t0_float), int(t0), int(row["id"]), str(best_name))
        )

    flash_t0s_by_charge_tpc: dict[int, list[int]] = {}
    details_by_tpc: dict[int, list[dict[str, Any]]] = {}
    for charge_tpcid, items in sorted(raw_by_charge_tpc.items()):
        # Favor earlier / stronger-separated times only by raw converted t0 ordering.
        items_sorted = sorted(items, key=lambda item: (int(item[1]), int(item[2])))
        kept_t0s: list[int] = []
        kept_rows: list[dict[str, Any]] = []
        limit = None if max_new_per_tpc is None or int(max_new_per_tpc) <= 0 else int(max_new_per_tpc)
        for raw_value, raw_t0_float, t0, flash_id, best_name in items_sorted:
            if any(abs(int(t0) - int(existing)) <= int(t0_resolution) for existing in kept_t0s):
                continue
            kept_t0s.append(int(t0))
            kept_rows.append(
                {
                    "flash_id": int(flash_id),
                    "best_t0_field": str(best_name),
                    "best_t0_value": float(raw_value),
                    "converted_t0_float": float(raw_t0_float),
                    "converted_t0": int(t0),
                    "light_tpcid": int(charge_tpc_to_light_tpc_for_flash(int(charge_tpcid))),
                    "charge_tpcid": int(charge_tpcid),
                }
            )
            if limit is not None and len(kept_t0s) >= limit:
                break
        if kept_t0s:
            flash_t0s_by_charge_tpc[int(charge_tpcid)] = kept_t0s
            details_by_tpc[int(charge_tpcid)] = kept_rows

    summary = {
        "eventid": int(eventid),
        "n_event_flash_refs": int(len(event_flash_ids)),
        "n_matched_flash_rows": int(len(matched)),
        "n_charge_tpcs_with_flash_seeds": int(len(flash_t0s_by_charge_tpc)),
        "n_flash_seed_t0s": int(sum(len(v) for v in flash_t0s_by_charge_tpc.values())),
        "dropped_missing_time": int(dropped_missing_time),
        "max_new_per_tpc": None if max_new_per_tpc is None else int(max_new_per_tpc),
        "t0_resolution": int(t0_resolution),
        "flash_tick_divisor": float(flash_tick_divisor),
        "flash_tick_offset": float(flash_tick_offset),
        "details_by_tpc": details_by_tpc,
    }
    return flash_t0s_by_charge_tpc, summary


def merge_flash_t0_candidates_after_primary(
    *,
    t0_candidates: list[list[int]],
    flash_seed_t0s_by_tpc: dict[int, list[int]] | None,
    candidate_min_sep: int = 5,
    max_t0: int = 800,
) -> tuple[list[list[int]], dict[int, list[int]], dict[str, Any]]:
    if not flash_seed_t0s_by_tpc:
        return t0_candidates, {}, {
            "n_tpcs_considered": 0,
            "n_tpcs_merged": 0,
            "n_flash_seed_t0s_kept": 0,
            "n_flash_seed_t0s_skipped_overlap": 0,
            "candidate_min_sep": int(candidate_min_sep),
        }

    kept_by_tpc: dict[int, list[int]] = {}
    skipped_overlap = 0
    for tpcid, t0s in sorted(flash_seed_t0s_by_tpc.items()):
        existing = [int(v) for v in t0_candidates[int(tpcid)]]
        kept: list[int] = []
        for t0 in t0s:
            if any(abs(int(t0) - int(existing_t0)) <= int(candidate_min_sep) for existing_t0 in existing):
                skipped_overlap += 1
                continue
            added = append_candidate_t0(
                t0_candidates[int(tpcid)],
                int(t0),
                min_sep=int(candidate_min_sep),
                max_t0=int(max_t0),
            )
            if added:
                kept.append(int(t0))
                existing.append(int(t0))
            else:
                skipped_overlap += 1
        if kept:
            kept_by_tpc[int(tpcid)] = kept

    summary = {
        "n_tpcs_considered": int(len(flash_seed_t0s_by_tpc)),
        "n_tpcs_merged": int(len(kept_by_tpc)),
        "n_flash_seed_t0s_kept": int(sum(len(v) for v in kept_by_tpc.values())),
        "n_flash_seed_t0s_skipped_overlap": int(skipped_overlap),
        "candidate_min_sep": int(candidate_min_sep),
    }
    return t0_candidates, kept_by_tpc, summary


def smooth_curve(curve: np.ndarray, width: int = 11) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float32)
    if int(width) <= 1:
        return arr
    kernel = np.ones(int(width), dtype=np.float32) / float(width)
    return np.convolve(arr, kernel, mode="same")


def top_unique_peaks(
    curve: np.ndarray,
    *,
    n_peaks: int = 2,
    min_sep: int = 10,
    min_height: float | None = None,
) -> list[int]:
    arr = np.asarray(curve, dtype=np.float32)
    if arr.size == 0:
        return []

    order = np.argsort(arr)[::-1]
    peaks: list[int] = []
    for idx in order:
        val = float(arr[int(idx)])
        if min_height is not None and val < float(min_height):
            break
        if any(abs(int(idx) - int(prev)) < int(min_sep) for prev in peaks):
            continue
        peaks.append(int(idx))
        if len(peaks) >= int(n_peaks):
            break
    return peaks


def build_residual_curves(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    smooth_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    act = np.asarray(actual, dtype=np.float32)
    pred = np.asarray(predicted, dtype=np.float32)
    residual = act - pred

    deficit = np.clip(residual, 0.0, None).sum(axis=0)
    overflow = np.clip(-residual, 0.0, None).sum(axis=0)
    return smooth_curve(deficit, width=int(smooth_width)), smooth_curve(overflow, width=int(smooth_width))


def candidate_t0s_from_residual(
    actual: np.ndarray,
    predicted: np.ndarray,
    existing_candidates: list[int],
    *,
    max_new: int = 2,
    search_range: int = 800,
    pulse_peak_tick: int = 105,
    t0_resolution: int = 5,
    peak_fraction: float = 0.30,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    smooth_width = max(2 * int(t0_resolution) + 1, 5)
    deficit, overflow = build_residual_curves(
        actual,
        predicted,
        smooth_width=int(smooth_width),
    )
    if deficit.size == 0 or float(deficit.max()) <= 0.0:
        return [], deficit, overflow

    peak_floor = max(
        float(peak_fraction) * float(deficit.max()),
        float(deficit.mean() + deficit.std()),
    )
    peak_idx = top_unique_peaks(
        deficit,
        n_peaks=int(max_new),
        min_sep=max(2 * int(t0_resolution), 8),
        min_height=float(peak_floor),
    )

    new_t0s: list[int] = []
    taken = [int(t0) for t0 in existing_candidates]
    for idx in peak_idx:
        cand_t0 = int(np.clip(int(idx) - int(pulse_peak_tick), 0, int(search_range)))
        if any(abs(int(cand_t0) - int(old)) <= max(2 * int(t0_resolution), 8) for old in taken + new_t0s):
            continue
        new_t0s.append(int(cand_t0))

    return new_t0s, deficit, overflow


def seed_t0_candidates_from_flash_residual(
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    t0_candidates: list[list[int]],
    search_range: int = 800,
    max_new_per_tpc: int = 2,
    pulse_peak_tick: int = 105,
    t0_resolution: int = 5,
    peak_fraction: float = 0.30,
    candidate_min_sep: int | None = None,
) -> tuple[list[list[int]], dict[int, list[int]], dict[str, Any]]:
    updated_candidates = t0_candidates
    flash_seed_t0s_by_tpc: dict[int, list[int]] = {}
    deficit_strengths: list[float] = []

    if candidate_min_sep is None:
        candidate_min_sep = max(2 * int(t0_resolution), 8)

    n_tpcs = int(np.asarray(base_image).shape[0])
    for tpc in range(n_tpcs):
        existing = [int(v) for v in updated_candidates[int(tpc)]]
        new_t0s, deficit, _ = candidate_t0s_from_residual(
            actual=np.asarray(full_light_waveform[int(tpc)], dtype=np.float32),
            predicted=np.asarray(base_image[int(tpc)], dtype=np.float32),
            existing_candidates=existing,
            max_new=int(max_new_per_tpc),
            search_range=int(search_range),
            pulse_peak_tick=int(pulse_peak_tick),
            t0_resolution=int(t0_resolution),
            peak_fraction=float(peak_fraction),
        )
        kept: list[int] = []
        for t0 in new_t0s:
            added = append_candidate_t0(
                updated_candidates[int(tpc)],
                int(t0),
                min_sep=int(candidate_min_sep),
                max_t0=int(search_range),
            )
            if added:
                kept.append(int(t0))
        if kept:
            flash_seed_t0s_by_tpc[int(tpc)] = kept
            deficit_strengths.append(float(np.max(deficit)))

    summary = {
        "n_tpcs_with_flash_seeds": int(len(flash_seed_t0s_by_tpc)),
        "n_flash_seed_t0s": int(sum(len(v) for v in flash_seed_t0s_by_tpc.values())),
        "max_new_per_tpc": int(max_new_per_tpc),
        "t0_resolution": int(t0_resolution),
        "peak_fraction": float(peak_fraction),
        "candidate_min_sep": int(candidate_min_sep),
        "mean_deficit_peak": float(np.mean(deficit_strengths)) if deficit_strengths else 0.0,
    }
    return updated_candidates, flash_seed_t0s_by_tpc, summary


__all__ = [
    "light_tpc_to_charge_tpc_for_flash",
    "charge_tpc_to_light_tpc_for_flash",
    "extract_flash_t0_candidates_from_table",
    "merge_flash_t0_candidates_after_primary",
    "smooth_curve",
    "top_unique_peaks",
    "build_residual_curves",
    "candidate_t0s_from_residual",
    "seed_t0_candidates_from_flash_residual",
]
