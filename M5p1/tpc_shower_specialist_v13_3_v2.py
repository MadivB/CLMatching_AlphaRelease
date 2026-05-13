from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from plottingTools import group_hits_by_time, plot_3d_clusters_with_t0
from tpc_shower_specialist_v13_3 import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_FILE,
    DEFAULT_PULSE_PATH,
    ADC_CLIP,
    _estimate_neighbor_aware_shower_apex,
    _estimate_provisional_shower_apex,
    _line_point_distance,
    _load_event_inputs,
    _normalize_direction,
    build_global_labels_toolbox,
    build_saturated_channel_cache_v12,
    extract_truth_vertex_and_t0_for_hits,
    load_v11_plotting_resources,
    run_local_tpc_shower_specialist,
    timeinterpolation,
)


def _fragment_geometry_rows(result: dict, provisional_apex: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for fragment in result["fragments"]:
        ray = np.asarray(fragment.centroid, dtype=np.float64) - np.asarray(provisional_apex, dtype=np.float64)
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm <= 1e-8:
            angle = 0.0
        else:
            angle = 1.0 - abs(float(np.dot(_normalize_direction(fragment.direction), ray / ray_norm)))
        rows.append(
            {
                "fragment_id": int(fragment.fragment_id),
                "original_label": int(fragment.original_label),
                "energy_mev": float(fragment.energy_mev),
                "truth_vertex_id": int(fragment.truth_vertex_id),
                "truth_t0": float(fragment.truth_t0),
                "truth_purity": float(fragment.truth_purity),
                "miss_cm": float(_line_point_distance(provisional_apex, fragment.start_point, fragment.direction)),
                "angle_penalty": float(angle),
                "centroid": [float(v) for v in np.asarray(fragment.centroid, dtype=np.float64).tolist()],
            }
        )
    return rows


def _scan_best_t0_for_block(
    *,
    predicted_block: np.ndarray,
    base_block: np.ndarray,
    actual_block: np.ndarray,
    std_block: np.ndarray,
    candidate_t0s: list[int],
    local_refine_radius_ticks: int = 5,
) -> tuple[int, dict[int, float]]:
    scan_t0s = sorted(set(int(v) for v in candidate_t0s))
    best_t0 = int(scan_t0s[0])
    best_score = None
    scores: dict[int, float] = {}

    def _score_one(t0: int) -> float:
        model = np.clip(
            np.asarray(base_block, dtype=np.float32)
            + timeinterpolation(np.asarray(predicted_block, dtype=np.float32), shift=float(t0), baseline=0.0).astype(np.float32),
            None,
            ADC_CLIP,
        )
        return float(
            np.mean(
                (model - np.asarray(actual_block, dtype=np.float32)) ** 2
                / np.maximum(np.asarray(std_block, dtype=np.float32), 1e-6)
            )
        )

    for t0 in scan_t0s:
        score = _score_one(int(t0))
        scores[int(t0)] = float(score)
        if best_score is None or score < best_score:
            best_score = float(score)
            best_t0 = int(t0)

    if int(local_refine_radius_ticks) > 0:
        lo = max(0, int(best_t0) - int(local_refine_radius_ticks))
        hi = min(800, int(best_t0) + int(local_refine_radius_ticks))
        for t0 in range(int(lo), int(hi) + 1):
            if int(t0) in scores:
                continue
            score = _score_one(int(t0))
            scores[int(t0)] = float(score)
            if best_score is None or score < best_score:
                best_score = float(score)
                best_t0 = int(t0)

    return int(best_t0), dict(scores)


def _nearest_candidate_t0(raw_t0: float, candidate_t0s: list[int], *, max_snap_ticks: float = 15.0) -> int | None:
    if not np.isfinite(raw_t0) or len(candidate_t0s) == 0:
        return None
    nearest = int(min([int(v) for v in candidate_t0s], key=lambda t0: abs(float(raw_t0) - float(t0))))
    if abs(float(raw_t0) - float(nearest)) > float(max_snap_ticks):
        return None
    return int(nearest)


def _dominant_hint_t0_for_fragment_ids(
    *,
    fragment_ids: list[int],
    fragment_map: dict[int, object],
    hit_timestamps_hint: np.ndarray | None,
    energies: np.ndarray,
    candidate_t0s: list[int],
    max_snap_ticks: float = 15.0,
) -> tuple[int | None, float, float]:
    if hit_timestamps_hint is None:
        return None, 0.0, 0.0
    hint = np.asarray(hit_timestamps_hint, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    energy_by_candidate: dict[int, float] = {}
    total_hint_energy = 0.0
    for fragment_id in [int(v) for v in fragment_ids]:
        fragment = fragment_map.get(int(fragment_id))
        if fragment is None:
            continue
        local_idx = np.asarray(fragment.hit_indices_local, dtype=np.int32)
        local_hint = np.asarray(hint[local_idx], dtype=np.float64)
        local_energy = np.asarray(energies[local_idx], dtype=np.float64)
        valid = np.isfinite(local_hint) & np.isfinite(local_energy) & (local_energy > 0.0)
        if not np.any(valid):
            continue
        snapped_t0 = _nearest_candidate_t0(
            float(np.median(local_hint[valid])),
            [int(v) for v in candidate_t0s],
            max_snap_ticks=float(max_snap_ticks),
        )
        if snapped_t0 is None:
            continue
        energy = float(np.sum(local_energy[valid]))
        total_hint_energy += float(energy)
        energy_by_candidate[int(snapped_t0)] = float(energy_by_candidate.get(int(snapped_t0), 0.0) + float(energy))
    if len(energy_by_candidate) == 0 or total_hint_energy <= 0.0:
        return None, 0.0, 0.0
    best_t0 = int(max(energy_by_candidate, key=lambda t0: float(energy_by_candidate[int(t0)])))
    best_energy = float(energy_by_candidate[int(best_t0)])
    return int(best_t0), float(best_energy / total_hint_energy), float(best_energy)


def _assign_fragment_t0s_v2(
    *,
    result: dict,
    event: dict,
    labels_global: np.ndarray,
    label_info: dict[int, dict],
    target_tpc: int,
) -> tuple[dict[int, int], dict]:
    provisional_apex = None
    if result.get("provisional_apex_xyz") is not None:
        provisional_apex = np.asarray(result["provisional_apex_xyz"], dtype=np.float64)
    if provisional_apex is None:
        provisional_apex, _ = _estimate_neighbor_aware_shower_apex(
            x=event["xset"],
            y=event["yset"],
            z=event["zset"],
            energies=event["Eset"],
            hit_tpc_ids=event["hitTPCid"],
            labels_global=labels_global,
            label_info=label_info,
            target_tpc=int(target_tpc),
        )
    if provisional_apex is None:
        provisional_apex = _estimate_provisional_shower_apex(
            x=event["xset"],
            y=event["yset"],
            z=event["zset"],
            energies=event["Eset"],
            hit_tpc_ids=event["hitTPCid"],
            labels_global=labels_global,
            label_info=label_info,
            target_tpc=int(target_tpc),
        )
    if provisional_apex is None:
        raise RuntimeError(f"Could not estimate a provisional shower apex on TPC {int(target_tpc)}.")

    geometry_rows = _fragment_geometry_rows(result, np.asarray(provisional_apex, dtype=np.float64))
    geometry_by_id = {int(row["fragment_id"]): dict(row) for row in geometry_rows}
    fragment_map = {int(fragment.fragment_id): fragment for fragment in result["fragments"]}

    dominant_shower_labels = [
        int(lbl)
        for lbl in sorted(set(int(fragment.original_label) for fragment in result["fragments"]))
        if str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() == "shower"
    ]
    dominant_shower_label = None
    if dominant_shower_labels:
        dominant_shower_label = max(
            dominant_shower_labels,
            key=lambda lbl: float(
                sum(
                    float(fragment.energy_mev)
                    for fragment in result["fragments"]
                    if int(fragment.original_label) == int(lbl)
                )
            ),
        )

    main_core_ids = [
        int(row["fragment_id"])
        for row in geometry_rows
        if (
            (
                float(row["miss_cm"]) < 20.0
                and float(row["angle_penalty"]) < 0.20
                and float(row["energy_mev"]) > 5.0
            )
            or (dominant_shower_label is not None and int(row["original_label"]) == int(dominant_shower_label))
        )
    ]
    main_core_ids = sorted(set(int(v) for v in main_core_ids))
    if len(main_core_ids) == 0:
        raise RuntimeError("No main-core fragments were identified for the v2 TPC shower specialist.")

    keep_idx = np.asarray(result["keep_channel_indices"], dtype=np.int32)
    track_base = np.asarray(result["track_base_image"][int(target_tpc), keep_idx], dtype=np.float32)
    actual = np.asarray(event["fullLightWaveform"][int(target_tpc), keep_idx], dtype=np.float32)
    std = np.asarray(event["fullLightStd"][int(target_tpc), keep_idx], dtype=np.float32)
    main_block = np.zeros((keep_idx.size, actual.shape[1]), dtype=np.float32)
    for fragment_id in main_core_ids:
        label = int(result["fragment_label_by_id"][int(fragment_id)])
        main_block += np.asarray(result["image_maps"][(int(label), int(target_tpc))], dtype=np.float32)[keep_idx]

    candidate_t0s = [int(v) for v in result["candidate_t0s"]]
    main_t0, main_scores = _scan_best_t0_for_block(
        predicted_block=main_block,
        base_block=track_base,
        actual_block=actual,
        std_block=std,
        candidate_t0s=candidate_t0s,
    )
    main_hint_t0, main_hint_frac, main_hint_energy = _dominant_hint_t0_for_fragment_ids(
        fragment_ids=main_core_ids,
        fragment_map=fragment_map,
        hit_timestamps_hint=event.get("hit_timestamps_hint"),
        energies=np.asarray(event["Eset"], dtype=np.float64),
        candidate_t0s=candidate_t0s,
    )
    if main_hint_t0 is not None and int(main_hint_t0) in main_scores:
        best_main_score = float(min(float(v) for v in main_scores.values()))
        hint_main_score = float(main_scores[int(main_hint_t0)])
        if (
            abs(int(main_hint_t0) - int(main_t0)) >= 20
            and float(main_hint_frac) >= 0.45
            and float(main_hint_energy) >= 40.0
            and float(hint_main_score) <= 1.35 * float(best_main_score)
        ):
            main_t0 = int(main_hint_t0)

    strength_profile = np.asarray(result.get("residual_profile_full", result["residual_profile"]), dtype=np.float64)
    peak_strength = {
        int(t0): float(strength_profile[105 + int(t0)])
        for t0 in candidate_t0s
        if 0 <= 105 + int(t0) < int(strength_profile.shape[0])
    }
    pre_main_peaks = sorted(
        [(int(t0), float(peak_strength[int(t0)])) for t0 in peak_strength if int(t0) < int(main_t0) - 40],
        key=lambda kv: -float(kv[1]),
    )
    if len(pre_main_peaks) == 0:
        raise RuntimeError("No meaningful pre-main residual peaks were found for the v2 TPC shower specialist.")

    detached_rows = [
        dict(row)
        for row in geometry_rows
        if int(row["fragment_id"]) not in set(main_core_ids)
    ]
    detached_centroids = np.asarray([row["centroid"] for row in detached_rows], dtype=np.float64)
    detached_group_labels = DBSCAN(eps=8.0, min_samples=1).fit_predict(detached_centroids).astype(np.int32)
    group_info: list[dict] = []
    for gid in sorted(int(v) for v in np.unique(detached_group_labels)):
        idx = np.flatnonzero(detached_group_labels == int(gid)).astype(np.int32)
        group_rows = [detached_rows[int(i)] for i in idx.tolist()]
        fragment_ids = [int(row["fragment_id"]) for row in group_rows]
        centroid = np.mean(np.asarray([row["centroid"] for row in group_rows], dtype=np.float64), axis=0)
        group_info.append(
            {
                "gid": int(gid),
                "fragment_ids": fragment_ids,
                "energy_mev": float(sum(float(row["energy_mev"]) for row in group_rows)),
                "median_miss_cm": float(np.median(np.asarray([row["miss_cm"] for row in group_rows], dtype=np.float64))),
                "centroid": [float(v) for v in centroid.tolist()],
            }
        )

    def _group_predicted_block(fragment_ids: list[int]) -> np.ndarray:
        block = np.zeros((keep_idx.size, actual.shape[1]), dtype=np.float32)
        for fragment_id in fragment_ids:
            label = int(result["fragment_label_by_id"][int(fragment_id)])
            block += np.asarray(result["image_maps"][(int(label), int(target_tpc))], dtype=np.float32)[keep_idx]
        return np.asarray(block, dtype=np.float32)

    group_scores_by_t0: dict[int, dict[int, float]] = {}
    group_best_t0: dict[int, int] = {}
    group_hint_t0: dict[int, int | None] = {}
    group_hint_fraction: dict[int, float] = {}
    group_hint_energy: dict[int, float] = {}
    for group in group_info:
        best_t0, scores = _scan_best_t0_for_block(
            predicted_block=_group_predicted_block(group["fragment_ids"]),
            base_block=track_base,
            actual_block=actual,
            std_block=std,
            candidate_t0s=candidate_t0s,
        )
        group_scores_by_t0[int(group["gid"])] = {int(k): float(v) for k, v in scores.items()}
        group_best_t0[int(group["gid"])] = int(best_t0)
        hint_t0, hint_frac, hint_energy = _dominant_hint_t0_for_fragment_ids(
            fragment_ids=[int(v) for v in group["fragment_ids"]],
            fragment_map=fragment_map,
            hit_timestamps_hint=event.get("hit_timestamps_hint"),
            energies=np.asarray(event["Eset"], dtype=np.float64),
            candidate_t0s=candidate_t0s,
        )
        group_hint_t0[int(group["gid"])] = None if hint_t0 is None else int(hint_t0)
        group_hint_fraction[int(group["gid"])] = float(hint_frac)
        group_hint_energy[int(group["gid"])] = float(hint_energy)

    activity_merge_ticks = 30
    activity_groups: list[list[int]] = []
    for t0 in sorted(int(v) for v in candidate_t0s):
        if len(activity_groups) == 0 or abs(int(t0) - int(activity_groups[-1][-1])) > int(activity_merge_ticks):
            activity_groups.append([int(t0)])
        else:
            activity_groups[-1].append(int(t0))

    peak_to_activity_rep: dict[int, int] = {}
    activity_group_strength_by_rep: dict[int, float] = {}
    for members in activity_groups:
        if int(main_t0) in set(int(v) for v in members):
            rep = int(main_t0)
        else:
            rep = int(max(members, key=lambda t0: float(peak_strength.get(int(t0), 0.0))))
        activity_group_strength_by_rep[int(rep)] = float(sum(float(peak_strength.get(int(t0), 0.0)) for t0 in members))
        for t0 in members:
            peak_to_activity_rep[int(t0)] = int(rep)

    detached_group_energy_by_activity_t0: dict[int, float] = {}
    for group in group_info:
        rep_t0 = int(peak_to_activity_rep.get(int(group_best_t0[int(group["gid"])]), int(group_best_t0[int(group["gid"])])))
        detached_group_energy_by_activity_t0[int(rep_t0)] = float(
            detached_group_energy_by_activity_t0.get(int(rep_t0), 0.0) + float(group["energy_mev"])
        )

    late_candidates = [
        dict(group)
        for group in group_info
        if float(group["energy_mev"]) > 50.0
        and float(group["median_miss_cm"]) > 35.0
        and float(group["centroid"][1]) > float(provisional_apex[1]) - 35.0
    ]
    if len(late_candidates) == 0:
        raise RuntimeError("No late displaced fragment group was found for the v2 TPC shower specialist.")
    late_seed = max(late_candidates, key=lambda group: float(group["energy_mev"]))
    late_ids = [
        int(group["gid"])
        for group in group_info
        if float(np.linalg.norm(np.asarray(group["centroid"], dtype=np.float64) - np.asarray(late_seed["centroid"], dtype=np.float64))) < 30.0
    ]

    remaining_groups = [dict(group) for group in group_info if int(group["gid"]) not in set(late_ids)]
    early_candidates = [
        dict(group)
        for group in remaining_groups
        if float(group["energy_mev"]) > 20.0
        and float(group["median_miss_cm"]) > 45.0
        and float(group["centroid"][1]) < float(provisional_apex[1]) - 35.0
    ]
    early_seed = None if len(early_candidates) == 0 else max(early_candidates, key=lambda group: float(group["energy_mev"]))
    early_ids: list[int] = []
    if early_seed is not None and len(pre_main_peaks) > 1:
        early_ids = [
            int(group["gid"])
            for group in remaining_groups
            if float(np.linalg.norm(np.asarray(group["centroid"], dtype=np.float64) - np.asarray(early_seed["centroid"], dtype=np.float64))) < 25.0
        ]

    assigned_t0_by_fragment: dict[int, int] = {int(fragment_id): int(main_t0) for fragment_id in main_core_ids}
    late_t0 = int(pre_main_peaks[0][0])
    early_t0 = int(pre_main_peaks[1][0]) if len(pre_main_peaks) > 1 else int(main_t0)
    total_detached_energy = float(sum(float(group["energy_mev"]) for group in group_info))
    strong_detached_energy_threshold = max(20.0, 0.08 * total_detached_energy)
    strong_activity_strength_threshold = 0.08 * max(float(v) for v in activity_group_strength_by_rep.values())
    selected_activity_t0s = [int(main_t0), int(late_t0)]
    if early_seed is not None and len(pre_main_peaks) > 1:
        selected_activity_t0s.append(int(early_t0))
    for rep_t0, strength in sorted(activity_group_strength_by_rep.items(), key=lambda kv: (-float(kv[1]), int(kv[0]))):
        if int(rep_t0) in set(int(v) for v in selected_activity_t0s):
            continue
        if (
            float(strength) >= float(strong_activity_strength_threshold)
            and float(detached_group_energy_by_activity_t0.get(int(rep_t0), 0.0)) >= float(strong_detached_energy_threshold)
        ):
            selected_activity_t0s.append(int(rep_t0))
    selected_activity_t0s = sorted(set(int(v) for v in selected_activity_t0s))

    for group in group_info:
        candidate_scores = group_scores_by_t0[int(group["gid"])]
        allowed_t0s = [int(v) for v in selected_activity_t0s if int(v) in candidate_scores]
        if len(allowed_t0s) == 0:
            allowed_t0s = [int(main_t0)]
        best_loss_t0 = int(min(allowed_t0s, key=lambda t0: float(candidate_scores[int(t0)])))
        group_t0 = int(best_loss_t0)
        if int(group["gid"]) in set(late_ids):
            group_t0 = int(late_t0)
        elif int(group["gid"]) in set(early_ids):
            group_t0 = int(early_t0)
        hint_t0 = group_hint_t0.get(int(group["gid"]))
        hint_frac = float(group_hint_fraction.get(int(group["gid"]), 0.0))
        hint_energy = float(group_hint_energy.get(int(group["gid"]), 0.0))
        if (
            hint_t0 is not None
            and int(hint_t0) in set(int(v) for v in allowed_t0s)
            and float(hint_frac) >= 0.55
            and float(hint_energy) >= 12.0
        ):
            hint_score = float(candidate_scores[int(hint_t0)])
            chosen_score = float(candidate_scores[int(group_t0)])
            best_score = float(candidate_scores[int(best_loss_t0)])
            if int(hint_t0) != int(group_t0):
                if float(min(float(chosen_score), float(best_score))) > 0.80 * float(hint_score):
                    group_t0 = int(hint_t0)
            elif float(best_score) > 0.80 * float(hint_score):
                group_t0 = int(hint_t0)
        for fragment_id in group["fragment_ids"]:
            assigned_t0_by_fragment[int(fragment_id)] = int(group_t0)

    debug = {
        "provisional_apex_xyz": [float(v) for v in np.asarray(provisional_apex, dtype=np.float64).tolist()],
        "main_core_fragment_ids": [int(v) for v in main_core_ids],
        "main_t0": int(main_t0),
        "main_scores": {int(k): float(v) for k, v in main_scores.items()},
        "peak_strength": {int(k): float(v) for k, v in peak_strength.items()},
        "pre_main_peaks": [(int(k), float(v)) for k, v in pre_main_peaks],
        "activity_groups": [[int(v) for v in members] for members in activity_groups],
        "peak_to_activity_rep": {int(k): int(v) for k, v in peak_to_activity_rep.items()},
        "group_best_t0": {int(k): int(v) for k, v in group_best_t0.items()},
        "group_scores_by_t0": {
            int(gid): {int(k): float(v) for k, v in scores.items()}
            for gid, scores in group_scores_by_t0.items()
        },
        "activity_group_strength_by_rep": {
            int(k): float(v) for k, v in activity_group_strength_by_rep.items()
        },
        "detached_group_energy_by_activity_t0": {
            int(k): float(v) for k, v in detached_group_energy_by_activity_t0.items()
        },
        "selected_activity_t0s": [int(v) for v in selected_activity_t0s],
        "main_hint_t0": None if main_hint_t0 is None else int(main_hint_t0),
        "main_hint_fraction": float(main_hint_frac),
        "main_hint_energy_mev": float(main_hint_energy),
        "strong_detached_energy_threshold_mev": float(strong_detached_energy_threshold),
        "strong_activity_strength_threshold": float(strong_activity_strength_threshold),
        "late_seed_gid": int(late_seed["gid"]),
        "late_ids": [int(v) for v in late_ids],
        "early_seed_gid": None if early_seed is None else int(early_seed["gid"]),
        "early_ids": [int(v) for v in early_ids],
        "group_hint_t0": {
            int(gid): (None if t0 is None else int(t0))
            for gid, t0 in group_hint_t0.items()
        },
        "group_hint_fraction": {
            int(gid): float(val) for gid, val in group_hint_fraction.items()
        },
        "group_hint_energy_mev": {
            int(gid): float(val) for gid, val in group_hint_energy.items()
        },
        "group_info": group_info,
        "fragment_geometry_rows": geometry_rows,
    }
    return assigned_t0_by_fragment, debug


def _build_hit_t0_for_tpc(
    *,
    result: dict,
    event: dict,
    labels_global: np.ndarray,
    target_tpc: int,
    fragment_t0_by_id: dict[int, int],
) -> np.ndarray:
    target_tpc = int(target_tpc)
    hit_t0 = np.full(event["xset"].shape[0], np.nan, dtype=np.float64)
    fragment_map = {int(fragment.fragment_id): fragment for fragment in result["fragments"]}

    for fragment_id, t0 in fragment_t0_by_id.items():
        fragment = fragment_map.get(int(fragment_id))
        if fragment is None:
            continue
        hit_t0[np.asarray(fragment.hit_indices_local, dtype=np.int32)] = float(t0)

    for label, t0 in result["track_t0_by_label"].items():
        mask = (np.asarray(event["hitTPCid"], dtype=np.int32) == int(target_tpc)) & (np.asarray(labels_global, dtype=np.int32) == int(label))
        hit_t0[mask] = float(t0)

    tpc_mask = np.asarray(event["hitTPCid"], dtype=np.int32) == int(target_tpc)
    hit_t0[tpc_mask & ~np.isfinite(hit_t0)] = float(
        np.median(np.asarray([float(v) for v in fragment_t0_by_id.values()], dtype=np.float64))
    )
    return np.asarray(hit_t0, dtype=np.float64)


def _build_fragment_t0_from_hint(
    *,
    result: dict,
    event: dict,
    fragment_t0_fallback: dict[int, int],
    max_snap_ticks: float = 15.0,
) -> dict[int, int]:
    candidate_t0s = [int(v) for v in result["candidate_t0s"]]
    fragment_map = {int(fragment.fragment_id): fragment for fragment in result["fragments"]}
    hint = np.asarray(event.get("hit_timestamps_hint"), dtype=np.float64)
    out: dict[int, int] = {}
    for fragment_id, fragment in fragment_map.items():
        local_idx = np.asarray(fragment.hit_indices_local, dtype=np.int32)
        local_hint = np.asarray(hint[local_idx], dtype=np.float64)
        valid = np.isfinite(local_hint)
        snapped_t0 = None
        if np.any(valid):
            snapped_t0 = _nearest_candidate_t0(
                float(np.median(local_hint[valid])),
                candidate_t0s,
                max_snap_ticks=float(max_snap_ticks),
            )
        if snapped_t0 is None:
            snapped_t0 = int(fragment_t0_fallback.get(int(fragment_id), int(candidate_t0s[0])))
        out[int(fragment_id)] = int(snapped_t0)
    return out


def _local_tpc_assignment_loss(
    *,
    result: dict,
    event: dict,
    target_tpc: int,
    fragment_t0_by_id: dict[int, int],
) -> float:
    target_tpc = int(target_tpc)
    keep_idx = np.asarray(result["keep_channel_indices"], dtype=np.int32)
    track_base = np.asarray(result["track_base_image"][int(target_tpc), keep_idx], dtype=np.float32)
    actual = np.asarray(event["fullLightWaveform"][int(target_tpc), keep_idx], dtype=np.float32)
    std = np.asarray(event["fullLightStd"][int(target_tpc), keep_idx], dtype=np.float32)

    model = np.asarray(track_base, dtype=np.float32).copy()
    for fragment_id, t0 in fragment_t0_by_id.items():
        label = int(result["fragment_label_by_id"][int(fragment_id)])
        block = np.asarray(result["image_maps"][(int(label), int(target_tpc))], dtype=np.float32)[keep_idx]
        model += timeinterpolation(block, shift=float(t0), baseline=0.0).astype(np.float32)
    model = np.clip(model, None, ADC_CLIP)
    return float(np.mean((model - actual) ** 2 / np.maximum(std, 1e-6)))


def run_case_v2(
    *,
    eventid: int,
    tpcid: int,
    data_file: str,
    checkpoint_path: str,
    pulse_path: str,
    outdir: Path,
) -> dict:
    resources = load_v11_plotting_resources(
        data_file=str(data_file),
        checkpoint_path=str(checkpoint_path),
        pulse_path=str(pulse_path),
    )
    try:
        event = _load_event_inputs(resources, int(eventid))
        truth = extract_truth_vertex_and_t0_for_hits(
            resources.h5,
            event["hit_refs"],
            convert_to_matching_ticks=True,
        )
        labels_global, _, label_info, _ = build_global_labels_toolbox(
            event["xset"],
            event["yset"],
            event["zset"],
            event["io_group"],
            lam=1.2,
            rss_threshold=1.5e6,
            iters=800,
            min_inliers=35,
            k_for_scale=8,
            attach_multiplier=1.15,
            seed=0,
            min_length_cm=30.0,
            n_tpcs=70,
            match_dist_tol=5.0,
            match_angle_deg=12.0,
            match_endpoint_dist_tol=40.0,
            match_endpoint_weight=0.45,
            match_angle_weight=0.35,
            match_quality_weight=0.15,
            match_max_tpc_gap=None,
            vertex_eps=10.0,
            vertex_min_samples=3,
            min_tracks_for_shower=3,
            split_track_components=True,
            split_radius_cm=4.0,
            split_min_component_hits=20,
            promote_line_like_leftovers=True,
            rescue_dbscan_eps=4.0,
            rescue_dbscan_min_samples=3,
            rescue_min_hits=15,
            rescue_min_length_cm=20.0,
            rescue_min_linearity=0.88,
            rescue_max_transverse_rms=5.0,
            track_noise_absorption_enable=True,
            track_noise_absorb_radius_scale=1.5,
            track_noise_absorb_min_base_radius_cm=1.2,
            track_noise_absorb_endpoint_margin_cm=4.0,
            leftover_dbscan_eps=4.0,
            leftover_dbscan_min_samples=3,
            return_label_info=True,
            return_debug_info=True,
        )
        saturated_channel_cache, _ = build_saturated_channel_cache_v12(
            event["fullLightWaveform"],
            clip_threshold=60700.0,
            max_clip_ticks=6,
        )
        result = run_local_tpc_shower_specialist(
            x=event["xset"],
            y=event["yset"],
            z=event["zset"],
            energies=event["Eset"],
            hit_tpc_ids=event["hitTPCid"],
            labels_global=labels_global,
            label_info=label_info,
            truth_vertex_id=np.asarray(truth["vertex_id"], dtype=np.int64),
            truth_t0=np.asarray(truth["truth_t0"], dtype=np.float64),
            full_light_waveform=event["fullLightWaveform"],
            full_light_std=event["fullLightStd"],
            model=resources.model,
            template=resources.wvfm_tmpl,
            target_tpc=int(tpcid),
            dbscan_eps_cm=4.0,
            dbscan_min_samples=3,
            seed_min_energy_mev=3.0,
            family_time_merge_ticks=10,
            peak_smooth_width=11,
            peak_min_fraction=0.06,
            w_time=1.0,
            w_miss=1.0,
            w_angle=1.0,
            saturated_channel_cache=saturated_channel_cache,
            hit_timestamps_hint=None,
        )
        fragment_t0_by_id, debug = _assign_fragment_t0s_v2(
            result=result,
            event=event,
            labels_global=np.asarray(labels_global, dtype=np.int32),
            label_info=label_info,
            target_tpc=int(tpcid),
        )
        hit_t0 = _build_hit_t0_for_tpc(
            result=result,
            event=event,
            labels_global=np.asarray(labels_global, dtype=np.int32),
            target_tpc=int(tpcid),
            fragment_t0_by_id=fragment_t0_by_id,
        )

        tpc_mask = np.asarray(event["hitTPCid"], dtype=np.int32) == int(tpcid)
        x_tpc = np.asarray(event["xset"][tpc_mask], dtype=np.float64)
        y_tpc = np.asarray(event["yset"][tpc_mask], dtype=np.float64)
        z_tpc = np.asarray(event["zset"][tpc_mask], dtype=np.float64)
        e_tpc = np.asarray(event["Eset"][tpc_mask], dtype=np.float64)
        t_tpc = np.asarray(hit_t0[tpc_mask], dtype=np.float64)
        cluster_label = group_hits_by_time(t_tpc)

        outdir.mkdir(parents=True, exist_ok=True)
        html_path = outdir / f"TPC_{int(tpcid)}_reco_result_v2.html"
        plot_3d_clusters_with_t0(
            x_tpc,
            y_tpc,
            z_tpc,
            cluster_label,
            t_tpc,
            energies=e_tpc,
            title=f"TPC {int(tpcid)} shower specialist v2 | event {int(eventid)}",
            save_path=str(html_path),
        )

        truth_rows: list[dict] = []
        tpc_truth_mask = tpc_mask & np.isfinite(np.asarray(truth["truth_t0"], dtype=np.float64))
        for truth_vertex in sorted(int(v) for v in np.unique(np.asarray(truth["vertex_id"])[tpc_truth_mask]) if int(v) >= 0):
            mask = tpc_truth_mask & (np.asarray(truth["vertex_id"], dtype=np.int64) == int(truth_vertex))
            total_energy = float(np.sum(np.asarray(event["Eset"], dtype=np.float64)[mask]))
            if total_energy <= 0.0:
                continue
            good_energy = float(
                np.sum(
                    np.asarray(event["Eset"], dtype=np.float64)[
                        mask
                        & np.isfinite(hit_t0)
                        & (np.abs(hit_t0 - np.asarray(truth["truth_t0"], dtype=np.float64)) <= 10.0)
                    ]
                )
            )
            truth_rows.append(
                {
                    "truth_vertex_id": int(truth_vertex),
                    "truth_t0": float(np.median(np.asarray(truth["truth_t0"], dtype=np.float64)[mask])),
                    "energy_mev": float(total_energy),
                    "completeness": float(good_energy / max(total_energy, 1e-9)),
                }
            )

        summary = {
            "eventid": int(eventid),
            "tpcid": int(tpcid),
            "html_path": str(html_path),
            "candidate_t0s": [int(v) for v in result["candidate_t0s"]],
            "track_t0_by_label": {int(k): int(v) for k, v in result["track_t0_by_label"].items()},
            "fragment_t0_by_id": {int(k): int(v) for k, v in fragment_t0_by_id.items()},
            "truth_rows": truth_rows,
            "debug": debug,
        }
        summary_path = outdir / f"TPC_{int(tpcid)}_reco_result_v2_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved {html_path}")
        print(f"Saved {summary_path}")
        return summary
    finally:
        try:
            resources.h5.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="TPC-local shower specialist v2 for staged geometry-plus-time reassignment.")
    parser.add_argument("--eventid", type=int, default=1)
    parser.add_argument("--tpcid", type=int, default=8)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--pulse-path", default=DEFAULT_PULSE_PATH)
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "tpc_shower_specialist_examples_v2"),
    )
    args = parser.parse_args()
    run_case_v2(
        eventid=int(args.eventid),
        tpcid=int(args.tpcid),
        data_file=str(args.data_file),
        checkpoint_path=str(args.checkpoint_path),
        pulse_path=str(args.pulse_path),
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
