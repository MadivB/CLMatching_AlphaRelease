from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plottingTools import group_hits_by_time, plot_3d_clusters_with_t0
from tpc_shower_specialist_v13_3 import run_local_tpc_shower_specialist
from tpc_shower_specialist_v13_3_v2 import (
    _assign_fragment_t0s_v2,
    _build_fragment_t0_from_hint,
    _build_hit_t0_for_tpc,
    _local_tpc_assignment_loss,
)


def find_shower_tpcs_v13(
    *,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    hit_tpc_ids: np.ndarray,
    energies: np.ndarray | None = None,
    min_shower_hits: int = 1,
    min_shower_energy_mev: float = 0.0,
) -> list[int]:
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    if energies is not None:
        energies = np.asarray(energies, dtype=np.float64)

    shower_tpcs: list[int] = []
    for tpcid in sorted(int(v) for v in np.unique(hit_tpc_ids) if int(v) >= 0):
        tpc_mask = hit_tpc_ids == int(tpcid)
        local_labels = sorted(int(v) for v in np.unique(labels_global[tpc_mask]) if int(v) >= 0)
        keep = False
        for label in local_labels:
            if str(label_info.get(int(label), {}).get("type", "cluster")).lower() != "shower":
                continue
            label_mask = tpc_mask & (labels_global == int(label))
            if int(np.count_nonzero(label_mask)) < int(min_shower_hits):
                continue
            if energies is not None and float(np.sum(energies[label_mask])) < float(min_shower_energy_mev):
                continue
            keep = True
            break
        if keep:
            shower_tpcs.append(int(tpcid))
    return shower_tpcs


def _freeze_multi_tpc_tracks_on_target_tpc(
    *,
    rescued_hit_t0: np.ndarray,
    original_hit_timestamps: np.ndarray,
    assignment_info: dict[tuple[int, int], dict[str, Any]] | None,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    hit_tpc_ids: np.ndarray,
    target_tpc: int,
) -> tuple[np.ndarray, list[int]]:
    target_tpc = int(target_tpc)
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    original_hit_timestamps = np.asarray(original_hit_timestamps, dtype=np.float64)
    frozen = np.asarray(rescued_hit_t0, dtype=np.float64).copy()

    frozen_labels: list[int] = []
    local_labels = sorted(int(v) for v in np.unique(labels_global[hit_tpc_ids == target_tpc]) if int(v) >= 0)
    for label in local_labels:
        if str(label_info.get(int(label), {}).get("type", "cluster")).lower() != "track":
            continue
        touched_tpcs = sorted(int(v) for v in np.unique(hit_tpc_ids[labels_global == int(label)]) if int(v) >= 0)
        if len(touched_tpcs) <= 1:
            continue
        mask = (hit_tpc_ids == target_tpc) & (labels_global == int(label))
        if not np.any(mask):
            continue
        override_t0 = None
        if assignment_info is not None:
            entry = assignment_info.get((int(label), target_tpc))
            if entry is not None:
                candidate_t0 = entry.get("t0", np.nan)
                if np.isfinite(candidate_t0):
                    override_t0 = float(candidate_t0)
        if override_t0 is None:
            frozen[mask] = original_hit_timestamps[mask]
        else:
            frozen[mask] = float(override_t0)
        frozen_labels.append(int(label))
    return np.asarray(frozen, dtype=np.float64), frozen_labels


def apply_tpc_shower_rescue_v13(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energies: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    label_info: dict[int, dict[str, Any]],
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    model: Any,
    template: np.ndarray,
    hit_timestamps: np.ndarray,
    assignment_info: dict[tuple[int, int], dict[str, Any]] | None = None,
    saturated_channel_cache: dict[str, Any] | None = None,
    target_tpcs: list[int] | tuple[int, ...] | np.ndarray | None = None,
    only_shower_tpcs: bool = True,
    min_shower_hits: int = 1,
    min_shower_energy_mev: float = 0.0,
    min_nontrack_hits: int = 50,
    dbscan_eps_cm: float = 4.0,
    dbscan_min_samples: int = 3,
    seed_min_energy_mev: float = 3.0,
    family_time_merge_ticks: int = 10,
    peak_smooth_width: int = 11,
    peak_min_fraction: float = 0.06,
    w_time: float = 1.0,
    w_miss: float = 1.0,
    w_angle: float = 1.0,
    truth_vertex_id: np.ndarray | None = None,
    truth_t0: np.ndarray | None = None,
    save_html: bool = False,
    outdir: str | Path | None = None,
    html_title_prefix: str = "TPC shower rescue v13_3",
    min_loss_improvement_frac: float = 0.01,
    max_selected_activity_t0s: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_timestamps_in = np.asarray(hit_timestamps, dtype=np.float64)
    hit_timestamps_out = np.asarray(hit_timestamps_in, dtype=np.float64).copy()

    if truth_vertex_id is None:
        truth_vertex_id = np.full(labels_global.shape[0], -1, dtype=np.int64)
    else:
        truth_vertex_id = np.asarray(truth_vertex_id, dtype=np.int64)
    if truth_t0 is None:
        truth_t0 = np.full(labels_global.shape[0], np.nan, dtype=np.float64)
    else:
        truth_t0 = np.asarray(truth_t0, dtype=np.float64)

    shower_tpcs = find_shower_tpcs_v13(
        labels_global=labels_global,
        label_info=label_info,
        hit_tpc_ids=hit_tpc_ids,
        energies=energies,
        min_shower_hits=min_shower_hits,
        min_shower_energy_mev=min_shower_energy_mev,
    )
    if target_tpcs is None:
        candidate_tpcs = list(shower_tpcs) if only_shower_tpcs else sorted(int(v) for v in np.unique(hit_tpc_ids) if int(v) >= 0)
    else:
        candidate_tpcs = [int(v) for v in np.asarray(target_tpcs, dtype=np.int32).tolist()]
        if only_shower_tpcs:
            candidate_tpcs = [int(v) for v in candidate_tpcs if int(v) in set(shower_tpcs)]

    if save_html:
        outdir = Path(outdir) if outdir is not None else Path("M5p1") / "v13_3_shower_tpc_rescue"
        outdir.mkdir(parents=True, exist_ok=True)

    rescue_log: list[dict[str, Any]] = []
    for tpcid in candidate_tpcs:
        tpc_mask = hit_tpc_ids == int(tpcid)
        local_labels = np.asarray(labels_global[tpc_mask], dtype=np.int32)
        local_types = np.asarray(
            [str(label_info.get(int(lbl), {}).get("type", "cluster")).lower() for lbl in local_labels],
            dtype=object,
        )
        nontrack_mask_tpc = local_types != "track"
        if int(np.count_nonzero(nontrack_mask_tpc)) < int(min_nontrack_hits):
            continue

        result = run_local_tpc_shower_specialist(
            x=x,
            y=y,
            z=z,
            energies=energies,
            hit_tpc_ids=hit_tpc_ids,
            labels_global=labels_global,
            label_info=label_info,
            truth_vertex_id=truth_vertex_id,
            truth_t0=truth_t0,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            model=model,
            template=template,
            target_tpc=int(tpcid),
            dbscan_eps_cm=float(dbscan_eps_cm),
            dbscan_min_samples=int(dbscan_min_samples),
            seed_min_energy_mev=float(seed_min_energy_mev),
            family_time_merge_ticks=int(family_time_merge_ticks),
            peak_smooth_width=int(peak_smooth_width),
            peak_min_fraction=float(peak_min_fraction),
            w_time=float(w_time),
            w_miss=float(w_miss),
            w_angle=float(w_angle),
            saturated_channel_cache=saturated_channel_cache,
            hit_timestamps_hint=hit_timestamps_out,
        )

        try:
            fragment_t0_by_id, debug = _assign_fragment_t0s_v2(
                result=result,
                event={
                    "xset": x,
                    "yset": y,
                    "zset": z,
                    "Eset": energies,
                    "hitTPCid": hit_tpc_ids,
                    "fullLightWaveform": full_light_waveform,
                    "fullLightStd": full_light_std,
                    "hit_timestamps_hint": hit_timestamps_in,
                },
                labels_global=labels_global,
                label_info=label_info,
                target_tpc=int(tpcid),
            )
        except RuntimeError as exc:
            rescue_log.append(
                {
                    "tpcid": int(tpcid),
                    "skipped": True,
                    "reason": str(exc),
                    "n_fragments": int(result.get("n_fragments", 0)),
                    "candidate_t0s": [int(v) for v in result.get("candidate_t0s", [])],
                    "track_t0_by_label": {int(k): int(v) for k, v in result.get("track_t0_by_label", {}).items()},
                }
            )
            continue

        selected_activity_t0s = [int(v) for v in debug.get("selected_activity_t0s", [])]
        if int(max_selected_activity_t0s) > 0 and len(selected_activity_t0s) > int(max_selected_activity_t0s):
            rescue_log.append(
                {
                    "tpcid": int(tpcid),
                    "skipped": True,
                    "reason": "Too many strong non-track activities for the specialized shower rescue.",
                    "n_fragments": int(result.get("n_fragments", 0)),
                    "candidate_t0s": [int(v) for v in result.get("candidate_t0s", [])],
                    "track_t0_by_label": {int(k): int(v) for k, v in result.get("track_t0_by_label", {}).items()},
                    "selected_activity_t0s": [int(v) for v in selected_activity_t0s],
                }
            )
            continue

        baseline_fragment_t0_by_id = _build_fragment_t0_from_hint(
            result=result,
            event={
                "hit_timestamps_hint": hit_timestamps_in,
            },
            fragment_t0_fallback=fragment_t0_by_id,
        )
        baseline_fragment_loss = _local_tpc_assignment_loss(
            result=result,
            event={
                "fullLightWaveform": full_light_waveform,
                "fullLightStd": full_light_std,
            },
            target_tpc=int(tpcid),
            fragment_t0_by_id=baseline_fragment_t0_by_id,
        )
        rescued_fragment_loss = _local_tpc_assignment_loss(
            result=result,
            event={
                "fullLightWaveform": full_light_waveform,
                "fullLightStd": full_light_std,
            },
            target_tpc=int(tpcid),
            fragment_t0_by_id=fragment_t0_by_id,
        )
        if (
            np.isfinite(baseline_fragment_loss)
            and np.isfinite(rescued_fragment_loss)
            and float(rescued_fragment_loss) > (1.0 - float(min_loss_improvement_frac)) * float(baseline_fragment_loss)
        ):
            rescue_log.append(
                {
                    "tpcid": int(tpcid),
                    "skipped": True,
                    "reason": "Rescue did not improve local fragment-level waveform loss over baseline.",
                    "n_fragments": int(result.get("n_fragments", 0)),
                    "candidate_t0s": [int(v) for v in result.get("candidate_t0s", [])],
                    "track_t0_by_label": {int(k): int(v) for k, v in result.get("track_t0_by_label", {}).items()},
                    "baseline_fragment_loss": float(baseline_fragment_loss),
                    "rescued_fragment_loss": float(rescued_fragment_loss),
                    "selected_activity_t0s": [int(v) for v in selected_activity_t0s],
                }
            )
            continue

        rescued_hit_t0 = _build_hit_t0_for_tpc(
            result=result,
            event={
                "xset": x,
                "yset": y,
                "zset": z,
                "Eset": energies,
                "hitTPCid": hit_tpc_ids,
            },
            labels_global=labels_global,
            target_tpc=int(tpcid),
            fragment_t0_by_id=fragment_t0_by_id,
        )
        rescued_hit_t0, frozen_track_labels = _freeze_multi_tpc_tracks_on_target_tpc(
            rescued_hit_t0=rescued_hit_t0,
            original_hit_timestamps=hit_timestamps_in,
            assignment_info=assignment_info,
            labels_global=labels_global,
            label_info=label_info,
            hit_tpc_ids=hit_tpc_ids,
            target_tpc=int(tpcid),
        )
        hit_timestamps_out[tpc_mask] = rescued_hit_t0[tpc_mask]

        log_row = {
            "tpcid": int(tpcid),
            "skipped": False,
            "n_fragments": int(result["n_fragments"]),
            "n_families": int(result["n_families"]),
            "candidate_t0s": [int(v) for v in result["candidate_t0s"]],
            "track_t0_by_label": {int(k): int(v) for k, v in result["track_t0_by_label"].items()},
            "frozen_multi_tpc_track_labels": [int(v) for v in frozen_track_labels],
            "main_t0": int(debug["main_t0"]),
            "main_hint_t0": None if debug.get("main_hint_t0") is None else int(debug["main_hint_t0"]),
            "main_hint_fraction": float(debug.get("main_hint_fraction", 0.0)),
            "baseline_fragment_loss": float(baseline_fragment_loss),
            "rescued_fragment_loss": float(rescued_fragment_loss),
            "pre_main_peaks": [(int(a), float(b)) for a, b in debug["pre_main_peaks"]],
            "selected_activity_t0s": [int(v) for v in selected_activity_t0s],
            "late_ids": [int(v) for v in debug["late_ids"]],
            "early_ids": [int(v) for v in debug["early_ids"]],
        }

        if np.any(np.isfinite(truth_t0[tpc_mask])) and np.any(truth_vertex_id[tpc_mask] >= 0):
            log_row["truth_rows"] = list(result.get("truth_rows", []))

        if save_html:
            t_tpc = np.asarray(hit_timestamps_out[tpc_mask], dtype=np.float64)
            cluster_label = group_hits_by_time(t_tpc, time_window=int(family_time_merge_ticks))
            html_path = Path(outdir) / f"TPC_{int(tpcid)}_reco_result_v13_3.html"
            plot_3d_clusters_with_t0(
                x=np.asarray(x[tpc_mask], dtype=np.float64),
                y=np.asarray(y[tpc_mask], dtype=np.float64),
                z=np.asarray(z[tpc_mask], dtype=np.float64),
                labels=cluster_label,
                t0s=t_tpc,
                energies=np.asarray(energies[tpc_mask], dtype=np.float64),
                title=f"{html_title_prefix} | TPC {int(tpcid)}",
                save_path=str(html_path),
            )
            log_row["html_path"] = str(html_path)

        rescue_log.append(log_row)

    return np.asarray(hit_timestamps_out, dtype=np.float64), rescue_log


__all__ = [
    "apply_tpc_shower_rescue_v13",
    "find_shower_tpcs_v13",
]
