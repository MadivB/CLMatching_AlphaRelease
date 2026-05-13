from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pulse_shapes import timeinterpolation
    from v12_saturation_mask import materialize_support_entry_v12
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.pulse_shapes import timeinterpolation
    from M5p1.v12_saturation_mask import materialize_support_entry_v12


def append_candidate_t0(
    candidate_list: list[int],
    t0: int,
    *,
    min_sep: int = 2,
    max_t0: int = 900,
) -> bool:
    t0 = int(np.clip(np.rint(t0), 0, int(max_t0)))
    for existing in candidate_list:
        if abs(int(existing) - t0) <= int(min_sep):
            return False
    candidate_list.append(t0)
    candidate_list.sort()
    return True


def compute_error_metric(model: np.ndarray, actual: np.ndarray, error_metric: np.ndarray) -> float:
    model = np.asarray(model, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    err = np.maximum(np.asarray(error_metric, dtype=np.float32), 1e-6)
    return float(np.mean((model - actual) ** 2 / err))


def _stack_tpcs(waveforms: np.ndarray, tpcids: np.ndarray) -> np.ndarray:
    tpcids = np.asarray(tpcids, dtype=int)
    return np.asarray(waveforms[tpcids], dtype=np.float32)


def _stack_cluster_images(
    image_maps: dict[tuple[int, int], np.ndarray],
    cluster_id: int,
    tpcids: np.ndarray,
) -> np.ndarray:
    tpcids = np.asarray(tpcids, dtype=int)
    return np.stack(
        [np.asarray(image_maps[(int(cluster_id), int(tpc))], dtype=np.float32) for tpc in tpcids],
        axis=0,
    )


def _shift_block(wave_block: np.ndarray, t0: int, baseline: float = 0.0) -> np.ndarray:
    wave_block = np.asarray(wave_block, dtype=np.float32)
    flat = wave_block.reshape(-1, wave_block.shape[-1])
    shifted = timeinterpolation(flat, shift=float(t0), baseline=baseline).astype(np.float32)
    return shifted.reshape(wave_block.shape)


def _snap_t0_to_local_peak(
    actual_block: np.ndarray,
    base_block: np.ndarray,
    opt_t0: int,
    *,
    pulse_peak_tick: int = 105,
    t0_resolution: int = 5,
    search_range: int = 900,
) -> int:
    expected_peak = int(pulse_peak_tick) + int(opt_t0)
    signal = np.clip(np.asarray(actual_block, dtype=np.float32) - np.asarray(base_block, dtype=np.float32), 0.0, None)
    if signal.ndim == 1:
        signal_1d = np.asarray(signal, dtype=np.float32)
    else:
        sum_axes = tuple(range(signal.ndim - 1))
        signal_1d = np.sum(signal, axis=sum_axes)

    s_start = max(0, expected_peak - int(t0_resolution))
    s_end = min(signal_1d.shape[0], expected_peak + int(t0_resolution) + 1)
    if s_end <= s_start:
        return int(np.clip(opt_t0, 0, search_range))

    local_peak = int(np.argmax(signal_1d[s_start:s_end]))
    actual_peak = s_start + local_peak
    snapped = int(opt_t0) + int(actual_peak - expected_peak)
    return int(np.clip(snapped, 0, search_range))


def _scan_best_shift_multi(
    cluster_block: np.ndarray,
    base_block: np.ndarray,
    actual_block: np.ndarray,
    error_block: np.ndarray,
    *,
    search_range: int = 900,
    adc_clip: float = 60780.0,
    return_curve: bool = False,
) -> tuple[int, float, np.ndarray | None]:
    best_t0 = 0
    best_score = np.inf
    loss_curve = None
    if return_curve:
        loss_curve = np.full(int(search_range) + 1, -1.0, dtype=np.float32)

    for t0 in range(int(search_range) + 1):
        shifted = _shift_block(cluster_block, int(t0))
        candidate_model = np.clip(base_block + shifted, None, adc_clip)
        score = compute_error_metric(candidate_model, actual_block, error_block)
        if loss_curve is not None:
            loss_curve[int(t0)] = float(score)
        if score < best_score:
            best_score = float(score)
            best_t0 = int(t0)

    return int(best_t0), float(best_score), loss_curve


def _build_scan_loss_entry(
    *,
    clusterid: int,
    stage: str,
    mode: str,
    tpcs: np.ndarray | list[int],
    energy: float,
    best_t0: int | None,
    assigned: bool,
    search_range: int,
    loss_curve: np.ndarray | None = None,
    best_t0_scan: int | None = None,
) -> dict[str, Any]:
    tpcs_list = [int(tpc) for tpc in np.asarray(tpcs, dtype=int).tolist()]
    entry: dict[str, Any] = {
        "clusterid": int(clusterid),
        "stage": str(stage),
        "mode": str(mode),
        "tpcs": tpcs_list,
        "energy": float(energy),
        "assigned": bool(assigned),
        "best_t0": None if best_t0 is None else int(best_t0),
        "best_t0_scan": None if best_t0_scan is None else int(best_t0_scan),
        "scan_performed": bool(loss_curve is not None),
        "t0_grid": -1,
        "loss_curve": -1.0,
    }
    if loss_curve is not None:
        entry["t0_grid"] = np.arange(int(search_range) + 1, dtype=np.int32)
        entry["loss_curve"] = np.asarray(loss_curve, dtype=np.float32)
    return entry


def _loss_matrix_single_tpc(
    image_maps: dict[tuple[int, int], np.ndarray],
    actual_block: np.ndarray,
    base_block: np.ndarray,
    error_block: np.ndarray,
    *,
    tpcid: int,
    clusters: np.ndarray,
    placed_mask: np.ndarray,
    t0_candidates: list[int],
    adc_clip: float = 60780.0,
) -> tuple[np.ndarray, np.ndarray]:
    remaining_clusters = np.asarray(clusters[~placed_mask], dtype=int)
    n_remaining = int(len(remaining_clusters))
    n_candidates = int(len(t0_candidates))

    loss_matrix = np.full((n_remaining, n_candidates), np.inf, dtype=np.float32)
    if n_remaining == 0 or n_candidates == 0:
        return loss_matrix, remaining_clusters

    cluster_images = np.stack(
        [np.asarray(image_maps[(int(cid), int(tpcid))], dtype=np.float32) for cid in remaining_clusters],
        axis=0,
    )
    param_batch = cluster_images.reshape(-1, cluster_images.shape[-1])
    norm_factor = float(cluster_images.shape[1] * cluster_images.shape[2])
    safe_error = np.maximum(np.asarray(error_block, dtype=np.float32), 1e-6)

    for j, t0 in enumerate(t0_candidates):
        shifted_flat = timeinterpolation(param_batch, shift=float(t0), baseline=0.0).astype(np.float32)
        shifted = shifted_flat.reshape(cluster_images.shape)
        model = np.clip(shifted + base_block[None, :, :], None, adc_clip)
        diff = model - actual_block[None, :, :]
        chi2 = (diff**2 / safe_error[None, :, :]).sum(axis=(1, 2))
        loss_matrix[:, j] = chi2 / norm_factor

    return loss_matrix, remaining_clusters


def _mark_cluster_unassigned(
    clusterid: int,
    tpcs: np.ndarray,
    *,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energy: float,
    mode: str,
    stage: str,
    candidate_t0: int | None = None,
    candidate_error: float | None = None,
    improvement: float | None = None,
) -> None:
    hit_timestamps[labels_global == int(clusterid)] = np.nan
    for tpc in np.asarray(tpcs, dtype=int):
        if int(clusterid) not in unassigned_by_tpc[int(tpc)]:
            unassigned_by_tpc[int(tpc)].append(int(clusterid))
        info = {
            "stage": str(stage),
            "mode": str(mode),
            "t0": np.nan,
            "energy": float(cluster_energy),
            "assigned": False,
        }
        if candidate_t0 is not None:
            info["suggested_t0"] = float(candidate_t0)
        if candidate_error is not None:
            info["error_after"] = float(candidate_error)
        if improvement is not None:
            info["improvement"] = float(improvement)
        assignment_info[(int(clusterid), int(tpc))] = info


def _assign_cluster_at_t0(
    clusterid: int,
    tpcs: np.ndarray,
    t0: int,
    *,
    cluster_block: np.ndarray,
    base_image: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energy: float,
    mode: str,
    stage: str,
    error_after: float,
    improvement: float,
    adc_clip: float,
    max_t0: int,
) -> None:
    shifted_block = _shift_block(cluster_block, int(t0))
    candidate_model = np.clip(_stack_tpcs(base_image, tpcs) + shifted_block, None, adc_clip)
    base_image[np.asarray(tpcs, dtype=int)] = candidate_model.astype(np.float32)
    hit_timestamps[labels_global == int(clusterid)] = float(t0)

    for tpc in np.asarray(tpcs, dtype=int):
        append_candidate_t0(t0_candidates[int(tpc)], int(t0), max_t0=max_t0)
        if int(clusterid) in unassigned_by_tpc[int(tpc)]:
            unassigned_by_tpc[int(tpc)] = [
                cid for cid in unassigned_by_tpc[int(tpc)] if int(cid) != int(clusterid)
            ]
        assignment_info[(int(clusterid), int(tpc))] = {
            "stage": str(stage),
            "mode": str(mode),
            "t0": float(t0),
            "energy": float(cluster_energy),
            "assigned": True,
            "error_after": float(error_after),
            "improvement": float(improvement),
        }


def _full_scan_assign(
    clusterid: int,
    tpcs: np.ndarray,
    *,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energy: float,
    stage: str,
    accepted_mode: str,
    forced_mode: str | None,
    rejected_mode: str,
    assignment_improvement_eps: float,
    search_range: int,
    adc_clip: float,
    t0_resolution: int,
    pulse_peak_tick: int,
    collect_scan_losses: bool,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None = None,
    channel_saturation_cache: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
    tpcs = np.asarray(tpcs, dtype=int)
    cluster_block = _stack_cluster_images(image_maps, int(clusterid), tpcs)

    fit_cluster_blocks: list[np.ndarray] = []
    fit_base_blocks: list[np.ndarray] = []
    fit_actual_blocks: list[np.ndarray] = []
    fit_error_blocks: list[np.ndarray] = []
    n_fit_channels_total = 0

    for block_idx, tpcid in enumerate(tpcs.tolist()):
        support_entry = materialize_support_entry_v12(
            waveform=np.asarray(cluster_block[block_idx], dtype=np.float32),
            tpcid=int(tpcid),
            base_entry=None if channel_support_cache is None else channel_support_cache.get((int(clusterid), int(tpcid))),
            saturated_channel_cache=channel_saturation_cache,
        )
        channel_indices = np.asarray(support_entry["channel_indices"], dtype=np.int32)
        if channel_indices.size == 0:
            continue
        fit_cluster_blocks.append(np.asarray(cluster_block[block_idx, channel_indices, :], dtype=np.float32))
        fit_base_blocks.append(np.asarray(base_image[int(tpcid), channel_indices, :], dtype=np.float32))
        fit_actual_blocks.append(np.asarray(full_light_waveform[int(tpcid), channel_indices, :], dtype=np.float32))
        fit_error_blocks.append(np.asarray(full_light_std[int(tpcid), channel_indices, :], dtype=np.float32))
        n_fit_channels_total += int(channel_indices.size)

    if not fit_cluster_blocks:
        best_t0 = 0
        current_error = np.inf
        candidate_error = np.inf
        improvement = -np.inf
        best_t0_scan = 0
        scan_curve = None
        accepted = False
        forced = False
    else:
        cluster_fit = np.concatenate(fit_cluster_blocks, axis=0)
        base_block = np.concatenate(fit_base_blocks, axis=0)
        actual_block = np.concatenate(fit_actual_blocks, axis=0)
        error_block = np.concatenate(fit_error_blocks, axis=0)

        current_error = compute_error_metric(base_block, actual_block, error_block)
        best_t0_scan, _, scan_curve = _scan_best_shift_multi(
            cluster_fit,
            base_block,
            actual_block,
            error_block,
            search_range=search_range,
            adc_clip=adc_clip,
            return_curve=collect_scan_losses,
        )
        best_t0 = _snap_t0_to_local_peak(
            actual_block,
            base_block,
            best_t0_scan,
            pulse_peak_tick=pulse_peak_tick,
            t0_resolution=t0_resolution,
            search_range=search_range,
        )
        shifted_block = _shift_block(cluster_fit, int(best_t0))
        candidate_model = np.clip(base_block + shifted_block, None, adc_clip)
        candidate_error = compute_error_metric(candidate_model, actual_block, error_block)
        improvement = float(current_error - candidate_error)

        forced = bool(forced_mode is not None and improvement <= float(assignment_improvement_eps))
        accepted = bool(improvement > float(assignment_improvement_eps) or forced)

    if accepted:
        final_mode = str(forced_mode if forced else accepted_mode)
        _assign_cluster_at_t0(
            clusterid,
            tpcs,
            int(best_t0),
            cluster_block=cluster_block,
            base_image=base_image,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energy=cluster_energy,
            mode=final_mode,
            stage=stage,
            error_after=float(candidate_error),
            improvement=float(improvement),
            adc_clip=adc_clip,
            max_t0=search_range,
        )
        for tpc in np.asarray(tpcs, dtype=int):
            assignment_info[(int(clusterid), int(tpc))]["n_fit_channels"] = int(n_fit_channels_total)
    else:
        _mark_cluster_unassigned(
            clusterid,
            tpcs,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energy=cluster_energy,
            mode=rejected_mode,
            stage=stage,
            candidate_t0=int(best_t0),
            candidate_error=float(candidate_error),
            improvement=float(improvement),
        )
        for tpc in np.asarray(tpcs, dtype=int):
            assignment_info[(int(clusterid), int(tpc))]["n_fit_channels"] = int(n_fit_channels_total)

    log = {
        "clusterid": int(clusterid),
        "tpcs": tpcs.tolist(),
        "energy": float(cluster_energy),
        "assigned": bool(accepted),
        "mode": str(forced_mode if forced else (accepted_mode if accepted else rejected_mode)),
        "label": str(stage),
        "t0": int(best_t0),
        "raw_t0": int(best_t0_scan),
        "improvement": float(improvement),
        "n_fit_channels": int(n_fit_channels_total),
    }

    scan_entry = None
    if collect_scan_losses:
        scan_entry = _build_scan_loss_entry(
            clusterid=int(clusterid),
            stage=stage,
            mode=str(log["mode"]),
            tpcs=tpcs,
            energy=float(cluster_energy),
            best_t0=int(best_t0),
            assigned=bool(accepted),
            search_range=search_range,
            loss_curve=scan_curve,
            best_t0_scan=int(best_t0_scan),
        )
        scan_entry["n_fit_channels"] = int(n_fit_channels_total)

    return bool(accepted), log, scan_entry


def assign_small_clusters_v32(
    cluster_labels: list[int],
    cluster_to_tpcs: dict[int, list[int]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    labels_global: np.ndarray,
    hit_timestamps: np.ndarray,
    t0_candidates: list[list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    unassigned_by_tpc: dict[int, list[int]],
    cluster_energies: dict[int, float],
    *,
    seed_full_scan_energy_mev: float = 10.0,
    assignment_improvement_eps: float = 1e-4,
    greedy_loss_increase_tolerance: float = 0.0,
    search_range: int = 900,
    adc_clip: float = 60780.0,
    t0_resolution: int = 5,
    pulse_peak_tick: int = 105,
    collect_scan_losses: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[int]],
    dict[tuple[int, int], dict[str, Any]],
    dict[int, list[int]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    assignment_log: list[dict[str, Any]] = []
    scan_loss_dict: dict[int, dict[str, Any]] = {}

    active_cluster_tpcs: dict[int, np.ndarray] = {}
    for clusterid in cluster_labels:
        tpcs = sorted(
            {
                int(tpc)
                for tpc in cluster_to_tpcs.get(int(clusterid), [])
                if int(tpc) < int(base_image.shape[0]) and (int(clusterid), int(tpc)) in image_maps
            }
        )
        if tpcs:
            active_cluster_tpcs[int(clusterid)] = np.asarray(tpcs, dtype=int)

    multi_tpc_clusters = sorted(
        [cid for cid, tpcs in active_cluster_tpcs.items() if len(tpcs) > 1],
        key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
        reverse=True,
    )

    single_tpc_by_tpc: dict[int, list[int]] = {}
    for clusterid, tpcs in active_cluster_tpcs.items():
        if len(tpcs) == 1:
            single_tpc_by_tpc.setdefault(int(tpcs[0]), []).append(int(clusterid))

    assigned_clusters: set[int] = set()
    stalled_tpcs: list[int] = []
    seed_clusters: list[int] = []
    greedy_clusters: list[int] = []
    greedy_tolerated_clusters: list[int] = []
    fallback_assigned_clusters: list[int] = []
    fallback_unassigned_clusters: list[int] = []

    for clusterid in multi_tpc_clusters:
        tpcs = active_cluster_tpcs[int(clusterid)]
        cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
        accepted, log, scan_entry = _full_scan_assign(
            int(clusterid),
            tpcs,
            image_maps=image_maps,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            labels_global=labels_global,
            hit_timestamps=hit_timestamps,
            t0_candidates=t0_candidates,
            assignment_info=assignment_info,
            unassigned_by_tpc=unassigned_by_tpc,
            cluster_energy=cluster_energy,
            stage="multi_tpc_full_scan",
            accepted_mode="multi_tpc_full_scan",
            forced_mode="multi_tpc_full_scan_forced",
            rejected_mode="multi_tpc_unassigned",
            assignment_improvement_eps=assignment_improvement_eps,
            search_range=search_range,
            adc_clip=adc_clip,
            t0_resolution=t0_resolution,
            pulse_peak_tick=pulse_peak_tick,
            collect_scan_losses=collect_scan_losses,
        )
        assignment_log.append(log)
        if scan_entry is not None:
            scan_loss_dict[int(clusterid)] = scan_entry
        if accepted:
            assigned_clusters.add(int(clusterid))

    for tpcid in sorted(single_tpc_by_tpc):
        clusters_here = np.asarray(
            sorted(
                single_tpc_by_tpc.get(int(tpcid), []),
                key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
                reverse=True,
            ),
            dtype=int,
        )
        if len(clusters_here) == 0:
            continue

        high_energy_clusters = [
            int(cid)
            for cid in clusters_here
            if int(cid) not in assigned_clusters
            and float(cluster_energies.get(int(cid), 0.0)) >= float(seed_full_scan_energy_mev)
        ]
        for clusterid in high_energy_clusters:
            tpcs = active_cluster_tpcs[int(clusterid)]
            cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
            accepted, log, scan_entry = _full_scan_assign(
                int(clusterid),
                tpcs,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                t0_candidates=t0_candidates,
                assignment_info=assignment_info,
                unassigned_by_tpc=unassigned_by_tpc,
                cluster_energy=cluster_energy,
                stage="seed_full_scan",
                accepted_mode="seed_full_scan",
                forced_mode="seed_full_scan_forced",
                rejected_mode="seed_full_scan_unassigned",
                assignment_improvement_eps=assignment_improvement_eps,
                search_range=search_range,
                adc_clip=adc_clip,
                t0_resolution=t0_resolution,
                pulse_peak_tick=pulse_peak_tick,
                collect_scan_losses=collect_scan_losses,
            )
            assignment_log.append(log)
            if scan_entry is not None:
                scan_loss_dict[int(clusterid)] = scan_entry
            if accepted:
                assigned_clusters.add(int(clusterid))
                seed_clusters.append(int(clusterid))

        remaining_clusters = np.asarray(
            [int(cid) for cid in clusters_here if int(cid) not in assigned_clusters],
            dtype=int,
        )
        if len(remaining_clusters) == 0:
            continue

        tpc_block_base = np.asarray(base_image[int(tpcid)], dtype=np.float32)
        tpc_block_actual = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
        tpc_block_error = np.asarray(full_light_std[int(tpcid)], dtype=np.float32)
        current_error = compute_error_metric(tpc_block_base, tpc_block_actual, tpc_block_error)
        frozen_candidates = sorted(int(t0) for t0 in t0_candidates[int(tpcid)])
        placed_mask = np.zeros(len(remaining_clusters), dtype=bool)

        if len(frozen_candidates) > 0:
            while True:
                loss_matrix, remaining_now = _loss_matrix_single_tpc(
                    image_maps,
                    tpc_block_actual,
                    tpc_block_base,
                    tpc_block_error,
                    tpcid=int(tpcid),
                    clusters=remaining_clusters,
                    placed_mask=placed_mask,
                    t0_candidates=frozen_candidates,
                    adc_clip=adc_clip,
                )
                if loss_matrix.size == 0 or len(remaining_now) == 0:
                    break

                delta_matrix = loss_matrix - float(current_error)
                min_idx_flat = int(np.argmin(delta_matrix))
                best_cluster_idx = int(min_idx_flat // loss_matrix.shape[1])
                best_t0_idx = int(min_idx_flat % loss_matrix.shape[1])
                best_delta = float(delta_matrix.reshape(-1)[min_idx_flat])

                if not np.isfinite(best_delta) or best_delta > float(greedy_loss_increase_tolerance):
                    stalled_tpcs.append(int(tpcid))
                    break

                clusterid = int(remaining_now[best_cluster_idx])
                opt_t0 = int(frozen_candidates[best_t0_idx])
                orig_idx = int(np.where(remaining_clusters == clusterid)[0][0])
                placed_mask[orig_idx] = True

                tpcs = active_cluster_tpcs[int(clusterid)]
                cluster_block = _stack_cluster_images(image_maps, int(clusterid), tpcs)
                shifted_block = _shift_block(cluster_block, int(opt_t0))
                candidate_model = np.clip(_stack_tpcs(base_image, tpcs) + shifted_block, None, adc_clip)
                candidate_error = compute_error_metric(candidate_model[0], tpc_block_actual, tpc_block_error)
                improvement = float(current_error - candidate_error)
                greedy_mode = "greedy_discrete"
                if improvement <= 0.0:
                    greedy_mode = "greedy_discrete_tolerated"
                    greedy_tolerated_clusters.append(int(clusterid))

                _assign_cluster_at_t0(
                    int(clusterid),
                    tpcs,
                    int(opt_t0),
                    cluster_block=cluster_block,
                    base_image=base_image,
                    labels_global=labels_global,
                    hit_timestamps=hit_timestamps,
                    t0_candidates=t0_candidates,
                    assignment_info=assignment_info,
                    unassigned_by_tpc=unassigned_by_tpc,
                    cluster_energy=float(cluster_energies.get(int(clusterid), 0.0)),
                    mode=greedy_mode,
                    stage="greedy_discrete",
                    error_after=float(candidate_error),
                    improvement=float(improvement),
                    adc_clip=adc_clip,
                    max_t0=search_range,
                )
                assignment_log.append(
                    {
                        "clusterid": int(clusterid),
                        "tpcs": tpcs.tolist(),
                        "energy": float(cluster_energies.get(int(clusterid), 0.0)),
                        "assigned": True,
                        "mode": greedy_mode,
                        "label": "greedy_discrete",
                        "t0": int(opt_t0),
                        "improvement": float(improvement),
                    }
                )
                if collect_scan_losses:
                    scan_loss_dict[int(clusterid)] = _build_scan_loss_entry(
                        clusterid=int(clusterid),
                        stage="greedy_discrete",
                        mode=greedy_mode,
                        tpcs=tpcs,
                        energy=float(cluster_energies.get(int(clusterid), 0.0)),
                        best_t0=int(opt_t0),
                        assigned=True,
                        search_range=search_range,
                        loss_curve=None,
                        best_t0_scan=None,
                    )

                assigned_clusters.add(int(clusterid))
                greedy_clusters.append(int(clusterid))
                tpc_block_base = np.asarray(base_image[int(tpcid)], dtype=np.float32)
                current_error = compute_error_metric(tpc_block_base, tpc_block_actual, tpc_block_error)
        else:
            stalled_tpcs.append(int(tpcid))

        fallback_remaining = [int(cid) for cid in remaining_clusters[~placed_mask] if int(cid) not in assigned_clusters]
        if len(fallback_remaining) == 0:
            continue

        fallback_remaining = sorted(
            fallback_remaining,
            key=lambda cid: (float(cluster_energies.get(int(cid), 0.0)), -int(cid)),
            reverse=True,
        )

        for clusterid in fallback_remaining:
            tpcs = active_cluster_tpcs[int(clusterid)]
            cluster_energy = float(cluster_energies.get(int(clusterid), 0.0))
            accepted, log, scan_entry = _full_scan_assign(
                int(clusterid),
                tpcs,
                image_maps=image_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                labels_global=labels_global,
                hit_timestamps=hit_timestamps,
                t0_candidates=t0_candidates,
                assignment_info=assignment_info,
                unassigned_by_tpc=unassigned_by_tpc,
                cluster_energy=cluster_energy,
                stage="fallback_full_scan",
                accepted_mode="fallback_full_scan",
                forced_mode=None,
                rejected_mode="fallback_unassigned",
                assignment_improvement_eps=assignment_improvement_eps,
                search_range=search_range,
                adc_clip=adc_clip,
                t0_resolution=t0_resolution,
                pulse_peak_tick=pulse_peak_tick,
                collect_scan_losses=collect_scan_losses,
            )
            assignment_log.append(log)
            if scan_entry is not None:
                scan_loss_dict[int(clusterid)] = scan_entry
            if accepted:
                assigned_clusters.add(int(clusterid))
                fallback_assigned_clusters.append(int(clusterid))
            else:
                fallback_unassigned_clusters.append(int(clusterid))

    stage_stats = {
        "multi_tpc_clusters": [int(cid) for cid in multi_tpc_clusters],
        "seed_clusters": seed_clusters,
        "greedy_clusters": greedy_clusters,
        "greedy_tolerated_clusters": sorted(set(int(cid) for cid in greedy_tolerated_clusters)),
        "fallback_assigned_clusters": fallback_assigned_clusters,
        "fallback_unassigned_clusters": fallback_unassigned_clusters,
        "stalled_tpcs": sorted(set(int(tpc) for tpc in stalled_tpcs)),
        "seed_full_scan_energy_mev": float(seed_full_scan_energy_mev),
        "assignment_improvement_eps": float(assignment_improvement_eps),
        "greedy_loss_increase_tolerance": float(greedy_loss_increase_tolerance),
    }

    return (
        base_image,
        hit_timestamps,
        t0_candidates,
        assignment_info,
        unassigned_by_tpc,
        assignment_log,
        scan_loss_dict,
        stage_stats,
    )
