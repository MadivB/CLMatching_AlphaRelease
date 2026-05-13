from __future__ import annotations

from typing import Any

import numpy as np

try:
    from cluster_fit import fit_noise_list
    from ML_NDfull_perceiver import process_clusters_to_imageMaps
    from v3_2_global_matching import compute_error_metric, _shift_block
    from var_prediction.inference_ndfl_v2 import select_continuous_side_channels_from_waveform
    from v12_saturation_mask import materialize_support_entry_v12
except ModuleNotFoundError:  # pragma: no cover - notebook import fallback
    from M5p1.cluster_fit import fit_noise_list
    from NewMLSection.ML_NDfull_perceiver import process_clusters_to_imageMaps
    from M5p1.v3_2_global_matching import compute_error_metric, _shift_block
    from NewMLSection.var_prediction.inference_ndfl_v2 import (
        select_continuous_side_channels_from_waveform,
    )
    from M5p1.v12_saturation_mask import materialize_support_entry_v12


def _label_type(label: int, label_info: dict[int, dict[str, Any]] | None) -> str:
    if label_info is None:
        return "cluster"
    return str(label_info.get(int(label), {}).get("type", "cluster"))


def _label_capacity(
    *,
    label: int,
    energy_mev: float,
    label_type: str,
    capacity_fraction_mev: float,
    shower_absorb_max_hits: int,
    huge_cluster_energy_mev: float,
    huge_cluster_absorb_max_hits: int,
) -> int:
    if str(label_type) == "track":
        return 0

    base_cap = int(np.floor(float(capacity_fraction_mev) * max(float(energy_mev), 0.0)))
    if base_cap <= 0:
        return 0

    if str(label_type) in {"shower", "vertex"}:
        return int(min(int(shower_absorb_max_hits), int(base_cap)))

    if float(energy_mev) > float(huge_cluster_energy_mev):
        return int(min(int(huge_cluster_absorb_max_hits), int(base_cap)))

    return int(base_cap)


def _assigned_label_tpcs_and_t0(
    *,
    label: int,
    cluster_to_tpcs: dict[int, list[int]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
) -> tuple[np.ndarray, int | None]:
    assigned_tpcs: list[int] = []
    t0_values: list[float] = []

    for tpc in cluster_to_tpcs.get(int(label), []):
        info = assignment_info.get((int(label), int(tpc)))
        if not info or not bool(info.get("assigned", False)):
            continue
        t0_val = float(info.get("t0", np.nan))
        if not np.isfinite(t0_val):
            continue
        assigned_tpcs.append(int(tpc))
        t0_values.append(float(t0_val))

    if not assigned_tpcs:
        return np.array([], dtype=int), None

    label_t0 = int(np.rint(np.median(np.asarray(t0_values, dtype=np.float32))))
    return np.asarray(sorted(set(assigned_tpcs)), dtype=int), label_t0


def _full_channel_support_entry(
    waveform: np.ndarray,
) -> dict[str, Any]:
    wave = np.asarray(waveform, dtype=np.float32)
    n_channels = int(wave.shape[0])
    channel_indices = np.arange(n_channels, dtype=np.int32)
    channel_labels = [
        f"L{int(ch)}" if int(ch) < 60 else f"R{int(ch) - 60}"
        for ch in channel_indices.tolist()
    ]
    return {
        "channel_indices": channel_indices,
        "channel_labels": channel_labels,
        "channel_selection_spec": [("L", 0, 59), ("R", 0, 59)],
        "selected_fraction": 1.0,
        "dominant_fraction": 1.0,
        "selected_waveform": np.asarray(wave, dtype=np.float32),
    }


def _build_support_for_waveform(
    waveform: np.ndarray,
    *,
    tpcid: int,
    use_support_selection: bool,
    support_light_fraction: float,
    support_max_gap: int,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wave = np.asarray(waveform, dtype=np.float32)
    if not bool(use_support_selection):
        return materialize_support_entry_v12(
            waveform=wave,
            tpcid=int(tpcid),
            base_entry=_full_channel_support_entry(wave),
            saturated_channel_cache=saturated_channel_cache,
        )

    info = select_continuous_side_channels_from_waveform(
        wave,
        light_fraction=float(support_light_fraction),
        max_gap=int(support_max_gap),
    )
    channel_indices = np.asarray(info.get("channel_indices", []), dtype=np.int32)
    if channel_indices.size == 0:
        return materialize_support_entry_v12(
            waveform=wave,
            tpcid=int(tpcid),
            base_entry=_full_channel_support_entry(wave),
            saturated_channel_cache=saturated_channel_cache,
        )

    return materialize_support_entry_v12(
        waveform=wave,
        tpcid=int(tpcid),
        base_entry={
            "channel_indices": channel_indices,
            "channel_labels": list(info.get("channel_labels", [])),
            "channel_selection_spec": list(info.get("channel_selection_spec", [])),
            "selected_fraction": float(info.get("selected_fraction", 1.0)),
            "dominant_fraction": float(info.get("dominant_fraction", 1.0)),
            "selected_waveform": np.asarray(info.get("selected_waveform", wave[channel_indices]), dtype=np.float32),
        },
        saturated_channel_cache=saturated_channel_cache,
    )


def _stack_parent_block(
    image_maps: dict[tuple[int, int], np.ndarray],
    label: int,
    tpcs: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [np.asarray(image_maps[(int(label), int(tpc))], dtype=np.float32) for tpc in np.asarray(tpcs, dtype=int)],
        axis=0,
    )


def _predict_augmented_waveforms(
    *,
    parent_ids: list[int],
    parent_hit_indices: dict[int, np.ndarray],
    candidate_hits_by_parent: dict[int, list[int]],
    parent_tpcs: dict[int, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    hit_tpc_ids: np.ndarray,
    model: Any,
    template: np.ndarray | None,
    target_scale: float,
    batch_size: int,
    raw_clip: tuple[float, float],
    min_prediction_threshold: float | None,
    device_policy: str,
    parent_batch_size: int,
) -> dict[int, dict[int, np.ndarray]]:
    output: dict[int, dict[int, np.ndarray]] = {}
    active_parents = [int(pid) for pid in parent_ids if len(candidate_hits_by_parent.get(int(pid), [])) > 0]
    if not active_parents:
        return output

    temp_offset = 1000000
    for start in range(0, len(active_parents), int(parent_batch_size)):
        batch_parents = active_parents[start : start + int(parent_batch_size)]
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        z_parts: list[np.ndarray] = []
        e_parts: list[np.ndarray] = []
        tpc_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        temp_to_parent: dict[int, int] = {}

        for local_idx, parent in enumerate(batch_parents):
            temp_label = int(temp_offset + start + local_idx)
            temp_to_parent[temp_label] = int(parent)
            hit_idx = np.asarray(parent_hit_indices[int(parent)], dtype=int)
            noisy_idx = np.asarray(candidate_hits_by_parent.get(int(parent), []), dtype=int)
            full_idx = np.concatenate([hit_idx, noisy_idx], axis=0)
            if full_idx.size == 0:
                continue
            x_parts.append(np.asarray(x[full_idx], dtype=np.float32))
            y_parts.append(np.asarray(y[full_idx], dtype=np.float32))
            z_parts.append(np.asarray(z[full_idx], dtype=np.float32))
            e_parts.append(np.asarray(E[full_idx], dtype=np.float32))
            tpc_parts.append(np.asarray(hit_tpc_ids[full_idx], dtype=np.int32))
            label_parts.append(np.full(full_idx.size, temp_label, dtype=np.int32))

        if not x_parts:
            continue

        aug_image_maps, _ = process_clusters_to_imageMaps(
            np.concatenate(x_parts, axis=0),
            np.concatenate(y_parts, axis=0),
            np.concatenate(z_parts, axis=0),
            np.concatenate(e_parts, axis=0),
            np.concatenate(tpc_parts, axis=0),
            np.concatenate(label_parts, axis=0),
            model=model,
            target_scale=float(target_scale),
            template=template,
            include_noise=False,
            batch_size=int(batch_size),
            raw_clip=tuple(raw_clip),
            min_prediction_threshold=min_prediction_threshold,
            device_policy=str(device_policy),
        )

        for temp_label, parent in temp_to_parent.items():
            output[int(parent)] = {}
            for tpc in np.asarray(parent_tpcs[int(parent)], dtype=int):
                wave = aug_image_maps.get((int(temp_label), int(tpc)))
                if wave is not None:
                    output[int(parent)][int(tpc)] = np.asarray(wave, dtype=np.float32)

    return output


def _evaluate_absorption_gain(
    *,
    parent: int,
    tpcs: np.ndarray,
    label_t0: int,
    image_maps: dict[tuple[int, int], np.ndarray],
    augmented_maps: dict[int, np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    support_by_tpc: dict[int, dict[str, Any]],
    adc_clip: float,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> tuple[float, float, float, int]:
    tpcs_arr = np.asarray(tpcs, dtype=int)
    core_block = _stack_parent_block(image_maps, int(parent), tpcs_arr)
    aug_block = np.stack(
        [np.asarray(augmented_maps[int(tpc)], dtype=np.float32) for tpc in tpcs_arr],
        axis=0,
    )

    shifted_core = _shift_block(core_block, int(label_t0))
    shifted_aug = _shift_block(aug_block, int(label_t0))
    base_current = np.asarray(base_image[tpcs_arr], dtype=np.float32)
    base_without = np.clip(base_current - shifted_core, 0.0, None)
    model_core = np.clip(base_without + shifted_core, None, float(adc_clip))
    model_aug = np.clip(base_without + shifted_aug, None, float(adc_clip))

    model_core_sel: list[np.ndarray] = []
    model_aug_sel: list[np.ndarray] = []
    actual_sel: list[np.ndarray] = []
    err_sel: list[np.ndarray] = []
    n_fit_channels = 0

    for block_idx, tpc in enumerate(tpcs_arr.tolist()):
        support = materialize_support_entry_v12(
            waveform=np.asarray(augmented_maps[int(tpc)], dtype=np.float32),
            tpcid=int(tpc),
            base_entry=support_by_tpc.get(int(tpc)),
            saturated_channel_cache=saturated_channel_cache,
        )
        channel_indices = np.asarray(support.get("channel_indices", np.arange(model_core.shape[1])), dtype=np.int32)
        if channel_indices.size == 0:
            continue

        n_fit_channels += int(channel_indices.size)
        model_core_sel.append(np.asarray(model_core[block_idx, channel_indices, :], dtype=np.float32))
        model_aug_sel.append(np.asarray(model_aug[block_idx, channel_indices, :], dtype=np.float32))
        actual_sel.append(np.asarray(full_light_waveform[int(tpc), channel_indices, :], dtype=np.float32))
        err_sel.append(np.asarray(full_light_std[int(tpc), channel_indices, :], dtype=np.float32))

    if n_fit_channels <= 0 or not model_core_sel:
        return (-np.inf, np.inf, np.inf, 0)

    current_error = compute_error_metric(
        np.concatenate(model_core_sel, axis=0),
        np.concatenate(actual_sel, axis=0),
        np.concatenate(err_sel, axis=0),
    )
    candidate_error = compute_error_metric(
        np.concatenate(model_aug_sel, axis=0),
        np.concatenate(actual_sel, axis=0),
        np.concatenate(err_sel, axis=0),
    )
    return (
        float(current_error - candidate_error),
        float(current_error),
        float(candidate_error),
        int(n_fit_channels),
    )


def _remove_claimed_hits_from_state(
    *,
    claimed_hits: np.ndarray,
    absorption_state: dict[str, Any],
) -> None:
    claimed = set(int(hit_idx) for hit_idx in np.asarray(claimed_hits, dtype=np.int32).tolist())
    if not claimed:
        return

    parents = absorption_state.get("parents", {})
    hit_to_parents = absorption_state.get("hit_to_parents", {})

    for hit_idx in claimed:
        for parent in hit_to_parents.get(int(hit_idx), []):
            parent_state = parents.get(int(parent))
            if parent_state is None:
                continue
            old_pairs = list(zip(parent_state.get("candidate_hits", []), parent_state.get("candidate_distances", [])))
            new_pairs = [(int(h), float(d)) for h, d in old_pairs if int(h) != int(hit_idx)]
            if len(new_pairs) == len(old_pairs):
                continue
            parent_state["candidate_hits"] = [int(h) for h, _ in new_pairs]
            parent_state["candidate_distances"] = [float(d) for _, d in new_pairs]


def prepare_leftover_absorption_state_v8(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    cluster_to_tpcs: dict[int, list[int]],
    label_info: dict[int, dict[str, Any]] | None,
    label_energies: dict[int, float],
    expand_frac: float = 0.10,
    capacity_fraction_mev: float = 0.20,
    shower_absorb_max_hits: int = 25,
    huge_cluster_energy_mev: float = 50.0,
    huge_cluster_absorb_max_hits: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_arr = np.asarray(labels_global, dtype=int)
    noise_mask = labels_arr == -1
    n_noise_hits = int(np.count_nonzero(noise_mask))

    if n_noise_hits == 0:
        state = {
            "parents": {},
            "parent_hit_indices": {},
            "hit_to_parents": {},
            "claimed_hits": set(),
        }
        stats = {
            "n_noise_hits": 0,
            "n_candidate_entries": 0,
            "n_eligible_parents": 0,
            "n_parents_with_candidates": 0,
        }
        return state, stats

    entries = fit_noise_list(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        np.asarray(E, dtype=np.float64),
        labels_arr,
        expand_frac=float(expand_frac),
    )

    parents: dict[int, dict[str, Any]] = {}
    parent_hit_indices: dict[int, np.ndarray] = {}

    for label in sorted(int(v) for v in np.unique(labels_arr) if int(v) >= 0):
        label_type = _label_type(int(label), label_info)
        energy_mev = float(label_energies.get(int(label), 0.0))
        capacity = _label_capacity(
            label=int(label),
            energy_mev=float(energy_mev),
            label_type=str(label_type),
            capacity_fraction_mev=float(capacity_fraction_mev),
            shower_absorb_max_hits=int(shower_absorb_max_hits),
            huge_cluster_energy_mev=float(huge_cluster_energy_mev),
            huge_cluster_absorb_max_hits=int(huge_cluster_absorb_max_hits),
        )
        if int(capacity) <= 0:
            continue

        parent_tpcs = sorted(int(tpc) for tpc in cluster_to_tpcs.get(int(label), []))
        if not parent_tpcs:
            continue

        use_support_selection = not (
            len(parent_tpcs) > 1
            or float(energy_mev) > float(huge_cluster_energy_mev)
            or str(label_type) in {"track", "shower", "vertex"}
        )

        parents[int(label)] = {
            "label": int(label),
            "label_type": str(label_type),
            "energy": float(energy_mev),
            "capacity": int(capacity),
            "tpcs": np.asarray(sorted(set(parent_tpcs)), dtype=int),
            "candidate_hits": [],
            "candidate_distances": [],
            "use_support_selection": bool(use_support_selection),
        }
        parent_hit_indices[int(label)] = np.flatnonzero(labels_arr == int(label)).astype(np.int32)

    for noise_idx, parent, dist in entries:
        parent_state = parents.get(int(parent))
        if parent_state is None:
            continue
        if len(parent_state["candidate_hits"]) >= int(parent_state["capacity"]):
            continue
        noise_idx = int(noise_idx)
        if noise_idx in parent_state["candidate_hits"]:
            continue
        if int(hit_tpc_ids[noise_idx]) not in set(int(tpc) for tpc in np.asarray(parent_state["tpcs"], dtype=int).tolist()):
            continue
        parent_state["candidate_hits"].append(int(noise_idx))
        parent_state["candidate_distances"].append(float(dist))

    parents = {
        int(parent): state
        for parent, state in parents.items()
        if len(state["candidate_hits"]) > 0
    }

    hit_to_parents: dict[int, list[int]] = {}
    for parent, state in parents.items():
        for hit_idx in state.get("candidate_hits", []):
            hit_to_parents.setdefault(int(hit_idx), []).append(int(parent))

    state = {
        "parents": parents,
        "parent_hit_indices": parent_hit_indices,
        "hit_to_parents": hit_to_parents,
        "claimed_hits": set(),
    }
    stats = {
        "n_noise_hits": int(n_noise_hits),
        "n_candidate_entries": int(len(entries)),
        "n_eligible_parents": int(len(parent_hit_indices)),
        "n_parents_with_candidates": int(len(parents)),
    }
    return state, stats


def try_absorb_leftovers_for_parent_v8(
    *,
    parent: int,
    absorption_state: dict[str, Any] | None,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    labels_with_leftovers: np.ndarray,
    absorbed_hit_parent: np.ndarray | None = None,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    hit_timestamps: np.ndarray,
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    model: Any,
    template: np.ndarray | None = None,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None = None,
    target_scale: float = 1e-3,
    batch_size: int = 4,
    raw_clip: tuple[float, float] = (0.0, 60780.0),
    min_prediction_threshold: float | None = 100.0,
    device_policy: str = "auto",
    support_light_fraction: float = 0.90,
    support_max_gap: int = 2,
    improvement_eps: float = 0.0,
    adc_clip: float = 60780.0,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if absorption_state is None:
        return None

    parents = absorption_state.get("parents", {})
    parent_state = parents.get(int(parent))
    if parent_state is None:
        return None

    candidate_hits = list(parent_state.get("candidate_hits", []))
    if len(candidate_hits) == 0:
        return None

    assigned_tpcs: list[int] = []
    t0_values: list[float] = []
    for tpc in np.asarray(parent_state.get("tpcs", []), dtype=int).tolist():
        info = assignment_info.get((int(parent), int(tpc)))
        if not info or not bool(info.get("assigned", False)):
            continue
        t0_val = float(info.get("t0", np.nan))
        if not np.isfinite(t0_val):
            continue
        assigned_tpcs.append(int(tpc))
        t0_values.append(float(t0_val))
    if not assigned_tpcs:
        return None

    label_t0 = int(np.rint(np.median(np.asarray(t0_values, dtype=np.float32))))
    assigned_tpcs_arr = np.asarray(sorted(set(assigned_tpcs)), dtype=int)

    predicted = _predict_augmented_waveforms(
        parent_ids=[int(parent)],
        parent_hit_indices=absorption_state.get("parent_hit_indices", {}),
        candidate_hits_by_parent={int(parent): candidate_hits},
        parent_tpcs={int(parent): assigned_tpcs_arr},
        x=x,
        y=y,
        z=z,
        E=E,
        hit_tpc_ids=hit_tpc_ids,
        model=model,
        template=template,
        target_scale=float(target_scale),
        batch_size=int(batch_size),
        raw_clip=tuple(raw_clip),
        min_prediction_threshold=min_prediction_threshold,
        device_policy=str(device_policy),
        parent_batch_size=1,
    )
    aug_maps = predicted.get(int(parent), {})
    if not aug_maps:
        return None

    support_by_tpc: dict[int, dict[str, Any]] = {}
    for tpc in assigned_tpcs_arr.tolist():
        wave = aug_maps.get(int(tpc))
        if wave is None:
            continue
        support_by_tpc[int(tpc)] = _build_support_for_waveform(
            np.asarray(wave, dtype=np.float32),
            tpcid=int(tpc),
            use_support_selection=bool(parent_state.get("use_support_selection", True)),
            support_light_fraction=float(support_light_fraction),
            support_max_gap=int(support_max_gap),
            saturated_channel_cache=saturated_channel_cache,
        )
    if not support_by_tpc:
        return None

    gain, current_error, candidate_error, n_fit_channels = _evaluate_absorption_gain(
        parent=int(parent),
        tpcs=assigned_tpcs_arr,
        label_t0=int(label_t0),
        image_maps=image_maps,
        augmented_maps=aug_maps,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        support_by_tpc=support_by_tpc,
        adc_clip=float(adc_clip),
        saturated_channel_cache=saturated_channel_cache,
    )
    if float(gain) <= float(improvement_eps):
        return None

    accepted_hits = np.asarray(candidate_hits, dtype=np.int32)
    core_block = _stack_parent_block(image_maps, int(parent), assigned_tpcs_arr)
    aug_block = np.stack(
        [np.asarray(aug_maps[int(tpc)], dtype=np.float32) for tpc in assigned_tpcs_arr.tolist()],
        axis=0,
    )
    shifted_core = _shift_block(core_block, int(label_t0))
    shifted_aug = _shift_block(aug_block, int(label_t0))
    base_image[assigned_tpcs_arr] = np.clip(
        np.asarray(base_image[assigned_tpcs_arr], dtype=np.float32) - shifted_core + shifted_aug,
        0.0,
        float(adc_clip),
    ).astype(np.float32)

    hit_timestamps[accepted_hits] = float(label_t0)
    labels_with_leftovers[accepted_hits] = int(parent)
    if absorbed_hit_parent is not None:
        absorbed_hit_parent[accepted_hits] = int(parent)

    hits_by_tpc: dict[int, int] = {}
    for hit_idx in accepted_hits.tolist():
        tpc = int(hit_tpc_ids[int(hit_idx)])
        hits_by_tpc[tpc] = int(hits_by_tpc.get(tpc, 0) + 1)

    for tpc in assigned_tpcs_arr.tolist():
        image_maps[(int(parent), int(tpc))] = np.asarray(aug_maps[int(tpc)], dtype=np.float32)
        info = assignment_info.get((int(parent), int(tpc)), {})
        info["leftover_absorption_applied"] = True
        info["leftover_absorption_gain"] = float(gain)
        info["leftover_absorption_hits_total"] = int(accepted_hits.size)
        info["leftover_absorption_hits_tpc"] = int(hits_by_tpc.get(int(tpc), 0))
        info["leftover_absorption_label_type"] = str(parent_state.get("label_type", "cluster"))
        assignment_info[(int(parent), int(tpc))] = info

        if channel_support_cache is not None and assigned_tpcs_arr.size == 1:
            support_entry = dict(support_by_tpc.get(int(tpc), _full_channel_support_entry(aug_maps[int(tpc)])))
            support_entry["clusterid"] = int(parent)
            support_entry["tpcid"] = int(tpc)
            channel_support_cache[(int(parent), int(tpc))] = support_entry

    _remove_claimed_hits_from_state(claimed_hits=accepted_hits, absorption_state=absorption_state)
    absorption_state.get("claimed_hits", set()).update(int(hit_idx) for hit_idx in accepted_hits.tolist())

    parent_state["candidate_hits"] = []
    parent_state["candidate_distances"] = []

    return {
        "clusterid": int(parent),
        "tpcs": [int(tpc) for tpc in assigned_tpcs_arr.tolist()],
        "t0": int(label_t0),
        "energy": float(parent_state.get("energy", 0.0)),
        "label_type": str(parent_state.get("label_type", "cluster")),
        "gain": float(gain),
        "current_error": float(current_error),
        "candidate_error": float(candidate_error),
        "n_absorbed_hits": int(accepted_hits.size),
        "hits_by_tpc": {int(k): int(v) for k, v in sorted(hits_by_tpc.items())},
        "n_fit_channels": int(n_fit_channels),
        "mode": "v8_leftover_absorption_inline",
    }


def absorb_leftover_hits_v8(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    hit_tpc_ids: np.ndarray,
    labels_global: np.ndarray,
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    hit_timestamps: np.ndarray,
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    cluster_to_tpcs: dict[int, list[int]],
    label_info: dict[int, dict[str, Any]] | None,
    label_energies: dict[int, float],
    model: Any,
    template: np.ndarray | None = None,
    channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None = None,
    target_scale: float = 1e-3,
    batch_size: int = 4,
    raw_clip: tuple[float, float] = (0.0, 60780.0),
    min_prediction_threshold: float | None = 100.0,
    device_policy: str = "auto",
    support_light_fraction: float = 0.90,
    support_max_gap: int = 2,
    expand_frac: float = 0.10,
    capacity_fraction_mev: float = 0.20,
    shower_absorb_max_hits: int = 25,
    huge_cluster_energy_mev: float = 50.0,
    huge_cluster_absorb_max_hits: int = 10,
    improvement_eps: float = 0.0,
    adc_clip: float = 60780.0,
    parent_batch_size: int = 64,
    saturated_channel_cache: dict[str, Any] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], dict[str, Any]] | None,
    dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    labels_with_leftovers = np.asarray(labels_global, dtype=int).copy()
    absorbed_hit_parent = np.full(labels_with_leftovers.shape, -1, dtype=np.int32)

    unassigned_mask = np.asarray(labels_global, dtype=int) == -1
    labels_for_candidates = np.asarray(labels_global, dtype=int).copy()

    entries = fit_noise_list(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        np.asarray(E, dtype=np.float64),
        labels_for_candidates,
        expand_frac=float(expand_frac),
    )

    parent_states: dict[int, dict[str, Any]] = {}
    parent_hit_indices: dict[int, np.ndarray] = {}

    for label in sorted(int(v) for v in np.unique(labels_global) if int(v) >= 0):
        label_type = _label_type(int(label), label_info)
        energy_mev = float(label_energies.get(int(label), 0.0))
        capacity = _label_capacity(
            label=int(label),
            energy_mev=float(energy_mev),
            label_type=str(label_type),
            capacity_fraction_mev=float(capacity_fraction_mev),
            shower_absorb_max_hits=int(shower_absorb_max_hits),
            huge_cluster_energy_mev=float(huge_cluster_energy_mev),
            huge_cluster_absorb_max_hits=int(huge_cluster_absorb_max_hits),
        )
        if int(capacity) <= 0:
            continue

        assigned_tpcs, label_t0 = _assigned_label_tpcs_and_t0(
            label=int(label),
            cluster_to_tpcs=cluster_to_tpcs,
            assignment_info=assignment_info,
        )
        if label_t0 is None or assigned_tpcs.size == 0:
            continue

        use_support_selection = not (
            assigned_tpcs.size > 1
            or float(energy_mev) > float(huge_cluster_energy_mev)
            or str(label_type) in {"track", "shower", "vertex"}
        )

        parent_hit_indices[int(label)] = np.flatnonzero(labels_global == int(label)).astype(np.int32)
        parent_states[int(label)] = {
            "label": int(label),
            "label_type": str(label_type),
            "energy": float(energy_mev),
            "capacity": int(capacity),
            "tpcs": assigned_tpcs,
            "t0": int(label_t0),
            "candidate_hits": [],
            "candidate_distances": [],
            "dirty": True,
            "use_support_selection": bool(use_support_selection),
            "augmented_maps": None,
            "support_by_tpc": None,
        }

    if not parent_states or not entries:
        stats = {
            "n_unassigned_hits": int(np.count_nonzero(unassigned_mask)),
            "n_candidate_entries": int(len(entries)),
            "n_eligible_parents": int(len(parent_states)),
            "n_parents_with_candidates": 0,
            "n_accepted_parents": 0,
            "n_absorbed_hits": 0,
            "accepted_parents": [],
        }
        return (
            base_image,
            hit_timestamps,
            labels_with_leftovers,
            absorbed_hit_parent,
            image_maps,
            channel_support_cache,
            assignment_info,
            [],
            stats,
        )

    for noise_idx, parent, dist in entries:
        state = parent_states.get(int(parent))
        if state is None:
            continue
        if len(state["candidate_hits"]) >= int(state["capacity"]):
            continue
        noise_idx = int(noise_idx)
        if noise_idx in state["candidate_hits"]:
            continue
        if int(hit_tpc_ids[noise_idx]) not in set(int(tpc) for tpc in np.asarray(state["tpcs"], dtype=int).tolist()):
            continue
        state["candidate_hits"].append(int(noise_idx))
        state["candidate_distances"].append(float(dist))

    parent_states = {
        int(parent): state
        for parent, state in parent_states.items()
        if len(state["candidate_hits"]) > 0
    }
    n_parents_with_candidates = int(len(parent_states))

    if not parent_states:
        stats = {
            "n_unassigned_hits": int(np.count_nonzero(unassigned_mask)),
            "n_candidate_entries": int(len(entries)),
            "n_eligible_parents": int(len(parent_hit_indices)),
            "n_parents_with_candidates": 0,
            "n_accepted_parents": 0,
            "n_absorbed_hits": 0,
            "accepted_parents": [],
        }
        return (
            base_image,
            hit_timestamps,
            labels_with_leftovers,
            absorbed_hit_parent,
            image_maps,
            channel_support_cache,
            assignment_info,
            [],
            stats,
        )

    hit_to_parents: dict[int, list[int]] = {}
    for parent, state in parent_states.items():
        for hit_idx in state["candidate_hits"]:
            hit_to_parents.setdefault(int(hit_idx), []).append(int(parent))

    def _refresh_dirty_states(dirty_parents: list[int]) -> None:
        active_dirty = [int(parent) for parent in dirty_parents if int(parent) in parent_states and len(parent_states[int(parent)]["candidate_hits"]) > 0]
        if not active_dirty:
            return

        predicted = _predict_augmented_waveforms(
            parent_ids=active_dirty,
            parent_hit_indices=parent_hit_indices,
            candidate_hits_by_parent={int(parent): list(parent_states[int(parent)]["candidate_hits"]) for parent in active_dirty},
            parent_tpcs={int(parent): np.asarray(parent_states[int(parent)]["tpcs"], dtype=int) for parent in active_dirty},
            x=x,
            y=y,
            z=z,
            E=E,
            hit_tpc_ids=hit_tpc_ids,
            model=model,
            template=template,
            target_scale=float(target_scale),
            batch_size=int(batch_size),
            raw_clip=tuple(raw_clip),
            min_prediction_threshold=min_prediction_threshold,
            device_policy=str(device_policy),
            parent_batch_size=int(parent_batch_size),
        )

        for parent in active_dirty:
            state = parent_states.get(int(parent))
            if state is None:
                continue
            aug_maps = predicted.get(int(parent), {})
            if not aug_maps:
                state["augmented_maps"] = None
                state["support_by_tpc"] = None
                state["dirty"] = False
                continue

            support_by_tpc: dict[int, dict[str, Any]] = {}
            for tpc in np.asarray(state["tpcs"], dtype=int):
                wave = aug_maps.get(int(tpc))
                if wave is None:
                    continue
                support_by_tpc[int(tpc)] = _build_support_for_waveform(
                    np.asarray(wave, dtype=np.float32),
                    tpcid=int(tpc),
                    use_support_selection=bool(state["use_support_selection"]),
                    support_light_fraction=float(support_light_fraction),
                    support_max_gap=int(support_max_gap),
                    saturated_channel_cache=saturated_channel_cache,
                )

            state["augmented_maps"] = {int(tpc): np.asarray(wave, dtype=np.float32) for tpc, wave in aug_maps.items()}
            state["support_by_tpc"] = support_by_tpc
            state["dirty"] = False

    absorption_log: list[dict[str, Any]] = []
    accepted_parents: list[int] = []
    accepted_hits_total = 0

    while True:
        dirty_parents = [int(parent) for parent, state in parent_states.items() if bool(state.get("dirty", False))]
        _refresh_dirty_states(dirty_parents)

        best_parent = None
        best_eval: dict[str, Any] | None = None

        for parent, state in sorted(parent_states.items()):
            candidate_hits = list(state.get("candidate_hits", []))
            aug_maps = state.get("augmented_maps")
            support_by_tpc = state.get("support_by_tpc")
            if not candidate_hits or not aug_maps or not support_by_tpc:
                continue

            gain, current_error, candidate_error, n_fit_channels = _evaluate_absorption_gain(
                parent=int(parent),
                tpcs=np.asarray(state["tpcs"], dtype=int),
                label_t0=int(state["t0"]),
                image_maps=image_maps,
                augmented_maps=aug_maps,
                base_image=base_image,
                full_light_waveform=full_light_waveform,
                full_light_std=full_light_std,
                support_by_tpc=support_by_tpc,
                adc_clip=float(adc_clip),
                saturated_channel_cache=saturated_channel_cache,
            )

            eval_entry = {
                "parent": int(parent),
                "gain": float(gain),
                "current_error": float(current_error),
                "candidate_error": float(candidate_error),
                "n_fit_channels": int(n_fit_channels),
            }
            state["last_eval"] = eval_entry

            if best_eval is None or float(gain) > float(best_eval["gain"]):
                best_parent = int(parent)
                best_eval = eval_entry

        if best_parent is None or best_eval is None:
            break
        if float(best_eval["gain"]) <= float(improvement_eps):
            break

        state = parent_states.pop(int(best_parent))
        accepted_hits = np.asarray(state["candidate_hits"], dtype=np.int32)
        tpcs_arr = np.asarray(state["tpcs"], dtype=int)
        label_t0 = int(state["t0"])

        core_block = _stack_parent_block(image_maps, int(best_parent), tpcs_arr)
        aug_block = np.stack(
            [np.asarray(state["augmented_maps"][int(tpc)], dtype=np.float32) for tpc in tpcs_arr],
            axis=0,
        )
        shifted_core = _shift_block(core_block, int(label_t0))
        shifted_aug = _shift_block(aug_block, int(label_t0))
        base_image[tpcs_arr] = np.clip(
            np.asarray(base_image[tpcs_arr], dtype=np.float32) - shifted_core + shifted_aug,
            0.0,
            float(adc_clip),
        ).astype(np.float32)

        hit_timestamps[accepted_hits] = float(label_t0)
        labels_with_leftovers[accepted_hits] = int(best_parent)
        absorbed_hit_parent[accepted_hits] = int(best_parent)

        hits_by_tpc = {}
        for hit_idx in accepted_hits.tolist():
            tpc = int(hit_tpc_ids[int(hit_idx)])
            hits_by_tpc[tpc] = int(hits_by_tpc.get(tpc, 0) + 1)

        for tpc in tpcs_arr.tolist():
            image_maps[(int(best_parent), int(tpc))] = np.asarray(state["augmented_maps"][int(tpc)], dtype=np.float32)
            info = assignment_info.get((int(best_parent), int(tpc)), {})
            info["leftover_absorption_applied"] = True
            info["leftover_absorption_gain"] = float(best_eval["gain"])
            info["leftover_absorption_hits_total"] = int(accepted_hits.size)
            info["leftover_absorption_hits_tpc"] = int(hits_by_tpc.get(int(tpc), 0))
            info["leftover_absorption_label_type"] = str(state["label_type"])
            assignment_info[(int(best_parent), int(tpc))] = info

            if channel_support_cache is not None and len(tpcs_arr) == 1:
                support_entry = dict(state["support_by_tpc"].get(int(tpc), _full_channel_support_entry(state["augmented_maps"][int(tpc)])))
                support_entry["clusterid"] = int(best_parent)
                support_entry["tpcid"] = int(tpc)
                channel_support_cache[(int(best_parent), int(tpc))] = support_entry

        absorption_log.append(
            {
                "clusterid": int(best_parent),
                "tpcs": [int(tpc) for tpc in tpcs_arr.tolist()],
                "t0": int(label_t0),
                "energy": float(state["energy"]),
                "label_type": str(state["label_type"]),
                "gain": float(best_eval["gain"]),
                "n_absorbed_hits": int(accepted_hits.size),
                "hits_by_tpc": {int(k): int(v) for k, v in sorted(hits_by_tpc.items())},
                "n_fit_channels": int(best_eval["n_fit_channels"]),
                "mode": "v8_leftover_absorption",
            }
        )
        accepted_parents.append(int(best_parent))
        accepted_hits_total += int(accepted_hits.size)

        claimed = set(int(hit_idx) for hit_idx in accepted_hits.tolist())
        for hit_idx in claimed:
            for other_parent in hit_to_parents.get(int(hit_idx), []):
                if int(other_parent) == int(best_parent):
                    continue
                other_state = parent_states.get(int(other_parent))
                if other_state is None:
                    continue
                old_pairs = list(zip(other_state["candidate_hits"], other_state["candidate_distances"]))
                new_hits = [int(h) for h in other_state["candidate_hits"] if int(h) != int(hit_idx)]
                if len(new_hits) != len(other_state["candidate_hits"]):
                    other_state["candidate_hits"] = new_hits
                    other_state["candidate_distances"] = [
                        float(d)
                        for h, d in old_pairs
                        if int(h) != int(hit_idx)
                    ]
                    other_state["dirty"] = True

        for parent in [int(pid) for pid, state in parent_states.items() if len(state.get("candidate_hits", [])) == 0]:
            parent_states.pop(int(parent), None)

    stats = {
        "n_unassigned_hits": int(np.count_nonzero(unassigned_mask)),
        "n_candidate_entries": int(len(entries)),
        "n_eligible_parents": int(len(parent_hit_indices)),
        "n_parents_with_candidates": int(n_parents_with_candidates),
        "n_accepted_parents": int(len(accepted_parents)),
        "n_absorbed_hits": int(accepted_hits_total),
        "accepted_parents": [int(parent) for parent in accepted_parents],
        "support_light_fraction": float(support_light_fraction),
        "support_max_gap": int(support_max_gap),
        "capacity_fraction_mev": float(capacity_fraction_mev),
        "shower_absorb_max_hits": int(shower_absorb_max_hits),
        "huge_cluster_energy_mev": float(huge_cluster_energy_mev),
        "huge_cluster_absorb_max_hits": int(huge_cluster_absorb_max_hits),
        "improvement_eps": float(improvement_eps),
    }

    return (
        base_image,
        hit_timestamps,
        labels_with_leftovers,
        absorbed_hit_parent,
        image_maps,
        channel_support_cache,
        assignment_info,
        absorption_log,
        stats,
    )


__all__ = [
    "prepare_leftover_absorption_state_v8",
    "try_absorb_leftovers_for_parent_v8",
    "absorb_leftover_hits_v8",
]
