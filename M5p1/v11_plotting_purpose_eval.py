from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from lut import LUT
from pulse_shapes import timeinterpolation
from v8_leftover_absorption import (
    prepare_leftover_absorption_state_v8,
    try_absorb_leftovers_for_parent_v8,
)
from v8_1_flash_seeding import extract_flash_t0_candidates_from_table
from v10_3_support import build_cluster_channel_support_cache_v10_3, stack_cluster_fit_inputs_v10_3
from v10_3_track_rescan import run_track_second_pass_rescan_v10_3
from v10_7_focused_track_swap import (
    build_track_swap_candidates_from_hits_v10_7_focused,
    run_track_overlap_swap_rescue_v10_7_focused,
)
from v11_phased_matching import (
    run_large_cluster_scan_phase_v11,
    run_small_cluster_matrix_phase_v11,
    snapshot_backbone_hits_v11,
    verify_backbone_hits_unchanged_v11,
)
from truth_plotting import (
    extract_event_hit_energy_and_truth_t0 as _extract_event_hit_energy_and_truth_t0_correct,
)

from ML_NDfull_perceiver import (
    DEFAULT_TPC_YAML,
    NUM_TARGETS,
    load_perceiver_model,
    load_tpc_geometries,
    light_tpc_to_charge_tpc,
    process_clusters_to_imageMaps,
)
# Keep the Python pipeline aligned with the current v11_2 notebook clustering
# configuration rather than the older toolbox defaults.
from global_track_clustering_toolbox_v11_2 import build_global_labels_toolbox


N_TPCS = 70
N_CHANNELS = NUM_TARGETS
WVFM_LEN = 1000
ADC_CLIP = 60780.0
T0_RESOLUTION = 5

DEFAULT_DATA_FILE = (
    "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/"
    "run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/"
    "MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000095.FLOW.hdf5"
)
DEFAULT_CHECKPOINT_PATH = (
    "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/"
    "runs/ndfull_run_distributed/checkpoint.pt"
)
DEFAULT_PULSE_PATH = (
    "/global/cfs/cdirs/dune/users/yuxuan/interactLevel/clusteringStudy/"
    "dataDrivenLUTtable/MLApproach/CNNApproach/avg_pulse.npy"
)


@dataclass
class V11PlottingResources:
    data_file: str
    h5: Any
    hits_full: Any
    hits_ref: np.ndarray
    charge_light_ref: np.ndarray
    geom_map: dict[int, Any]
    all_formatted_wvfms: np.ndarray
    model: Any
    wvfm_tmpl: np.ndarray


def _build_sipm_lut(h5: h5py.File) -> tuple[dict[tuple[int, int, int], tuple[int, int]], int]:
    meta = h5["geometry_info/sipm_rel_pos"].attrs["meta"]
    data = h5["geometry_info/sipm_rel_pos/data"]
    sipm_rel_pos = LUT.from_array(meta, data)
    samples = h5["light/wvfm/data"]["samples"]
    nadc = int(samples.shape[1])
    nchan_per_adc = 64
    lut: dict[tuple[int, int, int], tuple[int, int]] = {}

    for adc in range(nadc):
        for ch in range(nchan_per_adc):
            try:
                mapping = sipm_rel_pos[(adc, ch)]
            except Exception:
                continue
            if getattr(mapping, "size", None) == 0:
                continue
            tpc, side, y = mapping[0]
            tpc = int(tpc)
            side = int(side)
            y = int(y)
            if side not in (0, 1) or tpc < 0:
                continue
            lut[(tpc, side, y)] = (adc, ch)

    return lut, nadc


def get_formatted_light_waveforms(h5: h5py.File) -> np.ndarray:
    lut, _ = _build_sipm_lut(h5)
    ntpc = max(t for (t, _, _) in lut) + 1

    tpc_to_channels: dict[int, list[tuple[int, int]]] = {}
    for ltpc in range(ntpc):
        side0 = sorted(
            [(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == ltpc and side == 0],
            key=lambda item: item[0],
        )
        side1 = sorted(
            [(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == ltpc and side == 1],
            key=lambda item: item[0],
        )
        tpc_to_channels[ltpc] = [(adc, ch) for (_, adc, ch) in (side0 + side1)]

    light_wvfms = h5["light/wvfm/data"]["samples"][:]
    baseline = np.mean(light_wvfms[..., :75], axis=-1, keepdims=True)
    light_wvfms_bl = (light_wvfms - baseline).astype(np.float32)

    n_events = light_wvfms.shape[0]
    formatted = np.zeros((n_events, ntpc, N_CHANNELS, WVFM_LEN), dtype=np.float32)
    for ltpc in range(ntpc):
        chans = tpc_to_channels.get(ltpc, [])
        if len(chans) != N_CHANNELS:
            continue
        ctpc = light_tpc_to_charge_tpc(ltpc)
        adc_idx = np.array([adc for (adc, _) in chans], dtype=np.intp)
        ch_idx = np.array([ch for (_, ch) in chans], dtype=np.intp)
        formatted[:, ctpc, :, :] = light_wvfms_bl[:, adc_idx, ch_idx, :]

    return formatted


def load_v11_plotting_resources(
    data_file: str = DEFAULT_DATA_FILE,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    pulse_path: str = DEFAULT_PULSE_PATH,
    device: str | None = None,
) -> V11PlottingResources:
    h5 = h5py.File(data_file, "r")
    hits_dset = "calib_prompt_hits"
    hits_full = h5[f"charge/{hits_dset}/data"]
    hits_ref = h5[f"charge/events/ref/charge/{hits_dset}/ref"]
    charge_light_ref = h5["charge/events/ref/light/events/ref"]
    geom_map = load_tpc_geometries(DEFAULT_TPC_YAML)
    all_formatted_wvfms = get_formatted_light_waveforms(h5)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_perceiver_model(checkpoint_path, device=device)
    wvfm_tmpl = np.load(pulse_path).astype(np.float32) / 999.0

    return V11PlottingResources(
        data_file=str(data_file),
        h5=h5,
        hits_full=hits_full,
        hits_ref=np.asarray(hits_ref[:], dtype=np.int64),
        charge_light_ref=np.asarray(charge_light_ref[:], dtype=np.int64),
        geom_map=geom_map,
        all_formatted_wvfms=np.asarray(all_formatted_wvfms, dtype=np.float32),
        model=model,
        wvfm_tmpl=np.asarray(wvfm_tmpl, dtype=np.float32),
    )


def close_v11_plotting_resources(resources: V11PlottingResources) -> None:
    try:
        resources.h5.close()
    except Exception:
        pass


def charge_tpc_from_io_group(io_group: np.ndarray) -> np.ndarray:
    """Map ND-LAr io_group to charge-TPC id. This is the authoritative hit TPC."""
    io = np.asarray(io_group, dtype=np.int64)
    if np.any(io <= 0):
        bad = np.unique(io[io <= 0]).tolist()
        raise ValueError(f"Invalid io_group values for TPC assignment: {bad[:10]}")
    return ((io - 1) // 2).astype(np.int32)


def assign_hits_to_charge_tpc(x: np.ndarray, y: np.ndarray, z: np.ndarray, geom_map_: dict[int, Any]) -> np.ndarray:
    raise RuntimeError(
        "Geometry-based hit-to-TPC assignment is disabled. Use charge_tpc_from_io_group(io_group) instead."
    )


def max_likelihood_curve_with_base(
    predicted: np.ndarray,
    base: np.ndarray,
    actual: np.ndarray,
    error_metric: np.ndarray,
    search_range: int = 800,
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
        model = np.clip(shifted + base_, None, ADC_CLIP)
        errors[idx] = np.sum((model - act) ** 2 / err) / float(n_ticks)

    return shifts, errors


def fast_stack_images(image_maps: dict[tuple[int, int], np.ndarray], cluster_id: int, tpcids: Iterable[int]) -> np.ndarray:
    tpcids = np.atleast_1d(np.asarray(list(tpcids), dtype=int))
    images = [np.asarray(image_maps[(int(cluster_id), int(tpc))], dtype=np.float32) for tpc in tpcids]
    stacked = np.stack(images, axis=0)
    return stacked.reshape(-1, stacked.shape[-1])


def image_updater(new_image: np.ndarray, base_image: np.ndarray, sorted_tpcs: Iterable[int], num_ch: int = N_CHANNELS) -> np.ndarray:
    sorted_tpcs = np.atleast_1d(np.asarray(list(sorted_tpcs), dtype=int))
    new_image_r = np.asarray(new_image, dtype=np.float32).reshape(len(sorted_tpcs), num_ch, -1)
    base_image[sorted_tpcs] += new_image_r
    return base_image


def append_candidate_t0(candidate_list: list[int], t0: int, min_sep: int = 2) -> bool:
    t0 = int(np.clip(np.rint(t0), 0, 800))
    for existing in candidate_list:
        if abs(int(existing) - t0) <= int(min_sep):
            return False
    candidate_list.append(int(t0))
    candidate_list.sort()
    return True


def extract_event_hit_energy_and_truth_t0(
    h5: h5py.File,
    hits_ref: np.ndarray,
    eventid: int,
    *,
    hits_full: Any | None = None,
    convert_to_matching_ticks: bool = True,
) -> dict[str, np.ndarray]:
    return _extract_event_hit_energy_and_truth_t0_correct(
        h5,
        hits_ref,
        eventid=int(eventid),
        hits_full=hits_full,
        convert_to_matching_ticks=bool(convert_to_matching_ticks),
    )


def _load_event_inputs(resources: V11PlottingResources, ev_id: int) -> dict[str, Any]:
    hit_mask = np.asarray(resources.hits_ref[:, 0] == int(ev_id), dtype=bool)
    hit_refs = np.asarray(resources.hits_ref[hit_mask, 1], dtype=np.int64)
    if hit_refs.size == 0:
        raise RuntimeError(f"No charge hits found for event {ev_id}")

    hits_evt = resources.hits_full[hit_refs]
    xset = np.asarray(hits_evt["x"], dtype=np.float64)
    yset = np.asarray(hits_evt["y"], dtype=np.float64)
    zset = np.asarray(hits_evt["z"], dtype=np.float64)
    eset = np.asarray(hits_evt["E"], dtype=np.float64)
    io_group = np.asarray(hits_evt["io_group"], dtype=np.int64)
    hit_tpcid = charge_tpc_from_io_group(io_group)

    light_refs = np.asarray(resources.charge_light_ref[resources.charge_light_ref[:, 0] == int(ev_id)], dtype=np.int64)
    if light_refs.shape[0] == 0:
        raise RuntimeError(f"No light event found for charge event {ev_id}")
    light_id = int(light_refs[0, 1])
    full_light_waveform = np.asarray(resources.all_formatted_wvfms[light_id], dtype=np.float32).copy()
    full_light_std = np.ones_like(full_light_waveform, dtype=np.float32)

    return {
        "ev_id": int(ev_id),
        "hit_refs": hit_refs,
        "hits_evt": hits_evt,
        "xset": xset,
        "yset": yset,
        "zset": zset,
        "Eset": eset,
        "io_group": io_group,
        "hitTPCid": hit_tpcid,
        "light_id": int(light_id),
        "fullLightWaveform": full_light_waveform,
        "fullLightStd": full_light_std,
    }


def _run_phase1_track_shower(
    *,
    resources: V11PlottingResources,
    xset: np.ndarray,
    yset: np.ndarray,
    zset: np.ndarray,
    Eset: np.ndarray,
    io_group: np.ndarray,
    hitTPCid: np.ndarray,
    fullLightWaveform: np.ndarray,
    fullLightStd: np.ndarray,
    labels_global: np.ndarray,
    split_index: int,
    label_info: dict[int, dict[str, Any]],
    imageMaps: dict[tuple[int, int], np.ndarray],
    cluster_to_tpcs: dict[int, list[int]],
    cluster_channel_support_cache: dict[tuple[int, int], dict[str, Any]] | None,
    verbose: bool = True,
) -> dict[str, Any]:
    nhits = int(xset.shape[0])
    max_charge_tpc = int(fullLightWaveform.shape[0])

    track_shower_labels = list(range(int(split_index)))
    cluster_labels = sorted(int(v) for v in np.unique(labels_global) if int(v) >= int(split_index))
    track_shower_mask = np.isin(labels_global, np.asarray(track_shower_labels, dtype=int))

    baseImage = np.zeros_like(fullLightWaveform, dtype=np.float32)
    t0Candidates = [[] for _ in range(max_charge_tpc)]
    hit_timestamps = np.full(nhits, np.nan, dtype=np.float32)
    protected_track_shower_timestamps = np.full(nhits, np.nan, dtype=np.float32)

    assignment_info: dict[tuple[int, int], dict[str, Any]] = {}
    unassigned_by_tpc = {int(tpc): [] for tpc in range(max_charge_tpc)}
    labels_global_v8 = np.asarray(labels_global, dtype=int).copy()
    v8_absorbed_hit_parent = np.full(nhits, -1, dtype=np.int32)
    cluster_full_scan_loss_dict: dict[int, dict[str, Any]] = {}

    label_energies_all = {
        int(label): float(Eset[labels_global == int(label)].sum())
        for label in np.unique(labels_global)
        if int(label) >= 0
    }

    v8_leftover_state, _ = prepare_leftover_absorption_state_v8(
        x=xset,
        y=yset,
        z=zset,
        E=Eset,
        hit_tpc_ids=hitTPCid,
        labels_global=labels_global,
        cluster_to_tpcs=cluster_to_tpcs,
        label_info=label_info,
        label_energies=label_energies_all,
        expand_frac=0.20,
        capacity_fraction_mev=0.20,
        shower_absorb_max_hits=50,
        huge_cluster_energy_mev=50.0,
        huge_cluster_absorb_max_hits=20,
    )
    v8_absorption_log: list[dict[str, Any]] = []
    leftover_absorption_context = {
        "state": v8_leftover_state,
        "labels_with_leftovers": labels_global_v8,
        "absorbed_hit_parent": v8_absorbed_hit_parent,
        "x": xset,
        "y": yset,
        "z": zset,
        "E": Eset,
        "hit_tpc_ids": hitTPCid,
        "model": resources.model,
        "template": resources.wvfm_tmpl,
        "channel_support_cache": cluster_channel_support_cache,
        "target_scale": 1e-3,
        "batch_size": 4,
        "raw_clip": (0.0, ADC_CLIP),
        "min_prediction_threshold": 100.0,
        "device_policy": "auto",
        "support_light_fraction": 0.90,
        "support_max_gap": 2,
        "improvement_eps": 0.0,
        "absorption_log": v8_absorption_log,
    }

    if verbose:
        print(f"Starting Track/Shower Association ({len(track_shower_labels)} labels) ...")
    t0_start = time.time()

    for clusterid in track_shower_labels:
        sorted_tpcs = np.sort(
            np.asarray(
                [int(tpc) for tpc in cluster_to_tpcs.get(int(clusterid), []) if int(tpc) < max_charge_tpc],
                dtype=int,
            )
        )
        if sorted_tpcs.size == 0:
            continue

        cluster_energy = float(Eset[labels_global == int(clusterid)].sum())
        label_type = str(label_info.get(int(clusterid), {}).get("type", "track")).lower()
        use_support_mask = label_type == "track"

        tpc_image_full = fast_stack_images(imageMaps, int(clusterid), sorted_tpcs)
        tpc_image_fit, tpc_base_fit, tpc_actual_fit, tpc_std_fit, fit_support_meta = stack_cluster_fit_inputs_v10_3(
            clusterid=int(clusterid),
            tpcids=sorted_tpcs,
            image_maps=imageMaps,
            base_image=baseImage,
            full_light_waveform=fullLightWaveform,
            full_light_std=fullLightStd,
            channel_support_cache=cluster_channel_support_cache,
            use_support_mask=use_support_mask,
        )
        fit_meta_by_tpc = {int(item["tpcid"]): dict(item) for item in fit_support_meta}

        shifts, errors = max_likelihood_curve_with_base(
            predicted=tpc_image_fit,
            base=tpc_base_fit,
            actual=tpc_actual_fit,
            error_metric=tpc_std_fit,
            search_range=800,
        )
        raw_t0 = int(shifts[int(np.argmin(errors))])
        newt0 = int(raw_t0)

        expected_peak = 105 + int(newt0)
        signal_1d = np.sum(np.clip(tpc_actual_fit - tpc_base_fit, 0.0, None), axis=0)
        s_start = max(0, expected_peak - T0_RESOLUTION)
        s_end = min(WVFM_LEN, expected_peak + T0_RESOLUTION + 1)
        if s_end > s_start:
            actual_peak = int(s_start + np.argmax(signal_1d[s_start:s_end]))
            newt0 += int(actual_peak - expected_peak)

        if verbose:
            print(
                f"  Label {int(clusterid):4d} | type={label_type:<6} | "
                f"TPCs={sorted_tpcs.tolist()} | t0={int(newt0)} | fit_ch={int(tpc_image_fit.shape[0])}"
            )

        hit_timestamps[labels_global == int(clusterid)] = float(newt0)
        for tpc in sorted_tpcs.tolist():
            append_candidate_t0(t0Candidates[int(tpc)], int(newt0))
            fit_meta = dict(fit_meta_by_tpc.get(int(tpc), {}))
            assignment_info[(int(clusterid), int(tpc))] = {
                "stage": "track",
                "mode": "track_scan",
                "t0": float(newt0),
                "energy": float(cluster_energy),
                "assigned": True,
                "backbone_type": label_type,
                "n_fit_channels": int(fit_meta.get("n_fit_channels", fullLightWaveform.shape[1])),
                "channel_selection_spec": list(fit_meta.get("channel_selection_spec", [])),
                "selected_fraction": float(fit_meta.get("selected_fraction", 1.0)),
                "dominant_fraction": float(fit_meta.get("dominant_fraction", 1.0)),
                "support_mode": str(fit_meta.get("support_mode", "all_channels")),
            }

        new_image = timeinterpolation(tpc_image_full, shift=float(newt0), baseline=0).astype(np.float32)
        baseImage = image_updater(new_image, baseImage, sorted_tpcs)
        baseImage = np.clip(baseImage, None, ADC_CLIP)

        if v8_leftover_state is not None:
            try_absorb_leftovers_for_parent_v8(
                parent=int(clusterid),
                absorption_state=v8_leftover_state,
                x=xset,
                y=yset,
                z=zset,
                E=Eset,
                hit_tpc_ids=hitTPCid,
                labels_global=labels_global,
                labels_with_leftovers=labels_global_v8,
                absorbed_hit_parent=v8_absorbed_hit_parent,
                image_maps=imageMaps,
                base_image=baseImage,
                full_light_waveform=fullLightWaveform,
                full_light_std=fullLightStd,
                hit_timestamps=hit_timestamps,
                assignment_info=assignment_info,
                model=resources.model,
                template=resources.wvfm_tmpl,
                channel_support_cache=cluster_channel_support_cache,
                target_scale=1e-3,
                batch_size=4,
                raw_clip=(0.0, ADC_CLIP),
                min_prediction_threshold=100.0,
                device_policy="auto",
                support_light_fraction=0.90,
                support_max_gap=2,
                improvement_eps=0.0,
                adc_clip=ADC_CLIP,
            )

    if verbose:
        print()
        print("Starting v10_7_focused track second-pass rescan ...")
    (
        baseImage,
        hit_timestamps,
        t0Candidates,
        assignment_info,
        _,
        track_second_pass_log,
        track_second_pass_stats,
    ) = run_track_second_pass_rescan_v10_3(
        track_labels=track_shower_labels,
        cluster_to_tpcs=cluster_to_tpcs,
        image_maps=imageMaps,
        base_image=baseImage,
        full_light_waveform=fullLightWaveform,
        full_light_std=fullLightStd,
        labels_global=labels_global,
        hit_timestamps=hit_timestamps,
        t0_candidates=t0Candidates,
        assignment_info=assignment_info,
        label_info=label_info,
        channel_support_cache=cluster_channel_support_cache,
        cluster_full_scan_loss_dict=cluster_full_scan_loss_dict,
        absorbed_hit_parent=v8_absorbed_hit_parent,
        search_range=800,
        adc_clip=ADC_CLIP,
        t0_resolution=T0_RESOLUTION,
        waveform_len=WVFM_LEN,
        collect_scan_losses=False,
    )
    if verbose:
        for item in track_second_pass_log:
            print(
                f"  Track {int(item['clusterid']):4d} | TPCs={item['tpcs']} | "
                f"old={int(item['old_t0'])} -> raw={int(item['raw_t0'])} -> final={int(item['new_t0'])} | "
                f"dchi={float(item['improvement']):.4f} | fit_ch={int(item['n_fit_channels'])}"
            )
        print("Starting v10_7_focused overlap-aware track swap rescue ...")

    track_swap_candidate_rows, track_swap_candidate_stats = build_track_swap_candidates_from_hits_v10_7_focused(
        track_labels=track_shower_labels,
        labels_global=labels_global,
        hit_tpc_ids=hitTPCid,
        x=xset,
        y=yset,
        z=zset,
        energies=Eset,
        label_info=label_info,
        min_energy_ratio=0.50,
        min_shared_tpcs=1,
        max_tpc_sym_diff=3,
        max_angle_deg=20.0,
        max_shared_yz_dist_cm=35.0,
        max_endpoint_yz_dist_cm=45.0,
    )
    if verbose:
        print(
            f"  Candidate pairs found: {track_swap_candidate_stats['n_candidates']} "
            f"from {track_swap_candidate_stats['n_bucket_pairs']} shared-TPC bucket pairs"
        )

    (
        baseImage,
        hit_timestamps,
        t0Candidates,
        assignment_info,
        track_swap_log,
        track_swap_stats,
    ) = run_track_overlap_swap_rescue_v10_7_focused(
        track_labels=track_shower_labels,
        cluster_to_tpcs=cluster_to_tpcs,
        image_maps=imageMaps,
        base_image=baseImage,
        full_light_waveform=fullLightWaveform,
        full_light_std=fullLightStd,
        labels_global=labels_global,
        hit_tpc_ids=hitTPCid,
        x=xset,
        y=yset,
        z=zset,
        energies=Eset,
        hit_timestamps=hit_timestamps,
        t0_candidates=t0Candidates,
        assignment_info=assignment_info,
        label_info=label_info,
        channel_support_cache=cluster_channel_support_cache,
        protected_track_shower_timestamps=protected_track_shower_timestamps,
        absorbed_hit_parent=v8_absorbed_hit_parent,
        min_energy_ratio=0.50,
        min_shared_tpcs=1,
        max_tpc_sym_diff=3,
        max_angle_deg=20.0,
        max_shared_yz_dist_cm=35.0,
        max_endpoint_yz_dist_cm=45.0,
        min_t0_separation_ticks=8,
        max_passes=8,
        improvement_eps=0.0,
        adc_clip=ADC_CLIP,
        search_range=800,
        lock_swapped_clusters=True,
    )
    if verbose:
        for item in track_swap_log:
            print(
                f"  Tracks {int(item['cluster_a']):4d}/{int(item['cluster_b']):4d} | "
                f"shared={item['shared_tpcs']} | diff={item['sym_diff_tpcs']} | "
                f"{int(item['cluster_a'])}:{int(item['t0_a_old'])}->{int(item['t0_a_new'])} | "
                f"{int(item['cluster_b'])}:{int(item['t0_b_old'])}->{int(item['t0_b_new'])} | "
                f"dchi={float(item['improvement']):.4f}"
            )

    protected_track_shower_timestamps[track_shower_mask] = hit_timestamps[track_shower_mask]
    if verbose:
        print(f"Track/Shower loop done in {time.time()-t0_start:.1f}s")

    return {
        "baseImage": np.asarray(baseImage, dtype=np.float32),
        "t0Candidates": t0Candidates,
        "hit_timestamps": np.asarray(hit_timestamps, dtype=np.float32),
        "protected_track_shower_timestamps": np.asarray(protected_track_shower_timestamps, dtype=np.float32),
        "track_shower_mask": np.asarray(track_shower_mask, dtype=bool),
        "assignment_info": assignment_info,
        "unassigned_by_tpc": unassigned_by_tpc,
        "track_shower_labels": [int(v) for v in track_shower_labels],
        "cluster_labels": [int(v) for v in cluster_labels],
        "v8_absorbed_hit_parent": np.asarray(v8_absorbed_hit_parent, dtype=np.int32),
        "track_second_pass_stats": dict(track_second_pass_stats),
        "track_swap_candidate_stats": dict(track_swap_candidate_stats),
        "track_swap_stats": dict(track_swap_stats),
        "leftover_absorption_context": leftover_absorption_context,
        "v8_absorption_log": list(v8_absorption_log),
    }


def run_v11_pipeline_for_event(
    resources: V11PlottingResources,
    ev_id: int,
    *,
    lam: float = 1.2,
    verbose: bool = True,
    enable_phase23_leftover_absorption: bool = True,
) -> dict[str, Any]:
    event = _load_event_inputs(resources, int(ev_id))
    xset = event["xset"]
    yset = event["yset"]
    zset = event["zset"]
    Eset = event["Eset"]
    io_group = event["io_group"]
    hitTPCid = event["hitTPCid"]
    fullLightWaveform = event["fullLightWaveform"]
    fullLightStd = event["fullLightStd"]
    hit_refs = event["hit_refs"]

    if verbose:
        print(f"Running clustering for event {int(ev_id)} on {len(xset)} hits ...")
    (
        labels_global,
        split_index,
        label_info,
        clustering_debug_toolbox,
    ) = build_global_labels_toolbox(
        xset,
        yset,
        zset,
        io_group,
        lam=float(lam),
        rss_threshold=1.5e6,
        iters=800,
        min_inliers=35,
        k_for_scale=8,
        attach_multiplier=1.15,
        seed=0,
        min_length_cm=30.0,
        n_tpcs=N_TPCS,
        match_dist_tol=4.0,
        match_angle_deg=10.0,
        match_endpoint_dist_tol=25.0,
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
        rescue_min_length_cm=25.0,
        rescue_min_linearity=0.92,
        rescue_max_transverse_rms=3.5,
        track_noise_absorption_enable=True,
        track_noise_absorb_radius_scale=1.5,
        track_noise_absorb_min_base_radius_cm=1.2,
        track_noise_absorb_endpoint_margin_cm=4.0,
        leftover_dbscan_eps=4.0,
        leftover_dbscan_min_samples=3,
        return_label_info=True,
        return_debug_info=True,
    )

    imageMaps, _ = process_clusters_to_imageMaps(
        xset,
        yset,
        zset,
        Eset,
        hitTPCid,
        labels_global,
        model=resources.model,
        template=resources.wvfm_tmpl,
    )

    cluster_to_tpcs: dict[int, list[int]] = {}
    for (clid, tpcid) in imageMaps.keys():
        cluster_to_tpcs.setdefault(int(clid), []).append(int(tpcid))

    cluster_channel_support_cache, cluster_channel_support_summary = build_cluster_channel_support_cache_v10_3(
        imageMaps,
        cluster_to_tpcs,
        label_info,
        split_index=split_index,
        light_fraction=0.90,
        max_gap=2,
    )

    phase1 = _run_phase1_track_shower(
        resources=resources,
        xset=xset,
        yset=yset,
        zset=zset,
        Eset=Eset,
        io_group=io_group,
        hitTPCid=hitTPCid,
        fullLightWaveform=fullLightWaveform,
        fullLightStd=fullLightStd,
        labels_global=labels_global,
        split_index=split_index,
        label_info=label_info,
        imageMaps=imageMaps,
        cluster_to_tpcs=cluster_to_tpcs,
        cluster_channel_support_cache=cluster_channel_support_cache,
        verbose=verbose,
    )

    backbone_snapshot = snapshot_backbone_hits_v11(
        hit_timestamps=phase1["hit_timestamps"],
        labels_global=labels_global,
        v8_absorbed_hit_parent=phase1["v8_absorbed_hit_parent"],
        track_shower_labels=phase1["track_shower_labels"],
    )

    flash_seed_t0s_by_tpc, flash_seed_stats = extract_flash_t0_candidates_from_table(
        h5_file=resources.h5,
        eventid=int(ev_id),
        search_range=800,
        max_new_per_tpc=None,
        t0_resolution=T0_RESOLUTION,
        flash_tick_divisor=16.0,
        flash_tick_offset=-5.0,
        charge_tpc_min=0,
        charge_tpc_max=int(fullLightWaveform.shape[0]) - 1,
    )

    cluster_energies = {
        int(clid): float(Eset[labels_global == int(clid)].sum())
        for clid in phase1["cluster_labels"]
    }

    (
        baseImage,
        hit_timestamps,
        t0Candidates,
        assignment_info,
        unassigned_by_tpc,
        large_cluster_assignment_log,
        _,
        v11_large_phase_stats,
        v11_active_cluster_tpcs,
        v11_iterative_single_tpc,
        v11_pruned_iterative_clusters,
    ) = run_large_cluster_scan_phase_v11(
        cluster_labels=phase1["cluster_labels"],
        cluster_to_tpcs=cluster_to_tpcs,
        image_maps=imageMaps,
        base_image=phase1["baseImage"],
        full_light_waveform=fullLightWaveform,
        full_light_std=fullLightStd,
        channel_support_cache=cluster_channel_support_cache,
        labels_global=labels_global,
        hit_timestamps=phase1["hit_timestamps"],
        t0_candidates=phase1["t0Candidates"],
        assignment_info=phase1["assignment_info"],
        unassigned_by_tpc=phase1["unassigned_by_tpc"],
        cluster_energies=cluster_energies,
        large_cluster_energy_mev=50.0,
        minimum_iterative_energy_mev=0.5,
        search_range=800,
        adc_clip=ADC_CLIP,
        collect_scan_losses=False,
        assignment_improvement_eps=0.0,
        flash_seed_t0s_by_tpc=flash_seed_t0s_by_tpc,
        flash_seed_t0_resolution=T0_RESOLUTION,
        leftover_absorption_context=(
            phase1["leftover_absorption_context"] if bool(enable_phase23_leftover_absorption) else None
        ),
    )
    backbone_verify_after_phase2 = verify_backbone_hits_unchanged_v11(
        backbone_snapshot,
        hit_timestamps=hit_timestamps,
        stage_name="Phase 2 (large-cluster scan)",
    )

    (
        baseImage,
        hit_timestamps,
        t0Candidates,
        assignment_info,
        unassigned_by_tpc,
        small_cluster_assignment_log,
        _,
        v11_small_phase_stats,
    ) = run_small_cluster_matrix_phase_v11(
        active_cluster_tpcs=v11_active_cluster_tpcs,
        iterative_single_tpc=v11_iterative_single_tpc,
        pruned_iterative_clusters=v11_pruned_iterative_clusters,
        image_maps=imageMaps,
        base_image=baseImage,
        full_light_waveform=fullLightWaveform,
        full_light_std=fullLightStd,
        channel_support_cache=cluster_channel_support_cache,
        labels_global=labels_global,
        hit_timestamps=hit_timestamps,
        t0_candidates=t0Candidates,
        assignment_info=assignment_info,
        unassigned_by_tpc=unassigned_by_tpc,
        cluster_energies=cluster_energies,
        energy_band_fraction=0.20,
        positive_row_margin=-1e-4,
        matrix_worsen_tolerance_norm=0.15,
        search_range=800,
        adc_clip=ADC_CLIP,
        collect_scan_losses=False,
        full_scan_assign_eps=-1.0,
        backward_peak_align_ticks=5,
        leftover_absorption_context=(
            phase1["leftover_absorption_context"] if bool(enable_phase23_leftover_absorption) else None
        ),
    )
    backbone_verify_after_phase3 = verify_backbone_hits_unchanged_v11(
        backbone_snapshot,
        hit_timestamps=hit_timestamps,
        stage_name="Phase 3 (small-cluster matrix)",
    )

    truth = extract_event_hit_energy_and_truth_t0(
        resources.h5,
        resources.hits_ref,
        int(ev_id),
        hits_full=resources.hits_full,
        convert_to_matching_ticks=True,
    )

    return {
        "ev_id": int(ev_id),
        "hit_refs": hit_refs,
        "hits_evt": event["hits_evt"],
        "xset": xset,
        "yset": yset,
        "zset": zset,
        "Eset": Eset,
        "hitTPCid": hitTPCid,
        "fullLightWaveform": fullLightWaveform,
        "fullLightStd": fullLightStd,
        "baseImage": baseImage,
        "hit_timestamps": np.asarray(hit_timestamps, dtype=np.float32),
        "labels_global": np.asarray(labels_global, dtype=np.int32),
        "split_index": int(split_index),
        "label_info": label_info,
        "cluster_to_tpcs": cluster_to_tpcs,
        "cluster_channel_support_cache": cluster_channel_support_cache,
        "cluster_channel_support_summary": cluster_channel_support_summary,
        "imageMaps": imageMaps,
        "track_shower_labels": phase1["track_shower_labels"],
        "cluster_labels": phase1["cluster_labels"],
        "backbone_snapshot": backbone_snapshot,
        "backbone_verify_after_phase2": backbone_verify_after_phase2,
        "backbone_verify_after_phase3": backbone_verify_after_phase3,
        "large_cluster_assignment_log": large_cluster_assignment_log,
        "small_cluster_assignment_log": small_cluster_assignment_log,
        "v11_large_phase_stats": dict(v11_large_phase_stats),
        "v11_small_phase_stats": dict(v11_small_phase_stats),
        "track_second_pass_stats": dict(phase1["track_second_pass_stats"]),
        "track_swap_candidate_stats": dict(phase1["track_swap_candidate_stats"]),
        "track_swap_stats": dict(phase1["track_swap_stats"]),
        "v8_absorption_log": list(phase1["v8_absorption_log"]),
        "truth_t0": np.asarray(truth["truth_t0"], dtype=np.float64),
        "truth_hit_energy": np.asarray(truth["hit_energy"], dtype=np.float64),
        "flash_seed_stats": dict(flash_seed_stats),
        "clustering_debug_toolbox": clustering_debug_toolbox,
        "enable_phase23_leftover_absorption": bool(enable_phase23_leftover_absorption),
    }


def build_truth_t0_interaction_entries(
    event_result: dict[str, Any],
    *,
    tpcids: int | Iterable[int] | None = None,
    correct_ticks: float = 10.0,
    truth_rounding_ticks: float = 1.0,
) -> list[dict[str, Any]]:
    hit_tpcid = np.asarray(event_result["hitTPCid"], dtype=np.int64)
    reco_t0 = np.asarray(event_result["hit_timestamps"], dtype=np.float64)
    truth_t0 = np.asarray(event_result["truth_t0"], dtype=np.float64)
    energy = np.asarray(event_result["Eset"], dtype=np.float64)

    if truth_t0.shape[0] != reco_t0.shape[0]:
        raise ValueError("truth_t0 and reco_t0 lengths do not match")

    if tpcids is None:
        tpc_filter = np.ones(hit_tpcid.shape[0], dtype=bool)
    elif np.isscalar(tpcids):
        tpc_filter = hit_tpcid == int(tpcids)
    else:
        tpc_filter = np.isin(hit_tpcid, np.asarray(list(tpcids), dtype=np.int64))

    valid = (
        tpc_filter
        & np.isfinite(truth_t0)
        & np.isfinite(energy)
        & (energy > 0.0)
    )
    if not np.any(valid):
        return []

    if float(truth_rounding_ticks) <= 0.0:
        truth_group = truth_t0[valid]
    else:
        truth_group = np.rint(truth_t0[valid] / float(truth_rounding_ticks)).astype(np.int64)

    reco_valid = reco_t0[valid]
    truth_valid = truth_t0[valid]
    energy_valid = energy[valid]
    tpc_valid = hit_tpcid[valid]
    correct_mask_valid = np.isfinite(reco_valid) & (np.abs(reco_valid - truth_valid) <= float(correct_ticks))

    keys = np.column_stack([tpc_valid.astype(np.int64), truth_group.astype(np.int64)])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    rows: list[dict[str, Any]] = []
    for group_idx, (tpcid, truth_group_key) in enumerate(unique_keys):
        mask = inverse == int(group_idx)
        total_hits = int(np.count_nonzero(mask))
        correct_hits = int(np.count_nonzero(mask & correct_mask_valid))
        total_energy = float(np.sum(energy_valid[mask]))
        correct_energy = float(np.sum(energy_valid[mask & correct_mask_valid]))
        success_fraction = float(correct_energy / max(total_energy, 1e-12))
        hit_fraction = float(correct_hits / max(total_hits, 1))
        rows.append(
            {
                "eventid": int(event_result["ev_id"]),
                "TPCid": int(tpcid),
                "truth_t0_group": int(truth_group_key),
                "truth_t0_mean": float(np.mean(truth_valid[mask])),
                "total_hits": int(total_hits),
                "correct_hits": int(correct_hits),
                "total_energy_mev": float(total_energy),
                "correct_energy_mev": float(correct_energy),
                "success_fraction_energy": float(success_fraction),
                "success_fraction_hits": float(hit_fraction),
                "correct_ticks": float(correct_ticks),
            }
        )
    return rows


def evaluate_v11_interaction_success(
    resources: V11PlottingResources,
    *,
    event_ids: Iterable[int] = range(1, 11),
    tpcids: int | Iterable[int] | None = None,
    correct_ticks: float = 10.0,
    truth_rounding_ticks: float = 1.0,
    lam: float = 1.2,
    verbose: bool = True,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for ev_id in [int(v) for v in event_ids]:
        if verbose:
            print()
            print(f"=== Event {ev_id} ===")
        try:
            event_result = run_v11_pipeline_for_event(resources, ev_id, lam=float(lam), verbose=verbose)
            rows = build_truth_t0_interaction_entries(
                event_result,
                tpcids=tpcids,
                correct_ticks=float(correct_ticks),
                truth_rounding_ticks=float(truth_rounding_ticks),
            )
            all_rows.extend(rows)

            total_energy = float(np.sum(event_result["Eset"]))
            finite_mask = np.isfinite(event_result["truth_t0"]) & np.isfinite(event_result["hit_timestamps"])
            correct_mask = finite_mask & (
                np.abs(np.asarray(event_result["hit_timestamps"], dtype=np.float64) - np.asarray(event_result["truth_t0"], dtype=np.float64))
                <= float(correct_ticks)
            )
            total_truth_energy = float(np.sum(np.asarray(event_result["Eset"], dtype=np.float64)[np.isfinite(event_result["truth_t0"])]))
            correct_truth_energy = float(np.sum(np.asarray(event_result["Eset"], dtype=np.float64)[correct_mask]))
            energy_fraction = float(correct_truth_energy / max(total_truth_energy, 1e-12))
            event_summary = {
                "eventid": int(ev_id),
                "n_interactions": int(len(rows)),
                "mean_interaction_success": float(np.mean([row["success_fraction_energy"] for row in rows])) if rows else np.nan,
                "median_interaction_success": float(np.median([row["success_fraction_energy"] for row in rows])) if rows else np.nan,
                "truth_energy_mev": float(total_truth_energy),
                "correct_truth_energy_mev": float(correct_truth_energy),
                "energy_fraction_correct": float(energy_fraction),
                "total_event_energy_mev": float(total_energy),
                "backbone_changed_after_p2": int(event_result["backbone_verify_after_phase2"]["n_changed"]),
                "backbone_changed_after_p3": int(event_result["backbone_verify_after_phase3"]["n_changed"]),
            }
            event_summaries.append(event_summary)

            if verbose:
                print(
                    f"Event {ev_id}: interactions={event_summary['n_interactions']} | "
                    f"median success={100.0 * event_summary['median_interaction_success']:.2f}% | "
                    f"truth-energy correct={100.0 * event_summary['energy_fraction_correct']:.2f}%"
                )
        except Exception as exc:
            failures.append({"eventid": int(ev_id), "error": str(exc)})
            if verbose:
                print(f"Event {ev_id} failed: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "rows": all_rows,
        "event_summaries": event_summaries,
        "failures": failures,
        "settings": {
            "event_ids": [int(v) for v in event_ids],
            "tpcids": None if tpcids is None else ([int(tpcids)] if np.isscalar(tpcids) else [int(v) for v in tpcids]),
            "correct_ticks": float(correct_ticks),
            "truth_rounding_ticks": float(truth_rounding_ticks),
            "lam": float(lam),
        },
    }


def plot_interaction_success_histogram(
    rows: list[dict[str, Any]],
    *,
    min_energy_mev: float | None = None,
    max_energy_mev: float | None = None,
    bins: np.ndarray | None = None,
    figsize: tuple[float, float] = (8.8, 4.8),
    dpi: int = 150,
) -> tuple[Any, Any]:
    bins = np.asarray(bins if bins is not None else np.arange(0.0, 1.05, 0.05), dtype=np.float64)
    fractions = []
    for row in rows:
        energy = float(row["total_energy_mev"])
        if min_energy_mev is not None and energy < float(min_energy_mev):
            continue
        if max_energy_mev is not None and energy > float(max_energy_mev):
            continue
        fractions.append(100.0 * float(row["success_fraction_energy"]))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(fractions, bins=100.0 * bins, color="#4C78A8", edgecolor="white", linewidth=0.7)
    ax.set_xlabel("Correctly reconstructed energy fraction per interaction [%]")
    ax.set_ylabel("Number of interactions")
    ax.set_title("Interaction-level reconstruction success")
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, ax


def plot_interaction_success_vs_energy(
    rows: list[dict[str, Any]],
    *,
    min_energy_mev: float | None = None,
    max_energy_mev: float | None = None,
    logx: bool = True,
    figsize: tuple[float, float] = (8.8, 5.2),
    dpi: int = 150,
) -> tuple[Any, Any]:
    energy = []
    success = []
    for row in rows:
        e = float(row["total_energy_mev"])
        if min_energy_mev is not None and e < float(min_energy_mev):
            continue
        if max_energy_mev is not None and e > float(max_energy_mev):
            continue
        energy.append(e)
        success.append(100.0 * float(row["success_fraction_energy"]))

    energy_arr = np.asarray(energy, dtype=np.float64)
    success_arr = np.asarray(success, dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(energy_arr, success_arr, s=26, alpha=0.70, color="#E45756", edgecolors="none")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Interaction energy [MeV]")
    ax.set_ylabel("Correctly reconstructed energy fraction [%]")
    ax.set_title("Interaction reconstruction success vs. energy")
    ax.set_ylim(-2.0, 102.0)
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, ax


__all__ = [
    "V11PlottingResources",
    "assign_hits_to_charge_tpc",
    "charge_tpc_from_io_group",
    "build_truth_t0_interaction_entries",
    "close_v11_plotting_resources",
    "evaluate_v11_interaction_success",
    "extract_event_hit_energy_and_truth_t0",
    "get_formatted_light_waveforms",
    "load_v11_plotting_resources",
    "plot_interaction_success_histogram",
    "plot_interaction_success_vs_energy",
    "run_v11_pipeline_for_event",
]
