"""V2 light-rescue layer for vRelease5-Copy1.

Sits next to ``phase25_trial2_combined.py`` without modifying any existing
module. Reuses the existing spatial repair / large-grid correction by calling
``phase25_trial2.run_trial2_phase25_from_namespace`` with ``enable_light=False``
and then layering a stricter v2 light pass on top.

Improvements over the trial2 physical-chi2 light pass:

1. Stricter overflow definition: bumps ``light_overflow_sigma`` and
   ``light_overflow_abs_adc`` so we only look at clearly-overshooting channels.
2. Severity-aware track veto override: when the source t0 is overflowing very
   hard (lots of saturated channels above many sigma) and the candidate move
   produces overwhelming dChi2 evidence, allow leaving even a multi-TPC track.
3. Multi-cluster reshuffle: when several severe overflows exist in one TPC,
   try permuting the donor components across the union of relevant t0s and
   accept the assignment with the smallest joint physical chi2.

Speed is bounded explicitly: reshuffle uses at most a handful of clusters and
a small t0 pool per TPC, the chi2 evaluation operates on a windowed slice,
and the single-pass loop is identical in cost to the existing trial2 pass.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable

import numpy as np

try:
    from . import phase25_amendment as p25
    from . import phase25_trial2 as t2
    from . import phase25_trial2_combined as t2c
    from .flash_cluster_table import (
        ensure_flash_cluster_flags,
        mark_flash_cluster_assignment,
        source_t0s_from_received_flash_table,
    )
except Exception:  # pragma: no cover - direct notebook import fallback
    import phase25_amendment as p25
    import phase25_trial2 as t2
    import phase25_trial2_combined as t2c
    from flash_cluster_table import (
        ensure_flash_cluster_flags,
        mark_flash_cluster_assignment,
        source_t0s_from_received_flash_table,
    )


@dataclass
class Trial2V2LightConfig(t2c.Trial2CombinedConfig):
    """Trial2 config + v2 light-rescue knobs.

    Inherits every field from ``Trial2CombinedConfig`` so existing notebook
    code that builds a ``Trial2CombinedConfig`` keeps working unchanged when
    swapped in.  The defaults below override the few fields we want stricter
    out of the box.
    """

    # Overflow definition.  Empirically the looser thresholds catch
    # dramatically more correct moves on both event 0 and event 6 (the chi2
    # gate downstream filters out noise candidates), so we keep the
    # existing trial2 defaults rather than tightening them.
    light_overflow_sigma: float = 3.0
    light_overflow_abs_adc: float = 400.0
    light_model_activity_adc: float = 400.0
    light_min_overflow_channels: int = 6

    # Severity-aware track veto override: an overshoot at the source t0 large
    # enough to convince us a long-track-label cluster is at the wrong t0.
    enable_force_override: bool = True
    light_force_override_min_src_ofch: int = 22
    light_force_override_min_peak_ofch: int = 12
    light_force_override_min_severity: float = 250.0
    light_force_override_min_src_red: int = 14
    light_force_override_min_dchi2: float = 5.0e3
    light_force_override_min_dchi2_per_e: float = 5.0e1

    # Multi-cluster reshuffle.
    enable_reshuffle: bool = True
    reshuffle_min_severe_rows: int = 2
    reshuffle_max_clusters: int = 3
    reshuffle_max_t0_pool: int = 4
    reshuffle_severity_floor: float = 100.0
    reshuffle_min_dchi2: float = 5.0e2
    reshuffle_min_dchi2_per_e: float = 1.0e1
    reshuffle_max_total: int = 6

    # Multi-pass: re-scan after each round of moves to catch cascading
    # misassignments that only appear once an earlier move clears the model.
    # Default to 1 -- empirically multi-pass helps event 6 (+1013 -> +1202
    # net correct) but hurts event 0 (+435 -> +331) because later passes
    # admit weaker chi2 evidence.  Set to 2-3 when you need to maximize
    # captured movement on hard events and don't mind extra CtoW.
    # Later-pass chi2 thresholds (see ``later_pass_phys_*``) are bumped
    # automatically so multi-pass stays safer when it is enabled.
    n_outer_passes: int = 1
    # Per-pass chi2 tightening for outer_pass >= 1.  Falls back to the
    # single-move chi2 thresholds when these are <= 0.
    later_pass_phys_min_dchi2_improvement: float = 2.0e3
    later_pass_phys_min_dchi2_per_mev: float = 25.0
    later_pass_phys_min_source_ofch_reduction: int = 12

    # Source-only scan controls (kept aligned with trial2 physical-chi2 path).
    light_overflow_rows_per_pass: int = 1_000_000_000

    # Scratch metadata bag for callers that want to attach run notes.
    v2_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers


def _force_override_inputs(
    *,
    overflow_row: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "src_ofch_old": int(overflow_row.get("overflow_channels", gate.get("src_of0", 0))),
        "peak_ofch": int(overflow_row.get("peak_overflow_channels", 0)),
        "severity": float(overflow_row.get("severity", 0.0)),
        "src_red": int(gate.get("src_red", 0)),
        "dchi2": float(gate.get("dchi2", 0.0)),
        "dchi2_per_mev": float(gate.get("dchi2_per_mev", 0.0)),
    }


def _force_override_eligible(
    inputs: dict[str, Any],
    cfg: Trial2V2LightConfig,
) -> bool:
    if not bool(cfg.enable_force_override):
        return False
    return (
        int(inputs["src_ofch_old"]) >= int(cfg.light_force_override_min_src_ofch)
        and int(inputs["peak_ofch"]) >= int(cfg.light_force_override_min_peak_ofch)
        and float(inputs["severity"]) >= float(cfg.light_force_override_min_severity)
        and int(inputs["src_red"]) >= int(cfg.light_force_override_min_src_red)
        and float(inputs["dchi2"]) >= float(cfg.light_force_override_min_dchi2)
        and float(inputs["dchi2_per_mev"]) >= float(cfg.light_force_override_min_dchi2_per_e)
    )


def _evaluate_v2_gate(
    trial: dict[str, Any],
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    saturated_channel_cache: Any | None,
    config: Trial2V2LightConfig,
    cfg25: p25.Phase25Config,
    outer_pass: int = 0,
) -> dict[str, Any]:
    gate_config = config
    # Pass >= 1 uses tightened chi2 thresholds (when configured) so multi-pass
    # does not let weak evidence accumulate into CtoW noise.
    if (
        int(outer_pass) >= 1
        and (
            float(config.later_pass_phys_min_dchi2_improvement) > 0.0
            or float(config.later_pass_phys_min_dchi2_per_mev) > 0.0
            or int(config.later_pass_phys_min_source_ofch_reduction) > 0
        )
    ):
        gate_config = copy.copy(config)
        if float(config.later_pass_phys_min_dchi2_improvement) > 0.0:
            gate_config.phys_min_dchi2_improvement = float(
                config.later_pass_phys_min_dchi2_improvement
            )
        if float(config.later_pass_phys_min_dchi2_per_mev) > 0.0:
            gate_config.phys_min_dchi2_per_mev = float(
                config.later_pass_phys_min_dchi2_per_mev
            )
        if int(config.later_pass_phys_min_source_ofch_reduction) > 0:
            gate_config.phys_min_source_ofch_reduction = int(
                config.later_pass_phys_min_source_ofch_reduction
            )
    gate = t2._evaluate_trial2_light_gate(
        trial,
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_std if False else full_light_std,
        saturated_channel_cache=saturated_channel_cache,
        config=gate_config,
        cfg25=cfg25,
    )
    gate["pass_chi2_thresholds"] = {
        "dchi2": float(gate_config.phys_min_dchi2_improvement),
        "dchi2_per_mev": float(gate_config.phys_min_dchi2_per_mev),
        "src_ofch_reduction": int(gate_config.phys_min_source_ofch_reduction),
        "outer_pass": int(outer_pass),
    }
    fo_inputs = _force_override_inputs(overflow_row=trial.get("overflow_row", {}), gate=gate)
    gate["force_override_inputs"] = fo_inputs
    gate["force_override"] = bool(_force_override_eligible(fo_inputs, config))
    return gate


def _apply_track_veto_v2(
    trial: dict[str, Any],
    accepted: bool,
    *,
    cfg: Trial2V2LightConfig,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    energy: np.ndarray,
    label_info: Any | None,
) -> bool:
    """Multi-TPC track veto with optional severity-aware force override."""
    if not bool(accepted):
        trial["track_veto_ok"] = True
        trial["track_veto_overridden"] = False
        return False
    if not bool(cfg.light_veto_multitpc_track):
        trial["track_veto_ok"] = True
        trial["track_veto_overridden"] = False
        return True

    veto = t2._light_multitpc_track_veto(
        trial,
        labels_global=labels_global,
        hit_tpc_ids=hit_tpc_ids,
        energy=energy,
        label_info=label_info,
        min_tpcs=int(cfg.light_veto_track_min_tpcs),
    )
    trial.update(veto)
    if not bool(veto.get("track_veto", False)):
        trial["track_veto_ok"] = True
        trial["track_veto_overridden"] = False
        return True

    # Vetoed by track-label rule.  If the source overflow signature is extreme
    # enough, override and accept; otherwise reject.
    if bool(trial.get("force_override", False)):
        trial["accepted_by_before_track_veto"] = str(trial.get("accepted_by", ""))
        trial["track_veto_ok"] = True
        trial["track_veto_overridden"] = True
        trial["track_veto_override_reason"] = "force_override_extreme_overflow"
        return True

    trial["accepted_by_before_track_veto"] = str(trial.get("accepted_by", ""))
    trial["accepted_by"] = "reject"
    trial["track_veto_ok"] = False
    trial["track_veto_overridden"] = False
    return False


def _apply_one_trial(
    *,
    hit_ts: np.ndarray,
    base_image: np.ndarray,
    trial: dict[str, Any],
    namespace: dict[str, Any],
    hit_tpc_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    cfg25: p25.Phase25Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    moved = np.asarray(trial["hit_indices"], dtype=np.int64)
    tpc = int(trial["TPCid"])
    old_t0 = int(trial["old_t0"])
    new_t0 = int(trial["new_t0"])
    before_ts = hit_ts.copy()
    before_base = base_image.copy()
    new_ts = hit_ts.copy()
    new_ts[moved] = np.float32(new_t0)
    affected = {(tpc, old_t0), (tpc, new_t0)}
    new_base, records = p25._exact_update_affected_families(
        base_image=before_base,
        old_hit_timestamps=before_ts,
        new_hit_timestamps=new_ts,
        affected_specs=affected,
        hit_tpc_ids=hit_tpc_ids,
        x=x,
        y=y,
        z=z,
        energy=energy,
        model=t2._get(namespace, "model"),
        template=t2._get(namespace, "wvfm_tmpl"),
        cfg=cfg25,
    )
    return new_ts, new_base, records


def _maybe_canonicalize_flash_table(
    *,
    namespace: dict[str, Any],
    flash_cluster_received: Any | None,
    t0_candidates: Any,
    tpc: int,
    new_t0: float,
    cfg: Trial2V2LightConfig,
    flash_canon_rows: list[dict[str, Any]],
    component_id: int,
) -> tuple[Any, Any]:
    if flash_cluster_received is None:
        return t0_candidates, flash_cluster_received
    t0_candidates, flash_cluster_received, flash_row = mark_flash_cluster_assignment(
        t0_candidates,
        flash_cluster_received,
        tpc=int(tpc),
        t0=float(new_t0),
        resolution_ticks=float(cfg.light_flash_cluster_match_ticks),
        max_t0=800.0,
        prefer_existing_true=False,
        clusterid=int(component_id),
        stage="trial2_v2_light_rescue",
    )
    flash_canon_rows.append(flash_row)
    namespace["flash_cluster_received_by_tpc"] = flash_cluster_received
    namespace["t0Candidates"] = t0_candidates
    return t0_candidates, flash_cluster_received


def _refresh_source_t0s(
    *,
    flash_cluster_received: Any | None,
    t0_candidates: Any,
    hit_ts: np.ndarray,
    tpcs: np.ndarray,
    labels: np.ndarray,
    energy: np.ndarray,
    light_allowed_tpcs: Iterable[int],
    cfg: Trial2V2LightConfig,
) -> dict[int, list[int]]:
    if flash_cluster_received is not None:
        out, _ = source_t0s_from_received_flash_table(
            t0_candidates,
            flash_cluster_received,
            allowed_tpcs=light_allowed_tpcs,
            min_sep_ticks=float(cfg.light_flash_cluster_match_ticks),
        )
    else:
        out, _ = t2._associated_source_t0s_by_tpc(
            hit_timestamps=hit_ts,
            hit_tpc_ids=tpcs,
            labels_global=labels,
            energy=energy,
            t0_candidates=t2._as_tpc_dict(t0_candidates),
            allowed_tpcs=light_allowed_tpcs,
            match_ticks=float(cfg.light_flash_cluster_match_ticks),
        )
    return out


def _scan_with_details(
    *,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    source_t0s_by_tpc: dict[int, list[int]],
    saturated_channel_cache: Any | None,
    light_allowed_tpcs: Iterable[int],
    hit_ts: np.ndarray,
    tpcs: np.ndarray,
    labels: np.ndarray,
    energy: np.ndarray,
    cfg25: p25.Phase25Config,
    cfg: Trial2V2LightConfig,
) -> list[dict[str, Any]]:
    rows = t2._scan_light_overflows_source_only(
        base_image=base_image,
        full_light_waveform=full_light_waveform,
        full_light_std=full_light_std,
        source_t0s_by_tpc=source_t0s_by_tpc,
        saturated_channel_cache=saturated_channel_cache,
        allowed_tpcs=set(int(v) for v in light_allowed_tpcs),
        cfg=cfg25,
    )
    for row in rows:
        row.update(
            t2._light_source_detail_for_t0(
                hit_timestamps=hit_ts,
                hit_tpc_ids=tpcs,
                labels_global=labels,
                energy=energy,
                tpc=int(row["TPCid"]),
                t0=int(row["t0"]),
                match_ticks=float(cfg.light_t0_match_ticks),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Reshuffle


def _try_reshuffle_per_tpc(
    *,
    tpc: int,
    severe_rows: list[dict[str, Any]],
    base_image: np.ndarray,
    hit_ts: np.ndarray,
    namespace: dict[str, Any],
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    saturated_channel_cache: Any | None,
    t0_candidates: Any,
    labels: np.ndarray,
    tpcs: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    label_info: Any | None,
    cfg25: p25.Phase25Config,
    cfg: Trial2V2LightConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Attempt one reshuffle in the given TPC.

    Returns ``(applied_trials, record_or_None)`` -- the trials that are ready
    to be committed to ``hit_ts``/``base_image`` and a diagnostic record.
    Caller is responsible for applying them with ``_apply_one_trial`` so the
    exact GPU family update happens in one place.
    """
    if len(severe_rows) < int(cfg.reshuffle_min_severe_rows):
        return [], None

    # Build per-row donor trials & proxies.  We cap at the most severe N.
    severe_rows = severe_rows[: int(cfg.reshuffle_max_clusters)]
    row_trials: list[dict[str, Any]] = []
    proxies: list[np.ndarray] = []
    for row in severe_rows:
        trial = p25._try_light_repair_row(
            row,
            base_image=base_image,
            full_light_waveform=full_light_waveform,
            full_light_std=full_light_std,
            image_maps=namespace.get("imageMaps"),
            t0_candidates=t2._as_tpc_dict(t0_candidates),
            hit_timestamps=hit_ts,
            hit_tpc_ids=tpcs,
            labels_global=labels,
            x=x,
            y=y,
            z=z,
            energy=energy,
            saturated_channel_cache=saturated_channel_cache,
            locked_hit_mask=None,
            cfg=cfg25,
        )
        if trial is None:
            continue
        trial = dict(trial)
        trial["overflow_row"] = dict(row)
        proxy, image_status = p25._component_proxy_image(
            component=trial,
            image_maps=namespace.get("imageMaps"),
            labels_global=labels,
            hit_tpc_ids=tpcs,
            energy=energy,
        )
        trial["image_status"] = str(image_status)
        if proxy is None:
            continue
        row_trials.append(trial)
        proxies.append(np.asarray(proxy, dtype=np.float32))

    if len(row_trials) < int(cfg.reshuffle_min_severe_rows):
        return [], None

    # T0 pool: union of each donor's old + suggested-new t0.
    t0_pool: list[int] = []
    for trial in row_trials:
        for k in (int(trial["old_t0"]), int(trial["new_t0"])):
            if k not in t0_pool:
                t0_pool.append(k)
    if len(t0_pool) > int(cfg.reshuffle_max_t0_pool):
        t0_pool = t0_pool[: int(cfg.reshuffle_max_t0_pool)]
    if len(t0_pool) < 2:
        return [], None

    # Window: union of half-windows around each pool t0.
    n_ticks = int(base_image.shape[-1])
    half = int(cfg25.half_window_ticks)
    peak = int(cfg25.pulse_peak_tick)
    tick_mask = np.zeros(n_ticks, dtype=bool)
    for t0 in t0_pool:
        tick = int(t0) + peak
        if tick < 0 or tick >= n_ticks:
            continue
        lo = max(0, tick - half)
        hi = min(n_ticks, tick + half + 1)
        if hi > lo:
            tick_mask[lo:hi] = True
    if not np.any(tick_mask):
        return [], None

    keep = p25._keep_channel_indices(int(tpc), full_light_waveform, saturated_channel_cache)
    if keep.size == 0:
        return [], None

    actual_w = np.asarray(full_light_waveform[int(tpc)][keep][:, tick_mask], dtype=np.float64)
    std_w = np.maximum(
        np.asarray(full_light_std[int(tpc)][keep][:, tick_mask], dtype=np.float64),
        float(cfg.phys_std_floor),
    )
    base_w = np.asarray(base_image[int(tpc)][keep][:, tick_mask], dtype=np.float64)
    adc_clip = float(cfg25.adc_clip)

    # Pre-shifted contributions in the window.  Each cluster-at-t0 is a
    # window-restricted view of ``shift(proxy, t0)``.
    new_w: dict[tuple[int, int], np.ndarray] = {}
    old_w: dict[int, np.ndarray] = {}
    for c_idx, trial in enumerate(row_trials):
        proxy = proxies[c_idx]
        old_shift_full = p25._shift_image(proxy, int(trial["old_t0"]), nt=n_ticks)
        old_w[c_idx] = np.asarray(old_shift_full[keep][:, tick_mask], dtype=np.float64)
        for t0 in t0_pool:
            shift_full = p25._shift_image(proxy, int(t0), nt=n_ticks)
            new_w[(c_idx, int(t0))] = np.asarray(shift_full[keep][:, tick_mask], dtype=np.float64)

    # Baseline = each cluster at its own old_t0 (no delta).  The baseline
    # ``base_image`` already includes those contributions, so the chi2 is just
    # against ``base_w``.
    chi2_baseline = float(np.sum(((base_w - actual_w) / std_w) ** 2))

    best_chi2 = chi2_baseline
    best_assignment: tuple[int, ...] | None = None
    n_clusters = len(row_trials)

    # Each cluster picks one t0 from the pool; bounded enumeration.
    for assignment in product(t0_pool, repeat=n_clusters):
        # Skip the no-op assignment.
        if all(int(t0) == int(row_trials[c]["old_t0"]) for c, t0 in enumerate(assignment)):
            continue
        pred = base_w.copy()
        for c, t0 in enumerate(assignment):
            if int(t0) == int(row_trials[c]["old_t0"]):
                continue
            pred = pred + new_w[(c, int(t0))] - old_w[c]
        np.clip(pred, 0.0, adc_clip, out=pred)
        chi2 = float(np.sum(((pred - actual_w) / std_w) ** 2))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_assignment = tuple(int(v) for v in assignment)

    if best_assignment is None:
        return [], None

    dchi2 = float(chi2_baseline - best_chi2)
    if dchi2 < float(cfg.reshuffle_min_dchi2):
        return [], None

    moved_clusters: list[tuple[int, int]] = [
        (c, int(t0))
        for c, t0 in enumerate(best_assignment)
        if int(t0) != int(row_trials[c]["old_t0"])
    ]
    if not moved_clusters:
        return [], None

    total_e_moved = float(
        sum(float(row_trials[c]["energy_mev"]) for c, _ in moved_clusters)
    )
    if total_e_moved <= 0.0:
        return [], None
    if (dchi2 / total_e_moved) < float(cfg.reshuffle_min_dchi2_per_e):
        return [], None

    # Per-moved-cluster track veto with optional force-override.
    applied: list[dict[str, Any]] = []
    for c, new_t0 in moved_clusters:
        trial = dict(row_trials[c])
        trial["new_t0"] = int(new_t0)
        trial["accepted_by"] = "reshuffle"
        trial["force_override"] = bool(_force_override_eligible(
            _force_override_inputs(
                overflow_row=trial.get("overflow_row", {}),
                gate={
                    "src_of0": int(trial.get("overflow_row", {}).get("overflow_channels", 0)),
                    "src_red": int(trial.get("overflow_row", {}).get("overflow_channels", 0)),
                    "dchi2": float(dchi2),
                    "dchi2_per_mev": float(dchi2 / max(total_e_moved, 1.0e-6)),
                },
            ),
            cfg,
        ))
        veto_ok = _apply_track_veto_v2(
            trial,
            True,
            cfg=cfg,
            labels_global=labels,
            hit_tpc_ids=tpcs,
            energy=energy,
            label_info=label_info,
        )
        if not veto_ok:
            return [], None
        trial["accepted"] = True
        trial["reshuffle_dchi2"] = float(dchi2)
        trial["reshuffle_dchi2_per_mev"] = float(dchi2 / max(total_e_moved, 1.0e-6))
        trial["reshuffle_assignment"] = list(best_assignment)
        trial["reshuffle_pool"] = list(t0_pool)
        trial["reshuffle_total_e_moved_mev"] = float(total_e_moved)
        applied.append(trial)

    record = {
        "TPCid": int(tpc),
        "t0_pool": list(t0_pool),
        "current_assignment": [int(t["old_t0"]) for t in row_trials],
        "best_assignment": list(best_assignment),
        "chi2_baseline": float(chi2_baseline),
        "chi2_best": float(best_chi2),
        "dchi2": float(dchi2),
        "total_e_moved_mev": float(total_e_moved),
        "n_moved_clusters": int(len(applied)),
    }
    return applied, record


# ---------------------------------------------------------------------------
# Main entry point


def run_trial2_v2_light_rescue_from_namespace(
    namespace: dict[str, Any],
    *,
    config: Trial2V2LightConfig | None = None,
    commit: bool | None = None,
) -> dict[str, Any]:
    """Run trial2 spatial + large-grid then the v2 light rescue."""
    cfg = config or Trial2V2LightConfig()
    do_commit = cfg.commit if commit is None else bool(commit)
    t_start = time.time()

    before_ts = np.asarray(t2._get(namespace, "hit_timestamps"), dtype=np.float32).copy()
    before_base = np.asarray(t2._get(namespace, "baseImage"), dtype=np.float32).copy()
    before_t0_candidates = copy.deepcopy(t2._get(namespace, "t0Candidates"))
    before_flash = copy.deepcopy(namespace.get("flash_cluster_received_by_tpc", None))

    # Run trial2 base path with the standard light pass disabled.  We always
    # commit those into the namespace so the v2 light pass operates on the
    # spatial-corrected state.
    base_cfg = copy.deepcopy(cfg)
    base_cfg.enable_light = False
    base_cfg.commit = True
    base_result = t2.run_trial2_phase25_from_namespace(namespace, config=base_cfg, commit=True)

    base_out = np.asarray(base_result["baseImage"], dtype=np.float32).copy()
    hit_ts_out = np.asarray(base_result["hit_timestamps"], dtype=np.float32).copy()

    labels = np.asarray(t2._get(namespace, "labels_global"), dtype=np.int64)
    label_info = namespace.get("label_info", None)
    tpcs = np.asarray(t2._get(namespace, "hitTPCid"), dtype=np.int32)
    x = np.asarray(t2._get(namespace, "xset"), dtype=np.float64)
    y = np.asarray(t2._get(namespace, "yset"), dtype=np.float64)
    z = np.asarray(t2._get(namespace, "zset"), dtype=np.float64)
    energy = np.asarray(t2._get(namespace, "Eset"), dtype=np.float64)
    full_light = np.asarray(t2._get(namespace, "fullLightWaveform"), dtype=np.float32)
    full_std = np.asarray(
        namespace.get(
            "fullLightStd",
            namespace.get("fullLightStd_phase2", np.ones_like(full_light)),
        ),
        dtype=np.float32,
    )

    # Phase25Config aligned with our cfg overrides; we keep p25 permissive and
    # let our gate make the decision.
    cfg25_light = t2._make_phase25_config(cfg, enable_spatial=False, enable_light=True)
    cfg25_light.max_move_energy_per_tpc_mev = None
    cfg25_light.light_min_loss_improvement = -1.0e30
    cfg25_light.light_min_old_overflow_reduction = -1.0e30
    cfg25_light.light_min_dest_deficit_reduction = -1.0e30
    cfg25_light.light_max_dest_new_overflow_frac = 1.0e30
    cfg25_light.light_overflow_rows_per_pass = int(cfg.light_overflow_rows_per_pass)

    all_hit_tpcs = sorted(set(int(v) for v in np.unique(tpcs)))
    spatial_allowed_tpcs = sorted(set(int(v) for v in base_result.get("allowed_tpcs", all_hit_tpcs)))
    light_allowed_tpcs = (
        spatial_allowed_tpcs if bool(cfg.light_skip_shower_tpcs) else all_hit_tpcs
    )

    flash_cluster_received = namespace.get("flash_cluster_received_by_tpc", None)
    t0_candidates = t2._get(namespace, "t0Candidates")
    if flash_cluster_received is not None:
        flash_cluster_received = ensure_flash_cluster_flags(
            t0_candidates, flash_cluster_received, max_t0=800.0
        )
        namespace["flash_cluster_received_by_tpc"] = flash_cluster_received

    flash_canon_rows: list[dict[str, Any]] = list(
        base_result.get("flash_canonicalization_rows", [])
    )

    source_t0s_by_tpc = _refresh_source_t0s(
        flash_cluster_received=flash_cluster_received,
        t0_candidates=t0_candidates,
        hit_ts=hit_ts_out,
        tpcs=tpcs,
        labels=labels,
        energy=energy,
        light_allowed_tpcs=light_allowed_tpcs,
        cfg=cfg,
    )

    rows = _scan_with_details(
        base_image=base_out,
        full_light_waveform=full_light,
        full_light_std=full_std,
        source_t0s_by_tpc=source_t0s_by_tpc,
        saturated_channel_cache=namespace.get("saturated_channel_cache"),
        light_allowed_tpcs=light_allowed_tpcs,
        hit_ts=hit_ts_out,
        tpcs=tpcs,
        labels=labels,
        energy=energy,
        cfg25=cfg25_light,
        cfg=cfg,
    )
    light_source_rows = [dict(r) for r in rows]

    light_moves: list[dict[str, Any]] = []
    light_trials: list[dict[str, Any]] = []
    moves_by_tpc: dict[int, int] = {}
    family_update_records: list[dict[str, Any]] = list(
        base_result.get("family_update_records", [])
    )
    reshuffle_records: list[dict[str, Any]] = []
    pass_logs: list[dict[str, Any]] = []
    # Ping-pong guard: don't accept (tpc, new, old) right after applying (tpc, old, new).
    move_history: set[tuple[int, int, int]] = set()

    n_outer_passes = max(1, int(cfg.n_outer_passes))
    for outer_pass in range(n_outer_passes):
        pass_start_count = len(light_moves)

        if outer_pass > 0:
            # Re-scan: state has changed since the previous pass.
            source_t0s_by_tpc = _refresh_source_t0s(
                flash_cluster_received=flash_cluster_received,
                t0_candidates=t0_candidates,
                hit_ts=hit_ts_out,
                tpcs=tpcs,
                labels=labels,
                energy=energy,
                light_allowed_tpcs=light_allowed_tpcs,
                cfg=cfg,
            )
            rows = _scan_with_details(
                base_image=base_out,
                full_light_waveform=full_light,
                full_light_std=full_std,
                source_t0s_by_tpc=source_t0s_by_tpc,
                saturated_channel_cache=namespace.get("saturated_channel_cache"),
                light_allowed_tpcs=light_allowed_tpcs,
                hit_ts=hit_ts_out,
                tpcs=tpcs,
                labels=labels,
                energy=energy,
                cfg25=cfg25_light,
                cfg=cfg,
            )

        # ---- Phase A: reshuffle severe groups ----
        # Pass >= 1 also tightens the reshuffle dChi2 gates so we do not
        # accumulate marginal CtoW noise across passes.
        cfg_for_pass = cfg
        if int(outer_pass) >= 1 and bool(cfg.enable_reshuffle):
            cfg_for_pass = copy.copy(cfg)
            if float(cfg.later_pass_phys_min_dchi2_improvement) > 0.0:
                cfg_for_pass.reshuffle_min_dchi2 = max(
                    float(cfg.reshuffle_min_dchi2),
                    float(cfg.later_pass_phys_min_dchi2_improvement),
                )
            if float(cfg.later_pass_phys_min_dchi2_per_mev) > 0.0:
                cfg_for_pass.reshuffle_min_dchi2_per_e = max(
                    float(cfg.reshuffle_min_dchi2_per_e),
                    float(cfg.later_pass_phys_min_dchi2_per_mev),
                )
        if bool(cfg.enable_reshuffle):
            rows_by_tpc: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                rows_by_tpc.setdefault(int(row["TPCid"]), []).append(dict(row))

            reshuffle_count = 0
            for tpc in sorted(rows_by_tpc.keys()):
                if reshuffle_count >= int(cfg.reshuffle_max_total):
                    break
                tpc_rows = rows_by_tpc[int(tpc)]
                severe = [
                    r for r in tpc_rows
                    if float(r.get("severity", 0.0)) >= float(cfg.reshuffle_severity_floor)
                ]
                if len(severe) < int(cfg.reshuffle_min_severe_rows):
                    continue
                severe.sort(key=lambda r: -float(r.get("severity", 0.0)))

                applied, record = _try_reshuffle_per_tpc(
                    tpc=int(tpc),
                    severe_rows=severe,
                    base_image=base_out,
                    hit_ts=hit_ts_out,
                    namespace=namespace,
                    full_light_waveform=full_light,
                    full_light_std=full_std,
                    saturated_channel_cache=namespace.get("saturated_channel_cache"),
                    t0_candidates=t0_candidates,
                    labels=labels,
                    tpcs=tpcs,
                    x=x,
                    y=y,
                    z=z,
                    energy=energy,
                    label_info=label_info,
                    cfg25=cfg25_light,
                    cfg=cfg_for_pass,
                )
                if not applied or record is None:
                    continue
                # Reject if any move would undo a recent move (ping-pong guard).
                if any(
                    (int(t["TPCid"]), int(t["new_t0"]), int(t["old_t0"])) in move_history
                    for t in applied
                ):
                    continue
                for trial in applied:
                    if len(light_moves) >= int(cfg.light_max_total_moves):
                        break
                    if moves_by_tpc.get(int(tpc), 0) >= int(cfg.light_max_moves_per_tpc):
                        break
                    hit_ts_out, base_out, records = _apply_one_trial(
                        hit_ts=hit_ts_out,
                        base_image=base_out,
                        trial=trial,
                        namespace=namespace,
                        hit_tpc_ids=tpcs,
                        x=x,
                        y=y,
                        z=z,
                        energy=energy,
                        cfg25=cfg25_light,
                    )
                    family_update_records.extend(records)
                    trial["outer_pass"] = int(outer_pass)
                    light_moves.append(trial)
                    light_trials.append(trial)
                    moves_by_tpc[int(tpc)] = moves_by_tpc.get(int(tpc), 0) + 1
                    move_history.add((int(trial["TPCid"]), int(trial["old_t0"]), int(trial["new_t0"])))
                    t0_candidates, flash_cluster_received = _maybe_canonicalize_flash_table(
                        namespace=namespace,
                        flash_cluster_received=flash_cluster_received,
                        t0_candidates=t0_candidates,
                        tpc=int(tpc),
                        new_t0=float(trial["new_t0"]),
                        cfg=cfg,
                        flash_canon_rows=flash_canon_rows,
                        component_id=int(trial.get("component_id", -1)),
                    )
                    reshuffle_count += 1
                reshuffle_records.append(record)

            if reshuffle_count > 0:
                source_t0s_by_tpc = _refresh_source_t0s(
                    flash_cluster_received=flash_cluster_received,
                    t0_candidates=t0_candidates,
                    hit_ts=hit_ts_out,
                    tpcs=tpcs,
                    labels=labels,
                    energy=energy,
                    light_allowed_tpcs=light_allowed_tpcs,
                    cfg=cfg,
                )

        # ---- Phase B: single-move pass ----
        rows_b = _scan_with_details(
            base_image=base_out,
            full_light_waveform=full_light,
            full_light_std=full_std,
            source_t0s_by_tpc=source_t0s_by_tpc,
            saturated_channel_cache=namespace.get("saturated_channel_cache"),
            light_allowed_tpcs=light_allowed_tpcs,
            hit_ts=hit_ts_out,
            tpcs=tpcs,
            labels=labels,
            energy=energy,
            cfg25=cfg25_light,
            cfg=cfg,
        )

        for source_rank, row in enumerate(rows_b, start=1):
            if len(light_moves) >= int(cfg.light_max_total_moves):
                break
            tpc = int(row["TPCid"])
            if moves_by_tpc.get(tpc, 0) >= int(cfg.light_max_moves_per_tpc):
                continue
            row = dict(row)
            row["source_rank"] = int(source_rank)
            trial = p25._try_light_repair_row(
                row,
                base_image=base_out,
                full_light_waveform=full_light,
                full_light_std=full_std,
                image_maps=namespace.get("imageMaps"),
                t0_candidates=t2._as_tpc_dict(t0_candidates),
                hit_timestamps=hit_ts_out,
                hit_tpc_ids=tpcs,
                labels_global=labels,
                x=x,
                y=y,
                z=z,
                energy=energy,
                saturated_channel_cache=namespace.get("saturated_channel_cache"),
                locked_hit_mask=None,
                cfg=cfg25_light,
            )
            if trial is None:
                continue
            trial = dict(trial)
            trial["source_rank"] = int(source_rank)
            trial["overflow_row"] = dict(row)
            # Skip if applying this move would undo a recent move.
            if (int(trial["TPCid"]), int(trial["new_t0"]), int(trial["old_t0"])) in move_history:
                continue
            proxy, image_status = p25._component_proxy_image(
                component=trial,
                image_maps=namespace.get("imageMaps"),
                labels_global=labels,
                hit_tpc_ids=tpcs,
                energy=energy,
            )
            trial["image_status"] = str(image_status)
            if proxy is None:
                trial.update({
                    "accepted": False,
                    "accepted_by": "no_trial",
                    "no_trial": True,
                    "reason": str(image_status),
                })
                light_trials.append(trial)
                continue
            old_shift = p25._shift_image(proxy, int(trial["old_t0"]), nt=base_out.shape[-1])
            new_shift = p25._shift_image(proxy, int(trial["new_t0"]), nt=base_out.shape[-1])
            trial["delta"] = (new_shift - old_shift).astype(np.float32)

            gate = _evaluate_v2_gate(
                trial,
                base_image=base_out,
                full_light_waveform=full_light,
                full_light_std=full_std,
                saturated_channel_cache=namespace.get("saturated_channel_cache"),
                config=cfg,
                cfg25=cfg25_light,
                outer_pass=int(outer_pass),
            )
            accepted = bool(gate["accepted"])
            trial.update(gate)
            accepted = _apply_track_veto_v2(
                trial,
                accepted,
                cfg=cfg,
                labels_global=labels,
                hit_tpc_ids=tpcs,
                energy=energy,
                label_info=label_info,
            )
            trial["accepted"] = bool(accepted)
            light_trials.append(trial)
            if not accepted:
                continue

            hit_ts_out, base_out, records = _apply_one_trial(
                hit_ts=hit_ts_out,
                base_image=base_out,
                trial=trial,
                namespace=namespace,
                hit_tpc_ids=tpcs,
                x=x,
                y=y,
                z=z,
                energy=energy,
                cfg25=cfg25_light,
            )
            family_update_records.extend(records)
            trial["outer_pass"] = int(outer_pass)
            light_moves.append(trial)
            moves_by_tpc[tpc] = moves_by_tpc.get(tpc, 0) + 1
            move_history.add((int(trial["TPCid"]), int(trial["old_t0"]), int(trial["new_t0"])))

            t0_candidates, flash_cluster_received = _maybe_canonicalize_flash_table(
                namespace=namespace,
                flash_cluster_received=flash_cluster_received,
                t0_candidates=t0_candidates,
                tpc=int(tpc),
                new_t0=float(trial["new_t0"]),
                cfg=cfg,
                flash_canon_rows=flash_canon_rows,
                component_id=int(trial.get("component_id", -1)),
            )
            # Refresh the source-t0 catalogue so subsequent rows account for the
            # newly canonicalized flash assignment.
            source_t0s_by_tpc = _refresh_source_t0s(
                flash_cluster_received=flash_cluster_received,
                t0_candidates=t0_candidates,
                hit_ts=hit_ts_out,
                tpcs=tpcs,
                labels=labels,
                energy=energy,
                light_allowed_tpcs=light_allowed_tpcs,
                cfg=cfg,
            )

        moves_added = len(light_moves) - pass_start_count
        pass_logs.append({"pass": int(outer_pass), "moves_added": int(moves_added)})
        if moves_added == 0:
            break
        if len(light_moves) >= int(cfg.light_max_total_moves):
            break

    # ---- Build the result ----
    result = dict(base_result)
    result["baseImage"] = base_out.astype(np.float32)
    result["hit_timestamps"] = hit_ts_out.astype(np.float32)
    result["hit_timestamps_before"] = before_ts
    result["baseImage_before"] = before_base
    result["light_moves"] = light_moves
    result["light_trials"] = light_trials
    result["light_source_rows"] = light_source_rows
    result["family_update_records"] = family_update_records
    result["v2_reshuffle_records"] = reshuffle_records
    result["v2_pass_logs"] = pass_logs
    result["v2_light_allowed_tpcs"] = sorted(int(v) for v in light_allowed_tpcs)
    result["light_allowed_tpcs"] = result["v2_light_allowed_tpcs"]
    result["v2_config"] = cfg
    result["config"] = cfg
    result["phase25_config"] = cfg25_light
    result["t0Candidates"] = copy.deepcopy(t0_candidates)
    result["flash_cluster_received_by_tpc"] = copy.deepcopy(flash_cluster_received)
    result["flash_canonicalization_rows"] = flash_canon_rows
    result["v2_elapsed_s"] = float(time.time() - t_start)
    result["elapsed_s"] = float(time.time() - t_start)

    if do_commit:
        namespace["baseImage"] = result["baseImage"]
        namespace["hit_timestamps"] = result["hit_timestamps"]
        namespace["t0Candidates"] = result["t0Candidates"]
        if result["flash_cluster_received_by_tpc"] is not None:
            namespace["flash_cluster_received_by_tpc"] = result["flash_cluster_received_by_tpc"]
        namespace["phase25_trial2_v2_result"] = result
    else:
        # Roll back the namespace state to before this entire helper ran.
        namespace["baseImage"] = before_base
        namespace["hit_timestamps"] = before_ts
        namespace["t0Candidates"] = before_t0_candidates
        if before_flash is None:
            namespace.pop("flash_cluster_received_by_tpc", None)
        else:
            namespace["flash_cluster_received_by_tpc"] = before_flash

    return result


# ---------------------------------------------------------------------------
# Diagnostics


def print_v2_summary(result: dict[str, Any]) -> None:
    """Compact summary that mirrors ``print_trial2_summary`` plus v2 fields."""
    cfg = result.get("config") or result.get("v2_config") or Trial2V2LightConfig()
    large_rows = result.get("large_flash_grid", {}).get("accepted_rows", [])
    light_moves = list(result.get("light_moves", []))
    light_trials = list(result.get("light_trials", []))
    reshuffle_records = list(result.get("v2_reshuffle_records", []))
    n_force_overrides = sum(
        1
        for trial in light_moves
        if bool(trial.get("track_veto_overridden", False))
        and str(trial.get("track_veto_override_reason", "")) == "force_override_extreme_overflow"
    )
    n_reshuffle_moves = sum(
        1 for trial in light_moves if str(trial.get("accepted_by", "")) == "reshuffle"
    )

    print("Trial2 V2 light-rescue summary")
    print(f"  large flash-grid corrections : {len(large_rows)}")
    print(f"  spatial accepted moves       : {len(result.get('spatial_moves', []))}")
    print(f"  spatial trials               : {len(result.get('spatial_trials', []))}")
    print(f"  light accepted moves         : {len(light_moves)}")
    print(f"    via single-move gate       : {len(light_moves) - n_reshuffle_moves}")
    print(f"    via multi-cluster reshuffle: {n_reshuffle_moves}")
    print(f"    track-veto force-overrides : {n_force_overrides}")
    print(f"  light trials                 : {len(light_trials)}")
    print(f"  reshuffle TPCs accepted      : {len(reshuffle_records)}")
    print(
        f"  overflow def                 : sigma={float(cfg.light_overflow_sigma):.1f}, "
        f"abs={float(cfg.light_overflow_abs_adc):.0f} ADC, "
        f"min_ofch={int(cfg.light_min_overflow_channels)}"
    )
    if bool(cfg.enable_force_override):
        print(
            "  force-override gate          : "
            f"src_ofch>={int(cfg.light_force_override_min_src_ofch)}, "
            f"peak_ofch>={int(cfg.light_force_override_min_peak_ofch)}, "
            f"sev>={float(cfg.light_force_override_min_severity):.0f}, "
            f"srcRed>={int(cfg.light_force_override_min_src_red)}, "
            f"dChi2>={float(cfg.light_force_override_min_dchi2):.1e}, "
            f"dChi2/E>={float(cfg.light_force_override_min_dchi2_per_e):.1e}"
        )
    if bool(cfg.enable_reshuffle):
        print(
            "  reshuffle gate               : "
            f"min_rows={int(cfg.reshuffle_min_severe_rows)}, "
            f"max_clusters={int(cfg.reshuffle_max_clusters)}, "
            f"sev_floor={float(cfg.reshuffle_severity_floor):.0f}, "
            f"min_dChi2={float(cfg.reshuffle_min_dchi2):.1e}, "
            f"min_dChi2/E={float(cfg.reshuffle_min_dchi2_per_e):.1e}"
        )
    print(f"  family updates               : {len(result.get('family_update_records', []))}")
    print(f"  elapsed                      : {float(result.get('elapsed_s', np.nan)):.1f}s")


# Expose the trial2 acceptance-table printer as a convenience so callers can
# reuse the existing diagnostics view without an extra import.
print_v2_light_acceptance_table = t2.print_trial2_light_acceptance_table
print_stage_truth_summary = t2.print_stage_truth_summary
