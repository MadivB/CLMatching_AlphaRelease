from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotting is optional in batch contexts
    go = None

try:
    from v3_2_global_matching import _shift_block
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from M5p1.v3_2_global_matching import _shift_block


PULSE_PEAK_TICK = 105
ADC_CLIP = 60780.0


@dataclass
class ComponentCandidate:
    component_id: int
    clusterid: int
    tpcid: int
    hit_indices: np.ndarray
    energy_mev: float
    source_t0: int | None
    reason: str
    geometry: dict[str, float]
    energy_fraction_of_cluster: float


def _veto_mask_for_tpc(
    actual_tpc: np.ndarray,
    saturated_channel_cache: dict[str, Any] | None,
    tpcid: int,
) -> np.ndarray:
    if isinstance(saturated_channel_cache, dict) and "veto_mask" in saturated_channel_cache:
        return np.asarray(saturated_channel_cache["veto_mask"][int(tpcid)], dtype=bool)
    return np.sum(np.asarray(actual_tpc, dtype=np.float32) > 60700.0, axis=1) > 6


def _weighted_loss(
    model: np.ndarray,
    actual: np.ndarray,
    std: np.ndarray,
    time_mask: np.ndarray,
    *,
    overflow_weight: float = 3.0,
) -> float:
    if not np.any(time_mask):
        time_mask = np.ones(model.shape[-1], dtype=bool)
    m = np.asarray(model[:, time_mask], dtype=np.float32)
    a = np.asarray(actual[:, time_mask], dtype=np.float32)
    s = np.maximum(np.asarray(std[:, time_mask], dtype=np.float32), 1e-6)
    w = np.where(m > a, float(overflow_weight), 1.0).astype(np.float32)
    return float(np.sum(((m - a) ** 2 / s) * w))


def _focus_mask_for_t0s(
    n_ticks: int,
    t0s: list[int],
    *,
    pulse_peak_tick: int = PULSE_PEAK_TICK,
    half_window_ticks: int = 18,
    pad_ticks: int = 12,
) -> np.ndarray:
    mask = np.zeros(int(n_ticks), dtype=bool)
    for t0 in sorted(set(int(v) for v in t0s if v is not None)):
        tick = int(t0) + int(pulse_peak_tick)
        lo = max(0, tick - int(half_window_ticks) - int(pad_ticks))
        hi = min(int(n_ticks), tick + int(half_window_ticks) + int(pad_ticks) + 1)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def dominant_reco_t0(hit_timestamps: np.ndarray, hit_indices: np.ndarray) -> int | None:
    values = np.asarray(hit_timestamps[np.asarray(hit_indices, dtype=np.int64)], dtype=np.float32)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return None
    uniq, counts = np.unique(np.round(values).astype(np.int32), return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def cluster_geometry_metrics(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
) -> dict[str, float]:
    pts = np.column_stack([x, y, z]).astype(np.float64)
    if pts.shape[0] == 0:
        return {
            "n_hits": 0,
            "energy_mev": 0.0,
            "length_cm": 0.0,
            "linearity": 0.0,
            "planarity": 0.0,
            "transverse_rms_cm": 0.0,
            "branchiness": 0.0,
        }

    weights = np.asarray(energy, dtype=np.float64)
    weights = np.clip(weights, 1e-8, None)
    weights = weights / np.sum(weights)
    centroid = np.sum(pts * weights[:, None], axis=0)
    centered = pts - centroid

    if pts.shape[0] < 3:
        length = float(np.max(np.linalg.norm(centered, axis=1)) * 2.0) if pts.shape[0] else 0.0
        return {
            "n_hits": int(pts.shape[0]),
            "energy_mev": float(np.sum(energy)),
            "length_cm": length,
            "linearity": 1.0 if pts.shape[0] > 1 else 0.0,
            "planarity": 0.0,
            "transverse_rms_cm": 0.0,
            "branchiness": 0.0,
        }

    cov = (centered * weights[:, None]).T @ centered
    evals, evecs = np.linalg.eigh(cov)
    evals = np.sort(np.clip(evals, 0.0, None))[::-1]
    total = float(np.sum(evals))
    linearity = float(evals[0] / max(total, 1e-12))
    planarity = float((evals[0] + evals[1]) / max(total, 1e-12))

    direction = evecs[:, int(np.argmax(np.linalg.eigvalsh(cov)))]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    proj = centered @ direction
    length = float(np.max(proj) - np.min(proj))
    transverse = centered - proj[:, None] * direction[None, :]
    transverse_rms = float(np.sqrt(np.sum(weights * np.sum(transverse * transverse, axis=1))))
    branchiness = float(np.clip((1.0 - linearity) + transverse_rms / max(length, 1e-6), 0.0, 10.0))

    return {
        "n_hits": int(pts.shape[0]),
        "energy_mev": float(np.sum(energy)),
        "length_cm": length,
        "linearity": linearity,
        "planarity": planarity,
        "transverse_rms_cm": transverse_rms,
        "branchiness": branchiness,
    }


def build_phase3_assignment_audit(
    *,
    assignment_log: list[dict[str, Any]],
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    hit_timestamps: np.ndarray,
    cluster_energies: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    cluster_energies = {} if cluster_energies is None else cluster_energies
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)

    rows: list[dict[str, Any]] = []
    for item in assignment_log:
        cid = int(item.get("clusterid", item.get("cluster_id", -1)))
        if cid < 0:
            continue
        cmask = labels_global == cid
        hit_indices = np.flatnonzero(cmask).astype(np.int64)
        tpcs = [int(v) for v in np.unique(hit_tpc_ids[cmask]).tolist()] if np.any(cmask) else []
        row = {
            "clusterid": int(cid),
            "tpcs": [int(v) for v in item.get("tpcs", tpcs)],
            "n_hits": int(hit_indices.size),
            "energy_mev": float(item.get("energy", cluster_energies.get(cid, 0.0))),
            "assigned": bool(item.get("assigned", False)),
            "t0": None if item.get("t0", -1) is None else int(item.get("t0", -1)),
            "mode": str(item.get("mode", "")),
            "label": str(item.get("label", "")),
            "improvement": float(item.get("improvement", 0.0)),
            "improvement_per_mev": float(item.get("improvement", 0.0))
            / max(float(item.get("energy", cluster_energies.get(cid, 0.0))), 1e-6),
            "normalized_score": float(item.get("normalized_score", np.nan)),
            "objective_t0": item.get("objective_t0", None),
            "n_fit_channels": int(item.get("n_fit_channels", -1)),
            "dominant_reco_t0": dominant_reco_t0(hit_timestamps, hit_indices),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            not bool(r["assigned"]),
            -float(r["energy_mev"]),
            int(r["clusterid"]),
        )
    )
    return rows


def scan_residual_holes(
    *,
    full_light_waveform: np.ndarray,
    base_image: np.ndarray,
    t0_candidates: list[list[int]],
    hit_tpc_ids: np.ndarray | None = None,
    saturated_channel_cache: dict[str, Any] | None = None,
    min_peak_missing_fraction: float = 0.50,
    pulse_peak_tick: int = PULSE_PEAK_TICK,
    deficit_half_window_ticks: int = 18,
    exclude_t0_zero: bool = True,
    ignore_brightest_flash_peak: bool = True,
) -> list[dict[str, Any]]:
    actual_full = np.asarray(full_light_waveform, dtype=np.float32)
    model_full = np.asarray(base_image, dtype=np.float32)
    n_tpcs = min(actual_full.shape[0], model_full.shape[0], len(t0_candidates))

    hit_counts = None
    if hit_tpc_ids is not None:
        hit_counts = np.bincount(np.asarray(hit_tpc_ids, dtype=np.int64), minlength=n_tpcs)

    rows: list[dict[str, Any]] = []
    for tpc in range(n_tpcs):
        if hit_counts is not None and int(hit_counts[tpc]) == 0:
            continue

        cand = []
        seen = set()
        for value in t0_candidates[int(tpc)]:
            if value is None:
                continue
            t0 = int(round(float(value)))
            if bool(exclude_t0_zero) and t0 == 0:
                continue
            if t0 in seen:
                continue
            seen.add(t0)
            cand.append(t0)
        if not cand:
            continue

        actual_tpc = actual_full[int(tpc)]
        model_tpc = model_full[int(tpc)]
        veto_mask = _veto_mask_for_tpc(actual_tpc, saturated_channel_cache, int(tpc))
        keep_idx = np.flatnonzero(~veto_mask)
        if keep_idx.size == 0:
            continue

        actual_sum = np.sum(actual_tpc[keep_idx], axis=0)
        model_sum = np.sum(model_tpc[keep_idx], axis=0)

        peak_info = []
        for t0 in cand:
            tick = int(t0) + int(pulse_peak_tick)
            if 0 <= tick < actual_sum.shape[0]:
                peak_info.append((int(t0), int(tick), float(actual_sum[tick])))
        if not peak_info:
            continue

        brightest = None
        if bool(ignore_brightest_flash_peak):
            brightest = max(peak_info, key=lambda x: x[2])[0]

        for t0, tick, peak_actual in peak_info:
            if brightest is not None and int(t0) == int(brightest):
                continue
            peak_model = float(model_sum[tick])
            peak_missing = max(float(peak_actual) - peak_model, 0.0)
            peak_missing_fraction = peak_missing / max(float(peak_actual), 1e-9)
            if peak_missing_fraction < float(min_peak_missing_fraction):
                continue
            lo = max(0, int(tick) - int(deficit_half_window_ticks))
            hi = min(actual_sum.shape[0], int(tick) + int(deficit_half_window_ticks) + 1)
            window_actual = float(np.sum(actual_sum[lo:hi]))
            window_missing = float(np.sum(np.clip(actual_sum[lo:hi] - model_sum[lo:hi], 0.0, None)))
            rows.append(
                {
                    "TPCid": int(tpc),
                    "t0": int(t0),
                    "peak_tick": int(tick),
                    "window_lo": int(lo),
                    "window_hi": int(hi),
                    "peak_actual": float(peak_actual),
                    "peak_model": float(peak_model),
                    "peak_missing": float(peak_missing),
                    "peak_missing_fraction": float(peak_missing_fraction),
                    "window_actual": float(window_actual),
                    "window_missing": float(window_missing),
                    "window_missing_fraction": float(window_missing / max(window_actual, 1e-9)),
                    "n_unsaturated_channels": int(keep_idx.size),
                    "n_tpc_hits": int(hit_counts[tpc]) if hit_counts is not None else -1,
                }
            )

    rows.sort(key=lambda r: (-float(r["peak_missing_fraction"]), -float(r["peak_missing"]), int(r["TPCid"])))
    return rows


def find_suspicious_clusters(
    *,
    audit_rows: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    target_tpc: int | None = None,
    min_energy_mev: float = 1.0,
    min_hits: int = 8,
    low_linearity_threshold: float = 0.82,
    high_transverse_rms_cm: float = 4.0,
    weak_improvement_per_mev: float = 0.0,
    matrix_norm_suspicious: float = 0.10,
) -> list[dict[str, Any]]:
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    audit_by_cid = {int(r["clusterid"]): r for r in audit_rows}

    candidate_ids = sorted(audit_by_cid)
    out: list[dict[str, Any]] = []
    for cid in candidate_ids:
        cmask = labels_global == int(cid)
        if target_tpc is not None:
            cmask &= hit_tpc_ids == int(target_tpc)
        if int(np.count_nonzero(cmask)) < int(min_hits):
            continue

        geom = cluster_geometry_metrics(x[cmask], y[cmask], z[cmask], energy[cmask])
        if float(geom["energy_mev"]) < float(min_energy_mev):
            continue

        audit = audit_by_cid[int(cid)]
        norm = float(audit.get("normalized_score", np.nan))
        improvement_per_mev = float(audit.get("improvement_per_mev", 0.0))
        reasons = []
        if not bool(audit.get("assigned", False)):
            reasons.append("unassigned")
        if float(geom["linearity"]) < float(low_linearity_threshold):
            reasons.append("low_linearity")
        if float(geom["transverse_rms_cm"]) > float(high_transverse_rms_cm):
            reasons.append("wide_cluster")
        if np.isfinite(norm) and norm > float(matrix_norm_suspicious):
            reasons.append("weak_matrix_margin")
        if improvement_per_mev <= float(weak_improvement_per_mev):
            reasons.append("weak_light_improvement")

        if not reasons:
            continue
        row = dict(audit)
        row.update({f"geom_{k}": v for k, v in geom.items()})
        row["suspicious_reasons"] = ",".join(reasons)
        out.append(row)

    out.sort(
        key=lambda r: (
            -len(str(r["suspicious_reasons"]).split(",")),
            -float(r["energy_mev"]),
            int(r["clusterid"]),
        )
    )
    return out


def _connected_components_from_edges(n: int, edges: list[tuple[int, int]]) -> list[np.ndarray]:
    parent = np.arange(int(n), dtype=np.int32)

    def find(a: int) -> int:
        while int(parent[a]) != int(a):
            parent[a] = parent[int(parent[a])]
            a = int(parent[a])
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(int(a), int(b))

    groups: dict[int, list[int]] = {}
    for i in range(int(n)):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def split_cluster_into_spatial_components(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    global_hit_indices: np.ndarray,
    min_component_hits: int = 6,
    min_component_energy_mev: float = 0.5,
    edge_scale: float = 1.7,
    max_edge_cm: float = 5.0,
) -> list[dict[str, Any]]:
    pts = np.column_stack([x, y, z]).astype(np.float64)
    n = int(pts.shape[0])
    if n == 0:
        return []
    if n == 1:
        comps = [np.asarray([0], dtype=np.int64)]
    else:
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        nn = np.min(dist, axis=1)
        median_nn = float(np.median(nn[np.isfinite(nn)])) if np.any(np.isfinite(nn)) else float(max_edge_cm)
        edge_limit = min(float(max_edge_cm), float(edge_scale) * max(median_nn, 1e-6))
        ii, jj = np.where(dist <= edge_limit)
        edges = [(int(i), int(j)) for i, j in zip(ii.tolist(), jj.tolist()) if int(i) < int(j)]
        comps = _connected_components_from_edges(n, edges)

    rows: list[dict[str, Any]] = []
    for comp_id, local_idx in enumerate(comps):
        comp_energy = float(np.sum(np.asarray(energy, dtype=np.float64)[local_idx]))
        if local_idx.size < int(min_component_hits) or comp_energy < float(min_component_energy_mev):
            continue
        geom = cluster_geometry_metrics(
            np.asarray(x)[local_idx],
            np.asarray(y)[local_idx],
            np.asarray(z)[local_idx],
            np.asarray(energy)[local_idx],
        )
        rows.append(
            {
                "component_id": int(comp_id),
                "local_indices": np.asarray(local_idx, dtype=np.int64),
                "hit_indices": np.asarray(global_hit_indices, dtype=np.int64)[local_idx],
                "energy_mev": float(comp_energy),
                "geometry": geom,
            }
        )
    rows.sort(key=lambda r: (-float(r["energy_mev"]), int(r["component_id"])))
    return rows


def build_component_candidates_for_tpc(
    *,
    tpcid: int,
    suspicious_rows: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    hit_timestamps: np.ndarray,
    max_clusters: int = 20,
    max_components_per_cluster: int = 6,
    min_component_hits: int = 6,
    min_component_energy_mev: float = 0.5,
    edge_scale: float = 1.7,
    max_edge_cm: float = 5.0,
) -> list[ComponentCandidate]:
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    candidates: list[ComponentCandidate] = []

    for row in suspicious_rows[: int(max_clusters)]:
        cid = int(row["clusterid"])
        mask = (labels_global == cid) & (hit_tpc_ids == int(tpcid))
        hit_idx = np.flatnonzero(mask).astype(np.int64)
        if hit_idx.size == 0:
            continue
        total_energy = float(np.sum(np.asarray(energy, dtype=np.float64)[hit_idx]))
        components = split_cluster_into_spatial_components(
            x=np.asarray(x)[hit_idx],
            y=np.asarray(y)[hit_idx],
            z=np.asarray(z)[hit_idx],
            energy=np.asarray(energy)[hit_idx],
            global_hit_indices=hit_idx,
            min_component_hits=int(min_component_hits),
            min_component_energy_mev=float(min_component_energy_mev),
            edge_scale=float(edge_scale),
            max_edge_cm=float(max_edge_cm),
        )
        if len(components) <= 1:
            geom = cluster_geometry_metrics(
                np.asarray(x)[hit_idx],
                np.asarray(y)[hit_idx],
                np.asarray(z)[hit_idx],
                np.asarray(energy)[hit_idx],
            )
            components = [
                {
                    "component_id": 0,
                    "hit_indices": hit_idx,
                    "energy_mev": total_energy,
                    "geometry": geom,
                }
            ]

        for comp in components[: int(max_components_per_cluster)]:
            comp_hit_idx = np.asarray(comp["hit_indices"], dtype=np.int64)
            candidates.append(
                ComponentCandidate(
                    component_id=len(candidates),
                    clusterid=int(cid),
                    tpcid=int(tpcid),
                    hit_indices=comp_hit_idx,
                    energy_mev=float(comp["energy_mev"]),
                    source_t0=dominant_reco_t0(hit_timestamps, comp_hit_idx),
                    reason=str(row.get("suspicious_reasons", "")),
                    geometry=dict(comp["geometry"]),
                    energy_fraction_of_cluster=float(comp["energy_mev"]) / max(total_energy, 1e-9),
                )
            )

    candidates.sort(key=lambda c: (-float(c.energy_mev), int(c.clusterid), int(c.component_id)))
    return candidates


def _component_image_from_fraction(
    *,
    component: ComponentCandidate,
    image_maps: dict[tuple[int, int], np.ndarray],
) -> np.ndarray | None:
    key = (int(component.clusterid), int(component.tpcid))
    if key not in image_maps:
        return None
    return (
        np.asarray(image_maps[key], dtype=np.float32)
        * float(np.clip(component.energy_fraction_of_cluster, 0.0, 1.0))
    )


def score_component_reassignment_options(
    *,
    components: list[ComponentCandidate],
    residual_holes: list[dict[str, Any]],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    saturated_channel_cache: dict[str, Any] | None = None,
    overflow_weight: float = 3.0,
    source_protection_weight: float = 1.0,
    half_window_ticks: int = 18,
    pad_ticks: int = 12,
    adc_clip: float = ADC_CLIP,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    holes_by_tpc: dict[int, list[dict[str, Any]]] = {}
    for hole in residual_holes:
        holes_by_tpc.setdefault(int(hole["TPCid"]), []).append(hole)

    for comp in components:
        tpcid = int(comp.tpcid)
        holes = holes_by_tpc.get(tpcid, [])
        if not holes:
            continue
        image = _component_image_from_fraction(component=comp, image_maps=image_maps)
        if image is None:
            continue

        actual_tpc = np.asarray(full_light_waveform[tpcid], dtype=np.float32)
        model_tpc = np.asarray(base_image[tpcid], dtype=np.float32)
        std_tpc = np.asarray(full_light_std[tpcid], dtype=np.float32)
        veto_mask = _veto_mask_for_tpc(actual_tpc, saturated_channel_cache, tpcid)
        keep_idx = np.flatnonzero(~veto_mask)
        if keep_idx.size == 0:
            continue

        image_kept = np.asarray(image[keep_idx], dtype=np.float32)
        model_kept = np.asarray(model_tpc[keep_idx], dtype=np.float32)
        actual_kept = np.asarray(actual_tpc[keep_idx], dtype=np.float32)
        std_kept = np.asarray(std_tpc[keep_idx], dtype=np.float32)

        for hole in holes:
            target_t0 = int(hole["t0"])
            source_t0 = comp.source_t0
            if source_t0 is not None and int(source_t0) == int(target_t0):
                continue

            focus_t0s = [int(target_t0)]
            if source_t0 is not None:
                focus_t0s.append(int(source_t0))
            time_mask = _focus_mask_for_t0s(
                model_kept.shape[-1],
                focus_t0s,
                half_window_ticks=int(half_window_ticks),
                pad_ticks=int(pad_ticks),
            )

            target_add = _shift_block(image_kept[None, :, :], int(target_t0), baseline=0.0)[0]
            if source_t0 is None:
                source_sub = np.zeros_like(target_add, dtype=np.float32)
            else:
                source_sub = _shift_block(image_kept[None, :, :], int(source_t0), baseline=0.0)[0]

            before = _weighted_loss(
                model_kept,
                actual_kept,
                std_kept,
                time_mask,
                overflow_weight=float(overflow_weight),
            )
            after_model = np.clip(
                model_kept + target_add - float(source_protection_weight) * source_sub,
                0.0,
                float(adc_clip),
            )
            after = _weighted_loss(
                after_model,
                actual_kept,
                std_kept,
                time_mask,
                overflow_weight=float(overflow_weight),
            )

            peak_tick = int(hole["peak_tick"])
            target_peak_add = float(np.sum(target_add[:, peak_tick])) if 0 <= peak_tick < target_add.shape[1] else 0.0
            source_peak_sub = 0.0
            if source_t0 is not None:
                source_tick = int(source_t0) + PULSE_PEAK_TICK
                if 0 <= source_tick < source_sub.shape[1]:
                    source_peak_sub = float(np.sum(source_sub[:, source_tick]))

            overflow_after = np.clip(after_model - actual_kept, 0.0, None)
            overflow_sum = float(np.sum(overflow_after[:, time_mask]))
            improvement = float(before - after)
            score_per_mev = improvement / max(float(comp.energy_mev), 1e-9)
            fill_fraction = target_peak_add / max(float(hole["peak_missing"]), 1e-9)

            rows.append(
                {
                    "component_id": int(comp.component_id),
                    "clusterid": int(comp.clusterid),
                    "tpcid": int(tpcid),
                    "target_t0": int(target_t0),
                    "source_t0": None if source_t0 is None else int(source_t0),
                    "n_hits": int(comp.hit_indices.size),
                    "energy_mev": float(comp.energy_mev),
                    "loss_before": float(before),
                    "loss_after": float(after),
                    "delta_loss": float(improvement),
                    "delta_loss_per_mev": float(score_per_mev),
                    "target_peak_add": float(target_peak_add),
                    "source_peak_sub": float(source_peak_sub),
                    "target_peak_fill_fraction": float(fill_fraction),
                    "overflow_sum": float(overflow_sum),
                    "geometry_linearity": float(comp.geometry.get("linearity", np.nan)),
                    "geometry_transverse_rms_cm": float(comp.geometry.get("transverse_rms_cm", np.nan)),
                    "suspicious_reason": comp.reason,
                    "hit_indices": comp.hit_indices.copy(),
                }
            )

    rows.sort(
        key=lambda r: (
            -float(r["delta_loss_per_mev"]),
            -float(r["delta_loss"]),
            float(r["overflow_sum"]),
            -float(r["energy_mev"]),
        )
    )
    return rows


def _score_add_component(
    *,
    image: np.ndarray,
    tpcid: int,
    target_t0: int,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    saturated_channel_cache: dict[str, Any] | None,
    overflow_weight: float,
    half_window_ticks: int,
    pad_ticks: int,
    adc_clip: float,
) -> dict[str, Any] | None:
    actual_tpc = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
    model_tpc = np.asarray(base_image[int(tpcid)], dtype=np.float32)
    std_tpc = np.asarray(full_light_std[int(tpcid)], dtype=np.float32)
    veto_mask = _veto_mask_for_tpc(actual_tpc, saturated_channel_cache, int(tpcid))
    keep_idx = np.flatnonzero(~veto_mask)
    if keep_idx.size == 0:
        return None

    model_kept = model_tpc[keep_idx]
    actual_kept = actual_tpc[keep_idx]
    std_kept = std_tpc[keep_idx]
    shifted = _shift_block(np.asarray(image[keep_idx], dtype=np.float32)[None, :, :], int(target_t0), baseline=0.0)[0]
    time_mask = _focus_mask_for_t0s(
        model_kept.shape[-1],
        [int(target_t0)],
        half_window_ticks=int(half_window_ticks),
        pad_ticks=int(pad_ticks),
    )
    before = _weighted_loss(
        model_kept,
        actual_kept,
        std_kept,
        time_mask,
        overflow_weight=float(overflow_weight),
    )
    after_model = np.clip(model_kept + shifted, 0.0, float(adc_clip))
    after = _weighted_loss(
        after_model,
        actual_kept,
        std_kept,
        time_mask,
        overflow_weight=float(overflow_weight),
    )
    peak_tick = int(target_t0) + PULSE_PEAK_TICK
    peak_add = float(np.sum(shifted[:, peak_tick])) if 0 <= peak_tick < shifted.shape[1] else 0.0
    overflow = float(np.sum(np.clip(after_model[:, time_mask] - actual_kept[:, time_mask], 0.0, None)))
    return {
        "loss_before": float(before),
        "loss_after": float(after),
        "delta_loss": float(before - after),
        "peak_add": float(peak_add),
        "overflow_sum": float(overflow),
    }


def _score_move_component(
    *,
    image: np.ndarray,
    tpcid: int,
    source_t0: int,
    target_t0: int,
    base_image: np.ndarray,
    full_light_waveform: np.ndarray,
    full_light_std: np.ndarray,
    saturated_channel_cache: dict[str, Any] | None,
    overflow_weight: float,
    half_window_ticks: int,
    pad_ticks: int,
    adc_clip: float,
) -> dict[str, Any] | None:
    actual_tpc = np.asarray(full_light_waveform[int(tpcid)], dtype=np.float32)
    model_tpc = np.asarray(base_image[int(tpcid)], dtype=np.float32)
    std_tpc = np.asarray(full_light_std[int(tpcid)], dtype=np.float32)
    veto_mask = _veto_mask_for_tpc(actual_tpc, saturated_channel_cache, int(tpcid))
    keep_idx = np.flatnonzero(~veto_mask)
    if keep_idx.size == 0:
        return None

    model_kept = model_tpc[keep_idx]
    actual_kept = actual_tpc[keep_idx]
    std_kept = std_tpc[keep_idx]
    image_kept = np.asarray(image[keep_idx], dtype=np.float32)
    add = _shift_block(image_kept[None, :, :], int(target_t0), baseline=0.0)[0]
    sub = _shift_block(image_kept[None, :, :], int(source_t0), baseline=0.0)[0]
    time_mask = _focus_mask_for_t0s(
        model_kept.shape[-1],
        [int(source_t0), int(target_t0)],
        half_window_ticks=int(half_window_ticks),
        pad_ticks=int(pad_ticks),
    )
    before = _weighted_loss(
        model_kept,
        actual_kept,
        std_kept,
        time_mask,
        overflow_weight=float(overflow_weight),
    )
    after_model = np.clip(model_kept + add - sub, 0.0, float(adc_clip))
    after = _weighted_loss(
        after_model,
        actual_kept,
        std_kept,
        time_mask,
        overflow_weight=float(overflow_weight),
    )
    target_tick = int(target_t0) + PULSE_PEAK_TICK
    source_tick = int(source_t0) + PULSE_PEAK_TICK
    peak_add = float(np.sum(add[:, target_tick])) if 0 <= target_tick < add.shape[1] else 0.0
    peak_sub = float(np.sum(sub[:, source_tick])) if 0 <= source_tick < sub.shape[1] else 0.0
    overflow = float(np.sum(np.clip(after_model[:, time_mask] - actual_kept[:, time_mask], 0.0, None)))
    return {
        "loss_before": float(before),
        "loss_after": float(after),
        "delta_loss": float(before - after),
        "peak_add": float(peak_add),
        "peak_sub": float(peak_sub),
        "overflow_sum": float(overflow),
    }


def _build_v4_components_for_tpc(
    *,
    tpcid: int,
    cluster_ids: list[int],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    hit_timestamps: np.ndarray,
    image_maps: dict[tuple[int, int], np.ndarray],
    cluster_energies: dict[int, float],
    split_low_linearity_threshold: float,
    split_transverse_rms_cm: float,
    min_component_hits: int,
    min_component_energy_mev: float,
    edge_scale: float,
    max_edge_cm: float,
) -> list[ComponentCandidate]:
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)
    out: list[ComponentCandidate] = []

    for cid in sorted(set(int(v) for v in cluster_ids)):
        if (int(cid), int(tpcid)) not in image_maps:
            continue
        mask = (labels_global == int(cid)) & (hit_tpc_ids == int(tpcid))
        hit_idx = np.flatnonzero(mask).astype(np.int64)
        if hit_idx.size == 0:
            continue

        total_energy = float(np.sum(np.asarray(energy, dtype=np.float64)[hit_idx]))
        if total_energy <= 0:
            total_energy = float(cluster_energies.get(int(cid), 0.0))
        geom = cluster_geometry_metrics(
            np.asarray(x)[hit_idx],
            np.asarray(y)[hit_idx],
            np.asarray(z)[hit_idx],
            np.asarray(energy)[hit_idx],
        )
        should_split = (
            int(hit_idx.size) >= 2 * int(min_component_hits)
            and (
                float(geom["linearity"]) < float(split_low_linearity_threshold)
                or float(geom["transverse_rms_cm"]) > float(split_transverse_rms_cm)
            )
        )

        if should_split:
            raw_components = split_cluster_into_spatial_components(
                x=np.asarray(x)[hit_idx],
                y=np.asarray(y)[hit_idx],
                z=np.asarray(z)[hit_idx],
                energy=np.asarray(energy)[hit_idx],
                global_hit_indices=hit_idx,
                min_component_hits=int(min_component_hits),
                min_component_energy_mev=float(min_component_energy_mev),
                edge_scale=float(edge_scale),
                max_edge_cm=float(max_edge_cm),
            )
            reason = "split_low_linearity" if float(geom["linearity"]) < float(split_low_linearity_threshold) else "split_wide_cluster"
        else:
            raw_components = []
            reason = "whole_cluster"

        if len(raw_components) <= 1:
            raw_components = [
                {
                    "component_id": 0,
                    "hit_indices": hit_idx,
                    "energy_mev": total_energy,
                    "geometry": geom,
                }
            ]
            reason = "whole_cluster"

        for comp in raw_components:
            comp_hit_idx = np.asarray(comp["hit_indices"], dtype=np.int64)
            comp_energy = float(comp["energy_mev"])
            if comp_hit_idx.size < int(min_component_hits) or comp_energy < float(min_component_energy_mev):
                continue
            out.append(
                ComponentCandidate(
                    component_id=len(out),
                    clusterid=int(cid),
                    tpcid=int(tpcid),
                    hit_indices=comp_hit_idx,
                    energy_mev=comp_energy,
                    source_t0=dominant_reco_t0(hit_timestamps, comp_hit_idx),
                    reason=reason,
                    geometry=dict(comp["geometry"]),
                    energy_fraction_of_cluster=float(comp_energy) / max(float(total_energy), 1e-9),
                )
            )

    out.sort(key=lambda c: (-float(c.energy_mev), int(c.clusterid), int(c.component_id)))
    return out


def run_component_rescue_phase3_v4(
    *,
    active_cluster_tpcs: dict[int, np.ndarray],
    iterative_single_tpc: dict[int, list[int]],
    pruned_iterative_clusters: list[int],
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
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    hit_tpc_ids: np.ndarray,
    saturated_channel_cache: dict[str, Any] | None = None,
    include_pruned_clusters: bool = True,
    min_component_hits: int = 6,
    min_component_energy_mev: float = 0.5,
    split_low_linearity_threshold: float = 0.82,
    split_transverse_rms_cm: float = 4.0,
    edge_scale: float = 1.7,
    max_edge_cm: float = 5.0,
    initial_min_delta_loss: float = 0.0,
    initial_min_delta_loss_per_mev: float = 0.0,
    correction_min_peak_missing_fraction: float = 0.35,
    correction_min_delta_loss: float = 0.0,
    correction_min_delta_loss_per_mev: float = 0.0,
    max_initial_assignments_per_tpc: int | None = None,
    max_corrections_per_tpc: int = 20,
    overflow_weight: float = 3.0,
    half_window_ticks: int = 18,
    pad_ticks: int = 12,
    adc_clip: float = ADC_CLIP,
    verbose: bool = True,
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
    base_image = np.asarray(base_image, dtype=np.float32).copy()
    hit_timestamps = np.asarray(hit_timestamps, dtype=np.float32).copy()
    labels_global = np.asarray(labels_global, dtype=np.int32)
    hit_tpc_ids = np.asarray(hit_tpc_ids, dtype=np.int32)

    assignment_log: list[dict[str, Any]] = []
    scan_loss_dict: dict[int, dict[str, Any]] = {}
    stage_stats: dict[str, Any] = {
        "v4_component_phase_enabled": True,
        "v4_assigned_components": [],
        "v4_corrected_components": [],
        "v4_split_clusters": [],
        "v4_unassigned_components": [],
        "v4_tpc_summary": {},
    }

    cluster_ids_by_tpc: dict[int, set[int]] = {
        int(tpc): set(int(cid) for cid in cids)
        for tpc, cids in iterative_single_tpc.items()
    }
    if bool(include_pruned_clusters):
        for cid in pruned_iterative_clusters:
            tpcs = active_cluster_tpcs.get(int(cid), np.asarray([], dtype=int))
            if len(tpcs) == 1:
                cluster_ids_by_tpc.setdefault(int(tpcs[0]), set()).add(int(cid))

    assigned_components: dict[int, ComponentCandidate] = {}
    component_images: dict[int, np.ndarray] = {}
    component_current_t0: dict[int, int] = {}
    component_cluster_tpc: dict[int, tuple[int, int]] = {}
    component_energy_fraction: dict[int, float] = {}

    for tpcid in sorted(cluster_ids_by_tpc):
        candidate_grid = sorted(set(int(v) for v in t0_candidates[int(tpcid)] if int(v) >= 0))
        if not candidate_grid:
            continue

        components = _build_v4_components_for_tpc(
            tpcid=int(tpcid),
            cluster_ids=sorted(cluster_ids_by_tpc[int(tpcid)]),
            x=x,
            y=y,
            z=z,
            energy=energy,
            labels_global=labels_global,
            hit_tpc_ids=hit_tpc_ids,
            hit_timestamps=hit_timestamps,
            image_maps=image_maps,
            cluster_energies=cluster_energies,
            split_low_linearity_threshold=float(split_low_linearity_threshold),
            split_transverse_rms_cm=float(split_transverse_rms_cm),
            min_component_hits=int(min_component_hits),
            min_component_energy_mev=float(min_component_energy_mev),
            edge_scale=float(edge_scale),
            max_edge_cm=float(max_edge_cm),
        )
        remaining = {int(c.component_id): c for c in components}
        for comp in components:
            if comp.reason != "whole_cluster":
                stage_stats["v4_split_clusters"].append(int(comp.clusterid))

        tpc_assignments = 0
        if verbose:
            print(f"TPC {int(tpcid):3d} | v4 components={len(components)} | t0s={candidate_grid}")

        while remaining:
            if max_initial_assignments_per_tpc is not None and tpc_assignments >= int(max_initial_assignments_per_tpc):
                break

            best = None
            for comp_id, comp in list(remaining.items()):
                image = _component_image_from_fraction(component=comp, image_maps=image_maps)
                if image is None:
                    continue
                for t0 in candidate_grid:
                    score = _score_add_component(
                        image=image,
                        tpcid=int(tpcid),
                        target_t0=int(t0),
                        base_image=base_image,
                        full_light_waveform=full_light_waveform,
                        full_light_std=full_light_std,
                        saturated_channel_cache=saturated_channel_cache,
                        overflow_weight=float(overflow_weight),
                        half_window_ticks=int(half_window_ticks),
                        pad_ticks=int(pad_ticks),
                        adc_clip=float(adc_clip),
                    )
                    if score is None:
                        continue
                    delta = float(score["delta_loss"])
                    delta_per_mev = delta / max(float(comp.energy_mev), 1e-9)
                    row = {
                        "component": comp,
                        "image": image,
                        "target_t0": int(t0),
                        "delta_loss": delta,
                        "delta_loss_per_mev": float(delta_per_mev),
                        "score": score,
                    }
                    key = (float(delta_per_mev), float(delta), -float(score["overflow_sum"]), float(comp.energy_mev))
                    if best is None or key > best["key"]:
                        row["key"] = key
                        best = row

            if best is None:
                break
            if (
                float(best["delta_loss"]) <= float(initial_min_delta_loss)
                or float(best["delta_loss_per_mev"]) <= float(initial_min_delta_loss_per_mev)
            ):
                break

            comp = best["component"]
            image = np.asarray(best["image"], dtype=np.float32)
            shifted = _shift_block(image[None, :, :], int(best["target_t0"]), baseline=0.0)[0]
            base_image[int(tpcid)] = np.clip(base_image[int(tpcid)] + shifted, 0.0, float(adc_clip))
            hit_timestamps[np.asarray(comp.hit_indices, dtype=np.int64)] = np.float32(best["target_t0"])

            assigned_components[int(comp.component_id)] = comp
            component_images[int(comp.component_id)] = image
            component_current_t0[int(comp.component_id)] = int(best["target_t0"])
            component_cluster_tpc[int(comp.component_id)] = (int(comp.clusterid), int(tpcid))
            component_energy_fraction[int(comp.component_id)] = float(comp.energy_fraction_of_cluster)
            del remaining[int(comp.component_id)]
            tpc_assignments += 1

            key = (int(comp.clusterid), int(tpcid))
            previous = dict(assignment_info.get(key, {}))
            split_rows = list(previous.get("v4_components", []))
            split_rows.append(
                {
                    "component_id": int(comp.component_id),
                    "t0": int(best["target_t0"]),
                    "n_hits": int(comp.hit_indices.size),
                    "energy": float(comp.energy_mev),
                    "energy_fraction": float(comp.energy_fraction_of_cluster),
                    "reason": str(comp.reason),
                }
            )
            assignment_info[key] = {
                **previous,
                "clusterid": int(comp.clusterid),
                "tpcid": int(tpcid),
                "t0": int(best["target_t0"]) if len({int(r["t0"]) for r in split_rows}) == 1 else np.nan,
                "mode": "v4_component_phase3",
                "stage": "v4_component_phase3",
                "assigned": True,
                "energy": float(cluster_energies.get(int(comp.clusterid), comp.energy_mev)),
                "v4_components": split_rows,
            }
            if int(tpcid) in unassigned_by_tpc:
                unassigned_by_tpc[int(tpcid)] = [
                    int(v) for v in unassigned_by_tpc[int(tpcid)] if int(v) != int(comp.clusterid)
                ]

            log_row = {
                "clusterid": int(comp.clusterid),
                "component_id": int(comp.component_id),
                "tpcs": [int(tpcid)],
                "energy": float(comp.energy_mev),
                "assigned": True,
                "mode": "v4_component_phase3",
                "label": "v4_component_phase3",
                "t0": int(best["target_t0"]),
                "improvement": float(best["delta_loss"]),
                "improvement_per_mev": float(best["delta_loss_per_mev"]),
                "n_hits": int(comp.hit_indices.size),
                "reason": str(comp.reason),
                "peak_add": float(best["score"]["peak_add"]),
                "overflow_sum": float(best["score"]["overflow_sum"]),
            }
            assignment_log.append(log_row)
            stage_stats["v4_assigned_components"].append(dict(log_row))

        for comp in remaining.values():
            log_row = {
                "clusterid": int(comp.clusterid),
                "component_id": int(comp.component_id),
                "tpcs": [int(tpcid)],
                "energy": float(comp.energy_mev),
                "assigned": False,
                "mode": "v4_component_phase3_unassigned",
                "label": "v4_component_phase3",
                "t0": -1,
                "improvement": 0.0,
                "improvement_per_mev": 0.0,
                "n_hits": int(comp.hit_indices.size),
                "reason": str(comp.reason),
            }
            assignment_log.append(log_row)
            stage_stats["v4_unassigned_components"].append(dict(log_row))

        corrections = 0
        while corrections < int(max_corrections_per_tpc):
            holes = [
                row
                for row in scan_residual_holes(
                    full_light_waveform=full_light_waveform,
                    base_image=base_image,
                    t0_candidates=t0_candidates,
                    hit_tpc_ids=hit_tpc_ids,
                    saturated_channel_cache=saturated_channel_cache,
                    min_peak_missing_fraction=float(correction_min_peak_missing_fraction),
                )
                if int(row["TPCid"]) == int(tpcid)
            ]
            if not holes:
                break

            best_move = None
            for hole in holes:
                target_t0 = int(hole["t0"])
                for comp_id, comp in assigned_components.items():
                    cid, comp_tpc = component_cluster_tpc[int(comp_id)]
                    if int(comp_tpc) != int(tpcid):
                        continue
                    source_t0 = int(component_current_t0[int(comp_id)])
                    if int(source_t0) == int(target_t0):
                        continue
                    score = _score_move_component(
                        image=component_images[int(comp_id)],
                        tpcid=int(tpcid),
                        source_t0=int(source_t0),
                        target_t0=int(target_t0),
                        base_image=base_image,
                        full_light_waveform=full_light_waveform,
                        full_light_std=full_light_std,
                        saturated_channel_cache=saturated_channel_cache,
                        overflow_weight=float(overflow_weight),
                        half_window_ticks=int(half_window_ticks),
                        pad_ticks=int(pad_ticks),
                        adc_clip=float(adc_clip),
                    )
                    if score is None:
                        continue
                    delta = float(score["delta_loss"])
                    delta_per_mev = delta / max(float(comp.energy_mev), 1e-9)
                    key = (float(delta_per_mev), float(delta), -float(score["overflow_sum"]), float(comp.energy_mev))
                    row = {
                        "component": comp,
                        "component_id": int(comp_id),
                        "source_t0": int(source_t0),
                        "target_t0": int(target_t0),
                        "delta_loss": float(delta),
                        "delta_loss_per_mev": float(delta_per_mev),
                        "score": score,
                        "key": key,
                    }
                    if best_move is None or key > best_move["key"]:
                        best_move = row

            if best_move is None:
                break
            if (
                float(best_move["delta_loss"]) <= float(correction_min_delta_loss)
                or float(best_move["delta_loss_per_mev"]) <= float(correction_min_delta_loss_per_mev)
            ):
                break

            comp_id = int(best_move["component_id"])
            image = component_images[comp_id]
            target = _shift_block(image[None, :, :], int(best_move["target_t0"]), baseline=0.0)[0]
            source = _shift_block(image[None, :, :], int(best_move["source_t0"]), baseline=0.0)[0]
            base_image[int(tpcid)] = np.clip(base_image[int(tpcid)] + target - source, 0.0, float(adc_clip))
            comp = best_move["component"]
            hit_timestamps[np.asarray(comp.hit_indices, dtype=np.int64)] = np.float32(best_move["target_t0"])
            component_current_t0[comp_id] = int(best_move["target_t0"])
            corrections += 1

            log_row = {
                "clusterid": int(comp.clusterid),
                "component_id": int(comp_id),
                "tpcs": [int(tpcid)],
                "energy": float(comp.energy_mev),
                "assigned": True,
                "mode": "v4_component_correction",
                "label": "v4_component_correction",
                "source_t0": int(best_move["source_t0"]),
                "t0": int(best_move["target_t0"]),
                "improvement": float(best_move["delta_loss"]),
                "improvement_per_mev": float(best_move["delta_loss_per_mev"]),
                "n_hits": int(comp.hit_indices.size),
                "reason": str(comp.reason),
                "peak_add": float(best_move["score"]["peak_add"]),
                "peak_sub": float(best_move["score"]["peak_sub"]),
                "overflow_sum": float(best_move["score"]["overflow_sum"]),
            }
            assignment_log.append(log_row)
            stage_stats["v4_corrected_components"].append(dict(log_row))

        stage_stats["v4_tpc_summary"][int(tpcid)] = {
            "n_components": int(len(components)),
            "n_initial_assignments": int(tpc_assignments),
            "n_corrections": int(corrections),
            "n_remaining_unassigned": int(len(remaining)),
        }
        if verbose:
            print(
                f"  TPC {int(tpcid):3d} v4 done | assigned={tpc_assignments} "
                f"corrections={corrections} unassigned={len(remaining)}"
            )

    stage_stats["v4_split_clusters"] = sorted(set(int(v) for v in stage_stats["v4_split_clusters"]))
    stage_stats["step4_clusters"] = sorted(
        set(int(row["clusterid"]) for row in assignment_log if bool(row.get("assigned", False)))
    )
    stage_stats["step4_assigned_clusters"] = list(stage_stats["step4_clusters"])

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


def apply_component_reassignment_preview(
    *,
    row: dict[str, Any],
    image_maps: dict[tuple[int, int], np.ndarray],
    base_image: np.ndarray,
    hit_timestamps: np.ndarray,
    component_energy_fraction: float | None = None,
    adc_clip: float = ADC_CLIP,
) -> tuple[np.ndarray, np.ndarray]:
    base_out = np.asarray(base_image, dtype=np.float32).copy()
    t_out = np.asarray(hit_timestamps, dtype=np.float32).copy()
    tpcid = int(row["tpcid"])
    cid = int(row["clusterid"])
    key = (cid, tpcid)
    if key not in image_maps:
        raise KeyError(f"Missing imageMaps[{key}]")
    fraction = float(row.get("energy_fraction_of_cluster", 1.0) if component_energy_fraction is None else component_energy_fraction)
    image = np.asarray(image_maps[key], dtype=np.float32) * float(np.clip(fraction, 0.0, 1.0))
    target = _shift_block(image[None, :, :], int(row["target_t0"]), baseline=0.0)[0]
    source_t0 = row.get("source_t0")
    if source_t0 is not None:
        source = _shift_block(image[None, :, :], int(source_t0), baseline=0.0)[0]
    else:
        source = np.zeros_like(target, dtype=np.float32)
    base_out[tpcid] = np.clip(base_out[tpcid] + target - source, 0.0, float(adc_clip))
    t_out[np.asarray(row["hit_indices"], dtype=np.int64)] = np.float32(row["target_t0"])
    return base_out, t_out


def print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]], *, max_rows: int = 20) -> None:
    if not rows:
        print("(no rows)")
        return
    headers = [label for _, label, _ in columns]
    def _fmt_width(fmt: str, header: str) -> int:
        if not fmt:
            return len(header)
        digits = ""
        for ch in str(fmt):
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        return max(len(header), int(digits) if digits else len(header))

    widths = [_fmt_width(fmt, h) for h, (_, _, fmt) in zip(headers, columns)]
    print(" ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + len(widths) - 1))
    for row in rows[: int(max_rows)]:
        vals = []
        for key, _, fmt in columns:
            value = row.get(key, "")
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                vals.append("".rjust(max(1, len(fmt))))
                continue
            if fmt.endswith("f"):
                vals.append(f"{float(value):{fmt}}")
            elif fmt.endswith("d"):
                vals.append(f"{int(value):{fmt}}")
            else:
                vals.append(str(value)[: max(12, len(str(value)))])
        print(" ".join(v.rjust(w) for v, w in zip(vals, widths)))


def print_residual_hole_table(rows: list[dict[str, Any]], *, max_rows: int = 20) -> None:
    print_table(
        rows,
        [
            ("TPCid", "TPC", "4d"),
            ("t0", "t0", "5d"),
            ("peak_tick", "tick", "6d"),
            ("peak_missing_fraction", "peak%", "7.2f"),
            ("window_missing_fraction", "win%", "7.2f"),
            ("peak_missing", "peak_miss", "11.1f"),
            ("n_tpc_hits", "hits", "7d"),
        ],
        max_rows=max_rows,
    )


def print_suspicious_cluster_table(rows: list[dict[str, Any]], *, max_rows: int = 20) -> None:
    print_table(
        rows,
        [
            ("clusterid", "cluster", "8d"),
            ("energy_mev", "E", "9.2f"),
            ("t0", "t0", "6d"),
            ("assigned", "assign", ""),
            ("improvement_per_mev", "dL/E", "10.2f"),
            ("geom_linearity", "lin", "7.3f"),
            ("geom_transverse_rms_cm", "rms", "7.2f"),
            ("suspicious_reasons", "reason", ""),
        ],
        max_rows=max_rows,
    )


def print_component_score_table(rows: list[dict[str, Any]], *, max_rows: int = 20) -> None:
    print_table(
        rows,
        [
            ("component_id", "comp", "5d"),
            ("clusterid", "cluster", "8d"),
            ("target_t0", "to", "5d"),
            ("source_t0", "from", ""),
            ("n_hits", "hits", "6d"),
            ("energy_mev", "E", "8.2f"),
            ("delta_loss_per_mev", "dL/E", "12.2f"),
            ("delta_loss", "dLoss", "12.1f"),
            ("target_peak_fill_fraction", "fill", "7.2f"),
            ("overflow_sum", "overflow", "10.1f"),
            ("suspicious_reason", "reason", ""),
        ],
        max_rows=max_rows,
    )


def plot_component_candidates_3d(
    *,
    tpcid: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels_global: np.ndarray,
    hit_tpc_ids: np.ndarray,
    components: list[ComponentCandidate],
    score_rows: list[dict[str, Any]] | None = None,
    save_path: str | None = None,
):
    if go is None:
        raise RuntimeError("plotly is not available")

    tpc_mask = np.asarray(hit_tpc_ids, dtype=np.int32) == int(tpcid)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=np.asarray(z)[tpc_mask],
            y=np.asarray(y)[tpc_mask],
            z=np.asarray(x)[tpc_mask],
            mode="markers",
            marker=dict(size=2, color="lightgray", opacity=0.15),
            name=f"TPC {int(tpcid)} all hits",
            hoverinfo="skip",
        )
    )

    score_by_comp = {}
    for row in score_rows or []:
        score_by_comp.setdefault(int(row["component_id"]), row)

    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]
    for i, comp in enumerate(components):
        idx = np.asarray(comp.hit_indices, dtype=np.int64)
        row = score_by_comp.get(int(comp.component_id), {})
        name = f"comp {comp.component_id} | cl {comp.clusterid}"
        if row:
            name += f" -> t0 {row.get('target_t0')}"
        hover = (
            f"component={comp.component_id}<br>"
            f"cluster={comp.clusterid}<br>"
            f"source_t0={comp.source_t0}<br>"
            f"E={comp.energy_mev:.2f} MeV<br>"
            f"n_hits={idx.size}<br>"
            f"reason={comp.reason}"
        )
        if row:
            hover += (
                f"<br>target_t0={row.get('target_t0')}"
                f"<br>dL/E={float(row.get('delta_loss_per_mev', 0.0)):.2f}"
                f"<br>fill={float(row.get('target_peak_fill_fraction', 0.0)):.2f}"
            )
        fig.add_trace(
            go.Scatter3d(
                x=np.asarray(z)[idx],
                y=np.asarray(y)[idx],
                z=np.asarray(x)[idx],
                mode="markers",
                marker=dict(size=4.5, color=palette[i % len(palette)], opacity=0.95),
                name=name,
                hovertext=[hover] * idx.size,
                hoverinfo="text+x+y+z",
            )
        )

    fig.update_layout(
        title=f"TPC {int(tpcid)} Release4 component candidates",
        height=900,
        margin=dict(l=0, r=0, b=0, t=55),
        scene=dict(xaxis_title="z", yaxis_title="y", zaxis_title="x"),
        showlegend=True,
    )
    if save_path is not None:
        fig.write_html(save_path)
        print(f"Saved html: {save_path}")
    return fig


__all__ = [
    "ComponentCandidate",
    "build_phase3_assignment_audit",
    "scan_residual_holes",
    "find_suspicious_clusters",
    "build_component_candidates_for_tpc",
    "score_component_reassignment_options",
    "apply_component_reassignment_preview",
    "run_component_rescue_phase3_v4",
    "print_residual_hole_table",
    "print_suspicious_cluster_table",
    "print_component_score_table",
    "plot_component_candidates_3d",
]
