from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _is_valid_t0(value: Any, *, max_t0: float | None = None) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    if not np.isfinite(v) or int(round(v)) == 0:
        return False
    if max_t0 is not None and (v < 0.0 or v > float(max_t0)):
        return False
    return True


def _clean_t0(value: Any, *, max_t0: float | None = None) -> float:
    v = float(value)
    if max_t0 is not None:
        v = float(np.clip(v, 0.0, float(max_t0)))
    if abs(v - round(v)) < 1e-6:
        return float(int(round(v)))
    return float(v)


def _nearest_tick(value: Any) -> int:
    return int(np.floor(float(value) + 0.5))


def _sort_pairs(values: Iterable[Any], flags: Iterable[Any], *, max_t0: float | None = None) -> tuple[list[float], list[bool]]:
    pairs = []
    for value, flag in zip(values, flags):
        if _is_valid_t0(value, max_t0=max_t0):
            pairs.append((_clean_t0(value, max_t0=max_t0), bool(flag)))
    pairs.sort(key=lambda item: item[0])
    return [float(v) for v, _ in pairs], [bool(f) for _, f in pairs]


def ensure_flash_cluster_flags(
    t0_candidates: list[list[Any]],
    flash_cluster_received_by_tpc: list[list[Any]] | None = None,
    *,
    max_t0: float | None = None,
) -> list[list[bool]]:
    flags: list[list[bool]] = []
    for tpc, values in enumerate(t0_candidates):
        if flash_cluster_received_by_tpc is None or int(tpc) >= len(flash_cluster_received_by_tpc):
            raw_flags = [False] * len(values)
        else:
            raw_flags = list(flash_cluster_received_by_tpc[int(tpc)])
            if len(raw_flags) < len(values):
                raw_flags.extend([False] * (len(values) - len(raw_flags)))
            elif len(raw_flags) > len(values):
                raw_flags = raw_flags[: len(values)]
        cleaned_values, cleaned_flags = _sort_pairs(values, raw_flags, max_t0=max_t0)
        t0_candidates[int(tpc)] = cleaned_values
        flags.append(cleaned_flags)
    return flags


def mark_flash_cluster_assignment(
    t0_candidates: list[list[Any]],
    flash_cluster_received_by_tpc: list[list[Any]] | None,
    *,
    tpc: int,
    t0: float,
    resolution_ticks: float,
    max_t0: float | None = None,
    prefer_existing_true: bool = False,
    clusterid: int | None = None,
    stage: str = "",
) -> tuple[list[list[Any]], list[list[bool]], dict[str, Any]]:
    """Canonicalize a flash-table entry and mark it as receiving assigned charge.

    The t0 candidate and boolean lists are kept aligned. Any existing entries
    within ``resolution_ticks`` of the assigned t0 are removed, then a single
    canonical entry is inserted with ``received_cluster=True``.
    """
    flags = ensure_flash_cluster_flags(t0_candidates, flash_cluster_received_by_tpc, max_t0=max_t0)
    tpc = int(tpc)
    while len(t0_candidates) <= tpc:
        t0_candidates.append([])
        flags.append([])

    assigned_t0 = _clean_t0(t0, max_t0=max_t0)
    current_values = list(t0_candidates[tpc])
    current_flags = list(flags[tpc])
    removed = [
        (float(value), bool(flag))
        for value, flag in zip(current_values, current_flags)
        if abs(float(value) - float(assigned_t0)) <= float(resolution_ticks)
    ]
    survivors = [
        (float(value), bool(flag))
        for value, flag in zip(current_values, current_flags)
        if abs(float(value) - float(assigned_t0)) > float(resolution_ticks)
    ]

    canonical_t0 = float(assigned_t0)
    if prefer_existing_true:
        true_removed = [item for item in removed if bool(item[1])]
        if true_removed:
            canonical_t0 = min(true_removed, key=lambda item: abs(float(item[0]) - float(assigned_t0)))[0]

    survivors.append((float(canonical_t0), True))
    survivors.sort(key=lambda item: item[0])
    t0_candidates[tpc] = [float(v) for v, _ in survivors]
    flags[tpc] = [bool(f) for _, f in survivors]

    return t0_candidates, flags, {
        "TPCid": int(tpc),
        "clusterid": None if clusterid is None else int(clusterid),
        "stage": str(stage),
        "assigned_t0": float(assigned_t0),
        "canonical_t0": float(canonical_t0),
        "removed_flash_t0s": [float(v) for v, _ in removed],
        "removed_had_cluster": [bool(f) for _, f in removed],
        "resolution_ticks": float(resolution_ticks),
    }


def rebuild_flash_cluster_flags_from_assignments(
    t0_candidates: list[list[Any]],
    assignment_info: dict[tuple[int, int], dict[str, Any]],
    *,
    resolution_ticks: float,
    max_t0: float | None = None,
    initial_flags: list[list[Any]] | None = None,
    prefer_existing_true: bool = False,
    allowed_stages: set[str] | None = None,
) -> tuple[list[list[Any]], list[list[bool]], list[dict[str, Any]]]:
    flags = ensure_flash_cluster_flags(t0_candidates, initial_flags, max_t0=max_t0)
    rows: list[dict[str, Any]] = []

    assigned_rows = []
    for key, info in assignment_info.items():
        if not bool(info.get("assigned", False)):
            continue
        try:
            clusterid, tpc = int(key[0]), int(key[1])
            t0 = float(info.get("t0", np.nan))
        except Exception:
            continue
        if not _is_valid_t0(t0, max_t0=max_t0):
            continue
        stage = str(info.get("stage", info.get("mode", "")))
        if allowed_stages is not None and stage not in allowed_stages:
            continue
        assigned_rows.append(
            (
                -float(info.get("energy", 0.0)),
                clusterid,
                tpc,
                t0,
                stage,
            )
        )

    for _, clusterid, tpc, t0, stage in sorted(assigned_rows):
        t0_candidates, flags, row = mark_flash_cluster_assignment(
            t0_candidates,
            flags,
            tpc=int(tpc),
            t0=float(t0),
            resolution_ticks=float(resolution_ticks),
            max_t0=max_t0,
            prefer_existing_true=bool(prefer_existing_true),
            clusterid=int(clusterid),
            stage=str(stage),
        )
        rows.append(row)

    return t0_candidates, flags, rows


def flash_cluster_table_rows(
    t0_candidates: list[list[Any]],
    flash_cluster_received_by_tpc: list[list[Any]] | None,
) -> list[dict[str, Any]]:
    flags = ensure_flash_cluster_flags(t0_candidates, flash_cluster_received_by_tpc)
    rows: list[dict[str, Any]] = []
    for tpc, values in enumerate(t0_candidates):
        for value, flag in zip(values, flags[int(tpc)]):
            rows.append(
                {
                    "TPCid": int(tpc),
                    "t0": float(value),
                    "received_cluster": bool(flag),
                }
            )
    return rows


def source_t0s_from_received_flash_table(
    t0_candidates: list[list[Any]] | dict[int, list[Any]],
    flash_cluster_received_by_tpc: list[list[Any]] | dict[int, list[Any]] | None,
    *,
    allowed_tpcs: Iterable[int] | None = None,
    min_sep_ticks: float = 0.0,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    if isinstance(t0_candidates, dict):
        max_tpc = max([int(k) for k in t0_candidates.keys()] + [-1]) + 1
        cand_list = [[] for _ in range(max_tpc)]
        for key, values in t0_candidates.items():
            cand_list[int(key)] = list(values)
    else:
        cand_list = [list(values) for values in t0_candidates]

    if isinstance(flash_cluster_received_by_tpc, dict):
        max_tpc = max(len(cand_list), max([int(k) for k in flash_cluster_received_by_tpc.keys()] + [-1]) + 1)
        flag_list = [[] for _ in range(max_tpc)]
        for key, values in flash_cluster_received_by_tpc.items():
            flag_list[int(key)] = list(values)
    else:
        flag_list = None if flash_cluster_received_by_tpc is None else [list(values) for values in flash_cluster_received_by_tpc]

    flags = ensure_flash_cluster_flags(cand_list, flag_list)
    tpcs = sorted(int(v) for v in allowed_tpcs) if allowed_tpcs is not None else list(range(len(cand_list)))

    out: dict[int, list[int]] = {}
    rows: list[dict[str, Any]] = []
    def _merge_sources(values: list[tuple[float, int]]) -> list[int]:
        if not values:
            return []
        values = sorted(values, key=lambda item: item[0])
        if float(min_sep_ticks) <= 0.0:
            return sorted(set(int(source) for _, source in values))

        groups: list[list[tuple[float, int]]] = []
        for value, source in values:
            if not groups or abs(float(value) - float(groups[-1][-1][0])) > float(min_sep_ticks):
                groups.append([(float(value), int(source))])
            else:
                groups[-1].append((float(value), int(source)))

        merged = []
        for group in groups:
            center = float(np.median([float(value) for value, _ in group]))
            merged.append(_nearest_tick(center))
        return sorted(set(int(v) for v in merged if int(v) != 0))

    for tpc in tpcs:
        if int(tpc) >= len(cand_list):
            continue
        source_vals = []
        for value, flag in zip(cand_list[int(tpc)], flags[int(tpc)]):
            rows.append(
                {
                    "TPCid": int(tpc),
                    "t0": float(value),
                    "source_t0": _nearest_tick(value),
                    "received_cluster": bool(flag),
                }
            )
            if bool(flag):
                source = _nearest_tick(value)
                if source != 0:
                    source_vals.append((float(value), source))
        if source_vals:
            out[int(tpc)] = _merge_sources(source_vals)
    return out, rows


__all__ = [
    "ensure_flash_cluster_flags",
    "mark_flash_cluster_assignment",
    "rebuild_flash_cluster_flags_from_assignments",
    "flash_cluster_table_rows",
    "source_t0s_from_received_flash_table",
]
