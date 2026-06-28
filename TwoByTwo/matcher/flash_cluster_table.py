"""
Flash-cluster bookkeeping for the 2x2 charge-light matching v4 pipeline.

Each TPC owns a list of t0 candidates. Each candidate is annotated with a
boolean flag: True iff at least one charge cluster has been assigned to that
t0. The flag distinguishes "real PMT flash that received charge" from a
track-derived t0 seed that was never matched to non-track charge.

Adapted (simplified, no light<->charge remap, max_t0=700) from
M5p1/flash_cluster_table.py.
"""

from __future__ import annotations
from typing import Any, Iterable
import numpy as np


def _is_valid_t0(value: Any, max_t0: float | None = None) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    if not np.isfinite(v) or int(round(v)) == 0:
        return False
    if max_t0 is not None and (v < 0.0 or v > float(max_t0)):
        return False
    return True


def _clean_t0(value: Any, max_t0: float | None = None) -> float:
    v = float(value)
    if max_t0 is not None:
        v = float(np.clip(v, 0.0, float(max_t0)))
    if abs(v - round(v)) < 1e-6:
        return float(int(round(v)))
    return float(v)


def _sort_pairs(values, flags, max_t0=None):
    pairs = []
    for value, flag in zip(values, flags):
        if _is_valid_t0(value, max_t0=max_t0):
            pairs.append((_clean_t0(value, max_t0=max_t0), bool(flag)))
    pairs.sort(key=lambda item: item[0])
    return [float(v) for v, _ in pairs], [bool(f) for _, f in pairs]


def ensure_flash_cluster_flags(t0_candidates, flags_by_tpc=None, *, max_t0=None):
    """Align/clean a flag list against a t0_candidates list-of-lists.

    Returns the per-TPC flag list (mutates ``t0_candidates`` to drop zero/invalid).
    If ``flags_by_tpc`` is None, all flags default to False.
    """
    flags = []
    for tpc, values in enumerate(t0_candidates):
        if flags_by_tpc is None or int(tpc) >= len(flags_by_tpc):
            raw = [False] * len(values)
        else:
            raw = list(flags_by_tpc[int(tpc)])
            if len(raw) < len(values):
                raw.extend([False] * (len(values) - len(raw)))
            elif len(raw) > len(values):
                raw = raw[: len(values)]
        cleaned_v, cleaned_f = _sort_pairs(values, raw, max_t0=max_t0)
        t0_candidates[int(tpc)] = cleaned_v
        flags.append(cleaned_f)
    return flags


def merge_close_t0s(values, merge_ticks):
    """Group sorted t0s within ``merge_ticks`` and replace each group by its median."""
    vals = sorted(
        float(v) for v in values
        if v is not None and np.isfinite(float(v)) and int(round(float(v))) != 0
    )
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= float(merge_ticks):
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.median(g)) for g in groups]


def canonicalize_candidate_t0(t0_candidates, tpc, t0, *, merge_ticks):
    """Insert ``t0`` into TPC's candidate list and merge near-duplicates."""
    tpc = int(tpc)
    while len(t0_candidates) <= tpc:
        t0_candidates.append([])
    cur = [
        float(v) for v in t0_candidates[tpc]
        if v is not None and np.isfinite(float(v)) and int(round(float(v))) != 0
        and abs(float(v) - float(t0)) > float(merge_ticks)
    ]
    cur.append(float(t0))
    t0_candidates[tpc] = merge_close_t0s(cur, merge_ticks)
    return t0_candidates


def mark_flash_cluster_assignment(
    t0_candidates,
    flags_by_tpc,
    *,
    tpc,
    t0,
    resolution_ticks,
    max_t0=None,
    prefer_existing_true=False,
    clusterid=None,
    stage="",
):
    """Mark a single (tpc, t0) as having received an assigned cluster.

    Removes existing entries within ``resolution_ticks``, then inserts a single
    canonical entry with flag=True. Returns updated (t0_candidates, flags, row).
    """
    flags = ensure_flash_cluster_flags(t0_candidates, flags_by_tpc, max_t0=max_t0)
    tpc = int(tpc)
    while len(t0_candidates) <= tpc:
        t0_candidates.append([])
        flags.append([])

    assigned = _clean_t0(t0, max_t0=max_t0)
    cur_v = list(t0_candidates[tpc])
    cur_f = list(flags[tpc])
    removed = [(float(v), bool(f)) for v, f in zip(cur_v, cur_f)
               if abs(float(v) - assigned) <= float(resolution_ticks)]
    survivors = [(float(v), bool(f)) for v, f in zip(cur_v, cur_f)
                 if abs(float(v) - assigned) > float(resolution_ticks)]

    canonical = float(assigned)
    if prefer_existing_true:
        true_removed = [r for r in removed if r[1]]
        if true_removed:
            canonical = min(true_removed, key=lambda r: abs(r[0] - assigned))[0]

    survivors.append((float(canonical), True))
    survivors.sort(key=lambda r: r[0])
    t0_candidates[tpc] = [float(v) for v, _ in survivors]
    flags[tpc] = [bool(f) for _, f in survivors]

    row = {
        "TPCid": int(tpc),
        "clusterid": None if clusterid is None else int(clusterid),
        "stage": str(stage),
        "assigned_t0": float(assigned),
        "canonical_t0": float(canonical),
        "removed_flash_t0s": [float(v) for v, _ in removed],
        "removed_had_cluster": [bool(f) for _, f in removed],
        "resolution_ticks": float(resolution_ticks),
    }
    return t0_candidates, flags, row


def flash_cluster_table_rows(t0_candidates, flags_by_tpc):
    flags = ensure_flash_cluster_flags(t0_candidates, flags_by_tpc)
    rows = []
    for tpc, values in enumerate(t0_candidates):
        for value, flag in zip(values, flags[int(tpc)]):
            rows.append({"TPCid": int(tpc), "t0": float(value),
                         "received_cluster": bool(flag)})
    return rows


def associated_source_t0s_by_tpc(
    *,
    hit_timestamps,
    hit_tpc_ids,
    t0_candidates,
    allowed_tpcs,
    labels_global=None,
    match_ticks=5.0,
):
    """For each TPC, return the subset of t0_candidates that have at least one
    valid (finite, non-noise) charge hit within ``match_ticks``.

    ``t0_candidates`` may be either the per-TPC list-of-lists or a dict.
    """
    if isinstance(t0_candidates, dict):
        max_tpc = max([int(k) for k in t0_candidates] + [-1]) + 1
        cand_list = [[] for _ in range(max_tpc)]
        for k, v in t0_candidates.items():
            cand_list[int(k)] = list(v)
    else:
        cand_list = [list(v) for v in t0_candidates]

    out = {}
    hit_ts = np.asarray(hit_timestamps, dtype=np.float64)
    tpcs = np.asarray(hit_tpc_ids, dtype=np.int32)
    finite = np.isfinite(hit_ts) & (hit_ts >= 0)
    if labels_global is not None:
        finite &= np.asarray(labels_global, dtype=np.int64) >= 0

    for tpc in sorted(int(v) for v in allowed_tpcs):
        if int(tpc) >= len(cand_list):
            continue
        source_vals = []
        for flash_t0 in cand_list[int(tpc)]:
            try:
                ft0 = float(flash_t0)
            except Exception:
                continue
            if not np.isfinite(ft0):
                continue
            mask = (tpcs == int(tpc)) & finite & (np.abs(hit_ts - ft0) <= float(match_ticks))
            if mask.any():
                src = int(round(ft0))
                if src != 0:
                    source_vals.append(src)
        if source_vals:
            out[int(tpc)] = sorted({int(v) for v in source_vals})
    return out


__all__ = [
    "ensure_flash_cluster_flags",
    "merge_close_t0s",
    "canonicalize_candidate_t0",
    "mark_flash_cluster_assignment",
    "flash_cluster_table_rows",
    "associated_source_t0s_by_tpc",
]
