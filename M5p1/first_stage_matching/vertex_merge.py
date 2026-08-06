"""Opt-in vertex-track merging (v0.2.5).

Motivation: multi-prong structures (tracks emerging from one interaction
vertex) are placed independently by the backbone; each lone prong gambles on a
flash and one of them regularly loses.  A merged vertex object is brighter,
spans more channels/TPCs, and places as one, far more constrained, unit.

Geometry test — deliberately biased toward FRAGMENTATION: when in doubt, do
not merge.  Two track clusters merge only when
  1. an END of one lies within ``vm_end_radius_cm`` of an END of the other;
  2. both fitted end-directions pass within ``vm_line_tol_cm`` of the common
     point (the lines genuinely meet there, not merely two endpoints nearby);
  3. the opening angle is a real vertex angle
     (``vm_angle_min_deg``..``vm_angle_max_deg``): collinear joins (broken /
     pass-through tracks) and back-to-back configurations are excluded;
  4. neither endpoint touches the OTHER track's interior (a crossing, not a
     vertex) within ``vm_crossing_guard_cm``;
  5. both local end-fits are clean lines (linearity >= ``vm_min_linearity``).
Accepted pairs are grouped (union-find); any group larger than
``vm_max_group`` is dropped entirely.

Only backbone labels typed "track" participate.  Everything is pure numpy on
(x, y, z) in the drift frame — two tracks from the SAME interaction share the
unknown t0, so their relative geometry is exact; tracks from different
interactions can only converge by accident.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _end_fit(pts: np.ndarray, t: np.ndarray, lo_end: bool, end_len_cm: float,
             min_local_hits: int):
    """Fit the local line at one end of a track.

    Returns (endpoint, into_track_direction, linearity) or None."""
    tt = t - t.min()
    span = float(tt.max())
    if lo_end:
        sel = tt <= min(end_len_cm, 0.5 * span)
        extreme = np.argsort(tt)[:3]
    else:
        sel = tt >= span - min(end_len_cm, 0.5 * span)
        extreme = np.argsort(tt)[-3:]
    if int(sel.sum()) < min_local_hits:
        return None
    local = pts[sel]
    endpoint = pts[extreme].mean(axis=0)
    c = local - local.mean(axis=0)
    try:
        _, s, vt = np.linalg.svd(c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s.size < 2 or s[0] < 1e-9:
        return None
    linearity = float(s[0] ** 2 / max(np.sum(s ** 2), 1e-12))
    u = vt[0]
    into = local.mean(axis=0) - endpoint
    if np.dot(u, into) < 0:
        u = -u
    return endpoint, u / max(np.linalg.norm(u), 1e-12), linearity


def _point_line_dist(p: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    d = p - origin
    return float(np.linalg.norm(d - np.dot(d, direction) * direction))


def find_vertex_merges(x, y, z, labels, track_labels, *,
                       end_radius_cm=4.0, line_tol_cm=2.5,
                       angle_min_deg=15.0, angle_max_deg=165.0,
                       min_hits=15, min_length_cm=4.0, end_len_cm=8.0,
                       crossing_guard_cm=3.0, min_linearity=0.75,
                       min_local_hits=6, max_group=6):
    """Return (groups, pair_records): groups = lists of labels to merge."""
    pts_all = np.stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                        np.asarray(z, np.float64)], axis=1)
    labels = np.asarray(labels)
    ends = {}       # label -> list of (endpoint, dir_into, linearity)
    interior = {}   # label -> interior hit positions (for the crossing guard)
    for lab in track_labels:
        m = labels == lab
        n = int(m.sum())
        if n < min_hits:
            continue
        pts = pts_all[m]
        c = pts - pts.mean(axis=0)
        try:
            _, s, vt = np.linalg.svd(c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        t = c @ vt[0]
        length = float(t.max() - t.min())
        if length < min_length_cm:
            continue
        fits = []
        for lo in (True, False):
            f = _end_fit(pts, t, lo, end_len_cm, min_local_hits)
            if f is not None and f[2] >= min_linearity:
                fits.append(f)
            else:
                fits.append(None)
        ends[int(lab)] = fits
        q = np.quantile(t, [0.30, 0.70])
        core = pts[(t >= q[0]) & (t <= q[1])]
        interior[int(lab)] = core if core.shape[0] else pts

    labs = sorted(ends)
    parent = {l: l for l in labs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pair_records = []
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            best = None
            for fa in ends[a]:
                for fb in ends[b]:
                    if fa is None or fb is None:
                        continue
                    pa, ua, _ = fa
                    pb, ub, _ = fb
                    d_end = float(np.linalg.norm(pa - pb))
                    if d_end > end_radius_cm:
                        continue
                    v = 0.5 * (pa + pb)
                    if (_point_line_dist(v, pa, ua) > line_tol_cm
                            or _point_line_dist(v, pb, ub) > line_tol_cm):
                        continue
                    ang = float(np.degrees(np.arccos(np.clip(np.dot(ua, ub), -1, 1))))
                    if not (angle_min_deg <= ang <= angle_max_deg):
                        continue
                    # crossing guard: an endpoint near the OTHER track's core
                    # means we are looking at an X, not a V
                    if (np.min(np.linalg.norm(interior[b] - pa, axis=1)) < crossing_guard_cm
                            or np.min(np.linalg.norm(interior[a] - pb, axis=1)) < crossing_guard_cm):
                        continue
                    if best is None or d_end < best["end_dist_cm"]:
                        best = {"labels": [int(a), int(b)], "end_dist_cm": d_end,
                                "angle_deg": ang,
                                "vertex": [float(v[0]), float(v[1]), float(v[2])]}
            if best is not None:
                pair_records.append(best)
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    raw_groups = {}
    for l in labs:
        raw_groups.setdefault(find(l), []).append(l)
    groups = [sorted(g) for g in raw_groups.values() if len(g) >= 2]
    groups = [g for g in groups if len(g) <= max_group]   # oversized -> drop whole group
    keep = {l for g in groups for l in g}
    pair_records = [p for p in pair_records
                    if p["labels"][0] in keep and p["labels"][1] in keep]
    return sorted(groups), pair_records


def merge_vertex_tracks(*, labels_global, split_index, label_info, x, y, z, config):
    """Apply vertex merging in place-ish; returns (new_labels, stats)."""
    labels = np.asarray(labels_global).copy()
    track_labels = [int(l) for l in range(int(split_index))
                    if str(label_info.get(int(l), {}).get("type", "track")).lower() == "track"
                    and np.any(labels == l)]
    groups, pairs = find_vertex_merges(
        x, y, z, labels, track_labels,
        end_radius_cm=config.vm_end_radius_cm,
        line_tol_cm=config.vm_line_tol_cm,
        angle_min_deg=config.vm_angle_min_deg,
        angle_max_deg=config.vm_angle_max_deg,
        min_hits=config.vm_min_hits,
        min_length_cm=config.vm_min_length_cm,
        end_len_cm=config.vm_end_len_cm,
        crossing_guard_cm=config.vm_crossing_guard_cm,
        min_linearity=config.vm_min_linearity,
        min_local_hits=config.vm_min_local_hits,
        max_group=config.vm_max_group,
    )
    stats: dict[str, Any] = {"groups": groups, "pairs": pairs,
                             "labels_premerge": np.asarray(labels_global).copy()}
    for g in groups:
        keep = int(g[0])
        for lab in g[1:]:
            labels[labels == lab] = keep
            if int(lab) in label_info:
                label_info[int(lab)]["n_hits"] = 0
                label_info[int(lab)]["merged_into"] = keep
        if keep in label_info:
            label_info[keep]["vertex_merged"] = True
            label_info[keep]["merged_from"] = [int(l) for l in g[1:]]
    return labels, stats


__all__ = ["find_vertex_merges", "merge_vertex_tracks"]
