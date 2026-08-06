"""Opt-in shower noise absorption (v0.3.5).

Tracks already absorb nearby un-clustered hits (toolbox pass, transverse
radius based).  Showers had nothing: their fuzzy halo hits stay label -1 and
only get a t0 at the very end by geometry.  This pass lets each backbone
SHOWER admit noise hits through its own PCA ellipsoid — the shape showers
actually have — before prediction, so the absorbed hits contribute to the
shower's light image and inherit its placement.

Conservative rules:
  - only hits with label < 0 (never steal from clusters);
  - Mahalanobis distance in the shower's 3-component PCA frame <= k_sigma,
    with each axis floored at ``axis_floor_cm`` (thin showers get a minimum
    corridor) and padded by ``pad_frac``;
  - a hit admissible to several showers goes to the closest (Mahalanobis);
  - per-shower absorbed hits capped at ``max_frac`` of its own hit count.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def absorb_shower_noise(*, labels_global, split_index, label_info, x, y, z,
                        config) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels_global).copy()
    xyz = np.stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                    np.asarray(z, np.float64)], axis=1)
    noise_idx = np.flatnonzero(labels < 0)
    stats: dict[str, Any] = {"n_showers": 0, "n_absorbed": 0, "per_shower": []}
    if noise_idx.size == 0:
        return labels, stats

    k = float(config.shower_absorb_k_sigma)
    pad = 1.0 + float(config.shower_absorb_pad_frac)
    floor = float(config.shower_absorb_axis_floor_cm)
    max_frac = float(config.shower_absorb_max_frac)

    best_d = np.full(noise_idx.size, np.inf)
    best_lab = np.full(noise_idx.size, -1, dtype=np.int64)
    caps: dict[int, int] = {}

    for lab in range(int(split_index)):
        if str(label_info.get(int(lab), {}).get("type", "")).lower() != "shower":
            continue
        m = labels == lab
        n = int(m.sum())
        if n < 12:
            continue
        pts = xyz[m]
        mu = pts.mean(axis=0)
        c = pts - mu
        try:
            _, s, vt = np.linalg.svd(c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        sig = np.maximum(s / np.sqrt(max(n - 1, 1)), floor) * pad
        stats["n_showers"] += 1
        caps[int(lab)] = max(1, int(max_frac * n))
        # quick spatial pre-cut: bounding sphere
        r_max = k * float(sig[0]) + 1.0
        d0 = np.linalg.norm(xyz[noise_idx] - mu, axis=1)
        near = np.flatnonzero(d0 <= r_max)
        if near.size == 0:
            continue
        local = (xyz[noise_idx[near]] - mu) @ vt.T
        dm = np.sqrt(np.sum((local / sig) ** 2, axis=1))
        ok = dm <= k
        sel = near[ok]
        better = dm[ok] < best_d[sel]
        best_d[sel[better]] = dm[ok][better]
        best_lab[sel[better]] = int(lab)

    absorbed_by: dict[int, list[int]] = {}
    order = np.argsort(best_d)
    for i in order:
        lab = int(best_lab[i])
        if lab < 0 or not np.isfinite(best_d[i]):
            continue
        lst = absorbed_by.setdefault(lab, [])
        if len(lst) >= caps.get(lab, 0):
            continue
        lst.append(int(noise_idx[i]))
    for lab, lst in absorbed_by.items():
        labels[np.asarray(lst, dtype=np.int64)] = lab
        stats["n_absorbed"] += len(lst)
        stats["per_shower"].append({"label": int(lab), "n": len(lst)})
    return labels, stats


__all__ = ["absorb_shower_noise"]
