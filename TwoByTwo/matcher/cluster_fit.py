# cluster_fit.py
"""
Phase-2 and Phase-3 clustering helpers.

Public API:
    - fit_cluster_labels(x, y, z, E, labels=None, eps=2.5, min_samples=3, verbose=True)
    - fit_noise_list(x, y, z, E, labels, expand_frac=0.10, leafsize=16)
    - plot_phase3_candidate_hist(entries, labels, exclude_zero=False)
    - plot_cluster_result(x, y, z, labels=None, showNoise=True, maxClusterid=None, minClusterid=None)

Notes:
    - E is accepted for future energy-aware variants but currently unused.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


# --------------------------- Phase 2: DBSCAN merge ---------------------------

def fit_cluster_labels(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    eps: float = 2.5,
    min_samples: int = 3,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Phase-2 DBSCAN and merge into global labels without ID overlap.
    New clusters are sorted by total Energy (descending) before assigning IDs.

    - If labels is None:
        - DBSCAN over ALL hits.
        - Returns labels_out (0..K-1 sorted by E, -1 noise), DBSCAN_label (same), dbscan_noise_idx, stats.

    - If labels is provided (e.g. from Phase 1 tracks):
        - DBSCAN only on (labels == -1).
        - Offset cluster ids by start_id = max(existing)+1.
        - New clusters (ID >= start_id) are sorted by Energy descending.
        - Returns labels_out, DBSCAN_label (local ids on subset), dbscan_noise_idx, stats.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    X = np.column_stack([x, y, z])

    # --- Helper to sort labels by energy ---
    def _sort_labels_by_energy(local_labels, energy_array):
        """Remap local_labels (0..M-1) such that 0 is highest energy cluster."""
        unique_ids = np.unique(local_labels)
        unique_ids = unique_ids[unique_ids >= 0]
        if len(unique_ids) == 0:
            return local_labels

        # Calculate total energy for each cluster
        # Using bincount is faster than loop
        # Ensure labels are non-negative for bincount; we only care about >= 0
        max_id = unique_ids.max()
        e_sums = np.zeros(max_id + 1, dtype=float)
        
        # Iterate or mask? Iterate is safer for sparse IDs, but here IDs are 0..M (DBSCAN)
        # DBSCAN produces potentially non-contiguous IDs? No, usually 0..N-1. But let's be safe.
        for uid in unique_ids:
            e_sums[uid] = np.sum(energy_array[local_labels == uid])

        # Sort indices by energy descending
        # We want a map: old_id -> new_rank
        # sort existing IDs by their energy
        sorted_ids = sorted(unique_ids, key=lambda i: e_sums[i], reverse=True)
        
        # Create mapping array
        # map_arr[old_id] = new_id
        map_arr = np.full(max_id + 1, -1, dtype=int)
        for new_id, old_id in enumerate(sorted_ids):
            map_arr[old_id] = new_id
            
        # Apply mapping
        new_labels = np.full_like(local_labels, -1)
        mask_valid = (local_labels >= 0)
        if np.any(mask_valid):
            new_labels[mask_valid] = map_arr[local_labels[mask_valid]]
            
        return new_labels

    if labels is None:
        # Case A: No prior labels, cluster everything
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
        labels_sub = db.fit_predict(X)  # local ids 0..K-1, -1 noise

        # Sort by Energy
        labels_sub = _sort_labels_by_energy(labels_sub, E)

        labels_out = labels_sub.astype(int, copy=True)
        DBSCAN_label = labels_sub.astype(int, copy=True)
        dbscan_noise_idx = np.flatnonzero(labels_sub == -1)

        n_new_clusters = int(labels_sub.max() + 1) if np.any(labels_sub >= 0) else 0
        n_noise = int(dbscan_noise_idx.size)
        stats = dict(
            n_input=int(X.shape[0]),
            n_new_clusters=n_new_clusters,
            n_noise=n_noise,
            start_id=0,
            final_max_label=int(labels_out[labels_out >= 0].max()) if np.any(labels_out >= 0) else -1,
        )
        if verbose:
            print(f"[Phase2/DBSCAN] input:{stats['n_input']}, "
                  f"new_clusters:{stats['n_new_clusters']} (sorted by E), noise:{stats['n_noise']}, "
                  f"start_id:{stats['start_id']}, final_max_label:{stats['final_max_label']}")
        return labels_out, DBSCAN_label, dbscan_noise_idx, stats

    else:
        # Case B: Cluster only the noise (-1) from pre-existing labels
        labels = np.asarray(labels, dtype=int)
        mask_phase1_noise = (labels == -1)
        rem_idx = np.flatnonzero(mask_phase1_noise)

        DBSCAN_label = np.full_like(labels, -1, dtype=int)

        if rem_idx.size == 0:
            stats = dict(
                n_input=0, n_new_clusters=0, n_noise=0,
                start_id=int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0,
                final_max_label=int(labels[labels >= 0].max()) if np.any(labels >= 0) else -1,
            )
            if verbose:
                print(f"[Phase2/DBSCAN] input:0, new_clusters:0, noise:0, "
                      f"start_id:{stats['start_id']}, final_max_label:{stats['final_max_label']}")
            return labels.copy(), DBSCAN_label, np.array([], dtype=int), stats

        Xrem = X[mask_phase1_noise]
        Erem = E[mask_phase1_noise]
        
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
        labels_sub = db.fit_predict(Xrem)  # local: 0..K-1, -1 noise

        # Sort new clusters by Energy
        labels_sub = _sort_labels_by_energy(labels_sub, Erem)

        DBSCAN_label[mask_phase1_noise] = labels_sub

        # Determine offset (start_id)
        start_id = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
        
        # Apply offset to valid clusters
        labels_sub_global = np.where(labels_sub >= 0, labels_sub + start_id, -1)

        labels_out = labels.copy()
        labels_out[mask_phase1_noise] = labels_sub_global

        dbscan_noise_idx = rem_idx[labels_sub == -1]
        n_new_clusters = int(labels_sub.max() + 1) if np.any(labels_sub >= 0) else 0
        n_noise = int(np.count_nonzero(labels_sub == -1))
        stats = dict(
            n_input=int(Xrem.shape[0]),
            n_new_clusters=n_new_clusters,
            n_noise=n_noise,
            start_id=start_id,
            final_max_label=int(labels_out[labels_out >= 0].max()) if np.any(labels_out >= 0) else -1,
        )
        if verbose:
            print(f"[Phase2/DBSCAN] input:{stats['n_input']}, "
                  f"new_clusters:{stats['n_new_clusters']} (sorted by E), noise:{stats['n_noise']}, "
                  f"start_id:{stats['start_id']}, final_max_label:{stats['final_max_label']}")

        return labels_out, DBSCAN_label, dbscan_noise_idx, stats


# --------------------------- Phase 3: distance list ---------------------------
'''
The most original version of fit_noise_list, with no distance threashold and T/F expansion
def fit_noise_list(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    labels: np.ndarray,
    *,
    expand_frac: float = 0.20,
    leafsize: int = 16,
) -> List[Tuple[int, int, float]]:
    """
    Build the Phase-3 distance-ranking list from FINAL labels.

    For every noise hit (labels == -1):
        - Add (noise_index, nearest_cluster_id, distance),
        - Expand radius by (1+expand_frac)*distance and add entries for any *other*
          clusters within that ball using the closest point from each.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    X = np.column_stack([x, y, z])
    mask_lab = labels >= 0
    mask_noise = labels == -1

    if not np.any(mask_lab) or not np.any(mask_noise):
        return []

    X_lab = X[mask_lab]
    L_lab = labels[mask_lab]
    noise_idx = np.flatnonzero(mask_noise)

    tree = cKDTree(X_lab, leafsize=leafsize)
    dists, idxs = tree.query(X[noise_idx], k=1, workers=-1)

    entries: List[Tuple[int, int, float]] = []
    for global_i, d, j_lab in zip(noise_idx, dists, idxs):
        if not np.isfinite(d):
            continue
        cid_near = int(L_lab[j_lab])
        entries.append((int(global_i), cid_near, float(d)))

        r = float((1.0 + float(expand_frac)) * d)
        if r <= 0.0:
            continue

        inds_ball = tree.query_ball_point(X[global_i], r, workers=-1)
        if not inds_ball:
            continue

        cids_ball = set(int(L_lab[k]) for k in inds_ball)
        cids_ball.discard(cid_near)

        for cid in cids_ball:
            sel = [k for k in inds_ball if int(L_lab[k]) == cid]
            dd = np.linalg.norm(X_lab[sel] - X[global_i], axis=1).min()
            entries.append((int(global_i), int(cid), float(dd)))

    entries.sort(key=lambda t: t[2])
    return entries
'''
def fit_noise_list(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    labels: np.ndarray,
    *,
    expand_frac: float = 0.20,
    leafsize: int = 16,
    max_distance: float = 40.0,
) -> List[Tuple[int, int, float]]:
    """
    Build the Phase-3 distance-ranking list from FINAL labels.

    For every noise hit (labels == -1):
        - Add (noise_index, nearest_cluster_id, distance),
        - Expand radius by (1+expand_frac)*distance and add entries for any *other*
          clusters within that ball using the closest point from each.
        - Hits further than max_distance from any cluster are ignored.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    X = np.column_stack([x, y, z])
    mask_lab = labels >= 0
    mask_noise = labels == -1

    if not np.any(mask_lab) or not np.any(mask_noise):
        return []

    X_lab = X[mask_lab]
    L_lab = labels[mask_lab]
    noise_idx = np.flatnonzero(mask_noise)

    tree = cKDTree(X_lab, leafsize=leafsize)
    
    # Efficiently query with a distance cutoff. 
    # If no neighbor is found within max_distance, d will be inf.
    dists, idxs = tree.query(
        X[noise_idx], 
        k=1, 
        distance_upper_bound=max_distance, 
        workers=-1
    )

    entries: List[Tuple[int, int, float]] = []
    for global_i, d, j_lab in zip(noise_idx, dists, idxs):
        # 'd' is inf if no points were within max_distance
        if not np.isfinite(d):
            continue
            
        cid_near = int(L_lab[j_lab])
        entries.append((int(global_i), cid_near, float(d)))

        r = float((1.0 + float(expand_frac)) * d)
        if r <= 0.0:
            continue

        inds_ball = tree.query_ball_point(X[global_i], r, workers=-1)
        if not inds_ball:
            continue

        cids_ball = set(int(L_lab[k]) for k in inds_ball)
        cids_ball.discard(cid_near)

        for cid in cids_ball:
            sel = [k for k in inds_ball if int(L_lab[k]) == cid]
            dd = np.linalg.norm(X_lab[sel] - X[global_i], axis=1).min()
            entries.append((int(global_i), int(cid), float(dd)))

    entries.sort(key=lambda t: t[2])
    return entries

# --------------------------- Plotting helpers ---------------------------

def plot_phase3_candidate_hist(
    entries: List[Tuple[int, int, float]],
    labels: np.ndarray,
    *,
    exclude_zero: bool = False,
    title: str = "Phase-3 candidates per final noise hit"
):
    """
    Plot a histogram of 'number of candidate clusters per final noise hit'.

    Parameters
    ----------
    entries : list of (noise_index, cluster_id, distance) from Phase 3
    labels  : 1D array of final labels (to know which hits are noise, so zeros are included)
    exclude_zero : if True, omit hits that have 0 candidates (i.e., absent from entries)
    title   : plot title

    Returns (fig, ax, counts) where counts is the integer array of candidate counts per noise hit.
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    labels = np.asarray(labels, dtype=int)
    noise_idx_all = np.flatnonzero(labels == -1)

    counts_map = Counter(ni for (ni, _, _) in entries)
    counts = np.array([counts_map.get(ni, 0) for ni in noise_idx_all], dtype=int)
    if exclude_zero:
        counts = counts[counts > 0]

    max_k = int(counts.max()) if counts.size else 0
    bins = np.arange(-0.5, max_k + 1.5, 1.0)

    fig, ax = plt.subplots()
    ax.hist(counts, bins=bins)  # integer-aligned, bars touch
    ax.set_xticks(range(0, max_k + 1))
    ax.set_xlabel("Number of candidate parent clusters per noise hit")
    ax.set_ylabel("Number of noise hits")
    ax.set_title(title)

    # quick console summary
    if counts.size:
        unique, freq = np.unique(counts, return_counts=True)
        print("[Phase3/Histogram] count -> #hits")
        for k, n in zip(unique, freq):
            print(f"{k:>2} -> {n}")
        print(f"[Phase3/Histogram] min/mean/max: {counts.min()} / {counts.mean():.2f} / {counts.max()}")

    return fig, ax, counts


# --- paste this to REPLACE your existing plot_cluster_result in cluster_fit.py ---

def plot_cluster_result(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    showNoise: bool = True,
    maxClusterid: Optional[int] = None,
    minClusterid: Optional[int] = None,
    marker_size: int = 3,
    phase3_entries: Optional[List[Tuple[int, int, float]]] = None,
    use_lines: bool = False,
    line_width: int = 3,
    equal_axes: bool = False,   # <--- new flag
):
    """
    3D scatter with Plotly.

    Behavior:
      - If labels is None: plot all hits in one color (min/max/showNoise ignored).
      - If labels provided:
          * If showNoise=False, exclude labels == -1.
          * Apply minClusterid/maxClusterid to cluster ids (>=0) only.
          * If showNoise=True, always include noise (-1) regardless of minClusterid.
      - If phase3_entries is provided (list of (noise_idx, cluster_id, distance) sorted by distance),
        then for each noise hit we take its FIRST (nearest) candidate and overlay those noise hits
        using the parent cluster’s color (marker symbol = 'diamond').
        Optionally, draw short line segments from each such noise hit to the nearest labeled point
        in that parent cluster (use_lines=True).
      - If equal_axes=True: force x/y/z to share the same numeric span (based on the largest extent
        of the actually plotted points) and use a cubic aspect ratio for physical scale.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    N = x.size

    fig = go.Figure()

    # --- Helper: color for cluster id ---
    def _palette():
        return (px.colors.qualitative.Alphabet
                + px.colors.qualitative.Light24
                + px.colors.qualitative.Set3
                + px.colors.qualitative.Bold)
    palette = _palette()
    def color_for(lab: int) -> str:
        return palette[lab % len(palette)]

    # We'll also build a mask of points that are actually plotted, for equal_axes logic
    plotted_mask = np.zeros(N, dtype=bool)

    if labels is None:
        # All points in one color
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            name=f"points ({N})",
            marker=dict(size=marker_size)
        ))
        plotted_mask[:] = True
    else:
        labels = np.asarray(labels, dtype=int)
        uniq = np.unique(labels)
        color_noise = "rgba(120,120,120,0.7)"

        # Decide cluster IDs to plot (>=0)
        clabs = sorted(l for l in uniq if l >= 0)
        if minClusterid is not None:
            clabs = [l for l in clabs if l >= int(minClusterid)]
        if maxClusterid is not None:
            clabs = [l for l in clabs if l <= int(maxClusterid)]

        # Noise first (if requested)
        if showNoise and (-1 in uniq):
            mask_noise = (labels == -1)
            if np.any(mask_noise):
                fig.add_trace(go.Scatter3d(
                    x=x[mask_noise], y=y[mask_noise], z=z[mask_noise],
                    mode="markers", name=f"noise ({mask_noise.sum()})",
                    marker=dict(size=marker_size, color=color_noise)
                ))
                plotted_mask |= mask_noise

        # Base clusters
        for lab in clabs:
            mask = (labels == lab)
            if not np.any(mask):
                continue
            fig.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask],
                mode="markers", name=f"cluster {lab} ({mask.sum()})",
                marker=dict(size=marker_size, color=color_for(lab))
            ))
            plotted_mask |= mask

        # Optional overlay: Phase-3 first-candidate attachments
        if phase3_entries is not None and len(phase3_entries) > 0:
            # first candidate per noise index
            first_for_noise: Dict[int, Tuple[int, float]] = {}
            for ni, cid, dist in phase3_entries:
                if ni not in first_for_noise:
                    first_for_noise[ni] = (cid, dist)

            # group by chosen parent cluster, respecting min/max filters
            cid_to_noise: Dict[int, List[int]] = {}
            for ni, (cid, _) in first_for_noise.items():
                if (minClusterid is not None) and (cid < int(minClusterid)):
                    continue
                if (maxClusterid is not None) and (cid > int(maxClusterid)):
                    continue
                cid_to_noise.setdefault(cid, []).append(int(ni))

            X = np.column_stack([x, y, z])

            for cid, noise_list in cid_to_noise.items():
                if len(noise_list) == 0:
                    continue
                col = color_for(cid)

                fig.add_trace(go.Scatter3d(
                    x=x[noise_list], y=y[noise_list], z=z[noise_list],
                    mode="markers",
                    name=f"phase3→{cid} ({len(noise_list)})",
                    marker=dict(size=marker_size + 2, symbol="diamond", color=col)
                ))
                plotted_mask[noise_list] = True

                if use_lines:
                    idx_lab = np.flatnonzero(labels == cid)
                    if idx_lab.size > 0:
                        tree = cKDTree(X[idx_lab], leafsize=16)
                        q = X[noise_list]
                        _, j = tree.query(q, k=1)
                        nearest_pts = X[idx_lab[j]]

                        # Multi-segment polyline with None separators
                        xs, ys, zs = [], [], []
                        for (xn, yn, zn), (xp, yp, zp) in zip(q, nearest_pts):
                            xs += [xn, xp, None]
                            ys += [yn, yp, None]
                            zs += [zn, zp, None]

                        fig.add_trace(go.Scatter3d(
                            x=xs, y=ys, z=zs, mode="lines",
                            name=f"links→{cid}",
                            line=dict(width=line_width, color=col),
                            showlegend=False
                        ))

    # --- Layout ---
    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Cluster result"
    )

    # --- Equal numeric scale on all axes (physical view) ---
    if equal_axes and np.any(plotted_mask):
        xi, yi, zi = x[plotted_mask], y[plotted_mask], z[plotted_mask]

        # Compute per-axis min/max over the *actually plotted* points
        xmin, xmax = float(np.min(xi)), float(np.max(xi))
        ymin, ymax = float(np.min(yi)), float(np.max(yi))
        zmin, zmax = float(np.min(zi)), float(np.max(zi))

        # Centers
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        # Largest span across axes
        rx = max(xmax - xmin, 1e-9)
        ry = max(ymax - ymin, 1e-9)
        rz = max(zmax - zmin, 1e-9)
        R = max(rx, ry, rz)

        # Set all three axes to same numeric span (R), centered on data
        xr = [cx - 0.5 * R, cx + 0.5 * R]
        yr = [cy - 0.5 * R, cy + 0.5 * R]
        zr = [cz - 0.5 * R, cz + 0.5 * R]

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=xr),
                yaxis=dict(range=yr),
                zaxis=dict(range=zr),
                aspectmode="cube"  # equal aspect of the rendered box
            )
        )

    return fig




# --------------------------- Convenience ---------------------------

def dbscan_stats_string(stats: Dict[str, int]) -> str:
    """Pretty one-liner for Phase-2 stats dictionaries."""
    return (f"input:{stats.get('n_input', 0)}, new_clusters:{stats.get('n_new_clusters', 0)}, "
            f"noise:{stats.get('n_noise', 0)}, start_id:{stats.get('start_id', 0)}, "
            f"final_max_label:{stats.get('final_max_label', -1)}")


__all__ = [
    "fit_cluster_labels",
    "fit_noise_list",
    "plot_phase3_candidate_hist",
    "plot_cluster_result",
    "dbscan_stats_string",
]
