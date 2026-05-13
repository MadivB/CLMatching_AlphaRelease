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



# --- helper: io_group -> TPC id (pairs) ---
def _tpc_id_from_io(io_group):
    """
    Map io_group -> TPC id assuming pairs:
      io_group 1,2   -> TPC 0
      io_group 3,4   -> TPC 1
      ...
      io_group 139,140 -> TPC 69
    """
    io_group = np.asarray(io_group, dtype=int)
    return (io_group - 1) // 2

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

    If labels is None:
        - DBSCAN over ALL hits.
        - Returns labels_out (0..K-1, -1 noise), DBSCAN_label (same), dbscan_noise_idx, stats.

    If labels is provided:
        - DBSCAN only on (labels == -1).
        - Offset cluster ids by start_id = max(existing)+1 to avoid overlap.
        - Returns labels_out, DBSCAN_label (local ids on subset), dbscan_noise_idx, stats.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    X = np.column_stack([x, y, z]).astype(np.float64)

    if labels is None:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
        labels_sub = db.fit_predict(X)  # local ids 0..K-1, -1 noise

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
                  f"new_clusters:{stats['n_new_clusters']}, noise:{stats['n_noise']}, "
                  f"start_id:{stats['start_id']}, final_max_label:{stats['final_max_label']}")
        return labels_out, DBSCAN_label, dbscan_noise_idx, stats

    labels = np.asarray(labels, dtype=int)
    mask_phase1_noise = (labels == -1)
    rem_idx = np.flatnonzero(mask_phase1_noise)

    DBSCAN_label = np.full_like(labels, -1, dtype=int)

    if rem_idx.size == 0:
        stats = dict(
            n_input=0,
            n_new_clusters=0,
            n_noise=0,
            start_id=int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0,
            final_max_label=int(labels[labels >= 0].max()) if np.any(labels >= 0) else -1,
        )
        if verbose:
            print(f"[Phase2/DBSCAN] input:0, new_clusters:0, noise:0, "
                  f"start_id:{stats['start_id']}, final_max_label:{stats['final_max_label']}")
        return labels.copy(), DBSCAN_label, np.array([], dtype=int), stats

    Xrem = X[mask_phase1_noise]
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels_sub = db.fit_predict(Xrem)  # local: 0..K-1, -1 noise

    DBSCAN_label[mask_phase1_noise] = labels_sub

    start_id = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
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
              f"new_clusters:{stats['n_new_clusters']}, noise:{stats['n_noise']}, "
              f"start_id:{stats['start_id']}, final_max_label:{stats['final_max_label']}")

    return labels_out, DBSCAN_label, dbscan_noise_idx, stats


# --------------------------- Phase 3: distance list ---------------------------

def fit_noise_list(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    labels: np.ndarray,
    *,
    expand_frac: float = 0.10,
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
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    N = x.size

    fig = go.Figure()

    if labels is None:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            name=f"points ({N})",
            marker=dict(size=marker_size)
        ))
    else:
        labels = np.asarray(labels, dtype=int)

        # Palette + colors
        palette = (
            px.colors.qualitative.Alphabet
            + px.colors.qualitative.Light24
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Bold
        )
        def color_for(lab: int) -> str:
            return palette[lab % len(palette)]

        color_noise = "rgba(120,120,120,0.7)"
        uniq = np.unique(labels)

        # Noise first (if requested)
        if showNoise and (-1 in uniq):
            mask_noise = labels == -1
            if np.any(mask_noise):
                fig.add_trace(go.Scatter3d(
                    x=x[mask_noise], y=y[mask_noise], z=z[mask_noise],
                    mode="markers", name=f"noise ({mask_noise.sum()})",
                    marker=dict(size=marker_size, color=color_noise)
                ))

        # Cluster subset
        clabs = sorted(l for l in uniq if l >= 0)
        if minClusterid is not None:
            clabs = [l for l in clabs if l >= int(minClusterid)]
        if maxClusterid is not None:
            clabs = [l for l in clabs if l <= int(maxClusterid)]

        # Base cluster points
        for lab in clabs:
            mask = (labels == lab)
            if not np.any(mask):
                continue
            fig.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask],
                mode="markers", name=f"cluster {lab} ({mask.sum()})",
                marker=dict(size=marker_size, color=color_for(lab))
            ))

        # Optional overlay: Phase-3 first-candidate attachments
        if phase3_entries is not None and len(phase3_entries) > 0:
            # First candidate for each noise index
            first_for_noise: Dict[int, Tuple[int, float]] = {}
            for ni, cid, dist in phase3_entries:
                if ni not in first_for_noise:
                    first_for_noise[ni] = (cid, dist)

            # Group noise indices by chosen cluster id
            cid_to_noise: Dict[int, List[int]] = {}
            for ni, (cid, _) in first_for_noise.items():
                if (minClusterid is not None) and (cid < int(minClusterid)):
                    continue
                if (maxClusterid is not None) and (cid > int(maxClusterid)):
                    continue
                cid_to_noise.setdefault(cid, []).append(int(ni))

            # Overlay markers (and optional lines) per cluster
            X = np.column_stack([x, y, z])
            for cid, noise_list in cid_to_noise.items():
                if len(noise_list) == 0:
                    continue
                col = color_for(cid)

                # Draw the noise points colored by their parent cluster
                fig.add_trace(go.Scatter3d(
                    x=x[noise_list], y=y[noise_list], z=z[noise_list],
                    mode="markers",
                    name=f"phase3→{cid} ({len(noise_list)})",
                    marker=dict(size=marker_size + 2, symbol="diamond", color=col)
                ))

                if use_lines:
                    # Build KDTree for the labeled points of this cluster
                    idx_lab = np.flatnonzero(labels == cid)
                    if idx_lab.size > 0:
                        tree = cKDTree(X[idx_lab], leafsize=16)
                        q = X[noise_list]
                        _, j = tree.query(q, k=1)
                        nearest_pts = X[idx_lab[j]]

                        # Create a single multi-segment trace with None separators
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
    return fig


def auto_disband_single_cluster_gmm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    max_k: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Given hits from ONE over-merged cluster (x,y,z), automatically decide
    how many subclusters K to use and assign subcluster labels using
    a Gaussian Mixture Model (GMM) with BIC-based model selection.

    Parameters
    ----------
    x, y, z : (N,) arrays
        Coordinates of hits belonging to ONE cluster.
    max_k : int
        Maximum number of subclusters to consider (we scan K=1..max_k).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sub_labels : (N,) int
        Local subcluster labels 0..K*-1 (K* is chosen automatically).
    best_k : int
        Number of subclusters chosen (K*). If best_k == 1, the cluster
        is effectively unimodal (no meaningful split).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    N = x.size
    if N == 0:
        raise ValueError("auto_disband_single_cluster_gmm: no hits provided.")
    if N == 1:
        # Only one hit: there is nothing to split.
        return np.zeros(1, dtype=int), 1

    X = np.column_stack([x, y, z])

    # Limit max_k so that K <= N
    K_max = min(max_k, N)
    if K_max < 1:
        K_max = 1

    bics: List[float] = []
    gmms: List[GaussianMixture] = []

    for k in range(1, K_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        bics.append(float(bic))
        gmms.append(gmm)

    bics_arr = np.asarray(bics, dtype=float)
    best_idx = int(np.argmin(bics_arr))
    best_k = best_idx + 1  # because we started from k=1

    best_gmm = gmms[best_idx]
    sub_labels = best_gmm.predict(X)

    return sub_labels.astype(int), int(best_k)


def disband_clusters_gmm_global(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    target_cluster_ids,
    *,
    max_k: int = 5,
    random_state: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    Disband one or more over-merged clusters in a global label array,
    automatically deciding how many subclusters each should be split into
    using GMM + BIC on (x,y,z).

    Parameters
    ----------
    x, y, z : (N,) arrays
        Coordinates of ALL hits.
    labels : (N,) int
        Global cluster labels (e.g. labels after Phase 3).
    target_cluster_ids : int or sequence of int
        Which global cluster ids should be tested and potentially split.
    max_k : int
        Maximum number of subclusters per target cluster.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        If True, print a short summary for each processed cluster.

    Returns
    -------
    labels_new : (N,) int
        Updated global labels after disbanding.
    split_info : list of dict
        One entry per processed target cluster, e.g.:
          {
            "cluster_id": 140,
            "status": "split",
            "best_k": 3,
            "new_labels": [200, 201, 202],
          }
        or:
          {
            "cluster_id": 123,
            "status": "not_split",
            "best_k": 1,
            "new_labels": [123],
          }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # Normalize target_cluster_ids to a list[int]
    if np.isscalar(target_cluster_ids):
        target_list = [int(target_cluster_ids)]
    else:
        target_list = [int(c) for c in target_cluster_ids]

    labels_new = labels.copy()
    split_info: List[Dict[str, object]] = []

    # Start assigning new labels above the current max
    if np.any(labels_new >= 0):
        next_label = int(labels_new[labels_new >= 0].max()) + 1
    else:
        next_label = 0

    for cid in target_list:
        mask = labels_new == cid
        n_hits = int(mask.sum())
        if n_hits == 0:
            split_info.append(
                dict(cluster_id=cid, status="no_hits", best_k=0, new_labels=[])
            )
            if verbose:
                print(f"[disband_clusters_gmm_global] cluster {cid}: no hits found.")
            continue

        x_c = x[mask]
        y_c = y[mask]
        z_c = z[mask]

        sub_labels_local, best_k = auto_disband_single_cluster_gmm(
            x_c, y_c, z_c,
            max_k=max_k,
            random_state=random_state,
        )

        if best_k <= 1:
            # No meaningful split
            split_info.append(
                dict(
                    cluster_id=cid,
                    status="not_split",
                    best_k=best_k,
                    new_labels=[cid],
                )
            )
            if verbose:
                print(
                    f"[disband_clusters_gmm_global] cluster {cid}: best_k={best_k} "
                    "(no split applied)."
                )
            continue

        # Assign global labels for each local subcluster
        new_global_ids: List[int] = []
        for local_id in range(best_k):
            global_id = next_label
            next_label += 1
            new_global_ids.append(global_id)

            # indices of hits in this local subcluster (within the mask)
            take = mask.copy()
            take[mask] = sub_labels_local == local_id
            labels_new[take] = global_id

        split_info.append(
            dict(
                cluster_id=cid,
                status="split",
                best_k=best_k,
                new_labels=new_global_ids,
            )
        )

        if verbose:
            print(
                f"[disband_clusters_gmm_global] cluster {cid}: split into "
                f"{best_k} subclusters -> new global ids {new_global_ids}"
            )

    return labels_new, split_info

# --------------------------- Convenience ---------------------------

def dbscan_stats_string(stats: Dict[str, int]) -> str:
    """Pretty one-liner for Phase-2 stats dictionaries."""
    return (f"input:{stats.get('n_input', 0)}, new_clusters:{stats.get('n_new_clusters', 0)}, "
            f"noise:{stats.get('n_noise', 0)}, start_id:{stats.get('start_id', 0)}, "
            f"final_max_label:{stats.get('final_max_label', -1)}")

'''
Example usage
labels_phase4, split_info = disband_clusters_gmm_global(
    xset, yset, zset,
    labels_phase3,
    target_cluster_ids=[140, ...],
    max_k=5,
)
'''

__all__ = [
    "fit_cluster_labels",
    "fit_noise_list",
    "plot_phase3_candidate_hist",
    "plot_cluster_result",
    "dbscan_stats_string",
    "_tpc_id_from_io",   # <-- optional
    "auto_disband_single_cluster_gmm",
    "disband_clusters_gmm_global"
]

