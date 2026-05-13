"""
advanced_clustering.py
======================
Advanced interaction-vertex clustering for NDLAr charge hits.

Replaces the DBSCAN endpoint clustering in global_track_clustering.py with
four physics-motivated approaches. All methods are CONSERVATIVE (prefer
over-splitting over over-merging), which is ideal for charge-light matching.

Methods
-------
  nhc   – Neighbouring-Aware Hierarchical Clustering
            Graph-based: edges between tracks with small DCA + non-parallel angle.
            Connected components → vertex groups.
  hca   – Hierarchical Complete-Linkage on DCA
            Complete linkage = merges only when ALL pairs in two groups satisfy
            DCA < threshold. Maximally conservative.
  dvfs  – DCA Vertex-Finding with Seeding
            Compute closest-approach midpoints for all compatible track pairs,
            cluster those 3D seed points to find vertex candidates, assign tracks.
  spec  – Physics-Informed Spectral Clustering (SPINEX-inspired)
            DCA-based affinity matrix + spectral clustering with eigengap K.

All methods:
  - Keep the existing per-TPC RANSAC track fitting (unchanged).
  - Keep the cross-TPC segment matching (unchanged).
  - Replace only the endpoint/vertex grouping step.
  - Use KNN attachment (not DBSCAN) for leftover noise hits.

Drop-in API
-----------
    from advanced_clustering import build_advanced_labels
    labels, split_index = build_advanced_labels(
        x, y, z, io_group, method='nhc',
        plotting=True, out_prefix='ev0', plot_dir='./results'
    )
"""

from __future__ import annotations
import os
import sys
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage as sp_linkage, fcluster
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import plotly.express as px

# Re-use existing tracking code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_track_clustering import (
    _line_line_dca,
    _angle_diff_deg,
    _build_tpc_segments,
    _match_segments_across_tpcs,
    _PALETTE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _closest_approach_midpoint(p1, v1, p2, v2):
    """
    3D midpoint of the closest-approach segment between two infinite lines.
    Returns (midpoint_xyz, s, t, distance).
    """
    p1, v1 = np.asarray(p1, float), np.asarray(v1, float)
    p2, v2 = np.asarray(p2, float), np.asarray(v2, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.5 * (p1 + p2), 0.0, 0.0, float(np.linalg.norm(p1 - p2))
    v1, v2 = v1 / n1, v2 / n2
    w = p1 - p2
    a, b, c = np.dot(v1, v1), np.dot(v1, v2), np.dot(v2, v2)
    d, e   = np.dot(v1, w),  np.dot(v2, w)
    denom  = a * c - b * b
    if denom < 1e-12:
        s, t = 0.0, (e / c if c > 1e-12 else 0.0)
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    c1, c2 = p1 + s * v1, p2 + t * v2
    return 0.5 * (c1 + c2), s, t, float(np.linalg.norm(c1 - c2))


def _build_dca_matrix(tracks):
    """Symmetric M×M pairwise DCA matrix between global track infinite lines."""
    M = len(tracks)
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            d = _line_line_dca(tracks[i]['point'], tracks[i]['direction'],
                               tracks[j]['point'], tracks[j]['direction'])
            D[i, j] = D[j, i] = d
    return D


def _build_angle_matrix(tracks):
    """Symmetric M×M angle matrix (degrees, 0–90°)."""
    M = len(tracks)
    A = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            a = _angle_diff_deg(tracks[i]['direction'], tracks[j]['direction'])
            A[i, j] = A[j, i] = a
    return A


# ─────────────────────────────────────────────────────────────────────────────
#  Hit-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _absorb_noise_knn(x, y, z, labels, knn_radius=4.0):
    """
    Attach unassigned hits (label == -1) to the nearest assigned hit
    if within knn_radius. Leaves truly isolated hits as -1.
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    labels   = np.array(labels, dtype=int)
    assign_m = labels >= 0
    noise_m  = labels == -1
    if not np.any(assign_m) or not np.any(noise_m):
        return labels
    X = np.column_stack([x, y, z])
    tree = cKDTree(X[assign_m])
    dists, idxs = tree.query(X[noise_m], k=1, workers=-1)
    ok = dists <= knn_radius
    noise_idx = np.flatnonzero(noise_m)
    labels[noise_idx[ok]] = labels[assign_m][idxs[ok]]
    return labels


def _tracks_to_hitlabels(global_tracks, track_group_ids, x, y, z, knn_radius=4.0):
    """
    Convert per-track group IDs → per-hit labels.
    Compact sequential IDs; absorb noise via KNN.
    Returns (labels_global, split_index).
    """
    N = len(x)
    labels = np.full(N, -1, dtype=int)
    uniq   = sorted(set(track_group_ids))
    g_map  = {old: new for new, old in enumerate(uniq)}
    split_index = len(uniq)
    for i, gt in enumerate(global_tracks):
        labels[gt['hit_indices']] = g_map[track_group_ids[i]]
    labels = _absorb_noise_knn(x, y, z, labels, knn_radius=knn_radius)
    return labels, split_index


# ─────────────────────────────────────────────────────────────────────────────
#  Method 1 — NHC: Neighbouring-Aware Hierarchical Clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_tracks_nhc(
    global_tracks, *,
    dca_tol=5.0,
    angle_min_deg=5.0,
    endpoint_proximity_cm=20.0,
):
    """
    NHC: build proximity graph over global tracks and return connected components.

    An edge (i,j) is added when:
      1. DCA between infinite lines < dca_tol
      2. Opening angle > angle_min_deg  (filter same-particle parallel pairs)
      3. At least one endpoint pair is within endpoint_proximity_cm

    Connected components define vertex groups. Conservative by design.
    """
    M = len(global_tracks)
    if M <= 1:
        return list(range(M))

    D = _build_dca_matrix(global_tracks)
    A = _build_angle_matrix(global_tracks)

    parent = np.arange(M)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(M):
        pA_i, pB_i = global_tracks[i]['endpoints']
        for j in range(i + 1, M):
            if D[i, j] > dca_tol:
                continue
            if A[i, j] < angle_min_deg:
                continue
            pA_j, pB_j = global_tracks[j]['endpoints']
            ep_dist = min(
                np.linalg.norm(pA_i - pA_j), np.linalg.norm(pA_i - pB_j),
                np.linalg.norm(pB_i - pA_j), np.linalg.norm(pB_i - pB_j),
            )
            if ep_dist > endpoint_proximity_cm:
                continue
            union(i, j)

    groups = [find(i) for i in range(M)]
    uniq   = sorted(set(groups))
    g_map  = {g: k for k, g in enumerate(uniq)}
    return [g_map[g] for g in groups]


# ─────────────────────────────────────────────────────────────────────────────
#  Method 2 — HCA: Hierarchical Complete-Linkage on DCA
# ─────────────────────────────────────────────────────────────────────────────

def cluster_tracks_hca(
    global_tracks, *,
    dca_cut=6.0,
    angle_min_deg=5.0,
    linkage_method='complete',
):
    """
    HCA: complete-linkage hierarchical clustering on pairwise DCA.

    Complete linkage merges two groups only when ALL pairs satisfy DCA < dca_cut.
    Parallel-track pairs (angle < angle_min_deg) get DCA set to 1e9 (no merge).
    Most conservative scipy linkage option; extremely resistant to chaining.
    """
    M = len(global_tracks)
    if M <= 1:
        return list(range(M))

    D   = _build_dca_matrix(global_tracks)
    Ang = _build_angle_matrix(global_tracks)

    D_eff = D.copy()
    D_eff[Ang < angle_min_deg] = 1e9
    np.fill_diagonal(D_eff, 0.0)

    condensed = squareform(D_eff, checks=False)
    Z      = sp_linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=dca_cut, criterion='distance')
    return (labels - 1).tolist()   # 0-indexed


# ─────────────────────────────────────────────────────────────────────────────
#  Method 3 — DVFS: DCA Vertex-Finding with Seeding
# ─────────────────────────────────────────────────────────────────────────────

def cluster_tracks_dvfs(
    global_tracks, *,
    dca_tol=8.0,
    seed_eps=5.0,
    angle_min_deg=5.0,
    point_to_line_tol=6.0,
):
    """
    DVFS: collect 3D closest-approach midpoints for compatible track pairs,
    cluster those "DCA seeds" in 3D, then assign each track to the nearest
    vertex candidate.

    This is the most "physics vertex finder"-like approach:
      - Seeds accumulate near the true interaction vertex.
      - Seed clustering gives a robust 3D vertex position estimate.
      - Track assignment is based on distance from track line to vertex point.
    """
    M = len(global_tracks)
    if M <= 1:
        return list(range(M))

    D   = _build_dca_matrix(global_tracks)
    Ang = _build_angle_matrix(global_tracks)

    # Step 1: collect DCA seeds
    seeds = []
    for i in range(M):
        for j in range(i + 1, M):
            if D[i, j] > dca_tol or Ang[i, j] < angle_min_deg:
                continue
            mid, _, _, _ = _closest_approach_midpoint(
                global_tracks[i]['point'], global_tracks[i]['direction'],
                global_tracks[j]['point'], global_tracks[j]['direction'],
            )
            seeds.append((mid, i, j))

    if not seeds:
        return list(range(M))

    seed_pts = np.array([s[0] for s in seeds])

    # Step 2: single-linkage cluster seeds in 3D
    K = len(seeds)
    sp = np.arange(K)

    def sfind(a):
        while sp[a] != a:
            sp[a] = sp[sp[a]]; a = sp[a]
        return a

    def sunion(a, b):
        ra, rb = sfind(a), sfind(b)
        if ra != rb:
            sp[rb] = ra

    tree_s = cKDTree(seed_pts)
    for a in range(K):
        for b in tree_s.query_ball_point(seed_pts[a], seed_eps):
            if b > a:
                sunion(a, b)

    sc = [sfind(i) for i in range(K)]
    uniq_sc = sorted(set(sc))
    sc_map  = {g: k for k, g in enumerate(uniq_sc)}
    sc      = [sc_map[g] for g in sc]
    n_vtx   = len(uniq_sc)

    # Step 3: compute centroid of each seed cluster → vertex candidate
    vtx_pts = np.zeros((n_vtx, 3))
    vtx_cnt = np.zeros(n_vtx)
    for i, (mid, _, _) in enumerate(seeds):
        vtx_pts[sc[i]] += mid
        vtx_cnt[sc[i]] += 1.0
    vtx_pts /= np.maximum(vtx_cnt[:, None], 1.0)

    # Step 4: assign each track to nearest vertex candidate
    track_groups = np.full(M, -1, dtype=int)
    for i in range(M):
        pt, v = global_tracks[i]['point'], global_tracks[i]['direction']
        best_v, best_d = -1, point_to_line_tol + 1.0
        for vid, vp in enumerate(vtx_pts):
            diff = vp - pt
            proj = pt + np.dot(diff, v) * v
            d = float(np.linalg.norm(proj - vp))
            if d < best_d:
                best_d, best_v = d, vid
        track_groups[i] = best_v if best_v >= 0 else -1

    # Isolated tracks get unique IDs
    next_id = n_vtx
    for i in range(M):
        if track_groups[i] < 0:
            track_groups[i] = next_id
            next_id += 1

    return track_groups.tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  Method 4 — SPEC: Physics-Informed Spectral Clustering (SPINEX-inspired)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_tracks_spectral(
    global_tracks, *,
    sigma=4.0,
    max_dca=10.0,
    angle_min_deg=5.0,
    max_clusters=25,
    min_clusters=1,
    eigengap_threshold=0.08,
):
    """
    SPEC: spectral clustering on a physics-informed affinity graph.

    Affinity A[i,j] = exp(-DCA²/2σ²)  if DCA < max_dca and angle > min_deg
                      0                 otherwise.

    K is chosen by the eigengap heuristic on the normalised Laplacian,
    which naturally finds the number of well-separated communities.
    K-means is run in the spectral embedding space.
    """
    try:
        from scipy.sparse.csgraph import laplacian as sp_laplacian
        from scipy.linalg import eigh
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        return cluster_tracks_hca(global_tracks, dca_cut=max_dca,
                                   angle_min_deg=angle_min_deg)

    M = len(global_tracks)
    if M <= 1:
        return list(range(M))

    D   = _build_dca_matrix(global_tracks)
    Ang = _build_angle_matrix(global_tracks)

    Aff = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            if D[i, j] <= max_dca and Ang[i, j] >= angle_min_deg:
                w = np.exp(-D[i, j] ** 2 / (2.0 * sigma ** 2))
                Aff[i, j] = Aff[j, i] = w

    # If graph is totally disconnected, fall back to HCA
    if Aff.sum() == 0:
        return cluster_tracks_hca(global_tracks, dca_cut=max_dca,
                                   angle_min_deg=angle_min_deg)

    L, _ = sp_laplacian(Aff, normed=True, return_diag=True)
    n_eigs = min(max_clusters + 2, M)
    try:
        eigvals, eigvecs = eigh(L, subset_by_index=[0, n_eigs - 1])
    except Exception:
        return cluster_tracks_hca(global_tracks, dca_cut=max_dca,
                                   angle_min_deg=angle_min_deg)

    # Eigengap: last large gap → K
    gaps = np.diff(eigvals)
    K = min_clusters
    for k in range(1, len(gaps)):
        if gaps[k - 1] > eigengap_threshold:
            K = k + 1
        if gaps[k - 1] < eigengap_threshold * 0.5 and k > 1:
            break
    K = max(min_clusters, min(K, max_clusters, M))

    U = normalize(eigvecs[:, :K], norm='l2', axis=1)
    km = KMeans(n_clusters=K, random_state=0, n_init=10, max_iter=300)
    return km.fit_predict(U).tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_MAP = {
    'nhc':  cluster_tracks_nhc,
    'hca':  cluster_tracks_hca,
    'dvfs': cluster_tracks_dvfs,
    'spec': cluster_tracks_spectral,
}


def build_advanced_labels(
    x, y, z, io_group,
    method='nhc',
    # Per-TPC RANSAC tracking params
    lam=1.5, rss_threshold=1.5e6, iters=800, min_inliers=35,
    k_for_scale=8, attach_multiplier=1.3, seed=0,
    min_length_cm=30.0, n_tpcs=70,
    # Cross-TPC matching
    match_dist_tol=4.0, match_angle_deg=10.0,
    # Per-method params (dict → forwarded to method function)
    method_kwargs=None,
    # Noise absorption
    knn_radius=4.0,
    # Plotting
    plotting=False,
    out_prefix='event',
    plot_dir='.',
    global_tracks_out=None,   # if list provided, global_tracks written in-place
):
    """
    Full pipeline — drop-in replacement for build_global_labels().

    Returns
    -------
    labels_global : (N,) int  — hit-level cluster ID, -1 = noise/unassigned
    split_index   : int       — labels 0..split_index-1 are track/vertex clusters
    """
    x         = np.asarray(x, float)
    y         = np.asarray(y, float)
    z         = np.asarray(z, float)
    io_group  = np.asarray(io_group, int)
    N         = len(x)
    method_kwargs = method_kwargs or {}

    # Step 1 — per-TPC RANSAC segments (unchanged from original)
    segments = _build_tpc_segments(
        x, y, z, io_group,
        lam=lam, rss_threshold=rss_threshold, iters=iters,
        min_inliers=min_inliers, k_for_scale=k_for_scale,
        attach_multiplier=attach_multiplier, seed=seed,
        min_length_cm=min_length_cm, n_tpcs=n_tpcs,
    )
    if not segments:
        return np.full(N, -1, dtype=int), 0

    # Step 2 — cross-TPC matching (unchanged from original)
    global_tracks = _match_segments_across_tpcs(
        segments, x, y, z,
        dist_tol=match_dist_tol, angle_tol_deg=match_angle_deg,
    )
    if global_tracks_out is not None:
        global_tracks_out.clear()
        global_tracks_out.extend(global_tracks)

    if not global_tracks:
        labels = np.full(N, -1, dtype=int)
        for gid, seg in enumerate(segments):
            labels[seg['hits']] = gid
        return labels, len(segments)

    # Step 3 — advanced vertex clustering
    cluster_fn = _METHOD_MAP.get(method)
    if cluster_fn is None:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_METHOD_MAP)}")

    track_group_ids = cluster_fn(global_tracks, **method_kwargs)

    # Step 4 — propagate to hit level + KNN noise absorption
    labels_global, split_index = _tracks_to_hitlabels(
        global_tracks, track_group_ids, x, y, z, knn_radius=knn_radius,
    )

    # Step 5 — optional plotting
    if plotting:
        os.makedirs(plot_dir, exist_ok=True)
        fig = plot_clustering_result(
            x, y, z, labels_global, global_tracks,
            title=f"{method.upper()} — {out_prefix}",
            split_index=split_index,
        )
        fname = os.path.join(plot_dir, f"{out_prefix}_{method}.html")
        fig.write_html(fname)
        print(f"[plot] {fname}")

    return labels_global, split_index


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_clustering_result(x, y, z, labels, global_tracks=None,
                            title='', split_index=None):
    """Plotly 3D scatter coloured by cluster ID with optional track-fit lines."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    labels   = np.asarray(labels, int)
    fig = go.Figure()

    noise = labels == -1
    if np.any(noise):
        fig.add_scatter3d(
            x=x[noise], y=y[noise], z=z[noise], mode='markers',
            marker=dict(size=1.5, color='rgba(150,150,150,0.3)'),
            name=f'Noise ({noise.sum()})',
        )

    for lab in sorted(l for l in np.unique(labels) if l >= 0):
        color = _PALETTE[lab % len(_PALETTE)]
        m = labels == lab
        tag = 'V' if (split_index is None or lab < split_index) else 'C'
        fig.add_scatter3d(
            x=x[m], y=y[m], z=z[m], mode='markers',
            marker=dict(size=2, color=color, opacity=0.85),
            name=f'{tag}{lab} (N={m.sum()})',
            legendgroup=f'clus_{lab}',
        )

    if global_tracks:
        for gt in global_tracks:
            pA, pB = gt['endpoints']
            hit_labs = [l for l in np.unique(labels[gt['hit_indices']]) if l >= 0]
            col = _PALETTE[hit_labs[0] % len(_PALETTE)] if hit_labs else 'gray'
            fig.add_scatter3d(
                x=[pA[0], pB[0]], y=[pA[1], pB[1]], z=[pA[2], pB[2]],
                mode='lines', line=dict(width=4, color=col),
                showlegend=False,
            )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X (cm)', yaxis_title='Y (cm)', zaxis_title='Z (cm)'),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing='constant'),
    )
    return fig


def plot_truth_result(x, y, z, truth_labels, title='Truth Vertex IDs'):
    """Plot hits coloured by truth vertex_id."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    truth    = np.asarray(truth_labels, int)
    fig = go.Figure()

    noise = truth < 0
    if np.any(noise):
        fig.add_scatter3d(
            x=x[noise], y=y[noise], z=z[noise], mode='markers',
            marker=dict(size=1.5, color='rgba(150,150,150,0.4)'),
            name=f'No truth ({noise.sum()})',
        )

    for lab in sorted(l for l in np.unique(truth) if l >= 0):
        m = truth == lab
        color = _PALETTE[lab % len(_PALETTE)]
        fig.add_scatter3d(
            x=x[m], y=y[m], z=z[m], mode='markers',
            marker=dict(size=2, color=color, opacity=0.85),
            name=f'vtx_{lab} (N={m.sum()})',
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X (cm)', yaxis_title='Y (cm)', zaxis_title='Z (cm)'),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing='constant'),
    )
    return fig


def plot_metrics_table(metrics_df, title='Clustering Method Comparison'):
    """Plotly table summarising per-method evaluation metrics."""
    import pandas as pd
    if not isinstance(metrics_df, pd.DataFrame):
        metrics_df = pd.DataFrame(metrics_df)
    fig = go.Figure(go.Table(
        header=dict(
            values=['<b>' + c + '</b>' for c in metrics_df.columns],
            fill_color='#1a1a2e', font=dict(color='white', size=13),
            align='center', height=32,
        ),
        cells=dict(
            values=[metrics_df[c].tolist() for c in metrics_df.columns],
            fill_color=[['#16213e', '#0f3460'] * (len(metrics_df) // 2 + 1)],
            font=dict(color='white', size=12),
            align='center', height=28,
        ),
    ))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig
