# track_fit_ransac.py
import numpy as np
from scipy.spatial import cKDTree
# --- add this near the top of multi_track_fit.py ---
import plotly.graph_objects as go
import plotly.express as px


# ---------- helpers ----------
def _line_distances(points, p0, v):
    """Distance of points (N,3) to infinite line through p0 with unit dir v."""
    dif = points - p0
    return np.linalg.norm(np.cross(dif, v), axis=1)

def _refit_line(points):
    """Best-fit infinite line via SVD on centered points. Returns (c, v, (pA,pB))."""
    c = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
    v = Vt[0]
    v /= (np.linalg.norm(v) + 1e-12)
    t = (points - c) @ v
    return c, v, (c + t.min()*v, c + t.max()*v)

def _auto_dist_thresh(points, k_for_scale=8, mult=1.5):
    """kNN scale (median k-th neighbor) -> distance threshold."""
    n = len(points)
    if n <= k_for_scale:
        return 3.0
    tree = cKDTree(points)
    d = tree.query(points, k=k_for_scale+1, workers=-1)[0][:, -1]
    return mult * float(np.median(d))

# ---------- RANSAC single line ----------
def ransac_line_3d(points, iters=1200, dist_thresh=None, min_inliers=35,
                   k_for_scale=8, seed=None):
    """RANSAC a single 3D line on (N,3) points. Returns dict or None."""
    N = len(points)
    if N < 2:
        return None
    rng = np.random.default_rng(seed)
    if dist_thresh is None:
        dist_thresh = _auto_dist_thresh(points, k_for_scale=k_for_scale, mult=1.5)

    best_mask, best_cnt = None, 0
    for _ in range(iters):
        i, j = rng.choice(N, size=2, replace=False)
        p0, p1 = points[i], points[j]
        v = p1 - p0
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v /= nv
        d = _line_distances(points, p0, v)
        mask = d < dist_thresh
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_mask = cnt, mask

    if best_mask is None or best_cnt < min_inliers:
        return None

    in_points = points[best_mask]
    c, v, (pA, pB) = _refit_line(in_points)
    d_all = _line_distances(points, c, v)
    in_ref = d_all < dist_thresh
    rms = float(np.sqrt(np.mean(d_all[in_ref]**2))) if in_ref.any() else np.nan

    return {
        "point": c, "direction": v, "endpoints": (pA, pB),
        "inlier_mask": in_ref, "dist_thresh": float(dist_thresh),
        "n_inliers": int(in_ref.sum()), "n_total": int(N), "rms_dist": rms,
    }

# ---------- greedy multi-line extraction ----------
def fit_multiple_tracks(x, y, z, iters=1200, min_inliers=35,
                        dist_thresh=None, k_for_scale=8,
                        attach_outliers=True, attach_multiplier=1.3,
                        seed=0):
    """
    Greedy RANSAC extraction of an unbounded number of straight tracks.
    Optionally attach remaining points to nearest track.
    Returns tracks (list of dicts) and labels (N,).
    """
    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    N = len(X)
    alive = np.ones(N, dtype=bool)
    tracks = []
    rng = np.random.default_rng(seed)

    if dist_thresh is None:
        dist_thresh = _auto_dist_thresh(X, k_for_scale=k_for_scale, mult=1.5)

    while True:
        Xi = X[alive]
        if len(Xi) < min_inliers:
            break
        res = ransac_line_3d(Xi, iters=iters, dist_thresh=dist_thresh,
                             min_inliers=min_inliers, k_for_scale=k_for_scale,
                             seed=rng.integers(1 << 31))
        if res is None:
            break

        local_in = res["inlier_mask"]
        global_idx = np.flatnonzero(alive)
        chosen = global_idx[local_in]
        c, v, (pA, pB) = _refit_line(X[chosen])

        tracks.append({
            "name": f"Track {len(tracks)+1}",
            "point": c, "direction": v, "endpoints": (pA, pB),
            "indices": chosen,
            "rms_dist": float(np.sqrt(np.mean(_line_distances(X[chosen], c, v)**2))) if len(chosen) else np.nan,
            "dist_thresh": float(dist_thresh),
        })
        alive[chosen] = False

    labels = np.full(N, -1, dtype=int)
    for tid, t in enumerate(tracks):
        labels[t["indices"]] = tid

    if attach_outliers and len(tracks) > 0:
        remaining = np.flatnonzero(labels == -1)
        if len(remaining) > 0:
            M, T = len(remaining), len(tracks)
            D = np.empty((M, T), dtype=float)
            for j, t in enumerate(tracks):
                D[:, j] = _line_distances(X[remaining], t["point"], t["direction"])
            nearest = np.argmin(D, axis=1)
            dmin = D[np.arange(M), nearest]
            thr = np.array([tr["dist_thresh"] * attach_multiplier for tr in tracks], dtype=float)
            ok = dmin <= thr[nearest]
            for ii in np.flatnonzero(ok):
                labels[remaining[ii]] = int(nearest[ii])

        # final refit per track
        for tid, t in enumerate(tracks):
            members = np.flatnonzero(labels == tid)
            if len(members) >= 2:
                c, v, (pA, pB) = _refit_line(X[members])
                rms = float(np.sqrt(np.mean(_line_distances(X[members], c, v)**2)))
                tracks[tid].update({"point": c, "direction": v, "endpoints": (pA, pB),
                                    "indices": members, "rms_dist": rms})

    # sort by size
    order = np.argsort([-len(t["indices"]) for t in tracks])
    tracks = [tracks[i] for i in order]

    # remap labels to sorted order
    id_map = {old: i for i, old in enumerate(order)}
    if len(tracks) > 0:
        for i in range(N):
            if labels[i] != -1:
                labels[i] = id_map[labels[i]]
        for i, t in enumerate(tracks, 1):
            t["name"] = f"Track {i}"

    return tracks, labels

# ---------- evaluation & selection ----------
def evaluate_RSS_curve(x, y, z, tracks, param_count_per_track=5):
    """
    For ordered tracks (by size), compute RSS for K=1..T by assigning
    each point to the nearest among first K tracks.
    """
    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    N = len(X); T = len(tracks)
    if N == 0 or T == 0:
        return {"K": [], "RSS": [], "BIC": [], "rel_improve": []}

    D = np.empty((N, T), dtype=float)
    for j, tr in enumerate(tracks):
        D[:, j] = _line_distances(X, tr["point"], tr["direction"])

    Ks, RSSs, BICs, rel = [], [], [], []
    prev_RSS = None
    for K in range(1, T + 1):
        nearest = np.argmin(D[:, :K], axis=1)
        dmin = D[np.arange(N), nearest]
        RSS = float(np.sum(dmin**2))
        sigma2 = max(RSS / max(N, 1), 1e-12)
        BIC = N*np.log(sigma2) + (param_count_per_track*K)*np.log(max(N, 1))
        Ks.append(K); RSSs.append(RSS); BICs.append(BIC)
        rel.append(np.nan if prev_RSS is None else (prev_RSS - RSS) / max(prev_RSS, 1e-12))
        prev_RSS = RSS
    return {"K": Ks, "RSS": RSSs, "BIC": BICs, "rel_improve": rel}

def decide_k_by_rss_threshold(info, rss_threshold=1_500_000.0):
    """
    Choose the smallest K with RSS <= rss_threshold.
    If none qualifies, pick K with minimum BIC; else argmin RSS.
    """
    Ks = np.asarray(info["K"]); RSS = np.asarray(info["RSS"], dtype=float)
    ok = np.where(RSS <= rss_threshold)[0]
    if len(ok):
        return int(Ks[ok[0]]), "rss_cut"
    BIC = np.asarray(info.get("BIC", []), dtype=float)
    if len(BIC) == len(Ks) and len(BIC) > 0:
        return int(Ks[np.argmin(BIC)]), "bic_fallback"
    return int(Ks[np.argmin(RSS)]), "rss_min_fallback"

def select_top_k_tracks(x, y, z, tracks, labels, k=3,
                        reattach=True, attach_multiplier=1.25):
    """
    Keep only top-K tracks; reattach leftover points to kept tracks; refit.
    """
    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    N = len(X)
    if len(tracks) == 0 or k <= 0:
        return [], np.full(N, -1, dtype=int)

    sizes = np.array([len(t["indices"]) for t in tracks], dtype=int)
    order = np.argsort(-sizes)
    keep_ids = order[:min(k, len(tracks))]

    labels_k = np.full(N, -1, dtype=int)
    kept = []
    for new_tid, old_tid in enumerate(keep_ids):
        t = tracks[old_tid]
        members = np.asarray(t["indices"], dtype=int)
        labels_k[members] = new_tid
        kept.append({
            "name": f"Track {new_tid+1}",
            "point": np.asarray(t["point"], dtype=float),
            "direction": np.asarray(t["direction"], dtype=float),
            "endpoints": t["endpoints"],
            "indices": members,
            "rms_dist": float(t.get("rms_dist", np.nan)),
            "dist_thresh": float(t.get("dist_thresh", 3.0)),
        })

    if reattach and len(kept) > 0:
        remaining = np.flatnonzero(labels_k == -1)
        if len(remaining) > 0:
            M, T = len(remaining), len(kept)
            D = np.empty((M, T), dtype=float)
            for j, tr in enumerate(kept):
                D[:, j] = _line_distances(X[remaining], tr["point"], tr["direction"])
            nearest = np.argmin(D, axis=1)
            dmin = D[np.arange(M), nearest]
            thr = np.array([tr["dist_thresh"] * attach_multiplier for tr in kept], dtype=float)
            ok = dmin <= thr[nearest]
            for ii in np.flatnonzero(ok):
                labels_k[remaining[ii]] = int(nearest[ii])

    # final refit
    for tid, tr in enumerate(kept):
        members = np.flatnonzero(labels_k == tid)
        if len(members) >= 2:
            c, v, (pA, pB) = _refit_line(X[members])
            rms = float(np.sqrt(np.mean(_line_distances(X[members], c, v)**2)))
            tr.update({"point": c, "direction": v, "endpoints": (pA, pB),
                       "indices": members, "rms_dist": rms})
        else:
            tr.update({"indices": members, "rms_dist": np.nan})

    for i, tr in enumerate(kept, 1):
        tr["name"] = f"Track {i}"

    return kept, labels_k

# ---------- post-filter: distance cut (no regrouping) ----------
def filter_assignments_by_distance(x, y, z, tracks, labels, lam=1.0, update_tracks=True):
    """
    Remove assignments for hits farther than 'lam' (units of x,y,z) from their
    current track center line. Does NOT regroup or refit. If update_tracks is True,
    prunes tracks[i]['indices'] to match filtered labels.
    """
    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    labels_f = np.array(labels, copy=True)

    for tid, t in enumerate(tracks):
        members = np.flatnonzero(labels_f == tid)
        if len(members) == 0:
            continue
        d = _line_distances(X[members], t["point"], t["direction"])
        keep_mask = d <= float(lam)
        to_unassign = members[~keep_mask]
        labels_f[to_unassign] = -1

    if update_tracks:
        for tid in range(len(tracks)):
            tracks[tid]["indices"] = np.flatnonzero(labels_f == tid)

    return labels_f

def filter_tracks_by_min_length(x, y, z, labels, min_length_cm=0.0):
    """
    Unassign (set to -1) any labeled group whose fitted 3D line length
    is below 'min_length_cm'. No regrouping, just pruning.
    If min_length_cm <= 0, this is a no-op.
    """
    min_len = float(min_length_cm)
    if min_len <= 0.0:
        return np.array(labels, copy=True)

    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    labels_f = np.array(labels, copy=True)

    for lab in np.unique(labels_f):
        if lab < 0:
            continue
        idx = np.flatnonzero(labels_f == lab)
        if idx.size < 2:
            labels_f[idx] = -1
            continue
        # refit and measure the extent along the line
        _, _, (pA, pB) = _refit_line(X[idx])
        length_cm = float(np.linalg.norm(pB - pA))
        if length_cm < min_len:
            labels_f[idx] = -1

    return labels_f

# ---------- end-to-end: main entry ----------
def fit_tracks_labels(x, y, z,
                      lam=1.1,                   # cm (lambda cut)
                      rss_threshold=1_500_000.0, # same default as your current code
                      iters=1200,
                      min_inliers=35,
                      k_for_scale=8,
                      attach_multiplier=1.3,
                      seed=0,
                      min_length_cm=0.0          # NEW: minimum 3D track length (cm); 0 = disabled
                      ):
    """
    Full pipeline:
      discover tracks -> choose K by RSS threshold -> keep top-K -> reattach
      -> λ-distance prune (no regrouping)
      -> MIN-LENGTH prune (drop short tracks; no regrouping)
      -> return labels (N,).
    """
    # 1) discover
    tracks_all, labels_all = fit_multiple_tracks(
        x, y, z,
        iters=iters, min_inliers=min_inliers,
        dist_thresh=None, k_for_scale=k_for_scale,
        attach_outliers=True, attach_multiplier=attach_multiplier,
        seed=seed
    )

    # 2) choose K by hard RSS cut
    info = evaluate_RSS_curve(x, y, z, tracks_all, param_count_per_track=5)
    if len(info["K"]) == 0:
        return np.full(len(np.asarray(x)), -1, dtype=int)
    K, _ = decide_k_by_rss_threshold(info, rss_threshold=rss_threshold)

    # 3) keep top-K & reattach
    tracks_k, labels_k = select_top_k_tracks(
        x, y, z, tracks_all, labels_all,
        k=K, reattach=True, attach_multiplier=attach_multiplier
    )

    # 4) λ-distance prune (no regrouping)
    labels_k_f = filter_assignments_by_distance(
        x, y, z, tracks_k, labels_k, lam=float(lam), update_tracks=False
    )

    # 5) min-length prune (no regrouping)  <<< NEW
    labels_len_f = filter_tracks_by_min_length(
        x, y, z, labels_k_f, min_length_cm=float(min_length_cm)
    )

    return labels_len_f


# --- add this anywhere in multi_track_fit.py (e.g., after fit_tracks_labels) ---
import plotly.graph_objects as go
def plot_labeled_tracks_3d(x, y, z, labels=None,
                           draw_lines=True,
                           show_noise=True,
                           marker_size=3,
                           line_width=6,
                           name_prefix="Track"):
    """
    Plot hits in 3D.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates of points (same length).
    labels : array-like or None, default None
        If None, plot a single unlabeled scatter.
        If provided, integers where 0..K-1 are track IDs and -1 means unassigned.
    draw_lines : bool
        When labels are provided, draw a best-fit line per track label (SVD).
    show_noise : bool
        When labels are provided, show label == -1 points as "Unassigned".
    marker_size : int
    line_width : int
    name_prefix : str
        Prefix for labeled track legend entries.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # ---- prepare data ----
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length")

    X = np.column_stack([x, y, z]).astype(np.float64)
    fig = go.Figure()

    # palette for labeled tracks
    palette = (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Bold
    )

    # local, dependency-free refit (centroid + SVD; segment spans min/max projection)
    def _refit_line_local(points):
        c = points.mean(axis=0)
        U, S, Vt = np.linalg.svd(points - c, full_matrices=False)
        v = Vt[0]
        v = v / (np.linalg.norm(v) + 1e-12)
        t = (points - c) @ v
        pA = c + v * t.min()
        pB = c + v * t.max()
        return c, v, (pA, pB)

    # ---- unlabeled mode ----
    if labels is None:
        fig.add_scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers", name=f"Points (N={len(X)})",
            marker=dict(size=marker_size)
        )
        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(itemsizing="constant")
        )
        return fig

    # ---- labeled mode ----
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(X):
        raise ValueError("labels must have the same length as x/y/z")

    uniq = np.unique(labels)
    tracks = sorted([lab for lab in uniq if lab >= 0])

    for lab in tracks:
        idx = np.flatnonzero(labels == lab)
        color = palette[lab % len(palette)]
        name = f"{name_prefix} {lab+1}"

        # point cloud
        fig.add_scatter3d(
            x=X[idx, 0], y=X[idx, 1], z=X[idx, 2],
            mode="markers", name=f"{name} (N={len(idx)})",
            marker=dict(size=marker_size, color=color),
            legendgroup=name
        )

        # optional fitted line for visualization
        if draw_lines and len(idx) >= 2:
            _, _, (pA, pB) = _refit_line_local(X[idx])
            fig.add_scatter3d(
                x=[pA[0], pB[0]], y=[pA[1], pB[1]], z=[pA[2], pB[2]],
                mode="lines", name=f"{name} fit",
                line=dict(width=line_width, color=color),
                legendgroup=name, showlegend=False
            )

    # unassigned hits
    if show_noise:
        noise = np.flatnonzero(labels == -1)
        if len(noise):
            fig.add_scatter3d(
                x=X[noise, 0], y=X[noise, 1], z=X[noise, 2],
                mode="markers", name="Unassigned",
                marker=dict(size=max(1, marker_size-1), symbol="circle-open")
            )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing="constant")
    )
    return fig

