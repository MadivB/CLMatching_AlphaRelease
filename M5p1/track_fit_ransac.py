# track_fit_ransac.py

import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import plotly.express as px

# ---------- helpers ----------

def _line_distances(points, p0, v):
    """Distance of points (N,3) to infinite line through p0 with unit dir v."""
    dif = points - p0
    dot_prod = np.dot(dif, v)
    sq_dist = np.sum(dif**2, axis=1) - dot_prod**2
    return np.sqrt(np.maximum(sq_dist, 0.0))

def _refit_line(points):
    """
    Best-fit infinite line via SVD on centered points.
    Returns (center c, unit direction v, (endA, endB)) where ends span projections.
    """
    c = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
    v = Vt[0]
    v /= (np.linalg.norm(v) + 1e-12)
    t = (points - c) @ v
    pA = c + t.min() * v
    pB = c + t.max() * v
    return c, v, (pA, pB)

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

    idx = rng.integers(0, N, size=(iters, 2))
    diff = points[idx[:, 1]] - points[idx[:, 0]]
    nv = np.linalg.norm(diff, axis=1, keepdims=True)
    valid = (nv[:, 0] > 1e-12)
    p0s = points[idx[valid, 0]]
    vs = diff[valid] / nv[valid]

    points_sq = np.sum(points**2, axis=1)[:, None]
    
    batch_size = 200
    for b in range(0, len(vs), batch_size):
        v_batch = vs[b:b+batch_size]
        p0_batch = p0s[b:b+batch_size]
        
        points_dot_v = points @ v_batch.T
        p0_dot_v = np.sum(p0_batch * v_batch, axis=1)
        dot_prods = points_dot_v - p0_dot_v
        
        points_dot_p0 = points @ p0_batch.T
        p0_sq = np.sum(p0_batch**2, axis=1)[None, :]
        
        sq_dist_batch = points_sq - 2*points_dot_p0 + p0_sq - dot_prods**2
        mask = sq_dist_batch < dist_thresh**2
        
        cnts = np.sum(mask, axis=0)
        max_idx = np.argmax(cnts)
        if cnts[max_idx] > best_cnt:
            best_cnt = cnts[max_idx]
            best_mask = mask[:, max_idx]

    if best_mask is None or best_cnt < min_inliers:
        return None

    in_points = points[best_mask]
    c, v, (pA, pB) = _refit_line(in_points)
    d_all = _line_distances(points, c, v)
    in_ref = d_all < dist_thresh
    rms = float(np.sqrt(np.mean(d_all[in_ref] ** 2))) if in_ref.any() else np.nan

    return {
        "point": c,
        "direction": v,
        "endpoints": (pA, pB),
        "inlier_mask": in_ref,
        "dist_thresh": float(dist_thresh),
        "n_inliers": int(in_ref.sum()),
        "n_total": int(N),
        "rms_dist": rms,
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

        res = ransac_line_3d(
            Xi,
            iters=iters,
            dist_thresh=dist_thresh,
            min_inliers=min_inliers,
            k_for_scale=k_for_scale,
            seed=rng.integers(1 << 31),
        )
        if res is None:
            break

        local_in = res["inlier_mask"]
        global_idx = np.flatnonzero(alive)
        chosen = global_idx[local_in]

        c, v, (pA, pB) = _refit_line(X[chosen])

        tracks.append({
            "name": f"Track {len(tracks) + 1}",
            "point": c,
            "direction": v,
            "endpoints": (pA, pB),
            "indices": chosen,
            "rms_dist": float(np.sqrt(np.mean(_line_distances(X[chosen], c, v) ** 2))) if len(chosen) else np.nan,
            "dist_thresh": float(dist_thresh),
        })

        alive[chosen] = False

    labels = np.full(N, -1, dtype=int)
    for tid, t in enumerate(tracks):
        labels[t["indices"]] = tid

    # Attach outliers to nearest track if close enough
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

        # final refit per track after attaching
        for tid, t in enumerate(tracks):
            members = np.flatnonzero(labels == tid)
            if len(members) >= 2:
                c, v, (pA, pB) = _refit_line(X[members])
                rms = float(np.sqrt(np.mean(_line_distances(X[members], c, v) ** 2)))
                t.update({
                    "point": c,
                    "direction": v,
                    "endpoints": (pA, pB),
                    "indices": members,
                    "rms_dist": rms,
                })

    # sort tracks by size (descending)
    order = np.argsort([-len(t["indices"]) for t in tracks])
    tracks = [tracks[i] for i in order]

    # remap labels to sorted track order
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
    N = len(X)
    T = len(tracks)
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
        RSS = float(np.sum(dmin ** 2))
        sigma2 = max(RSS / max(N, 1), 1e-12)
        BIC = N * np.log(sigma2) + (param_count_per_track * K) * np.log(max(N, 1))

        Ks.append(K)
        RSSs.append(RSS)
        BICs.append(BIC)
        rel.append(np.nan if prev_RSS is None else (prev_RSS - RSS) / max(prev_RSS, 1e-12))
        prev_RSS = RSS

    return {"K": Ks, "RSS": RSSs, "BIC": BICs, "rel_improve": rel}

def decide_k_by_rss_threshold(info, rss_threshold=1_500_000.0):
    """
    Choose the smallest K with RSS <= rss_threshold.
    If none qualifies, pick K with minimum BIC; else argmin RSS.
    """
    Ks = np.asarray(info["K"])
    RSS = np.asarray(info["RSS"], dtype=float)

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
            "name": f"Track {new_tid + 1}",
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
            rms = float(np.sqrt(np.mean(_line_distances(X[members], c, v) ** 2)))
            tr.update({
                "point": c,
                "direction": v,
                "endpoints": (pA, pB),
                "indices": members,
                "rms_dist": rms,
            })
        else:
            tr.update({"indices": members, "rms_dist": np.nan})

    for i, tr in enumerate(kept, 1):
        tr["name"] = f"Track {i}"

    return kept, labels_k

# ---------- post-filters ----------

def filter_assignments_by_distance(x, y, z, tracks, labels, lam=1.0, update_tracks=True):
    """
    Remove assignments for hits farther than 'lam' from their
    current track center line. Does NOT regroup.
    """
    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    labels_f = np.array(labels, copy=True)

    for tid, t in enumerate(tracks):
        members = np.flatnonzero(labels_f == tid)
        if len(members) == 0:
            continue
        d = _line_distances(X[members], t["point"], t["direction"])
        keep_mask = d <= float(lam)
        labels_f[members[~keep_mask]] = -1

    if update_tracks:
        for tid in range(len(tracks)):
            tracks[tid]["indices"] = np.flatnonzero(labels_f == tid)

    return labels_f

def filter_tracks_by_min_length(x, y, z, labels, min_length_cm=0.0):
    """
    Unassign (set -1) any label whose fitted line length < min_length_cm.
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
        _, _, (pA, pB) = _refit_line(X[idx])
        length_cm = float(np.linalg.norm(pB - pA))
        if length_cm < min_len:
            labels_f[idx] = -1

    return labels_f
def build_track_params(x, y, z, labels):
    """
    From final labels, build compact line parameters per surviving track.

    Returns dict with:
      - 'track_ids': (T,) int32
      - 'point':     (T,3) float64   # centroid on line
      - 'direction': (T,3) float64   # unit vector
      - 'endpoints': (T,2,3) float64 # segment spanning min/max projection
      - 'n_hits':    (T,) int32
    """
    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    lab_arr = np.asarray(labels, dtype=int)

    uniq = np.unique(lab_arr)
    track_ids = np.array([l for l in uniq if l >= 0], dtype=int)
    T = len(track_ids)

    if T == 0:
        return {
            "track_ids": np.zeros((0,), dtype=int),
            "point":     np.zeros((0, 3), dtype=float),
            "direction": np.zeros((0, 3), dtype=float),
            "endpoints": np.zeros((0, 2, 3), dtype=float),
            "n_hits":    np.zeros((0,), dtype=int),
        }

    points     = np.empty((T, 3), dtype=float)
    directions = np.empty((T, 3), dtype=float)
    endpoints  = np.empty((T, 2, 3), dtype=float)
    n_hits     = np.empty((T,), dtype=int)

    for k, lab in enumerate(track_ids):
        idx = np.flatnonzero(lab_arr == lab)
        n_hits[k] = idx.size

        if idx.size >= 2:
            pts = X[idx]
            c, v, (pA, pB) = _refit_line(pts)
            points[k] = c
            directions[k] = v
            endpoints[k, 0] = pA
            endpoints[k, 1] = pB
        elif idx.size == 1:
            # Degenerate single-point "track" (should be rare after pruning)
            p = X[idx[0]]
            points[k] = p
            directions[k] = np.array([1.0, 0.0, 0.0])
            endpoints[k, 0] = p
            endpoints[k, 1] = p
        else:
            # Shouldn't happen, but fill safely
            points[k] = 0.0
            directions[k] = np.array([1.0, 0.0, 0.0])
            endpoints[k, :, :] = 0.0
            n_hits[k] = 0

    return {
        "track_ids": track_ids,
        "point":     points,
        "direction": directions,
        "endpoints": endpoints,
        "n_hits":    n_hits,
    }

def filter_tracks_by_gap_density(x, y, z, labels,
                                 gap_cm=2.0,
                                 min_seg_size=25):
    """
    For each track:
      - Project hits onto its fitted line (1D coordinate t).
      - Sort by t; split into segments where gaps > gap_cm.
      - For segments with >= min_seg_size hits, compute density = N / length.
      - Keep ONLY the segment with highest density; all other hits for that track -> -1.
      - If no segment passes min_seg_size, drop the whole track.

    This breaks spurious connections to distant blobs along the track direction.
    """
    X = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)]).astype(np.float64)
    labels_f = np.array(labels, copy=True)
    gap = float(gap_cm)
    min_seg = int(min_seg_size)

    for lab in np.unique(labels):
        if lab < 0:
            continue
        idx = np.flatnonzero(labels == lab)
        if idx.size < 2:
            labels_f[idx] = -1
            continue

        pts = X[idx]
        c, v, _ = _refit_line(pts)
        t = (pts - c) @ v

        order = np.argsort(t)
        t_sorted = t[order]
        idx_sorted = idx[order]

        # find segment boundaries where gap > gap_cm
        boundaries = [0]
        for i in range(len(t_sorted) - 1):
            if (t_sorted[i+1] - t_sorted[i]) > gap:
                boundaries.append(i+1)
        boundaries.append(len(t_sorted))

        if len(boundaries) <= 2:
            # no large gaps -> nothing to split
            continue

        best_seg_idx = None
        best_density = -np.inf

        for s in range(len(boundaries) - 1):
            a, b = boundaries[s], boundaries[s+1]
            if b <= a:
                continue
            seg_idx = idx_sorted[a:b]
            n = b - a
            if n < min_seg:
                continue
            length = max(t_sorted[b-1] - t_sorted[a], 1e-6)
            density = n / length
            if density > best_density:
                best_density = density
                best_seg_idx = seg_idx

        if best_seg_idx is None:
            # no sufficiently populated segment -> drop entire track
            labels_f[idx] = -1
            continue

        # keep only best segment, drop all other hits of this track
        keep_mask = np.zeros_like(labels_f, dtype=bool)
        keep_mask[best_seg_idx] = True
        drop_mask = (labels_f == lab) & (~keep_mask)
        labels_f[drop_mask] = -1

    return labels_f

# ---------- end-to-end ----------

def fit_tracks_labels(x, y, z,
                      lam=1.0,                   # distance cut to clean assignments
                      rss_threshold=1_500_000.0,
                      iters=1200,
                      min_inliers=35,
                      k_for_scale=8,
                      attach_multiplier=1.3,
                      seed=0,
                      min_length_cm=0.0):
    """
    Full pipeline:
      1) RANSAC multi-track discovery.
      2) Choose K via RSS threshold.
      3) Keep top-K & reattach.
      4) λ-distance prune.
      5) Min-length prune.
      6) Gap-density prune: remove low-density segments separated by ~4 cm.

    Returns
    -------
    labels : (N,) int
        Final assignment, -1 = noise / discarded.
    track_params : dict of arrays
        Compact parameters of best-fit line for each surviving track ID:
          - track_ids, point, direction, endpoints, n_hits
    """
    # 1) discover
    tracks_all, labels_all = fit_multiple_tracks(
        x, y, z,
        iters=iters,
        min_inliers=min_inliers,
        dist_thresh=None,
        k_for_scale=k_for_scale,
        attach_outliers=True,
        attach_multiplier=attach_multiplier,
        seed=seed,
    )

    # 2) select K
    info = evaluate_RSS_curve(x, y, z, tracks_all, param_count_per_track=5)
    if len(info["K"]) == 0:
        labels_empty = np.full(len(np.asarray(x)), -1, dtype=int)
        params_empty = build_track_params(x, y, z, labels_empty)
        return labels_empty, params_empty

    K, _ = decide_k_by_rss_threshold(info, rss_threshold=rss_threshold)

    # 3) keep top-K & reattach
    tracks_k, labels_k = select_top_k_tracks(
        x, y, z,
        tracks_all, labels_all,
        k=K,
        reattach=True,
        attach_multiplier=attach_multiplier,
    )

    # 4) λ-distance prune
    labels_lam = filter_assignments_by_distance(
        x, y, z,
        tracks_k, labels_k,
        lam=float(lam),
        update_tracks=False,
    )

    # 5) min-length prune
    labels_len = filter_tracks_by_min_length(
        x, y, z,
        labels_lam,
        min_length_cm=float(min_length_cm),
    )

    # 6) gap-density prune (kills track+far-blob connections)
    labels_final = filter_tracks_by_gap_density(
        x, y, z,
        labels_len,
        gap_cm=4.0,
        min_seg_size=25,
    )

    # 7) build line parameters for surviving tracks
    track_params = build_track_params(x, y, z, labels_final)

    return labels_final, track_params


# ---------- visualization ----------

def plot_labeled_tracks_3d(x, y, z, labels=None,
                           draw_lines=True,
                           show_noise=True,
                           marker_size=3,
                           line_width=6,
                           name_prefix="Track"):
    """
    3D scatter + optional fitted lines for labeled tracks.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length")
    X = np.column_stack([x, y, z]).astype(np.float64)

    fig = go.Figure()

    palette = (
        px.colors.qualitative.Alphabet
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Bold
    )

    def _refit_line_local(points):
        c = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
        v = Vt[0]
        v /= (np.linalg.norm(v) + 1e-12)
        t = (points - c) @ v
        pA = c + t.min() * v
        pB = c + t.max() * v
        return c, v, (pA, pB)

    if labels is None:
        fig.add_scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            name=f"Points (N={len(X)})",
            marker=dict(size=marker_size),
        )
        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(itemsizing="constant"),
        )
        return fig

    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(X):
        raise ValueError("labels must have same length as x,y,z")

    uniq = np.unique(labels)
    tracks = sorted([lab for lab in uniq if lab >= 0])

    for lab in tracks:
        idx = np.flatnonzero(labels == lab)
        color = palette[lab % len(palette)]
        name = f"{name_prefix} {lab+1}"

        fig.add_scatter3d(
            x=X[idx, 0], y=X[idx, 1], z=X[idx, 2],
            mode="markers",
            name=f"{name} (N={len(idx)})",
            marker=dict(size=marker_size, color=color),
            legendgroup=name,
        )

        if draw_lines and len(idx) >= 2:
            _, _, (pA, pB) = _refit_line_local(X[idx])
            fig.add_scatter3d(
                x=[pA[0], pB[0]],
                y=[pA[1], pB[1]],
                z=[pA[2], pB[2]],
                mode="lines",
                name=f"{name} fit",
                line=dict(width=line_width, color=color),
                legendgroup=name,
                showlegend=False,
            )

    if show_noise:
        noise = np.flatnonzero(labels == -1)
        if len(noise):
            fig.add_scatter3d(
                x=X[noise, 0],
                y=X[noise, 1],
                z=X[noise, 2],
                mode="markers",
                name="Unassigned",
                marker=dict(size=max(1, marker_size - 1), symbol="circle-open"),
            )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing="constant"),
    )
    return fig
