# global_track_clustering.py

import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px

from track_fit_ransac import fit_tracks_labels  # per-TPC tracking


# ---------- basic helpers ----------

def _line_line_dca(p1, v1, p2, v2):
    """
    Distance of closest approach between two infinite 3D lines:
      L1(s) = p1 + s v1
      L2(t) = p2 + t v2
    """
    p1 = np.asarray(p1, float)
    v1 = np.asarray(v1, float)
    p2 = np.asarray(p2, float)
    v2 = np.asarray(v2, float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float(np.linalg.norm(p1 - p2))

    v1 = v1 / n1
    v2 = v2 / n2

    w0 = p1 - p2
    a = np.dot(v1, v1)
    b = np.dot(v1, v2)
    c = np.dot(v2, v2)
    d = np.dot(v1, w0)
    e = np.dot(v2, w0)

    denom = a * c - b * b
    if denom < 1e-12:
        # Nearly parallel: perpendicular distance from w0 to v1
        return float(np.linalg.norm(np.cross(w0, v1)))

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    c1 = p1 + s * v1
    c2 = p2 + t * v2

    return float(np.linalg.norm(c1 - c2))


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


def _angle_diff_deg(u, v):
    """
    Absolute angle between two directions in degrees.
    Treats v and -v as equivalent (uses |dot|).
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 180.0
    u /= nu
    v /= nv
    cosang = abs(float(np.dot(u, v)))
    cosang = max(min(cosang, 1.0), -1.0)
    return float(np.degrees(np.arccos(cosang)))


def _segment_segment_distance(p1, q1, p2, q2):
    """
    Minimum distance between two 3D line segments [p1,q1] and [p2,q2].
    """
    p1 = np.asarray(p1, float)
    q1 = np.asarray(q1, float)
    p2 = np.asarray(p2, float)
    q2 = np.asarray(q2, float)

    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b
    if denom < 1e-12:
        # nearly parallel: approximate via distances to endpoints
        dists = [
            np.linalg.norm(np.cross((p1 - p2), v)) / (np.linalg.norm(v) + 1e-12),
            np.linalg.norm(np.cross((q1 - p2), v)) / (np.linalg.norm(v) + 1e-12),
            np.linalg.norm(np.cross((p2 - p1), u)) / (np.linalg.norm(u) + 1e-12),
            np.linalg.norm(np.cross((q2 - p1), u)) / (np.linalg.norm(u) + 1e-12),
        ]
        return float(min(dists))

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    # clamp to [0,1] segment range
    s = min(max(s, 0.0), 1.0)
    t = min(max(t, 0.0), 1.0)

    c1 = p1 + s * u
    c2 = p2 + t * v
    return float(np.linalg.norm(c1 - c2))


def _extend_segment_to_active_box(p1, p2, bounds, eps=1e-9):
    """
    Extend the segment [p1, p2] along its direction until it hits
    the global active volume bounding box.

    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    We use the infinite line through p1,p2 and clip it to the box.
    Result: a finite segment inside the box, typically a superset
    of [p1, p2], used ONLY in cross-TPC matching.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    v = p2 - p1
    nv = np.linalg.norm(v)
    if nv < eps:
        return p1, p2  # degenerate

    v = v / nv
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Line: p(t) = p1 + t v
    tmin, tmax = -np.inf, np.inf

    def clip_axis(p_comp, v_comp, lo, hi):
        nonlocal tmin, tmax
        if abs(v_comp) < eps:
            # parallel to axis; if outside slab, no intersection
            if p_comp < lo or p_comp > hi:
                tmin, tmax = 1.0, 0.0
            return
        t1 = (lo - p_comp) / v_comp
        t2 = (hi - p_comp) / v_comp
        t_lo = min(t1, t2)
        t_hi = max(t1, t2)
        tmin = max(tmin, t_lo)
        tmax = min(tmax, t_hi)

    clip_axis(p1[0], v[0], x_min, x_max)
    clip_axis(p1[1], v[1], y_min, y_max)
    clip_axis(p1[2], v[2], z_min, z_max)

    if tmax < tmin:
        # no intersection with box: fall back
        return p1, p2

    e1 = p1 + tmin * v
    e2 = p1 + tmax * v
    return e1, e2


# ---------- palette ----------

_PALETTE = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Light24
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Bold
)


# ---------- segment building (per TPC) ----------

def _build_tpc_segments(x, y, z, io_group,
                        lam=1.5,
                        rss_threshold=1.5e6,
                        iters=800,
                        min_inliers=35,
                        k_for_scale=8,
                        attach_multiplier=1.3,
                        seed=0,
                        min_length_cm=30.0,
                        n_tpcs=70):
    """
    Run fit_tracks_labels TPC-by-TPC and return a list of segment dicts:
      {
        'tpc': int,
        'local_id': int,          # per-TPC track label
        'hits': np.ndarray[int],  # global hit indices
        'point': (3,),
        'direction': (3,),
        'endpoints': (2,3),
        'n_hits': int
      }
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    io_group = np.asarray(io_group, int)
    N = len(x)
    if not (len(y) == N and len(z) == N and len(io_group) == N):
        raise ValueError("x, y, z, io_group must have the same length")

    tpc_ids = _tpc_id_from_io(io_group)
    segments = []

    for tpc in range(n_tpcs):
        mask = (tpc_ids == tpc)
        if not np.any(mask):
            continue

        idx_global = np.flatnonzero(mask)
        x_tpc = x[mask]
        y_tpc = y[mask]
        z_tpc = z[mask]

        labels_tpc, params = fit_tracks_labels(
            x_tpc, y_tpc, z_tpc,
            lam=lam,
            rss_threshold=rss_threshold,
            iters=iters,
            min_inliers=min_inliers,
            k_for_scale=k_for_scale,
            attach_multiplier=attach_multiplier,
            seed=seed,
            min_length_cm=min_length_cm,
        )

        track_ids = params["track_ids"]
        points = params["point"]
        directions = params["direction"]
        endpoints = params["endpoints"]
        n_hits_arr = params["n_hits"]

        for k, tid in enumerate(track_ids):
            loc = np.flatnonzero(labels_tpc == tid)
            if loc.size == 0:
                continue

            seg_hits = idx_global[loc]

            segments.append({
                "tpc": int(tpc),
                "local_id": int(tid),
                "hits": seg_hits,
                "point": np.asarray(points[k], float),
                "direction": np.asarray(directions[k], float),
                "endpoints": np.asarray(endpoints[k], float),
                "n_hits": int(n_hits_arr[k]),
            })

    return segments


# ---------- match segments across TPCs -> global tracks ----------

def _match_segments_across_tpcs(segments,
                                x, y, z,
                                dist_tol=4.0,
                                angle_tol_deg=10.0):
    """
    Group TPC-local segments into global tracks based on:

      - segments from different TPCs only
      - angular difference < angle_tol_deg
      - DOCA between extended finite segments < dist_tol

    Extended segments:
      For matching only, each per-TPC segment [p1,p2] is extended along its
      fitted direction until it hits the global (x,y,z) bounding box.
      This approximates crossing gaps without using truly infinite lines.

    NOTE:
      Global track 'endpoints' stored below are still based on actual hits,
      NOT the extended-box endpoints; shower / vertex clustering uses those.
    """
    if not segments:
        return []

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    # Global active volume bounds from ALL hits
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    z_min, z_max = float(z.min()), float(z.max())
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    M = len(segments)
    parent = np.arange(M, dtype=int)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Precompute extended endpoints for matching
    ext_endpoints = []
    for seg in segments:
        p1, p2 = seg["endpoints"]
        e1, e2 = _extend_segment_to_active_box(p1, p2, bounds)
        ext_endpoints.append((e1, e2))

    # pairwise matching using extended segments
    for i in range(M):
        si = segments[i]
        ti = si["tpc"]
        vi = si["direction"]
        e1i, e2i = ext_endpoints[i]

        for j in range(i + 1, M):
            sj = segments[j]
            tj = sj["tpc"]
            if ti == tj:
                continue  # no merging within the same TPC here

            vj = sj["direction"]
            e1j, e2j = ext_endpoints[j]

            # 1) angle consistency (treat ±v the same)
            ang = _angle_diff_deg(vi, vj)
            if ang > angle_tol_deg:
                continue

            # 2) DOCA between extended finite segments
            d_seg = _segment_segment_distance(e1i, e2i, e1j, e2j)
            if d_seg > dist_tol:
                continue

            union(i, j)

    # collect DSU groups into global tracks
    groups = {}
    for i in range(M):
        r = find(i)
        groups.setdefault(r, []).append(i)

    global_tracks = []
    for members in groups.values():
        all_hits = np.unique(
            np.concatenate([segments[m]["hits"] for m in members])
        )
        if all_hits.size == 0:
            continue

        pts = np.column_stack([x[all_hits], y[all_hits], z[all_hits]])
        c = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        v = Vt[0]
        v /= (np.linalg.norm(v) + 1e-12)
        t = (pts - c) @ v
        pA = c + t.min() * v
        pB = c + t.max() * v

        global_tracks.append({
            "segments": [segments[m] for m in members],
            "hit_indices": all_hits,
            "point": c,
            "direction": v,
            "endpoints": (pA, pB),  # from actual hits
        })

    return global_tracks


# ---------- DBSCAN endpoints -> vertex clusters ----------

def _cluster_track_endpoints(global_tracks,
                             eps=10.0,
                             min_samples=3,
                             dca_max=4.0):
    """
    Run DBSCAN on ALL start/endpoints of global tracks.

    A DBSCAN endpoint cluster is kept as a 'vertex cluster' only if:
      - it is touched by >= 2 distinct global tracks, and
      - the median pairwise infinite-line DOCA between those tracks
        is <= dca_max (cm).

    Uses the stored global_tracks endpoints/lines (no segment extension here).
    """
    endpoint_xyz = []
    endpoint_meta = []  # (gid, end_id) where end_id in {0,1}

    for gid, gt in enumerate(global_tracks):
        pA, pB = gt["endpoints"]
        endpoint_xyz.append(pA)
        endpoint_meta.append((gid, 0))
        endpoint_xyz.append(pB)
        endpoint_meta.append((gid, 1))

    if not endpoint_xyz:
        return {}, {}, np.zeros((0, 3)), [], np.zeros((0,), dtype=int)

    endpoint_xyz = np.asarray(endpoint_xyz, float)

    db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(endpoint_xyz)
    ep_labels = db.labels_

    # raw clusters: cid -> list of endpoint indices
    raw_clusters = {}
    for idx, cid in enumerate(ep_labels):
        if cid >= 0:
            raw_clusters.setdefault(int(cid), []).append(idx)

    clusters = {}
    cluster_tracks = {}

    for cid, idcs in raw_clusters.items():
        gids = sorted({endpoint_meta[i][0] for i in idcs})
        if len(gids) < 2:
            continue

        dcas = []
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                gi = global_tracks[gids[i]]
                gj = global_tracks[gids[j]]
                d = _line_line_dca(
                    gi["point"], gi["direction"],
                    gj["point"], gj["direction"]
                )
                dcas.append(d)

        if not dcas:
            continue

        med_dca = float(np.median(dcas))
        if med_dca > float(dca_max):
            # lines don't actually converge -> likely fake
            continue

        clusters[cid] = idcs
        cluster_tracks[cid] = set(gids)

    return clusters, cluster_tracks, endpoint_xyz, endpoint_meta, ep_labels


# ---------- assign vertex IDs ----------

def _assign_vertex_ids(global_tracks, cluster_tracks, min_tracks_for_shower=3):
    """
    Map each global track -> vertex-cluster id.
    Tracks sharing an accepted endpoint cluster -> same vertex id.
    Others get unique IDs.
    """
    n_gt = len(global_tracks)
    if n_gt == 0:
        return {}, {}

    track_to_vertex = {}
    vertex_meta = {}
    next_vid = 0

    # group tracks connected via endpoint clusters
    for cid in sorted(cluster_tracks.keys()):
        gids = sorted(cluster_tracks[cid])
        if not gids:
            continue
        vid = next_vid
        next_vid += 1
        vertex_meta[vid] = {
            "cluster_id": int(cid),
            "track_ids": gids,
            "n_tracks": int(len(gids)),
            "type": "shower" if len(gids) >= int(min_tracks_for_shower) else "vertex",
        }
        for gid in gids:
            if gid not in track_to_vertex:
                track_to_vertex[gid] = vid

    # standalone tracks (not in any cluster) each get their own id
    for gid in range(n_gt):
        if gid not in track_to_vertex:
            track_to_vertex[gid] = next_vid
            vertex_meta[next_vid] = {
                "cluster_id": None,
                "track_ids": [int(gid)],
                "n_tracks": 1,
                "type": "track",
            }
            next_vid += 1

    return track_to_vertex, vertex_meta


# ---------- plotting helpers ----------

def _plot_matched_tracks_only(global_tracks, x, y, z, filename):
    """
    Plot only matched global tracks (those spanning >= 2 TPCs).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    fig = go.Figure()
    any_plotted = False

    for gid, gt in enumerate(global_tracks):
        tpcs = {s["tpc"] for s in gt["segments"]}
        if len(tpcs) <= 1:
            continue

        color = _PALETTE[gid % len(_PALETTE)]
        idx = gt["hit_indices"]

        fig.add_scatter3d(
            x=x[idx],
            y=y[idx],
            z=z[idx],
            mode="markers",
            marker=dict(size=2, color=color),
            name=f"Matched {gid}",
            legendgroup="matched",
            showlegend=True,
        )

        pA, pB = gt["endpoints"]
        fig.add_scatter3d(
            x=[pA[0], pB[0]],
            y=[pA[1], pB[1]],
            z=[pA[2], pB[2]],
            mode="lines",
            line=dict(width=4, color=color),
            name=f"Matched {gid} fit",
            legendgroup="matched",
            showlegend=False,
        )

        any_plotted = True

    if not any_plotted:
        return

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing="constant"),
        title="Matched Tracks Only (Pre-Vertex / DBSCAN)"
    )
    fig.write_html(filename)


def _plot_dbscan_result(global_tracks,
                        clusters,
                        cluster_tracks,
                        endpoint_xyz,
                        ep_labels,
                        x, y, z,
                        filename):
    """
    All hits grey; clustered endpoints + their tracks colored.
    """
    if endpoint_xyz.size == 0:
        return

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    fig = go.Figure()

    # all hits grey
    fig.add_scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=1.5, color="lightgray"),
        name="All hits",
        legendgroup="base",
        showlegend=True,
    )

    # overlay vertex clusters
    for cid, idcs in clusters.items():
        color = _PALETTE[cid % len(_PALETTE)]
        gids = cluster_tracks.get(cid, set())
        if not gids:
            continue

        pts_ep = endpoint_xyz[idcs]
        fig.add_scatter3d(
            x=pts_ep[:, 0],
            y=pts_ep[:, 1],
            z=pts_ep[:, 2],
            mode="markers",
            marker=dict(size=4, color=color, symbol="diamond"),
            name=f"V{cid} endpoints",
            legendgroup=f"V{cid}",
            showlegend=True,
        )

        hits = np.unique(
            np.concatenate([global_tracks[gid]["hit_indices"] for gid in gids])
        )
        fig.add_scatter3d(
            x=x[hits],
            y=y[hits],
            z=z[hits],
            mode="markers",
            marker=dict(size=2, color=color),
            name=f"V{cid} tracks",
            legendgroup=f"V{cid}",
            showlegend=False,
        )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing="constant"),
        title="Endpoint DBSCAN Result"
    )
    fig.write_html(filename)


def _plot_shower_clusters(global_tracks,
                          cluster_tracks,
                          x, y, z,
                          filename,
                          min_tracks_for_shower=3):
    """
    Only shower-like vertex clusters:
      clusters touched by >= min_tracks_for_shower tracks.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    shower_cids = [
        cid for cid, gids in cluster_tracks.items()
        if len(gids) >= min_tracks_for_shower
    ]
    if not shower_cids:
        return

    fig = go.Figure()

    for idx_c, cid in enumerate(sorted(shower_cids)):
        gids = sorted(cluster_tracks[cid])
        if not gids:
            continue
        color = _PALETTE[idx_c % len(_PALETTE)]
        hits = np.unique(
            np.concatenate([global_tracks[gid]["hit_indices"] for gid in gids])
        )

        fig.add_scatter3d(
            x=x[hits],
            y=y[hits],
            z=z[hits],
            mode="markers",
            marker=dict(size=2.5, color=color),
            name=f"Shower V{cid} (tracks={len(gids)})",
            legendgroup=f"V{cid}",
            showlegend=True,
        )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing="constant"),
        title=f"Shower-like Vertex Clusters (≥{min_tracks_for_shower} tracks)"
    )
    fig.write_html(filename)


def _plot_shower_endpoints(global_tracks,
                           clusters,
                           cluster_tracks,
                           endpoint_xyz,
                           ep_labels,
                           filename,
                           min_tracks_for_shower=3):
    """
    Endpoint-only view:
      - all endpoints in grey,
      - endpoints in shower-like clusters colored.
    """
    if endpoint_xyz.size == 0:
        return

    shower_cids = [
        cid for cid, gids in cluster_tracks.items()
        if len(gids) >= min_tracks_for_shower
    ]

    fig = go.Figure()

    fig.add_scatter3d(
        x=endpoint_xyz[:, 0],
        y=endpoint_xyz[:, 1],
        z=endpoint_xyz[:, 2],
        mode="markers",
        marker=dict(size=3, color="lightgray"),
        name="All endpoints",
        legendgroup="all",
        showlegend=True,
    )

    for idx_c, cid in enumerate(sorted(shower_cids)):
        mask = (ep_labels == cid)
        pts = endpoint_xyz[mask]
        if pts.size == 0:
            continue
        color = _PALETTE[idx_c % len(_PALETTE)]
        fig.add_scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=5, color=color, symbol="diamond"),
            name=f"Shower V{cid}",
            legendgroup=f"V{cid}",
            showlegend=True,
        )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing="constant"),
        title=f"Shower Endpoint Clusters (colored), Others Grey (≥{min_tracks_for_shower} tracks)"
    )
    fig.write_html(filename)


# ---------- label-info helper ----------

def build_global_labels(x, y, z, io_group,
                        # per-TPC tracking params
                        lam=1.5,
                        rss_threshold=1.5e6,
                        iters=800,
                        min_inliers=35,
                        k_for_scale=8,
                        attach_multiplier=1.3,
                        seed=0,
                        min_length_cm=30.0,
                        n_tpcs=70,
                        # segment matching across TPCs
                        match_dist_tol=4.0,
                        match_angle_deg=10.0,
                        # vertex (endpoint) clustering
                        vertex_eps=10.0,
                        vertex_min_samples=3,
                        # shower threshold
                        min_tracks_for_shower=3,
                        # plotting
                        plotting=False,
                        out_prefix="event",
                        return_label_info=False):
    """
    Full pipeline:

      1) Per-TPC RANSAC tracks via fit_tracks_labels.
      2) Cross-TPC matching:
           - extend segments to active box
           - match if angle < match_angle_deg & DOCA(ext segs) < match_dist_tol
      3) DBSCAN on global-track endpoints to find interaction vertices/showers.
      4) Map each global track to a vertex ID (cluster of tracks).
      5) Return hit-level labels (vertex IDs), -1 for noise/unassigned.

    Outputs
    -------
    labels_global : (N,) int
        Hit-level cluster ID; -1 = noise/unassigned.
    split_index : int
        The integer index that separates tracks/showers from small clusters.
        Labels in range `[0, split_index)` are tracks or showers.
        Labels in range `[split_index, max_label+1)` are small (leftover) clusters.
    label_info : dict, optional
        Returned only when `return_label_info=True`.
        Metadata keyed by final label id, including `type`, `tpcs`, `n_hits`,
        and `n_tracks` for backbone labels.

    When plotting=True, also writes:
      <prefix>_matched_tracks.html
      <prefix>_dbscan.html
      <prefix>_showers.html
      <prefix>_shower_endpoints.html
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    io_group = np.asarray(io_group, int)

    N = len(x)
    if not (len(y) == N and len(z) == N and len(io_group) == N):
        raise ValueError("x, y, z, io_group must have the same length")

    # 1) per-TPC segments
    segments = _build_tpc_segments(
        x, y, z, io_group,
        lam=lam,
        rss_threshold=rss_threshold,
        iters=iters,
        min_inliers=min_inliers,
        k_for_scale=k_for_scale,
        attach_multiplier=attach_multiplier,
        seed=seed,
        min_length_cm=min_length_cm,
        n_tpcs=n_tpcs,
    )

    if not segments:
        if return_label_info:
            return np.full(N, -1, dtype=int), 0, {}
        return np.full(N, -1, dtype=int), 0

    # 2) cross-TPC matching -> global tracks
    global_tracks = _match_segments_across_tpcs(
        segments,
        x, y, z,
        dist_tol=match_dist_tol,
        angle_tol_deg=match_angle_deg,
    )

    if plotting:
        _plot_matched_tracks_only(
            global_tracks,
            x, y, z,
            filename=f"{out_prefix}_matched_tracks.html",
        )

    if not global_tracks:
        # fallback: treat each segment as its own cluster
        labels = np.full(N, -1, dtype=int)
        label_info = {}
        for gid, seg in enumerate(segments):
            labels[seg["hits"]] = gid
            label_info[int(gid)] = {
                "type": "track",
                "tpcs": [int(seg["tpc"])],
                "n_hits": int(len(seg["hits"])),
                "n_tracks": 1,
                "track_ids": [int(gid)],
                "backbone": True,
            }
        if return_label_info:
            return labels, len(segments), label_info
        return labels, len(segments)

    # 3) DBSCAN endpoints -> vertex clusters (with DCA sanity)
    clusters, cluster_tracks, endpoint_xyz, endpoint_meta, ep_labels = _cluster_track_endpoints(
        global_tracks,
        eps=vertex_eps,
        min_samples=vertex_min_samples,
    )

    if plotting:
        _plot_dbscan_result(
            global_tracks,
            clusters,
            cluster_tracks,
            endpoint_xyz,
            ep_labels,
            x, y, z,
            filename=f"{out_prefix}_dbscan.html",
        )
        _plot_shower_clusters(
            global_tracks,
            cluster_tracks,
            x, y, z,
            filename=f"{out_prefix}_showers.html",
        )
        _plot_shower_endpoints(
            global_tracks,
            clusters,
            cluster_tracks,
            endpoint_xyz,
            ep_labels,
            filename=f"{out_prefix}_shower_endpoints.html",
            min_tracks_for_shower=3,
        )

    # 4) assign vertex IDs to global tracks
    track_to_vertex, vertex_meta = _assign_vertex_ids(
        global_tracks,
        cluster_tracks,
        min_tracks_for_shower=min_tracks_for_shower,
    )

    # 5) propagate to hit-level labels with sequential remapping
    labels_global = np.full(N, -1, dtype=int)
    
    unique_vids = sorted(set(vid for vid in track_to_vertex.values()))
    vid_map = {old: new for new, old in enumerate(unique_vids)}
    split_index = len(unique_vids)
    label_info = {}

    for gid, gt in enumerate(global_tracks):
        vid = track_to_vertex.get(gid, None)
        if vid is None:
            continue
        new_vid = vid_map[vid]
        labels_global[gt["hit_indices"]] = new_vid

    for old_vid in unique_vids:
        meta = vertex_meta.get(old_vid, {})
        track_ids = [int(gid) for gid in meta.get("track_ids", [])]
        if track_ids:
            hit_indices = np.unique(
                np.concatenate([global_tracks[gid]["hit_indices"] for gid in track_ids])
            )
            tpcs = sorted(
                {
                    int(seg["tpc"])
                    for gid in track_ids
                    for seg in global_tracks[gid]["segments"]
                }
            )
        else:
            hit_indices = np.zeros((0,), dtype=int)
            tpcs = []
        label_info[int(vid_map[old_vid])] = {
            "type": str(meta.get("type", "track")),
            "tpcs": tpcs,
            "n_hits": int(hit_indices.size),
            "n_tracks": int(meta.get("n_tracks", 1)),
            "track_ids": track_ids,
            "cluster_id": meta.get("cluster_id"),
            "backbone": True,
        }

    # 6) cluster leftover hits into small clusters using DBSCAN
    leftover_mask = (labels_global == -1)
    if np.any(leftover_mask):
        leftover_indices = np.where(leftover_mask)[0]
        pts = np.column_stack([x[leftover_mask], y[leftover_mask], z[leftover_mask]])
        db = DBSCAN(eps=4.0, min_samples=3).fit(pts)
        
        max_vid = split_index - 1
        for cid in np.unique(db.labels_):
            if cid == -1:
                continue
            max_vid += 1
            idx_in_leftover = np.where(db.labels_ == cid)[0]
            global_hit_idx = leftover_indices[idx_in_leftover]
            labels_global[global_hit_idx] = max_vid
            label_info[int(max_vid)] = {
                "type": "cluster",
                "tpcs": sorted({int(tpc) for tpc in _tpc_id_from_io(io_group[global_hit_idx])}),
                "n_hits": int(len(global_hit_idx)),
                "n_tracks": 0,
                "track_ids": [],
                "cluster_id": None,
                "backbone": False,
            }
    if return_label_info:
        return labels_global, split_index, label_info
    return labels_global, split_index
