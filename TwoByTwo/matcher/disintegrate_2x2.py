#!/usr/bin/env python3
"""STEP 6 - disintegrate a matched FAMILY into span-bounded PIECES for
between-piece velocity fitting, WITHOUT re-running the GPU light model.

Rules
-----
* Atomic unit = an existing matcher cluster; a piece is a UNION of whole
  clusters, so its predicted light = sum of the clusters' images.
* Connectivity (single-linkage on min inter-hit gap) splits the family
  into groups -> isolates DISPLACED deposits (neutron candidates).
* main family = highest-energy connected group.
* Inside a group, whole clusters are agglomerated until the span would
  exceed D_MAX (span cap).
* A single cluster longer than D_MAX is SPLIT into span-bounded pieces
  (PCA-axis slices if track-like, spatial k-means if blobby); its light
  is APPORTIONED by charge fraction (the 'deduced' amplitude).
Each piece: hits, member cluster ids, centroid, E, span, tag, light-plan.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

THIN_CM = 3.0    # transverse RMS below this + elongated => a real (thin) track

def _pca(P, w):
    c = np.average(P, axis=0, weights=w)
    Xc = P - c
    cov = (Xc * w[:, None]).T @ Xc / w.sum()
    val, vec = np.linalg.eigh(cov)
    axis = vec[:, -1]; s = Xc @ axis
    span = float(s.max() - s.min())
    aspect = float(np.sqrt(val[-1] / max(val[0] + val[1], 1e-6)))
    thick = float(np.sqrt(max(val[0], 0) + max(val[1], 0)))   # transverse RMS
    return c, axis, s, span, aspect, thick

def is_track(aspect, thick):
    return aspect >= 3.0 and thick < THIN_CM

def cluster_table(labels, xyz, E, tpc):
    """One row per matcher cluster."""
    cl = {}
    for l in np.unique(labels[labels >= 0]):
        m = np.flatnonzero(labels == l)
        P = xyz[m]; w = np.maximum(E[m], 1e-6)
        c, ax, s, span, aspect, thick = _pca(P, w)
        cl[int(l)] = dict(hits=m, P=P, E=float(w.sum()), centroid=c,
                          axis=ax, span=span, aspect=aspect, thick=thick,
                          tpc=int(np.bincount(tpc[m]).argmax()))
    return cl

def _adjacency(cl, link):
    """LOCAL cluster adjacency via one KD-tree neighbor query over all
    family hits: O(N log N); a cluster only ever sees hits within
    `link`. Returns {cid: set(neighbor cids)}."""
    ids = list(cl)
    P = np.vstack([cl[i]["P"] for i in ids])
    owner = np.concatenate([np.full(len(cl[i]["P"]), k) for k, i in enumerate(ids)])
    tree = cKDTree(P)
    pairs = tree.query_pairs(link, output_type="ndarray")
    adj = {i: set() for i in ids}
    if pairs.size:
        oa, ob = owner[pairs[:, 0]], owner[pairs[:, 1]]
        diff = oa != ob
        for a, b in zip(oa[diff], ob[diff]):
            adj[ids[a]].add(ids[b]); adj[ids[b]].add(ids[a])
    return adj

def _connected(cl, link):
    """Connected components from the local adjacency (union-find)."""
    adj = _adjacency(cl, link)
    parent = {i: i for i in cl}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, nb in adj.items():
        for j in nb:
            parent[find(i)] = find(j)
    groups = {}
    for i in cl: groups.setdefault(find(i), []).append(i)
    return list(groups.values()), adj

def _split_big_cluster(info, D_MAX, xyz, E, labels, lid):
    """Split ONE over-long cluster into span-bounded sub-pieces; light
    apportioned by charge fraction. Track-like -> PCA slices; else kmeans."""
    m = info["hits"]; P = info["P"]; w = np.maximum(E[m], 1e-6)
    nsub = int(np.ceil(info["span"] / D_MAX))
    track = is_track(info["aspect"], info["thick"])
    kind = "TRACK-SEG" if track else "SHOWER-SEG"
    if track:                                       # thin track: slice along axis
        s = (P - info["centroid"]) @ info["axis"]
        edges = np.linspace(s.min(), s.max(), nsub + 1)
        assign = np.clip(np.searchsorted(edges, s) - 1, 0, nsub - 1)
    else:                                           # fat shower/blob: spatial kmeans
        rng = np.random.default_rng(lid)
        ctr = P[rng.choice(len(P), nsub, replace=False)]
        for _ in range(12):
            assign = np.argmin(cdist(P, ctr), axis=1)
            for k in range(nsub):
                if (assign == k).any():
                    ctr[k] = np.average(P[assign == k], axis=0,
                                        weights=w[assign == k])
    pieces = []
    Etot = w.sum()
    for k in range(nsub):
        sub = assign == k
        if sub.sum() < 3: continue
        frac = float(w[sub].sum() / Etot)
        cc, _, _, span, _, _ = _pca(P[sub], w[sub])
        pieces.append(dict(hits=m[sub], clusters=[lid], centroid=cc,
            E=float(w[sub].sum()), span=span, tag=kind,
            light={lid: frac}))     # image[lid] * frac  (GPU re-predict for accuracy)
    return pieces

def disintegrate(labels, xyz, E, tpc, D_MAX=25.0, LINK=10.0,
                 DISP=30.0, MAIN_FRAC=0.5, E_KEEP_FRAC=0.03, E_ABS_MIN=8.0):
    cl = cluster_table(labels, xyz, E, tpc)
    if not cl: return [], None
    # drop tiny background clusters (keep significant deposits only);
    # always keep the single most energetic (the vertex/main).
    Etot = sum(v["E"] for v in cl.values())
    floor = max(E_KEEP_FRAC * Etot, E_ABS_MIN)
    keep_ids = {i for i, v in cl.items() if v["E"] >= floor}
    keep_ids |= {max(cl, key=lambda i: cl[i]["E"])}
    cl = {i: cl[i] for i in keep_ids}
    if not cl: return [], None
    for v in cl.values():                       # bbox for O(1) span tests
        v["bmin"] = v["P"].min(0); v["bmax"] = v["P"].max(0)
    groups, adj = _connected(cl, LINK)
    gE = [sum(cl[i]["E"] for i in g) for g in groups]
    main_gi = int(np.argmax(gE))
    main_ctr = np.average(np.vstack([cl[i]["centroid"] for i in groups[main_gi]]),
        axis=0, weights=[cl[i]["E"] for i in groups[main_gi]])
    pieces = []
    for gi, g in enumerate(groups):
        is_main = (gi == main_gi) or (gE[gi] >= MAIN_FRAC * gE[main_gi])
        gc = np.average(np.vstack([cl[i]["centroid"] for i in g]), axis=0,
                        weights=[cl[i]["E"] for i in g])
        displaced = (not is_main) and (np.linalg.norm(gc - main_ctr) >= DISP)
        # span-capped agglomeration of WHOLE clusters within the group
        order = sorted(g, key=lambda i: cl[i]["E"], reverse=True)
        used = set()
        for seed in order:
            if seed in used: continue
            if cl[seed]["span"] > D_MAX:            # big cluster -> split
                base = "DISPLACED" if displaced else ("MAIN" if is_main else "SATELLITE")
                for pc in _split_big_cluster(cl[seed], D_MAX, xyz, E, labels, seed):
                    pc["group"] = gi
                    pc["tag"] = base + "/" + pc["tag"]   # preserve TRACK-SEG / SHOWER-SEG
                    pieces.append(pc)
                used.add(seed); continue
            members = [seed]; hits = [cl[seed]["hits"]]; used.add(seed)
            bmin = cl[seed]["bmin"].copy(); bmax = cl[seed]["bmax"].copy()
            # only LOCAL candidates: KD-tree neighbors of the seed
            for cand in sorted(adj.get(seed, ()), key=lambda i: -cl[i]["E"]):
                if cand in used or cl[cand]["span"] > D_MAX: continue
                nmin = np.minimum(bmin, cl[cand]["bmin"])
                nmax = np.maximum(bmax, cl[cand]["bmax"])
                if np.linalg.norm(nmax - nmin) <= D_MAX:   # O(1) bbox span test
                    members.append(cand); hits.append(cl[cand]["hits"])
                    used.add(cand); bmin, bmax = nmin, nmax
            allh = np.concatenate(hits); w = np.maximum(E[allh], 1e-6)
            cc, _, _, span, _, _ = _pca(xyz[allh], w)
            pieces.append(dict(hits=allh, clusters=members, centroid=cc,
                E=float(w.sum()), span=span, group=gi,
                tag="DISPLACED" if displaced else ("MAIN" if is_main else "SATELLITE"),
                light={i: 1.0 for i in members}))
    return pieces, dict(main_ctr=main_ctr, n_groups=len(groups),
                        main_gi=main_gi)
