"""
v0.1 post-pass — spatial-guided cluster assignment for ND-LAr vAlpha.

RULE (required): the post-pass NEVER overrides an existing t0 assignment
from the main chain — assigned clusters are frozen anchors; only clusters
with no t0 can be assigned (respect_assigned=True default). Opt-in via
`--postpass v0.1`; with the flag off the pipeline is bit-identical to the
vAlpha baseline.

Two arms, both operating on the vAlpha namespace `ns` AFTER _run_phase3, both
moving ONLY blob clusters (labels >= split_index) with a uniform finite t0;
the Phase-1 backbone (labels < split_index) is never touched.

1. region_grow_nd   — the cosine region-grow (2x2-tuned gates): confident seeds
                      propagate t0 to spatially-adjacent blobs, whitened-cosine
                      light arbitration at zero base.
2. family_expand_nd — the cosine-free chi2 FAMILY EXPANSION with the
                      remove-and-rescore base fix: every chi2 is scored against
                      observed light with the running baseImage MINUS the
                      cluster's own current contribution (so pile-up light from
                      other interactions is explained away, unlike the 2x2
                      base=0 prototype that regressed on multi-flash events).
                      Candidates = the pipeline's own per-TPC t0Candidates
                      (flash table + backbone t0s + Phase-2/3 appends), so the
                      candidate-coverage gap of the 2x2 prototype is closed by
                      construction.

ND conventions honoured (from recon of M5p1):
  - chi2 = MEAN((model-actual)^2 / max(err,1e-6))  (compute_error_metric)
  - model = clip(base + shift(image, t0), None, ADC_CLIP=60780)
  - shift = _shift_block (fractional linear interp, baseline 0)
  - support channels from ns["cluster_channel_support_cache"] (already
    saturation-vetoed at build time); missing entry -> all 120 channels
  - err = ns["fullLightStd"] (predicted variance if model present, ones in the
    release env — matches what produced the 93% baseline)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

# flat M5p1 imports (resolved by M5p1.phase25_trial2_v_alpha_test._configure_paths)
from v3_2_global_matching import _shift_block  # fractional shift, baseline=0

ADC_CLIP = 60780.0


# ---------------------------------------------------------------------------
# shared bookkeeping
# ---------------------------------------------------------------------------

def _support_rows(ns, cid: int, tpc: int) -> Optional[np.ndarray]:
    ent = (ns.get("cluster_channel_support_cache") or {}).get((cid, tpc))
    if ent is None:
        return None
    idx = np.asarray(ent.get("channel_indices", []), np.int64)
    return idx if idx.size else None


def _cluster_state(ns, *, min_hits: int = 1):
    """Per-cluster bookkeeping over ALL labelled clusters.

    Returns dict cid -> {hits(bool mask), n, tpcs(with images), t0(uniform or
    None), E, backbone(bool)}. Blob clusters with mixed per-hit t0 (V2 split a
    sub-group off) get t0=None and are treated as immovable.
    """
    labels = np.asarray(ns["labels_global"])
    ts = np.asarray(ns["hit_timestamps"])
    E = np.asarray(ns["Eset"], np.float64)
    split = int(ns["split_index"])

    cid_tpcs: Dict[int, list] = defaultdict(list)      # one pass over the keys
    for (c, tp) in ns["imageMaps"]:
        cid_tpcs[int(c)].append(int(tp))

    st: Dict[int, dict] = {}
    for cid in (int(c) for c in np.unique(labels) if c >= 0):
        m = labels == cid
        if int(m.sum()) < min_hits:
            continue
        t = ts[m]
        fin = np.isfinite(t)
        if fin.any():
            tf = t[fin]
            uni = float(tf.max() - tf.min()) <= 1e-3
            t0 = float(np.median(tf)) if uni else None
        else:
            t0 = None
        st[cid] = {"mask": m, "n": int(m.sum()), "tpcs": sorted(cid_tpcs.get(cid, [])),
                   "t0": t0, "E": float(E[m].sum()),
                   "backbone": cid < split}
    return st


def _blob_adjacency(ns, st, *, rmax: float, include_unassigned: bool = False):
    """adj[e] = {other_cid: min hit-hit distance} for every movable blob e,
    against ANY cluster within rmax.  Built by querying each blob's (few) hits
    against one global KD-tree — avoids an O(N^2) pair sweep on ~2e5-hit events.
    """
    from scipy.spatial import cKDTree
    labels = np.asarray(ns["labels_global"])
    XYZ = np.column_stack([np.asarray(ns["xset"], np.float64),
                           np.asarray(ns["yset"], np.float64),
                           np.asarray(ns["zset"], np.float64)])
    tree = cKDTree(XYZ)
    adj: Dict[int, Dict[int, float]] = {}
    for cid, s in st.items():
        if s["backbone"] or not s["tpcs"]:
            continue
        if s["t0"] is None and not include_unassigned:
            continue
        pts = XYZ[s["mask"]]
        neigh = tree.query_ball_point(pts, r=float(rmax))
        best: Dict[int, float] = {}
        for i, nb in enumerate(neigh):
            if not nb:
                continue
            nb = np.asarray(nb)
            lab = labels[nb]
            keep = (lab >= 0) & (lab != cid)
            if not keep.any():
                continue
            d = np.linalg.norm(XYZ[nb[keep]] - pts[i], axis=1)
            for l, dd in zip(lab[keep], d):
                l = int(l)
                if dd < best.get(l, np.inf):
                    best[l] = float(dd)
        adj[cid] = best
    return adj


def _shift_rows(img_rows: np.ndarray, t0: float) -> np.ndarray:
    return _shift_block(np.ascontiguousarray(img_rows, np.float32), float(t0))


# ---------------------------------------------------------------------------
# chi2 (error-matrix) scoring with remove-and-rescore base
# ---------------------------------------------------------------------------

class _Chi2Scorer:
    def __init__(self, ns, st):
        self.ns = ns
        self.st = st
        self.act = np.asarray(ns["fullLightWaveform"], np.float32)
        self.err = np.maximum(np.asarray(ns["fullLightStd"], np.float32), 1e-6)
        # working copy: post-pass moves update it so later decisions see them
        self.base = np.asarray(ns["baseImage"], np.float32).copy()
        self.version = 0                       # bumped on every committed move
        self._sup = {}
        self._img = {}

    def _blocks(self, cid, tp):
        key = (cid, tp)
        if key not in self._img:
            sup = _support_rows(self.ns, cid, tp)
            img = np.asarray(self.ns["imageMaps"][key], np.float32)
            rows = slice(None) if sup is None else sup
            self._sup[key] = rows
            self._img[key] = img[rows]
        return self._img[key], self._sup[key]

    def chi2(self, cid: int, t: float) -> float:
        """MEAN chi2 of cluster cid placed at t, own contribution removed."""
        s = self.st[cid]
        num = 0.0
        cnt = 0
        for tp in s["tpcs"]:
            img_s, rows = self._blocks(cid, tp)
            base_s = self.base[tp][rows]
            if s["t0"] is not None:
                base_s = np.clip(base_s - _shift_rows(img_s, s["t0"]), 0.0, None)
            model = np.clip(base_s + _shift_rows(img_s, t), None, ADC_CLIP)
            act_s = self.act[tp][rows]
            err_s = self.err[tp][rows]
            num += float(((model - act_s) ** 2 / err_s).sum())
            cnt += int(model.size)
        return num / cnt if cnt else np.inf

    def move(self, cid: int, t_new: float):
        """Commit a move: update working base and the cluster's state t0."""
        s = self.st[cid]
        for tp in s["tpcs"]:
            img_s, rows = self._blocks(cid, tp)
            blk = self.base[tp][rows]
            if s["t0"] is not None:
                blk = blk - _shift_rows(img_s, s["t0"])
            blk = np.clip(blk + _shift_rows(img_s, t_new), 0.0, ADC_CLIP)
            self.base[tp][rows] = blk
        s["t0"] = float(t_new)
        self.version += 1


def family_expand_nd(ns, hit_t0: np.ndarray, *,
                     contact_cm: float = 3.5, accept_margin: float = 0.15,
                     reconcile_hyst: float = 0.05, try_next_neighbor: bool = True,
                     min_blob_hits: int = 1, max_outer: int = 20000,
                     respect_assigned: bool = True) -> List[dict]:
    """chi2 family expansion post-pass. Mutates hit_t0 in place; returns log.

    respect_assigned=True (DEFAULT, required rule): clusters that already
    carry a t0 from the main chain are FROZEN anchors — the pass may only
    ASSIGN clusters with no t0 (adoption into adjacent families); it never
    reassigns. respect_assigned=False is the legacy override mode."""
    labels = np.asarray(ns["labels_global"])
    st = _cluster_state(ns, min_hits=min_blob_hits)
    ns_ts_saved = ns["hit_timestamps"]
    ns["hit_timestamps"] = hit_t0            # st masks reference labels only; fine
    sc = _Chi2Scorer(ns, st)
    adj = _blob_adjacency(ns, st, rmax=max(contact_cm, 8.0),
                          include_unassigned=respect_assigned)

    t0c = ns["t0Candidates"]

    def candidates(cid) -> List[float]:
        s = st[cid]
        out = set()
        for tp in s["tpcs"]:
            if tp < len(t0c):
                out.update(round(float(x)) for x in t0c[tp])
        if s["t0"] is not None:
            out.add(round(s["t0"]))
        return sorted(float(x) for x in out)

    if respect_assigned:
        movable = [c for c, s in st.items()
                   if (not s["backbone"]) and s["t0"] is None and s["tpcs"]
                   and c in adj]
    else:
        movable = [c for c, s in st.items()
                   if (not s["backbone"]) and s["t0"] is not None and s["tpcs"]
                   and c in adj]
    cand_of = {c: candidates(c) for c in movable}
    movable = [c for c in movable if len(cand_of[c]) >= 1]

    # upfront error matrix (seeding order only; live decisions re-score fresh)
    emat = {c: np.array([sc.chi2(c, t) for t in cand_of[c]], float) for c in movable}

    def decisiveness(c) -> float:
        r = np.sort(emat[c])
        return float((r[1] - r[0]) / max(r[0], 1e-9)) if r.size >= 2 else 0.0

    _best_cache: Dict[int, tuple] = {}

    def best_now(c):
        hit = _best_cache.get(c)
        if hit is not None and hit[0] == sc.version:
            return hit[1], hit[2]
        vals = [sc.chi2(c, t) for t in cand_of[c]]
        j = int(np.argmin(vals))
        out = (sc.version, float(cand_of[c][j]), float(vals[j]))
        _best_cache[c] = out
        return out[1], out[2]

    def wants(c, t) -> bool:
        _, bchi = best_now(c)
        return sc.chi2(c, t) <= bchi * (1.0 + accept_margin)

    fam_of: Dict[int, Optional[int]] = {c: None for c in st}
    families: List[dict] = []
    for c, s in st.items():                       # backbone = immutable anchors
        if s["backbone"] and s["t0"] is not None:
            families.append({"clusters": {c}, "t0": s["t0"], "anchor": True})
            fam_of[c] = len(families) - 1
    if respect_assigned:                          # ALL assigned clusters = frozen anchors
        for c, s in st.items():
            if fam_of[c] is None and s["t0"] is not None:
                families.append({"clusters": {c}, "t0": s["t0"], "anchor": True})
                fam_of[c] = len(families) - 1

    rows: List[dict] = []
    unassigned = sorted((c for c in movable if fam_of[c] is None),
                        key=lambda c: (-decisiveness(c), -st[c]["E"]))
    pending = list(unassigned)
    guard = 0
    while pending and guard < max_outer:
        guard += 1
        seed = pending.pop(0)
        if fam_of[seed] is not None:
            continue
        bt, _ = best_now(seed)
        if st[seed]["t0"] is None or abs(bt - st[seed]["t0"]) > 0.5:  # assign/move seed
            sc.move(seed, bt)
            hit_t0[st[seed]["mask"]] = np.float32(bt)
            rows.append({"event": "seed_move", "cluster": seed,
                         "to_t0": bt, "energy_mev": st[seed]["E"]})
        fidx = len(families)
        families.append({"clusters": {seed}, "t0": st[seed]["t0"], "anchor": False})
        fam_of[seed] = fidx
        rows.append({"event": "seed", "cluster": seed, "t0": st[seed]["t0"],
                     "energy_mev": st[seed]["E"]})

        while guard < max_outer:                  # expand this family
            guard += 1
            F = families[fidx]
            ext = []
            for e in movable:
                if fam_of[e] == fidx:
                    continue
                dmins = [adj[e].get(c, np.inf) for c in F["clusters"]]
                d = min(dmins) if dmins else np.inf
                if d <= contact_cm:
                    ext.append((d, e))
            ext.sort()
            progressed = False
            for d, e in ext:
                fe = fam_of[e]
                if fe is None:                    # case a/b
                    if wants(e, F["t0"]):
                        if st[e]["t0"] is None or abs(st[e]["t0"] - F["t0"]) > 0.5:
                            sc.move(e, F["t0"])
                            hit_t0[st[e]["mask"]] = np.float32(F["t0"])
                        F["clusters"].add(e); fam_of[e] = fidx
                        rows.append({"event": "absorb", "cluster": e, "into": fidx,
                                     "t0": F["t0"], "dist_cm": round(d, 2),
                                     "energy_mev": st[e]["E"]})
                        progressed = True
                        break
                    rows.append({"event": "reject", "cluster": e,
                                 "family": fidx, "dist_cm": round(d, 2)})
                    if try_next_neighbor:
                        continue
                    break
                else:                             # case c: contested contact
                    F2 = families[fe]
                    if F2.get("anchor"):
                        continue
                    chi_a = sc.chi2(e, F["t0"])
                    chi_b = sc.chi2(e, F2["t0"])
                    move = chi_a < chi_b * (1.0 - reconcile_hyst)
                    rows.append({"event": "contact", "cluster": e, "famA": fidx,
                                 "famB": fe, "chi_A": chi_a, "chi_B": chi_b,
                                 "moved": bool(move), "dist_cm": round(d, 2)})
                    if move:
                        if st[e]["t0"] is None or abs(st[e]["t0"] - F["t0"]) > 0.5:
                            sc.move(e, F["t0"])
                            hit_t0[st[e]["mask"]] = np.float32(F["t0"])
                        F2["clusters"].discard(e)
                        F["clusters"].add(e); fam_of[e] = fidx
                        progressed = True
                        break
                    continue
            if not progressed:
                break

    ns["hit_timestamps"] = ns_ts_saved
    return rows


# ---------------------------------------------------------------------------
# cosine region-grow (2x2-tuned) — comparison arm
# ---------------------------------------------------------------------------

def _cosine_at(ns, st, cid: int, t: float) -> float:
    """Whitened cosine of cid's predicted pattern vs observed light at t
    (zero base — 2x2 region-grow convention), best over its TPCs."""
    act = np.asarray(ns["fullLightWaveform"], np.float32)
    err = np.maximum(np.asarray(ns["fullLightStd"], np.float32), 1e-6)
    best = 0.0
    for tp in st[cid]["tpcs"]:
        sup = _support_rows(ns, cid, tp)
        rows = slice(None) if sup is None else sup
        img_s = np.asarray(ns["imageMaps"][(cid, tp)], np.float32)[rows]
        a = _shift_rows(img_s, t)
        b = np.clip(act[tp][rows], 0.0, None)
        w = 1.0 / np.sqrt(err[tp][rows])
        aw, bw = a * w, b * w
        da = float(np.sqrt((aw * aw).sum()))
        db = float(np.sqrt((bw * bw).sum()))
        if da > 0 and db > 0:
            best = max(best, float((aw * bw).sum() / (da * db)))
    return best


def region_grow_nd(ns, hit_t0: np.ndarray, *,
                   contact_cm: float = 3.5, conf_cos: float = 0.55,
                   light_margin: float = 0.04, min_seed_cos: float = 0.20,
                   max_iter: int = 4) -> List[dict]:
    """Cosine region-grow post-pass (2x2-tuned defaults). In-place on hit_t0."""
    st = _cluster_state(ns)
    ns_ts_saved = ns["hit_timestamps"]
    ns["hit_timestamps"] = hit_t0
    adj = _blob_adjacency(ns, st, rmax=contact_cm)

    # symmetric adjacency incl. blob->backbone (from blob side)
    sym = defaultdict(dict)
    for e, m in adj.items():
        for c, d in m.items():
            if d <= contact_cm and c in st:
                sym[e][c] = d
                sym[c][e] = d

    info = {}
    for c, s in st.items():
        if s["t0"] is None or not s["tpcs"]:
            continue
        info[c] = {"t0": s["t0"],
                   "cos": _cosine_at(ns, st, c, s["t0"]),
                   "track": s["backbone"], "E": s["E"]}

    rows = []
    for _ in range(int(max_iter)):
        moved = 0
        for c in sorted(info, key=lambda i: (info[i]["track"], info[i]["cos"]),
                        reverse=True):
            if not (info[c]["track"] or info[c]["cos"] >= conf_cos):
                continue
            seed = info[c]
            for n in sym.get(c, {}):
                nb = info.get(n)
                if nb is None or nb["track"] or abs(nb["t0"] - seed["t0"]) < 1.0:
                    continue
                cs = _cosine_at(ns, st, n, seed["t0"])
                co = _cosine_at(ns, st, n, nb["t0"])
                if cs >= co - light_margin and cs >= min_seed_cos:
                    hit_t0[st[n]["mask"]] = np.float32(seed["t0"])
                    rows.append({"event": "grow", "cluster": n,
                                 "from_t0": nb["t0"], "to_t0": seed["t0"],
                                 "seed": c, "cos_seed": cs, "cos_own": co,
                                 "energy_mev": nb["E"]})
                    nb["t0"] = seed["t0"]
                    nb["cos"] = max(nb["cos"], conf_cos)
                    moved += 1
        if moved == 0:
            break
    ns["hit_timestamps"] = ns_ts_saved
    return rows
