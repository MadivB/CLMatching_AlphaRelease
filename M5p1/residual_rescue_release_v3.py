from __future__ import annotations

from typing import Any

import time
import numpy as np
import plotly.graph_objects as go

try:
    from v3_2_global_matching import _shift_block
except ModuleNotFoundError:
    from M5p1.v3_2_global_matching import _shift_block

try:
    from plottingTools import VALID_GROUP_COLORS
except ModuleNotFoundError:
    from M5p1.plottingTools import VALID_GROUP_COLORS

try:
    from ML_NDfull_perceiver import group_voxelize_pairs, predict_phi
except ModuleNotFoundError:
    from NewMLSection.ML_NDfull_perceiver import group_voxelize_pairs, predict_phi


# ============================================================================
# Spatial hash grid - O(n) build, O(1) neighbor lookup
# ============================================================================

class _SpatialGrid:
    """
    Fixed-cell spatial hash. Cell size is chosen from the median nearest-neighbor
    distance so queries touch only ~27 cells and return O(k) candidates.
    """
    __slots__ = ("cell", "origin", "buckets", "pts")

    def __init__(self, pts, cell_size):
        self.pts = pts
        self.cell = float(cell_size)
        self.origin = pts.min(axis=0) - self.cell
        ijk = np.floor((pts - self.origin) / self.cell).astype(np.int64)
        self.buckets = {}
        for idx, (i, j, k) in enumerate(ijk):
            key = (int(i), int(j), int(k))
            lst = self.buckets.get(key)
            if lst is None:
                self.buckets[key] = [idx]
            else:
                lst.append(idx)

    def query_radius(self, p, r):
        """Return indices within distance r of point p."""
        c = self.cell
        ext = int(np.ceil(r / c))
        base = np.floor((p - self.origin) / c).astype(np.int64)
        bx, by, bz = int(base[0]), int(base[1]), int(base[2])
        out = []
        r2 = r * r
        for di in range(-ext, ext + 1):
            for dj in range(-ext, ext + 1):
                for dk in range(-ext, ext + 1):
                    lst = self.buckets.get((bx + di, by + dj, bz + dk))
                    if lst is None:
                        continue
                    for idx in lst:
                        d = self.pts[idx] - p
                        if d[0]*d[0] + d[1]*d[1] + d[2]*d[2] <= r2:
                            out.append(idx)
        return out


# ============================================================================
# Main rescue
# ============================================================================

def _run_residual_rescue_tpc_bound(
    TPCid,
    *,
    flash_t0s=None,
    min_peak_missing_fraction=0.50,
    pulse_peak_tick=105,
    deficit_half_window_ticks=18,
    focus_pad_ticks=12,
    ignore_brightest_flash_peak=True,
    exclude_t0_zero=True,

    max_rescues=5,
    top_cluster_count=15,
    min_cosine_similarity=0.05,
    overflow_weight=3.0,
    min_loss_improvement=0.0,

    # --- NEW topology-growth parameters ---
    # Initial edge scale relative to the seed's local spacing. Larger -> more permissive adjacency.
    topo_edge_scale=2.0,
    # Hard cap on edge length as a multiple of the median local spacing over the whole donor cluster.
    topo_edge_global_cap=2.5,
    # Gap break: if the shortest link from frontier into the rest-of-donor exceeds this multiple
    # of the median edge length seen so far inside the grown component, stop growing.
    topo_gap_scale=2.2,
    # Energy-density bridge cut:
    # Once the component has at least `bridge_core_size` hits, reject a neighbor whose local
    # energy density (sum of E in a small ball of radius `bridge_probe_radius_scale` *
    # local_spacing) exceeds `bridge_density_jump` times the current component's mean density.
    bridge_core_size=8,
    bridge_density_jump=3.0,
    bridge_probe_radius_scale=2.0,
    # Shape / minimum-size floor
    min_component_size=6,
    max_component_size=400,  # hard safety cap

    # --- Legacy aggressive-accept knobs (kept) ---
    old_subtraction_scale=0.35,
    extreme_hole_fraction=0.90,
    extreme_accept_remaining_fraction=0.35,
    max_negative_loss_for_extreme=1.0e8,
    allow_add_only_for_extreme=True,

    max_tries_per_t0=4,
    allow_keep_best_partial=True,

    target_scale=None,
    batch_size=4,
    raw_clip=(0.0, 60780.0),
    min_prediction_threshold=100.0,
    device_policy="auto",

    exclude_track_shower_labels=True,
    use_unit_std=False,
    mutate=False,
    save_path=None,
    show=True,
    verbose=True,
):
    """
    Residual-driven rescue v5: topology-first growth.

    Growth philosophy:
      - Expand by pure BFS through geometric adjacency.
      - Stop at natural spatial gaps (edge-length jump).
      - Stop if crossing into a denser object (energy-density bridge cut).
      - Enforce a minimum size so we never return 2-3 hits.
      - Trust the downstream ML prediction + weighted-loss check to reject
        bad topology picks. No fill-score gating during growth.
    """
    t_start = time.time()
    TPCid = int(TPCid)

    if target_scale is None:
        try:
            target_scale = float(FIRST_STAGE_CONFIG.prediction.target_scale)
        except Exception:
            target_scale = 1e-3

    actual_full = np.asarray(fullLightWaveform[TPCid], dtype=np.float32)
    model_full  = np.asarray(baseImage[TPCid], dtype=np.float32).copy()
    base_out    = np.asarray(baseImage, dtype=np.float32).copy()
    hit_ts_out  = np.asarray(hit_timestamps, dtype=np.float32).copy()

    if use_unit_std:
        std_full = np.ones_like(actual_full, dtype=np.float32)
    elif "fullLightStd" in globals():
        std_full = np.asarray(fullLightStd[TPCid], dtype=np.float32)
    elif "fullLightStd_phase2" in globals():
        std_full = np.asarray(fullLightStd_phase2[TPCid], dtype=np.float32)
    else:
        std_full = np.ones_like(actual_full, dtype=np.float32)

    if "saturated_channel_cache" in globals() and saturated_channel_cache is not None:
        veto_mask = np.asarray(saturated_channel_cache["veto_mask"][TPCid], dtype=bool)
    else:
        veto_mask = np.sum(actual_full > 60700.0, axis=1) > 6

    keep_idx = np.flatnonzero(~veto_mask).astype(np.int32)
    if keep_idx.size == 0:
        raise RuntimeError(f"TPC {TPCid}: no unsaturated channels.")

    actual_kept = actual_full[keep_idx]
    std_kept    = np.maximum(std_full[keep_idx], 1e-6)

    def unique_t0s(values):
        out, seen = [], set()
        for v in values:
            if v is None or not np.isfinite(float(v)):
                continue
            t0 = int(round(float(v)))
            if exclude_t0_zero and t0 == 0:
                continue
            if t0 not in seen:
                seen.add(t0); out.append(t0)
        return sorted(out)

    flash_t0s_here = unique_t0s(
        t0Candidates[TPCid] if flash_t0s is None else flash_t0s
    )
    if not flash_t0s_here:
        if verbose:
            print(f"TPC {TPCid}: no flash t0s to inspect.")
        return None

    closed_t0s = set()

    # ------------------------------------------------------------------
    # Hole finder (unchanged)
    # ------------------------------------------------------------------
    def find_holes(cur_model):
        asum = np.sum(actual_kept, axis=0)
        msum = np.sum(np.asarray(cur_model[keep_idx], dtype=np.float32), axis=0)

        peak_info = []
        for t0 in flash_t0s_here:
            tick = int(t0 + pulse_peak_tick)
            if 0 <= tick < asum.shape[0]:
                peak_info.append(dict(t0=t0, tick=tick, pa=float(asum[tick])))

        brightest = None
        if ignore_brightest_flash_peak and peak_info:
            brightest = max(peak_info, key=lambda r: r["pa"])["t0"]

        rows = []
        for t0 in flash_t0s_here:
            if t0 in closed_t0s:
                continue
            tick = int(t0 + pulse_peak_tick)
            if tick < 0 or tick >= asum.shape[0]:
                continue
            if brightest is not None and t0 == brightest:
                continue

            lo = max(0, tick - deficit_half_window_ticks)
            hi = min(asum.shape[0], tick + deficit_half_window_ticks + 1)
            pa = float(asum[tick]); pm = float(msum[tick])
            pmiss = max(pa - pm, 0.0)
            pfrac = pmiss / max(pa, 1e-9)
            wa = float(np.sum(asum[lo:hi]))
            wmiss = float(np.sum(np.clip(asum[lo:hi] - msum[lo:hi], 0, None)))
            wfrac = wmiss / max(wa, 1e-9)
            if pfrac >= min_peak_missing_fraction:
                rows.append(dict(
                    t0=t0, peak_tick=tick, window_lo=lo, window_hi=hi,
                    peak_actual=pa, peak_model=pm, peak_missing=pmiss,
                    peak_missing_fraction=pfrac, window_actual=wa,
                    window_missing=wmiss, window_missing_fraction=wfrac,
                ))
        return sorted(rows, key=lambda r: (-r["peak_missing_fraction"], -r["peak_missing"]))

    def weighted_loss(mk, tmask):
        m = mk[:, tmask].astype(np.float32)
        a = actual_kept[:, tmask].astype(np.float32)
        s = std_kept[:, tmask].astype(np.float32)
        w = np.where(m > a, float(overflow_weight), 1.0).astype(np.float32)
        return float(np.sum(((m - a) ** 2 / s) * w))

    def focus_mask(hole_rows, extra_t0=None):
        nt = actual_full.shape[-1]
        mask = np.zeros(nt, dtype=bool)
        for r in hole_rows:
            tick = int(r["peak_tick"])
            lo = max(0, tick - deficit_half_window_ticks - focus_pad_ticks)
            hi = min(nt, tick + deficit_half_window_ticks + focus_pad_ticks + 1)
            mask[lo:hi] = True
        if extra_t0 is not None:
            tick = int(extra_t0 + pulse_peak_tick)
            lo = max(0, tick - deficit_half_window_ticks - focus_pad_ticks)
            hi = min(nt, tick + deficit_half_window_ticks + focus_pad_ticks + 1)
            mask[lo:hi] = True
        if not np.any(mask):
            mask[:] = True
        return mask

    tpc_mask   = (np.asarray(hitTPCid, dtype=np.int32) == TPCid)
    labels_arr = np.asarray(labels_global, dtype=np.int32)

    protected = set()
    if exclude_track_shower_labels and "track_shower_labels" in globals():
        protected = set(int(v) for v in track_shower_labels)

    _xset = np.asarray(xset, np.float64)
    _yset = np.asarray(yset, np.float64)
    _zset = np.asarray(zset, np.float64)
    _Eset = np.asarray(Eset, np.float32)

    # ------------------------------------------------------------------
    # Build TPC-wide spatial grid ONCE (for bridge-density probes across
    # the full environment, including other clusters).
    # ------------------------------------------------------------------
    tpc_hit_indices = np.flatnonzero(tpc_mask).astype(np.int64)
    if tpc_hit_indices.size == 0:
        raise RuntimeError(f"TPC {TPCid}: no hits.")
    tpc_pts = np.column_stack([
        _xset[tpc_hit_indices], _yset[tpc_hit_indices], _zset[tpc_hit_indices],
    ]).astype(np.float64)

    # Estimate a typical spacing in the TPC using a random subsample.
    n_tpc = tpc_pts.shape[0]
    sub_n = min(256, n_tpc)
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(n_tpc, size=sub_n, replace=False)
    sub_pts = tpc_pts[sub_idx]
    # Nearest-neighbor distance inside the subsample (O(sub_n^2), small).
    if sub_n >= 2:
        d2 = np.sum((sub_pts[:, None, :] - sub_pts[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(np.min(d2, axis=1))
        tpc_median_spacing = float(np.median(nn))
    else:
        tpc_median_spacing = 1.0
    tpc_median_spacing = max(tpc_median_spacing, 1e-6)

    tpc_grid = _SpatialGrid(tpc_pts, cell_size=max(tpc_median_spacing * 1.5, 1e-6))
    tpc_E = _Eset[tpc_hit_indices].astype(np.float32)

    # ------------------------------------------------------------------
    # Per-cluster info
    # ------------------------------------------------------------------
    cluster_info = {}
    for (cid_key, tpc_key), img in imageMaps.items():
        cid_key, tpc_key = int(cid_key), int(tpc_key)
        if tpc_key != TPCid or cid_key in protected:
            continue
        cmask = tpc_mask & (labels_arr == cid_key)
        hit_idx_arr = np.flatnonzero(cmask)
        if hit_idx_arr.size == 0:
            continue
        e_total = float(_Eset[cmask].sum())
        if e_total < 1e-6:
            continue
        vals = hit_ts_out[cmask]
        vals_fin = vals[np.isfinite(vals) & (vals >= 0)]
        if vals_fin.size == 0:
            ct0 = None
        else:
            uniq, cnts = np.unique(np.round(vals_fin).astype(np.int32), return_counts=True)
            ct0 = int(uniq[np.argmax(cnts)])
        img_np = np.asarray(img, dtype=np.float32)
        peak_amps_kept = np.maximum(img_np[:, pulse_peak_tick], 0.0)[keep_idx]
        peak_total_kept = float(np.sum(peak_amps_kept))
        if peak_total_kept > 0:
            peak_normed_kept = (peak_amps_kept / peak_total_kept).astype(np.float32)
            peak_norm_kept = float(np.linalg.norm(peak_normed_kept))
        else:
            peak_normed_kept = peak_amps_kept.astype(np.float32)
            peak_norm_kept = 0.0
        cluster_info[cid_key] = dict(
            hit_indices=hit_idx_arr,
            energy_total=e_total,
            current_t0=ct0,
            image=img_np,
            peak_amps_kept=peak_amps_kept,
            peak_total_kept=peak_total_kept,
            peak_normed_kept=peak_normed_kept,
            peak_norm_kept=peak_norm_kept,
        )

    # ------------------------------------------------------------------
    # Build adjacency for a donor cluster using the grid.
    # Returns points, per-hit local spacing, adjacency list, global cap.
    # Complexity: O(n * k) where k is avg neighbors per cell.
    # ------------------------------------------------------------------
    def build_donor_adjacency(hit_indices_cluster):
        hit_indices_cluster = np.asarray(hit_indices_cluster, dtype=np.int64)
        n = hit_indices_cluster.size
        if n == 0:
            return hit_indices_cluster, None, None, [], 0.0
        pts = np.column_stack([
            _xset[hit_indices_cluster],
            _yset[hit_indices_cluster],
            _zset[hit_indices_cluster],
        ]).astype(np.float64)
        if n == 1:
            return hit_indices_cluster, pts, np.array([1.0]), [[]], tpc_median_spacing

        # Build a small grid for this cluster.
        cell = max(tpc_median_spacing * 1.5, 1e-6)
        grid = _SpatialGrid(pts, cell_size=cell)

        # Local spacing = distance to nearest intra-cluster neighbor.
        # Search a small radius; expand if needed.
        local_spacing = np.empty(n, dtype=np.float64)
        search_r = cell * 2.5
        for i in range(n):
            cand = grid.query_radius(pts[i], search_r)
            best = np.inf
            for j in cand:
                if j == i:
                    continue
                d = pts[i] - pts[j]
                dd = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
                if dd < best:
                    best = dd
            if not np.isfinite(best):
                # fallback: expand
                cand = grid.query_radius(pts[i], search_r * 4)
                for j in cand:
                    if j == i:
                        continue
                    d = pts[i] - pts[j]
                    dd = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
                    if dd < best:
                        best = dd
            local_spacing[i] = np.sqrt(best) if np.isfinite(best) else cell

        median_local = float(np.median(local_spacing))
        global_cap = topo_edge_global_cap * median_local

        # Build adjacency: connect i-j iff dist <= min(scale*max(s_i,s_j), global_cap)
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            ri = topo_edge_scale * local_spacing[i]
            r_query = min(max(ri, topo_edge_scale * median_local), global_cap)
            cand = grid.query_radius(pts[i], r_query)
            for j in cand:
                if j <= i:
                    continue
                d = pts[i] - pts[j]
                dij = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
                rj = topo_edge_scale * local_spacing[j]
                thresh = min(max(ri, rj), global_cap)
                if dij <= thresh:
                    adjacency[i].append((j, float(dij)))
                    adjacency[j].append((i, float(dij)))

        return hit_indices_cluster, pts, local_spacing, adjacency, median_local

    # ------------------------------------------------------------------
    # Topology-first growth.
    # ------------------------------------------------------------------
    donor_adjacency_cache = {}
    local_density_cache = {}

    def grow_topology(hit_indices_cluster, seed_hidx):
        """
        Pure topology BFS from seed with:
          - natural-gap stop (edge length jump)
          - energy-density bridge cut (uses TPC-wide grid)
          - min/max component size
        """
        hit_indices_cluster = np.asarray(hit_indices_cluster, dtype=np.int64)
        if hit_indices_cluster.size <= 1:
            return hit_indices_cluster.copy(), dict(stop_reason="singleton",
                                                    median_edge=0.0, n_bridge_cuts=0)

        cache_key = tuple(int(v) for v in hit_indices_cluster.tolist())
        cached = donor_adjacency_cache.get(cache_key)
        if cached is None:
            cached = build_donor_adjacency(hit_indices_cluster)
            donor_adjacency_cache[cache_key] = cached
        hit_indices_cluster, pts, local_spacing, adjacency, median_local = cached

        seed_loc_arr = np.flatnonzero(hit_indices_cluster == int(seed_hidx))
        if seed_loc_arr.size == 0:
            return np.array([int(seed_hidx)], dtype=np.int64), dict(
                stop_reason="seed_missing", median_edge=0.0, n_bridge_cuts=0
            )
        seed_loc = int(seed_loc_arr[0])

        n = hit_indices_cluster.size
        selected = np.zeros(n, dtype=bool)
        selected[seed_loc] = True

        # Priority queue of candidate edges (dist, frontier_hit, neighbor).
        # We use a simple list-as-heap via sort  cluster sizes are small (<<1000).
        import heapq
        heap = []
        for (j, dij) in adjacency[seed_loc]:
            heapq.heappush(heap, (dij, seed_loc, j))

        # Running stats.
        edge_lengths = []   # edges we've accepted
        comp_energy = float(_Eset[hit_indices_cluster[seed_loc]])
        comp_volume_est = (4.0 / 3.0) * np.pi * (local_spacing[seed_loc] ** 3)
        bridge_cuts = 0
        stop_reason = "exhausted"

        probe_radius = bridge_probe_radius_scale * median_local

        def local_density_around(global_hit_idx, pt):
            """Energy density in a small ball around pt, using TPC-wide grid."""
            global_hit_idx = int(global_hit_idx)
            density_key = (global_hit_idx, float(probe_radius))
            cached_density = local_density_cache.get(density_key)
            if cached_density is not None:
                return cached_density
            cand = tpc_grid.query_radius(pt, probe_radius)
            if not cand:
                out_density = (0.0, 0.0)
                local_density_cache[density_key] = out_density
                return out_density
            cand = np.asarray(cand, dtype=np.int64)
            e_sum = float(np.sum(tpc_E[cand]))
            vol = (4.0 / 3.0) * np.pi * (probe_radius ** 3)
            out_density = (e_sum / max(vol, 1e-9), e_sum)
            local_density_cache[density_key] = out_density
            return out_density

        while heap and int(selected.sum()) < max_component_size:
            dij, i_from, j_to = heapq.heappop(heap)
            if selected[j_to]:
                continue

            n_selected_now = int(selected.sum())

            # --- gap break ---
            if edge_lengths and n_selected_now >= min_component_size:
                med_edge = float(np.median(edge_lengths))
                if dij > topo_gap_scale * max(med_edge, 1e-9):
                    stop_reason = "gap_break"
                    # But DON'T break the loop  other shorter edges in the heap
                    # may still be valid. Skip this one only.
                    continue

            # --- density bridge cut (only after we have a reasonable core) ---
            do_bridge_check = n_selected_now >= bridge_core_size
            if do_bridge_check:
                # current comp mean density
                comp_mean_dens = comp_energy / max(comp_volume_est, 1e-9)
                # density around the candidate neighbor
                neigh_dens, _ = local_density_around(hit_indices_cluster[j_to], pts[j_to])
                if comp_mean_dens > 0 and neigh_dens > bridge_density_jump * comp_mean_dens:
                    bridge_cuts += 1
                    stop_reason = "bridge_cut"
                    continue

            # accept
            selected[j_to] = True
            edge_lengths.append(float(dij))
            comp_energy += float(_Eset[hit_indices_cluster[j_to]])
            comp_volume_est += (4.0 / 3.0) * np.pi * (local_spacing[j_to] ** 3)

            for (k, dk) in adjacency[j_to]:
                if not selected[k]:
                    heapq.heappush(heap, (dk, j_to, k))

        # --- min-size floor: if tiny, force take nearest neighbors regardless ---
        if int(selected.sum()) < min_component_size:
            # distances from seed to everyone
            dseed = np.sqrt(np.sum((pts - pts[seed_loc]) ** 2, axis=1))
            order = np.argsort(dseed)
            need = min(min_component_size, n)
            for k in order:
                if int(selected.sum()) >= need:
                    break
                selected[k] = True
            stop_reason = stop_reason + "+min_floor"

        out = hit_indices_cluster[np.flatnonzero(selected)].astype(np.int64)
        if out.size == 0:
            out = np.array([int(seed_hidx)], dtype=np.int64)

        med_edge_final = float(np.median(edge_lengths)) if edge_lengths else 0.0
        return out, dict(
            stop_reason=stop_reason,
            median_edge=med_edge_final,
            n_bridge_cuts=int(bridge_cuts),
            n_selected=int(out.size),
        )

    # ------------------------------------------------------------------
    # Main loop (mostly unchanged from v4, only the growth call differs)
    # ------------------------------------------------------------------
    initial_holes = find_holes(model_full)
    if verbose:
        print(f"TPC {TPCid} residual rescue v5  topology-first growth")
        print(f"Flash t0s checked      : {flash_t0s_here}")
        print(f"Unsaturated channels   : {keep_idx.size}")
        print(f"Non-protected clusters : {len(cluster_info)}")
        print(f"TPC median spacing     : {tpc_median_spacing:.3f}")
        print(f"Deficit threshold      : {min_peak_missing_fraction:.0%}")
        print(f"Max tries per t0       : {max_tries_per_t0}")
        print(f"Min component size     : {min_component_size}")
        print(f"Max component size     : {max_component_size}")
        print(f"Gap scale              : {topo_gap_scale}x median edge")
        print(f"Bridge density jump    : {bridge_density_jump}x")
        print(f"Bridge core threshold  : {bridge_core_size} hits")
        print()

    moves = []
    rescued_hits = set()
    failed_components = set()

    for step in range(1, max_rescues + 1):
        holes = find_holes(model_full)
        if not holes:
            break

        hole = holes[0]
        target_t0 = int(hole["t0"])
        target_tick = int(hole["peak_tick"])

        deficit_ch = np.clip(
            actual_full[keep_idx, target_tick] - model_full[keep_idx, target_tick],
            0, None,
        ).astype(np.float32)

        total_deficit = float(deficit_ch.sum())
        if total_deficit <= 0:
            if verbose:
                print(f"step {step}: no positive deficit at t0={target_t0}.")
            closed_t0s.add(target_t0)
            continue

        deficit_norm = deficit_ch / total_deficit
        deficit_norm_norm = float(np.linalg.norm(deficit_norm))

        if verbose:
            print(f"step {step}: rescuing t0={target_t0} "
                  f"(current miss={100*hole['peak_missing_fraction']:.1f}%)")

        cluster_scores = []
        for cid, ci in cluster_info.items():
            if ci["current_t0"] is not None and ci["current_t0"] == target_t0:
                continue
            if float(ci["peak_total_kept"]) <= 0 or float(ci["peak_norm_kept"]) <= 0:
                continue
            dot = float(np.dot(deficit_norm, ci["peak_normed_kept"]))
            csim = dot / (deficit_norm_norm * float(ci["peak_norm_kept"])) if deficit_norm_norm > 0 else 0.0
            if csim < min_cosine_similarity:
                continue
            cluster_scores.append((cid, csim, ci["energy_total"]))

        cluster_scores.sort(key=lambda x: -(x[1] * x[2]))

        if not cluster_scores:
            if verbose:
                print(f"  No correlated donor clusters for t0={target_t0}.")
            closed_t0s.add(target_t0)
            continue

        hit_scores = []
        for cid, csim, cenergy in cluster_scores[:top_cluster_count]:
            ci = cluster_info[cid]
            pa = ci["peak_amps_kept"]
            et = ci["energy_total"]
            peak_total = float(ci["peak_total_kept"])
            if peak_total <= 0:
                continue
            for hidx in ci["hit_indices"]:
                if int(hidx) in rescued_hits:
                    continue
                he = float(_Eset[hidx])
                if he < 1e-6:
                    continue
                frac = he / et
                hc = frac * pa
                fill = np.minimum(hc, deficit_ch)
                overflow = np.maximum(hc - deficit_ch, 0)
                score = float(np.sum(fill)) - overflow_weight * float(np.sum(overflow))
                est_peak = frac * peak_total
                if score > 0 and est_peak > 0:
                    hit_scores.append((int(hidx), cid, he, score, frac, est_peak))

        if not hit_scores:
            if verbose:
                print("  No positive-scored seed candidates found.")
            closed_t0s.add(target_t0)
            continue

        hit_scores.sort(key=lambda x: -x[3])

        accepted_this_t0 = False
        best_try = None
        tries_done = 0
        tried_this_t0 = set()
        blocked_seed_hits_this_t0 = set()

        for seed_hidx, seed_cid, _, seed_score, _, _ in hit_scores:
            if tries_done >= int(max_tries_per_t0):
                break
            if int(seed_hidx) in blocked_seed_hits_this_t0:
                continue

            donor_hits_full = np.asarray(cluster_info[seed_cid]["hit_indices"], dtype=np.int64)
            donor_hits_full = donor_hits_full[
                np.array([int(h) not in rescued_hits for h in donor_hits_full], dtype=bool)
            ]

            selected_idx, growth_meta = grow_topology(donor_hits_full, seed_hidx)

            comp_key = (int(seed_cid), tuple(sorted(map(int, selected_idx.tolist()))))
            if comp_key in failed_components or comp_key in tried_this_t0:
                continue
            tried_this_t0.add(comp_key)

            if selected_idx.size == 0:
                failed_components.add(comp_key)
                continue

            donor_total_energy = max(cluster_info[seed_cid]["energy_total"], 1e-9)
            sel_energy = float(np.sum(_Eset[selected_idx]))
            donor_fraction = sel_energy / donor_total_energy

            tries_done += 1

            if verbose:
                print(
                    f"  try {tries_done}/{max_tries_per_t0}: "
                    f"seed cluster {seed_cid}, hit {seed_hidx}, "
                    f"score={seed_score:.1f}, hits={selected_idx.size}, "
                    f"E={sel_energy:.2f} MeV, donor_fraction={100*donor_fraction:.1f}%, "
                    f"stop={growth_meta['stop_reason']}, "
                    f"bridges_cut={growth_meta['n_bridge_cuts']}"
                )

            resc_label = max(int(labels_arr.max()) + 1, 90000) + step * 10 + tries_done

            maps_4d, grp_cls, grp_tpcs = group_voxelize_pairs(
                _xset[selected_idx], _yset[selected_idx], _zset[selected_idx], _Eset[selected_idx],
                np.full(len(selected_idx), TPCid, dtype=np.int64),
                np.full(len(selected_idx), resc_label, dtype=np.int64),
            )
            if maps_4d.shape[0] == 0:
                failed_components.add(comp_key)
                blocked_seed_hits_this_t0.update(map(int, selected_idx.tolist()))
                continue

            amps = predict_phi(
                maps_4d, model, grp_tpcs,
                target_scale=target_scale, batch_size=batch_size,
                raw_clip=raw_clip, min_prediction_threshold=min_prediction_threshold,
                device_policy=device_policy,
            )

            tmpl = np.asarray(wvfm_tmpl, np.float32)
            rescue_img = (amps[:, :, None] * tmpl[None, None, :])[0]
            rescue_shifted = _shift_block(
                rescue_img[None, :, :], target_t0, baseline=0.0,
            )[0]

            old_contribution = np.zeros_like(model_full, dtype=np.float32)
            old_t0 = cluster_info[seed_cid]["current_t0"]
            if old_t0 is not None:
                old_shifted = _shift_block(
                    cluster_info[seed_cid]["image"][None, :, :], int(old_t0), baseline=0.0,
                )[0]
                old_contribution += old_subtraction_scale * donor_fraction * old_shifted

            delta_sub = rescue_shifted - old_contribution
            model_kept = model_full[keep_idx].astype(np.float32)
            tmask = focus_mask(holes)

            before_loss = weighted_loss(model_kept, tmask)
            after_sub = weighted_loss(model_kept + delta_sub[keep_idx], tmask)
            imp_sub = before_loss - after_sub
            peak_add_sub = float(np.sum(rescue_shifted[keep_idx, target_tick]))
            rem_sub = max(
                hole["peak_actual"] - (hole["peak_model"] + peak_add_sub), 0.0,
            ) / max(hole["peak_actual"], 1e-9)

            current_delta = delta_sub
            current_after = after_sub
            current_improvement = imp_sub
            current_peak_add = peak_add_sub
            current_remaining = rem_sub
            current_mode = "subtract"

            if verbose:
                print(f"    subtract: peak_add={peak_add_sub:.1f}, "
                      f"remaining miss={100*rem_sub:.1f}%, loss ={imp_sub:.3e}")

            if allow_add_only_for_extreme:
                delta_add = rescue_shifted
                after_add = weighted_loss(model_kept + delta_add[keep_idx], tmask)
                imp_add = before_loss - after_add
                peak_add_add = float(np.sum(delta_add[keep_idx, target_tick]))
                rem_add = max(
                    hole["peak_actual"] - (hole["peak_model"] + peak_add_add), 0.0,
                ) / max(hole["peak_actual"], 1e-9)
                if verbose:
                    print(f"    add-only: peak_add={peak_add_add:.1f}, "
                          f"remaining miss={100*rem_add:.1f}%, loss ={imp_add:.3e}")
                if (after_add < current_after) or (
                    abs(after_add - current_after) < 1e-6 and rem_add < current_remaining
                ):
                    current_delta = delta_add
                    current_after = after_add
                    current_improvement = imp_add
                    current_peak_add = peak_add_add
                    current_remaining = rem_add
                    current_mode = "add_only"

            if verbose:
                print(f"    chosen mode={current_mode}, "
                      f"peak_add={current_peak_add:.1f}, "
                      f"remaining miss={100*current_remaining:.1f}%, "
                      f"loss ={current_improvement:.3e}")

            score_tuple = (current_after, current_remaining, -current_peak_add)
            if (best_try is None) or (score_tuple < best_try["score_tuple"]):
                best_try = dict(
                    score_tuple=score_tuple,
                    seed_cid=int(seed_cid),
                    seed_hidx=int(seed_hidx),
                    selected_idx=np.asarray(selected_idx, dtype=np.int64),
                    donor_fraction=float(donor_fraction),
                    sel_energy=float(sel_energy),
                    delta=np.asarray(current_delta, dtype=np.float32),
                    before_loss=float(before_loss),
                    after_loss=float(current_after),
                    improvement=float(current_improvement),
                    peak_add=float(current_peak_add),
                    remaining_missing_fraction=float(current_remaining),
                    mode=current_mode,
                    tries_used=int(tries_done),
                )

            converged_here = current_remaining <= float(min_peak_missing_fraction)
            accept = (current_improvement > float(min_loss_improvement)) and converged_here

            if (
                not accept
                and hole["peak_missing_fraction"] >= extreme_hole_fraction
                and current_remaining <= extreme_accept_remaining_fraction
                and current_improvement >= -abs(max_negative_loss_for_extreme)
            ):
                accept = True
                if verbose:
                    print(f"    -> ACCEPTED by extreme-hole rule ({current_mode})")

            if not accept:
                failed_components.add(comp_key)
                blocked_seed_hits_this_t0.update(map(int, selected_idx.tolist()))
                continue

            model_full = np.clip(model_full + current_delta, 0.0, ADC_CLIP)
            for hidx in selected_idx:
                hit_ts_out[int(hidx)] = np.float32(target_t0)
                rescued_hits.add(int(hidx))

            moves.append(dict(
                step=step, target_t0=target_t0,
                seed_cluster=int(seed_cid), seed_hit=int(seed_hidx),
                n_hits=int(selected_idx.size),
                rescue_energy=float(sel_energy),
                donor_fraction=float(donor_fraction),
                peak_add=float(current_peak_add),
                target_peak_missing=hole["peak_missing"],
                remaining_missing_fraction=float(current_remaining),
                loss_improvement=float(current_improvement),
                hit_indices=selected_idx.tolist(),
                forced_partial=False, converged_here=True,
                mode=current_mode, tries_used=int(tries_done),
                stop_reason=growth_meta["stop_reason"],
            ))

            if verbose:
                print(f"  ACCEPTED: try {tries_done}, cluster {seed_cid} -> t0={target_t0} | "
                      f"hits={selected_idx.size} | E={sel_energy:.2f} MeV | "
                      f"remaining miss={100*current_remaining:.1f}% | mode={current_mode}")
                print()

            accepted_this_t0 = True
            break

        if (not accepted_this_t0) and allow_keep_best_partial and (best_try is not None):
            model_full = np.clip(model_full + best_try["delta"], 0.0, ADC_CLIP)
            for hidx in best_try["selected_idx"]:
                hit_ts_out[int(hidx)] = np.float32(target_t0)
                rescued_hits.add(int(hidx))

            moves.append(dict(
                step=step, target_t0=target_t0,
                seed_cluster=int(best_try["seed_cid"]),
                seed_hit=int(best_try["seed_hidx"]),
                n_hits=int(best_try["selected_idx"].size),
                rescue_energy=float(best_try["sel_energy"]),
                donor_fraction=float(best_try["donor_fraction"]),
                peak_add=float(best_try["peak_add"]),
                target_peak_missing=hole["peak_missing"],
                remaining_missing_fraction=float(best_try["remaining_missing_fraction"]),
                loss_improvement=float(best_try["improvement"]),
                hit_indices=best_try["selected_idx"].tolist(),
                forced_partial=True, converged_here=False,
                mode=best_try["mode"], tries_used=int(best_try["tries_used"]),
                stop_reason="partial_kept",
            ))
            closed_t0s.add(target_t0)
            if verbose:
                print(f"  KEEP BEST PARTIAL after {best_try['tries_used']} tries: "
                      f"cluster {best_try['seed_cid']} -> t0={target_t0} | "
                      f"hits={best_try['selected_idx'].size} | "
                      f"remaining miss={100*best_try['remaining_missing_fraction']:.1f}% | "
                      f"mode={best_try['mode']}")
                print()
            continue

        if not accepted_this_t0:
            closed_t0s.add(target_t0)
            if verbose:
                print(f"  No acceptable rescue for t0={target_t0} after {tries_done} tries.")
                print()
            continue

    base_out[TPCid] = model_full
    final_holes = find_holes(model_full)
    converged = len(final_holes) == 0

    if verbose:
        print()
        hdr2 = (f"{'step':>4} {'target_t0':>10} {'seed_cl':>9} {'n_hits':>8} "
                f"{'E[MeV]':>10} {'dLoss':>14} {'remain%':>9} {'mode':>10} {'stop':>14}")
        print("Rescue summary")
        print(hdr2)
        print("-" * len(hdr2))
        for mv in moves:
            print(f"{mv['step']:4d} {mv['target_t0']:10d} {mv['seed_cluster']:9d} "
                  f"{mv['n_hits']:8d} {mv['rescue_energy']:10.2f} "
                  f"{mv['loss_improvement']:14.3e} "
                  f"{100*mv['remaining_missing_fraction']:8.1f}% "
                  f"{mv['mode']:>10s} {mv.get('stop_reason','?'):>14s}")
        if len(closed_t0s) > 0:
            print()
            print(f"Closed without convergence: {sorted(closed_t0s)}")
        if final_holes:
            print()
            print("Remaining deficient t0s")
            for r in final_holes:
                print(f"  t0={r['t0']:4d}  tick={r['peak_tick']:4d}  "
                      f"peak_miss={100*r['peak_missing_fraction']:.1f}%  "
                      f"win_miss={100*r['window_missing_fraction']:.1f}%")
        print(f"\nElapsed: {time.time() - t_start:.2f}s")

    write_plot = save_path is not None
    if save_path is None and bool(show):
        save_path = f"TPC{TPCid}_residual_rescue_v5.html"
        write_plot = True

    if moves and write_plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=_zset[tpc_mask], y=_yset[tpc_mask], z=_xset[tpc_mask],
            mode="markers",
            marker=dict(size=2.0, color="lightgray", opacity=0.16),
            name=f"TPC {TPCid} all hits", hoverinfo="skip",
        ))
        for i, mv in enumerate(moves):
            idx = np.array(mv["hit_indices"], dtype=np.int64)
            color = VALID_GROUP_COLORS[i % len(VALID_GROUP_COLORS)][0]
            name = (f"cluster {mv['seed_cluster']} -> t0={mv['target_t0']} "
                    f"({mv['n_hits']} hits, {mv['rescue_energy']:.1f} MeV)")
            hover = (
                f"step={mv['step']}<br>target_t0={mv['target_t0']}<br>"
                f"seed_cluster={mv['seed_cluster']}<br>seed_hit={mv['seed_hit']}<br>"
                f"n_hits={mv['n_hits']}<br>E={mv['rescue_energy']:.2f} MeV<br>"
                f"donor_fraction={100*mv['donor_fraction']:.1f}%<br>"
                f"remaining miss={100*mv['remaining_missing_fraction']:.1f}%<br>"
                f"mode={mv['mode']}<br>stop={mv.get('stop_reason','?')}<br>"
                f"forced_partial={mv['forced_partial']}<br>"
                f"tries_used={mv['tries_used']}<br>"
                f"dLoss={mv['loss_improvement']:.3e}"
            )
            fig.add_trace(go.Scatter3d(
                x=_zset[idx], y=_yset[idx], z=_xset[idx],
                mode="markers",
                marker=dict(size=5.0, color=color, opacity=0.98),
                name=name, hovertext=[hover]*len(idx), hoverinfo="text+x+y+z",
            ))
        fig.update_layout(
            title=(f"TPC {TPCid} residual rescue v5 (topology) | rescues={len(moves)} "
                   f"| converged={converged}"),
            height=900, margin=dict(l=0, r=0, b=0, t=55),
            scene=dict(xaxis_title="z", yaxis_title="y", zaxis_title="x"),
            showlegend=True,
        )
        fig.write_html(save_path)
        if verbose:
            print(f"Saved html: {save_path}")
        if show:
            fig.show()
    elif not moves:
        if verbose:
            print("No rescue moves made; no HTML written.")

    if mutate:
        globals()["baseImage"] = base_out
        globals()["hit_timestamps"] = hit_ts_out

    return dict(
        TPCid=TPCid, moves=moves,
        initial_holes=initial_holes, final_holes=final_holes,
        converged=converged, closed_t0s=sorted(closed_t0s),
        baseImage=base_out, hit_timestamps=hit_ts_out,
        save_path=save_path if moves else None,
        flash_t0s_used=flash_t0s_here,
        keep_channel_indices=keep_idx,
    )


_REQUIRED_NAMESPACE_NAMES = (
    "fullLightWaveform",
    "baseImage",
    "hit_timestamps",
    "t0Candidates",
    "hitTPCid",
    "labels_global",
    "xset",
    "yset",
    "zset",
    "Eset",
    "imageMaps",
    "model",
    "wvfm_tmpl",
)

_OPTIONAL_NAMESPACE_NAMES = (
    "fullLightStd",
    "fullLightStd_phase2",
    "saturated_channel_cache",
    "track_shower_labels",
    "FIRST_STAGE_CONFIG",
    "ADC_CLIP",
)

_BOUND_NAMESPACE: dict[str, Any] = {}


def bind_notebook_namespace(namespace: dict[str, Any]) -> None:
    """Bind notebook globals used by the residual rescue implementation."""
    missing = [name for name in _REQUIRED_NAMESPACE_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(f"Missing required residual-rescue namespace values: {missing}")

    _BOUND_NAMESPACE.clear()
    _BOUND_NAMESPACE.update(namespace)

    for name in _REQUIRED_NAMESPACE_NAMES:
        globals()[name] = namespace[name]

    for name in _OPTIONAL_NAMESPACE_NAMES:
        if name in namespace:
            globals()[name] = namespace[name]
        elif name in globals():
            del globals()[name]


def _write_back_namespace(namespace: dict[str, Any], result: dict[str, Any] | None) -> None:
    if result is None:
        return
    if "baseImage" in result:
        namespace["baseImage"] = result["baseImage"]
        globals()["baseImage"] = result["baseImage"]
    if "hit_timestamps" in result:
        namespace["hit_timestamps"] = result["hit_timestamps"]
        globals()["hit_timestamps"] = result["hit_timestamps"]


def run_residual_rescue_tpc(
    TPCid: int,
    *,
    namespace: dict[str, Any] | None = None,
    mutate: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """
    Run the tested topology-growth residual rescue on one TPC.

    If namespace is supplied and mutate=True, this updates namespace["baseImage"]
    and namespace["hit_timestamps"] after the move. The implementation also
    updates the returned light prediction image for every accepted rescued
    component by adding the newly predicted shifted light and subtracting the
    configured old donor contribution.
    """
    if namespace is not None:
        bind_notebook_namespace(namespace)
    elif not _BOUND_NAMESPACE:
        raise RuntimeError("Pass namespace=globals() or call bind_notebook_namespace(globals()) first.")

    kwargs.pop("mutate", None)
    result = _run_residual_rescue_tpc_bound(int(TPCid), mutate=False, **kwargs)
    if mutate and namespace is not None:
        _write_back_namespace(namespace, result)
    return result


def run_residual_rescue_all_tpcs(
    *,
    namespace: dict[str, Any],
    target_tpcs: list[int] | tuple[int, ...] | None = None,
    rescue_kwargs: dict[str, Any] | None = None,
    mutate: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run residual rescue sequentially over TPCs with candidate t0s."""
    bind_notebook_namespace(namespace)
    rescue_kwargs = {} if rescue_kwargs is None else dict(rescue_kwargs)
    rescue_kwargs.pop("mutate", None)
    rescue_kwargs.setdefault("show", False)
    rescue_kwargs.setdefault("save_path", None)
    rescue_kwargs.setdefault("verbose", bool(verbose))

    if target_tpcs is None:
        candidate_tpcs = [
            int(tpc)
            for tpc, values in enumerate(namespace["t0Candidates"])
            if len(values) > 0
        ]
    else:
        candidate_tpcs = sorted(set(int(v) for v in target_tpcs))

    results_by_tpc: dict[int, dict[str, Any] | None] = {}
    rescue_log: list[dict[str, Any]] = []
    total_moves = 0
    unconverged_tpcs: list[int] = []

    for tpc in candidate_tpcs:
        # Re-bind before each TPC so the next TPC sees the updated image from
        # previous accepted moves.
        bind_notebook_namespace(namespace)
        result = _run_residual_rescue_tpc_bound(int(tpc), mutate=False, **rescue_kwargs)
        results_by_tpc[int(tpc)] = result
        if result is None:
            continue

        moves = [dict(row) for row in result.get("moves", [])]
        if moves:
            total_moves += len(moves)
            for row in moves:
                row["TPCid"] = int(tpc)
                rescue_log.append(row)
            if mutate:
                _write_back_namespace(namespace, result)

        if moves and not bool(result.get("converged", True)):
            unconverged_tpcs.append(int(tpc))

    out = {
        "results_by_tpc": results_by_tpc,
        "rescue_log": rescue_log,
        "n_tpcs_scanned": int(len(candidate_tpcs)),
        "n_tpcs_with_moves": int(sum(1 for r in results_by_tpc.values() if r is not None and len(r.get("moves", [])) > 0)),
        "n_moves": int(total_moves),
        "unconverged_tpcs": sorted(set(unconverged_tpcs)),
        "baseImage": namespace["baseImage"],
        "hit_timestamps": namespace["hit_timestamps"],
    }

    namespace["residual_rescue_v3_results"] = out
    namespace["residual_rescue_v3_log"] = rescue_log
    return out


__all__ = [
    "bind_notebook_namespace",
    "run_residual_rescue_tpc",
    "run_residual_rescue_all_tpcs",
]
