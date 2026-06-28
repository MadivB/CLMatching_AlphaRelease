"""
2x2 charge-light matching pipeline (vAlpha port) — orchestrator.

`run_pipeline_for_event(h5, ev_id, light_model=...)` mirrors the ND
`run_v11_pipeline_for_event`: it clusters the charge, predicts each cluster's
light with the 2x2 perceiver, and assigns every hit a t0 (matching ticks) by
variance-weighted chi2 matching against the observed light, with explicit
small-blob handling.

Phases
------
  1. clustering        : RANSAC tracks + DBSCAN blobs + noise list
  2. noise absorption  : each blob may adopt <=20% nearby noise hits
  3. light prediction  : perceiver -> imageMaps[(cid,tpc)] (clean & noisy)
  4. Stage 4 (tracks)  : backbone t0 per track (multi-TPC, single shared t0)
  5. Stage 5 (blobs)   : per-TPC greedy placement; faint blobs matched to
                         flash-seed t0s on their support channels
  6. Phase 8 (refine)  : per-cluster full scan + sub-tick refine

Returns a dict with `hit_timestamps` (per-hit t0) and `hit_refs`, plus
diagnostics and the assignment logs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import geometry_2x2 as geo
import data_2x2 as data
import matching_2x2 as match
import phase25_2x2 as p25
from track_fit_ransac import fit_tracks_labels
from cluster_fit import fit_cluster_labels, fit_noise_list


# ----------------------------------------------------------------------------
# Clustering helpers
# ----------------------------------------------------------------------------
def cluster_charge(xset, yset, zset, Eset, *, lam=1.1, min_length_cm=40.0,
                   dbscan_eps=2.5, dbscan_min_samples=3, expand_frac=0.10):
    """3-stage clustering: tracks -> DBSCAN blobs -> noise adoption list."""
    labels_p1 = fit_tracks_labels(xset, yset, zset, lam=lam,
                                  min_length_cm=min_length_cm)
    labels, _dbl, _noise_idx, _stats = fit_cluster_labels(
        xset, yset, zset, Eset, labels=labels_p1,
        eps=dbscan_eps, min_samples=dbscan_min_samples, verbose=False)
    noise_list = fit_noise_list(xset, yset, zset, Eset, labels,
                                expand_frac=expand_frac)
    track_id_max = int(labels_p1.max())
    return labels, noise_list, track_id_max


def build_noisy_labels(labels, noise_list, track_id_max, *, frac=0.20):
    """Each non-track cluster adopts up to ``frac`` of its size in nearby noise."""
    labels_noisy = np.asarray(labels, dtype=np.int64).copy()
    last = int(labels_noisy.max())
    for j in range(track_id_max + 1, last + 1):
        n_hit = int(np.count_nonzero(labels_noisy == j))
        n_max = int(n_hit * float(frac))
        if n_max <= 0:
            continue
        n_cur = 0
        for nidx, cid, _dist in noise_list:
            if int(cid) == j and labels_noisy[int(nidx)] == -1:
                labels_noisy[int(nidx)] = j
                n_cur += 1
                if n_cur >= n_max:
                    break
    return labels_noisy


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run_pipeline_for_event(h5, ev_id: int, *, light_model,
                           dead_yaml: str = "", lam: float = 1.1,
                           min_length_cm: float = 40.0, dbscan_eps: float = 2.5,
                           dbscan_min_samples: int = 3,
                           noise_absorb_frac: float = 0.20,
                           small_energy_mev: float = 8.0,
                           t0_resolution: int = 5,
                           support_fraction: float = 0.90,
                           support_floor: float = 25.0,
                           rel_sigma: float = 0.10,
                           variance_model=None,
                           unit_variance: bool = True,   # plain unit var wins (== ND vAlpha)
                           enable_refine: bool = True,
                           tiebreak_variance_model=None, tie_frac: float = 0.08,
                           enable_region_grow: bool = False,
                           enable_phase25: bool = False,
                           enable_colocation: bool = False,
                           light_support_mode: Optional[str] = None,  # None|'global'|'small'
                           light_support_keep_tiles: int = 4,
                           phase25_cfg=None,
                           verbose: bool = False) -> Optional[Dict[str, Any]]:
    ev = data.load_event(h5, ev_id, dead_yaml=dead_yaml, rel_sigma=rel_sigma,
                         variance_model=variance_model, unit_variance=unit_variance)
    if ev is None:
        if verbose:
            print(f"[ev {ev_id}] no light / no hits — skipped")
        return None

    x, y, z, E = ev.xset, ev.yset, ev.zset, ev.Eset
    tpcid = ev.hitTPCid

    # ND-style "light support": keep only the brightest light-detector tiles.
    tile_masks = None
    if light_support_mode in ("global", "small"):
        tile_masks = np.stack([
            match.light_support_tile_mask(ev.fullLightWaveform[t],
                                          n_keep_tiles=light_support_keep_tiles)
            for t in range(geo.N_TPCS)])
        if light_support_mode == "global":
            # mask the weak channels everywhere by inflating their variance
            for t in range(geo.N_TPCS):
                ev.fullLightVar[t][~tile_masks[t]] = np.float32(1.0e12)

    # 1) clustering
    labels, noise_list, track_id_max = cluster_charge(
        x, y, z, E, lam=lam, min_length_cm=min_length_cm,
        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples)
    n_clusters = int(labels.max()) + 1
    track_labels = list(range(track_id_max + 1))

    # 2) noise absorption
    labels_noisy = build_noisy_labels(labels, noise_list, track_id_max,
                                      frac=noise_absorb_frac)

    cluster_energies = {int(c): float(E[labels == c].sum())
                        for c in np.unique(labels) if c >= 0}

    # 3) light prediction (clean + noise-absorbed)
    image_maps, _ = light_model.predict_image_maps(x, y, z, E, tpcid, labels)
    image_maps_noisy, _ = light_model.predict_image_maps(
        x, y, z, E, tpcid, labels_noisy)

    cluster_to_tpcs: Dict[int, List[int]] = {}
    tpc_to_clusters: Dict[int, List[int]] = {}
    for (cid, tpc) in image_maps.keys():
        cluster_to_tpcs.setdefault(int(cid), []).append(int(tpc))
    for cid, tpcs in cluster_to_tpcs.items():
        if cid > track_id_max:                      # blobs only
            for t in tpcs:
                tpc_to_clusters.setdefault(int(t), []).append(int(cid))

    # state
    base_image = np.zeros((geo.N_TPCS, geo.N_CHANNELS, geo.WVFM_LEN), np.float32)
    hit_t0 = np.full(x.size, -1.0, dtype=np.float32)
    t0_candidates: List[List[float]] = [[] for _ in range(geo.N_TPCS)]

    # 4) Stage 4 — tracks
    track_rows = match.match_tracks(
        track_labels=track_labels, cluster_to_tpcs=cluster_to_tpcs,
        image_maps=image_maps, base_image=base_image,
        full_wvfm=ev.fullLightWaveform, full_var=ev.fullLightVar,
        labels=labels, hit_t0=hit_t0, t0_candidates=t0_candidates,
        flash_seeds=ev.flash_seeds, t0_resolution=t0_resolution)

    # learned variance for the ambiguous-t0 tiebreaker (optional test)
    tiebreak_var = None
    if tiebreak_variance_model is not None:
        tbl = data.get_tables(h5, dead_yaml=dead_yaml)
        tiebreak_var = tiebreak_variance_model.predict_variance(
            ev.extras["raw_sub"], tbl, dead_mask=tbl.dead_mask)

    # 5) Stage 5 — greedy blob placement (small-blob aware, flash-seeded)
    blob_rows = match.match_clusters_greedy(
        tpc_to_clusters=tpc_to_clusters, image_maps=image_maps,
        image_maps_noisy=image_maps_noisy, base_image=base_image,
        full_wvfm=ev.fullLightWaveform, full_var=ev.fullLightVar,
        labels=labels, labels_noisy=labels_noisy, hit_t0=hit_t0,
        t0_candidates=t0_candidates, cluster_energies=cluster_energies,
        flash_seeds=ev.flash_seeds, t0_resolution=t0_resolution,
        small_energy_mev=small_energy_mev, support_fraction=support_fraction,
        support_floor=support_floor,
        tile_masks=(tile_masks if light_support_mode == "small" else None),
        tiebreak_var=tiebreak_var, tie_frac=tie_frac)

    # 6) Phase 8 — refinement
    refine_rows = []
    if enable_refine:
        refine_rows = match.refine_clusters(
            tpc_to_clusters=tpc_to_clusters, image_maps=image_maps,
            base_image=base_image, full_wvfm=ev.fullLightWaveform,
            full_var=ev.fullLightVar, labels=labels, labels_noisy=labels_noisy,
            hit_t0=hit_t0, cluster_energies=cluster_energies,
            small_energy_mev=small_energy_mev, t0_resolution=t0_resolution)

    # 6b) Cluster-guided spatial region growing (alternative after-track
    #     association): confident clusters propagate their t0 to adjacent
    #     uncertain clusters, light-arbitrated. Off by default (baseline = greedy).
    region_grow_rows = []
    if enable_region_grow:
        region_grow_rows = match.region_grow_association(
            labels=labels, xset=x, yset=y, zset=z, Eset=E, hitTPCid=tpcid,
            hit_t0=hit_t0, cluster_energies=cluster_energies, image_maps=image_maps,
            base_image=base_image, full_wvfm=ev.fullLightWaveform,
            full_var=ev.fullLightVar, track_labels=track_labels)

    # 7) Phase 2.5 — Trial2 combined rescue (large flash-grid, spatial mixed-t0,
    #    physical-chi2 light repair). Recovers wrong-flash assignments in the
    #    higher-pile-up regions.
    phase25_result = None
    if enable_phase25:
        cfg = phase25_cfg or p25.Trial2Config(
            verbose=False, light_veto_track_min_tpcs=2)
        sat_cache = {"veto_mask": {t: (np.sum(ev.fullLightWaveform[t] > 60700.0,
                                              axis=1) > 6)
                                   for t in range(geo.N_TPCS)}}
        namespace = {
            "baseImage": base_image.astype(np.float32),
            "hit_timestamps": hit_t0.astype(np.float32),
            "fullLightWaveform": ev.fullLightWaveform.astype(np.float32),
            "fullLightStd": ev.fullLightVar.astype(np.float32),   # variance map
            "labels_global": labels.astype(np.int64),
            "hitTPCid": tpcid.astype(np.int32),
            "xset": x.astype(np.float64), "yset": y.astype(np.float64),
            "zset": z.astype(np.float64), "Eset": E.astype(np.float64),
            "imageMaps": image_maps, "cluster_to_tpcs": cluster_to_tpcs,
            "t0Candidates": t0_candidates,
            "track_shower_labels": track_labels,
            "saturated_channel_cache": sat_cache,
        }

        def _predict_family(xs, ys, zs, es, tpc):
            return light_model.predict_single_image(xs, ys, zs, es, tpc)

        phase25_result = p25.run_trial2_combined_rescue(
            namespace=namespace, predict_family_image_fn=_predict_family,
            cfg=cfg, commit=True)
        hit_t0 = np.asarray(namespace["hit_timestamps"], np.float32)
        base_image = np.asarray(namespace["baseImage"], np.float32)
        t0_candidates = namespace["t0Candidates"]

    # 8) Spatial co-location repair — snap a mis-matched large cluster to the
    #    consensus t0 of its touching neighbours (fixes split-shower fragments
    #    whose own light is too ambiguous to match).
    coloc_rows = []
    if enable_colocation:
        coloc_rows = match.colocation_repair(
            labels=labels, xset=x, yset=y, zset=z, hit_t0=hit_t0,
            cluster_energies=cluster_energies)

    n_matched = int(np.count_nonzero(np.isfinite(hit_t0) & (hit_t0 >= 0)))
    if verbose:
        print(f"[ev {ev_id}] hits={x.size} clusters={n_clusters} "
              f"(tracks={track_id_max + 1}, blobs={n_clusters - track_id_max - 1}) "
              f"matched_hits={n_matched}")

    return {
        "ev_id": int(ev_id),
        "hit_timestamps": hit_t0,
        "hit_refs": ev.hit_refs,
        "event": ev,
        "labels": labels,
        "labels_noisy": labels_noisy,
        "track_id_max": track_id_max,
        "track_labels": track_labels,
        "cluster_energies": cluster_energies,
        "cluster_to_tpcs": cluster_to_tpcs,
        "tpc_to_clusters": tpc_to_clusters,
        "t0_candidates": t0_candidates,
        "base_image": base_image,
        "image_maps": image_maps,
        "logs": {"tracks": track_rows, "blobs": blob_rows, "refine": refine_rows,
                 "region_grow": region_grow_rows,
                 "phase25": phase25_result, "colocation": coloc_rows},
        "n_matched_hits": n_matched,
    }


def run_multiple(h5, ev_ids, *, light_model, **kw):
    """Run several events; return concatenated (t0, hit_refs) and per-event results."""
    all_t0, all_refs, results = [], [], []
    for ev in np.atleast_1d(ev_ids):
        r = run_pipeline_for_event(h5, int(ev), light_model=light_model, **kw)
        if r is None:
            continue
        results.append(r)
        all_t0.append(np.asarray(r["hit_timestamps"], np.float32))
        all_refs.append(np.asarray(r["hit_refs"], np.int64))
    if not all_t0:
        return np.empty(0, np.float32), np.empty(0, np.int64), results
    return np.concatenate(all_t0), np.concatenate(all_refs), results


__all__ = ["run_pipeline_for_event", "run_multiple", "cluster_charge",
           "build_noisy_labels"]
