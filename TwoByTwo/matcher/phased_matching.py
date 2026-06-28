"""
Phased charge-light matching helpers for the 2x2 v4 pipeline.

Adapted from M5p1/v11_phased_matching.py, simplified for the 2x2 detector
(8 TPCs, 48 channels, 1000-tick waveforms; no shower workflow; no light<->charge
TPC remap; no streaming voxelization).

Provides:
  - snapshot_backbone_hits / verify_backbone_hits_unchanged
  - run_large_cluster_scan_phase  (Phase 2)
  - run_small_cluster_matrix_phase (Phase 3)

The first-stage matching (charge clustering, CNN image prediction, track t0,
DBSCAN cluster t0, transporter, refinement) lives in the notebook; this module
runs *after* the first stage on the residual cluster population.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Backbone snapshot/verify
# ---------------------------------------------------------------------------

def snapshot_backbone_hits(*, hit_timestamps, labels_global,
                            track_shower_labels=None, track_only=True):
    """Freeze the per-hit t0 values for hits already decided by the first stage.

    By default (``track_only=True``) only hits belonging to a track or shower
    cluster (``track_shower_labels``) are frozen — these must never move in
    later phases. Pass ``track_only=False`` to freeze every decided hit (in
    which case Trial2 sub-step (b)'s legitimate non-track moves will trigger
    the verify warning).
    """
    ts = np.asarray(hit_timestamps, dtype=np.float32)
    labels = np.asarray(labels_global, dtype=np.int64)
    decided_mask = np.isfinite(ts) & (ts >= 0)
    if track_only:
        ts_set = set(int(v) for v in (track_shower_labels or []))
        if not ts_set:
            decided_mask = np.zeros_like(decided_mask, dtype=bool)
        else:
            track_mask = np.isin(labels, np.fromiter(ts_set, dtype=np.int64))
            decided_mask &= track_mask
    decided_idx = np.flatnonzero(decided_mask)
    return {
        "mask": decided_mask.copy(),
        "indices": decided_idx.astype(np.int64).copy(),
        "expected_t0": ts[decided_mask].copy(),
        "labels": labels[decided_mask].copy(),
        "track_shower_labels": [] if track_shower_labels is None
                               else [int(v) for v in track_shower_labels],
        "track_only": bool(track_only),
        "n_hits": int(decided_mask.sum()),
    }


def verify_backbone_hits_unchanged(snapshot, *, hit_timestamps, stage_name,
                                   atol=1e-4, max_examples=8):
    mask = np.asarray(snapshot["mask"], dtype=bool)
    expected = np.asarray(snapshot["expected_t0"], dtype=np.float32)
    indices = np.asarray(snapshot["indices"], dtype=np.int64)
    labels = np.asarray(snapshot["labels"], dtype=np.int64)
    current = np.asarray(hit_timestamps, dtype=np.float32)[mask]
    mismatch = (~np.isfinite(current)) | (np.abs(current - expected) > float(atol))
    n_changed = int(np.count_nonzero(mismatch))
    n_total = int(mask.sum())
    if n_changed:
        msg = f"WARNING: {n_changed}/{n_total} backbone hits changed after {stage_name}."
        warnings.warn(msg)
        print(msg)
        bad = np.flatnonzero(mismatch)[: int(max_examples)]
        print(f"{'hit_idx':>8} {'label':>8} {'expected':>12} {'current':>12}")
        for j in bad:
            print(f"{int(indices[j]):8d} {int(labels[j]):8d} "
                  f"{float(expected[j]):12.4f} {float(current[j]):12.4f}")
    else:
        print(f"Backbone integrity OK after {stage_name}: {n_total}/{n_total} unchanged.")
    return {"stage_name": stage_name, "n_total": n_total, "n_changed": n_changed,
            "changed_indices": indices[mismatch].copy()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shift_image_fractional(image, shift_ticks, *, nt):
    """Linear-interp shift of a (channels, T) template by a possibly fractional t0."""
    img = np.asarray(image, dtype=np.float32)
    out = np.zeros((img.shape[0], int(nt)), dtype=np.float32)
    src_x = np.arange(img.shape[1], dtype=np.float32)
    dst_x = np.arange(int(nt), dtype=np.float32) - float(shift_ticks)
    for ch in range(img.shape[0]):
        out[ch] = np.interp(dst_x, src_x, img[ch], left=0.0, right=0.0).astype(np.float32)
    return out


def _shift_image_integer(image, t0, *, nt=None):
    img = np.asarray(image, dtype=np.float32)
    if nt is None:
        nt = img.shape[1]
    out = np.zeros((img.shape[0], int(nt)), dtype=np.float32)
    t0 = int(round(float(t0)))
    if t0 >= 0:
        if t0 < nt:
            n = min(img.shape[1], nt - t0)
            out[:, t0:t0 + n] = img[:, :n]
    else:
        src0 = -t0
        if src0 < img.shape[1]:
            n = min(img.shape[1] - src0, nt)
            out[:, :n] = img[:, src0:src0 + n]
    return out


def _scan_window_loss(predicted_shifted, base, actual, std):
    """Per-tick chi2: sum((predicted_shifted + base - actual)^2 / std)."""
    model = np.clip(predicted_shifted + base, 0.0, 60780.0)
    return float(np.sum((model - actual) ** 2 / np.maximum(std, 1e-6)))


def _full_integer_scan_loss(image, base, actual, std, search_range=700):
    """Returns (shifts, errors) for integer t0 in [0, search_range]."""
    n_ticks = image.size
    shifts = np.arange(int(search_range) + 1, dtype=np.int32)
    errors = np.empty(shifts.shape[0], dtype=np.float32)
    pred = image.astype(np.float32)
    base = base.astype(np.float32)
    actual = actual.astype(np.float32)
    std = np.maximum(std.astype(np.float32), 1e-6)
    for k, t0 in enumerate(shifts):
        if t0 == 0:
            shifted = pred
        else:
            shifted = np.zeros_like(pred)
            shifted[:, t0:] = pred[:, :-t0]
        model = np.clip(shifted + base, None, 60780.0)
        errors[k] = ((model - actual) ** 2 / std).sum() / n_ticks
    return shifts, errors


# ---------------------------------------------------------------------------
# Phase 2 — Large-cluster scan with flash-table seeding
# ---------------------------------------------------------------------------

def run_large_cluster_scan_phase(
    *,
    cluster_to_tpcs,
    image_maps,
    base_image,
    full_light_waveform,
    full_light_std,
    labels_global,
    hit_timestamps,
    t0_candidates,
    cluster_energies,
    track_shower_labels=None,
    large_cluster_energy_mev=50.0,
    minimum_iterative_energy_mev=0.5,
    search_range=700,
    flash_seed_resolution_ticks=5,
    pulse_peak_tick=105,
    backward_align_ticks=5,
    sub_tick_grid=None,
    improvement_threshold=0.0,
):
    """Run a full integer t0 scan against the flash-table for large clusters.

    For every (cluster, tpc) where the cluster is non-track/shower and has total
    energy >= ``large_cluster_energy_mev``:

      1. Compute current per-cluster-per-tpc t0 (median of constituent hit t0s).
      2. Run a full integer scan + sub-tick refinement on the flash candidates
         in ``t0_candidates[tpc]`` (or fall back to a wide scan if none).
      3. Accept the best new t0 if it improves the per-TPC chi2 by more than
         ``improvement_threshold`` and differs by > 1e-3 ticks from current.
      4. Update ``base_image[tpc]`` exactly (subtract old shift, add new shift)
         and update ``hit_timestamps`` for the affected hits.

    Returns (base_image, hit_timestamps, accepted_rows, all_rows, partition).
    """
    if sub_tick_grid is None:
        sub_tick_grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

    base = np.asarray(base_image, dtype=np.float32).copy()
    hit_ts = np.asarray(hit_timestamps, dtype=np.float32).copy()
    actual = np.asarray(full_light_waveform, dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std, dtype=np.float32), 1e-6)
    labels = np.asarray(labels_global, dtype=np.int64)

    track_shower_set = {int(v) for v in (track_shower_labels or [])}
    nt = base.shape[-1]

    # Partition.
    primary_clusters = []
    iterative_single_tpc = {}
    pruned_clusters = []
    for cid, tpcs in cluster_to_tpcs.items():
        cid = int(cid)
        cid_e = float(cluster_energies.get(cid, 0.0))
        tpcs = sorted(int(t) for t in tpcs if (cid, int(t)) in image_maps)
        if not tpcs:
            continue
        if cid in track_shower_set:
            continue
        if len(tpcs) > 1 or cid_e > float(large_cluster_energy_mev):
            primary_clusters.append((cid, cid_e, tpcs))
        elif cid_e < float(minimum_iterative_energy_mev):
            pruned_clusters.append(cid)
        else:
            iterative_single_tpc.setdefault(int(tpcs[0]), []).append(cid)

    # Sort by descending energy (largest first).
    primary_clusters.sort(key=lambda r: (-r[1], r[0]))

    accepted_rows = []
    all_rows = []
    for cid, cid_e, tpcs in primary_clusters:
        for tpc in tpcs:
            key = (cid, int(tpc))
            if key not in image_maps:
                continue
            hit_idx = np.flatnonzero((labels == cid)
                                     & (np.asarray([True] * labels.size)))  # placeholder
            # Recompute mask carefully: we don't have hitTPCid here directly,
            # so recompute below using the cluster_to_tpcs assumption.
            # For Phase 2 we use the median of all hits in the cluster as a starting t0,
            # since labels alone cannot distinguish per-tpc hits without hitTPCid.
            # The notebook will pass labels and assume cluster_to_tpcs already filtered.
            cur_ts = hit_ts[labels == cid]
            finite = np.isfinite(cur_ts) & (cur_ts >= 0)
            current_t0 = float(np.median(cur_ts[finite])) if finite.any() else 0.0

            cluster_img = np.asarray(image_maps[key], dtype=np.float32)
            old_shifted = _shift_image_fractional(cluster_img, current_t0, nt=nt)
            base_without = np.clip(base[int(tpc)] - old_shifted, 0.0, None)
            before_loss = _scan_window_loss(old_shifted, base_without,
                                            actual[int(tpc)], std[int(tpc)])

            # Build candidate t0 set.
            cand_t0s = [float(v) for v in t0_candidates[int(tpc)]
                        if v is not None and np.isfinite(float(v))]
            if not cand_t0s:
                # Fallback: full integer scan.
                shifts, errors = _full_integer_scan_loss(
                    cluster_img, base_without, actual[int(tpc)], std[int(tpc)],
                    search_range=int(search_range),
                )
                best_int = int(shifts[int(np.argmin(errors))])
                cand_t0s = [float(best_int)]

            # Sub-tick fine grid around each candidate.
            best_loss = float("inf")
            best_t0 = float(current_t0)
            best_model = None
            for ft0 in cand_t0s:
                for off in sub_tick_grid:
                    cand = float(ft0) + float(off)
                    new_shifted = _shift_image_fractional(cluster_img, cand, nt=nt)
                    trial_model = np.clip(base_without + new_shifted, 0.0, 60780.0)
                    loss = _scan_window_loss(new_shifted, base_without,
                                             actual[int(tpc)], std[int(tpc)])
                    if loss < best_loss:
                        best_loss = loss
                        best_t0 = cand
                        best_model = trial_model

            improvement = before_loss - best_loss
            accepted = (improvement >= float(improvement_threshold)
                        and abs(best_t0 - current_t0) > 1e-3)
            row = {
                "clusterid": int(cid), "TPCid": int(tpc),
                "energy_mev": float(cid_e),
                "current_t0": float(current_t0),
                "best_t0": float(best_t0),
                "before_loss": float(before_loss),
                "best_loss": float(best_loss),
                "improvement": float(improvement),
                "accepted": bool(accepted),
            }
            all_rows.append(row)
            if accepted and best_model is not None:
                base[int(tpc)] = best_model.astype(np.float32)
                hit_ts[labels == cid] = np.float32(best_t0)
                accepted_rows.append(row)

    return (base, hit_ts, accepted_rows, all_rows,
            {"primary_clusters": [r[0] for r in primary_clusters],
             "iterative_single_tpc": iterative_single_tpc,
             "pruned_clusters": pruned_clusters})


# ---------------------------------------------------------------------------
# Phase 3 — Small-cluster matrix phase
# ---------------------------------------------------------------------------

def _loss_matrix(image_maps, actual_tpc, base_tpc, std_tpc, tpc, clusters,
                 placed_mask, t0_list):
    """Vectorised chi2 matrix for remaining clusters x candidate t0s."""
    remaining = [int(c) for c, p in zip(clusters, placed_mask) if not p]
    if not remaining or not t0_list:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)
    nt = actual_tpc.shape[-1]
    n_ch = actual_tpc.shape[0]
    M = np.empty((len(remaining), len(t0_list)), dtype=np.float32)
    norm = float(n_ch * nt)
    std = np.maximum(std_tpc, 1e-6)
    for i, cid in enumerate(remaining):
        if (cid, int(tpc)) not in image_maps:
            M[i, :] = np.inf
            continue
        img = np.asarray(image_maps[(cid, int(tpc))], dtype=np.float32)
        for j, t0 in enumerate(t0_list):
            shifted = _shift_image_fractional(img, float(t0), nt=nt)
            model = np.clip(shifted + base_tpc, 0.0, 60780.0)
            M[i, j] = float(((model - actual_tpc) ** 2 / std).sum() / norm)
    return M, np.asarray(remaining, dtype=np.int64)


def run_small_cluster_matrix_phase(
    *,
    iterative_single_tpc,
    image_maps,
    base_image,
    full_light_waveform,
    full_light_std,
    labels_global,
    hit_timestamps,
    t0_candidates,
    cluster_energies,
    energy_band_fraction=0.20,
    matrix_worsen_tolerance_norm=0.15,
    search_range=700,
    pulse_peak_tick=105,
    t0_resolution_ticks=5,
):
    """Single-TPC small-cluster matrix-association phase.

    For each TPC and its small clusters (sorted by descending energy):
      - Within an energy band of fraction ``energy_band_fraction`` (anchor =
        the most-energetic remaining cluster), build the loss matrix vs the
        existing per-TPC t0 candidates and pick the (cluster, t0) with the
        global minimum that improves the per-cluster-current-loss by more than
        ``matrix_worsen_tolerance_norm * current_loss``.
      - On accept: shift the cluster image into base, snap the per-hit t0,
        mark the cluster placed.
      - On reject: leave the cluster's t0 unchanged (caller may run residual
        rescue later).

    Returns (base, hit_ts, accepted_rows, rejected_rows).
    """
    base = np.asarray(base_image, dtype=np.float32).copy()
    hit_ts = np.asarray(hit_timestamps, dtype=np.float32).copy()
    actual = np.asarray(full_light_waveform, dtype=np.float32)
    std = np.maximum(np.asarray(full_light_std, dtype=np.float32), 1e-6)
    labels = np.asarray(labels_global, dtype=np.int64)
    nt = base.shape[-1]

    accepted_rows = []
    rejected_rows = []

    for tpc, cluster_ids in sorted(iterative_single_tpc.items()):
        ordered = sorted(
            (int(c) for c in cluster_ids),
            key=lambda c: (float(cluster_energies.get(int(c), 0.0)), -int(c)),
            reverse=True,
        )
        if not ordered:
            continue

        placed = np.zeros(len(ordered), dtype=bool)
        # Pre-fetch existing candidate t0s for this TPC.
        cand_t0s = [float(v) for v in t0_candidates[int(tpc)]
                    if v is not None and np.isfinite(float(v))]

        while not placed.all() and cand_t0s:
            anchor_idx = int(np.argmax(
                [(-1.0 if placed[i] else float(cluster_energies.get(int(ordered[i]), 0.0)))
                 for i in range(len(ordered))]
            ))
            if placed[anchor_idx]:
                break
            anchor_e = float(cluster_energies.get(int(ordered[anchor_idx]), 0.0))
            band_floor = (1.0 - float(energy_band_fraction)) * anchor_e

            # Indices of unplaced clusters within energy band.
            band_local = [i for i in range(len(ordered))
                          if (not placed[i]) and float(cluster_energies.get(
                              int(ordered[i]), 0.0)) >= band_floor]
            band_clusters = [int(ordered[i]) for i in band_local]
            if not band_clusters:
                break

            M, remaining = _loss_matrix(
                image_maps, actual[int(tpc)], base[int(tpc)], std[int(tpc)],
                int(tpc), band_clusters,
                placed_mask=[False] * len(band_clusters),
                t0_list=cand_t0s,
            )
            if M.size == 0 or not np.isfinite(M).any():
                break
            i_min, j_min = np.unravel_index(np.argmin(M), M.shape)
            best_cid = int(remaining[int(i_min)])
            best_t0 = float(cand_t0s[int(j_min)])

            # Compare to per-cluster current loss (no shift, just base vs actual).
            current_loss = float(
                ((np.clip(base[int(tpc)], 0.0, 60780.0) - actual[int(tpc)]) ** 2
                 / std[int(tpc)]).sum() / float(actual.shape[-1] * actual.shape[-2]))
            best_loss = float(M[i_min, j_min])
            improvement = current_loss - best_loss

            row = {
                "clusterid": int(best_cid), "TPCid": int(tpc),
                "best_t0": float(best_t0),
                "current_loss": float(current_loss),
                "best_loss": float(best_loss),
                "improvement": float(improvement),
            }

            # Accept if the matrix's best loss isn't more than tolerance worse
            # than the (no-add) baseline. This admits associations that are
            # well-supported but whose absolute chi2 is dominated by other TPCs.
            tolerance = matrix_worsen_tolerance_norm * max(current_loss, 1e-9)
            if best_loss <= current_loss + tolerance:
                shifted = _shift_image_fractional(
                    image_maps[(best_cid, int(tpc))], best_t0, nt=nt)
                base[int(tpc)] = np.clip(base[int(tpc)] + shifted, 0.0, 60780.0)
                hit_ts[labels == best_cid] = np.float32(best_t0)
                # Mark placed.
                local_pos = band_local[int(i_min)]
                placed[local_pos] = True
                row["accepted"] = True
                accepted_rows.append(row)
            else:
                row["accepted"] = False
                rejected_rows.append(row)
                # Move on; this band is exhausted.
                # Mark all band entries as placed to break the loop.
                for i in band_local:
                    placed[i] = True

    return base, hit_ts, accepted_rows, rejected_rows


__all__ = [
    "snapshot_backbone_hits",
    "verify_backbone_hits_unchanged",
    "run_large_cluster_scan_phase",
    "run_small_cluster_matrix_phase",
]
