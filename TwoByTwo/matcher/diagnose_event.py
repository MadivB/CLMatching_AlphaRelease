#!/usr/bin/env python3
"""
Per-stage diagnostic for ONE event of the 2x2 charge-light matcher.

Shows, stage by stage:
  0. flash inventory (real flashes per TPC from light/flash) + truth t0s present
  1. clustering           — tracks vs DBSCAN blobs, per-cluster E / nhits / TPCs
  2. Stage 4 (tracks)     — t0 scan result vs truth, per track
  3. Stage 5 (blobs)      — for each blob the candidate t0s actually considered
                            with their matched-filter score AND chi2 (the
                            "association matrix"), which one was chosen, and the
                            truth t0 it SHOULD have gone to
  4. Phase 2.5 (rescue)   — what each sub-step changed
  5. final per-cluster table with residual (ns) + correct/WRONG flag

Run: python diagnose_event.py --in-h5 <FLOW> --event 14 --mode sim
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py
import geometry_2x2 as geo
import data_2x2 as data
import matching_2x2 as match
import light_model_2x2 as lm
import truth_2x2 as truth
import phase25_2x2 as p25
from pipeline_2x2 import cluster_charge, build_noisy_labels

NS = geo.NS_PER_TICK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5", required=True)
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--mode", default="sim")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--small-energy-mev", type=float, default=8.0)
    ap.add_argument("--phase25", action="store_true")
    args = ap.parse_args()

    model = lm.load_light_model(args.mode, device=args.device)
    h5 = h5py.File(args.in_h5, "r")
    tt = truth.TruthTables(h5)
    ev = data.load_event(h5, args.event)
    th_hit = tt.per_hit_truth_t0(ev.hit_refs, args.event)
    x, y, z, E, tpcid = ev.xset, ev.yset, ev.zset, ev.Eset, ev.hitTPCid
    print(f"\n{'='*78}\nEVENT {args.event}: {x.size} hits, light_event {ev.light_event_id}\n{'='*78}")

    # ---- 0. flash inventory ----
    print("\n[0] FLASH INVENTORY (real flashes -> t0 seeds per TPC):")
    for t in range(geo.N_TPCS):
        if ev.flash_seeds[t]:
            print(f"   TPC{t}: flash t0 seeds = {[round(s,1) for s in ev.flash_seeds[t]]}")

    # ---- 1. clustering ----
    labels, noise_list, track_id_max = cluster_charge(x, y, z, E)
    labels_noisy = build_noisy_labels(labels, noise_list, track_id_max)
    cluster_energies = {int(c): float(E[labels == c].sum())
                        for c in np.unique(labels) if c >= 0}

    def cluster_truth(cid):
        m = labels == cid
        thc, we = th_hit[m], E[m]
        ok = np.isfinite(thc)
        return float((thc[ok] * we[ok]).sum() / we[ok].sum()) if ok.any() and we[ok].sum() > 0 else np.nan

    n_blob = int(labels.max()) - track_id_max
    print(f"\n[1] CLUSTERING: {track_id_max+1} track(s) [ids 0..{track_id_max}], "
          f"{n_blob} DBSCAN blob(s).  noise hits adoptable: {len(noise_list)}")
    print(f"   {'cid':>4} {'type':>6} {'nhit':>5} {'E(MeV)':>8} {'TPCs':>10} {'truth_t0':>9}")
    for cid in np.unique(labels):
        if cid < 0:
            continue
        m = labels == cid
        typ = "track" if cid <= track_id_max else "blob"
        tpcs = sorted(set(int(t) for t in tpcid[m]))
        print(f"   {int(cid):>4} {typ:>6} {int(m.sum()):>5} {cluster_energies[int(cid)]:>8.2f} "
              f"{str(tpcs):>10} {cluster_truth(int(cid)):>9.1f}")

    # ---- predict + state ----
    image_maps, _ = model.predict_image_maps(x, y, z, E, tpcid, labels)
    image_maps_noisy, _ = model.predict_image_maps(x, y, z, E, tpcid, labels_noisy)
    cluster_to_tpcs, tpc_to_clusters = {}, {}
    for (cid, tpc) in image_maps.keys():
        cluster_to_tpcs.setdefault(int(cid), []).append(int(tpc))
    for cid, tpcs in cluster_to_tpcs.items():
        if cid > track_id_max:
            for t in tpcs:
                tpc_to_clusters.setdefault(int(t), []).append(int(cid))
    base = np.zeros((8, 48, 1000), np.float32)
    hit_t0 = np.full(x.size, -1.0, np.float32)
    cand = [[] for _ in range(8)]

    # ---- 2. tracks ----
    print(f"\n[2] STAGE 4 — TRACK t0 (full scan over [0,700], peak-snap):")
    for cid in range(track_id_max + 1):
        if cid not in cluster_to_tpcs:
            continue
        tpcs = sorted(cluster_to_tpcs[cid])
        img = np.concatenate([image_maps[(cid, t)] for t in tpcs], 0)
        b = np.concatenate([base[t] for t in tpcs], 0)
        a = np.concatenate([ev.fullLightWaveform[t] for t in tpcs], 0)
        v = np.concatenate([ev.fullLightVar[t] for t in tpcs], 0)
        shifts, errs = match.full_integer_scan(img, b, a, v, 700)
        free_t0 = float(shifts[int(np.argmin(errs))])
        # candidate flash seeds across the track's TPCs (the real association set)
        seedset = match._merge_close(
            [s for t in tpcs for s in ev.flash_seeds[t]], 5)
        cand_scores = [(c, match.matched_filter_at(img, b, a, v, c, None),
                        match.score_at(img, b, a, v, c, None)) for c in seedset]
        t0s = match.peak_snap_t0(a, b, free_t0, 5)
        tr = cluster_truth(cid)
        res_ns = (tr - t0s) * NS
        flag = "OK" if abs(res_ns) <= 160 else "*** WRONG ***"
        print(f"   track {cid} TPCs{tpcs} (nhit={int(np.sum(labels==cid))}): "
              f"free_scan_t0={t0s:.1f}  truth={tr:.1f}  resid={res_ns:+.0f}ns  {flag}")
        print(f"      flash-seed candidates (t0: matched-filter cos / chi2):")
        for c, cs, ch in sorted(cand_scores, key=lambda r: -r[1]):
            mark = " <-- TRUTH flash" if abs(c - tr) <= 8 else ""
            print(f"        t0={c:6.1f}: cos={cs:.3f}  chi2={ch:.3g}{mark}")
        hit_t0[labels == cid] = t0s
        for t in tpcs:
            cand[t].append(t0s)
        placed = match.shift_frac(img, t0s).reshape(len(tpcs), 48, -1)
        for i, t in enumerate(tpcs):
            base[t] = np.clip(base[t] + placed[i], None, geo.ADC_CLIP)

    # ---- 3. blobs (association matrix) ----
    print(f"\n[3] STAGE 5 — BLOB t0 (per cluster: candidate t0 -> "
          f"matched-filter cos / chi2; chosen vs truth):")
    for tpc in sorted(tpc_to_clusters):
        clusters = sorted((int(c) for c in tpc_to_clusters[tpc]),
                          key=lambda c: -cluster_energies.get(c, 0))
        match._ensure_seeds(cand[tpc], ev.flash_seeds[tpc], 5)
        for cid in clusters:
            if (cid, tpc) not in image_maps:
                continue
            img = image_maps[(cid, tpc)]
            e = cluster_energies.get(cid, 0.0)
            small = e < args.small_energy_mev
            sup = match.support_channels(img) if small else None
            cl = list(cand[tpc])
            tr = cluster_truth(cid)
            rows = []
            for c in cl:
                cos = match.matched_filter_at(img, base[tpc], ev.fullLightWaveform[tpc],
                                              ev.fullLightVar[tpc], c, sup)
                chi = match.score_at(img, base[tpc], ev.fullLightWaveform[tpc],
                                     ev.fullLightVar[tpc], c, sup)
                rows.append((c, cos, chi))
            # decide as the matcher does
            if small and cl:
                pk = float(np.clip(img if sup is None else img[sup], 0, None).max())
                if pk < 40.0:
                    chosen = max(cl, key=lambda c: match.observed_brightness_at(
                        ev.fullLightWaveform[tpc], sup, c))
                    how = "brightest(faint)"
                else:
                    chosen = max(rows, key=lambda r: r[1])[0]
                    how = "max matched-filter"
            else:
                shifts, errs = match.full_integer_scan(
                    img, base[tpc], ev.fullLightWaveform[tpc], ev.fullLightVar[tpc], 700, sup)
                chosen = float(shifts[int(np.argmin(errs))])
                how = "free chi2 scan"
            res_ns = (tr - chosen) * NS
            flag = "OK" if abs(res_ns) <= 160 else "*** WRONG ***"
            print(f"   blob {cid} TPC{tpc} E={e:.2f} small={small} via {how}:")
            print(f"      chosen_t0={chosen:.1f}  truth={tr:.1f}  resid={res_ns:+.0f}ns  {flag}")
            srt = sorted(rows, key=lambda r: -r[1])[:6]
            print(f"      candidates (t0: cos / chi2):  " +
                  "  ".join(f"{r[0]:.0f}:{r[1]:.2f}/{r[2]:.2g}" for r in srt))
            # place it (greedy) so the base evolves like the real run
            hit_t0[labels == cid] = chosen
            base[tpc] = np.clip(base[tpc] + match.shift_frac(img, chosen), None, geo.ADC_CLIP)

    # ---- 4/5 efficiency before/after rescue ----
    def eff(ht):
        asg = np.isfinite(th_hit) & np.isfinite(ht) & (ht >= 0) & (E > 0)
        d = np.abs(th_hit - ht) * NS
        return 100 * E[asg][d[asg] <= 160].sum() / max(E[asg].sum(), 1e-9)

    print(f"\n[4] EVENT EFFICIENCY (energy within 160 ns):")
    print(f"   after Stage 5 (no rescue): {eff(hit_t0):.1f}%")
    if args.phase25:
        ns = {"baseImage": base.copy(), "hit_timestamps": hit_t0.copy(),
              "fullLightWaveform": ev.fullLightWaveform, "fullLightStd": ev.fullLightVar,
              "labels_global": labels, "hitTPCid": tpcid.astype(np.int32),
              "xset": x, "yset": y, "zset": z, "Eset": E,
              "imageMaps": image_maps, "cluster_to_tpcs": cluster_to_tpcs,
              "t0Candidates": cand, "track_shower_labels": list(range(track_id_max + 1)),
              "saturated_channel_cache": {"veto_mask": {t: (np.sum(
                  ev.fullLightWaveform[t] > 60700.0, axis=1) > 6) for t in range(8)}}}
        cfg = p25.Trial2Config(verbose=True)
        r = p25.run_trial2_combined_rescue(
            namespace=ns, predict_family_image_fn=model.predict_single_image, cfg=cfg)
        print(f"   phase25: (a) accepted {len(r['large_flash_grid']['accepted_rows'])}"
              f", (b) moved {len(r['spatial']['spatial_moves'])}"
              f", (c) light moves {len(r['light']['light_moves'])}")
        print(f"   after phase 2.5         : {eff(np.asarray(ns['hit_timestamps'])):.1f}%")


if __name__ == "__main__":
    main()
