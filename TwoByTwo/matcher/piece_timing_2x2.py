#!/usr/bin/env python3
"""STEP 6.5 — per-piece timing of disintegrated families (sim, truth-scored).

Model:
  * v_in = c: every MAIN/SATELLITE hit is timed  t_vtx + |hit - vtx|/c
    (vertex = E-weighted centroid of MAIN pieces; t_vtx fit from the
    main-dominated channels' onsets).
  * DISPLACED pieces: ONE free time shift each (== v_out / TOF), found in
    closed form as the weighted median of (onset_j - d_j/c_LAr) over the
    piece-dominated channels. Validity: shift within the physical window
    [W_LO, W_HI] around the family flash and loss <= LOSS_MAX.
  * CORRECTOR (--corrector): a piece failing the in-window fit is checked
    against every OTHER family's flash time: bounded re-extraction of the
    piece's top channels' waveforms in [T'-10, T'+45] ticks, refit; if a
    consistent time is found there the piece is re-timed (mis-association
    fixed). No full-axis search anywhere.

Metrics vs truth (per hit, matching ticks):
  1. assignment ACCURACY : fraction with |reco - true| <= 5 (also 2, 1) ticks
  2. assignment PRECISION: std (and robust sigma) of reco-true for the
     correctly-assigned hits, in ns.
Arms: A = stage-F per-hit t0 baseline; B = 6.5 no corrector; C = 6.5 + corrector.
"""
import os, sys, glob, pickle, argparse
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np, h5py
sys.path.insert(0, "/pscratch/sd/y/yuxuan/2x2QLMatching/LightChannelMappingSpatially")
sys.path.insert(0, "/pscratch/sd/y/yuxuan/2x2QLMatching/QLMatching2x2")
sys.path.insert(0, "/pscratch/sd/y/yuxuan/2x2QLMatching/Charge2Light")
sys.path.insert(0, "/pscratch/sd/y/yuxuan/2x2QLMatching/StartTimeFit")
import data_2x2 as data, geometry_2x2 as geo
import pipeline_2x2 as pipe, light_model_2x2 as lm
import assoc_threshold_2x2 as at, truth_2x2 as truth
from disintegrate import disintegrate

PROD = ("/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/"
        "MiniRun6.4_1E19_RHC/MiniRun6.4_1E19_RHC.flow/FLOW/0000000")
NS, C, CLAR = 16.0, 29.9792458, 8.0
# waveform-sample <-> matching-tick frame: a flash at matching tick T has its
# pulse PEAK at sample T+105 (geometry_2x2 convention) and its CF onset at
# ~T+102 (empirical: ev28 pulses at samples 210/673 vs matching 108/571).
WF_SAMPLE_OFFSET = 102.0
AMP_MIN = 600.0
W_LO, W_HI = -160.0, 650.0      # ns, physical displaced-shift window vs family flash
LOSS_MAX = 4.0                   # ns, robust scatter acceptance
RE_LOSS_MAX = 6.0                # ns, coarser for re-extracted onsets
MIN_CH = 3

def wmedian(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    cs = np.cumsum(w); return float(x[np.searchsorted(cs, 0.5 * cs[-1])])

def rsig(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan

def piece_peaks(p, image_maps, tpc):
    """(48,) assembled predicted-light peak of a piece (deduced amplitude)."""
    pk = np.zeros(48)
    for cid, frac in p["light"].items():
        im = image_maps.get((int(cid), int(tpc)))
        if im is not None:
            pk += np.asarray(im).max(axis=1) * float(frac)
    return pk

def fit_shift(tau_ns, d_cm, w):
    """Closed-form free-shift fit: t* = wmedian(tau - d/CLAR); loss = MAD."""
    r = tau_ns - d_cm / CLAR
    t = wmedian(r, w)
    return t, rsig(r), r.size

def extract_onset(wf_ch, lo, hi):
    """Bounded-window constant-fraction onset [ticks] on one channel; NaN if no pulse."""
    seg = np.asarray(wf_ch[lo:hi], float)
    if seg.size < 8: return np.nan
    base = np.median(seg[:4]); s = seg - base
    pk = s.max()
    if pk < 300: return np.nan
    thr = 0.25 * pk
    idx = np.flatnonzero(s >= thr)
    if idx.size == 0: return np.nan
    i = idx[0]
    if i == 0: return float(lo)
    f = (thr - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)   # sub-tick interp
    return float(lo + i - 1 + f)

def run(args):
    m = lm.load_light_model("sim", device="cuda")
    files = sorted(glob.glob(os.path.join(PROD, "*.FLOW.hdf5")))[:args.nfiles]
    rows = []          # per-hit: ev, fam, tag, dA, dB, dC (ticks)
    piece_log = []     # per displaced piece: fit outcome
    corr_stats = dict(pieces=0, failed=0, reassigned=0, unresolved=0)
    for fp in files:
        h5 = h5py.File(fp, "r")
        sxyz = geo.build_sipm_positions(h5); tt = truth.TruthTables(h5)
        hr = np.asarray(h5["charge/events/ref/charge/calib_prompt_hits/ref"][()], np.int64)
        for ev in np.unique(hr[:, 0])[:args.nev]:
            try:
                r = pipe.run_pipeline_for_event(h5, int(ev), light_model=m)
            except Exception:
                continue
            if r is None: continue
            e = r["event"]; reco = np.full(e.Eset.size, np.nan); clog = []
            try:
                at.threshold_family_association(hit_t0=reco, hit_refine="channel6",
                  sipm_xyz=sxyz, chan_log=clog, light_model=m, labels=r["labels"],
                  xset=e.xset, yset=e.yset, zset=e.zset, Eset=e.Eset,
                  hitTPCid=e.hitTPCid, cluster_energies=r["cluster_energies"],
                  image_maps=r["image_maps"], labels_noisy=r["labels_noisy"],
                  full_wvfm=e.fullLightWaveform, full_var=e.fullLightVar,
                  flash_seeds=e.flash_seeds)
            except Exception:
                continue
            TH = np.asarray(tt.per_hit_truth_t0(r["hit_refs"], int(ev)), float)  # ticks
            xyz = np.column_stack([e.xset, e.yset, e.zset])
            labels = np.asarray(r["labels"]); tpcid = np.asarray(e.hitTPCid)
            Es = np.asarray(e.Eset, float)
            fams = [g for g in clog if len(g["hits"]) >= 80]
            # corrector lookup table: candidate flash times (ns) = reconstructed
            # family flashes (any TPC) + the LIGHT-LEVEL flash seeds of each TPC
            # (charge-poor pile-up has a seed but never becomes a family)
            cand_times = []
            for g in clog:
                hhc = np.asarray(g["hits"]); rc = reco[hhc]; rc = rc[np.isfinite(rc)]
                if rc.size >= 20:
                    cand_times.append((int(g["tpc"]), float(np.median(rc) * NS)))
            seed_times = {}
            try:
                for tp in range(8):
                    seed_times[tp] = [float(s) * NS for s in e.flash_seeds[tp]]
            except Exception:
                seed_times = {tp: [] for tp in range(8)}
            for fi, g in enumerate(fams):
                hh = np.asarray(g["hits"]); tpc = int(g["tpc"])
                rfam = reco[hh]; rfam = rfam[np.isfinite(rfam)]
                if rfam.size < 20: continue
                t_fam = float(np.median(rfam) * NS)
                amp = np.asarray(g["amps"], float)
                tau = np.asarray(g["tch"], float) * NS
                chans = np.asarray(g["chans"], int)
                ok_ch = (amp >= AMP_MIN) & np.isfinite(tau)
                pieces, meta = disintegrate(labels[hh], xyz[hh], Es[hh], tpcid[hh])
                if meta is None or not pieces: continue
                fam_pk = np.zeros(48)
                for p in pieces: fam_pk += piece_peaks(p, r["image_maps"], tpc)
                # ---------- vertex fit (v_in = c) ----------
                mainp = [p for p in pieces if p["tag"].startswith(("MAIN", "SATELLITE"))]
                dispp = [p for p in pieces if p["tag"].startswith("DISPLACED")]
                if not mainp: continue
                mh = np.concatenate([p["hits"] for p in mainp])
                wv = np.maximum(Es[hh][mh], 1e-6)
                vtx = np.average(xyz[hh][mh], axis=0, weights=wv)
                main_pk = np.zeros(48)
                for p in mainp: main_pk += piece_peaks(p, r["image_maps"], tpc)
                pcs = np.vstack([p["centroid"] for p in mainp])
                L1 = np.linalg.norm(pcs - vtx, axis=1) / C          # ns per piece
                sel = ok_ch & (main_pk[chans] > 0.6 * fam_pk[chans]) & (main_pk[chans] > 200)
                if sel.sum() < MIN_CH:
                    sel = ok_ch
                if sel.sum() < 1:
                    continue                        # no usable channels at all
                dj = np.linalg.norm(pcs[:, None, :] - sxyz[tpc][chans[sel]][None, :, :],
                                    axis=2)                          # (npiece, nch)
                arr = (L1[:, None] + dj / CLAR).min(axis=0)
                t_vtx = wmedian(tau[sel] - arr, np.sqrt(amp[sel]))
                # per-hit times, arms B and C start identical
                tB = np.full(hh.size, np.nan); tag_of = np.full(hh.size, "", dtype=object)
                for p in mainp:
                    d_h = np.linalg.norm(xyz[hh][p["hits"]] - vtx, axis=1)
                    tB[p["hits"]] = (t_vtx + d_h / C) / NS
                    for ii in p["hits"]: tag_of[ii] = "MAIN"
                tC = tB.copy()
                # ---------- displaced pieces: free shift ----------
                for p in dispp:
                    corr_stats["pieces"] += 1
                    ppk = piece_peaks(p, r["image_maps"], tpc)
                    psel = ok_ch & (ppk[chans] > 0.3 * fam_pk[chans]) & (ppk[chans] > 50)
                    stat = "fit"
                    t_p = np.nan
                    if psel.sum() >= MIN_CH:
                        d_p = np.linalg.norm(sxyz[tpc][chans[psel]] - p["centroid"], axis=1)
                        t_p, loss, nch = fit_shift(tau[psel], d_p, np.sqrt(amp[psel]))
                        good = (loss <= LOSS_MAX) and (W_LO <= t_p - t_fam <= W_HI)
                    else:
                        loss, nch, good = np.nan, int(psel.sum()), False
                    if not good:
                        stat = "failed"; corr_stats["failed"] += 1
                    # arm B: accept fit if good, else fall back to family flash
                    tB[p["hits"]] = (t_p if good else t_fam) / NS
                    tC[p["hits"]] = (t_p if good else t_fam) / NS
                    for ii in p["hits"]: tag_of[ii] = "DISPLACED"
                    # arm C corrector: bounded re-extraction vs other families
                    if not good and args.corrector:
                        top = [j for j in np.argsort(-ppk)[:8] if ppk[j] > 30]
                        # candidates: this TPC's flash seeds + all family flashes
                        cands = [(tpc, T) for T in seed_times.get(tpc, [])] + cand_times
                        best = None
                        for ctpc, T2 in cands:
                            if abs(T2 - t_fam) < 3 * NS: continue   # itself
                            base_s = int(T2 / NS + WF_SAMPLE_OFFSET)
                            lo = max(0, base_s - 10); hi = min(1000, base_s + 45)
                            ons, dd = [], []
                            for j in top:
                                o = extract_onset(e.fullLightWaveform[tpc][j], lo, hi)
                                if np.isfinite(o):
                                    ons.append((o - WF_SAMPLE_OFFSET) * NS)  # back to matching frame
                                    dd.append(np.linalg.norm(sxyz[tpc][j] - p["centroid"]))
                            if len(ons) < MIN_CH: continue
                            t2, l2, n2 = fit_shift(np.asarray(ons), np.asarray(dd),
                                                   np.ones(len(ons)))
                            if l2 <= RE_LOSS_MAX and W_LO <= t2 - T2 <= W_HI:
                                if best is None or l2 < best[1]:
                                    best = (t2, l2, T2)
                        if best is not None:
                            tC[p["hits"]] = best[0] / NS
                            stat = "reassigned"; corr_stats["reassigned"] += 1
                            # log the CORRECTED fit; TOF is vs the matched
                            # candidate flash T2 (its new family), not t_fam
                            t_p, loss = best[0], best[1]
                            tof_log = best[0] - best[2]
                        else:
                            stat = "unresolved"; corr_stats["unresolved"] += 1
                            tof_log = (t_p - t_fam) if np.isfinite(t_p) else np.nan
                    else:
                        tof_log = (t_p - t_fam) if np.isfinite(t_p) else np.nan
                    piece_log.append(dict(ev=int(ev), fam=fi, tpc=tpc, E=p["E"],
                        nch=int(nch) if np.isfinite(nch) else 0,
                        t_fam=t_fam, t_fit=t_p, loss=float(loss) if np.isfinite(loss) else np.nan,
                        status=stat, tof=tof_log,
                        dist=float(np.linalg.norm(p["centroid"] - vtx))))
                # ---------- score vs truth ----------
                THf = TH[hh]
                for ii in range(hh.size):
                    if not np.isfinite(THf[ii]) or not tag_of[ii]: continue
                    dA = reco[hh][ii] - THf[ii] if np.isfinite(reco[hh][ii]) else np.nan
                    rows.append((int(ev), fi, tag_of[ii],
                                 float(dA) if np.isfinite(dA) else np.nan,
                                 float(tB[ii] - THf[ii]),
                                 float(tC[ii] - THf[ii])))
            data.clear_cache()
        h5.close()
        print(f"[{os.path.basename(fp)}] cum hits {len(rows)}, pieces {corr_stats['pieces']}",
              flush=True)
    return rows, piece_log, corr_stats

def report(rows, piece_log, corr_stats, out_prefix):
    tags = np.array([r[2] for r in rows])
    D = {"A stage-F baseline": np.array([r[3] for r in rows]),
         "B 6.5 no corrector": np.array([r[4] for r in rows]),
         "C 6.5 + corrector ": np.array([r[5] for r in rows])}
    def line(d, mask, label):
        d = d[mask]; d = d[np.isfinite(d)]
        if d.size == 0: return f"    {label:12s}  (no hits)"
        acc5 = np.mean(np.abs(d) <= 5); acc2 = np.mean(np.abs(d) <= 2)
        acc1 = np.mean(np.abs(d) <= 1)
        good = d[np.abs(d) <= 5] * NS
        return (f"    {label:12s} n={d.size:6d}  acc(5t)={acc5:.3f} acc(2t)={acc2:.3f} "
                f"acc(1t)={acc1:.3f}  prec(std)={np.std(good):5.2f}ns rob={rsig(good):5.2f}ns")
    print("\n================ STEP 6.5 METRICS (vs truth, ticks) ================")
    for arm, d in D.items():
        print(f"  {arm}:")
        print(line(d, np.ones(d.size, bool), "ALL"))
        print(line(d, tags == "MAIN", "MAIN"))
        print(line(d, tags == "DISPLACED", "DISPLACED"))
    print(f"\n  corrector: {corr_stats}")
    disp = [p for p in piece_log if p["status"] != "fit"]
    print(f"  displaced pieces: {len(piece_log)} total; non-clean: {len(disp)}")
    for p in piece_log[:0]: pass
    pickle.dump(dict(rows=rows, piece_log=piece_log, corr=corr_stats),
                open(out_prefix + ".pkl", "wb"))
    print(f"  saved {out_prefix}.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=2)
    ap.add_argument("--nev", type=int, default=40)
    ap.add_argument("--corrector", action="store_true")
    ap.add_argument("--out", default="/pscratch/sd/y/yuxuan/2x2QLMatching/StartTimeFit/step6_5_results")
    args = ap.parse_args()
    rows, piece_log, corr_stats = run(args)
    report(rows, piece_log, corr_stats, args.out)
