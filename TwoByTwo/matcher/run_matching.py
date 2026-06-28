#!/usr/bin/env python3
"""
Driver / demo for the 2x2 charge-light matcher (vAlpha port).

Runs the matcher on one or more charge events of a FLOW file, optionally
validates against mc_truth, and prints an energy-binned success report that
highlights the small/low-energy clusters (the 2x2 priority).

Examples
--------
# one event, sim model, with truth check
python run_matching.py --in-h5 <FLOW.hdf5> --events 5 --mode sim

# first 30 light-matched events, aggregate truth metrics
python run_matching.py --in-h5 <FLOW.hdf5> --events auto --max-events 30 --mode sim

# data file, data model, no truth, dump (hit_ref, t0)
python run_matching.py --in-h5 <FLOW.hdf5> --events auto --mode data \
    --dead-yaml ../Charge2Light/dead_channels_2x2.yaml --no-truth --out matched.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py

import geometry_2x2 as geo
import data_2x2 as data
import pipeline_2x2 as pipe
import light_model_2x2 as lm
import truth_2x2 as truth


DEFAULT_SIM_FILE = ("/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/"
                    "MiniRun6.4_1E19_RHC/MiniRun6.4_1E19_RHC.flow/FLOW/0000000/"
                    "MiniRun6.4_1E19_RHC.flow.0000541.FLOW.hdf5")


def events_with_light(h5, max_events=None):
    ref = np.asarray(h5["charge/events/ref/light/events/ref"][()], dtype=np.int64)
    evs = np.unique(ref[:, 0])
    if max_events:
        evs = evs[:int(max_events)]
    return evs.tolist()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-h5", default=DEFAULT_SIM_FILE)
    ap.add_argument("--events", nargs="+", default=["auto"],
                    help="event ids, or 'auto' for all light-matched events")
    ap.add_argument("--max-events", type=int, default=10)
    ap.add_argument("--mode", choices=["sim", "data"], default="sim")
    ap.add_argument("--ckpt", default=None, help="override perceiver checkpoint")
    ap.add_argument("--dead-yaml", default="")
    ap.add_argument("--device", default=None, help="cpu|cuda (default auto)")
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="matching-tick tolerance for a 'correct' truth match")
    ap.add_argument("--small-energy-mev", type=float, default=8.0)
    ap.add_argument("--no-truth", action="store_true")
    ap.add_argument("--out", default="", help="optional .npy dump of (hit_ref, t0)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"[load] perceiver ({args.mode}) ...")
    model = lm.load_light_model(args.mode, checkpoint=args.ckpt, device=args.device)
    print(f"[load] checkpoint OK, out_dim={model.meta.get('out_dim')}")

    h5 = h5py.File(args.in_h5, "r")
    if args.events == ["auto"] or args.events == "auto":
        ev_ids = events_with_light(h5, args.max_events)
    else:
        ev_ids = [int(e) for e in args.events][: args.max_events]
    print(f"[run] {len(ev_ids)} event(s) from {os.path.basename(args.in_h5)}")

    tt = None if args.no_truth else truth.TruthTables(h5)
    all_rows = []
    all_t0, all_refs = [], []

    for ev in ev_ids:
        res = pipe.run_pipeline_for_event(
            h5, ev, light_model=model, dead_yaml=args.dead_yaml,
            small_energy_mev=args.small_energy_mev, verbose=args.verbose)
        if res is None:
            continue
        all_t0.append(np.asarray(res["hit_timestamps"], np.float32))
        all_refs.append(np.asarray(res["hit_refs"], np.int64))
        if tt is not None:
            ev_eval = truth.evaluate_clusters(
                tt, ev_id=int(ev), hit_refs=res["hit_refs"], labels=res["labels"],
                hit_t0=res["hit_timestamps"], Eset=res["event"].Eset,
                hitTPCid=res["event"].hitTPCid, tolerance_ticks=args.tolerance)
            all_rows.extend(ev_eval["rows"])

    if args.out and all_t0:
        refs = np.concatenate(all_refs)
        t0 = np.concatenate(all_t0)
        np.save(args.out, np.stack([refs.astype(np.float64), t0.astype(np.float64)], 1))
        print(f"[out] wrote {refs.size} (hit_ref, t0) rows -> {args.out}")

    if tt is not None and all_rows:
        agg = {"rows": all_rows, "tolerance_ticks": args.tolerance,
               "summary": truth._summarize(all_rows, args.tolerance)}
        print()
        print(truth.format_report(agg))


if __name__ == "__main__":
    main()
