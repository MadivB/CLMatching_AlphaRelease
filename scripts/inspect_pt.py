"""Quick-inspect a v_alpha_test per-file .pt.

Prints schema, sentinel counts, coverage fractions, processed event count.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def inspect(path: str) -> None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    print(f"=== {path} ===")
    print(f"version            : {d.get('version')}")
    print(f"input_file         : {d.get('input_file')}")
    print(f"n_calib_hits       : {d.get('n_calib_hits')}")
    print(f"n_assigned         : {d.get('n_assigned')}")
    print(f"n_unassigned       : {d.get('n_unassigned')}")
    print(f"events processed   : {len(d.get('processed_event_ids', []))}"
          f" of {len(d.get('all_event_ids', []))}")
    print(f"failed events      : {len(d.get('failed_events', []))}")
    print(f"event shards       : {d.get('n_event_shards')}")
    print(f"aggregator elapsed : {d.get('aggregator_elapsed_s', 0):.1f}s")
    print()
    print("--- arrays ---")
    for k, v in d.items():
        if hasattr(v, "shape"):
            arr = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            extra = ""
            if k == "calib_hit_t0_reco":
                n_assigned = int(np.isfinite(arr).sum() & (arr >= 0).sum())
                n_assigned = int(((arr >= 0) & np.isfinite(arr)).sum())
                extra = (
                    f"  finite&>=0: {n_assigned}/{arr.size} "
                    f"({100*n_assigned/max(arr.size,1):.2f}%)"
                )
            if k == "prompt_hit_t_cluster_id":
                n_assigned = int((arr != -1).sum())
                extra = (
                    f"  != -1: {n_assigned}/{arr.size} "
                    f"({100*n_assigned/max(arr.size,1):.2f}%)"
                )
            print(f"  {k:30s} shape={tuple(arr.shape)} dtype={arr.dtype}{extra}")
    print()
    if d.get("event_summaries"):
        print("--- first 3 event summaries (truncated) ---")
        for s in d["event_summaries"][:3]:
            print(
                f"  ev{int(s.get('event_id', -1)):04d} hits={s.get('n_hits')} "
                f"elapsed={s.get('elapsed_s', 0):.1f}s "
                f"summary_keys={sorted((s.get('summary') or {}).keys())[:5]}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    for p in args.paths:
        inspect(p)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
