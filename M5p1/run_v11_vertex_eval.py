from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

import QLmatchingND as QL
from truth_plotting import extract_truth_for_selected_hits

DEFAULT_DATA_FILE = (
    "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/"
    "run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/"
    "MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000095.FLOW.hdf5"
)
DEFAULT_OUTPUT = str(THIS_DIR / "v11_vertex_success_eval_events1_10.pt")


def _segment_cache_key(h5: h5py.File) -> tuple[str, str]:
    return (str(h5.filename), "mc_truth/segments/data")


def extract_truth_vertex_and_t0_for_hits(
    h5: h5py.File,
    hit_ids: np.ndarray,
    *,
    convert_to_matching_ticks: bool = True,
    event_id: int | None = None,
) -> dict[str, np.ndarray]:
    hit_ids = np.asarray(hit_ids, dtype=np.int64).reshape(-1)
    if hit_ids.size == 0:
        raise ValueError("hit_ids is empty")

    hits_full = h5["charge/calib_prompt_hits/data"]
    hits_evt = hits_full[hit_ids]
    hit_energy = np.asarray(hits_evt["E"], dtype=np.float32)

    truth = extract_truth_for_selected_hits(
        h5,
        hit_ids,
        convert_t0_to_matching_ticks=bool(convert_to_matching_ticks),
        event_id=event_id,
    )

    return {
        "hit_ids": hit_ids,
        "hit_energy_mev": hit_energy,
        "best_segment_id": np.asarray(truth["best_segment_id"], dtype=np.int64),
        "best_fraction": np.asarray(truth["best_fraction"], dtype=np.float32),
        "truth_t0": np.asarray(truth["true_t0_rel"], dtype=np.float64),
        "vertex_id": np.asarray(truth["vertex_id"], dtype=np.int64),
    }


def build_vertex_rows(
    *,
    eventid: int,
    reco_t0: np.ndarray,
    truth_info: dict[str, np.ndarray],
    correct_ticks: float = 10.0,
) -> list[dict[str, Any]]:
    reco_t0 = np.asarray(reco_t0, dtype=np.float64)
    truth_t0 = np.asarray(truth_info["truth_t0"], dtype=np.float64)
    vertex_id = np.asarray(truth_info["vertex_id"], dtype=np.int64)
    hit_energy = np.asarray(truth_info["hit_energy_mev"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    valid_vertices = sorted(int(v) for v in np.unique(vertex_id) if int(v) >= 0)
    for vid in valid_vertices:
        mask = vertex_id == int(vid)
        truth_mask = mask & np.isfinite(truth_t0) & np.isfinite(hit_energy) & (hit_energy > 0.0)
        if not np.any(truth_mask):
            continue

        correct_mask = (
            truth_mask
            & np.isfinite(reco_t0)
            & (np.abs(reco_t0 - truth_t0) <= float(correct_ticks))
        )

        total_energy = float(np.sum(hit_energy[truth_mask]))
        correct_energy = float(np.sum(hit_energy[correct_mask]))
        n_truth_hits = int(np.count_nonzero(truth_mask))
        n_correct_hits = int(np.count_nonzero(correct_mask))

        rows.append(
            {
                "eventid": int(eventid),
                "vertex_id": int(vid),
                "n_hits_truth": n_truth_hits,
                "n_hits_correct": n_correct_hits,
                "total_energy_mev": total_energy,
                "correct_energy_mev": correct_energy,
                "success_fraction_energy": float(correct_energy / max(total_energy, 1e-12)),
                "success_fraction_hits": float(n_correct_hits / max(n_truth_hits, 1)),
            }
        )

    return rows


def evaluate_events(
    h5: h5py.File,
    *,
    event_ids: np.ndarray,
    lam: float = 1.2,
    correct_ticks: float = 10.0,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    event_payloads: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for ev_id in [int(v) for v in event_ids]:
        t0 = time.time()
        print()
        print(f"=== Event {ev_id} ===", flush=True)
        try:
            reco_t0, hit_ids = QL.run(h5, eventid=int(ev_id), lam=float(lam), verbose=False)
            truth_info = extract_truth_vertex_and_t0_for_hits(
                h5,
                hit_ids,
                convert_to_matching_ticks=True,
                event_id=int(ev_id),
            )
            rows = build_vertex_rows(
                eventid=int(ev_id),
                reco_t0=reco_t0,
                truth_info=truth_info,
                correct_ticks=float(correct_ticks),
            )
            all_rows.extend(rows)

            truth_t0 = np.asarray(truth_info["truth_t0"], dtype=np.float64)
            hit_energy = np.asarray(truth_info["hit_energy_mev"], dtype=np.float64)
            finite_truth = np.isfinite(truth_t0) & np.isfinite(hit_energy) & (hit_energy > 0.0)
            correct_mask = finite_truth & np.isfinite(reco_t0) & (np.abs(np.asarray(reco_t0, dtype=np.float64) - truth_t0) <= float(correct_ticks))
            total_energy = float(np.sum(hit_energy[finite_truth]))
            correct_energy = float(np.sum(hit_energy[correct_mask]))
            success_energy = float(correct_energy / max(total_energy, 1e-12))

            event_summary = {
                "eventid": int(ev_id),
                "n_vertices": int(len(rows)),
                "truth_energy_mev": total_energy,
                "correct_energy_mev": correct_energy,
                "success_fraction_energy": success_energy,
                "median_vertex_success": float(np.median([row["success_fraction_energy"] for row in rows])) if rows else np.nan,
                "runtime_sec": float(time.time() - t0),
            }
            event_summaries.append(event_summary)

            event_payloads[int(ev_id)] = {
                "hit_ids": np.asarray(hit_ids, dtype=np.int64),
                "reco_t0": np.asarray(reco_t0, dtype=np.float32),
                "truth_t0": np.asarray(truth_info["truth_t0"], dtype=np.float64),
                "vertex_id": np.asarray(truth_info["vertex_id"], dtype=np.int64),
                "hit_energy_mev": np.asarray(truth_info["hit_energy_mev"], dtype=np.float32),
                "best_segment_id": np.asarray(truth_info["best_segment_id"], dtype=np.int64),
                "best_fraction": np.asarray(truth_info["best_fraction"], dtype=np.float32),
                "correct_mask": np.asarray(correct_mask, dtype=bool),
                "vertex_rows": rows,
            }

            print(
                f"Event {ev_id}: vertices={event_summary['n_vertices']} | "
                f"truth-energy correct={100.0 * event_summary['success_fraction_energy']:.2f}% | "
                f"runtime={event_summary['runtime_sec']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "eventid": int(ev_id),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Event {ev_id} failed: {exc}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "rows": all_rows,
        "event_summaries": event_summaries,
        "event_payloads": event_payloads,
        "failures": failures,
        "settings": {
            "event_ids": [int(v) for v in event_ids],
            "correct_ticks": float(correct_ticks),
            "lam": float(lam),
            "grouping": "global_truth_vertex_id",
            "truth_t0_convention": "event-normalized matching ticks",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v11 global vertex-level reconstruction success.")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--event-ids", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--lam", type=float, default=1.2)
    parser.add_argument("--correct-ticks", type=float, default=10.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data_file, "r") as h5:
        results = evaluate_events(
            h5,
            event_ids=np.asarray(args.event_ids, dtype=np.int64),
            lam=float(args.lam),
            correct_ticks=float(args.correct_ticks),
        )

    torch.save(results, output_path)
    print()
    print(f"Saved results to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
