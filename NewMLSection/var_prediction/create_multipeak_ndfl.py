#!/usr/bin/env python3
"""
create_multipeak_ndfl.py
========================
Step 2 of the ND-full variance prediction pipeline.

Reads single-flash (actual, predicted) waveform zarrs produced by
process_data_ndfl.py and overlays multiple events (with random time shifts)
from the same TPC to create realistic multi-peak training samples.

TPC grouping is inferred from, in order:
  1. charge_tpc_ids
  2. tpc_ids
  3. event_ids[:, 1]
  4. light_tpc_ids

Older zarrs that only contain `event_ids` or `light_tpc_ids` remain usable,
but a missing TPC field is now treated as an error instead of silently
collapsing everything into a fake TPC `0`.

Output zarr schema:
  inputs  : (N, 120, T)  float32  — multi-peak actual waveform
  targets : (N, 120, T)  float32  — multi-peak predicted waveform
  tpc_ids : (N,)         int32    — grouping TPC id used during synthesis
  source_counts  : (N,)          int16  — number of overlaid source events
  source_indices : (N, Pmax)     int64  — global source-sample indices
  source_shifts  : (N, Pmax)     int32  — per-source applied shifts in ticks

Usage
-----
python3 create_multipeak_ndfl.py \\
    --input-dir ./var_zarrs \\
    --out-zarr  ./var_multipeak/multi_train.zarr \\
    --n-samples 20000 \\
    [--min-peaks 2] \\
    [--max-peaks 10] \\
    [--max-shift 200] \\
    [--clip-max 60780] \\
    [--seed 42]
"""

import argparse, os, collections
import numpy as np
import zarr


# ─────────────────────────────────────────────────────────────────────────────
def _make_store(path: str):
    try:
        from zarr.storage import LocalStore
        return LocalStore(path)
    except ImportError:
        return zarr.DirectoryStore(path)


def shift_zeros_fast(x: np.ndarray, t: int, axis: int = -1) -> np.ndarray:
    """Zero-padding integer shift along an axis."""
    t = int(t)
    if t == 0:
        return x.copy()
    L = x.shape[axis]
    if abs(t) >= L:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    idx_src = [slice(None)] * x.ndim
    idx_dst = [slice(None)] * x.ndim
    if t > 0:
        idx_src[axis] = slice(0, L - t)
        idx_dst[axis] = slice(t, L)
    else:
        t = -t
        idx_src[axis] = slice(t, L)
        idx_dst[axis] = slice(0, L - t)
    out[tuple(idx_dst)] = x[tuple(idx_src)]
    return out


def open_zarr_r(path: str):
    try:
        return zarr.open(path, mode='r')
    except Exception:
        return zarr.open_group(zarr.DirectoryStore(path), mode='r')


# ─────────────────────────────────────────────────────────────────────────────
def infer_grouping_tpc_ids(root, zarr_path: str):
    basename = os.path.basename(zarr_path)

    if "charge_tpc_ids" in root:
        tpc_ids = np.asarray(root["charge_tpc_ids"], dtype=np.int32)
        source_name = "charge_tpc_ids"
    elif "tpc_ids" in root:
        tpc_ids = np.asarray(root["tpc_ids"], dtype=np.int32)
        source_name = "tpc_ids"
    elif "event_ids" in root:
        event_ids = np.asarray(root["event_ids"], dtype=np.int64)
        if event_ids.ndim != 2 or event_ids.shape[1] < 2:
            raise RuntimeError(
                f"{basename}: event_ids must have shape (N, >=2), got {event_ids.shape}"
            )
        tpc_ids = np.asarray(event_ids[:, 1], dtype=np.int32)
        source_name = "event_ids[:,1]"
    elif "light_tpc_ids" in root:
        tpc_ids = np.asarray(root["light_tpc_ids"], dtype=np.int32)
        source_name = "light_tpc_ids"
    else:
        raise RuntimeError(
            f"{basename}: no usable TPC metadata found. Need one of "
            "'charge_tpc_ids', 'tpc_ids', 'event_ids', or 'light_tpc_ids'."
        )

    if "event_ids" in root:
        event_ids = np.asarray(root["event_ids"], dtype=np.int64)
        if event_ids.ndim == 2 and event_ids.shape[1] >= 2:
            charge_from_event = np.asarray(event_ids[:, 1], dtype=np.int32)
            if charge_from_event.shape == tpc_ids.shape and source_name == "light_tpc_ids":
                expected = np.where((tpc_ids % 2) == 0, tpc_ids + 1, tpc_ids - 1)
                mismatch = int(np.count_nonzero(charge_from_event != expected))
                if mismatch > 0:
                    print(
                        f"  [warn] {basename}: light_tpc_ids -> charge parity check failed "
                        f"for {mismatch:,} samples"
                    )

    return tpc_ids, source_name


def draw_source_indices(
    pool: np.ndarray,
    n_sources: int,
    rng: np.random.Generator,
    pointer: int,
):
    if pool.size == 0:
        raise RuntimeError("Cannot draw from an empty TPC pool.")

    n_sources = int(n_sources)
    if pool.size < n_sources:
        selected = rng.choice(pool, size=n_sources, replace=True).astype(np.int64, copy=False)
        return np.asarray(selected, dtype=np.int64), int(pointer)

    pointer = int(pointer)
    if pointer + n_sources > pool.size:
        rng.shuffle(pool)
        pointer = 0

    selected = np.asarray(pool[pointer: pointer + n_sources], dtype=np.int64).copy()
    pointer += n_sources
    return selected, int(pointer)


# ─────────────────────────────────────────────────────────────────────────────
def load_all_from_dir(input_dir: str):
    """Load and concatenate all *_var.zarr files in input_dir."""
    zarr_files = sorted([
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir) if d.endswith(".zarr")
    ])
    if not zarr_files:
        raise RuntimeError(f"No .zarr files found in {input_dir}")

    all_inputs  = []
    all_targets = []
    all_tpc_ids = []
    tpc_source_counts = collections.Counter()

    for zp in zarr_files:
        root = open_zarr_r(zp)
        if 'inputs' not in root or 'targets' not in root:
            print(f"  [skip] {os.path.basename(zp)} missing inputs/targets")
            continue
        inputs = np.asarray(root['inputs'], dtype=np.float32)
        targets = np.asarray(root['targets'], dtype=np.float32)
        if inputs.shape != targets.shape:
            raise RuntimeError(
                f"{os.path.basename(zp)}: inputs shape {inputs.shape} does not match "
                f"targets shape {targets.shape}"
            )
        tpc_ids, tpc_source = infer_grouping_tpc_ids(root, zp)
        if tpc_ids.shape[0] != inputs.shape[0]:
            raise RuntimeError(
                f"{os.path.basename(zp)}: TPC metadata length {tpc_ids.shape[0]} does not "
                f"match sample count {inputs.shape[0]}"
            )
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_tpc_ids.append(tpc_ids)
        tpc_source_counts[tpc_source] += 1
        print(
            f"  loaded {inputs.shape[0]:,} samples from {os.path.basename(zp)} "
            f"(TPC source: {tpc_source})"
        )

    inputs  = np.concatenate(all_inputs,  axis=0)
    targets = np.concatenate(all_targets, axis=0)
    tpc_ids = np.concatenate(all_tpc_ids, axis=0)
    print(f"[info] total source samples: {inputs.shape[0]:,}  shape: {inputs.shape}")
    print(f"[info] TPC metadata sources used: {dict(tpc_source_counts)}")
    return inputs, targets, tpc_ids


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-dir",  required=True,  help="Directory of *_var.zarr files")
    ap.add_argument("--out-zarr",   default="./var_multipeak/multi_train.zarr")
    ap.add_argument("--n-samples",  type=int, default=20000)
    ap.add_argument("--min-peaks",  type=int, default=2,
                    help="Minimum number of overlaid source events")
    ap.add_argument("--max-peaks",  type=int, default=10,
                    help="Maximum number of overlaid source events")
    ap.add_argument("--max-shift",  type=int, default=200,
                    help="Max random time shift (ticks)")
    ap.add_argument("--clip-max",   type=float, default=60780.0,
                    help="Clip multi-peak waveforms to this physical maximum after summing")
    ap.add_argument("--write-chunk", type=int, default=128,
                    help="How many output samples to buffer before writing to zarr")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    if args.min_peaks < 1:
        raise ValueError("--min-peaks must be >= 1")
    if args.max_peaks < args.min_peaks:
        raise ValueError("--max-peaks must be >= --min-peaks")
    if args.write_chunk < 1:
        raise ValueError("--write-chunk must be >= 1")

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_zarr)), exist_ok=True)

    print(f"Loading source zarrs from {args.input_dir} ...")
    inputs, targets, tpc_ids = load_all_from_dir(args.input_dir)

    _, N_CH, T = inputs.shape

    # Group sample indices by TPC
    indices_by_tpc = collections.defaultdict(list)
    for i, tpc in enumerate(tpc_ids):
        indices_by_tpc[int(tpc)].append(i)

    available_tpcs = np.asarray(sorted(indices_by_tpc.keys()), dtype=np.int32)
    if available_tpcs.size == 0:
        raise RuntimeError("No source samples available after grouping by TPC.")

    for tpc in available_tpcs.tolist():
        pool = np.asarray(indices_by_tpc[int(tpc)], dtype=np.int64)
        rng.shuffle(pool)
        indices_by_tpc[int(tpc)] = pool
        print(f"  TPC {int(tpc)}: {pool.size:,} events")

    tpc_probs = np.asarray(
        [indices_by_tpc[int(tpc)].size for tpc in available_tpcs],
        dtype=np.float64,
    )
    tpc_probs /= np.sum(tpc_probs)
    tpc_pointers = {int(tpc): 0 for tpc in available_tpcs.tolist()}

    print(f"Saving to {args.out_zarr} ...")
    store = _make_store(args.out_zarr)
    root  = zarr.open(store, mode='w')
    out_chunk = min(int(args.write_chunk), int(args.n_samples))
    wave_chunks = (out_chunk, N_CH, T)
    id_chunks = (out_chunk,)
    src_chunks = (out_chunk, int(args.max_peaks))

    out_inputs = root.create_array(
        "inputs",
        shape=(args.n_samples, N_CH, T),
        chunks=wave_chunks,
        dtype=np.float32,
        overwrite=True,
    )
    out_targets = root.create_array(
        "targets",
        shape=(args.n_samples, N_CH, T),
        chunks=wave_chunks,
        dtype=np.float32,
        overwrite=True,
    )
    out_tpc_ids = root.create_array(
        "tpc_ids",
        shape=(args.n_samples,),
        chunks=id_chunks,
        dtype=np.int32,
        overwrite=True,
    )
    out_source_counts = root.create_array(
        "source_counts",
        shape=(args.n_samples,),
        chunks=id_chunks,
        dtype=np.int16,
        overwrite=True,
    )
    out_source_indices = root.create_array(
        "source_indices",
        shape=(args.n_samples, int(args.max_peaks)),
        chunks=src_chunks,
        dtype=np.int64,
        overwrite=True,
    )
    out_source_shifts = root.create_array(
        "source_shifts",
        shape=(args.n_samples, int(args.max_peaks)),
        chunks=src_chunks,
        dtype=np.int32,
        overwrite=True,
    )

    root.attrs.put({
        "version": "var-multipeak-v2",
        "n_samples": int(args.n_samples),
        "min_peaks": int(args.min_peaks),
        "max_peaks": int(args.max_peaks),
        "max_shift": int(args.max_shift),
        "clip_max": float(args.clip_max),
        "seed": int(args.seed),
        "tpc_sampling": "weighted_by_source_sample_count",
        "tpc_ids_semantics": "Grouping TPC id inferred from charge_tpc_ids/tpc_ids/event_ids[:,1]/light_tpc_ids",
        "source_counts_semantics": "How many source events were overlaid into each output sample",
        "source_indices_semantics": "Global source-sample indices into the concatenated single-event pool; valid prefix length is source_counts[i]",
        "source_shifts_semantics": "Per-source integer tick shifts aligned with source_indices",
    })

    print(f"Generating {args.n_samples:,} multi-peak samples ...")
    buf_inputs = np.zeros((out_chunk, N_CH, T), dtype=np.float32)
    buf_targets = np.zeros((out_chunk, N_CH, T), dtype=np.float32)
    buf_tpc_ids = np.zeros(out_chunk, dtype=np.int32)
    buf_source_counts = np.zeros(out_chunk, dtype=np.int16)
    buf_source_indices = np.full((out_chunk, int(args.max_peaks)), -1, dtype=np.int64)
    buf_source_shifts = np.zeros((out_chunk, int(args.max_peaks)), dtype=np.int32)
    write_start = 0

    for i in range(int(args.n_samples)):
        local_i = i - write_start
        tpc = int(rng.choice(available_tpcs, p=tpc_probs))
        n_peaks = int(rng.integers(args.min_peaks, args.max_peaks + 1))
        pool = indices_by_tpc[tpc]
        selected, new_ptr = draw_source_indices(pool, n_peaks, rng, tpc_pointers[tpc])
        tpc_pointers[tpc] = new_ptr

        combined_in = np.zeros((N_CH, T), dtype=np.float32)
        combined_tgt = np.zeros((N_CH, T), dtype=np.float32)
        shifts = np.asarray(
            rng.integers(-args.max_shift, args.max_shift + 1, size=n_peaks),
            dtype=np.int32,
        )

        for idx, shift in zip(selected.tolist(), shifts.tolist()):
            combined_in += shift_zeros_fast(inputs[int(idx)], int(shift))
            combined_tgt += shift_zeros_fast(targets[int(idx)], int(shift))

        np.clip(combined_in, 0.0, float(args.clip_max), out=combined_in)
        np.clip(combined_tgt, 0.0, float(args.clip_max), out=combined_tgt)

        buf_inputs[local_i] = combined_in
        buf_targets[local_i] = combined_tgt
        buf_tpc_ids[local_i] = int(tpc)
        buf_source_counts[local_i] = int(n_peaks)
        buf_source_indices[local_i].fill(-1)
        buf_source_shifts[local_i].fill(0)
        buf_source_indices[local_i, :n_peaks] = selected
        buf_source_shifts[local_i, :n_peaks] = shifts

        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,} / {args.n_samples:,}")

        is_flush = ((local_i + 1) == out_chunk) or ((i + 1) == args.n_samples)
        if is_flush:
            count = local_i + 1
            write_stop = write_start + count
            out_inputs[write_start:write_stop] = buf_inputs[:count]
            out_targets[write_start:write_stop] = buf_targets[:count]
            out_tpc_ids[write_start:write_stop] = buf_tpc_ids[:count]
            out_source_counts[write_start:write_stop] = buf_source_counts[:count]
            out_source_indices[write_start:write_stop] = buf_source_indices[:count]
            out_source_shifts[write_start:write_stop] = buf_source_shifts[:count]
            write_start = write_stop

    print(f"[done] {args.n_samples:,} samples → {args.out_zarr}")
    print(f"        inputs  shape: {out_inputs.shape}")
    print(f"        targets shape: {out_targets.shape}")


if __name__ == "__main__":
    main()
