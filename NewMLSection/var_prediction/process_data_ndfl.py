#!/usr/bin/env python3
"""
process_data_ndfl.py
====================
Step 1 of the ND-full variance prediction pipeline (Standalone HDF5 Version).

For each raw unseen ND-full HDF5 file:
  - Find valid single-flash events
  - Voxelize the charge hits for each valid event into (50, 300, 100) grids
  - Load raw true waveforms from the HDF5 (120 channels, baseline corrected)
  - Run the ML model on the voxels to get predicted peak amplitudes
  - Expand predictions to (120, T) waveforms by multiplying by a pulse template
    and shifting that template to the actual event peak tick
  - Save to a new zarr: inputs (actual unscaled raw waveforms), targets (predicted scaled waveforms), metadata

Output zarr schema (per file):
  inputs         : (S, 120, T)  float32  — actual waveform  (raw from HDF5, baseline corrected)
  targets        : (S, 120, T)  float32  — ML-predicted waveform (scaled by template,
                                           clipped only after waveform formation)
  event_ids      : (S, 2)       int64    — [charge_event_id, charge_tpc_id]
  charge_tpc_ids : (S,)         int64    — charge-side TPC id (same as event_ids[:,1])
  tpc_ids        : (S,)         int64    — alias of charge_tpc_ids for downstream compatibility
  light_tpc_ids  : (S,)         int64
  light_event_ids: (S,)         int64

Usage
-----
python3 process_data_ndfl.py \
    --h5-dir    /path/to/original/hdf5s \
    --out-dir   ./var_zarrs \
    --tpc-yaml  /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/samplePreparation/tpc_boundaries.yaml \
    --model     latent \
    --model-path runs/latent_run/best_model.pt \
    [--perceiver-path runs/ndfull_run_distributed/best_model.pt] \
    [--pulse-template /global/cfs/cdirs/dune/users/yuxuan/interactLevel/clusteringStudy/dataDrivenLUTtable/MLApproach/CNNApproach/avg_pulse.npy] \
    [--waveform-len 1000] \
    [--peak-tick 105] \
    [--batch-size 16]
"""

import argparse, os, sys, glob
import numpy as np
import zarr
import torch
import h5py
from tqdm import tqdm

# ── locate modules in the parent dir ──────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import voxelizer utilities
try:
    from compile_samples_to_zarr import (
        load_tpc_geometries, build_event_to_hitids, build_light_to_charge_map,
        get_single_flash_events_enhanced, build_new_lookup_table,
        get_hits_for_event_TPC, voxelize_subhits
    )
except ImportError as e:
    print(f"Warning: Could not import methods from compile_samples_to_zarr. Make sure it's in the parent directory. {e}")
    sys.exit(1)

NUM_CHANNELS = 120
DEFAULT_WAVEFORM_LEN = 1000
DEFAULT_PEAK_TICK    = 105
AUTO_BASELINE_NSAMP  = 75
DEFAULT_PULSE_TEMPLATE = (
    "/global/cfs/cdirs/dune/users/yuxuan/interactLevel/"
    "clusteringStudy/dataDrivenLUTtable/MLApproach/CNNApproach/avg_pulse.npy"
)


# ─────────────────────────────────────────────────────────────────────────────
def build_default_template(tau: float = 15.0, T: int = DEFAULT_WAVEFORM_LEN,
                            peak_tick: int = DEFAULT_PEAK_TICK) -> np.ndarray:
    """Simple RC-like exponential pulse centred at peak_tick, normalised to 1."""
    t = np.arange(T, dtype=np.float32)
    before = t < peak_tick
    tmpl = np.where(before, 0.0, np.exp(-(t - peak_tick) / tau))
    rise = 3
    for k in range(rise):
        tick = peak_tick - rise + k
        if 0 <= tick < T:
            tmpl[tick] = (k + 1) / rise
    mx = tmpl.max()
    if mx > 0:
        tmpl /= mx
    return tmpl.astype(np.float32)


def load_template(path: str | None, T: int = DEFAULT_WAVEFORM_LEN,
                  peak_tick: int = DEFAULT_PEAK_TICK) -> np.ndarray:
    if path is None:
        path = DEFAULT_PULSE_TEMPLATE
    raw = np.load(path).astype(np.float32).ravel()
    if raw.size < T:
        raw = np.pad(raw, (0, T - raw.size))
    else:
        raw = raw[:T]
    raw /= 1000.0
    print(f"[template] loaded from {path}, shape {raw.shape}, scaled by 1/1000")
    return raw


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


def actual_event_peak_tick(waveforms: np.ndarray) -> int:
    """Use the strongest actual sample across all channels as the event peak tick."""
    return int(np.unravel_index(np.argmax(waveforms), waveforms.shape)[1])


def expand_to_waveform(
    peaks: np.ndarray,
    template: np.ndarray,
    peak_ticks: np.ndarray | None = None,
    clip_max: float = 60780.0,
) -> np.ndarray:
    wave = peaks[:, :, None] * template[None, None, :]
    if peak_ticks is not None:
        template_peak_tick = int(np.argmax(template))
        shifted = np.zeros_like(wave)
        for i, peak_tick in enumerate(peak_ticks):
            shifted[i] = shift_zeros_fast(
                wave[i],
                int(peak_tick) - template_peak_tick,
                axis=-1,
            )
        wave = shifted
    return np.clip(wave, 0.0, clip_max).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def process_h5_file(h5_path: str, model,
                    template: np.ndarray, batch_size: int,
                    target_scale: float, device, geometry_map,
                    clip_max: float = 60780.0, min_hits: int = 4):
    """
    Reads an HDF5 directly, extracts single flash events, builds voxels, runs ML,
    and grabs true raw waveforms.
    Returns tracking dict containing actual_wave, pred_wave, etc.
    """
    
    T = len(template)
    batch_voxels = []
    batch_info = [] # stores (cevent, c_tpc_id, levent, l_tpc_id, raw_wav, peak_tick)
    
    all_actual = []
    all_pred = []
    all_ev_ids = []
    all_charge_tpcs = []
    all_l_tpcs = []
    all_l_evids = []
    
    def process_batch(voxels, infos):
        bS = len(voxels)
        if bS == 0: return

        # Forward Pass
        vox_arr = np.stack(voxels, axis=0) # (B, 1, 50, 300, 100)
        for i in range(bS):
            mx = vox_arr[i].max()
            # if mx > 1e-6:
            #     vox_arr[i] /= mx
        
        xt = torch.from_numpy(vox_arr).to(device, dtype=torch.float32)
        tt = torch.tensor([info[1] for info in infos], dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = model(xt, tt)
            if isinstance(out, dict): out = out["final"]
            pred_scaled = out.detach().cpu().float().numpy()
            
        inv = 1.0 / target_scale
        # Keep the raw channel prediction unconstrained here so tall pulses can
        # saturate only after the waveform has been formed.
        pred_raw = pred_scaled * inv # (B, 120)
        peak_ticks = np.asarray([info[5] for info in infos], dtype=np.int32)
        pred_wave = expand_to_waveform(
            pred_raw,
            template,
            peak_ticks=peak_ticks,
            clip_max=clip_max,
        ) # (B, 120, T)
        
        # Package and append
        for i in range(bS):
            all_actual.append(infos[i][4])
            all_pred.append(pred_wave[i])
            all_ev_ids.append((infos[i][0], infos[i][1]))
            all_charge_tpcs.append(infos[i][1])
            all_l_tpcs.append(infos[i][3])
            all_l_evids.append(infos[i][2])


    with h5py.File(h5_path, 'r') as h5:
        if "light" not in h5:
            print(f"  [skip] no 'light' group in {os.path.basename(h5_path)}")
            return None
            
        hits_full = h5["charge/calib_prompt_hits/data"]
        hits_ref  = h5["charge/events/ref/charge/calib_prompt_hits/ref"]
        wvfm_data = h5['light/wvfm/data']['samples']
        
        evt2hit = build_event_to_hitids(hits_ref)
        l2c = build_light_to_charge_map(h5)
        lut = build_new_lookup_table(h5)
        NTPC = max(t for (t,_,_) in lut.keys() if t>=0) + 1
        
        # Create map of expected channels
        tpc_to_channels = {}
        for t in range(NTPC):
            s0 = sorted([(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == t and side == 0], key=lambda item: item[0])
            s1 = sorted([(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == t and side == 1], key=lambda item: item[0])
            tpc_to_channels[t] = [(adc, ch) for (y, adc, ch) in (s0 + s1)]
            
        pairs = get_single_flash_events_enhanced(h5)
        if len(pairs) == 0:
            return None
            
        print(f"    Found {len(pairs)} single-flash pairs. Voxelizing & Inferring...")
        
        for p in tqdm(pairs, desc="Processing Events", leave=False):
            light_event_id = int(p[0])
            light_TPCid = int(p[1])
            
            cevent = l2c.get(light_event_id)
            if cevent is None: continue
            
            if light_TPCid % 2 == 0:
                charge_TPCid = light_TPCid + 1
            else:
                charge_TPCid = light_TPCid - 1
                
            geom = geometry_map.get(charge_TPCid)
            if geom is None: continue
            
            # Hit fetching
            sub = get_hits_for_event_TPC(cevent, charge_TPCid, hits_full=hits_full, event_to_hitids=evt2hit)
            if sub.size < min_hits: continue
            
            # Voxelize
            vol = voxelize_subhits(sub, geom)
            if not np.any(vol): continue
            
            # Grabbing waveforms
            chans = tpc_to_channels.get(light_TPCid, [])
            if len(chans) == 0: continue
            
            event_wvfm = wvfm_data[light_event_id]
            raw_target = np.zeros((NUM_CHANNELS, event_wvfm.shape[-1]), dtype=np.float32)
            for j, (adc, ch) in enumerate(chans):
                if j < NUM_CHANNELS:
                    raw_target[j, :] = event_wvfm[adc, ch, :]
                    
            baseline = np.mean(raw_target[:, :AUTO_BASELINE_NSAMP], axis=1, keepdims=True)
            bc_wvfm = raw_target - baseline
            
            if bc_wvfm.shape[1] > T:
                bc_wvfm = bc_wvfm[:, :T]
            elif bc_wvfm.shape[1] < T:
                bc_wvfm = np.pad(bc_wvfm, ((0, 0), (0, T - bc_wvfm.shape[1])))
                
            clipped_wvfm = np.clip(bc_wvfm, 0.0, clip_max)
            
            batch_voxels.append(vol)
            peak_tick = actual_event_peak_tick(clipped_wvfm)
            batch_info.append(
                (
                    cevent,
                    charge_TPCid,
                    light_event_id,
                    light_TPCid,
                    clipped_wvfm,
                    peak_tick,
                )
            )
            
            if len(batch_voxels) >= batch_size:
                process_batch(batch_voxels, batch_info)
                batch_voxels = []
                batch_info = []

        if len(batch_voxels) > 0:
            process_batch(batch_voxels, batch_info)

    if len(all_actual) == 0:
        return None
        
    return {
        "inputs": np.stack(all_actual, axis=0),
        "targets": np.stack(all_pred, axis=0),
        "event_ids": np.array(all_ev_ids, dtype=np.int64),
        "charge_tpc_ids": np.array(all_charge_tpcs, dtype=np.int64),
        "tpc_ids": np.array(all_charge_tpcs, dtype=np.int64),
        "light_tpc_ids": np.array(all_l_tpcs, dtype=np.int64),
        "light_event_ids": np.array(all_l_evids, dtype=np.int64)
    }

# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--h5-dir",         required=True,  help="Directory containing the original HDF5 files")
    ap.add_argument("--out-dir",        default="./var_zarrs")
    ap.add_argument("--tpc-yaml",       default="/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/samplePreparation/tpc_boundaries.yaml", help="Path to tpc_boundaries.yaml")
    ap.add_argument("--n-files",        type=int, default=10,
                    help="Number of hdf5 files to process (0=all)")
    ap.add_argument("--model",          choices=["latent", "perceiver"], default="latent")
    ap.add_argument("--model-path",     required=True,  help="Path to best_model.pt")
    ap.add_argument("--target-scale",   type=float, default=1e-3)
    ap.add_argument("--batch-size",     type=int,   default=16)
    ap.add_argument(
        "--pulse-template",
        default=DEFAULT_PULSE_TEMPLATE,
        help="Path to the average pulse template .npy file",
    )
    ap.add_argument("--waveform-len",   type=int,   default=DEFAULT_WAVEFORM_LEN)
    ap.add_argument("--peak-tick",      type=int,   default=DEFAULT_PEAK_TICK)
    ap.add_argument("--clip-max",       type=float, default=60780.0)
    ap.add_argument("--device",         default=None,   help="cuda or cpu (auto-detect if omitted)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")

    geometry_map = load_tpc_geometries(args.tpc_yaml)

    # ── Load model ───────────────────────────────────────────────────────────
    if args.model == "latent":
        from ML_NDfull_latent import load_latent_model
        model, meta = load_latent_model(args.model_path, target_scale=args.target_scale,
                                        device=str(device))
    else:
        from ML_NDfull_perceiver import load_perceiver_model
        model, meta = load_perceiver_model(args.model_path, target_scale=args.target_scale,
                                           device=str(device))
    model.eval()
    print(f"[model] loaded {args.model} from {args.model_path}")

    # ── Pulse template ───────────────────────────────────────────────────────
    template = load_template(args.pulse_template, T=args.waveform_len,
                             peak_tick=args.peak_tick)

    # ── Find H5 files ──────────────────────────────────────────────────────
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, "*.hdf5")) + glob.glob(os.path.join(args.h5_dir, "*.h5")))
    if args.n_files > 0:
        h5_files = h5_files[:args.n_files]
    print(f"[info] processing {len(h5_files)} HDF5 files → {args.out_dir}")

    total_samples = 0
    for h5_path in h5_files:
        bname  = os.path.splitext(os.path.basename(h5_path))[0]
        out_path = os.path.join(args.out_dir, bname + "_var.zarr")

        if os.path.isdir(out_path):
            print(f"  [skip] exists: {out_path}")
            continue

        print(f"  processing {os.path.basename(h5_path)}")
        result = process_h5_file(
            h5_path, model, template, args.batch_size, args.target_scale, device, geometry_map, args.clip_max
        )
        if result is None:
            print(f"  [skip] empty or missing datasets in {bname}")
            continue

        actual_wave = result["inputs"]
        pred_wave = result["targets"]
        
        S = actual_wave.shape[0]
        T = actual_wave.shape[2]

        root = zarr.open(out_path, mode='w')
        chunks = (min(S, 100), NUM_CHANNELS, T)
        root.create_array("inputs",  data=actual_wave, chunks=chunks,
                            overwrite=True)
        root.create_array("targets", data=pred_wave,   chunks=chunks,
                            overwrite=True)
        root.create_array("event_ids", data=result["event_ids"], chunks=(min(S, 4096), 2),
                            overwrite=True)
        root.create_array("charge_tpc_ids", data=result["charge_tpc_ids"], chunks=(min(S, 4096),),
                            overwrite=True)
        root.create_array("tpc_ids", data=result["tpc_ids"], chunks=(min(S, 4096),),
                            overwrite=True)
        root.create_array("light_tpc_ids", data=result["light_tpc_ids"], chunks=(min(S, 4096),),
                            overwrite=True)
        root.create_array("light_event_ids", data=result["light_event_ids"], chunks=(min(S, 4096),),
                            overwrite=True)

        total_samples += S
        print(f"  → saved {S:,} samples to {out_path}")

    print(f"\n[done] total samples written: {total_samples:,}")

if __name__ == "__main__":
    main()
