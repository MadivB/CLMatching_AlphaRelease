#!/usr/bin/env python3
"""
prep_fullModule_baselineVaried_with_TPCid.py  (DEAD-CHANNEL AWARE + TPCid embedding)

Adds:
  • Writes a per-sample `tpc_ids` dataset (int16).
  • Optional embedding of TPC id into voxel channels:
        --embed-tpc-into-voxels none   -> vox shape (1, NX, NY, NZ)  [default]
        --embed-tpc-into-voxels id     -> vox shape (2, NX, NY, NZ), channel 1 = constant TPCid (0..7)
        --embed-tpc-into-voxels onehot -> vox shape (1+8, NX, NY, NZ), channels 1..8 are one-hot planes for TPCid
  • Preserves existing behavior, including dead-channel masking via YAML and sentinel.
  • event_ids[:,0]=charge_event_id, event_ids[:,1]=TPCid   (kept for compatibility)

If you choose id/onehot embedding, remember to update your training `c_in` accordingly
(2 for id, 9 for onehot). With `none`, your current training code works unchanged.
"""

from __future__ import annotations
import argparse, os, sys
from typing import Dict, List, Tuple, Sequence, Optional, Set
import numpy as np, h5py, zarr, pandas as pd
from numcodecs import Blosc
from lut import LUT

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ----------------------------- Geometry / binning -----------------------------
C_BASE, NX, NY, NZ = 1, 32, 128, 64
_X_RANGE_BY_TPC = {
    0: ( 32.5,  64.5), 2: ( 32.5,  64.5),
    1: (  2.5,  34.5), 3: (  2.5,  34.5),
    4: (-34.5,  -2.5), 6: (-34.5,  -2.5),
    5: (-64.5, -32.5), 7: (-64.5, -32.5),
}
_POSZ_TPCS = {0, 1, 4, 5}
_NEGZ_TPCS = {2, 3, 6, 7}

# ---- 48-target setup: side {0,1}, y_rel {0..23} ----
Y_RANGE = range(24)
ORDERED_KEYS = [(0, y) for y in Y_RANGE] + [(1, y) for y in Y_RANGE]  # 48
TYPEA_YRELS = set(range(0,6)) | set(range(12,18))  # Type-A definition

# ----------------------------- Helpers ---------------------------------------
def build_event_to_hitids(hits_ref) -> Dict[int, np.ndarray]:
    pairs = np.asarray(hits_ref)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("hits_ref must be (N,2) [charge_event_id, hit_id].")
    ceids = pairs[:, 0].astype(np.int64, copy=False)
    hids  = pairs[:, 1].astype(np.int64, copy=False)
    order = np.argsort(ceids, kind="mergesort")
    ceids, hids = ceids[order], hids[order]
    change = np.nonzero(np.diff(ceids) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends   = np.concatenate((change, [ceids.size]))
    out: Dict[int, np.ndarray] = {}
    for s, e in zip(starts, ends):
        out[int(ceids[s])] = hids[s:e]
    return out

def build_light_to_charge_map(h5: h5py.File) -> Dict[int, int]:
    arr = h5['charge/events/ref/light/events/ref'][()]
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError("Unexpected shape for charge/events/ref/light/events/ref; expected (N,2).")
    charge_ids = arr[:, 0].astype(np.int64, copy=False)
    light_ids  = arr[:, 1].astype(np.int64, copy=False)
    order = np.argsort(light_ids, kind="mergesort")
    charge_ids, light_ids = charge_ids[order], light_ids[order]
    l2c: Dict[int, int] = {}
    for c, l in zip(charge_ids, light_ids):
        l = int(l); c = int(c)
        if l not in l2c:
            l2c[l] = c
    return l2c

def get_hits_for_event_TPC(charge_event_id: int, tpc_id: int, *, hits_full, event_to_hitids: Dict[int, np.ndarray]):
    hitids = event_to_hitids.get(int(charge_event_id), None)
    if hitids is None or hitids.size == 0:
        return np.empty(0, dtype=hits_full.dtype)
    hf = hits_full[hitids]
    sel = (hf["io_group"].astype(np.int32) - 1) == int(tpc_id)
    return hf[sel]

def _bin_y(y: np.ndarray) -> np.ndarray:
    idx = np.floor(y + 64.0).astype(np.int32)  # [-64,64) -> [0..127]
    return np.clip(idx, 0, NY - 1)

def _bin_z(z: np.ndarray, tpc_id: int) -> np.ndarray:
    if tpc_id in _POSZ_TPCS:
        idx = np.floor(z).astype(np.int32)           # [0,64) -> [0..63]
    elif tpc_id in _NEGZ_TPCS:
        idx = np.floor(z + 64.0).astype(np.int32)    # [-64,0) -> [0..63]
    else:
        raise ValueError(f"Unknown TPC id: {tpc_id}")
    return np.clip(idx, 0, NZ - 1)

def _bin_x(x: np.ndarray, tpc_id: int) -> np.ndarray:
    x_lo, _ = _X_RANGE_BY_TPC[tpc_id]
    idx = np.floor(x - x_lo).astype(np.int32)
    return np.clip(idx, 0, NX - 1)

def voxelize_subhits(sub: np.ndarray, tpc_id: int) -> np.ndarray:
    if sub.size == 0:
        return np.zeros((1, NX, NY, NZ), dtype=np.float32)
    x = sub["x"].astype(np.float64, copy=False)
    y = sub["y"].astype(np.float64, copy=False)
    z = sub["z"].astype(np.float64, copy=False)
    e = sub["E"].astype(np.float32, copy=False)
    ix = _bin_x(x, tpc_id); iy = _bin_y(y); iz = _bin_z(z, tpc_id)
    lin = ix.astype(np.int64) * (NY * NZ) + iy.astype(np.int64) * NZ + iz.astype(np.int64)
    acc = np.bincount(lin, weights=e, minlength=NX * NY * NZ).astype(np.float32, copy=False)
    grid = acc.reshape(NX, NY, NZ)
    out = np.zeros((1, NX, NY, NZ), dtype=np.float32)
    out[0] = grid
    return out

def build_lookup(h5: h5py.File):
    rel_meta = h5["geometry_info/sipm_rel_pos"].attrs["meta"]
    rel_data = h5["geometry_info/sipm_rel_pos/data"]
    sipm_rel_pos = LUT.from_array(rel_meta, rel_data)
    lut = {}
    for adc in range(8):
        for ch in range(64):
            try:
                TPC, side, y = sipm_rel_pos[(adc, ch)][0]
                lut[(int(TPC), int(side), int(y))] = (adc, ch)
            except Exception:
                pass
    return lut

def adc_kept(adc: int, mode: str) -> bool:
    if mode == "odd":  return (adc % 2) == 1
    if mode == "even": return (adc % 2) == 0
    return True

def build_tpc_channel_index(lut: Dict[Tuple[int,int,int], Tuple[int,int]], adc_mode: str):
    uniq_pairs: List[Tuple[int,int]] = []
    pair_to_idx: Dict[Tuple[int,int], int] = {}
    for tpc in range(8):
        for (side, y_rel) in ORDERED_KEYS:
            key = (tpc, side, y_rel)
            if key not in lut:
                continue
            adc, ch = lut[key]
            if not adc_kept(adc, adc_mode):
                continue
            p = (int(adc), int(ch))
            if p not in pair_to_idx:
                pair_to_idx[p] = len(uniq_pairs)
                uniq_pairs.append(p)
    tpc_to_idx: Dict[int, List[int]] = {}
    for tpc in range(8):
        idxs = []
        for (side, y_rel) in ORDERED_KEYS:
            key = (tpc, side, y_rel)
            if key not in lut:
                idxs.append(-1); continue
            adc, ch = lut[key]
            if not adc_kept(adc, adc_mode):
                idxs.append(-1); continue
            idxs.append(pair_to_idx[(int(adc), int(ch))])
        tpc_to_idx[tpc] = idxs
    adcs = np.array([a for a, _ in uniq_pairs], dtype=np.int64)
    chans = np.array([c for _, c in uniq_pairs], dtype=np.int64)
    return adcs, chans, tpc_to_idx

def print_mapping_coverage(lut: Dict[Tuple[int,int,int], Tuple[int,int]], adc_mode: str):
    print("\n[diag] Mapping coverage by (side,y_rel): existing vs kept under --adc-parity =", adc_mode)
    for side in (0,1):
        line_exist = []; line_kept = []; line_tag = []
        for y in range(24):
            has = any((tpc, side, y) in lut for tpc in range(8))
            kept = any((tpc, side, y) in lut and adc_kept(lut[(tpc,side,y)][0], adc_mode) for tpc in range(8))
            line_exist.append('✔' if has else '·')
            line_kept.append('✔' if kept else '·')
            line_tag.append('A' if y in TYPEA_YRELS else 'B')
            

# ----------------------------- Light selection --------------------------------
def get_single_flash_events(h5: h5py.File) -> pd.DataFrame:
    evt_flash_pairs = h5['light/events/ref/light/flash/ref'][()]
    flashes = h5['light/flash/data']
    tpc_arr = flashes['tpc'][()]
    time_ranges = flashes['hit_time_range'][()]
    event_to_flashids: Dict[int, List[int]] = {}
    for eventid, flashid in evt_flash_pairs:
        event_to_flashids.setdefault(int(eventid), []).append(int(flashid))
    rows = []
    for eventid, fids in event_to_flashids.items():
        tpc_to: Dict[int, List[int]] = {}
        for fid in fids:
            tpc_to.setdefault(int(tpc_arr[fid]), []).append(fid)
        for tpcid, group in tpc_to.items():
            if len(group) == 1:
                rows.append((eventid, tpcid, group[0]))
            else:
                times = time_ranges[group]
                all_windows = sorted(times, key=lambda x: x[0])
                merged = [all_windows[0].copy()]
                for w in all_windows[1:]:
                    prev = merged[-1]
                    if w[0] - prev[1] <= 81:
                        prev[1] = max(prev[1], w[1])
                    else:
                        merged.append(w.copy())
                if len(merged) == 1:
                    chosen = group[int(np.argmin(times[:, 0]))]
                    rows.append((eventid, tpcid, chosen))
    return pd.DataFrame(rows, columns=['eventid', 'TPCid', 'flashid'])

# ----------------------------- DEAD YAML --------------------------------------
def _parse_dead_yaml(path: str) -> Set[Tuple[int,int]]:
    if not path:
        return set()
    if yaml is None:
        raise SystemExit("PyYAML not available; install 'pyyaml' or omit --dead-yaml.")
    with open(path, "r") as f:
        doc = yaml.safe_load(f) or {}
    raw = doc.get("dead_channels", [])
    out: Set[Tuple[int,int]] = set()
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            a, c = int(item[0]), int(item[1]); out.add((a,c))
        elif isinstance(item, dict):
            a, c = int(item.get("adc")), int(item.get("ch", item.get("channel")))
            out.add((a,c))
        elif isinstance(item, str):
            try:
                a_s, c_s = item.split(",")
                out.add((int(a_s.strip()), int(c_s.strip())))
            except Exception:
                pass
    return out

# ----------------------------- Per-file builder -------------------------------
def build_cnn48_dataset(in_h5: str, out_zarr: str, *,
                        hits_dset: str = "calib_prompt_hits",
                        min_phi: float = 400.0,
                        saturation_included: bool = True,
                        store_dtype: str = "float16",
                        phi_batch_events: int = 64,
                        zstd_clevel: int = 1,
                        Mod0: bool = True,
                        adc_parity: str = "both",
                        baseline_ticks: int = 75,
                        min_hits: int = 4,
                        dead_yaml: Optional[str] = None,
                        dead_sentinel: float = -10000.0,
                        embed_tpc_into_voxels: str = "none") -> int:
    """
    Writes -10000 (or chosen sentinel) for targets mapped to dead (adc, ch) pairs.
    Also writes tpc_ids (int16). Optionally embeds TPC id into voxels.
    """
    if embed_tpc_into_voxels not in ("none", "id", "onehot"):
        raise SystemExit("--embed-tpc-into-voxels must be one of: none, id, onehot")

    h5 = h5py.File(in_h5, "r")

    if "light" not in h5.keys():
        print(f"[skip] no 'light' group in {os.path.basename(in_h5)}; skipping.")
        return 0

    hits_full = h5[f"charge/{hits_dset}/data"]
    hits_ref  = h5[f"charge/events/ref/charge/{hits_dset}/ref"]
    wvf       = h5['light/wvfm/data']['samples']
    N_light   = wvf.shape[0]

    event_to_hitids  = build_event_to_hitids(hits_ref)
    lut              = build_lookup(h5)

    print_mapping_coverage(lut, adc_mode=adc_parity)

    adcs_uni, chans_uni, tpc_to_idx = build_tpc_channel_index(lut, adc_mode=adc_parity)
    l2c_map          = build_light_to_charge_map(h5)
    pairs            = get_single_flash_events(h5)

    # Mod0 filter
    if bool(Mod0):
        pairs = pairs[pairs["TPCid"].isin([0,1])].reset_index(drop=True)
    else:
        pairs = pairs[~pairs["TPCid"].isin([0,1])].reset_index(drop=True)
    if len(pairs) == 0:
        print("[info] no pairs after Mod0 filter; skipping.")
        return 0
    by_light = pairs.groupby('eventid').indices

    # Dead-channel set
    dead_pairs: Set[Tuple[int,int]] = _parse_dead_yaml(dead_yaml) if dead_yaml else set()
    if dead_pairs:
        print(f"[dead] loaded {len(dead_pairs)} dead channels from {dead_yaml}: {sorted(dead_pairs)}")

    # Prepare Zarr
    compressor = Blosc(cname="zstd", clevel=int(zstd_clevel), shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=zarr.DirectoryStore(out_zarr), overwrite=True)
    N_est = len(pairs); N_out = len(ORDERED_KEYS)

    chan_extra = 0
    if embed_tpc_into_voxels == "id":      chan_extra = 1
    elif embed_tpc_into_voxels == "onehot": chan_extra = 8
    channels = C_BASE + chan_extra

    vox_dtype = "f2" if store_dtype=="float16" else "f4"
    vox = root.create_dataset("voxels",  shape=(N_est, channels, NX, NY, NZ),
                              chunks=(1, channels, NX, NY, NZ),
                              dtype=vox_dtype,
                              compressor=compressor, overwrite=True)
    tgt = root.create_dataset("targets", shape=(N_est, N_out),
                              chunks=(max(1, 2048//N_out), N_out),
                              dtype="f4", compressor=compressor, overwrite=True)
    ids = root.create_dataset("event_ids", shape=(N_est, 2), dtype="i8",
                              chunks=(max(1, 4096), 2), compressor=compressor, overwrite=True)
    light_ids = root.create_dataset("light_event_ids", shape=(N_est,), dtype="i8",
                                    chunks=(max(1, 4096),), compressor=compressor, overwrite=True)
    # NEW: tpc_ids per sample
    tpc_ids = root.create_dataset("tpc_ids", shape=(N_est,), dtype="i2",
                                  chunks=(max(1, 4096),), compressor=compressor, overwrite=True)

    root.attrs.put({
        "bins": (NX, NY, NZ),
        "ordered_keys": ORDERED_KEYS,
        "min_phi": float(min_phi),
        "selection_rule": "keep if any (max - baseline_mean_first_ticks) >= min_phi AND charge voxel grid has >= min_hits nonzero hits",
        "store_dtype": store_dtype,
        "hits_dset": hits_dset,
        "x_ranges_by_tpc": _X_RANGE_BY_TPC,
        "posz_tpcs": sorted(_POSZ_TPCS),
        "negz_tpcs": sorted(_NEGZ_TPCS),
        "version": "cnn48-full-fast-mapped-v2-tpcid",
        "phi_batch_events": int(phi_batch_events),
        "zstd_clevel": int(zstd_clevel),
        "ids_semantics": "event_ids[:,0]=charge_event_id, event_ids[:,1]=TPCid; light_event_ids[i]=light_event_id",
        "Mod0": bool(Mod0),
        "Mod0_filter": "True→TPCid in {0,1}; False→TPCid not in {0,1}",
        "targets_len": N_out,
        "adc_parity": adc_parity,
        "baseline_mode": "per-event per-channel mean over first baseline_ticks samples",
        "baseline_ticks": int(baseline_ticks),
        "min_hits": int(min_hits),
        "dead_yaml_path": (os.path.abspath(dead_yaml) if dead_yaml else ""),
        "dead_sentinel": float(dead_sentinel),
        "dead_channels": sorted(list(dead_pairs)),
        # NEW:
        "channels": int(channels),
        "tpc_embed_mode": embed_tpc_into_voxels,
        "tpc_ids_dtype": "int16",
    })

    write_i = 0
    req_fields = {"x","y","z","E","io_group"}
    if adcs_uni.size == 0:
        print("[info] no usable (ADC,CHAN) from LUT after parity filter; skipping.")
    else:
        start = 0
        while start < N_light:
            end = min(N_light, start + int(phi_batch_events))
            chunk = wvf[start:end, :, :, :]   # (m, 8, 64, T)

            T = chunk.shape[-1]
            w = int(baseline_ticks) if int(baseline_ticks) <= T else T

            maxima_all = chunk.max(axis=-1).astype(np.float32, copy=False)
            baseline_all = chunk[:, :, :, :w].mean(axis=-1).astype(np.float32, copy=False)

            if saturation_included:
                valid_mask_sel = None
            else:
                plateau = (chunk == maxima_all[..., None]).sum(axis=-1)
                valid_mask_sel = plateau[:, adcs_uni, chans_uni] <= 5

            vmax_sel      = maxima_all[:, adcs_uni, chans_uni]
            vbaseline_sel = baseline_all[:, adcs_uni, chans_uni]
            phi_sel = vmax_sel - vbaseline_sel

            for levent in range(start, end):
                idxs_rows = by_light.get(levent)
                if idxs_rows is None:
                    continue
                row_phi   = phi_sel[levent - start]
                row_valid = valid_mask_sel[levent - start] if valid_mask_sel is not None else None

                cevent = l2c_map.get(int(levent))
                if cevent is None:
                    continue

                for ridx in idxs_rows:
                    tpid = int(pairs.at[ridx, "TPCid"])
                    idxs = tpc_to_idx.get(tpid)
                    if idxs is None:
                        continue
                    sel = [k for k in idxs if k >= 0]
                    if not sel:
                        continue

                    sub = get_hits_for_event_TPC(int(cevent), tpid, hits_full=hits_full, event_to_hitids=event_to_hitids)
                    if sub.size < int(min_hits):
                        continue
                    if not req_fields.issubset(sub.dtype.names):
                        missing = req_fields - set(sub.dtype.names)
                        raise SystemExit(f"Missing hit fields: {missing}")

                    keep = np.any(row_phi[sel] >= float(min_phi)) if (row_valid is None) else np.any((row_phi[sel] >= float(min_phi)) & row_valid[sel])
                    if not keep:
                        continue

                    # Base voxel grid (1, NX, NY, NZ)
                    vol = voxelize_subhits(sub, tpid)

                    # Optional TPC embedding planes
                    if embed_tpc_into_voxels == "id":
                        id_plane = np.full((1, NX, NY, NZ), float(tpid), dtype=np.float32)
                        vol = np.concatenate([vol, id_plane], axis=0)
                    elif embed_tpc_into_voxels == "onehot":
                        oh = np.zeros((8, NX, NY, NZ), dtype=np.float32)
                        if 0 <= tpid < 8:
                            oh[tpid, :, :, :] = 1.0
                        vol = np.concatenate([vol, oh], axis=0)

                    if not np.any(vol[0]):
                        continue

                    # Build 48-dim phi vector ordered by ORDERED_KEYS
                    phi_vec = np.zeros(len(ORDERED_KEYS), dtype=np.float32)
                    for j, k in enumerate(idxs):
                        if k < 0:
                            continue
                        adc_j, ch_j = int(adcs_uni[k]), int(chans_uni[k])
                        if (adc_j, ch_j) in dead_pairs:
                            phi_vec[j] = float(dead_sentinel)
                            continue
                        if (row_valid is None) or bool(row_valid[k]):
                            phi_vec[j] = row_phi[k]

                    # Write
                    vox[write_i, :, :, :, :] = vol.astype(vox.dtype, copy=False)
                    tgt[write_i, :] = phi_vec
                    ids[write_i, :] = (int(cevent), tpid)
                    light_ids[write_i] = int(levent)
                    tpc_ids[write_i] = int(tpid)
                    write_i += 1
            start = end

    if write_i == 0:
        store = root.store
        if hasattr(store, 'path'):
            import shutil; shutil.rmtree(store.path, ignore_errors=True)
        print(f"[info] wrote 0 samples for {os.path.basename(in_h5)}; removed empty zarr.")
        return 0

    vox.resize((write_i, channels, NX, NY, NZ))
    tgt.resize((write_i, len(ORDERED_KEYS)))
    ids.resize((write_i, 2))
    light_ids.resize((write_i,))
    tpc_ids.resize((write_i,))

    # Diagnostics
    nz = np.asarray(tgt[:write_i] > 0, dtype=np.bool_).sum(axis=0)  # (48,)
    total = write_i
    print("\n[diag] Nonzero φ counts per target column (out of", total, "samples):")
    def is_typeA_idx(j):
        side, y = ORDERED_KEYS[j]
        return y in TYPEA_YRELS
    for side in (0,1):
        idxs = [j for j,(s,_y) in enumerate(ORDERED_KEYS) if s == side]
        line = " ".join(f"{nz[j]:5d}" for j in idxs)
        print(f"  side {side}: {line}")
    nzA = sum(nz[j] for j in range(48) if is_typeA_idx(j))
    nzB = sum(nz[j] for j in range(48) if not is_typeA_idx(j))
    print(f"  Type-A total nonzero across columns: {nzA}  |  Type-B: {nzB}\n")

    print(f"[done] {os.path.basename(in_h5)} -> {out_zarr} | samples={write_i}")
    return write_i

# ----------------------------- DLIST + CLI ------------------------------------
def looks_like_h5(path: str) -> bool:
    n = path.lower()
    return n.endswith(".h5") or n.endswith(".hdf5")

def collect_h5(in_dir: str) -> List[str]:
    cand = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if looks_like_h5(f)]
    return sorted(cand)

def write_dlist(dlist_path: str, zarr_paths: Sequence[str]):
    if not dlist_path: return
    os.makedirs(os.path.dirname(os.path.abspath(dlist_path)) or ".", exist_ok=True)
    with open(dlist_path, "w") as f:
        for p in zarr_paths:
            f.write(os.path.abspath(p) + "\n")
    print(f"[dlist] wrote {len(zarr_paths)} entries to {dlist_path}")

def is_48_target_zarr(zpath: str) -> bool:
    try:
        g = zarr.open_group(zarr.DirectoryStore(zpath), "r")
        return ("targets" in g) and (g["targets"].shape[1] == 48) and (g["targets"].shape[0] > 0)
    except Exception:
        return False

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # single-file mode
    ap.add_argument("--in-h5", type=str)
    ap.add_argument("--out-zarr", type=str)
    # directory mode
    ap.add_argument("--in-dir", type=str)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="./preparedData")
    # shared options
    ap.add_argument("--hits-dset", default="calib_prompt_hits")
    ap.add_argument("--min-phi", type=float, default=400.0)
    ap.add_argument("--saturation-included", type=lambda s: s.lower() not in ("false","0","no"), default=True)
    ap.add_argument("--store-dtype", choices=["float16","float32"], default="float16")
    ap.add_argument("--phi-batch-events", type=int, default=64)
    ap.add_argument("--zstd-clevel", type=int, default=1)
    ap.add_argument("--Mod0", type=lambda s: s.lower() not in ("false","0","no"), required=True)
    ap.add_argument("--adc-parity", choices=["odd","even","both"], default="both")
    ap.add_argument("--baseline-ticks", type=int, default=75)
    ap.add_argument("--min-hits", type=int, default=4)
    ap.add_argument("--dlist-out", type=str, default="./dlist_48.txt")
    # Dead channels
    ap.add_argument("--dead-yaml", type=str, default="", help="YAML with dead_channels: [[adc,ch], ...]. Optional.")
    ap.add_argument("--dead-sentinel", type=float, default=-10000.0, help="Target value for dead channels.")
    # NEW: TPC embedding mode
    ap.add_argument("--embed-tpc-into-voxels", choices=["none","id","onehot"], default="none",
                    help="If not 'none', append TPC-id planes to voxel input: 'id' adds one constant plane; 'onehot' adds 8 one-hot planes.")
    return ap.parse_args()

def main():
    args = parse_args()

    produced: List[str] = []

    if args.in_h5 and args.out_zarr:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_zarr)) or ".", exist_ok=True)
        n = build_cnn48_dataset(
            args.in_h5, args.out_zarr,
            hits_dset=args.hits_dset,
            min_phi=args.min_phi,
            saturation_included=args.saturation_included,
            store_dtype=args.store_dtype,
            phi_batch_events=args.phi_batch_events,
            zstd_clevel=args.zstd_clevel,
            Mod0=args.Mod0,
            adc_parity=args.adc_parity,
            baseline_ticks=args.baseline_ticks,
            min_hits=args.min_hits,
            dead_yaml=(args.dead_yaml or None),
            dead_sentinel=args.dead_sentinel,
            embed_tpc_into_voxels=args.embed_tpc_into_voxels,
        )
        if n > 0 and is_48_target_zarr(args.out_zarr):
            produced.append(args.out_zarr)

    elif args.in_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        files = collect_h5(args.in_dir)
        if args.max_files and args.max_files > 0:
            files = files[:args.max_files]
        print(f"[info] Found {len(files)} HDF5 files in {args.in_dir}")

        for f in files:
            base = os.path.basename(f)
            name = os.path.splitext(base)[0] + ".zarr"
            out_z = os.path.join(args.out_dir, name)
            if os.path.isdir(out_z) and is_48_target_zarr(out_z):
                print(f"[skip] exists (48 targets, non-empty): {out_z}")
                produced.append(out_z)
                continue
            n = build_cnn48_dataset(
                f, out_z,
                hits_dset=args.hits_dset,
                min_phi=args.min_phi,
                saturation_included=args.saturation_included,
                store_dtype=args.store_dtype,
                phi_batch_events=args.phi_batch_events,
                zstd_clevel=args.zstd_clevel,
                Mod0=args.Mod0,
                adc_parity=args.adc_parity,
                baseline_ticks=args.baseline_ticks,
                min_hits=args.min_hits,
                dead_yaml=(args.dead_yaml or None),
                dead_sentinel=args.dead_sentinel,
                embed_tpc_into_voxels=args.embed_tpc_into_voxels,
            )
            if n > 0 and is_48_target_zarr(out_z):
                produced.append(out_z)
    else:
        print("ERROR: Provide either (--in-h5 AND --out-zarr) OR (--in-dir [--max-files] --out-dir).", file=sys.stderr)
        sys.exit(2)

    if args.dlist_out:
        write_dlist(args.dlist_out, produced)

if __name__ == "__main__":
    main()
