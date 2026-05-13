#!/usr/bin/env python3

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import h5py
import numpy as np
import yaml
import zarr
import pandas as pd

from lut import LUT

try:
    from zarr.storage import LocalStore as _StoreCls  # Zarr ≥ 3
    def make_store(path: str | os.PathLike, *, read_only: bool = False):
        return _StoreCls(str(path), read_only=read_only)
except Exception:
    try:
        from zarr.storage import DirectoryStore as _StoreCls  # Zarr 2.x
    except Exception:
        from zarr import DirectoryStore as _StoreCls
    def make_store(path: str | os.PathLike, *, read_only: bool = False):
        return _StoreCls(str(path))

def _parse_version_tuple(v: str) -> Tuple[int, ...]:
    out = []
    for part in v.split("."):
        d = "".join(ch for ch in part if ch.isdigit())
        if not d: break
        out.append(int(d))
    return tuple(out)

_ZARR_GE_3 = _parse_version_tuple(zarr.__version__) >= (3,)
if _ZARR_GE_3:
    from zarr.codecs import BloscCodec as _BloscCompressor
    def make_blosc(cname: str, clevel: int, *, shuffle: str = "bitshuffle"):
        return _BloscCompressor(cname=cname, clevel=clevel, shuffle=shuffle)
else:
    from numcodecs import Blosc as _BloscCompressor
    def make_blosc(cname: str, clevel: int, *, shuffle: str = "bitshuffle"):
        smap = {
            "bitshuffle": _BloscCompressor.BITSHUFFLE,
            "shuffle": _BloscCompressor.SHUFFLE,
            "noshuffle": _BloscCompressor.NOSHUFFLE,
        }
        return _BloscCompressor(cname=cname, clevel=clevel, shuffle=smap.get(shuffle, _BloscCompressor.BITSHUFFLE))

# ------------------------------- Config --------------------------------------
C = 1
NX, NY, NZ = 50, 300, 100
EXPECTED_TARGETS_PER_TPC = 120
NUM_CHANNELS_PER_ADC = 64
UNITS_PER_METER = 100.0
TPC_EXTENTS_M = (0.5, 3.0, 1.0)
TPC_EXTENTS_UNITS = tuple(dim * UNITS_PER_METER for dim in TPC_EXTENTS_M)

DEFAULT_TPC_CENTERS_PATH = Path("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/samplePreparation/tpc_boundaries.yaml")
AUTO_BASELINE_NSAMP = 75

# ------------------------------ Geometry -------------------------------------
@dataclass(frozen=True)
class TPCGeometry:
    tpc_id: int
    x_center: float; y_center: float; z_center: float
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float
    dx: float; dy: float; dz: float
    inv_dx: float; inv_dy: float; inv_dz: float
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (NX, NY, NZ)

def _voxel_metrics():
    hx, hy, hz = (TPC_EXTENTS_UNITS[0]/2.0, TPC_EXTENTS_UNITS[1]/2.0, TPC_EXTENTS_UNITS[2]/2.0)
    dx = TPC_EXTENTS_UNITS[0]/NX; dy = TPC_EXTENTS_UNITS[1]/NY; dz = TPC_EXTENTS_UNITS[2]/NZ
    return hx, hy, hz, dx, dy, dz, 1.0/dx, 1.0/dy, 1.0/dz

def load_tpc_geometries(path: Path | str) -> Dict[int, TPCGeometry]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"TPC geometry YAML not found: {p}")
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Malformed TPC YAML.")

    hx, hy, hz, dx, dy, dz, inv_dx, inv_dy, inv_dz = _voxel_metrics()
    out: Dict[int, TPCGeometry] = {}
    for k, entry in data.items():
        t = int(k)
        if {"x_center","y_center","z_center"} <= entry.keys():
            cx, cy, cz = float(entry["x_center"]), float(entry["y_center"]), float(entry["z_center"])
        elif {"x_min","x_max","y_min","y_max","z_min","z_max"} <= entry.keys():
            cx = 0.5*(float(entry["x_min"])+float(entry["x_max"]))
            cy = 0.5*(float(entry["y_min"])+float(entry["y_max"]))
            cz = 0.5*(float(entry["z_min"])+float(entry["z_max"]))
        else:
            raise ValueError(f"TPC {t} missing center/boundaries.")
        out[t] = TPCGeometry(
            tpc_id=t, x_center=cx, y_center=cy, z_center=cz,
            x_min=cx-hx, x_max=cx+hx, y_min=cy-hy, y_max=cy+hy, z_min=cz-hz, z_max=cz+hz,
            dx=dx, dy=dy, dz=dz, inv_dx=1.0/dx, inv_dy=1.0/dy, inv_dz=1.0/dz
        )
    return out

def build_event_to_hitids(hits_ref) -> Dict[int, np.ndarray]:
    pairs = np.asarray(hits_ref)
    if pairs.size == 0:
        return {}
    ce = pairs[:,0].astype(np.int64); hi = pairs[:,1].astype(np.int64)
    order = np.argsort(ce, kind="mergesort")
    ce, hi = ce[order], hi[order]
    change = np.nonzero(np.diff(ce)!=0)[0] + 1
    starts = np.concatenate(([0], change))
    ends   = np.concatenate((change, [ce.size]))
    out: Dict[int, np.ndarray] = {}
    for s,e in zip(starts, ends):
        out[int(ce[s])] = hi[s:e]
    return out

def build_light_to_charge_map(h5: h5py.File) -> Dict[int,int]:
    arr = h5['charge/events/ref/light/events/ref'][()]
    c = arr[:,0].astype(np.int64); l = arr[:,1].astype(np.int64)
    order = np.argsort(l, kind="mergesort")
    c, l = c[order], l[order]
    out: Dict[int,int] = {}
    for cc, ll in zip(c,l):
        if int(ll) not in out: out[int(ll)] = int(cc)
    return out

def get_hits_for_event_TPC(charge_event_id: int, charge_tpc_id: int, *,
                           hits_full, event_to_hitids: Dict[int,np.ndarray]) -> np.ndarray:
    hitids = event_to_hitids.get(int(charge_event_id))
    if hitids is None or hitids.size == 0: return np.empty(0, dtype=hits_full.dtype)
    hf = hits_full[hitids]
    g1 = 2 * charge_tpc_id + 1
    g2 = 2 * charge_tpc_id + 2
    io_vals = hf["io_group"].astype(np.int32)
    sel = (io_vals==int(g1)) | (io_vals==int(g2))
    return hf[sel]

def voxelize_subhits(sub: np.ndarray, geom: TPCGeometry) -> np.ndarray:
    out = np.zeros((1,NX,NY,NZ), dtype=np.float32)
    if sub.size == 0: return out
    req = {"x","y","z","E","io_group"}
    if not req.issubset(sub.dtype.names):
        raise RuntimeError(f"Missing hit fields: {req - set(sub.dtype.names)}")
    x = sub["x"].astype(np.float64); y=sub["y"].astype(np.float64); z=sub["z"].astype(np.float64)
    e = sub["E"].astype(np.float32)
    # Hits are already selected by io_group. Reconstructed coordinates can drift
    # across TPC boundaries, so geometry is used only for binning and clipping.
    ix = np.clip(((x-geom.x_min)*geom.inv_dx).astype(np.int32), 0, NX-1)
    iy = np.clip(((y-geom.y_min)*geom.inv_dy).astype(np.int32), 0, NY-1)
    iz = np.clip(((z-geom.z_min)*geom.inv_dz).astype(np.int32), 0, NZ-1)
    lin = ix.astype(np.int64)*(NY*NZ) + iy.astype(np.int64)*NZ + iz.astype(np.int64)
    acc = np.bincount(lin, weights=e, minlength=NX*NY*NZ).astype(np.float32)
    out[0] = acc.reshape(NX,NY,NZ)
    return out

def build_new_lookup_table(h5: h5py.File) -> Dict[Tuple[int,int,int], Tuple[int,int]]:
    meta = h5["geometry_info/sipm_rel_pos"].attrs["meta"]
    data = h5["geometry_info/sipm_rel_pos/data"]
    sipm_rel_pos = LUT.from_array(meta, data)
    samples = h5["light/wvfm/data"]["samples"]
    NADC = int(samples.shape[1])

    lut = {}
    for adc in range(NADC):
        for ch in range(NUM_CHANNELS_PER_ADC):
            try:
                mapping = sipm_rel_pos[(adc, ch)]
            except Exception:
                continue
            if getattr(mapping,"size",None)==0: continue
            tpc, side, y = mapping[0]
            tpc=int(tpc); side=int(side); y=int(y)
            if side not in (0,1) or tpc<0: continue
            lut[(tpc,side,y)] = (adc,ch)
    return lut

def get_single_flash_events_enhanced(h5_file):
    ref_ev_flash = h5_file['light/events/ref/light/flash/ref'][...]
    flash_ids    = h5_file['light/flash/data']['id'][...]
    flash_tpcs   = h5_file['light/flash/data']['tpc'][...]
    flash_hits   = h5_file['light/flash/data']['n_sum_hits'][...]
    
    id_to_tpc = dict(zip(flash_ids.astype(int), flash_tpcs.astype(int)))
    id_to_hits = dict(zip(flash_ids.astype(int), flash_hits.astype(int)))
    
    ev = ref_ev_flash[:, 0].astype(np.int64)
    fid = ref_ev_flash[:, 1].astype(int)
    
    tpc = np.array([id_to_tpc.get(int(f), -1) for f in fid], dtype=np.int32)
    hits = np.array([id_to_hits.get(int(f), 0) for f in fid], dtype=np.int32)
    
    mask = tpc >= 0
    ev = ev[mask]
    tpc = tpc[mask]
    hits = hits[mask]
    
    # Filter out all noise flashes with n_sum_hits == 1
    valid_hits_mask = hits > 1
    ev_valid = ev[valid_hits_mask]
    tpc_valid = tpc[valid_hits_mask]
    
    pairs = np.stack([ev_valid, tpc_valid], axis=1)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
        
    pairs_unique, counts_per_tpcev = np.unique(pairs, axis=0, return_counts=True)
    single_flash_pairs = pairs_unique[counts_per_tpcev == 1]
    return single_flash_pairs

def build_cnn120_dataset(in_h5: str, out_zarr: str, *,
                         tpc_yaml: str | os.PathLike = DEFAULT_TPC_CENTERS_PATH,
                         hits_dset: str = "calib_prompt_hits",
                         min_phi: float = 400.0,
                         store_dtype: str = "float16",
                         zstd_clevel: int = 1,
                         min_hits: int = 4,
                         phi_batch_events: int = 64) -> int:
    
    geometry_map = load_tpc_geometries(tpc_yaml)
    
    with h5py.File(in_h5, 'r') as h5:
        if "light" not in h5:
            print(f"[skip] no 'light' group in {os.path.basename(in_h5)}")
            return 0
            
        hits_full = h5[f"charge/{hits_dset}/data"]
        hits_ref  = h5[f"charge/events/ref/charge/{hits_dset}/ref"]
        wvf       = h5['light/wvfm/data']['samples']
        N_light   = wvf.shape[0]
        
        evt2hit = build_event_to_hitids(hits_ref)
        l2c = build_light_to_charge_map(h5)
        
        lut = build_new_lookup_table(h5)
        NTPC = max(t for (t,_,_) in lut.keys() if t>=0) + 1
        
        pairs = get_single_flash_events_enhanced(h5)  # (light_event_id, light_TPCid)
        if len(pairs) == 0:
            print(f"[info] {os.path.basename(in_h5)} empty single flash events.")
            return 0
            
        # create tpc to channel mapping: tpc -> [(adc, ch), ...] ordered by side and y
        tpc_to_channels = {}
        for t in range(NTPC):
            # Side 0
            s0 = sorted([(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == t and side == 0], key=lambda item: item[0])
            # Side 1
            s1 = sorted([(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == t and side == 1], key=lambda item: item[0])
            tpc_to_channels[t] = [(adc, ch) for (y, adc, ch) in (s0 + s1)]

        compressor = make_blosc("zstd", int(zstd_clevel))
        root = zarr.group(store=make_store(out_zarr), overwrite=True)

        N_est = len(pairs)
        vox = root.create_dataset("voxels", shape=(N_est, C, NX, NY, NZ),
                                  chunks=(1,C,NX,NY,NZ),
                                  dtype=("f2" if store_dtype=="float16" else "f4"),
                                  compressor=compressor, overwrite=True)
        tgt = root.create_dataset("targets", shape=(N_est, EXPECTED_TARGETS_PER_TPC),
                                  chunks=(max(1, 2048//EXPECTED_TARGETS_PER_TPC), EXPECTED_TARGETS_PER_TPC),
                                  dtype="f4", compressor=compressor, overwrite=True)
        ids = root.create_dataset("event_ids", shape=(N_est, 2),
                                  dtype="i8", chunks=(4096,2),
                                  compressor=compressor, overwrite=True)
        light_ids = root.create_dataset("light_event_ids", shape=(N_est,),
                                        dtype="i8", chunks=(4096,),
                                        compressor=compressor, overwrite=True)
        light_tpc_ids = root.create_dataset("light_tpc_ids", shape=(N_est,),
                                            dtype="i8", chunks=(4096,),
                                            compressor=compressor, overwrite=True)
        charge_tpc_ids = root.create_dataset("charge_tpc_ids", shape=(N_est,),
                                             dtype="i8", chunks=(4096,),
                                             compressor=compressor, overwrite=True)

        root.attrs.put({
            "version": "cnn120-full-single-flash-v1",
            "bins": (NX,NY,NZ),
            "voxel_extents_units": TPC_EXTENTS_UNITS,
            "voxel_extents_m": TPC_EXTENTS_M,
            "units_per_meter": UNITS_PER_METER,
            "targets_len": EXPECTED_TARGETS_PER_TPC,
            "selection": f"single_flash_events (exactly 1 flash), min_phi={float(min_phi)}",
            "baseline": "auto-first-75",
            "tpc_yaml": str(Path(tpc_yaml).resolve()),
            "event_ids_semantics": "event_ids[:,0]=charge_event_id, event_ids[:,1]=charge_tpc_id",
            "light_ids_semantics": "light_event_ids[i]=light event id",
            "light_tpc_ids_semantics": "light_tpc_ids[i]=light-side TPC id",
            "charge_tpc_ids_semantics": "charge_tpc_ids[i]=charge-side TPC id (swapped 0<->1, 2<->3...)",
        })

        df = pd.DataFrame(pairs, columns=['light_event_id', 'light_TPCid'])
        by_light = df.groupby('light_event_id').indices
        
        write_i = 0
        start = 0
        while start < N_light:
            end = min(N_light, start + int(phi_batch_events))
            chunk = wvf[start:end, :, :, :]  # (m, 140, 64, T)
            
            # Baseline correct all
            b = chunk[:, :, :, :AUTO_BASELINE_NSAMP].mean(axis=-1, keepdims=True)
            chunk_b = (chunk - b).astype(np.float32)
            maxima = chunk_b.max(axis=-1)
            
            for levent in range(start, end):
                idxs_rows = by_light.get(levent)
                if idxs_rows is None:
                    continue
                    
                cevent = l2c.get(int(levent))
                if cevent is None:
                    continue
                    
                row_maxima = maxima[levent - start]  # (140, 64)
                
                for ridx in idxs_rows:
                    light_TPCid = int(df.at[ridx, "light_TPCid"])
                    
                    # Convert to charge TPCid using the even-odd mismatch relation requested
                    if light_TPCid % 2 == 0:
                        charge_TPCid = light_TPCid + 1
                    else:
                        charge_TPCid = light_TPCid - 1
                        
                    geom = geometry_map.get(int(charge_TPCid))
                    if geom is None:
                        continue
                        
                    sub = get_hits_for_event_TPC(int(cevent), charge_TPCid,
                                                 hits_full=hits_full,
                                                 event_to_hitids=evt2hit)
                                                 
                    if sub.size < int(min_hits):
                        continue
                        
                    chans = tpc_to_channels.get(light_TPCid, [])
                    if len(chans) == 0:
                        continue
                        
                    # Build phi vector
                    phi_vec = np.zeros(EXPECTED_TARGETS_PER_TPC, dtype=np.float32)
                    keep = False
                    for j, (adc, ch) in enumerate(chans):
                        val = row_maxima[adc, ch]
                        phi_vec[j] = val
                        if val >= min_phi:
                            keep = True
                            
                    if not keep:
                        continue
                        
                    vol = voxelize_subhits(sub, geom)
                    if not np.any(vol):
                        continue
                        
                    vox[write_i, :, :, :, :] = vol.astype(vox.dtype, copy=False)
                    tgt[write_i, :] = phi_vec
                    ids[write_i, :] = (int(cevent), int(charge_TPCid))
                    light_ids[write_i] = int(levent)
                    light_tpc_ids[write_i] = int(light_TPCid)
                    charge_tpc_ids[write_i] = int(charge_TPCid)
                    write_i += 1
            
            start = end
            
        if write_i == 0:
            store = root.store
            path_attr = getattr(store, "path", None)
            if path_attr and os.path.isdir(path_attr):
                import shutil; shutil.rmtree(path_attr, ignore_errors=True)
            print(f"[info] wrote 0 samples for {os.path.basename(in_h5)}; removed empty zarr.")
            return 0
            
        # Trim datasets
        vox.resize((write_i, C, NX, NY, NZ))
        tgt.resize((write_i, EXPECTED_TARGETS_PER_TPC))
        ids.resize((write_i, 2))
        light_ids.resize((write_i,))
        light_tpc_ids.resize((write_i,))
        charge_tpc_ids.resize((write_i,))

        print(f"[done] {os.path.basename(in_h5)} -> {out_zarr} | samples={write_i}")
        return write_i

def collect_h5(in_dir: str) -> List[str]:
    cand = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".h5") or f.endswith(".hdf5")]
    return sorted(cand)

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-h5", type=str, help="Input HDF5")
    ap.add_argument("--out-zarr", type=str, help="Output Zarr directory")
    ap.add_argument("--in-dir", type=str, help="Input directory of HDF5s")
    ap.add_argument("--out-dir", type=str, help="Output directory of Zarrs")
    ap.add_argument("--tpc-yaml", type=str, default=str(DEFAULT_TPC_CENTERS_PATH), help="TPC centers YAML")
    ap.add_argument("--hits-dset", type=str, default="calib_prompt_hits")
    ap.add_argument("--min-phi", type=float, default=400.0, help="Min peak to count (after baseline)")
    ap.add_argument("--store-dtype", choices=["float16","float32"], default="float16")
    ap.add_argument("--zstd-clevel", type=int, default=1)
    ap.add_argument("--phi-batch-events", type=int, default=64)
    ap.add_argument("--min-hits", type=int, default=4)
    return ap.parse_args()

def main():
    a = parse_args()
    
    if a.in_h5 and a.out_zarr:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_zarr)) or ".", exist_ok=True)
        build_cnn120_dataset(
            a.in_h5, a.out_zarr,
            tpc_yaml=a.tpc_yaml,
            hits_dset=a.hits_dset,
            min_phi=a.min_phi,
            store_dtype=a.store_dtype,
            zstd_clevel=a.zstd_clevel,
            min_hits=a.min_hits,
            phi_batch_events=a.phi_batch_events
        )
    elif a.in_dir and a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        files = collect_h5(a.in_dir)
        print(f"[info] Found {len(files)} HDF5 files in {a.in_dir}")
        for f in files:
            base = os.path.basename(f)
            name = os.path.splitext(base)[0] + ".zarr"
            out_z = os.path.join(a.out_dir, name)
            if os.path.isdir(out_z):
                print(f"[skip] exists: {out_z}")
                continue
            build_cnn120_dataset(
                f, out_z,
                tpc_yaml=a.tpc_yaml,
                hits_dset=a.hits_dset,
                min_phi=a.min_phi,
                store_dtype=a.store_dtype,
                zstd_clevel=a.zstd_clevel,
                min_hits=a.min_hits,
                phi_batch_events=a.phi_batch_events
            )
    else:
        print("ERROR: Provide either (--in-h5 AND --out-zarr) OR (--in-dir AND --out-dir).", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
