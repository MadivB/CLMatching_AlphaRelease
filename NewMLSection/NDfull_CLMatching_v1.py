import os
import sys
import h5py
import numpy as np
import torch
import pandas as pd

# Load ND-full utilities
sys.path.append("/global/cfs/cdirs/dune/users/yuxuan/2x2CLMatching")
from pulse_shapes import timeinterpolation

from ML_NDfull_latent import (
    load_latent_model, 
    process_clusters_to_imageMaps,
    light_tpc_to_charge_tpc,
    load_tpc_geometries,
    DEFAULT_TPC_YAML,
    NUM_TARGETS
)

def charge_tpc_from_io_group(io_group):
    """Map ND-LAr io_group to charge-TPC id. This is the authoritative hit TPC."""
    io = np.asarray(io_group, dtype=np.int64)
    if np.any(io <= 0):
        bad = np.unique(io[io <= 0]).tolist()
        raise ValueError(f"Invalid io_group values for TPC assignment: {bad[:10]}")
    return ((io - 1) // 2).astype(np.int32)


def assign_hits_to_charge_tpc(x, y, z, yaml_path=DEFAULT_TPC_YAML):
    raise RuntimeError(
        "Geometry-based hit-to-TPC assignment is disabled. Use charge_tpc_from_io_group(io_group) instead."
    )

from track_fit_ransac import fit_tracks_labels
from cluster_fit import fit_cluster_labels, fit_noise_list
from lut import LUT

def frontEndChargeClustering(xset, yset, zset, Eset, lam=1.1, min_length_cm=40, expand_frac=0.10): 
    # --- Phase 1 (RANSAC) ---
    labels_phase1, track_params_phase1 = fit_tracks_labels(xset, yset, zset, lam=lam, min_length_cm=min_length_cm)
    # --- Phase 2 (DBSCAN merge) ---
    labels_phase2, DBSCAN_label, dbscan_noise_idx, stats = fit_cluster_labels(
        xset, yset, zset, Eset, labels=labels_phase1, eps=2.5, min_samples=3)   
    # --- Phase 3 (distance-based candidates for remaining noise) ---
    entries = fit_noise_list(xset, yset, zset, Eset, labels_phase2, expand_frac=expand_frac)
    return labels_phase2, entries, labels_phase1.max()

def max_likelihood_curve_with_base(predicted, base, actual, errorMetric, search_range=700, smoothing=False):
    xp = np
    pred_norm = predicted.astype(xp.float32) 
    base_norm = base.astype(xp.float32)      
    act_norm  = actual.astype(xp.float32)   

    if smoothing:
        kernel = xp.ones(5, dtype=xp.float32) / 5.0
        conv   = lambda x: xp.convolve(x, kernel, mode='same')
        pred_norm = xp.apply_along_axis(conv, 1, pred_norm)
        base_norm = xp.apply_along_axis(conv, 1, base_norm)
        act_norm  = xp.apply_along_axis(conv, 1, act_norm)

    n_ticks = pred_norm.size
    shifts  = xp.arange(search_range + 1, dtype=int)
    errors  = xp.empty_like(shifts, dtype=xp.float32)

    for k, t0 in enumerate(shifts):
        shifted = xp.zeros_like(pred_norm)
        if t0 > 0:
            shifted[:, t0:] = pred_norm[:, :-t0]
        else:
            shifted[:] = pred_norm

        model = shifted + base_norm
        model = np.clip(model, None, 60780)
        err_total = ((model - act_norm)**2 / errorMetric).sum() 
        errors[k] = err_total / n_ticks

    return shifts, errors

def lossMatrixFormation(imageMaps, TPCactual, TPCbase, TPCstd, TPCid, clusters, placedMask, tpct0Candidates):
    remaining_clusters = clusters[~placedMask]
    N_remaining = len(remaining_clusters)
    M_candidates = len(tpct0Candidates)
    
    lossMatrix = np.zeros((N_remaining, M_candidates), dtype=np.float32)
    
    if N_remaining == 0 or M_candidates == 0:
        return lossMatrix, remaining_clusters

    cluster_images = np.stack([imageMaps[(cid, TPCid)] for cid in remaining_clusters])
    param_batch = cluster_images.reshape(-1, 1000) 

    normalization_factor = cluster_images.shape[1] * cluster_images.shape[2] 
    
    for j, t0 in enumerate(tpct0Candidates):
        shifted_flat = timeinterpolation(param_batch, shift=t0, baseline=0)
        shifted = shifted_flat.reshape(N_remaining, NUM_TARGETS, 1000)
        
        model = shifted + TPCbase
        model = np.clip(model, None, 60780)
        
        diff = model - TPCactual
        chi2 = (diff**2 / TPCstd).sum(axis=(1, 2))
        
        lossMatrix[:, j] = chi2 / normalization_factor
        
    return lossMatrix, remaining_clusters

def fast_stack(waveforms, tpcids):
    tpcids = np.atleast_1d(tpcids)
    stacked = waveforms[tpcids]
    return stacked.reshape(-1, stacked.shape[-1])

def fast_stack_images(imageMaps, cluster_id, tpcids):
    tpcids = np.atleast_1d(tpcids)
    images = [imageMaps[(cluster_id, tpc)] for tpc in tpcids]
    stacked = np.stack(images, axis=0) 
    return stacked.reshape(-1, stacked.shape[-1])

def imageUpdater(newImage, baseImage, sorted_tpcs, num_targets=NUM_TARGETS):
    sorted_tpcs = np.atleast_1d(sorted_tpcs)
    newImage_reshaped = newImage.reshape(len(sorted_tpcs), num_targets, -1)
    baseImage[sorted_tpcs] += newImage_reshaped
    return baseImage

def build_new_lookup_table(h5):
    meta = h5["geometry_info/sipm_rel_pos"].attrs["meta"]
    data = h5["geometry_info/sipm_rel_pos/data"]
    sipm_rel_pos = LUT.from_array(meta, data)
    samples = h5["light/wvfm/data"]["samples"]
    NADC = int(samples.shape[1])
    NUM_CHANNELS_PER_ADC = 64

    lut = {}
    for adc in range(NADC):
        for ch in range(NUM_CHANNELS_PER_ADC):
            try:
                mapping = sipm_rel_pos[(adc, ch)]
            except:
                continue
            if getattr(mapping,"size",None)==0: continue
            tpc, side, y = mapping[0]
            tpc=int(tpc); side=int(side); y=int(y)
            if side not in (0,1) or tpc<0: continue
            lut[(tpc,side,y)] = (adc,ch)
    return lut

def get_formatted_light_waveforms(h5_file):
    print("Formatting light waveforms...")
    lut = build_new_lookup_table(h5_file)
    NTPC = max(t for (t,_,_) in lut.keys() if t>=0) + 1
    tpc_to_channels = {}
    for t in range(NTPC):
        s0 = sorted([(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == t and side == 0], key=lambda item: item[0])
        s1 = sorted([(y, adc, ch) for (tpc, side, y), (adc, ch) in lut.items() if tpc == t and side == 1], key=lambda item: item[0])
        tpc_to_channels[t] = [(adc, ch) for (y, adc, ch) in (s0 + s1)]

    light_wvfms = h5_file['light/wvfm/data']['samples'][:] # (N_events, 140, 64, 1000)
    baseline = np.mean(light_wvfms[..., :75], axis=-1, keepdims=True)
    light_wvfms_baselined = light_wvfms - baseline

    N_events = light_wvfms.shape[0]
    # Estimate max charge TPC id (since charge tpc can be light_tpc + 1)
    max_charge_tpc = NTPC + 2
    
    formatted_wvfms = np.zeros((N_events, max_charge_tpc, NUM_TARGETS, 1000), dtype=np.float32)
    
    for light_tpc in range(NTPC):
        chans = tpc_to_channels.get(light_tpc, [])
        if len(chans) != NUM_TARGETS:
            continue
        charge_tpc = light_tpc_to_charge_tpc(light_tpc)
        
        # Build array of adcs, chs
        adc_idx = [x[0] for x in chans]
        ch_idx = [x[1] for x in chans]
        
        # Populate formatted array mapped by CHARGE TPC
        formatted_wvfms[:, charge_tpc, :, :] = light_wvfms_baselined[:, adc_idx, ch_idx, :]
        
    print(f"Formatted light waveforms array shape: {formatted_wvfms.shape}")
    return formatted_wvfms

def main():
    print("Starting ND-full Charge-Light Matching...")
    
    data_file = "/global/cfs/cdirs/dune/users/yuxuan/singleTPCCLMatching/2x2_sim/run-ndlar-flow/Tutorial.flow/FLOW/0000000/Tutorial.flow.0000000.FLOW.hdf5"
    data_file = "/pscratch/sd/d/dunepro/yuxuan/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift/FLOW/0000000/MiniProdN5p1_NDComplex_FHC.flow.full.lowintensity.sanddrift.0000001.FLOW.hdf5"
    
    # Load HDF5 components
    h5 = h5py.File(data_file,'r')
    hits_dset = 'calib_prompt_hits'
    hits_full = h5[f'charge/{hits_dset}/data']
    hits_ref = h5[f'charge/events/ref/charge/{hits_dset}/ref']
    charge_light_ref = h5['charge/events/ref/light/events/ref']
    
    all_formatted_wvfms = get_formatted_light_waveforms(h5)
    
    # Load Latent Model
    checkpoint_path = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/NewMLSection/runs/latent_run/best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Latent Model from {checkpoint_path} on {device}")
    modelMod123, meta123 = load_latent_model(checkpoint_path, device=device)
    wvfm_tmpl = np.load("/global/cfs/cdirs/dune/users/yuxuan/interactLevel/clusteringStudy/dataDrivenLUTtable/MLApproach/CNNApproach/avg_pulse.npy")

    # Select an event for mapping, e.g., EV=1
    ev_id = 1
    print(f"\nProcessing event {ev_id}...")
    
    hit_mask = hits_ref[:, 0] == ev_id
    hit_refs = hits_ref[hit_mask, 1]
    if len(hit_refs) == 0:
        print("No hits found for event", ev_id)
        return
        
    hits_evt = hits_full[hit_refs]
    
    # Get light waveforms for this event
    light_refs = charge_light_ref[charge_light_ref[:,0] == ev_id]
    if len(light_refs) == 0:
        print("No light event found for charge event", ev_id)
        return
    lightID = light_refs[0, 1]
    
    fullLightWaveform = all_formatted_wvfms[lightID] # (MAX_CHARGE_TPC, 120, 1000)
    
    # Temporarily set Variance to 1s
    fullLightStd = np.ones_like(fullLightWaveform)
    print("Set placeholder Variance (Array of 1s)")
    
    xset = hits_evt['x']
    yset = hits_evt['y']
    zset = hits_evt['z']
    Eset = hits_evt['E']
    io_group = np.asarray(hits_evt['io_group'], dtype=np.int64)
    hitTPCid = charge_tpc_from_io_group(io_group)
    
    # Keep only hits with valid DAQ io_group. Do not reassign by reconstructed x/y/z.
    valid_mask = io_group > 0
    n_total = len(xset)
    n_valid = valid_mask.sum()
    print(f"Hit count: {n_total} total, {n_valid} with valid io_group, {n_total - n_valid} discarded")
    xset = xset[valid_mask]
    yset = yset[valid_mask]
    zset = zset[valid_mask]
    Eset = Eset[valid_mask]
    hitTPCid = hitTPCid[valid_mask]
    
    # 3-stage charge clustering
    labels, noiseList, trackidMax = frontEndChargeClustering(xset, yset, zset, Eset)
    Nclusters = labels.max() + 1
    print(f"Found {Nclusters} clusters (Tracks <= {trackidMax})")
    
    # Generate image maps using Latent Model
    print("Generating predicted Light Waveforms using Latent Model...")
    imageMaps, meta = process_clusters_to_imageMaps(
        xset, yset, zset, Eset,
        hitTPCid, labels,
        model=modelMod123,
        target_scale=1e-3,
        template=wvfm_tmpl/1000,
        include_noise=False,
        batch_size=1
    )
    
    tpc_to_clusters = {}
    cluster_to_tpcs = {}
    for (cluster_id, tpc_id) in imageMaps.keys():
        cluster_to_tpcs.setdefault(cluster_id, []).append(tpc_id)
    for cluster_id, tpc_list in cluster_to_tpcs.items():
        if cluster_id > trackidMax:
            for tpc in tpc_list:
                tpc_to_clusters.setdefault(tpc, []).append(cluster_id)
                
    baseImage = np.full_like(fullLightWaveform, 0, dtype=np.float32)
    max_charge_tpc = fullLightWaveform.shape[0]
    t0Candidates = [[] for _ in range(max_charge_tpc)]
    t0Resolution = 5
    
    print("Starting Track Association...")
    for clusterid in range(trackidMax + 1):
        if clusterid not in cluster_to_tpcs:
            continue
        sorted_tpcs = np.sort(np.asarray(cluster_to_tpcs[clusterid], dtype=int))
        TPCbase = fast_stack(baseImage, sorted_tpcs)
        TPCactual = fast_stack(fullLightWaveform, sorted_tpcs)
        TPCstd = fast_stack(fullLightStd, sorted_tpcs)
        TPCimage = fast_stack_images(imageMaps, clusterid, sorted_tpcs)
        
        shifts, errors = max_likelihood_curve_with_base(
            predicted=TPCimage,
            actual=TPCactual,
            base=TPCbase,
            search_range=700,
            errorMetric=TPCstd
        )
        minError = np.argmin(errors)
        newt0 = shifts[minError]
        
        expected_peak_idx = int(105 + newt0)
        signal_1d = np.sum(np.clip(TPCactual - TPCbase, 0, None), axis=0)
        search_start = max(0, expected_peak_idx - t0Resolution)
        search_end = min(1000, expected_peak_idx + 1)
        
        if search_end > search_start:
            local_peak_offset = np.argmax(signal_1d[search_start:search_end])
            actual_peak_idx = search_start + local_peak_offset
            correction = actual_peak_idx - expected_peak_idx
            newt0 += correction
            
        print(f"Track {clusterid} associated with t0 = {newt0}")
        for tpc in sorted_tpcs:
            t0Candidates[tpc].append(newt0)
            
        newImage = timeinterpolation(TPCimage, shift=newt0, baseline=0).astype(np.float32)
        baseImage = imageUpdater(newImage, baseImage, sorted_tpcs)
        baseImage = np.clip(baseImage, a_min=None, a_max=60780)

    print("Starting Cluster Association...")
    for TPCid in tpc_to_clusters:
        clusters = np.array(tpc_to_clusters[TPCid])
        placedMask = np.zeros(len(clusters), dtype=bool)
        tpct0Candidates = t0Candidates[TPCid]
        
        TPCbase = fast_stack(baseImage, TPCid)
        TPCactual = fast_stack(fullLightWaveform, TPCid)
        TPCstd = fast_stack(fullLightStd, TPCid)
        
        # Calculate current background error
        CurrentError = ((TPCbase - TPCactual)**2 / TPCstd).sum()
        
        for idx in range(len(clusters)):
            if len(tpct0Candidates) == 0:
                print(f"TPC {TPCid} has 0 t0 Candidates.")
                break
                
            lossMatrix, remaining_clusters = lossMatrixFormation(
                imageMaps, TPCactual, TPCbase, TPCstd, TPCid, clusters, placedMask, tpct0Candidates
            )
            lossMatrix = lossMatrix - CurrentError
            min_idx_flat = np.argmin(lossMatrix)
            best_cluster_idx = min_idx_flat // lossMatrix.shape[1]
            best_t0_idx = min_idx_flat % lossMatrix.shape[1]
            
            clusterid = remaining_clusters[best_cluster_idx]
            original_idx = np.where(clusters == clusterid)[0][0]
            placedMask[original_idx] = True
            opt_t0 = tpct0Candidates[best_t0_idx]
            
            print(f"Cluster {clusterid} assigned to t0 = {opt_t0} (TPC {TPCid})")
            
            TPCimage = imageMaps[(clusterid, TPCid)]
            newImage = timeinterpolation(TPCimage, shift=opt_t0, baseline=0).astype(np.float32)
            # Reconstruct (1, 120, 1000) for baseImage update
            baseImage[TPCid] += newImage.reshape(NUM_TARGETS, 1000)
            baseImage = np.clip(baseImage, a_min=None, a_max=60780)
            
            CurrentError = ((fast_stack(baseImage, TPCid) - TPCactual)**2 / TPCstd).sum()
            
    print("\nCharge-Light Matching execution completed.")

if __name__ == "__main__":
    main()
