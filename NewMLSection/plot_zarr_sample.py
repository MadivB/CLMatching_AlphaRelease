#!/usr/bin/env python3
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import zarr

def plot_zarr_sample(zarr_path, index, out_file):
    # Support both new and old versions of zarr depending on python env
    try:
        from zarr.storage import LocalStore
        store = LocalStore(zarr_path, read_only=True)
    except ImportError:
        store = zarr.DirectoryStore(zarr_path)
        
    try:
        root = zarr.open_group(store=store, mode='r')
    except Exception:
        root = zarr.open_group(store=store)  # fallback


    
    total_samples = root['voxels'].shape[0]
    if index >= total_samples or index < 0:
        print(f"Error: Index {index} is out of bounds. The dataset has {total_samples} samples.")
        return
        
    voxels = root['voxels'][index, 0]  # Shape (NX, NY, NZ) -> (50, 300, 100)
    targets = root['targets'][index]  # Shape (120,)
    light_tpc_id = root['light_tpc_ids'][index]
    charge_tpc_id = root['charge_tpc_ids'][index]
    charge_event_id = root['event_ids'][index, 0]
    
    print(f"Plotting sample {index}:")
    print(f"  Charge Event ID: {charge_event_id}")
    print(f"  Charge TPC ID: {charge_tpc_id}")
    print(f"  Light TPC ID : {light_tpc_id}")
    
    fig = plt.figure(figsize=(15, 8))
    # We want a layout like yours: Left bars | Center 2D view | Right bars | Colorbar
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 3, 1.5, 0.15], wspace=0.3)
    
    # Left bar layout (side 0: channels 0-59)
    ax_l = fig.add_subplot(gs[0])
    ax_l.barh(np.arange(60), targets[0:60], height=0.8, color='dodgerblue', edgecolor='black', linewidth=0.5)
    ax_l.set_ylim(-1, 60)
    ax_l.set_yticks(np.arange(0, 60, 5))
    ax_l.set_yticklabels([f"L{i}" for i in range(0, 60, 5)])
    ax_l.set_xlabel("Max Amplitude (PE)")
    ax_l.set_title("Side 0 (Left Channels)")
    # Make left channels point inwards (towards center)
    ax_l.invert_xaxis()
    ax_l.yaxis.tick_right()
    
    # Center 2D voxel layout
    ax_c = fig.add_subplot(gs[1])
    # voxels is (NX, NY, NZ) corresponding to (x, y, z).
    # To get a 2D y-z view, we sum over x (axis 0).
    voxel_yz = np.sum(voxels, axis=0) # Shape (NY, NZ)
    
    # y_z_img = voxel_yz array where rows are Y and columns are Z.
    y_z_img = voxel_yz
    im = ax_c.imshow(y_z_img, origin='lower', aspect='auto', cmap='viridis')
    ax_c.set_xlabel("Z Voxel Index (NZ=100)")
    ax_c.set_ylabel("Y Voxel Index (NY=300)")
    ax_c.set_title(f"Charge Voxels (Summed over X)\nCharge Event {charge_event_id} | TPC {charge_tpc_id}")
    
    cax = fig.add_subplot(gs[3])
    fig.colorbar(im, cax=cax, label='Charge Sum')

    # Right bar layout (side 1: channels 60-119)
    ax_r = fig.add_subplot(gs[2])
    ax_r.barh(np.arange(60), targets[60:120], height=0.8, color='tomato', edgecolor='black', linewidth=0.5)
    ax_r.set_ylim(-1, 60)
    ax_r.set_yticks(np.arange(0, 60, 5))
    ax_r.set_yticklabels([f"R{i}" for i in range(0, 60, 5)])
    ax_r.set_xlabel("Max Amplitude (PE)")
    ax_r.set_title("Side 1 (Right Channels)")
    ax_r.yaxis.tick_left()
    
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot checkout to: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D Y-Z view and light targets of a Zarr sample")
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to the .zarr directory")
    parser.add_argument("--index", type=int, default=0, help="Index of the event to plot")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the plot image")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"zarr_sample_{args.index}.png")
    plot_zarr_sample(args.zarr_path, args.index, out_file)
