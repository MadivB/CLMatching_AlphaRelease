#!/usr/bin/env python3
import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import zarr
import glob

def plot_distributions(zarr_dir, out_file):
    # Support both new and old versions of zarr depending on python env
    zarr_files = glob.glob(os.path.join(zarr_dir, "*.zarr"))
    if not zarr_files:
        print(f"No .zarr files found in {zarr_dir}")
        return
        
    print(f"Found {len(zarr_files)} zarr files in {zarr_dir}")
    print(zarr_files)
    
    total_energies = []
    lit_voxels_counts = []
    
    for zarr_path in zarr_files:
        try:
            from zarr.storage import LocalStore
            store = LocalStore(zarr_path, read_only=True)
        except ImportError:
            store = zarr.DirectoryStore(zarr_path)
            
        try:
            root = zarr.open_group(store=store, mode='r')
        except Exception:
            root = zarr.open_group(store=store)
            
        voxels = root['voxels'][:]  # (N, C, NX, NY, NZ)
        
        # Calculate sum of energy (over axes 1,2,3,4) for each sample
        e_sums = np.sum(voxels, axis=(1, 2, 3, 4))
        
        # Calculate number of non-zero voxels for each sample
        v_counts = np.sum(voxels > 0, axis=(1, 2, 3, 4))
        
        total_energies.extend(e_sums)
        lit_voxels_counts.extend(v_counts)
        
    total_energies = np.array(total_energies)
    lit_voxels_counts = np.array(lit_voxels_counts)
    
    print(f"Processed {len(total_energies)} total single-flash samples.")
    print(f"Energy sum stats - Min: {np.min(total_energies):.1f}, Max: {np.max(total_energies):.1f}, Mean: {np.mean(total_energies):.1f}")
    print(f"Lit voxels stats - Min: {np.min(lit_voxels_counts)}, Max: {np.max(lit_voxels_counts)}, Mean: {np.mean(lit_voxels_counts):.1f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Energy Distribution
    bins_e = np.linspace(0, np.percentile(total_energies, 99), 50)
    ax1.hist(total_energies, bins=bins_e, color='teal', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Total Voxel Energy Sum (per sample)')
    ax1.set_ylabel('Count')
    ax1.set_title('Energy Distribution of Single-Flash Events\n(Bottom 99% of values)')
    
    # 2. Lit Voxel Counts
    bins_v = np.linspace(0, np.percentile(lit_voxels_counts, 99), 50)
    ax2.hist(lit_voxels_counts, bins=bins_v, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Number of Lit Voxels (energy > 0)')
    ax2.set_ylabel('Count')
    ax2.set_title('Lit Voxel Distribution of Single-Flash Events\n(Bottom 99% of values)')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved distribution plot to: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot distributions of total energy and lit voxels from compiled Zarrs")
    parser.add_argument("--zarr-dir", type=str, required=True, help="Path to the directory containing .zarr files")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the plot image")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "zarr_distributions.png")
    plot_distributions(args.zarr_dir, out_file)
