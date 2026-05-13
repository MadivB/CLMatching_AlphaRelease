#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# Import model and dataset from the training script
from train_ndfull import HybridPerceiver3D, MultiZarrDataset, fit_target_standardizer_masked, set_seed

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = "./zarr_outputs_enhanced"
    checkpoint_path = "./runs/ndfull_run_distributed/checkpoint.pt"
    
    print(f"Loading data from {input_dir}")
    zpaths = sorted([os.path.join(input_dir, d) for d in os.listdir(input_dir) if d.endswith(".zarr")])
    if not zpaths:
        print("No zarr files found!")
        return

    # Use same target scale as training default
    target_scale = 1e-3
    ds_full = MultiZarrDataset(zpaths, target_scale=target_scale)
    
    val_frac = 0.1
    n_val = max(1, int(val_frac * len(ds_full)))
    n_train = len(ds_full) - n_val
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    print(f"Train/Val split: {n_train}/{n_val}")

    # Standardizer parameters from training
    dead_sentinel = -10000.0
    zero_below_raw = 100.0
    sentinel_scaled = dead_sentinel * target_scale
    zero_below_scaled = zero_below_raw * target_scale

    # Fit standardizer on train set (using a small loader to make it fast like train_ndfull)
    # The original script uses targets-only logic internally
    # We pass ds_train to the DataLoader but fit_target_standardizer_masked will extract the full dataset
    # Wait, fit_target_standardizer_masked expects something with loader.dataset.dataset 
    # to access the full dataset targets. 
    train_loader = DataLoader(ds_train, batch_size=256, shuffle=False, num_workers=4)
    print("Fitting Target Standardizers...")
    stdzr = fit_target_standardizer_masked(train_loader, device=device, 
                                           sentinel_scaled=sentinel_scaled, 
                                           zero_below_scaled=zero_below_scaled)

    # Initialize Model
    print(f"Loading Model from {checkpoint_path}...")
    model = HybridPerceiver3D(c_in=ds_full.C, num_targets=ds_full.K).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
        
    # Handle DDP "module." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    val_loader = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=4)
    
    all_preds = []
    all_true = []
    
    inv_target_scale = 1.0 / target_scale

    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for i, (x, y, tpc) in enumerate(val_loader):
            if i >= 50:  # Only take 50 batches for visualization to save time
                break
                
            x, y, tpc = x.to(device), y.to(device), tpc.to(device)
            valid_mask = (y != sentinel_scaled)
            y_eff = torch.where(valid_mask & (y < zero_below_scaled), torch.zeros_like(y), y)
            
            # Use AMP autocast just like in training
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=False):
                out = model(x, tpc_ids=tpc)
                pred = out["final"]

            pred_raw = pred * inv_target_scale
            y_raw = y_eff * inv_target_scale
            
            for b in range(x.shape[0]):
                valid_indices = valid_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    all_preds.extend(pred_raw[b, valid_indices].cpu().numpy())
                    all_true.extend(y_raw[b, valid_indices].cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    print(f"Collected {len(all_preds)} valid points.")
    
    # Visualization
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    
    if len(all_true) > 0:
        max_val = np.percentile(all_true, 99) * 1.5
        min_val = min(0, np.min(all_true))
        
        # Calculate R^2 for display
        correlation_matrix = np.corrcoef(all_true, all_preds)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        
        plt.hist2d(all_true, all_preds, bins=100, range=[[min_val, max_val], [min_val, max_val]], cmap='viridis', cmin=1)
        plt.colorbar(label='Number of Targets')
        
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label=f'Ideal Prediction (R²={r_squared:.3f})')
        
        plt.xlabel('Actual Amplitude: True Light / Charge')
        plt.ylabel('Predicted Amplitude')
        plt.title('Predicted vs Actual Amplitude (Validation Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_name = "predicted_vs_actual_amplitude.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {out_name}")
    else:
        print("No valid points found to plot.")

if __name__ == "__main__":
    main()
