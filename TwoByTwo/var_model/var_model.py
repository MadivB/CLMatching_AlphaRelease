
import math
import torch
import torch.nn as nn
import numpy as np
import os

# ==============================================================================
# MODEL ARCHITECTURE (v2.0 TPC-Aware)
# Copied from var2D/train_weight_vGPT_2D_v2.py
# ==============================================================================


# ==============================================================================
# MODEL ARCHITECTURE (v3.test Two-Sided TPC-Aware)
# Copied from train.py
# ==============================================================================

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, dim*expansion_factor), 
            Swish(), 
            nn.Dropout(dropout), 
            nn.Linear(dim*expansion_factor, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2*dim, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=(kernel_size-1)//2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.swish = Swish()
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        # x: (Batch, Length, Dim) -> (Batch, Dim, Length) for Conv1d
        x = self.ln(x).transpose(1, 2)
        return self.do(self.pw2(self.swish(self.bn(self.dw(self.glu(self.pw1(x))))))).transpose(1, 2)

class MultiHeadAttributes(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        x_ln = self.ln(x)
        return self.do(self.mha(x_ln, x_ln, x_ln, need_weights=False)[0])

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, dropout=dropout)
        self.attn = MultiHeadAttributes(dim, num_heads=num_heads, dropout=dropout)
        self.conv = ConvolutionModule(dim, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, dropout=dropout)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)

class ConformerVarPredictor2D_TwoSided(nn.Module):
    def __init__(self, length=1000, num_channels=48, model_dim=128, num_layers=4, num_heads=4, num_tpcs=8):
        super().__init__()
        # Input Projection: 48 channels -> model_dim
        self.input_proj = nn.Linear(num_channels, model_dim)
        self.register_buffer('pos_encoding', self._create_sinusoidal_embeddings(length, model_dim), persistent=False)
        self.layers = nn.ModuleList([ConformerBlock(model_dim, num_heads=num_heads) for _ in range(num_layers)])
        
        # Output Projection: model_dim -> 96 channels (48 Up, 48 Down)
        # We interleave: [ch0_up, ch0_down, ch1_up, ch1_down, ...] ? 
        # Wait, check train.py structure.
        # "logvar_reshaped = logvar_all.view(B, 48, 2, T)" implies the output is (B, 96, T) or (B, T, 96).
        # In forward: "out = self.output_proj(x_emb)" -> (B, T, 96)
        # "return logvar.permute(0, 2, 1)" -> (B, 96, T)
        # Then view(B, 48, 2, T) means the 96 dimension is split into 48x2.
        # So linearly it is [ch0, ch1, ... ch47] but doubled? No.
        # It depends on how view maps memory. 
        # If shape is (48, 2), then index 0 is (0,0), index 1 is (0,1), index 2 is (1,0).
        # So yes, interleaved [ch0_0, ch0_1, ch1_0, ch1_1...]
        
        self.output_proj = nn.Linear(model_dim, num_channels * 2) 
        
        # TPC-Dependent Gain Scaling
        self.tpc_gain = nn.Embedding(num_tpcs, num_channels)
        nn.init.constant_(self.tpc_gain.weight, 1.0)

    def _create_sinusoidal_embeddings(self, length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe.unsqueeze(0)

    def forward(self, x, tpc_ids):
        # x: (Batch, 48, 1000)
        
        # Apply TPC Gain
        gains = self.tpc_gain(tpc_ids)
        x = x * gains.unsqueeze(-1)
        
        x = x.permute(0, 2, 1) 
        x_emb = self.input_proj(x) + self.pos_encoding.to(x.device)[:, :x.shape[1], :]
        for layer in self.layers: x_emb = layer(x_emb)
        out = self.output_proj(x_emb) # (Batch, 1000, 96)
        
        # Clamp logvar
        logvar = torch.clamp(out, min=-12.0, max=12.0)
        
        # Permute back to (Batch, 96, 1000)
        return logvar.permute(0, 2, 1)

# ==============================================================================
# LIBRARY FUNCTIONS
# ==============================================================================

def load_model(checkpoint_path, device='cpu', length=1000, num_channels=48, num_tpcs=8):
    """
    Loads the ConformerVarPredictor2D_TwoSided model from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Initialize Model - Two Sided
    model = ConformerVarPredictor2D_TwoSided(length=length, num_channels=num_channels, num_tpcs=num_tpcs)
    model.to(device)
    
    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint dict and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Handle DDP 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Model loaded from {checkpoint_path} to {device}")
    return model

def predict(model, waveforms, tpc_ids, batch_size=64, input_scale=1e-3, device='cpu'):
    """
    Runs inference and returns the AVERAGE sigma from the two-sided output.
    """
    
    # --- Input Handling ---
    if isinstance(waveforms, torch.Tensor):
        waveforms_np = waveforms.detach().cpu().numpy()
    else:
        waveforms_np = np.array(waveforms)

    # Check Dimensions
    if waveforms_np.ndim == 2: # (48, 1000) -> Add batch dim
        if waveforms_np.shape != (48, 1000):
            raise ValueError(f"Expected single sample shape (48, 1000), got {waveforms_np.shape}")
        waveforms_np = waveforms_np[np.newaxis, ...] # (1, 48, 1000)
    elif waveforms_np.ndim == 3: # (N, 48, 1000)
        if waveforms_np.shape[1:] != (48, 1000):
            raise ValueError(f"Expected batch shape (N, 48, 1000), got {waveforms_np.shape}")
    else:
         raise ValueError(f"Input waveforms must be 2D (48, 1000) or 3D (N, 48, 1000). Got ndim={waveforms_np.ndim}")

    N_samples = waveforms_np.shape[0]

    # --- TPC ID Handling ---
    if isinstance(tpc_ids, int):
        tpc_ids_np = np.full((N_samples,), tpc_ids, dtype=int)
    else:
        if isinstance(tpc_ids, torch.Tensor):
            tpc_ids_np = tpc_ids.detach().cpu().numpy().astype(int)
        else:
            tpc_ids_np = np.array(tpc_ids, dtype=int)
        
        # Handle scalar array
        if tpc_ids_np.ndim == 0:
             tpc_ids_np = np.full((N_samples,), tpc_ids_np.item(), dtype=int)
        elif tpc_ids_np.ndim == 1:
            if len(tpc_ids_np) == 1 and N_samples > 1:
                 tpc_ids_np = np.full((N_samples,), tpc_ids_np[0], dtype=int)
            elif len(tpc_ids_np) != N_samples:
                 raise ValueError(f"Length of tpc_ids ({len(tpc_ids_np)}) does not match number of samples ({N_samples})")

    # --- Inference Loop ---
    predictions = []
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i in range(0, N_samples, batch_size):
            end = min(i + batch_size, N_samples)
            
            # Prepare Batch
            batch_wave = waveforms_np[i:end].astype(np.float32)
            batch_tpc = tpc_ids_np[i:end].astype(int) 
            
            # To Tensor
            batch_wave_t = torch.from_numpy(batch_wave).to(device)
            batch_tpc_t = torch.from_numpy(batch_tpc).to(device).long()
            
            # Scale
            batch_wave_t = batch_wave_t * input_scale
            
            # Forward -> (B, 96, 1000)
            logvar_all = model(batch_wave_t, batch_tpc_t)
            
            # Split outputs
            B, _, T = logvar_all.shape
            logvar_reshaped = logvar_all.view(B, 48, 2, T)
            
            logvar_up = logvar_reshaped[:, :, 0, :]
            logvar_down = logvar_reshaped[:, :, 1, :]
            
            std_up = torch.exp(0.5 * logvar_up)
            std_down = torch.exp(0.5 * logvar_down)
            
            # Return Average Sigma
            std_avg = (std_up + std_down) / 2.0
            
            predictions.append(std_avg.cpu().numpy())
            
    # Concatenate
    all_preds = np.concatenate(predictions, axis=0) # (N, 48, 1000)
    
    # If input was single sample, return single sample
    if N_samples == 1 and waveforms.ndim == 2: 
        return all_preds[0]
        
    return all_preds

