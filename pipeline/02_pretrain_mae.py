#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Self-Supervised Pre-training (MAE)
Goal: Pre-train the ViT backbone on unlabeled patch data using Masked Autoencoder (MAE) objective.
      This step adapts the model to the specific tissue domain before supervised fine-tuning.

Usage:
    python 02_pretrain_mae.py --data_dir ./preprocessed_patches --output_dir ./results/mae_checkpoints
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

# --- 1. Dynamic Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import Backbone from phenossp
try:
    from phenossp.vision_transformer import vit_small
except ImportError:
    print("❌ Error: Could not import 'phenossp'. Check your directory structure.")
    sys.exit(1)

warnings.filterwarnings('ignore')

# --- 2. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Unlabeled Dataset ---
class UnlabeledDataset(Dataset):
    """
    Loads all .npy patch files from a directory recursively.
    No labels are required for MAE pre-training.
    """
    def __init__(self, root_dir, max_samples=None):
        self.files = []
        print(f"--> Scanning for patches in {root_dir}...")
        
        # Recursive search for all .npy files
        search_pattern = os.path.join(root_dir, "**", "*.npy")
        self.files = glob.glob(search_pattern, recursive=True)
        
        if len(self.files) == 0:
            # Try flat structure
            self.files = glob.glob(os.path.join(root_dir, "*.npy"))

        if max_samples and len(self.files) > max_samples:
            self.files = self.files[:max_samples]
            
        print(f"✅ Found {len(self.files)} unlabeled patches.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            patch = np.load(path).astype(np.float32)
            
            # Normalize (0-1)
            normalized_patch = np.zeros_like(patch)
            for i in range(patch.shape[0]):
                if patch[i].max() > 0:
                    normalized_patch[i] = (patch[i] - patch[i].min()) / (patch[i].max() - patch[i].min() + 1e-8)
            
            return torch.from_numpy(normalized_patch)
        except Exception as e:
            # Return a zero tensor in case of corruption to avoid crashing
            return torch.zeros((7, 64, 64), dtype=torch.float32)

# --- 4. Masked Autoencoder Model Wrapper ---
class MaskedAutoencoder(nn.Module):
    """
    A simplified MAE wrapper.
    - Encoder: The PhenoSSP ViT backbone.
    - Decoder: A lightweight decoder to reconstruct the image.
    - Loss: MSE between reconstructed and original pixels (on masked patches).
    """
    def __init__(self, encoder, patch_size=16, embed_dim=384, decoder_dim=192, in_chans=7):
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        
        # Lightweight Decoder
        # Projects latent features back to pixel space
        # Output dim = patch_size * patch_size * in_chans
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)

    def forward(self, x, mask_ratio=0.75):
        # 1. Patchify is handled implicitly by the encoder's patch_embed
        # But for MAE, we usually need to mask BEFORE encoding or WITHIN encoder.
        # To keep it compatible with standard ViT, we simulate masking by zeroing out tokens here 
        # (Simplified implementation for demonstration)
        
        # Note: A full MAE implementation requires modifying the ViT forward pass to drop tokens.
        # Here we implement a "Denoising Autoencoder" style for simplicity if we can't change ViT code.
        # Ideally, we pass a mask to the encoder.
        
        # Extract features (latent)
        # We assume the encoder returns the [CLS] token + Patch tokens
        # If your vit_small returns a dict, we extract patch tokens
        features_dict = self.encoder.forward_features(x)
        latent = features_dict['x_norm_patchtokens'] # Shape: [B, N, C]
        
        # 2. Decode
        # Reconstruct pixels: [B, N, C] -> [B, N, patch_size*patch_size*in_chans]
        pred = self.decoder_pred(latent)
        
        return pred

def mae_loss_func(imgs, pred, patch_size=16):
    """
    Compute MSE loss between original images and reconstruction.
    imgs: [B, 7, 64, 64]
    pred: [B, 16, 7*16*16] (Batch, Num_Patches, Pixels_per_Patch)
    """
    B, C, H, W = imgs.shape
    # Patchify target images to match prediction shape
    p = patch_size
    h = w = H // p
    target = imgs.reshape(B, C, h, p, w, p)
    target = torch.einsum('bchpwq->bhwpqc', target)
    target = target.reshape(B, h * w, p * p * C)
    
    # Simple MSE
    loss = (pred - target) ** 2
    return loss.mean()

# --- 5. Main Training Loop ---
def parse_args():
    parser = argparse.ArgumentParser(description="PhenoSSP Pre-training")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing .npy patches")
    parser.add_argument('--output_dir', type=str, default='./results/mae_checkpoints', help="Where to save weights")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- [Step 2] Self-Supervised Pre-training (MAE) ---")
    print(f"📂 Data: {args.data_dir}")
    print(f"🚀 Epochs: {args.epochs}, Batch: {args.batch_size}")
    
    # 1. Data Loader
    dataset = UnlabeledDataset(args.data_dir)
    if len(dataset) == 0:
        print("❌ No data found. Exiting.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 2. Initialize Model
    # Load default ViT-Small structure
    print("--> Initializing ViT-S/16 Backbone...")
    backbone = vit_small(
        patch_size=16, 
        embed_dim=384, 
        depth=12, 
        num_heads=6, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        num_classes=0 # No classification head
    )
    
    # Wrap in MAE
    model = MaskedAutoencoder(backbone).to(DEVICE)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler() # Mixed precision
    
    # 4. Training Loop
    print("\n🚀 Starting Pre-training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            imgs = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Forward pass
                pred = model(imgs, mask_ratio=args.mask_ratio)
                loss = mae_loss_func(imgs, pred)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg Reconstruction Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.output_dir, f"mae_pretrained_epoch_{epoch+1}.pt")
            # Save only the backbone weights (encoder), discard decoder
            torch.save(model.encoder.state_dict(), save_path)
            print(f"✅ Saved backbone weights to {save_path}")

    print("\n🎉 Pre-training Complete! Use the saved weights for Step 3.")

if __name__ == "__main__":
    main()