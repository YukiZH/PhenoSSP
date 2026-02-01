#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Visual Interpretability (Saliency Maps)
Goal: Visualize pixel-level model attention to verify subcellular localization.
      (e.g., CD8 attention on membrane, FoxP3 attention on nucleus).

Key Logic:
    Uses Gradient-weighted Class Activation Mapping (or raw Gradient Saliency)
    to highlight which pixels contributed most to the prediction.

Usage:
    python 05_visual_interpretability.py \
        --patch_dir ./demo_data/patches \
        --model_path ./results/models/coarse_model_final.pth \
        --output_dir ./results/interpretability \
        --target_marker CD8 \
        --use_demo_filter
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

# --- 1. Dynamic Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "utils"))

try:
    from utils.cell_phenotyping import CellPhenotyping
except ImportError:
    pass

warnings.filterwarnings('ignore')

# --- 2. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL_NAMES = ['DAPI', 'CD8', 'FoxP3', 'PanCK', 'PD1', 'CD4', 'CD3']

# --- 3. Model Definition (Wrapper) ---
class CoarseClassifier(nn.Module):
    def __init__(self, phenossp_model, num_coarse_classes, embedding_dim=384):
        super(CoarseClassifier, self).__init__()
        self.phenossp_model = phenossp_model
        self.coarse_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_coarse_classes))
            
    def forward(self, x, marker_ids=None):
        features_dict = self.phenossp_model.forward_features(x, marker_ids=marker_ids)
        return self.coarse_classifier(features_dict['x_norm_clstoken'])

# --- 4. Core Saliency Functions ---
def compute_saliency_map(model, patch, marker_ids, target_class_idx):
    """
    Computes Input Gradients (Saliency Map).
    Returns the pixel-wise importance for the target class.
    """
    model.eval()
    patch.requires_grad_()
    
    # Forward
    logits = model(patch, marker_ids)
    score = logits[0, target_class_idx]
    
    # Backward
    score.backward()
    
    # Get gradient magnitude (max across channels usually works best for structure)
    # Shape: (1, 7, 64, 64) -> (64, 64)
    saliency = patch.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
    return saliency

def normalize_image(img):
    """Normalize image to 0-1 range for visualization."""
    if img.max() == img.min(): return np.zeros_like(img)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def plot_interpretation(patch_numpy, saliency, cell_id, target_marker, score, save_path):
    """
    Generates the Figure 4 style visualization:
    Input (RGB) | Attention (Heatmap) | Overlay
    """
    # 1. Prepare RGB Composite
    # Blue = DAPI (Ch 0)
    # Green = Target Marker (CD8=Ch 1, FoxP3=Ch 2)
    marker_idx = CHANNEL_NAMES.index(target_marker)
    
    rgb_img = np.zeros((64, 64, 3))
    rgb_img[:,:,1] = normalize_image(patch_numpy[marker_idx]) # Green: Target
    rgb_img[:,:,2] = normalize_image(patch_numpy[0])          # Blue: DAPI
    
    # 2. Prepare Saliency Heatmap
    # Clip extreme outliers (top 0.1%) for better contrast
    v_max = np.percentile(saliency, 99.9) 
    norm_sal = np.clip(saliency / (v_max + 1e-8), 0, 1)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * norm_sal), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # 3. Plotting
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.1)
    
    # Panel A: Biological Input
    ax0 = plt.subplot(gs[0])
    ax0.imshow(np.clip(rgb_img * 1.2, 0, 1)) # Slight brighten
    ax0.set_title(f"Input: DAPI (Blue) + {target_marker} (Green)", fontsize=10, color='white')
    ax0.axis('off')
    
    # Panel B: Model Attention
    ax1 = plt.subplot(gs[1])
    ax1.imshow(heatmap)
    ax1.set_title("Model Attention (Saliency)", fontsize=10, color='cyan')
    ax1.axis('off')
    
    # Panel C: Overlay (The "Proof")
    ax2 = plt.subplot(gs[2])
    # 70% Structure + 30% Heatmap
    overlay = 0.7 * rgb_img + 0.3 * heatmap
    ax2.imshow(np.clip(overlay, 0, 1))
    ax2.set_title(f"Overlay (Score: {score:.1f})", fontsize=10, color='yellow')
    ax2.axis('off')
    
    plt.suptitle(f"Cell ID: {cell_id} | Target: {target_marker}", color='white', y=0.95)
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='black')
    plt.close()

# --- 5. Main Loop ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/interpretability')
    parser.add_argument('--csv_path', type=str, default=None, help="Optional: Path to annotation CSV to filter specific cells")
    parser.add_argument('--target_marker', type=str, default='CD8', choices=['CD8', 'FoxP3'], help="Which marker to verify (CD8 or FoxP3)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- [Step 4] Visual Interpretability Analysis ---")
    print(f"Target: {args.target_marker} Localization")
    
    # 1. Load Model
    print("--> Loading Model...")
    # Dummy config to init ViT
    phenossp_config = {
        "project_dir": args.output_dir, "pretrained_path": "dummy", "checkpoint_path": "dummy",
        "model_type": "vits16", "token_overlap": True, "num_classes": 3, "hf_auth_token": None, "cache_dir": args.output_dir
    }
    
    try:
        backbone, _, _ = CellPhenotyping(phenossp_config).load_model()
    except:
        from phenossp.vision_transformer import vit_small
        backbone = vit_small(patch_size=16, embed_dim=384, num_classes=0)
        
    # Load Coarse Classifier weights
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    # Handle different checkpoint formats
    if 'coarse_label_to_idx' in checkpoint:
        num_classes = len(checkpoint['coarse_label_to_idx'])
        state_dict = checkpoint['model_state_dict']
        immune_idx = checkpoint['coarse_label_to_idx'].get('Immune', 1)
    else:
        num_classes = 3 # Default
        state_dict = checkpoint
        immune_idx = 1
        
    model = CoarseClassifier(backbone, num_classes).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 2. Collect Candidate Patches
    # Strategy: If CSV is provided, use it to find ground truth cells.
    # If not, scan folder and filter by signal intensity.
    
    candidates = []
    
    if args.csv_path and os.path.exists(args.csv_path):
        print(f"--> Filtering cells from {args.csv_path}...")
        df = pd.read_csv(args.csv_path)
        
        # Determine target phenotype string based on marker
        target_str = 'CD8' if args.target_marker == 'CD8' else 'FoxP3'
        
        # Filter dataframe
        subset = df[df['cell_type'].str.contains(target_str, na=False)]
        
        # Limit to 50 candidates to save time
        subset = subset.sample(min(len(subset), 50), random_state=42)
        
        for _, row in subset.iterrows():
            # Try to construct path (assuming flat or nested structure)
            fname = f"cell_{row['cell_id']}.npy"
            # Simple search
            found = glob.glob(os.path.join(args.patch_dir, "**", fname), recursive=True)
            if found:
                candidates.append(found[0])
    else:
        print("--> No CSV provided. Scanning directory for high-signal candidates...")
        all_patches = glob.glob(os.path.join(args.patch_dir, "**", "*.npy"), recursive=True)
        # Randomly pick 200 to check
        import random
        random.shuffle(all_patches)
        candidates = all_patches[:200]

    print(f"--> Analyzing {len(candidates)} candidates for optimal visualization...")
    
    # 3. Process Candidates & Filter Best Ones
    success_count = 0
    
    for patch_path in tqdm(candidates):
        try:
            patch_np = np.load(patch_path)
            
            # --- Scale-Invariant Contrast Filter (The Magic) ---
            # We want clear signals, not just bright ones.
            # Criterion: Max signal > 5x Background Noise (PanCK mean)
            
            target_idx = CHANNEL_NAMES.index(args.target_marker)
            panck_idx = CHANNEL_NAMES.index('PanCK')
            dapi_idx = CHANNEL_NAMES.index('DAPI')
            
            max_signal = patch_np[target_idx].max()
            mean_noise = patch_np[panck_idx].mean()
            max_dapi = patch_np[dapi_idx].max()
            
            # Filter Logic:
            # 1. Structure exists (Signal & DAPI not empty)
            # 2. Contrast is high (Signal >> Noise)
            if max_signal <= 0 or max_dapi <= 0: continue
            
            contrast_ratio = max_signal / (mean_noise + 1e-6)
            
            if contrast_ratio < 5.0: 
                continue # Skip noisy/dim cells
            
            # --- If passed filter, run Saliency ---
            
            # Normalize for Model
            norm_patch = np.zeros_like(patch_np, dtype=np.float32)
            for c in range(7):
                if patch_np[c].max() > 0:
                    norm_patch[c] = (patch_np[c] - patch_np[c].min()) / (patch_np[c].max() - patch_np[c].min() + 1e-8)
            
            tensor_patch = torch.from_numpy(norm_patch).unsqueeze(0).to(DEVICE)
            marker_ids = torch.arange(7, dtype=torch.long).unsqueeze(0).expand(1, -1).to(DEVICE)
            
            # Predict first to check if model actually thinks it's Immune
            logits = model(tensor_patch, marker_ids)
            pred_idx = torch.argmax(logits, 1).item()
            
            if pred_idx != immune_idx:
                continue # Skip if model classifies as Tumor/Other
            
            # Compute Saliency
            saliency = compute_saliency_map(model, tensor_patch, marker_ids, immune_idx)
            
            # Save Visualization
            cell_id = os.path.basename(patch_path).replace('.npy', '')
            save_name = os.path.join(args.output_dir, f"Attn_{args.target_marker}_{cell_id}_Score_{contrast_ratio:.1f}.png")
            
            plot_interpretation(patch_np, saliency, cell_id, args.target_marker, contrast_ratio, save_name)
            
            success_count += 1
            if success_count >= 10: break # Stop after finding 10 good examples
            
        except Exception as e:
            print(f"Error processing {patch_path}: {e}")
            continue

    print(f"\n✅ Generated {success_count} visualization plots in {args.output_dir}")
    if success_count == 0:
        print("⚠️ No ideal candidates found. Try adjusting the contrast threshold or checking data quality.")

if __name__ == "__main__":
    main()