#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Cohort Inference
Goal: Apply the trained hierarchical PhenoSSP model to a new cohort of image patches.
      This script generates cell phenotype predictions for every cell in the input directory.

Usage:
    python 04_inference_cohort.py \
        --data_dir ./preprocessed_patches \
        --model_dir ./results/models \
        --output_dir ./results/inference
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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

# --- 3. Model Definitions (Must match training structure) ---
class CoarseClassifier(nn.Module):
    def __init__(self, phenossp_model, num_coarse_classes, embedding_dim=384, dropout_rate=0.0):
        super(CoarseClassifier, self).__init__()
        self.phenossp_model = phenossp_model
        # Note: Dropout usually disabled during inference, but structure must match
        self.coarse_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, num_coarse_classes))
            
    def forward(self, x, marker_ids=None):
        features_dict = self.phenossp_model.forward_features(x, marker_ids=marker_ids)
        features = features_dict['x_norm_clstoken']
        return self.coarse_classifier(features)

class ImmuneFineTuner(nn.Module):
    def __init__(self, phenossp_model, num_immune_fine_classes, embedding_dim=384, dropout_rate=0.0):
        super(ImmuneFineTuner, self).__init__()
        self.phenossp_model = phenossp_model
        self.immune_fine_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate * 0.6),
            nn.Linear(64, num_immune_fine_classes))
            
    def forward(self, x, marker_ids=None):
        features_dict = self.phenossp_model.forward_features(x, marker_ids=marker_ids)
        features = features_dict['x_norm_clstoken']
        return self.immune_fine_classifier(features)

# --- 4. Inference Dataset ---
class InferenceDataset(Dataset):
    """
    Loads patches for inference. Returns (patch_tensor, patch_path, metadata).
    Metadata includes inferred cohort/sample/cell_id from path.
    """
    def __init__(self, root_dir):
        self.files = []
        print(f"--> Scanning for patches in {root_dir}...")
        
        # Recursive search
        self.files = glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)
        
        print(f"✅ Found {len(self.files)} patches to process.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        
        # Parse Metadata from path
        # Expected formats:
        # 1. root/cohort/sample/cell_123.npy
        # 2. root/sample/cell_123.npy
        # 3. root/cell_123.npy
        
        filename = os.path.basename(path)
        cell_id = filename.replace('cell_', '').replace('.npy', '')
        
        parts = path.split(os.sep)
        if len(parts) >= 3 and parts[-3] != 'patches': # Cohort/Sample/Cell
            cohort = parts[-3]
            sample = parts[-2]
        elif len(parts) >= 2: # Sample/Cell
            cohort = 'Unknown'
            sample = parts[-2]
        else:
            cohort = 'Unknown'
            sample = 'Unknown'

        try:
            patch = np.load(path).astype(np.float32)
            # Normalize
            normalized_patch = np.zeros_like(patch)
            for i in range(patch.shape[0]):
                if patch[i].max() > 0:
                    normalized_patch[i] = (patch[i] - patch[i].min()) / (patch[i].max() - patch[i].min() + 1e-8)
            
            return torch.from_numpy(normalized_patch), path, cohort, sample, cell_id
            
        except Exception as e:
            # Return dummy if failed
            print(f"Error loading {path}: {e}")
            return torch.zeros((7, 64, 64)), path, cohort, sample, cell_id

# --- 5. Main Inference Logic ---
def parse_args():
    parser = argparse.ArgumentParser(description="PhenoSSP Inference Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing input .npy patches")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing trained .pth models")
    parser.add_argument('--output_dir', type=str, default='./results/inference', help="Where to save predictions")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for inference")
    parser.add_argument('--pretrained_mae', type=str, default=None, 
                        help="Optional: Path to MAE weights. If not provided, uses default init.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- [Step 3] Cohort Inference ---")
    print(f"📂 Data Source: {args.data_dir}")
    print(f"🤖 Models: {args.model_dir}")
    
    # 1. Load Data
    dataset = InferenceDataset(args.data_dir)
    if len(dataset) == 0:
        print("❌ No data found. Exiting.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Load Models
    print("\n--> Loading Models...")
    
    # Paths
    coarse_path = os.path.join(args.model_dir, "coarse_model_final.pth")
    immune_path = os.path.join(args.model_dir, "immune_model_final.pth")
    
    if not os.path.exists(coarse_path) or not os.path.exists(immune_path):
        print(f"❌ Missing model files in {args.model_dir}. Expected 'coarse_model_final.pth' and 'immune_model_final.pth'")
        return

    # Load Checkpoints to get Label Maps
    coarse_ckpt = torch.load(coarse_path, map_location=DEVICE)
    immune_ckpt = torch.load(immune_path, map_location=DEVICE)
    
    coarse_map = coarse_ckpt['coarse_label_to_idx'] # {'Epithelial': 0, ...}
    immune_map = immune_ckpt['immune_fine_label_to_idx'] # {'CD4+ Treg': 4, ...}
    
    # Reverse maps for decoding
    idx_to_coarse = {v: k for k, v in coarse_map.items()}
    idx_to_immune = {v: k for k, v in immune_map.items()}
    
    print(f"✅ Loaded Label Maps.")
    print(f"   Coarse: {list(coarse_map.keys())}")
    print(f"   Immune: {list(immune_map.keys())}")

    # Init Backbone
    # Note: We use a dummy config to init the ViT structure using CellPhenotyping utils if available
    # Or just init standard ViT. Here we assume standard initialization.
    phenossp_config = {
        "project_dir": args.output_dir,
        "pretrained_path": args.pretrained_mae if args.pretrained_mae else "dummy",
        "checkpoint_path": args.pretrained_mae if args.pretrained_mae else "dummy",
        "model_type": "vits16", "token_overlap": True, "num_classes": 3, "hf_auth_token": None, "cache_dir": args.output_dir
    }
    
    try:
        backbone, _, _ = CellPhenotyping(phenossp_config).load_model()
    except:
        # Fallback manual init if utils fail
        from phenossp.vision_transformer import vit_small
        backbone = vit_small(patch_size=16, embed_dim=384, num_classes=0)

    backbone.to(DEVICE)
    
    # Init Heads
    coarse_model = CoarseClassifier(backbone, len(coarse_map)).to(DEVICE)
    coarse_model.load_state_dict(coarse_ckpt['model_state_dict'])
    coarse_model.eval()
    
    immune_model = ImmuneFineTuner(backbone, len(immune_map)).to(DEVICE)
    immune_model.load_state_dict(immune_ckpt['model_state_dict'])
    immune_model.eval()
    
    print("✅ Models initialized successfully.")

    # 3. Inference Loop
    results = []
    
    print("\n🚀 Starting Batch Inference...")
    with torch.no_grad():
        for batch_idx, (patches, paths, cohorts, samples, cell_ids) in enumerate(tqdm(loader)):
            patches = patches.to(DEVICE)
            marker_ids = torch.arange(7, dtype=torch.long).unsqueeze(0).expand(patches.size(0), -1).to(DEVICE)
            
            # -- Stage 1: Coarse --
            coarse_logits = coarse_model(patches, marker_ids)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            coarse_preds = torch.argmax(coarse_probs, dim=1)
            
            # Prepare Stage 2 inputs
            immune_indices = (coarse_preds == coarse_map['Immune']).nonzero(as_tuple=True)[0]
            
            # Default predictions (Stage 1)
            batch_final_labels = [idx_to_coarse[p.item()] for p in coarse_preds]
            batch_final_confs = [coarse_probs[i, p].item() for i, p in enumerate(coarse_preds)]
            
            # -- Stage 2: Fine (Only for Immune) --
            if len(immune_indices) > 0:
                immune_input = patches[immune_indices]
                immune_marker_ids = marker_ids[immune_indices]
                
                immune_logits = immune_model(immune_input, immune_marker_ids)
                immune_probs = torch.softmax(immune_logits, dim=1)
                immune_preds = torch.argmax(immune_probs, dim=1)
                
                # Update predictions
                for i, idx in enumerate(immune_indices):
                    idx = idx.item()
                    fine_pred = immune_preds[i].item()
                    fine_prob = immune_probs[i, fine_pred].item()
                    
                    batch_final_labels[idx] = idx_to_immune[fine_pred]
                    batch_final_confs[idx] = fine_prob # Or combine with coarse prob
            
            # Collect Batch Results
            for i in range(len(patches)):
                results.append({
                    'cohort': cohorts[i],
                    'sample_id': samples[i],
                    'cell_id': cell_ids[i],
                    'patch_path': paths[i],
                    'phenotype': batch_final_labels[i],
                    'confidence': batch_final_confs[i]
                })

    # 4. Save Results
    df = pd.DataFrame(results)
    
    # Save raw predictions
    csv_path = os.path.join(args.output_dir, "cohort_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Predictions saved to: {csv_path}")
    
    # 5. Generate Summary Report
    print("\n📊 Cohort Summary:")
    summary = df['phenotype'].value_counts()
    print(summary)
    
    summary_path = os.path.join(args.output_dir, "phenotype_summary.csv")
    summary.to_csv(summary_path)
    print(f"✅ Summary stats saved to: {summary_path}")

if __name__ == "__main__":
    main()