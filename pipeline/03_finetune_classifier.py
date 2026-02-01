#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Supervised Fine-tuning
Goal: Train the hierarchical classifiers (Coarse & Immune Expert) using labeled data.

Usage:
    python 03_finetune_classifier.py \
        --patch_dir ./demo_data/patches \
        --annotation_csv ./demo_data/cell_annotations.csv \
        --pretrained_weights ./phenossp/weights/kronos_vits16_model.pt \
        --output_dir ./results/finetuned_models
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from tqdm import tqdm
import warnings
import glob

# --- 1. Setup Project Path (Dynamic) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "utils"))

try:
    from utils.cell_phenotyping import CellPhenotyping
except ImportError:
    # Fallback if utils structure is different
    pass

warnings.filterwarnings('ignore')

# --- 2. Configurations & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR_HEAD = 1e-4
LR_BACKBONE = 2e-6

# --- 3. Model Definitions ---
class CoarseClassifier(nn.Module):
    def __init__(self, phenossp_model, num_coarse_classes, embedding_dim=384, dropout_rate=0.5):
        super(CoarseClassifier, self).__init__()
        self.phenossp_model = phenossp_model
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
    def __init__(self, phenossp_model, num_immune_fine_classes, embedding_dim=384, dropout_rate=0.5):
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

# --- 4. Helper Classes & Functions ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience, self.min_delta, self.restore_best_weights = patience, min_delta, restore_best_weights
        self.best_score, self.counter, self.best_weights = None, 0, None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights: self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights: model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            if self.restore_best_weights: self.best_weights = model.state_dict().copy()
            self.counter = 0
        return False

def create_two_stage_labels(df):
    """Maps raw cell types to Coarse and Immune-Fine labels."""
    coarse_mapping = {
        'Epithelial': ['Epithelial cell'],
        'Immune': ['CD3+CD4+CD8- cell', 'CD3+CD4-CD8+ cell', 'CD3+CD4-CD8- cell','CD3+CD4+CD8+ cell', 
                   'CD3-CD4+CD8+ cell', 'CD3-CD4+CD8- cell','CD3-CD4-CD8+ cell', 'PANCK+CD3+ cell', 'CD4+ Treg'],
        'Other': ['other']
    }
    df['coarse_label'] = 'Other'
    for coarse_type, cell_types in coarse_mapping.items(): 
        df.loc[df['cell_type'].isin(cell_types), 'coarse_label'] = coarse_type
        
    immune_fine_mapping = {
        'CD3+CD4+CD8-': ['CD3+CD4+CD8- cell'], 
        'CD3+CD4-CD8-': ['CD3+CD4-CD8- cell'],
        'CD3-CD4+CD8-': ['CD3-CD4+CD8- cell'], 
        'CD3+CD4-CD8+': ['CD3+CD4-CD8+ cell'],
        'CD3+CD4+CD8+': ['CD3+CD4+CD8+ cell'], 
        'CD3-CD4+CD8+': ['CD3-CD4+CD8+ cell'],
        'CD3-CD4-CD8+': ['CD3-CD4-CD8+ cell'], 
        'PANCK+CD3+': ['PANCK+CD3+ cell'],
        'CD4+FOXP3+': ['CD4+ Treg']
    }
    df['fine_label'] = np.nan
    for fine_type, cell_types in immune_fine_mapping.items(): 
        df.loc[df['cell_type'].isin(cell_types), 'fine_label'] = fine_type
    return df

class FinetuneDataset(Dataset):
    def __init__(self, patches, labels, label_map):
        self.patches = patches
        self.labels = [label_map.get(l, -1) for l in labels]
        
    def __len__(self): return len(self.patches)
    
    def __getitem__(self, idx):
        patch, label = self.patches[idx].astype(np.float32), self.labels[idx]
        # Normalize: (0, 1) scaling per channel
        normalized_patch = np.zeros_like(patch)
        for i in range(patch.shape[0]):
            channel = patch[i]
            if channel.max() > 0: 
                normalized_patch[i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        return torch.from_numpy(normalized_patch), torch.tensor(label, dtype=torch.long)

def load_patch_data(df, patch_dir):
    """
    Efficiently loads .npy patches matched with the annotation dataframe.
    Looks for patches in: patch_dir/<cohort_id>/<sample_id>/cell_<cell_id>.npy
    OR flat structure: patch_dir/cell_<cell_id>.npy (for demo)
    """
    patches = []
    coarse_labels = []
    fine_labels = []
    
    print(f"--> Loading patches from {patch_dir}...")
    
    found_count = 0
    missing_count = 0
    
    # Pre-scan directory to speed up lookup (optional optimization)
    # Here we assume standard structure
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Data"):
        sample_id = str(row['sample_id'])
        cell_id = int(row['cell_id'])
        
        # Try finding the file using glob (flexible for folder structures)
        # Search pattern 1: patch_dir/cohort/sample/cell_id.npy
        # Search pattern 2: patch_dir/sample/cell_id.npy
        # Search pattern 3: patch_dir/cell_id.npy (Demo mode)
        
        # Construct specific filename
        filename = f"cell_{cell_id}.npy"
        
        # Check if file exists directly (Demo Mode)
        direct_path = os.path.join(patch_dir, filename)
        
        # Check nested (Production Mode) - Requires searching
        # Since searching every time is slow, we try specific paths first
        # Extract cohort from image_name if possible, e.g. "245517"
        cohort_guess = str(row.get('image_name', '')).split('.')[0] 
        
        nested_path_1 = os.path.join(patch_dir, cohort_guess, sample_id, filename)
        nested_path_2 = os.path.join(patch_dir, sample_id, filename)
        
        final_path = None
        if os.path.exists(direct_path):
            final_path = direct_path
        elif os.path.exists(nested_path_1):
            final_path = nested_path_1
        elif os.path.exists(nested_path_2):
            final_path = nested_path_2
        else:
            # Fallback: glob search (slower)
            search = glob.glob(os.path.join(patch_dir, "**", filename), recursive=True)
            if search:
                final_path = search[0]
        
        if final_path:
            try:
                patch = np.load(final_path)
                if patch.shape == (7, 64, 64): # Ensure correct shape
                    patches.append(patch)
                    coarse_labels.append(row['coarse_label'])
                    fine_labels.append(row['fine_label'])
                    found_count += 1
                else:
                    pass # Skip invalid shapes
            except:
                pass
        else:
            missing_count += 1
            
    print(f"✅ Loaded {found_count} patches. (Missing: {missing_count})")
    
    return {
        'patches': patches,
        'coarse_labels': coarse_labels,
        'fine_labels': fine_labels
    }

def train_engine(model, train_loader, val_loader, optimizer, model_name, save_path):
    """Generic Training Loop"""
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    early_stopper = EarlyStopping(patience=5, restore_best_weights=True)
    
    print(f"\n🚀 Training {model_name}...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        for patches, labels in train_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)
            marker_ids = torch.arange(7, dtype=torch.long).unsqueeze(0).expand(patches.size(0), -1).to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(patches, marker_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(DEVICE), labels.to(DEVICE)
                marker_ids = torch.arange(7, dtype=torch.long).unsqueeze(0).expand(patches.size(0), -1).to(DEVICE)
                logits = model(patches, marker_ids)
                preds = torch.argmax(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Filter -1 (unlabeled)
        valid_mask = np.array(all_labels) != -1
        if valid_mask.sum() > 0:
            val_acc = balanced_accuracy_score(np.array(all_labels)[valid_mask], np.array(all_preds)[valid_mask])
        else:
            val_acc = 0
            
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f} | Val Bal Acc: {val_acc:.4f}")
        
        if early_stopper(val_acc, model):
            print("🛑 Early stopping triggered.")
            break
            
    # Save Best Model
    torch.save(early_stopper.best_weights, save_path)
    print(f"✅ Saved best {model_name} to {save_path}")
    return model

# --- 5. Main Pipeline ---
def parse_args():
    parser = argparse.ArgumentParser(description="PhenoSSP Supervised Fine-tuning")
    parser.add_argument('--patch_dir', type=str, required=True, help="Directory containing preprocessed .npy patches")
    parser.add_argument('--annotation_csv', type=str, required=True, help="Path to cell annotations CSV")
    parser.add_argument('--pretrained_weights', type=str, required=True, help="Path to MAE pretrained weights (.pt)")
    parser.add_argument('--output_dir', type=str, default='./results/models', help="Directory to save finetuned models")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- [Step 2] Supervised Fine-Tuning ---")
    print(f"📂 Patches: {args.patch_dir}")
    print(f"📂 Annotations: {args.annotation_csv}")
    print(f"🔧 Device: {DEVICE}")

    # 1. Load Annotations
    df = pd.read_csv(args.annotation_csv)
    df = create_two_stage_labels(df)
    
    # Helper to extract sample_id if needed (customize regex based on your filename format)
    # Assumes format "245517.ome_A-1.ome.tiff" or similar
    if 'image_name' in df.columns and 'sample_id' not in df.columns:
        # Example extraction: try to split by underscore
        try:
            df['sample_id'] = df['image_name'].astype(str).apply(lambda x: x.split('_')[1] if '_' in str(x) else x)
        except:
            df['sample_id'] = df['image_name']
    
    # 2. Load Data (Patches)
    data_dict = load_patch_data(df, args.patch_dir)
    if len(data_dict['patches']) == 0:
        print("❌ No matching patches found! Check paths and IDs.")
        return

    # 3. Load Backbone (PhenoSSP)
    print("\n--> Loading Backbone...")
    # Initialize barebones ViT structure to load weights
    # We use CellPhenotyping helper to ensure config matches DINOv2 requirements
    phenossp_config = {
        "project_dir": args.output_dir,
        "pretrained_path": args.pretrained_weights, # Path to .pt
        "checkpoint_path": args.pretrained_weights,
        "model_type": "vits16", 
        "token_overlap": True, 
        "num_classes": 3, 
        "hf_auth_token": None,
        "cache_dir": args.output_dir
    }
    
    # Load backbone
    # Note: If this fails due to path issues, ensure 'utils.cell_phenotyping' is accessible
    backbone, _, _ = CellPhenotyping(phenossp_config).load_model()
    
    # If loading fine-tuned weights directly (not MAE), state dict keys might differ. 
    # Here we assume loading MAE pretrained weights.
    
    # 4. Stage 1: Coarse Classification
    print("\n--> Stage 1: Training Coarse Classifier...")
    coarse_labels_map = {'Epithelial': 0, 'Immune': 1, 'Other': 2}
    
    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        data_dict['patches'], data_dict['coarse_labels'], test_size=0.2, stratify=data_dict['coarse_labels']
    )
    
    train_dataset = FinetuneDataset(X_train, y_train, coarse_labels_map)
    val_dataset = FinetuneDataset(X_val, y_val, coarse_labels_map)
    
    # Weighted Sampler for Imbalance
    class_counts = pd.Series(y_train).value_counts()
    sample_weights = [1.0 / class_counts.get(l, 1e-6) for l in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Init Model
    coarse_model = CoarseClassifier(backbone, len(coarse_labels_map)).to(DEVICE)
    optimizer = optim.AdamW(coarse_model.parameters(), lr=LR_HEAD)
    
    # Train
    save_path_coarse = os.path.join(args.output_dir, "coarse_model_final.pth")
    best_coarse_model = train_engine(coarse_model, train_loader, val_loader, optimizer, "Coarse Classifier", save_path_coarse)
    
    # Save metadata
    torch.save({
        'model_state_dict': best_coarse_model.state_dict(),
        'coarse_label_to_idx': coarse_labels_map
    }, save_path_coarse)


    # 5. Stage 2: Immune Fine-tuning
    print("\n--> Stage 2: Training Immune Expert Classifier...")
    
    # Filter only Immune cells
    immune_indices = [i for i, label in enumerate(data_dict['coarse_labels']) if label == 'Immune']
    immune_patches = [data_dict['patches'][i] for i in immune_indices]
    immune_subtypes = [data_dict['fine_labels'][i] for i in immune_indices]
    
    # Filter out NaNs (unknown subtypes)
    valid_indices = [i for i, label in enumerate(immune_subtypes) if pd.notna(label)]
    immune_patches = [immune_patches[i] for i in valid_indices]
    immune_subtypes = [immune_subtypes[i] for i in valid_indices]
    
    if len(immune_patches) < 50:
        print("⚠️ Not enough immune cells for fine-tuning. Skipping Stage 2.")
    else:
        immune_labels_map = {label: idx for idx, label in enumerate(sorted(set(immune_subtypes)))}
        print(f"Immune Subtypes found: {immune_labels_map}")
        
        # Split
        X_train_imm, X_val_imm, y_train_imm, y_val_imm = train_test_split(
            immune_patches, immune_subtypes, test_size=0.2, stratify=immune_subtypes
        )
        
        train_ds_imm = FinetuneDataset(X_train_imm, y_train_imm, immune_labels_map)
        val_ds_imm = FinetuneDataset(X_val_imm, y_val_imm, immune_labels_map)
        
        # Sampler
        cls_counts_imm = pd.Series(y_train_imm).value_counts()
        weights_imm = [1.0 / cls_counts_imm.get(l, 1e-6) for l in y_train_imm]
        sampler_imm = WeightedRandomSampler(weights_imm, len(weights_imm))
        
        train_loader_imm = DataLoader(train_ds_imm, batch_size=BATCH_SIZE, sampler=sampler_imm, num_workers=4)
        val_loader_imm = DataLoader(val_ds_imm, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Init Model (Share backbone from best coarse model)
        immune_model = ImmuneFineTuner(backbone, len(immune_labels_map)).to(DEVICE)
        # Optional: Freeze backbone for fine-tuning? Usually we fine-tune all but with lower LR.
        optimizer_imm = optim.AdamW(immune_model.parameters(), lr=LR_HEAD)
        
        # Train
        save_path_immune = os.path.join(args.output_dir, "immune_model_final.pth")
        best_immune_model = train_engine(immune_model, train_loader_imm, val_loader_imm, optimizer_imm, "Immune Expert", save_path_immune)
        
        torch.save({
            'model_state_dict': best_immune_model.state_dict(),
            'immune_fine_label_to_idx': immune_labels_map
        }, save_path_immune)

    print("\n🎉 All training stages completed!")

if __name__ == "__main__":
    main()