#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils: Cell Phenotyping Model Manager

[Refactored for Release]
This module handles the initialization and loading of the PhenoSSP Vision Transformer backbone.
Note: 
- Training loops have been moved to `pipeline/03_finetune_classifier.py` for modularity.
- Data loaders are defined locally in pipeline scripts to reduce coupling.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import warnings

# Import backbone from the local package
try:
    from phenossp.vision_transformer import vit_small
except ImportError:
    # Fallback for relative imports if needed
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from phenossp.vision_transformer import vit_small

class CellPhenotypingDataset(Dataset):
    """
    A generic dataset class for loading .npy cell patches.
    (Kept as a utility for custom experiments)
    """
    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        try:
            # Load patch (C, H, W)
            patch = np.load(path).astype(np.float32)
            # Normalize to 0-1
            if patch.max() > 0:
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
            
            patch = torch.from_numpy(patch)
            if self.transform:
                patch = self.transform(patch)
            return patch
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((7, 64, 64))

class CellPhenotyping:
    """
    Main wrapper to manage PhenoSSP model initialization and weight loading.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary containing 'model_type', 'pretrained_path', etc.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_weights(self, model, path):
        """Helper to load weights safely, handling prefix mismatches."""
        if not os.path.exists(path):
            print(f"⚠️ Warning: Weight file not found at {path}. Using random initialization.")
            return model

        print(f"--> Loading backbone weights from: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # 1. Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 2. Fix 'module.' prefix (from DataParallel training)
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                # Only load backbone weights, ignore classification heads if they mismatch
                if "head" not in new_key: 
                    clean_state_dict[new_key] = v
            
            # 3. Load with strict=False to allow flexibility (e.g. MAE encoder -> Classifier)
            msg = model.load_state_dict(clean_state_dict, strict=False)
            print(f"    Weight loading status: {msg}")
            
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            print("    Proceeding with random weights (Warning!)")
            
        return model

    def load_model(self):
        """
        Initializes the ViT backbone and loads weights.
        Returns:
            model (nn.Module): The backbone model.
            optimizer (None): Deprecated placeholder.
            scheduler (None): Deprecated placeholder.
        """
        model_type = self.config.get('model_type', 'vits16')
        pretrained_path = self.config.get('pretrained_path', None)
        
        # 1. Initialize Architecture
        if model_type == 'vits16':
            # Initialize standard ViT-S/16 with 7 input channels
            model = vit_small(
                patch_size=16, 
                embed_dim=384, 
                num_classes=0 # No head, just features
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 2. Load Weights
        if pretrained_path:
            model = self._load_weights(model, pretrained_path)
            
        model.to(self.device)
        
        # Return signature matches the old code for compatibility
        return model, None, None