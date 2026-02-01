#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5: Spatial Statistics & Survival Analysis (Figure 5)
Goal: 
    1. Calculate the "Spatial Interaction Score" (Treg enrichment within 30um of CD8+ T cells).
    2. Perform Kaplan-Meier survival analysis to validate clinical relevance.
    3. Generate spatial plots for high/low risk samples.

Usage:
    python 06_spatial_analysis.py \
        --prediction_csv ./results/inference/cohort_predictions.csv \
        --clinical_csv ./demo_data/clinical_data.csv \
        --output_dir ./results/spatial_analysis \
        --radius 30
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import spearmanr
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings

# --- 1. Configuration ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# --- 2. Core Spatial Functions ---

def parse_coordinates_from_path(path_str):
    """
    Extracts (x, y) coordinates from patch filenames if embedded.
    Assumption: Filename format contains coordinates or we have a mapping.
    
    *CRITICAL*: Since patches are cropped, we usually need the original centroid coordinates.
    Option A: Read from the original `cell_annotations.csv` match by cell_id.
    Option B: If prediction_csv doesn't have coordinates, we must merge with geometry data.
    
    Here we assume the input CSV has 'centroid_x' and 'centroid_y' columns.
    If not, we try to merge with a metadata file.
    """
    return 0, 0 # Placeholder if needed, but we prefer pandas merge

def calculate_interaction_score(df, sample_id, radius=30, pixel_size=0.345):
    """
    Calculates the Density-Normalized Interaction Score for a single sample.
    Definition: Enrichment of Tregs within 'radius' (microns) of CD8+ T cells,
    normalized by the global density of Tregs to correct for abundance.
    
    Score = (Observed Neighbor Count / Expected Neighbor Count)
    """
    sample_df = df[df['sample_id'] == sample_id].copy()
    
    if len(sample_df) < 10: return np.nan # Skip empty/sparse samples
    
    # 1. Identify Populations
    # Note: Adjust phenotype labels based on your exact string in cohort_predictions.csv
    # e.g., 'CD3+CD4-CD8+' for Cytotoxic, 'CD4+ Treg' for Regulatory
    effectors = sample_df[sample_df['phenotype'].str.contains('CD8', na=False)]
    suppressors = sample_df[sample_df['phenotype'].str.contains('Treg|FoxP3', na=False, regex=True)]
    
    if len(effectors) == 0 or len(suppressors) == 0:
        return 0.0 # No interaction possible
        
    # 2. Get Coordinates (converted to microns)
    # Assumes columns 'centroid_x', 'centroid_y' exist in pixels
    eff_coords = effectors[['centroid_x', 'centroid_y']].values * pixel_size
    sup_coords = suppressors[['centroid_x', 'centroid_y']].values * pixel_size
    
    # 3. Build KDTree for fast spatial query
    tree_sup = KDTree(sup_coords)
    
    # 4. Count neighbors within radius
    # k=None returns all points within distance
    # output is a list of lists of neighbor indices
    neighbors_list = tree_sup.query_ball_point(eff_coords, r=radius)
    total_neighbors = sum([len(n) for n in neighbors_list])
    avg_neighbors_per_effector = total_neighbors / len(effectors)
    
    # 5. Density Normalization (The "PhenoSSP" Logic)
    # Global density of suppressors = Total Suppressors / Tissue Area
    # Here we estimate Area roughly by the bounding box of all cells to avoid mask dependency
    # Area (mm^2) approx = (max_x - min_x) * (max_y - min_y) / 1e6
    all_coords = sample_df[['centroid_x', 'centroid_y']].values * pixel_size
    min_x, max_x = all_coords[:,0].min(), all_coords[:,0].max()
    min_y, max_y = all_coords[:,1].min(), all_coords[:,1].max()
    estimated_area_mm2 = ((max_x - min_x) * (max_y - min_y)) / 1e6
    
    if estimated_area_mm2 <= 0: return np.nan
    
    global_treg_density = len(suppressors) / estimated_area_mm2
    
    # Calculate "Spatial Enrichment"
    # Expected neighbors (random distribution) ~ Density * Area_of_Circle
    circle_area_mm2 = (np.pi * (radius/1000)**2)
    expected_neighbors = global_treg_density * circle_area_mm2
    
    if expected_neighbors == 0: return 0.0
    
    interaction_score = avg_neighbors_per_effector / expected_neighbors
    
    # Log transform to normalize distribution (optional, but good for biological data)
    return np.log1p(interaction_score) # log(1+x)

# --- 3. Plotting Functions ---

def plot_survival_curve(df, score_col, time_col, event_col, cutoff, output_path):
    """Generates Kaplan-Meier plot stratified by high/low score."""
    kmf = KaplanMeierFitter()
    
    high_group = df[df[score_col] >= cutoff]
    low_group = df[df[score_col] < cutoff]
    
    if len(high_group) == 0 or len(low_group) == 0:
        print("⚠️ Cannot plot survival: One group is empty.")
        return

    plt.figure(figsize=(8, 7))
    
    # Fit Low Risk (Low Interaction)
    kmf.fit(low_group[time_col], low_group[event_col], label=f'Loose Niche (Low Score) (n={len(low_group)})')
    ax = kmf.plot_survival_function(ci_show=True, color='blue')
    
    # Fit High Risk (High Interaction)
    kmf.fit(high_group[time_col], high_group[event_col], label=f'Suppressive Niche (High Score) (n={len(high_group)})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='red')
    
    # Log-rank Test
    results = logrank_test(high_group[time_col], low_group[time_col], 
                           event_observed_A=high_group[event_col], event_observed_B=low_group[event_col])
    p_value = results.p_value
    
    plt.title(f"Kaplan-Meier Analysis (Cutoff={cutoff:.2f})\nLog-rank P = {p_value:.4f}", fontsize=14)
    plt.xlabel(f"Time ({time_col})", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.ylim(0, 1.05)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return p_value

# --- 4. Main Pipeline ---

def parse_args():
    parser = argparse.ArgumentParser(description="PhenoSSP Spatial Analysis")
    parser.add_argument('--prediction_csv', type=str, required=True, 
                        help="CSV generated by Step 3 (must contain 'sample_id', 'phenotype')")
    parser.add_argument('--metadata_csv', type=str, default=None, 
                        help="Optional CSV containing 'cell_id', 'centroid_x', 'centroid_y' if not in prediction_csv")
    parser.add_argument('--clinical_csv', type=str, required=True,
                        help="CSV with clinical data. Must columns: 'sample_id', 'OS_months', 'OS_status'")
    parser.add_argument('--output_dir', type=str, default='./results/spatial_analysis')
    parser.add_argument('--radius', type=float, default=30.0, help="Interaction radius in microns (default: 30)")
    parser.add_argument('--pixel_size', type=float, default=0.345, help="Microns per pixel")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- [Step 5] Spatial Interaction & Survival Analysis ---")
    
    # 1. Load Predictions
    print(f"--> Loading Predictions: {args.prediction_csv}")
    pred_df = pd.read_csv(args.prediction_csv)
    
    # Merge Coordinates if provided separately
    if 'centroid_x' not in pred_df.columns:
        if args.metadata_csv and os.path.exists(args.metadata_csv):
            print(f"--> Merging coordinates from {args.metadata_csv}...")
            meta_df = pd.read_csv(args.metadata_csv)
            # Ensure keys match (e.g., sample_id + cell_id)
            # This logic depends on your specific CSV structure
            if 'cell_id' in meta_df.columns:
                pred_df = pd.merge(pred_df, meta_df[['cell_id', 'sample_id', 'centroid_x', 'centroid_y']], 
                                   on=['sample_id', 'cell_id'], how='left')
        else:
            print("❌ Error: 'centroid_x/y' not found in predictions and no metadata CSV provided.")
            print("   Spatial analysis requires cell coordinates.")
            return

    # 2. Calculate Spatial Scores per Sample
    print(f"--> Calculating Interaction Scores (Radius={args.radius}μm)...")
    sample_ids = pred_df['sample_id'].unique()
    spatial_results = []
    
    for sid in tqdm(sample_ids):
        score = calculate_interaction_score(pred_df, sid, radius=args.radius, pixel_size=args.pixel_size)
        
        # Also calculate density for correlation check
        # (Simplified count here)
        sample_data = pred_df[pred_df['sample_id'] == sid]
        cd8_count = len(sample_data[sample_data['phenotype'].str.contains('CD8', na=False)])
        
        spatial_results.append({
            'sample_id': sid,
            'Interaction_Score': score,
            'CD8_Count': cd8_count
        })
        
    spatial_df = pd.DataFrame(spatial_results)
    
    # Save Spatial Scores
    score_path = os.path.join(args.output_dir, "spatial_scores.csv")
    spatial_df.to_csv(score_path, index=False)
    print(f"✅ Spatial scores saved to {score_path}")
    
    # 3. Clinical Correlation & Survival
    print(f"--> Loading Clinical Data: {args.clinical_csv}")
    if os.path.exists(args.clinical_csv):
        clinical_df = pd.read_csv(args.clinical_csv)
        
        # Merge
        merged_df = pd.merge(spatial_df, clinical_df, on='sample_id', how='inner')
        print(f"   Matched {len(merged_df)} samples with clinical outcome.")
        
        if len(merged_df) > 5:
            # A. Correlation Analysis (Independence Check)
            corr, p_corr = spearmanr(merged_df['Interaction_Score'], merged_df['CD8_Count'])
            print(f"   Independence Check: Correlation with Cell Count R={corr:.2f} (p={p_corr:.3f})")
            
            # Plot Scatter (Supplementary Fig S1A style)
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=merged_df, x='CD8_Count', y='Interaction_Score')
            plt.title(f"Independence Check\nR={corr:.2f}")
            plt.savefig(os.path.join(args.output_dir, "correlation_check.png"))
            plt.close()
            
            # B. Survival Analysis
            # Determine Cutoff (Median or Optimized)
            cutoff = merged_df['Interaction_Score'].median()
            
            print(f"   Running Survival Analysis (Cutoff={cutoff:.2f})...")
            # Assumes columns 'OS_months' and 'OS_status' (1=Dead, 0=Alive)
            # Adjust column names if yours differ
            time_col = 'OS_months' if 'OS_months' in merged_df.columns else 'Time'
            event_col = 'OS_status' if 'OS_status' in merged_df.columns else 'Status'
            
            if time_col in merged_df.columns and event_col in merged_df.columns:
                p_val = plot_survival_curve(merged_df, 'Interaction_Score', time_col, event_col, cutoff, 
                                            os.path.join(args.output_dir, "KM_Curve_Interaction.png"))
                print(f"✅ Survival Plot Generated. Log-rank P-value: {p_val:.4f}")
            else:
                print(f"⚠️ Clinical columns '{time_col}' or '{event_col}' not found. Skipping KM plot.")
        else:
            print("⚠️ Not enough matched samples for survival analysis.")
    else:
        print("⚠️ Clinical CSV not found. Skipping survival analysis.")

if __name__ == "__main__":
    main()