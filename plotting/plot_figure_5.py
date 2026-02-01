#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Script for Figure 5: Spatial Analysis & Clinical Survival
==================================================================
This script reproduces the key clinical findings:
1. Figure 5B/E: Kaplan-Meier Survival Curves (Density vs. Spatial Score).
2. Figure 5C: Interaction Score comparison (Survivor vs. Non-Survivor).
3. Figure 5D: Spatial Map visualization of the "Suppressive Niche".

Usage:
    python plotting/plot_figure_5.py \
        --spatial_csv ./results/spatial_analysis/spatial_scores.csv \
        --clinical_csv ./demo_data/clinical_data.csv \
        --output_dir ./results/figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu

# --- Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

COLORS = {
    'Survivor': '#1f77b4',      # Blue
    'Non-Survivor': '#d62728',  # Red
    'High Risk': '#d62728',     # Red
    'Low Risk': '#2ca02c',      # Green
    'CD8': 'red',
    'Treg': 'blue',
    'Other': 'lightgrey'
}

def plot_km_curve(df, stratify_col, time_col, event_col, title, save_path):
    """
    Figure 5B / 5E: Kaplan-Meier Survival Curve
    """
    if len(df) < 5:
        print(f"⚠️ Not enough data to plot KM curve for {title}")
        return

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 7))
    
    # Determine cutoff (Median)
    cutoff = df[stratify_col].median()
    
    group_high = df[df[stratify_col] >= cutoff]
    group_low = df[df[stratify_col] < cutoff]
    
    # Plot High Group
    kmf.fit(group_high[time_col], group_high[event_col], label=f'High {stratify_col} (n={len(group_high)})')
    ax = kmf.plot_survival_function(ci_show=True, color=COLORS['High Risk'], linewidth=2.5)
    
    # Plot Low Group
    kmf.fit(group_low[time_col], group_low[event_col], label=f'Low {stratify_col} (n={len(group_low)})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS['Low Risk'], linewidth=2.5)
    
    # Log-rank Test
    results = logrank_test(group_high[time_col], group_low[time_col], 
                           event_observed_A=group_high[event_col], event_observed_B=group_low[event_col])
    p_value = results.p_value
    
    # Formatting
    plt.title(f'{title}\n(Log-rank P = {p_value:.4f})', fontsize=14, fontweight='bold')
    plt.xlabel(f'Time ({time_col})', fontsize=12)
    plt.ylabel('Overall Survival Probability', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add P-value text
    plt.text(0.1, 0.1, f'p = {p_value:.4f}', transform=ax.transAxes, 
             fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved KM Curve to {save_path}")

def plot_interaction_boxplot(df, score_col, group_col, save_path):
    """
    Figure 5C: Boxplot comparison (Survivor vs Non-Survivor)
    """
    plt.figure(figsize=(6, 6))
    
    # Run Stats
    groups = df[group_col].unique()
    if len(groups) == 2:
        g1 = df[df[group_col] == groups[0]][score_col]
        g2 = df[df[group_col] == groups[1]][score_col]
        stat, p_val = mannwhitneyu(g1, g2)
        title_suffix = f"\n(P = {p_val:.4f})"
    else:
        title_suffix = ""

    sns.boxplot(data=df, x=group_col, y=score_col, palette=[COLORS['Survivor'], COLORS['Non-Survivor']], width=0.5)
    sns.stripplot(data=df, x=group_col, y=score_col, color='black', alpha=0.6, jitter=True)
    
    plt.title(f'Figure 5C: Interaction Score by Outcome{title_suffix}', fontsize=12, fontweight='bold')
    plt.ylabel('Spatial Interaction Score')
    plt.xlabel('Patient Outcome')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Figure 5C to {save_path}")

def plot_spatial_niche_demo(save_path):
    """
    Figure 5D: Synthetic Spatial Map (if no real coordinates provided)
    Visualizes the "Suppressive Niche": CD8s surrounded by Tregs.
    """
    print("ℹ️ Generating synthetic spatial map for Figure 5D...")
    np.random.seed(42)
    
    # Create a "Suppressive Niche" scenario
    # Center: CD8 T cell
    # Surround: Tregs
    
    n_cd8 = 20
    n_treg = 50
    n_other = 200
    
    # CD8s clustered in center
    cd8_x = np.random.normal(500, 50, n_cd8)
    cd8_y = np.random.normal(500, 50, n_cd8)
    
    # Tregs surrounding them (Ring)
    theta = np.random.uniform(0, 2*np.pi, n_treg)
    r = np.random.normal(100, 10, n_treg)
    treg_x = 500 + r * np.cos(theta)
    treg_y = 500 + r * np.sin(theta)
    
    # Background cells
    other_x = np.random.uniform(0, 1000, n_other)
    other_y = np.random.uniform(0, 1000, n_other)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(other_x, other_y, c='lightgrey', s=20, alpha=0.5, label='Other')
    plt.scatter(treg_x, treg_y, c='blue', s=60, label='Treg', edgecolors='white')
    plt.scatter(cd8_x, cd8_y, c='red', s=60, marker='*', label='CD8+ T', edgecolors='white')
    
    # Draw "Interaction Radius" circles around CD8s
    for x, y in zip(cd8_x, cd8_y):
        circle = plt.Circle((x, y), 30*2, color='red', fill=False, linestyle='--', alpha=0.3) # 30um scale approx
        plt.gca().add_patch(circle)
        
    plt.title('Figure 5D: The Suppressive Niche (30μm Contact)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.axis('off') # Turn off axis for map look
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Figure 5D to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 5 (Spatial & Survival)")
    parser.add_argument('--spatial_csv', type=str, default=None, 
                        help="CSV with 'sample_id', 'Interaction_Score', 'CD8_Count'")
    parser.add_argument('--clinical_csv', type=str, default=None, 
                        help="CSV with 'sample_id', 'OS_months', 'OS_status', 'Outcome_Group'")
    parser.add_argument('--output_dir', type=str, default='./results/figures', help="Where to save figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- Generating Figure 5 (Spatial & Clinical) ---")
    
    # 1. Figure 5D: Spatial Map (Always generate demo if no specific coordinates input)
    # In a real pipeline, you would read cell coordinates. Here we simulate for visualization.
    plot_spatial_niche_demo(os.path.join(args.output_dir, 'Figure_5D_Suppressive_Niche.png'))
    
    # 2. Survival & Boxplots (Need Data)
    if args.spatial_csv and args.clinical_csv:
        if os.path.exists(args.spatial_csv) and os.path.exists(args.clinical_csv):
            print("--> Loading spatial and clinical data...")
            spatial_df = pd.read_csv(args.spatial_csv)
            clinical_df = pd.read_csv(args.clinical_csv)
            
            # Merge
            df = pd.merge(spatial_df, clinical_df, on='sample_id', how='inner')
            print(f"   Matched {len(df)} samples.")
            
            if len(df) > 5:
                # Figure 5C: Boxplot
                # Ensure we have a grouping column (e.g. 'OS_status' -> Survivor/Non-Survivor)
                if 'OS_status' in df.columns:
                    df['Outcome'] = df['OS_status'].apply(lambda x: 'Non-Survivor' if x==1 else 'Survivor')
                    plot_interaction_boxplot(df, 'Interaction_Score', 'Outcome', 
                                             os.path.join(args.output_dir, 'Figure_5C_Interaction_Boxplot.png'))
                
                # Figure 5B: CD8 Density Survival (The Paradox)
                if 'CD8_Count' in df.columns:
                    plot_km_curve(df, 'CD8_Count', 'OS_months', 'OS_status', 
                                  'Figure 5B: CD8+ T Density (Traditional)', 
                                  os.path.join(args.output_dir, 'Figure_5B_CD8_Survival.png'))
                
                # Figure 5E: Interaction Score Survival (The Resolution)
                if 'Interaction_Score' in df.columns:
                    plot_km_curve(df, 'Interaction_Score', 'OS_months', 'OS_status', 
                                  'Figure 5E: Spatial Interaction Score (PhenoSSP)', 
                                  os.path.join(args.output_dir, 'Figure_5E_Interaction_Survival.png'))
            else:
                print("⚠️ Not enough matched data for statistical plots.")
        else:
            print("⚠️ Spatial or Clinical CSV file not found.")
    else:
        print("ℹ️ No input CSVs provided. Skipping KM curves and Boxplots.")
        print("   (Run with --spatial_csv and --clinical_csv to generate them)")

if __name__ == "__main__":
    main()