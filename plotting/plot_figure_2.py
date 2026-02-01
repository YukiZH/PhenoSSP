#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Script for Figure 2: Model Performance & Benchmarking
==============================================================
This script reproduces the key performance figures from the manuscript:
1. Figure 2A: Coarse Classification Confusion Matrix.
2. Figure 2B: Per-Class Performance (F1-Scores).
3. Figure 2C: Benchmark Comparison against Baseline Models.

Usage:
    python plotting/plot_figure_2.py \
        --pred_csv ./results/evaluation/test_predictions.csv \
        --output_dir ./results/figures
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Color Palette (Paper Style)
COLORS = {
    'PhenoSSP': '#2ca02c',  # Green
    'Flat CNN': '#d62728',  # Red
    'Flat ViT': '#7f7f7f',  # Grey
    'SVM': '#ff7f0e',       # Orange
    'Random Forest': '#8c564b' # Brown
}

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Figure 2A: Coarse Classification Confusion Matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Greens', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Recall (Normalized)'}, vmin=0, vmax=1)
    
    plt.title('Figure 2A: Coarse Lineage Separation', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Figure 2A to {save_path}")

def plot_per_class_metrics(report_dict, save_path):
    """
    Figure 2B: Per-Class F1 Scores (Detailed)
    """
    # Filter classes to plot
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    f1_scores = [report_dict[k]['f1-score'] for k in classes]
    
    # Sort for better visualization
    sorted_indices = np.argsort(f1_scores)[::-1]
    classes = [classes[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(classes, f1_scores, color='#1f77b4', alpha=0.8, width=0.6)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.title('Figure 2B: Per-Class F1-Score', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Figure 2B to {save_path}")

def plot_benchmark_comparison(save_path):
    """
    Figure 2C: Benchmark Comparison (Circular Bar Plot or Standard Bar Plot)
    Uses hardcoded data from the manuscript if no external file is provided.
    """
    # Data from Manuscript Section 3.2
    data = {
        'Model': ['PhenoSSP', 'Flat CNN', 'Flat ViT', 'Random Forest', 'SVM'],
        'Balanced Accuracy': [0.710, 0.625, 0.595, 0.482, 0.451],
        'F1-Macro': [0.707, 0.610, 0.582, 0.465, 0.430]
    }
    df = pd.DataFrame(data)
    
    # Prepare Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Plot Bars
    rects1 = ax.bar(x - width/2, df['Balanced Accuracy'], width, label='Balanced Acc', color=COLORS['PhenoSSP'], alpha=0.9)
    rects2 = ax.bar(x + width/2, df['F1-Macro'], width, label='F1-Macro', color='#17becf', alpha=0.9)
    
    # Labels
    ax.set_ylabel('Score')
    ax.set_title('Figure 2C: Overall Performance Benchmark', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'])
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right')
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved Figure 2C to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 2 (Performance)")
    parser.add_argument('--pred_csv', type=str, default=None, 
                        help="Path to CSV with columns 'true_label' and 'pred_label'")
    parser.add_argument('--output_dir', type=str, default='./results/figures', help="Where to save figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- Generating Figure 2 (Performance Benchmarks) ---")
    
    # 1. Figure 2C: Benchmarks (Always generated using manuscript data)
    plot_benchmark_comparison(os.path.join(args.output_dir, 'Figure_2C_Benchmark.png'))
    
    # 2. Figure 2A & 2B: Require Prediction Data
    if args.pred_csv and os.path.exists(args.pred_csv):
        print(f"--> Loading predictions from {args.pred_csv}")
        df = pd.read_csv(args.pred_csv)
        
        # Check columns
        if 'phenotype' in df.columns and 'true_label' not in df.columns:
             # Assume 'phenotype' is pred, and we might be missing ground truth
             # Or check if 'cell_type' (GT) exists
             if 'cell_type' in df.columns:
                 y_true = df['cell_type']
                 y_pred = df['phenotype']
             else:
                 print("⚠️ CSV missing 'cell_type' (ground truth). Skipping Fig 2A/2B.")
                 return
        elif 'true_label' in df.columns and 'pred_label' in df.columns:
            y_true = df['true_label']
            y_pred = df['pred_label']
        else:
            print("⚠️ CSV column format unknown. Expected 'cell_type'/'phenotype' or 'true_label'/'pred_label'.")
            return

        # Generate Metrics
        print("--> Calculating classification metrics...")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Plot 2B
        plot_per_class_metrics(report, os.path.join(args.output_dir, 'Figure_2B_PerClass_F1.png'))
        
        # Prepare Coarse Labels for 2A
        # Map fine-grained to coarse for confusion matrix
        def to_coarse(label):
            if 'Epithelial' in label: return 'Epithelial'
            if any(x in label for x in ['CD8', 'CD4', 'FoxP3', 'CD3', 'Immune']): return 'Immune'
            return 'Other'
            
        y_true_coarse = [to_coarse(l) for l in y_true]
        y_pred_coarse = [to_coarse(l) for l in y_pred]
        labels_coarse = ['Epithelial', 'Immune', 'Other']
        
        # Plot 2A
        plot_confusion_matrix(y_true_coarse, y_pred_coarse, labels_coarse, 
                              os.path.join(args.output_dir, 'Figure_2A_ConfusionMatrix.png'))
                              
    else:
        print("ℹ️ No prediction CSV provided (or file not found). Skipping Figure 2A & 2B.")
        print("   (Run 'python plotting/plot_figure_2.py --pred_csv ...' to generate them)")

if __name__ == "__main__":
    main()