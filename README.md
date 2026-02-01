这份 README 写得已经很不错了，结构清晰，逻辑顺畅。但是为了匹配我们之前修改过的**文稿术语（Terminology）以及提升专业度**，我有以下几点修改建议：

### 🛠️ 主要修改点 (Key Changes)

1. **术语统一**：
* 将 `FoxP3` 改为全大写 `FOXP3`。
* 将 `Contact-Dependent` 改为 `Proximity-Dependent`（与你最新的 Title 和 Cover Letter 保持一致，因为 30μm 属于邻近而非物理接触）。
* 将 `dictated by` 弱化为 `strongly associated with`（避免因果关系太绝对）。


2. **逻辑修正**：
* 在 **Step 4 (Interpretability)** 中，你原来的命令是加载 `coarse_model` 来查看 `CD8`。但根据你的 Method，CD8 是在 **Expert Model (Fine-grained)** 里区分的，Coarse Model 只分 Immune/Epithelial。所以我把示例命令改成了加载 `expert_model`。


3. **增加专业感**：
* 添加了 Badges（徽章），让仓库看起来更像一个成熟的开源项目。
* 在 Note to Reviewers 里强调了 "Minimal Demo Data"，避免误会。



---

### ✅ 修改后的 README.md (直接复制即可)

```markdown
# PhenoSSP: Uncovering a Proximity-Dependent Suppressive Niche in RCC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Framework](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/)

**PhenoSSP** is a hierarchical deep learning framework designed to resolve the "CD8+ T cell Paradox" in Renal Cell Carcinoma (RCC). By combining fine-grained single-cell phenotyping with spatial interaction analysis, PhenoSSP reveals that patient prognosis is not determined by the simple abundance of cytotoxic T cells, but is strongly associated with their spatial entrapment within **proximity-dependent suppressive niches** driven by Regulatory T cells (Tregs).

> **Note to Reviewers**: This repository contains the official implementation of the source code used in the manuscript. While the full clinical dataset (1,633 TMA cores) is restricted due to privacy regulations, a **minimal anonymized demo dataset** is provided in the `demo_data/` folder to verify the pipeline's functionality and reproducibility.

## 🚀 Key Features

* **Hierarchical Classification**: A "coarse-to-fine" architecture that mimics pathological logic to effectively handle class imbalance, enabling precise identification of rare subsets like **CD4+FOXP3+ Tregs** and CD3+CD4-CD8+ T cells.
* **Domain-Adaptive Spatial Features**: Utilizes a Vision Transformer (ViT-S/16) backbone initialized with self-supervised learning (MAE) to capture channel-specific morphological patterns from 7-color multiplex immunofluorescence (mIF) images.
* **Pixel-Level Interpretability**: Features a saliency map engine (Guided Backpropagation) that validates model decisions based on biologically relevant subcellular structures (e.g., verifying membrane vs. nuclear localization).
* **Spatial Interaction Scoring**: Introduces a density-normalized **Spatial Interaction Score ($S_{inter}$)** to quantify the **30 μm suppressive niche**, serving as a robust prognostic biomarker independent of TNM stage.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone [https://github.com/YourUsername/PhenoSSP.git](https://github.com/YourUsername/PhenoSSP.git)
cd PhenoSSP

```

### 2. Install dependencies

It is recommended to use a virtual environment (Conda/venv).

```bash
pip install -r requirements.txt

```

## 💻 Pipeline Usage

The pipeline is designed to run sequentially. Below are example commands using the provided demo data.

### 1. Data Preprocessing

Extract 64x64 single-cell patches from raw mIF images (using DeepCell Mesmer segmentation masks).

```bash
python pipeline/01_data_preprocessing.py --input_dir ./demo_data/raw --output_dir ./demo_data/patches

```

### 2. Model Training

PhenoSSP employs a two-stage training strategy:

**Stage 1: Self-Supervised Pre-training (MAE)**

```bash
python pipeline/02_pretrain_mae.py --data_dir ./demo_data/patches --epochs 50

```

**Stage 2: Supervised Fine-tuning (Hierarchical)**

```bash
# Fine-tune the Expert Classifier (e.g., for Immune subtypes)
python pipeline/03_finetune_classifier.py \
    --patch_dir ./demo_data/patches \
    --annotation_csv ./demo_data/annotations.csv \
    --pretrained_weights ./results/mae_checkpoints/backbone.pt \
    --mode expert

```

### 3. Inference

Run the trained hierarchical classifier on a new cohort.

```bash
python pipeline/04_inference_cohort.py --data_dir ./demo_data/patches --model_dir ./results/models

```

### 4. Interpretability (Figure 4)

Generate saliency maps to visualize subcellular attention patterns.
*Note: Use the Expert Model to visualize specific markers like CD8 or FOXP3.*

```bash
python pipeline/05_visual_interpretability.py \
    --patch_dir ./demo_data/patches \
    --model_path ./results/models/expert_immune_model.pth \
    --target_marker CD8 \
    --output_dir ./results/saliency_maps

```

### 5. Spatial Analysis (Figure 5)

Calculate the Spatial Interaction Score (30 μm radius) and perform survival analysis.

```bash
python pipeline/06_spatial_analysis.py \
    --prediction_csv ./results/inference/cohort_predictions.csv \
    --clinical_csv ./demo_data/clinical.csv \
    --radius 30 \
    --output_dir ./results/spatial_analysis

```

## 📊 Reproducing Figures

Scripts to reproduce the main figures from the manuscript are located in the `plotting/` directory.

| Figure | Description | Command |
| --- | --- | --- |
| **Figure 2** | Performance Benchmarks (Confusion Matrices, F1 Scores) | `python plotting/plot_figure_2.py` |
| **Figure 3** | Robustness & Domain Adaptation Analysis | `python plotting/plot_figure_3.py` |
| **Figure 4** | Feature Space (t-SNE) & Attention Maps | `python plotting/plot_figure_4.py` |
| **Figure 5** | **Spatial Interaction & Survival Analysis (The Paradox)** | `python plotting/plot_figure_5.py` |

## 📜 Citation

If you find this code or analysis useful, please cite our manuscript:

> Zhang Y., et al. "PhenoSSP Uncovers a Proximity-Dependent Suppressive Niche Decoding the CD8+ T Cell Paradox in Renal Cell Carcinoma." (Under Review, 2026).

## 📧 Contact

For technical questions or issues, please open a [GitHub issue](https://github.com/YourUsername/PhenoSSP/issues) or contact the authors.
