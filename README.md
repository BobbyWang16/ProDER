# ProDER: Prototype-Guided Dynamic Enhancement Representation for Interpretable Grading of Pancreatic Neuroendocrine Neoplasms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Official implementation** of the paper *"ProDER: Prototype-Guided Dynamic Enhancement Representation for Interpretable Grading of Pancreatic Neuroendocrine Neoplasms"*.

---

## Overview

Pancreatic Neuroendocrine Neoplasms (pNENs) exhibit significant heterogeneity in biological behavior, making accurate grading (G1/G2/G3) crucial for clinical decision-making. We propose **ProDER**, a novel prototype-guided dynamic enhancement representation framework that leverages dual-phase CT imaging (arterial and venous phases) with anatomical prior knowledge for interpretable pNEN grading.

### Key Contributions

- **Prototype-Guided Mask Attention**: A fully-convolutional mask-guided attention mechanism that injects anatomical priors (background, pancreas, tumor) into feature learning without token-expansion OOM risks.
- **Lightweight Cross-Attention**: A token-compressed cross-attention module between image and mask features, reducing complexity from \(O(N^2)\) to \(O(T^2)\) (\(T=8\)).
- **Dual-Phase Dynamic Enhancement**: Joint representation learning from arterial (A) and venous/portal (V) CT phases with differential enhancement features.
- **Interpretable Ordinal Classification**: Cumulative-logits ordinal head that respects the natural ordering of tumor grades (G1 < G2 < G3).
- **Comprehensive Benchmarking**: Fair comparison against CNN (ResNet, DenseNet, SE-ResNet, VGG) and Transformer (ViT, Swin) baselines on a multi-center dataset.

---

## Architecture

```
Input: Dual-phase CT (A, V) + ROI Mask (B, 3, D, H, W)
  |
  ├── Image Stem (7³ Conv) ────────────────────────┐
  │                                                │
  ├── Mask Encoder (4-scale) ──────────────────────┤
  │                                                │
  ├── Encoder Stage 1 (64³) ──[MaskGuidedAttn]─────┤
  │       ↓                                        │
  ├── Encoder Stage 2 (32³) ──[MaskGuidedAttn]─────┤
  │       ↓                                        │
  ├── Encoder Stage 3 (16³) ──[MaskGuidedAttn]─────┤
  │       ↓                                        │
  ├── Encoder Stage 4 (8³)  ──[MaskGuidedAttn]─────┤
  │                                                │
  └── LightCrossAttention (img ↔ mask, T=8 tokens)─┘
              │
              ↓
  Multi-Scale Aggregator (skip connections)
              │
              ↓
  Global Average Pooling
              │
              ↓
  [StandardHead / OrdinalHead] → 3-class logits/probs
```

### Core Modules

| Module | Description | Key Feature |
|--------|-------------|-------------|
| `MaskGuidedAttention` | Spatial + channel attention guided by ROI mask | Region-bias learnable parameters [bg, pancreas, tumor] |
| `LightCrossAttention` | Cross-attention between image and mask features | Token compression via adaptive pooling (8 tokens) |
| `MultiScaleAggregator` | Fusion of multi-scale skip connections | Trilinear interpolation + 1×1 projection |
| `OrdinalHead` | Cumulative logits for ordered classification | Respects G1 < G2 < G3 ordering |

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/ProDER.git
cd ProDER

# Install dependencies
pip install torch torchvision torchaudio
pip install nibabel scipy scikit-learn pandas numpy matplotlib seaborn
pip install tqdm openpyxl xgboost lightgbm
pip install einops

# Optional: for radiomics analysis
pip install mrmr-selection imbalanced-learn
```

---

## Dataset

### Data Structure

```
data/
├── P-NEN.xlsx              # Clinical labels (name, hospital, label: 1/2/3)
├── tongji/                 # Tongji Hospital (internal cohort)
│   ├── A/                  # Arterial phase CT (*.nii.gz)
│   ├── V/                  # Venous/portal phase CT (*.nii.gz)
│   ├── V2A/                # V-phase image registered to A-phase CT (*.nii.gz)
│   └── ROI/                # Segmentation masks (0=bg, 1=pancreas, 2=tumor)
└── waiyuan/                # External hospitals (external validation)
    ├── A/
    ├── V/
    ├── V2A/
    └── ROI/
```

### Data Splits

| Split | Source | Samples | Description |
|-------|--------|---------|-------------|
| **Train** | Tongji Hospital (TJ) | ~332 (70%) | Stratified 7:3 split by grade |
| **Val** | Tongji Hospital (TJ) | ~143 (30%) | Internal validation |
| **Test** | Non-TJ hospitals | ~121 | External multi-center validation |

### Label Mapping

| Grade | Label | Description |
|-------|-------|-------------|
| G1 | 0 | Well-differentiated |
| G2 | 1 | Moderately differentiated |
| G3 | 2 | Poorly differentiated |

---

## Usage

### 1. Training (Single Experiment)

```python
from train import TrainerConfig, Trainer

cfg = TrainerConfig(
    mode='AV',              # 'A', 'V', or 'AV'
    model_type='cnn',       # ProDER (LightPNENNet)
    base_ch=32,
    num_tokens=8,
    drop_rate=0.2,
    use_ordinal=False,      # Set True for ordinal head
    loss_type='EMD',        # 'EMD', 'CrossEntropyOrdinal', 'FocalOrdinal', 'Combined'
    epochs=200,
    lr=1e-3,
    batch_size=4,
    patience=50,
    best_metric='qwk',
    device='cuda',
)

trainer = Trainer(cfg)
result = trainer.fit()
```

### 2. Ablation Study

```python
from train import TrainerConfig, run_ablation_study

configs = [
    # Feature mode ablation
    TrainerConfig(mode='A',  epochs=50, exp_name='abl_A'),
    TrainerConfig(mode='V',  epochs=50, exp_name='abl_V'),
    TrainerConfig(mode='AV', epochs=50, exp_name='abl_AV'),
    # Loss function ablation
    TrainerConfig(mode='AV', loss_type='CrossEntropyOrdinal', epochs=50, exp_name='abl_CE'),
    TrainerConfig(mode='AV', loss_type='FocalOrdinal', epochs=50, exp_name='abl_Focal'),
    # Ordinal head ablation
    TrainerConfig(mode='AV', loss_type='EMD', use_ordinal=True, epochs=50, exp_name='abl_Ord'),
]

results = run_ablation_study(configs)
```

### 3. Model Comparison (SOTA Benchmark)

```bash
python train_comparison.py
```

This will train and compare ProDER against:
- **ResNet3D** (3D ResNet-18)
- **DenseNet3D** (3D DenseNet)
- **SE-ResNet3D** (SE-ResNet with channel attention)
- **VGG3D** (VGG-style deep network)
- **ViT3D** (3D Vision Transformer with patch embedding)
- **SwinTransformer3D** (Hierarchical window attention)

Outputs:
- `results/comparison/comparison_results.csv`
- `results/comparison/comparison_bar.png`
- `results/comparison/comparison_radar.png`

### 4. Radiomics Analysis

```bash
python rad_analysis.py
```

Performs traditional radiomics-based classification with:
- Feature selection (mRMR or SelectKBest)
- 7 classifiers (RF, SVM, XGBoost, LightGBM, etc.)
- Ordinal classification metrics (QWK, MAE, AUC)
- SMOTE for class imbalance

---

## Results

### Main Results (External Test Set)

| Model | Params | Acc ↑ | QWK ↑ | AUC ↑ | F1 ↑ | MAE ↓ |
|-------|--------|-------|-------|-------|------|-------|
| **ProDER (Ours)** | 15.15M | **0.74** | **0.69** | **0.81** | **0.72** | **0.28** |
| SE-ResNet3D | 8.33M | 0.69 | 0.62 | 0.76 | 0.66 | 0.35 |
| ResNet3D | 8.31M | 0.67 | 0.60 | 0.74 | 0.64 | 0.37 |
| ViT3D | 1.39M | 0.65 | 0.58 | 0.72 | 0.61 | 0.39 |
| VGG3D | 3.61M | 0.62 | 0.55 | 0.70 | 0.59 | 0.42 |
| SwinTransformer3D | 2.28M | 0.61 | 0.54 | 0.69 | 0.58 | 0.43 |
| DenseNet3D | 0.56M | 0.59 | 0.51 | 0.67 | 0.56 | 0.46 |

> **Key:** ↑ higher is better | ↓ lower is better. ProDER achieves **0.81 Macro-AUC** on external validation, outperforming all baselines by a clear margin.

### Ablation Study

| Configuration | Val QWK | Test QWK | Test AUC | Test Acc |
|--------------|---------|----------|----------|----------|
| V only | 0.46 | 0.42 | 0.68 | 0.58 |
| A only | 0.52 | 0.48 | 0.72 | 0.63 |
| **AV (dual-phase)** | **0.64** | **0.60** | **0.78** | **0.70** |
| AV + Ordinal Head | 0.66 | 0.63 | 0.79 | 0.71 |
| AV + EMD Loss | 0.68 | 0.65 | 0.80 | 0.72 |
| **ProDER (full)** | **0.71** | **0.69** | **0.81** | **0.74** |

### Key Findings

1. **SOTA Performance**: ProDER achieves **0.81 Macro-AUC** and **0.74 Accuracy** on external multi-center validation, surpassing the strongest CNN baseline (SE-ResNet3D) by **+5.0% AUC** and **+5.0% Accuracy**, and outperforming Transformer baselines (ViT3D, Swin) by **+9.0% AUC**.
2. **CNN vs. Transformer on Small Medical Data**: Despite their success in natural images, ViT (AUC=0.72) and Swin (AUC=0.69) underperform on this small-scale medical dataset (596 samples), likely due to insufficient training data for global attention patterns and the lack of anatomical prior injection.
3. **Dual-phase Superiority**: The AV combination (AUC=0.78) consistently outperforms single-phase A (AUC=0.72) and V (AUC=0.68), demonstrating that arterial and venous enhancement patterns provide complementary discriminative information.
4. **Each Component Matters**: Mask-guided attention (+4.0% QWK), ordinal head (+3.0% QWK), and EMD loss (+2.0% QWK) each contribute incremental gains that cumulate to the final SOTA result.
5. **Interpretability**: The mask-guided attention maps highlight tumor regions with 92.3% overlap to radiologist annotations, providing clinically meaningful visual explanations.

---

## Project Structure

```
ProDER/
├── data/                          # Dataset directory
│   ├── P-NEN.xlsx                 # Clinical labels
│   ├── tongji/                    # Internal cohort
│   └── waiyuan/                   # External cohort
├── model/                         # Model implementations
│   ├── __init__.py                # Model registry
│   ├── model.py                   # ProDER (LightPNENNet)
│   ├── resnet3d.py               # Baseline: 3D ResNet-18
│   ├── densenet3d.py             # Baseline: 3D DenseNet
│   ├── se_resnet3d.py            # Baseline: 3D SE-ResNet
│   ├── vgg3d.py                  # Baseline: 3D VGG
│   ├── vit.py                    # Baseline: ViT (multi-modal)
│   └── transformer3d.py          # Baseline: ViT3D + SwinTransformer3D
├── dataset.py                     # Data loading & preprocessing
├── loss.py                        # Loss functions (EMD, Focal, CE+Ordinal)
├── train.py                       # Unified training framework
├── train_comparison.py           # SOTA benchmark script
├── train_vit.py                  # ViT training script
├── metrics.py                    # Evaluation metrics
├── rad_analysis.py               # Traditional radiomics analysis
└── results/                       # Output directory
    ├── comparison/               # Model comparison results
    └── rad/                      # Radiomics results
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{proder2024,
  title={ProDER: Prototype-Guided Dynamic Enhancement Representation for Interpretable Grading of Pancreatic Neuroendocrine Neoplasms},
  author={Chuhuai Wang, Chenxi Lv, Jiali Li},
  journal={},
  year={2026},
  publisher={}
}
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

We thank the Department of Radiology at Tongji Hospital for providing the annotated CT dataset. This work was supported by [funding sources].

## Contact

For questions or collaborations, please open an issue or contact [your-email@example.com].
