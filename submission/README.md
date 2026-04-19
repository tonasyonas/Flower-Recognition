# Flower Recognition — SC4001 Neural Networks and Deep Learning

**Team:** Jonas Tua · Jia Ying Thor · F Syed Abu Thahir
**Course:** SC4001 Neural Networks and Deep Learning, NTU, AY2025/2026

A systematic comparison of CNN architectures, regularisation techniques, and Visual Prompt Tuning (VPT) for few-shot classification on Oxford Flowers 102 (1 020 train / 1 020 val / 6 149 test images, 102 classes).

---

## Headline Results

| Method | Test Acc | Mean Per-Class Acc | Top-5 | Trainable Params |
|---|---|---|---|---|
| **VPT-Deep** | **91.64%** | **92.83%** | **97.93%** | **170,598** |
| Frozen ResNet18 | 88.52% | 90.82% | 96.85% | 10.5 M |
| Label Smoothing (α=0.1) | 87.27% | 89.40% | 96.16% | 11.2 M |
| ResNet50 | 85.93% | 88.07% | 96.70% | 23.7 M |
| ResNet18 Baseline | 84.42% | 86.11% | 94.84% | 11.2 M |

VPT-Deep beats every CNN configuration while training only **0.2% of the backbone parameters**. Its margin widens as the per-class training budget shrinks — at k=1 it leads the best CNN by 11 points.

---

## Project Structure

```
Flower-Recognition/
├── models/
│   ├── resnet_baseline.py      # ResNet18 — full fine-tune
│   ├── resnet_dilated.py       # ResNet18 + dilated convolutions (layer3+4)
│   ├── resnet_deformable.py    # ResNet18 + deformable convolutions (layer3+4)
│   ├── resnet_depthwise.py     # ResNet18 + depthwise-separable convs (layer2-4)
│   ├── resnet_hybrid.py        # Novel: dilated (layer3) + deformable (layer4)
│   ├── resnet_frozen.py        # ResNet18 with layer1+2 frozen
│   ├── resnet50_baseline.py    # ResNet50 — full fine-tune
│   ├── vit_model.py            # ViT-B/16 + VPT-Shallow
│   ├── vit_model_deep.py       # ViT-B/16 + VPT-Deep
│   └── __init__.py             # MODEL_REGISTRY
├── utils/
│   ├── data_loader.py          # DataLoaders + few-shot subsampling
│   ├── augmentations.py        # MixUp collate function
│   ├── losses.py               # Cross-entropy + batch-hard triplet loss
│   └── set_seed.py             # Reproducibility helper
├── notebooks/
│   ├── train_cnn.ipynb         # Training notebook (all CNN variants)
│   ├── evaluate_cnn.ipynb      # Evaluation notebook with inline plots
│   └── vpt_evaluation.ipynb    # VPT-specific evaluation notebook
├── scripts/
│   ├── reproduce_all.sh        # Reproduce every experiment in the report
│   ├── evaluate_all.py         # Evaluate all checkpoints, print results table
│   └── evaluate_tta.py         # Test-time augmentation evaluation
├── results/
│   └── logs/                   # Per-epoch training CSVs + plots
├── submission/                 # Final submission package
├── train_cnn.py                # Unified training CLI
├── evaluate_cnn.py             # Single-model evaluation CLI
└── requirements.txt
```

---

## Setup

Requires Python 3.10+ and an NVIDIA GPU. Tested on an RTX 3070 Ti with CUDA 12.8.

```bash
# 1. Clone the repo
git clone https://github.com/tonasyonas/Flower-Recognition.git
cd Flower-Recognition

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# source venv/Scripts/activate  # Windows (Git Bash)

# 3. Install PyTorch with CUDA support
#    Replace cu128 with your CUDA version (e.g. cu121, cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install remaining dependencies
pip install -r requirements.txt
```

The Oxford Flowers 102 dataset (~330 MB) is downloaded automatically to `./data/` on the first run.

---

## Available Models

| Key | Description | Trainable Params |
|---|---|---|
| `baseline` | ResNet18, full fine-tune | 11,228,838 |
| `frozen` | ResNet18 with `layer1+2` frozen | 10,545,766 |
| `resnet50` | ResNet50, full fine-tune | 23,717,030 |
| `dilated` | Dilated conv (d=2) in `layer3+4` | 11,228,838 |
| `deformable` | Deformable conv in `layer3+4` | 11,415,534 |
| `depthwise` | Depthwise-separable conv in `layer2-4` | 1,623,270 |
| `hybrid` | Dilated (`layer3`) + Deformable (`layer4`) | 11,353,290 |
| `vpt_shallow` | ViT-B/16 + VPT-Shallow (10 prompts) | 86,118 |
| `vpt_deep` | ViT-B/16 + VPT-Deep (10 prompts/layer) | 170,598 |

---

## Usage

### Train a model

```bash
# Standard 50-epoch run
python train_cnn.py --model baseline --epochs 50

# Few-shot (k images per class)
python train_cnn.py --model vpt_deep --k_shot 5 --epochs 30

# Regularisation variants
python train_cnn.py --model baseline --label_smoothing 0.1
python train_cnn.py --model baseline --strong_aug
python train_cnn.py --model baseline --mixup 0.2

# VPT prompt length ablation
python train_cnn.py --model vpt_deep --num_prompts 50
```

Outputs saved to:
- `results/checkpoints/best_<model>[_kshot][_mixup][_ls][_aug][_p].pth`
- `results/logs/<model>[...}_training_log.csv`

### Evaluate a single model

```bash
python evaluate_cnn.py --model vpt_deep --checkpoint results/checkpoints/best_vpt_deep.pth
```

### Evaluate with test-time augmentation (TTA)

```bash
python scripts/evaluate_tta.py --model vpt_deep --checkpoint results/checkpoints/best_vpt_deep.pth
```

TTA averages predictions over five views (original, horizontal flip, ±5° and +10° rotations) and typically adds 0.4–0.9 MPC points at no extra training cost.

### Evaluate all trained models

```bash
python scripts/evaluate_all.py
```

Prints a full results table with accuracy, mean per-class accuracy, top-5, and parameter counts.

---

## Reproducing All Experiments

To reproduce every number in the report (~90 minutes on an RTX 3070 Ti):

```bash
bash scripts/reproduce_all.sh
```

This runs:
1. Full-data training (50 epochs) for all 9 model variants
2. Regularisation ablation on the baseline (label smoothing, MixUp, strong augmentation)
3. Few-shot experiments at k ∈ {1, 2, 5} for 8 models (24 runs)
4. VPT-Deep prompt length ablation (p ∈ {1, 5, 50})

Then evaluate everything:

```bash
python scripts/evaluate_all.py
```

---

## Reproducibility

- **Seed:** 69 — applied to `random`, `numpy`, `torch`, and `torch.cuda` via `utils/set_seed.py`.
- **cuDNN:** deterministic mode enabled (slightly slower, exactly reproducible).
- **Hyperparameters:** AdamW lr=1e-3, cosine annealing, batch size 32, mixed precision. 50 epochs for full-data runs, 30 epochs for few-shot.
- **Dataset split:** canonical `torchvision.datasets.Flowers102` split.
