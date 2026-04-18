# Advanced Few-Shot Flower Recognition

SC4001 Neural Networks and Deep Learning, NTU, Sem 2 2026.
**Team:** Jonas Tua, Jia Ying Thor, F Syed Abu Thahir.

A systematic comparison of CNN modifications, regularization, and Visual Prompt Tuning on Oxford Flowers 102 (1020 train / 1020 val / 6149 test). The full report is in [`report/main.pdf`](report/main.pdf).

## Headline Result

| Method | Test Acc | Mean Per-Class Acc | Top-5 | Trainable Params |
|---|---|---|---|---|
| **VPT-Deep** | **91.64%** | **92.83%** | **97.93%** | **170,598** |
| Frozen ResNet18 | 88.52% | 90.82% | 96.85% | 10.5M |
| ResNet18 baseline | 84.42% | 86.11% | 94.84% | 11.2M |

VPT-Deep beats every CNN configuration we tried while training only 0.2% of the backbone parameters.

## Project Structure

```
Flower-Recognition/
├── models/                        # Model definitions
│   ├── resnet_baseline.py         # ResNet18 baseline (full fine-tune)
│   ├── resnet_dilated.py          # ResNet18 + dilated convolutions
│   ├── resnet_deformable.py       # ResNet18 + deformable convolutions
│   ├── resnet_depthwise.py        # ResNet18 + depthwise-separable convs
│   ├── resnet_hybrid.py           # Novel: dilated (l3) + deformable (l4)
│   ├── resnet_frozen.py           # ResNet18 with layer1+2 frozen
│   ├── resnet50_baseline.py       # ResNet50 baseline
│   ├── vit_model.py               # ViT-B/16 + VPT-Shallow
│   ├── vit_model_deep.py          # ViT-B/16 + VPT-Deep
│   └── __init__.py                # MODEL_REGISTRY
├── utils/
│   ├── data_loader.py             # DataLoaders + few-shot sampling
│   ├── augmentations.py           # MixUp collate function
│   └── set_seed.py                # Reproducibility helper
├── notebooks/
│   ├── train_cnn.ipynb            # Training notebook (all variants)
│   └── evaluate_cnn.ipynb         # Evaluation notebook
├── scripts/
│   ├── reproduce_all.sh           # Reproduce every experiment in the report
│   └── evaluate_all.py            # Evaluate every checkpoint, print results
├── results/
│   ├── checkpoints/               # Trained .pth files (gitignored, ~12-345 MB each)
│   └── logs/                      # Per-epoch training CSVs
├── report/
│   ├── main.tex                   # LaTeX root
│   ├── sections/                  # One .tex per section
│   ├── figures/                   # Generated PDF figures
│   └── README.md                  # Build instructions
├── train_cnn.py                   # Unified training CLI
├── evaluate_cnn.py                # Single-model evaluation CLI
└── requirements.txt
```

## Setup

You need Python 3.10+ and (ideally) an NVIDIA GPU. Tested on RTX 3070 Ti with CUDA 12.8.

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/tonasyonas/Flower-Recognition.git
cd Flower-Recognition
python -m venv venv

# 2. Activate (choose your OS)
source venv/Scripts/activate    # Windows (Git Bash)
# OR
source venv/bin/activate        # Linux / macOS

# 3. Install PyTorch with CUDA support
#    (Replace cu128 with your CUDA version, e.g. cu121 / cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install remaining dependencies
pip install -r requirements.txt
```

The first training run downloads the Oxford Flowers 102 dataset to `./data/` (~330 MB).

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

## Quick Examples

### Train a single model

```bash
# Standard 50-epoch training
python train_cnn.py --model baseline --epochs 50

# Few-shot (k=5 images per class)
python train_cnn.py --model vpt_deep --k_shot 5 --epochs 30

# Regularization variants
python train_cnn.py --model baseline --label_smoothing 0.1
python train_cnn.py --model baseline --strong_aug
python train_cnn.py --model baseline --mixup 0.2

# VPT prompt length ablation
python train_cnn.py --model vpt_deep --num_prompts 50
```

Outputs:
- `results/checkpoints/best_<model>{_kshot}{_mixup}{_ls}{_aug}{_p}.pth`
- `results/logs/<model>{...}_training_log.csv`

### Evaluate a single model

```bash
python evaluate_cnn.py --model vpt_deep --checkpoint results/checkpoints/best_vpt_deep.pth
```

### Evaluate every trained model at once

```bash
python scripts/evaluate_all.py
```

Prints a results table with accuracy, mean per-class accuracy, top-5, and parameter counts.

## Reproducing the Full Report

To reproduce every number in the report (~90 minutes on an RTX 3070 Ti):

```bash
bash scripts/reproduce_all.sh
```

This runs:
1. Full-data training (50 epochs) for all 9 model variants
2. Regularization ablation on baseline (4 configurations)
3. Few-shot experiments (k=1, 2, 5) for 8 models = 24 runs
4. VPT-Deep prompt length ablation (3 additional configurations)
5. Regenerates all PDF figures in `report/figures/`

You can re-run `python scripts/evaluate_all.py` afterwards to get the test-set table.

## Building the Report

See [`report/README.md`](report/README.md). Quick version:

```bash
cd report
latexmk -pdf main.tex
```

Or upload the `report/` folder to Overleaf and compile there.

## Notebooks

`notebooks/train_cnn.ipynb` and `notebooks/evaluate_cnn.ipynb` mirror the CLI scripts but with inline plots and tables. Useful for exploration but the CLI scripts are the source of truth for the report numbers.

## Reproducibility

- Random seed: 69 (set in `utils/set_seed.py`, applied to `random`, `numpy`, `torch`, and `torch.cuda`).
- cuDNN deterministic mode is enabled (slightly slower but exactly reproducible).
- All experiments use identical hyperparameters (AdamW lr=1e-3, cosine annealing, batch 32, 50 epochs full / 30 epochs few-shot, mixed precision).
- The Flowers 102 split is the canonical one from `torchvision.datasets.Flowers102`.
