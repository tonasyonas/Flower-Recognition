# SC4001: Neural Networks and Deep Learning - Group Project

## Track F: Advanced Few-Shot Flower Recognition

**Team Members:** Jonas Tua, Jia Ying Thor, F Syed Abu Thahir

**Dataset:** Oxford Flowers 102 (1020 train, 1020 val, 6149 test)

## Project Structure

```
Flower-Recognition/
├── models/                     # Model definitions
│   ├── resnet_baseline.py      # ResNet18 baseline (full fine-tune)
│   ├── resnet_dilated.py       # ResNet18 + dilated convolutions
│   ├── resnet_deformable.py    # ResNet18 + deformable convolutions
│   ├── resnet_depthwise.py     # ResNet18 + depthwise-separable convolutions
│   ├── resnet_hybrid.py        # ResNet18 + dilated + deformable (novel)
│   ├── vit_model.py            # ViT-B/16 + Visual Prompt Tuning
│   └── __init__.py             # Model registry
├── utils/                      # Utilities
│   ├── data_loader.py          # DataLoaders + few-shot sampling
│   ├── losses.py               # Combined CE + Triplet loss
│   ├── augmentations.py        # MixUp augmentation
│   └── set_seed.py             # Reproducibility
├── notebooks/                  # Jupyter notebooks
│   ├── train_cnn.ipynb         # Train & compare all CNN variants
│   └── evaluate_cnn.ipynb      # Evaluate models on test set
├── results/
│   ├── checkpoints/            # Saved .pth model weights (gitignored)
│   └── logs/                   # Training CSVs and plots
├── train_cnn.py                # CLI training script
├── evaluate_cnn.py             # CLI evaluation script
└── requirements.txt
```

## Models

| Model | Description | Params |
|-------|------------|--------|
| `baseline` | ResNet18, full fine-tune | 11.2M |
| `dilated` | Dilated conv (dilation=2) in layer3/4 | 11.2M |
| `deformable` | Deformable conv in layer3/4 | 11.4M |
| `depthwise` | Depthwise-separable conv in layer2/3/4 | 1.6M |
| `hybrid` | Dilated (layer3) + Deformable (layer4) | 11.4M |

## Quick Start

```bash
# Setup
python -m venv venv
source venv/Scripts/activate  # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install tqdm scikit-learn matplotlib pandas

# Train a model
python train_cnn.py --model baseline --epochs 50

# Train with few-shot
python train_cnn.py --model baseline --k_shot 5 --epochs 30

# Evaluate
python evaluate_cnn.py --model baseline --checkpoint results/checkpoints/best_baseline.pth
```

## Results (Validation Accuracy)

| Model | 1-shot | 2-shot | 5-shot | 10-shot |
|-------|--------|--------|--------|---------|
| Baseline | 35.20% | 59.31% | 79.61% | 88.33% |
| Deformable | 26.96% | 54.22% | 77.45% | 84.61% |
| Hybrid | 24.80% | 46.37% | 68.73% | 82.45% |
| Dilated | 22.94% | 47.25% | 65.59% | 79.71% |
| Depthwise | 6.96% | 12.75% | 39.41% | 60.78% |
