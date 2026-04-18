#!/usr/bin/env bash
# scripts/reproduce_all.sh
# Reproduces every experiment reported in the SC4001 Flowers 102 paper.
#
# Usage:
#   bash scripts/reproduce_all.sh
#
# Prerequisites:
#   - Python venv with deps from requirements.txt installed
#   - PyTorch with CUDA support (RTX 3070 Ti or similar)
#   - First run will download the Flowers 102 dataset to ./data (~330 MB)
#
# Total runtime on a single RTX 3070 Ti: ~90 minutes
# Outputs:
#   - results/checkpoints/best_<model>.pth         (model weights, ~12-345 MB each)
#   - results/logs/<model>_training_log.csv        (per-epoch metrics)
#   - report/figures/*.pdf                         (regenerated figures)

set -euo pipefail

cd "$(dirname "$0")/.."

# Activate venv if present (Linux/macOS or Git Bash on Windows)
if [ -f "venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

echo "================================================"
echo "1. Full-data training (50 epochs each, k=10)"
echo "================================================"
for model in baseline dilated deformable depthwise hybrid frozen resnet50 vpt_shallow vpt_deep; do
    echo ""
    echo "--- Training $model ---"
    python train_cnn.py --model "$model" --epochs 50
done

echo ""
echo "================================================"
echo "2. Regularization ablation (baseline only)"
echo "================================================"
python train_cnn.py --model baseline --epochs 50 --label_smoothing 0.1
python train_cnn.py --model baseline --epochs 50 --strong_aug
python train_cnn.py --model baseline --epochs 50 --label_smoothing 0.1 --strong_aug
python train_cnn.py --model baseline --epochs 50 --mixup 0.2

echo ""
echo "================================================"
echo "3. Few-shot experiments (k=1,2,5, 30 epochs each)"
echo "================================================"
for model in baseline dilated deformable depthwise hybrid frozen vpt_shallow vpt_deep; do
    for k in 1 2 5; do
        echo ""
        echo "--- $model k=$k ---"
        python train_cnn.py --model "$model" --k_shot "$k" --epochs 30
    done
done

echo ""
echo "================================================"
echo "4. VPT-Deep prompt length ablation"
echo "================================================"
for p in 1 5 50; do
    echo ""
    echo "--- vpt_deep num_prompts=$p ---"
    python train_cnn.py --model vpt_deep --num_prompts "$p" --epochs 50
done

echo ""
echo "================================================"
echo "5. Regenerate report figures from CSV logs"
echo "================================================"
python report/figures/generate_figures.py

echo ""
echo "================================================"
echo "All experiments complete."
echo "  - Checkpoints: results/checkpoints/"
echo "  - Logs:        results/logs/"
echo "  - Figures:     report/figures/*.pdf"
echo "================================================"
