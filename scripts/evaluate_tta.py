"""
scripts/evaluate_tta.py
Test-time augmentation (TTA) evaluation: average predictions over multiple
augmented views (original, horizontal flip, small rotations) at inference.

Usage:
    python scripts/evaluate_tta.py --model vpt_deep --checkpoint results/checkpoints/best_vpt_deep.pth
"""
import argparse
import os
import sys
import torch
import numpy as np
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MODEL_REGISTRY
from utils.data_loader import get_dataloaders


def tta_predict(model, inputs):
    """Produce softmax predictions averaged over TTA views.
    Views: original, horizontal flip, +5deg rotation, -5deg rotation, +10deg rotation."""
    views = [
        inputs,
        TF.hflip(inputs),
        TF.rotate(inputs, 5),
        TF.rotate(inputs, -5),
        TF.rotate(inputs, 10),
    ]
    probs_sum = None
    for v in views:
        logits, _ = model(v)
        probs = torch.softmax(logits, dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(views)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_prompts', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_vpt = args.model.startswith('vpt_')
    model_cls = MODEL_REGISTRY[args.model]
    if is_vpt:
        model = model_cls(num_classes=102, num_prompts=args.num_prompts).to(device)
    else:
        model = model_cls(num_classes=102).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    base_preds, tta_preds, labels, base_logits_all = [], [], [], []
    top5_correct_base, top5_correct_tta = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="TTA Evaluating"):
            inputs = inputs.to(device, non_blocking=True)

            # Baseline forward (no TTA).
            base_logits, _ = model(inputs)
            base_probs = torch.softmax(base_logits, dim=1)

            # TTA-averaged probabilities.
            tta_probs = tta_predict(model, inputs)

            base_pred = base_probs.argmax(1).cpu().numpy()
            tta_pred = tta_probs.argmax(1).cpu().numpy()
            targets_np = targets.numpy()

            base_preds.extend(base_pred)
            tta_preds.extend(tta_pred)
            labels.extend(targets_np)
            base_logits_all.append(base_logits.cpu().numpy())

            # Top-5 bookkeeping.
            base_top5 = np.argsort(base_probs.cpu().numpy(), axis=1)[:, -5:]
            tta_top5 = np.argsort(tta_probs.cpu().numpy(), axis=1)[:, -5:]
            for i, t in enumerate(targets_np):
                if t in base_top5[i]:
                    top5_correct_base += 1
                if t in tta_top5[i]:
                    top5_correct_tta += 1

    base_preds = np.array(base_preds)
    tta_preds = np.array(tta_preds)
    labels = np.array(labels)
    n = len(labels)

    print(f"\n=== {args.model} ===")
    print(f"{'Metric':<24} {'No TTA':>10} {'+TTA':>10} {'Delta':>8}")
    print('-' * 56)
    b_acc, t_acc = accuracy_score(labels, base_preds), accuracy_score(labels, tta_preds)
    b_mpc, t_mpc = balanced_accuracy_score(labels, base_preds), balanced_accuracy_score(labels, tta_preds)
    b_top5, t_top5 = top5_correct_base / n, top5_correct_tta / n
    print(f"{'Accuracy':<24} {b_acc*100:>9.2f}% {t_acc*100:>9.2f}% {(t_acc-b_acc)*100:>+7.2f}")
    print(f"{'Mean Per-Class Acc':<24} {b_mpc*100:>9.2f}% {t_mpc*100:>9.2f}% {(t_mpc-b_mpc)*100:>+7.2f}")
    print(f"{'Top-5 Accuracy':<24} {b_top5*100:>9.2f}% {t_top5*100:>9.2f}% {(t_top5-b_top5)*100:>+7.2f}")


if __name__ == '__main__':
    main()
