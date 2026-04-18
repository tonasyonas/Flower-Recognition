"""
scripts/evaluate_all.py
Evaluates every trained checkpoint on the test set and prints a results table
matching the one in the report.

Usage:
    python scripts/evaluate_all.py
"""
import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MODEL_REGISTRY
from utils.data_loader import get_dataloaders


# (display_name, model_key, checkpoint_filename)
CONFIGS = [
    ('VPT-Deep',                'vpt_deep',     'best_vpt_deep.pth'),
    ('Frozen ResNet18',         'frozen',       'best_frozen.pth'),
    ('Label Smoothing (a=0.1)', 'baseline',     'best_baseline_ls0.1.pth'),
    ('LS + Strong Aug',         'baseline',     'best_baseline_ls0.1_strongaug.pth'),
    ('ResNet50',                'resnet50',     'best_resnet50.pth'),
    ('VPT-Shallow',             'vpt_shallow',  'best_vpt_shallow.pth'),
    ('Baseline',                'baseline',     'best_baseline.pth'),
    ('Strong Augmentation',     'baseline',     'best_baseline_strongaug.pth'),
    ('Deformable',              'deformable',   'best_deformable.pth'),
    ('MixUp (a=0.2)',           'baseline',     'best_baseline_mixup0.2.pth'),
    ('Hybrid',                  'hybrid',       'best_hybrid.pth'),
    ('Dilated',                 'dilated',      'best_dilated.pth'),
    ('Depthwise',               'depthwise',    'best_depthwise.pth'),
]

CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'checkpoints'
)


def evaluate(model, dataloader, device):
    """Run inference and return preds, labels, logits."""
    model.eval()
    preds, labels, all_logits = [], [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            logits, _ = model(inputs.to(device, non_blocking=True))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(targets.numpy())
            all_logits.append(logits.cpu().numpy())
    return np.array(preds), np.array(labels), np.concatenate(all_logits, axis=0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}\n")

    _, _, test_loader = get_dataloaders(batch_size=32)

    header = f"{'Method':<26} {'Acc':>7} {'MPC':>7} {'Top-5':>7} {'Params':>14}"
    print(header)
    print('=' * len(header))

    for display_name, model_key, ckpt_name in CONFIGS:
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"{display_name:<26} (checkpoint not found: {ckpt_name})")
            continue

        model = MODEL_REGISTRY[model_key](num_classes=102).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        preds, labels, logits = evaluate(model, test_loader, device)
        acc = accuracy_score(labels, preds)
        mpc = balanced_accuracy_score(labels, preds)
        top5 = np.argsort(logits, axis=1)[:, -5:]
        top5_acc = float(np.mean([labels[i] in top5[i] for i in range(len(labels))]))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{display_name:<26} {acc*100:>6.2f}% {mpc*100:>6.2f}% {top5_acc*100:>6.2f}% {params:>14,}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
