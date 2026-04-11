# evaluate_cnn.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from models import MODEL_REGISTRY
from utils.data_loader import get_dataloaders


def evaluate(model, dataloader, device):
    """Run inference and return all predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN model on Flowers 102 test set")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(num_classes=102).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    print(f"Loaded {args.model} from {args.checkpoint}")

    # Load test data
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Evaluate
    preds, labels = evaluate(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Classification report
    report = classification_report(labels, preds)
    report_path = f'results/logs/{args.model}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Model: {args.model}\nTest Accuracy: {acc:.4f}\n\n{report}")
    print(f"Report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{args.model} - Confusion Matrix (Acc: {acc:.4f})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = f'results/logs/{args.model}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=200)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == '__main__':
    main()
