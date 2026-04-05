# evaluation_baseline.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from models.resnet_baseline import ResNet18Baseline
from utils.data_loader import get_dataloaders

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get dataloaders
    _, _, test_loader = get_dataloaders(batch_size=32)

    # Initialize model
    model = ResNet18Baseline(num_classes=102)

    # Load the best checkpoint
    checkpoint_path = 'best_resnet_baseline.pth'
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded checkpoint from '{checkpoint_path}'")
    except FileNotFoundError:
        print(f"Error: '{checkpoint_path}' not found. Make sure you have trained the model first.")
        exit(1)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating Baseline"):
            inputs, targets = inputs.to(device), targets.to(device)

            logits, _ = model(inputs)
            _, predicted = torch.max(logits.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_accuracy = np.mean(all_preds == all_targets)
    print(f"Baseline Test Accuracy: {test_accuracy * 100:.2f}%")

    report = classification_report(all_targets, all_preds, zero_division=0)
    
    with open("classification_report_baseline.txt", "w") as f:
        f.write(report)
    print("Report saved to classification_report_baseline.txt")

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap="Reds", cbar=True)
    plt.title("Baseline Confusion Matrix (102 Classes)", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.tight_layout()
    plt.savefig("confusion_matrix_baseline.png", dpi=300)
