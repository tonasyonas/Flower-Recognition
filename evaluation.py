import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from models.vit_model import ViTForFlowerRecognition
from utils.data_loader import get_dataloaders

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get dataloaders
    _, _, test_loader = get_dataloaders(batch_size=32)

    # Initialize model
    model = ViTForFlowerRecognition(num_classes=102, num_prompts=10)

    # Load the best checkpoint
    checkpoint_path = "best_vit_vpt_model.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded checkpoint from '{checkpoint_path}'")
    except FileNotFoundError:
        print(
            f"Error: '{checkpoint_path}' not found. Make sure you have trained the model first."
        )
        exit(1)

    model = model.to(device)
    model.eval()

    import sklearn.manifold as manifold
    from torchvision.utils import make_grid

    all_preds = []
    all_targets = []
    all_embeddings = []
    all_logits = []

    sample_images = None
    sample_targets = None
    sample_preds = None

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            logits, embeddings = model(inputs)

            # Get predictions
            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

            # Save the first batch for visual gallery plotting
            if i == 0:
                sample_images = inputs.cpu()
                sample_targets = targets.cpu()
                sample_preds = predicted.cpu()

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_embeddings = np.array(all_embeddings)
    all_logits = np.array(all_logits)

    test_accuracy = np.mean(all_preds == all_targets)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # ------------------
    # Top-K Accuracy
    # ------------------
    def top_k_accuracy(logits, targets, k):
        sorted_indices = np.argsort(logits, axis=1)[:, ::-1]  # descending
        top_k_indices = sorted_indices[:, :k]
        correct = np.any(top_k_indices == targets[:, None], axis=1)
        return np.mean(correct)

    top1_acc = top_k_accuracy(all_logits, all_targets, k=1)
    top3_acc = top_k_accuracy(all_logits, all_targets, k=3)
    top5_acc = top_k_accuracy(all_logits, all_targets, k=5)

    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")

    # ------------------
    # Per-Class Accuracy Breakdown
    # ------------------
    report_dict = classification_report(
        all_targets, all_preds, zero_division=0, output_dict=True
    )
    class_accuracies = []
    for cls_idx in range(102):
        if str(cls_idx) in report_dict:
            class_accuracies.append((cls_idx, report_dict[str(cls_idx)]["f1-score"]))

    # Sort by f1-score
    class_accuracies.sort(key=lambda x: x[1])

    worst_10 = class_accuracies[:10]
    best_10 = class_accuracies[-10:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Worst 10
    axes[0].barh(
        [str(x[0]) for x in worst_10], [x[1] for x in worst_10], color="salmon"
    )
    axes[0].set_title("Top 10 Worst Performing Classes (F1-Score)")
    axes[0].set_xlim(0, 1.0)

    # Best 10
    axes[1].barh(
        [str(x[0]) for x in best_10], [x[1] for x in best_10], color="lightgreen"
    )
    axes[1].set_title("Top 10 Best Performing Classes (F1-Score)")
    axes[1].set_xlim(0, 1.0)

    plt.tight_layout()
    plt.savefig("top_bottom_10_classes.png", dpi=300)
    plt.close()
    print("Saved top_bottom_10_classes.png")

    report = classification_report(all_targets, all_preds, zero_division=0)

    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("Report saved to classification_report.txt")

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix (102 Classes)", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
