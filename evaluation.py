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
            _, predicted = torch.max(logits.data, 1)

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
