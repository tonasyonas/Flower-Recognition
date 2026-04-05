# Enhanced Evaluation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine the `evaluation.ipynb` script to include Top-K accuracy, per-class accuracy bar charts, a visual prediction gallery, and a t-SNE latent space plot.

**Architecture:** Python script using sklearn metrics, matplotlib, seaborn, and TSNE manifold. Since the output format is Jupyter, we will create the raw python blocks and use nbconvert or build a `.py` script that acts functionally identical but creates the images. *To keep it simple and runnable headlessly on the GPU, we will update `evaluation.py` instead of the ipynb directly.*

**Tech Stack:** PyTorch, scikit-learn, matplotlib, seaborn

---

### Task 1: Extract Embeddings and Logits during Inference

**Files:**
- Modify: `evaluation.py`

- [ ] **Step 1: Update the imports and inference loop**
Modify `evaluation.py` to capture `all_logits`, `all_embeddings`, and a small sample of images/labels for the gallery. We also need to add TSNE imports.

Find the `if __name__ == '__main__':` block. Update the `with torch.no_grad():` section to the following:

```python
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
```

- [ ] **Step 2: Commit**
```bash
git add evaluation.py
git commit -m "feat: capture embeddings and logits for advanced metrics"
```

### Task 2: Add Top-K Accuracy and Per-Class Bar Charts

**Files:**
- Modify: `evaluation.py`

- [ ] **Step 1: Calculate Top-1, 3, 5 and Per-class metrics**
Append the following directly after the `test_accuracy` calculation in `evaluation.py`.

```python
    # ------------------
    # Top-K Accuracy
    # ------------------
    def top_k_accuracy(logits, targets, k):
        sorted_indices = np.argsort(logits, axis=1)[:, ::-1] # descending
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
    report_dict = classification_report(all_targets, all_preds, zero_division=0, output_dict=True)
    class_accuracies = []
    for cls_idx in range(102):
        if str(cls_idx) in report_dict:
            class_accuracies.append((cls_idx, report_dict[str(cls_idx)]['f1-score']))
            
    # Sort by f1-score
    class_accuracies.sort(key=lambda x: x[1])
    
    worst_10 = class_accuracies[:10]
    best_10 = class_accuracies[-10:]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Worst 10
    axes[0].barh([str(x[0]) for x in worst_10], [x[1] for x in worst_10], color='salmon')
    axes[0].set_title('Top 10 Worst Performing Classes (F1-Score)')
    axes[0].set_xlim(0, 1.0)
    
    # Best 10
    axes[1].barh([str(x[0]) for x in best_10], [x[1] for x in best_10], color='lightgreen')
    axes[1].set_title('Top 10 Best Performing Classes (F1-Score)')
    axes[1].set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig("top_bottom_10_classes.png", dpi=300)
    plt.close()
    print("Saved top_bottom_10_classes.png")
```

- [ ] **Step 2: Commit**
```bash
git add evaluation.py
git commit -m "feat: add top-k accuracy and per-class performance charts"
```

### Task 3: Visual Predictions Gallery & t-SNE Plot

**Files:**
- Modify: `evaluation.py`

- [ ] **Step 1: Add gallery and t-SNE logic**
Append the following directly after the previous block.

```python
    # ------------------
    # Example Predictions Grid
    # ------------------
    import torchvision.transforms.functional as F
    
    # Denormalize images for plotting
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    num_samples = min(15, len(sample_images))
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = sample_images[i] * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        
        pred = sample_preds[i].item()
        true = sample_targets[i].item()
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f"Pred: {pred} | True: {true}", color=color)
        
    plt.tight_layout()
    plt.savefig("example_predictions.png", dpi=300)
    plt.close()
    print("Saved example_predictions.png")

    # ------------------
    # t-SNE Embeddings Plot
    # ------------------
    print("Calculating t-SNE embeddings (this may take a minute)...")
    tsne = manifold.TSNE(n_components=2, random_state=42, perplexity=30)
    # Use a subset if dataset is too large, but 6149 should be fast enough for t-sne
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_targets, cmap='tab20', alpha=0.7, s=15)
    plt.colorbar(scatter, label='True Class Label')
    plt.title('t-SNE Projection of ViT-VPT Latent Embeddings\n(Showing effect of Triplet Loss)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig("tsne_embeddings.png", dpi=300)
    plt.close()
    print("Saved tsne_embeddings.png")
```

- [ ] **Step 2: Commit**
```bash
git add evaluation.py
git commit -m "feat: add t-SNE visualization and example prediction gallery"
```
