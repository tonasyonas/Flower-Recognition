# Enhanced Evaluation Notebook Design

## Objective
Refine the `evaluation.ipynb` notebook to generate advanced, publication-ready visualizations and metrics for the SC4001 final university report, proving the effectiveness of the ViT-VPT architecture paired with Triplet Margin Loss.

## Approach
We will update the existing `evaluation.ipynb` notebook to include four new evaluation sections. These additions directly support the claims made in the `RESEARCH_PLAN.md` regarding few-shot learning performance, geometric separation in latent space, and fine-grained classification.

## Components to Add

### 1. Top-K Accuracy Metrics
Standard Top-1 accuracy is already computed. We will add Top-3 and Top-5 accuracy calculation logic.
- **Why:** In fine-grained recognition with 102 similar classes, a "near miss" (e.g., confusing two types of yellow daisies) is expected. Showing high Top-5 accuracy proves the model has learned robust representations even if the absolute highest logit is slightly off.

### 2. Per-Class Accuracy Breakdown (Bar Charts)
Extract the individual F1-score/Accuracy for each of the 102 classes from `classification_report`.
- **Visualization:** A horizontal bar chart displaying the Top 10 Best performing classes and the Top 10 Worst performing classes.
- **Why:** This provides a deep dive into *where* the model struggles (likely due to extreme similarity or lack of training data in the few-shot setup) and where it excels.

### 3. Example Predictions Grid (Visual Gallery)
Sample 10-15 images from the `test_loader`. Run them through the model.
- **Visualization:** A matplotlib grid (e.g., 3x5) displaying the actual flower image, overlaid with its True label and its Predicted label. Incorrect predictions will have their labels colored in red, and correct ones in green.
- **Why:** Qualitative analysis. Seeing actual failure cases and successes adds necessary context for the discussion section of the report.

### 4. Latent Space Visualization (t-SNE)
The core contribution of Triplet Margin Loss is enforcing geometric separation. We must extract the embeddings tensor returned from `logits, embeddings = model(inputs)`.
- **Visualization:** A 2D scatter plot using `sklearn.manifold.TSNE`. Points will be colored by their True Class Label (102 distinct colors using a large colormap).
- **Why:** This is the ultimate proof that the Triplet Loss worked. A successful t-SNE plot will show tight, distinct clusters of the same color, proving that intra-class variance was minimized and inter-class variance was maximized in the few-shot regime.

## Output Artifacts
The notebook will automatically save these new high-resolution plots:
- `top_bottom_10_classes.png`
- `example_predictions.png`
- `tsne_embeddings.png`

## Data Flow Updates
We need to modify the initial `test_loader` inference loop to correctly track and store:
1. `all_logits` (for Top-K sorting)
2. `all_embeddings` (for the t-SNE projection)
3. `all_images` (for the prediction gallery - but only a small batch needs to be saved to avoid memory overflow)
