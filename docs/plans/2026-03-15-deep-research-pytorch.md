# Few-Shot Flower Recognition Implementation Plan

> **For Claude/OpenClaw:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Few-Shot Image Classification model on the Oxford Flowers 102 dataset using Vision Transformers (ViT), Visual Prompt Tuning (VPT), MixUp augmentation, and Triplet Loss.

**Architecture:** A pre-trained ViT-B/16 backbone with frozen weights, modified with trainable visual prompt embeddings and a fine-tuned linear classification head. Data augmented with MixUp. Loss calculated using a combination of CrossEntropy and Triplet Loss.

**Tech Stack:** PyTorch, TorchVision, NumPy, scikit-learn

---

### Task 1: Setup Dataset & Dataloaders

**Files:**
- Modify: `utils/data_loader.py`
- Test: `tests/test_data_loader.py` (To be created)

**Step 1: Write the failing test**
Create `tests/test_data_loader.py` to assert dataset lengths (Train=1020, Val=1020, Test=6149) and batch shapes.

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_data_loader.py`
Expected: FAIL (NotImplementedError)

**Step 3: Write minimal implementation**
Implement `get_dataloaders` in `utils/data_loader.py` using `torchvision.datasets.Flowers102`. Apply `Resize(224)`, `ToTensor()`, and `Normalize` for ViT.

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_data_loader.py`
Expected: PASS

**Step 5: Commit**
git commit -m "feat: implement Flowers102 dataset loading and transforms"

---

### Task 2: Implement Visual Prompt Tuning (VPT)

**Files:**
- Modify: `models/vit_model.py`
- Test: `tests/test_vit_vpt.py` (To be created)

**Step 1: Write the failing test**
Create `tests/test_vit_vpt.py` to assert that only the prompt embeddings and classification head require gradients, and output shape is `[batch_size, 102]`.

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_vit_vpt.py`
Expected: FAIL 

**Step 3: Write minimal implementation**
Modify `ViTForFlowerRecognition` in `models/vit_model.py`. Initialize an `nn.Parameter` of shape `[1, num_prompts, embed_dim]`. Inject this parameter into the forward pass by concatenating it with the image patch embeddings before feeding into the transformer blocks. Ensure backbone is frozen.

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_vit_vpt.py`
Expected: PASS

**Step 5: Commit**
git commit -m "feat: implement visual prompt tuning layer in ViT"

---

### Task 3: Implement Combined Loss (Triplet + CrossEntropy)

**Files:**
- Modify: `utils/losses.py`
- Test: `tests/test_losses.py` (To be created)

**Step 1: Write the failing test**
Create `tests/test_losses.py` to assert loss calculation given dummy anchor, positive, negative embeddings and logits.

**Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_losses.py`
Expected: FAIL

**Step 3: Write minimal implementation**
Implement `TripletLoss` calculation logic in `utils/losses.py`. Add a wrapper class `CombinedLoss` that takes `logits` for CrossEntropy and `embeddings` for TripletLoss, weighting them (e.g., `loss = ce_loss + alpha * triplet_loss`).

**Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_losses.py`
Expected: PASS

**Step 5: Commit**
git commit -m "feat: implement combined triplet and cross entropy loss"

---

### Task 4: Complete Training Loop with MixUp

**Files:**
- Modify: `train.py`
- Test: Manual dry-run of 1 epoch

**Step 1: Write minimal implementation**
In `train.py`, integrate `get_dataloaders`, `ViTForFlowerRecognition`, `CombinedLoss`, and `AdamW` optimizer. Implement the epoch loop. Apply `mixup_data` from `utils/augmentations.py` to inputs and calculate MixUp-adjusted loss.

**Step 2: Run a dry-run test**
Run: `python train.py --epochs 1 --dry-run`
Expected: Prints loss decreasing, script completes without crashing.

**Step 3: Commit**
git commit -m "feat: complete end-to-end training loop with mixup"
