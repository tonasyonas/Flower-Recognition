# SC4001 Few-Shot Flower Recognition: Setup & Run Guide

This guide covers how to clone this repository, set up a GPU-enabled environment (like Google Colab or a local machine with CUDA), and execute the training pipeline.

## 1. Environment Setup

### Option A: Google Colab (Recommended for Free GPUs)
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Go to **Runtime > Change runtime type** and select **T4 GPU** (or better).
4. Run the following in the first cell to clone the repo and enter the directory:
   ```bash
   !git clone https://github.com/tonasyonas/SC4001-Flower-Recognition.git
   %cd SC4001-Flower-Recognition
   ```
5. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```

### Option B: Local Machine (Requires NVIDIA GPU)
1. Clone the repository:
   ```bash
   git clone https://github.com/tonasyonas/SC4001-Flower-Recognition.git
   cd SC4001-Flower-Recognition
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Running the Training Pipeline

By default, the `train.py` script has the actual execution commented out to prevent accidental CPU lockups on headless servers. 

To run the training:
1. Open `train.py` in your editor (or Colab cell).
2. Scroll to the very bottom.
3. Change the code from this:
   ```python
   if __name__ == '__main__':
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       # train_model(epochs=50, batch_size=32, device=device)
       print("Training loop logic written. Script is set to skip execution...")
   ```
   To this:
   ```python
   if __name__ == '__main__':
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       train_model(epochs=50, batch_size=32, device=device)
   ```
4. Execute the script:
   ```bash
   python train.py
   ```

## 3. What to Expect
- **First Run:** The script will automatically download the Oxford Flowers 102 dataset (around 330MB) into a local `data/` folder.
- **Epochs 1-40:** The model will train using **MixUp augmentation** and the **Combined Loss** (Cross-Entropy + Triplet Loss). You will see the loss and accuracy metrics printed to the console.
- **Epochs 41-50:** MixUp is disabled automatically for clean fine-tuning.
- **Checkpoints:** Whenever validation accuracy hits a new high, the model weights are saved to `best_vit_vpt_model.pth`.

## 4. Next Steps (For the Report)
Once training is complete, you should load the best checkpoint and evaluate it on the `test_loader`. You can use `scikit-learn`'s `confusion_matrix` and `classification_report` to generate visualizations for the final 10-page university submission.
