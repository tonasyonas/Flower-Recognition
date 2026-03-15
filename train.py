import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.vit_model import ViTForFlowerRecognition
from utils.data_loader import get_dataloaders
from utils.losses import CombinedLoss
from utils.augmentations import mixup_data

def train_model(epochs=50, batch_size=32, device='cpu'):
    print(f"Initializing training on {device}...")
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    # 2. Init Model
    model = ViTForFlowerRecognition(num_classes=102, num_prompts=10).to(device)
    
    # Verify what is trainable (should only be prompts and head)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    # 3. Setup Loss and Optimizer
    criterion = CombinedLoss(alpha=0.2).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # Optional learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("Pipeline ready. Dry-run complete.")
    # Real training loop logic goes here when ready to compute.
    
if __name__ == '__main__':
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Just running a pipeline check without doing the heavy lifting yet
    print("Training script structured. Awaiting execution command.")
