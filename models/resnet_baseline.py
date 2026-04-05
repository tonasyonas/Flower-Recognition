# models/resnet_baseline.py
import torch.nn as nn
import torchvision.models as models

class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=102):
        super(ResNet18Baseline, self).__init__()
        # Load pre-trained ResNet18
        # Using the standard weights (ImageNet)
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        # We only need the logits for the baseline (no embeddings needed for standard CE loss)
        logits = self.backbone(x)
        return logits, None  # Returning None for embeddings to match ViT tuple unpacking in eval if needed
