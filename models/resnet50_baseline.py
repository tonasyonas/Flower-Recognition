# models/resnet50_baseline.py
import torch.nn as nn
import torchvision.models as models


class ResNet50Baseline(nn.Module):
    """
    ResNet50 baseline with full fine-tuning.
    Larger backbone (23.5M params) vs ResNet18 (11.2M) for comparison.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        # Replace classifier head (2048 -> num_classes for ResNet50)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
