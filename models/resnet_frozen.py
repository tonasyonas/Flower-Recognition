# models/resnet_frozen.py
import torch.nn as nn
import torchvision.models as models


class ResNet18Frozen(nn.Module):
    """
    ResNet18 with early layers frozen (layer1, layer2).
    Only layer3, layer4, and classification head are trainable.
    Reduces overfitting by limiting trainable parameters while
    preserving low-level pretrained features.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer3, layer4, and fc
        for module in [self.backbone.layer3, self.backbone.layer4]:
            for param in module.parameters():
                param.requires_grad = True

        # Replace and unfreeze classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
