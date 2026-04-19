# models/resnet_hybrid.py
import torch
import torch.nn as nn
import torchvision.models as models
from models.resnet_deformable import DeformableConv2d

class ResNet18Hybrid(nn.Module):
    """
    Novel hybrid: dilated convolutions in layer3 + deformable convolutions in layer4.
    Rationale: layer3 benefits from wider receptive field (dilation),
    layer4 benefits from input-adaptive spatial sampling (deformable).
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Layer3: dilated convolutions for wider receptive field
        self._apply_dilation(self.backbone.layer3, dilation=2)

        # Layer4: deformable convolutions for adaptive sampling
        self._apply_deformable(self.backbone.layer4)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_dilation(self, layer, dilation):
        """Replace 3x3 convolutions with dilated versions, preserving pretrained weights."""
        for block in layer:
            if isinstance(block.conv1, nn.Conv2d) and block.conv1.kernel_size == (3, 3):
                old_conv = block.conv1
                new_conv = nn.Conv2d(
                    old_conv.in_channels, old_conv.out_channels,
                    kernel_size=3, stride=old_conv.stride[0],
                    padding=dilation, dilation=dilation, bias=False
                )
                new_conv.weight.data.copy_(old_conv.weight.data)
                block.conv1 = new_conv
            if isinstance(block.conv2, nn.Conv2d) and block.conv2.kernel_size == (3, 3):
                old_conv = block.conv2
                new_conv = nn.Conv2d(
                    old_conv.in_channels, old_conv.out_channels,
                    kernel_size=3, stride=1,
                    padding=dilation, dilation=dilation, bias=False
                )
                new_conv.weight.data.copy_(old_conv.weight.data)
                block.conv2 = new_conv

    def _apply_deformable(self, layer):
        """Replace conv1 in each BasicBlock with DeformableConv2d, preserving pretrained weights."""
        for block in layer:
            old_conv = block.conv1
            block.conv1 = DeformableConv2d(
                old_conv.in_channels, old_conv.out_channels,
                kernel_size=3, stride=old_conv.stride[0], padding=1,
                pretrained_weight=old_conv.weight.data
            )

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
