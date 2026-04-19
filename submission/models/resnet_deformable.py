# models/resnet_deformable.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    """
    Wrapper around torchvision.ops.deform_conv2d.
    Learns per-pixel offsets to adapt the convolution sampling grid.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, pretrained_weight=None):
        super().__init__()
        self.stride = stride
        self.padding = padding

        # Offset prediction: 2 offsets (x, y) per kernel position
        # Initialized to zeros so it starts as a standard convolution
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # Main convolution weight — copy pretrained if available
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if pretrained_weight is not None:
            self.weight.data.copy_(pretrained_weight)
        else:
            nn.init.kaiming_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=(self.stride, self.stride),
                             padding=(self.padding, self.padding))


class ResNet18Deformable(nn.Module):
    """
    ResNet18 with deformable convolutions replacing the first 3x3 conv
    in each BasicBlock of layer3 and layer4.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Replace conv1 in each block of layer3 and layer4
        self._apply_deformable(self.backbone.layer3)
        self._apply_deformable(self.backbone.layer4)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_deformable(self, layer):
        """Replace conv1 in each BasicBlock with DeformableConv2d,
        preserving pretrained weights."""
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
