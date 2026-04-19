# models/resnet_depthwise.py
import torch.nn as nn
import torchvision.models as models

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable convolution: depthwise (per-channel) + pointwise (1x1).
    Reduces parameters by ~8-9x compared to standard 3x3 conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResNet18Depthwise(nn.Module):
    """
    ResNet18 with depthwise-separable convolutions in layer2/layer3/layer4.
    Significantly reduces trainable parameters while maintaining structure.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Replace 3x3 convs in layer2, layer3, layer4 with depthwise-separable
        self._apply_depthwise(self.backbone.layer2)
        self._apply_depthwise(self.backbone.layer3)
        self._apply_depthwise(self.backbone.layer4)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_depthwise(self, layer):
        """Replace 3x3 convolutions with depthwise-separable versions."""
        for block in layer:
            if isinstance(block.conv1, nn.Conv2d) and block.conv1.kernel_size == (3, 3):
                block.conv1 = DepthwiseSeparableConv(
                    block.conv1.in_channels, block.conv1.out_channels,
                    kernel_size=3, stride=block.conv1.stride[0], padding=1
                )
            if isinstance(block.conv2, nn.Conv2d) and block.conv2.kernel_size == (3, 3):
                block.conv2 = DepthwiseSeparableConv(
                    block.conv2.in_channels, block.conv2.out_channels,
                    kernel_size=3, stride=1, padding=1
                )

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
