# models/resnet_dilated.py
import torch.nn as nn
import torchvision.models as models

class ResNet18Dilated(nn.Module):
    """
    ResNet18 with dilated convolutions (dilation=2) in layer3 and layer4.
    Increases receptive field without adding parameters.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Replace convolutions in layer3 and layer4 with dilation=2
        self._apply_dilation(self.backbone.layer3, dilation=2)
        self._apply_dilation(self.backbone.layer4, dilation=2)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_dilation(self, layer, dilation):
        """Replace 3x3 convolutions in a ResNet layer with dilated versions,
        preserving pretrained weights (same kernel shape)."""
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

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
