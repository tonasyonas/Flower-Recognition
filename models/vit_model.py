import torch
import torch.nn as nn
import torchvision.models as models

class ViTForFlowerRecognition(nn.Module):
    def __init__(self, num_classes=102, pretrained=True):
        super(ViTForFlowerRecognition, self).__init__()
        # Base ViT model
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        
        # Freeze base model parameters for prompt tuning (will refine with VPT later)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the classification head for 102 flower classes
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
