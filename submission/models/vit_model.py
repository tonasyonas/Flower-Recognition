import torch
import torch.nn as nn
import torchvision.models as models

class ViTForFlowerRecognition(nn.Module):
    def __init__(self, num_classes=102, num_prompts=10, embed_dim=768, pretrained=True):
        super(ViTForFlowerRecognition, self).__init__()
        
        # Load pre-trained ViT
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the classification head
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
        # Initialize Visual Prompts (VPT - Shallow)
        # Shape: [1, num_prompts, embed_dim]
        self.num_prompts = num_prompts
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, embed_dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)
        
    def forward(self, x):
        # We need to manually inject the prompts before the transformer encoder.
        # This requires overriding the standard forward pass of torchvision's ViT.
        
        # 1. Patch embedding
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. Add class token and positional embeddings
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Add positional embedding (only to class token + image patches)
        x = x + self.model.encoder.pos_embedding
        
        # 3. Inject Visual Prompts (VPT)
        # Expand prompts to batch size
        batch_prompts = self.prompts.expand(n, -1, -1)
        
        # Concatenate: [Class Token, Prompts, Image Patches]
        # x is currently [n, 1 + num_patches, embed_dim]
        # We split it to insert prompts
        cls_token = x[:, :1, :]
        img_patches = x[:, 1:, :]
        
        x = torch.cat([cls_token, batch_prompts, img_patches], dim=1)
        
        # 4. Pass through Transformer Encoder
        # The encoder processes the sequence
        x = self.model.encoder.dropout(x)
        x = self.model.encoder.layers(x)
        
        # 5. Extract output (only the class token matters for classification)
        # The class token is still at index 0
        x = self.model.encoder.ln(x)
        x = x[:, 0]
        
        # 6. Classification Head
        logits = self.model.heads(x)
        
        return logits, x # Returning logits for CE, x (embeddings) for Triplet Loss
