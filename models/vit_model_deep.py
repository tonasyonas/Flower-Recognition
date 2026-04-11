import torch
import torch.nn as nn
import torchvision.models as models


class ViTDeepForFlowerRecognition(nn.Module):
    def __init__(self, num_classes=102, num_prompts=10, embed_dim=768, pretrained=True):
        super(ViTDeepForFlowerRecognition, self).__init__()

        # Load pre-trained ViT
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)

        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classification head
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

        self.num_prompts = num_prompts
        num_layers = len(self.model.encoder.layers)

        # VPT-Deep: one set of learnable prompts per transformer layer
        # Shape: [num_layers, 1, num_prompts, embed_dim]
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.empty(1, num_prompts, embed_dim))
            for _ in range(num_layers)
        ])
        for p in self.prompts:
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x):
        # 1. Patch embedding
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # 2. Add class token and positional embeddings
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding

        x = self.model.encoder.dropout(x)

        # 3. Pass through each transformer layer, injecting prompts at every layer
        for i, layer in enumerate(self.model.encoder.layers):
            batch_prompts = self.prompts[i].expand(n, -1, -1)

            # Split: [CLS, img_patches] — discard previous layer's prompts (indices 1:1+num_prompts)
            if i == 0:
                cls_token = x[:, :1, :]
                img_patches = x[:, 1:, :]
            else:
                cls_token = x[:, :1, :]
                img_patches = x[:, 1 + self.num_prompts:, :]

            # Inject fresh prompts for this layer: [CLS, prompts_i, img_patches]
            x = torch.cat([cls_token, batch_prompts, img_patches], dim=1)

            x = layer(x)

        # 4. Layer norm and extract CLS token
        x = self.model.encoder.ln(x)
        cls_output = x[:, 0]

        # 5. Classification head
        logits = self.model.heads(cls_output)

        return logits, cls_output
