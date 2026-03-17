import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50_Embedder(nn.Module):
    """
    ResNet-50 based embedding model for metric learning.

    Args:
        embedding_dim (int): Dimension of the output embedding vector
        pretrained (bool): Whether to use pretrained ImageNet weights
        dropout (float): Dropout probability before embedding layer (0 to disable)
    """
    def __init__(self, embedding_dim=128, pretrained=True, dropout=0.3):
        super().__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)           # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)      # (B, 2048)
        x = self.dropout(x)
        x = self.embedding(x)          # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_embedding_dim(self):
        return self.embedding.out_features


class DINOv2_Embedder(nn.Module):
    """
    DINOv2 (ViT-S/14) based embedding model for metric learning.
    Uses self-supervised features from Meta's DINOv2.

    Args:
        embedding_dim (int): Dimension of the output embedding vector
        model_name (str): DINOv2 variant — 'dinov2_vits14' (22M params, 384-d)
        dropout (float): Dropout probability before embedding layer
        freeze_backbone (bool): Whether to freeze DINOv2 weights
    """
    def __init__(self, embedding_dim=256, model_name='dinov2_vits14',
                 dropout=0.2, freeze_backbone=False):
        super().__init__()
        # Load DINOv2 from Meta's hub (downloads weights automatically)
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        feature_dim = self.backbone.embed_dim  # 384 for vits14

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, 3, H, W)  — H,W should be multiples of 14

        Returns:
            L2-normalized embeddings of shape (B, embedding_dim)
        """
        x = self.backbone(x)               # (B, 384) CLS token
        x = self.dropout(x)
        x = self.embedding(x)              # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1)     # L2 normalize
        return x

    def get_embedding_dim(self):
        """Returns the dimension of the embedding vector"""
        return self.embedding.out_features


# Example usage:
# model = DINOv2_Embedder(embedding_dim=256, dropout=0.2)
# out = model(torch.randn(8, 3, 224, 224))
# print(f"Output shape: {out.shape}, L2 norm: {torch.norm(out, p=2, dim=1)}")
