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

        # Freeze early layers for faster training (optional)
        # Uncomment to freeze layers 1-6
        # for name, param in self.backbone.named_parameters():
        #     if 'layer1' in name or 'layer2' in name or 'conv1' in name or 'bn1' in name:
        #         param.requires_grad = False

        # Projection head
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            L2-normalized embeddings of shape (B, embedding_dim)
        """
        x = self.backbone(x)           # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)      # (B, 2048)
        x = self.dropout(x)            # Apply dropout
        x = self.embedding(x)          # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1) # L2 normalize
        return x

    def get_embedding_dim(self):
        """Returns the dimension of the embedding vector"""
        return self.embedding.out_features

# Example usage:
# model = ResNet50_Embedder(embedding_dim=128, pretrained=True, dropout=0.3)
# out = model(torch.randn(8, 3, 128, 128))
# print(f"Output shape: {out.shape}, L2 norm: {torch.norm(out, p=2, dim=1)}")
