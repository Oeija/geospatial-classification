"""Vision Transformer components and CNN-ViT hybrid model."""

import math
from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Project CNN feature maps into transformer embedding dimension
    using a 1×1 convolution.
    """

    def __init__(self, in_ch: int = 1024, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1024, H, W) -> (B, embed_dim, H, W) -> (B, L, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MHSA(nn.Module):
    """Multi-Head Self-Attention."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, self.heads, -1).transpose(1, 2)  # (B, heads, N, d)
        k = k.reshape(B, N, self.heads, -1).transpose(1, 2)
        v = v.reshape(B, N, self.heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer head with CLS token and learnable positional embeddings."""

    def __init__(
        self,
        in_ch: int = 1024,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_tokens: int = 50,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(in_ch, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, max_tokens, embed_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # initialise cls token
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)  # (B, L, D)
        B, L, D = x.shape
        cls = self.cls.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # prepend CLS token
        x = x + self.pos[:, : L + 1, :]  # add positional embeddings
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # classify on CLS token


class CNN_ViT_Hybrid(nn.Module):
    """End-to-end CNN-ViT hybrid for geospatial classification."""

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 6,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.cnn = ConvNet(num_classes)
        self.vit = ViT(
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(self.cnn.forward_features(x))


# re-export ConvNet so ``models`` module contains both
from geospatial_classification.models.cnn import ConvNet  # noqa: E402,F401
