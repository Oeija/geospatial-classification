"""Unit tests for model architectures."""

import torch

from geospatial_classification.models import CNN_ViT_Hybrid, ConvNet, ViT

BATCH = 2
IMG_SIZE = 64


class TestConvNet:
    def test_forward_shape(self) -> None:
        model = ConvNet(num_classes=2)
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        out = model(x)
        assert out.shape == (BATCH, 2)

    def test_forward_features_shape(self) -> None:
        model = ConvNet(num_classes=2)
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        feats = model.forward_features(x)
        assert feats.dim() == 4
        assert feats.size(0) == BATCH
        assert feats.size(1) == 1024


class TestViT:
    def test_vit_forward(self) -> None:
        vit = ViT(
            in_ch=1024,
            num_classes=2,
            embed_dim=768,
            depth=2,
            heads=4,
            max_tokens=50,
        )
        # Simulate CNN feature maps: (B, 1024, H, W)
        x = torch.randn(BATCH, 1024, 2, 2)
        out = vit(x)
        assert out.shape == (BATCH, 2)

    def test_patch_embed(self) -> None:
        from geospatial_classification.models.vit import PatchEmbed

        pe = PatchEmbed(1024, 768)
        x = torch.randn(BATCH, 1024, 2, 2)
        out = pe(x)
        assert out.shape == (BATCH, 4, 768)


class TestHybrid:
    def test_hybrid_forward(self) -> None:
        model = CNN_ViT_Hybrid(num_classes=2, embed_dim=256, depth=2, heads=4)
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        out = model(x)
        assert out.shape == (BATCH, 2)
