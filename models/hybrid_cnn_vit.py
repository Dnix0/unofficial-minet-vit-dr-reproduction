import torch
import torch.nn as nn
from torchvision import models


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                  # [B, D, H/P, W/P]
        x = x.flatten(2)                  # [B, D, N]
        x = x.transpose(1, 2)             # [B, N, D]
        return x


class LightweightViTBranch(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)           # [B, N, D]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x)

        cls_feature = x[:, 0]             # [B, D]
        return cls_feature


class CNNBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = models.resnet18(weights=weights)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = backbone.fc.in_features

    def forward(self, x):
        x = self.features(x)              # [B, 512, 1, 1]
        x = torch.flatten(x, 1)           # [B, 512]
        return x


class HybridCNNViT(nn.Module):
    def __init__(
        self,
        num_classes=5,
        cnn_pretrained=True,
        vit_embed_dim=256,
        vit_depth=4,
        vit_heads=4,
        dropout=0.3
    ):
        super().__init__()

        self.cnn_branch = CNNBranch(pretrained=cnn_pretrained)

        self.vit_branch = LightweightViTBranch(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            dropout=0.1
        )

        fusion_dim = self.cnn_branch.out_dim + vit_embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn_branch(x)
        vit_feat = self.vit_branch(x)

        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        logits = self.classifier(fused)

        return logits


def build_hybrid_cnn_vit(num_classes=5, cnn_pretrained=True):
    model = HybridCNNViT(
        num_classes=num_classes,
        cnn_pretrained=cnn_pretrained,
        vit_embed_dim=256,
        vit_depth=4,
        vit_heads=4,
        dropout=0.3
    )

    return model
