# src/models/lmvt.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.cbam import CBAM
from src.models.components.sgldm import SGLDMExtractor

class MobileConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=128, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x

class LMVT(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, use_sgldm=True):
        super(LMVT, self).__init__()
        self.use_sgldm = use_sgldm

        # CNN Feature Extractor
        self.conv1 = MobileConvBlock(in_channels, 32)
        self.conv2 = MobileConvBlock(32, 64)
        self.conv3 = MobileConvBlock(64, 128)

        # CBAM
        self.cbam = CBAM(128)

        # Transformer Input Projector
        self.flatten = nn.Flatten(2)
        self.transpose = lambda x: x.transpose(1, 2)
        self.transformer = TransformerEncoder(dim=128)

        # Pool and FC
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # SGLDM
        if self.use_sgldm:
            self.sgldm = SGLDMExtractor(num_features=6)
            self.fc = nn.Linear(128 + 6, num_classes)
        else:
            self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN feature extractor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # shape: (B, 128, H, W)

        # Attention
        x = self.cbam(x)

        # Transformer encoding
        x = self.flatten(x)  # (B, 128, H*W)
        x = self.transpose(x)  # (B, H*W, 128)
        x = self.transformer(x)

        x = x.transpose(1, 2)  # (B, 128, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (B, 128)

        if self.use_sgldm:
            sgldm_feat = self.sgldm(x.view(batch_size, 1, int(x.shape[1] ** 0.5), -1))
            x = torch.cat([x, sgldm_feat], dim=1)  # (B, 128+6)

        return self.fc(x)
