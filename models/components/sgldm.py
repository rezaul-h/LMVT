# src/models/components/sgldm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SGLDMExtractor(nn.Module):
    def __init__(self, num_features=6):
        super(SGLDMExtractor, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, H, W) — expected to be grayscale
        returns: Tensor of shape (B, num_features)
        """
        B, C, H, W = x.shape
        assert C == 1, "SGLDM input must be a single-channel grayscale image"

        features = []

        for b in range(B):
            img = x[b, 0]  # Shape: (H, W)
            diff = F.pad(img, (1, 0, 1, 0)) - F.pad(img, (0, 1, 0, 1))
            diff = diff[1:, 1:]  # remove padded borders

            # Convert to absolute differences
            diff = diff.abs()

            # Normalize
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)

            # Compute SGLDM features
            mean_diff = diff.mean()
            std_diff = diff.std()
            energy = (diff ** 2).sum()
            entropy = -(diff * (diff + 1e-8).log()).sum()
            contrast = ((diff - mean_diff) ** 2).sum()
            smoothness = 1 - 1 / (1 + std_diff ** 2)

            feature_vector = torch.tensor(
                [mean_diff, std_diff, energy, entropy, contrast, smoothness],
                dtype=torch.float32,
                device=x.device,
            )

            features.append(feature_vector)

        return torch.stack(features)  # Shape: (B, 6)
