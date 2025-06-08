
import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules

        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

    def forward(self, x):
        priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
        logits = torch.zeros(*priors.size()).to(x.device)
        for i in range(3):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
            if i != 2:
                logits = logits + (priors * outputs).sum(dim=-1, keepdim=True)
        return outputs.squeeze()

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=9, stride=1)
        self.primary_capsules = nn.Conv2d(256, 8 * 32, kernel_size=9, stride=2)
        self.capsules = CapsuleLayer(num_classes, 32 * 6 * 6, 8, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x).view(x.size(0), 32 * 6 * 6, 8)
        x = self.capsules(x)
        return x.norm(dim=-1)
