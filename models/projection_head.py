"""
Projection head for contrastive learning.

Important:
- Small capacity
- Does NOT replace the representation used for analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        proj_dim: int = 64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)
