"""
Symmentric AutoEncoder using the Same MLP encoder backbone

Purpose:
 - Preserve data Geometry
 - No semantics pressure
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim   : int = 784,
        hidden_dim  : List[int] = [512, 256, 128],
        use_layer_norm : bool = False
    ):
        super().__init__()

        dims = [input_dim] + hidden_dim

        # Encoder Design
        self.encoder = nn.ModuleList()
        self.enc_norm = nn.ModuleList() if use_layer_norm else None

        for i in range(len(dims) - 1):
            self.encoder.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm:
                self.enc_norm.append(nn.LayerNorm(dims[i + 1]))

        # Decoder Design
        self.decoder = nn.ModuleList(
            [nn.Linear(dims[i + 1], dims[i]) for i in reversed(range(len(dims) - 1))]
        )

        self.use_layer_norm = use_layer_norm


        
        
    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        activations = []

        # Encoder Design
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.use_layer_norm:
                x = self.enc_norm[i](x)
            x = F.relu(x)
            activations.append(x)

        z = x

        # Decoder Design
        for layer in self.decoder[:-1]:
            z = F.relu(layer(z))

        recon = torch.sigmoid(self.decoder[-1](z))

        if return_activations:
            return recon, activations

        return recon
    




        