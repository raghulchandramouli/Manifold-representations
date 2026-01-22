"""
Plain MLP Architecture used accross all expirements
No CNN - Spatial Norm is not Needed, We want to isolate the objective's effect
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MLP(nn.Module):
    """
    Simple Feedforward MLP 

    Args:
        input_dims  : Input dimensions (784 for MNIST)
        hidden_dims : List of hidden Layer dimensions [512, 256, 128]
        output_dim  : Output_dim (10)
        use_layer_norm : set to False
    """

    def __init__(self, input_dim:  int = 768, 
                       hidden_dim: List[int] = [512, 256, 128],
                       output_dim: int = 10,
                       use_layer_norm: bool = False):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        dims = [input_dim] + hidden_dim

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.output_layer = nn.Linear(hidden_dim[-1], 
                                      output_dim)


    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input Tensor [B, 1, 28, 28]
            return_activations (bool): If True, return hidden activations

        Return:
            logits (Tensor): Output logits [B, output_dim]
            activations (List[Tensor], Optional): Hidden Layer Activations
        """

        # Flatten if Image
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        activations = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norms[idx](x)
            x = F.relu(x)
            activations.append(x)

        logits = self.output_layer(x)

        if return_activations:
            return logits, activations

        return logits

class EncoderOnlyMLP(nn.Module):
    """
    Encoder-only MLP used for:
        - AutoEncoder
        - Contrasive Learning
        - Representation Extraction
    """

    def __init__(
        self,
        input_dim  : int = 784,
        hidden_dim : List[int] = [512, 256, 128],
        use_layer_norm : bool = False
    ):

        super().__init__()

        self.backbone  = MLP(
            input_dim  = input_dim,
            hidden_dim = hidden_dim,
            output_dim = hidden_dim[-1],
            use_layer_norm = use_layer_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ):

        if return_activations:
            _, activations = self.backbone(x, return_activations=True)
            return activations[-1], activations

        return self.backbone(x)


        