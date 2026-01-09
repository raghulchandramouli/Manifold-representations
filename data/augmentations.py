"""
Data Augmentations used in controlled exp

Principles:
- Minimal Transformation
- No Semanatic destruction
- Reusable for perturbation analysis
"""

import torch
import torchvision.transforms as T
import random

class GaussianNoise:
    "Additive Gaussian Noise"
    
    def __init__(self, std: float):
        self.std = std
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std
    
class ContrasiveAugmentation:
    "Minimal SimCLR-style augmentation"
    
    def __init__(self,
                 gaussian_noise_std: float = 0.1,
                 random_crop: bool = True
                ):
        
        transforms = []
        
        if random_crop:
            transforms.append(
                T.RandomResizedCrop(
                    size=28,
                    scale=(0.9, 1.0),
                )
            )
            
        transforms.append(GaussianNoise(gaussian_noise_std))
        
        self.transform = T.Compose(transforms)
        
    def __call__(self, x: torch.Tensor):
        x1 = self.transform(x)
        x2 = self.transform(x)
        
        return x1, x2
    
class InputPerturbation:
    """
    Small perturbations for manifold smoothness tests.
    """

    def __init__(self, std: float):
        self.noise = GaussianNoise(std)

    def __call__(self, x: torch.Tensor):
        return self.noise(x)