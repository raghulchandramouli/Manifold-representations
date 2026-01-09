"""
Dataset utilities for MNIST Manifold experiments

Key Guarentees
- Single dataset implementation
- Determinstic subset selection
- Objective-specific outputs handled cleanly
"""

from typing import Optional
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

from data.augmentations import ContrasiveAugmentation

class MNISTDataset(Dataset):
    """
    Unified MNIST Dataset wrapper:
    
    Supports:
        - supervised
        - random labels
        - autoencoder
        - contrasive
    """
    
    def __init__(
        self,
        root: str,
        train: bool,
        objective: str,
        subset_size: Optional[int] = None,
        seed : int = 42,
        contrasive_aug : Optional[ContrasiveAugmentation] = None,
        
    ):
        assert objective in {
            "supervised",
            "random_labels",
            "contrasive",
            "random_labels",
        }
        
        self.objective = objective
        self.seed = seed
        
        self.base_transform = transforms.ToTensor()
        
        self.dataset = datasets.MNIST(
            root  = root,
            train = train,
            download = True,
            transform = self.base_transform
        )
        
        # Determinstic subset
        if subset_size is not None:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.dataset), subset_size, replace=False)
            self.dataset = torch.utils.data.Subset(self.dataset, indices)
            
        # Random Labels (controlling)
        if objective == 'random_labels':
            rng = np.random.default_rng(seed)
            self.random_labels = rng.integers(
                low = 0,
                high = 10,
                size = len(self.dataset),
            )
            
        else:
            self.random_labels = None
            
        # Contrastive augmentation
        self.contrasive_aug = contrasive_aug
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        if self.objective == "supervised":
            return x, y
        
        if self.objective == "random_labels":
            return x, (self.random_labels[idx])
        
        if self.objective == "autoencoder":
            return x, x
        
        if self.objective == 'contrasive':
            assert self.contrasive_aug is not None
            x1, x2 = self.contrasive_aug(x)
            return x1, x2
        
        raise RuntimeError("Invalid objective")
