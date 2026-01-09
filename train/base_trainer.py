"""
BaseTrainer

Responsible:
    - Determinstic Setup
    - Step-based training loop
    - logging
    - Checkpointing
    - Device Management
    
DOES NOT:
    - Define Loss
    - Define Model architecture
    - Touch dataset semantics
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Iterable
import torch
from tqdm import tqdm

class BaseTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: Iterable,
        cfg: Dict[str, Any],
        experiment_name: str,
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.cfg = cfg
        
        self.device = self._resolve_device(cfg['experiment']['device'])
        self.model.to(self.device)
        
        self.max_steps = cfg['experiment']['max_steps']
        self.log_every = cfg['experiment']['log_every']
        self.checkpoint_steps = set(cfg['training']['checkpoint_steps'])
        
        self.global_step = 0
        
        self.ckpt_dir = Path('checkpoint') / experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def training_step(self, batch) -> Dict[str, float]:
        """
        One optmization step
        
        Must:
         - compute loss
         - call backward()
         - call optimizer.step()
         
         Returns:
         - dict of scaler metrics
        """
        
        pass
    
    # CORE TRAINING LOOP
    def train(self):
        self.model.train()
        
        data_iter = iter(self.dataloader)
        pbar = tqdm(total=self.max_steps, desc="Training")
        
        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            
            batch = self._move_to_device(batch)
            
            metrics = self.training_step(batch)
            self.global_step += 1
            pbar.update(1)
            
            if self.global_step % self.log_every == 0:
                self._log(metrics)
                
            if self.global_step in self.checkpoint_steps:
                self._save_checkpoint()
                
        pbar.close()
        self._save_checkpoint(final=True)
        
    # Utilities
    def _move_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) for b in batch]
        return batch.to(self.device)
    
    def _resolve_device(self, device):
        if device_cfg == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _log(self, metrics: Dict[str, float]):
        msg = f"[step {self.global_step}]"
        msg += " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(msg)
        
    def _save_checkpoint(self, final: bool = False):
        name = "final.pt" if final else f"step_{self.global_step}.pt"
        path = self.ckpt_dir / name
        
        torch.save(
            {
                "model_state" : self.model.state_dict(),
                "optimizer_state" : self.optimizer.state_dict(),
                "global_step" : self.global_step,
                "cfg" : self.cfg
            },
            path,
        )