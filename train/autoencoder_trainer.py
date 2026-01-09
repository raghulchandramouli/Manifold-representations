import torch
import torch.nn.functional as F
from train.base_trainer import BaseTrainer

class AutoEncoderTrainer(BaseTrainer):
    """
    Recontruction-only objective
    
    No labels
    """
    
    def training_step(self, batch):
        x1, target = batch
        
        self.optimizer.zero_grad()
        
        recon = self.model(x1)
        loss = F.mse_loss(recon, target.view(target.size(0), -1))
        
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss" : loss.item(),
        }