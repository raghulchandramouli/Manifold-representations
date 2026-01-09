import torch
import torch.nn.functional as F
from train.base_trainer import BaseTrainer

class SupervisedTrainer(BaseTrainer):
    """
    Standard supervised classification trainer.
    """ 
    
    def training_step(self, batch):
        x, y = batch
        
        self.optimizer.zero_grad()
        
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        loss.backward()
        self.optimizer.step()
        
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        return {
            "loss" : loss.item(),
            "acc" : acc.item(),
        }