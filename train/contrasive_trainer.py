import torch
import torch.nn.functional as F
from train.base_trainer import BaseTrainer


class ContrastiveTrainer(BaseTrainer):
    """
    SimCLR-style contrastive trainer using NT-Xent loss.
    """

    def __init__(self, *args, projection_head=None, temperature: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # projection_head is optional; move it to device if provided
        self.projection_head = projection_head
        if self.projection_head is not None:
            self.projection_head.to(self.device)
        self.temperature = temperature

    def training_step(self, batch):
        x1, x2 = batch

        self.optimizer.zero_grad()

        z1 = self.model(x1)
        z2 = self.model(x2)

        # Optionally apply projection head before computing contrastive loss
        if self.projection_head is not None:
            z1 = self.projection_head(z1)
            z2 = self.projection_head(z2)

        loss = self.nt_xent_loss(z1, z2)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
        }

    def nt_xent_loss(self, z1, z2):
        """
        Normalized Temperature-scaled Cross Entropy Loss
        """

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        representations = torch.cat([z1, z2], dim=0)

        similarity = torch.matmul(representations, representations.T)
        similarity /= self.temperature

        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity = similarity.masked_fill(mask, -9e15)

        loss = F.cross_entropy(similarity, labels)

        return loss
