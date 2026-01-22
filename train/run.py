"""
Unified experiment runner.

This is the ONLY entry point for training.
All objectives go through this file.
"""

import argparse
import torch
import os
from utils.config import load_config
from utils.seed import seed_everything
from data.dataset import MNISTDataset
from data.augmentations import ContrasiveAugmentation
from models.mlp import MLP, EncoderOnlyMLP
from models.autoencoder import AutoEncoder
from models.projection_head import ProjectionHead

from train.supervised_trainer import SupervisedTrainer
from train.autoencoder_trainer import AutoEncoderTrainer
from train.contrasive_trainer import ContrastiveTrainer
from train.random_label_trainer import RandomLabelTrainer



# Builders
def build_model(cfg, objective):
    if objective in ["supervised", "random_labels"]:
        model_cfg = {k: v for k, v in cfg["model"].items() if k != "type"}
        return MLP(**model_cfg)

    if objective == "autoencoder":
        return AutoEncoder(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            use_layer_norm=cfg["model"]["use_layer_norm"],
        )

    # contrastive
    return EncoderOnlyMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        use_layer_norm=cfg["model"]["use_layer_norm"],
    )


def build_dataloader(cfg, objective):
    contrastive_aug = None

    if objective == "contrastive":
        contrastive_aug = ContrasiveAugmentation(
            gaussian_noise_std=cfg["objective"]["contrastive"]["gaussian_noise_std"],
            random_crop=cfg["objective"]["contrastive"]["random_crop"],
        )

    dataset = MNISTDataset(
        root=cfg["data"]["root"],
        train=True,
        objective=objective,
        subset_size=None,
        seed=cfg["experiment"]["seed"],
        contrastive_aug=contrastive_aug,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        # On Windows, worker subprocesses frequently cause issues; default to 0 there.
        num_workers=(0 if os.name == 'nt' else cfg["data"]["num_workers"]),
        drop_last=True,
    )

def build_optimizer(cfg, model, projection_head=None):
    params = list(model.parameters())
    if projection_head is not None:
        params += list(projection_head.parameters())

    return torch.optim.Adam(
        params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )



# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True,
                        choices=["supervised", "autoencoder",
                                 "contrastive", "random_labels"])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed_everything(
        cfg["experiment"]["seed"],
        cfg["experiment"]["deterministic"],
    )

    objective = args.objective

    
    # Build components
    model = build_model(cfg, objective)
    dataloader = build_dataloader(cfg, objective)

    projection_head = None
    temperature = None

    if objective == "contrastive":
        projection_head = ProjectionHead(
            input_dim=cfg["model"]["hidden_dim"][-1],
            proj_dim=cfg["objective"]["contrastive"]["projection_dim"],
        )
        temperature = cfg["objective"]["contrastive"]["temperature"]

    optimizer = build_optimizer(cfg, model, projection_head)

    # Select trainer
    experiment_name = objective

    if objective == "supervised":
        trainer = SupervisedTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            cfg=cfg,
            experiment_name=experiment_name,
        )

    elif objective == "random_labels":
        trainer = RandomLabelTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            cfg=cfg,
            experiment_name=experiment_name,
        )

    elif objective == "autoencoder":
        trainer = AutoEncoderTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            cfg=cfg,
            experiment_name=experiment_name,
        )

    elif objective == "contrastive":
        trainer = ContrastiveTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            cfg=cfg,
            experiment_name=experiment_name,
            projection_head=projection_head,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
