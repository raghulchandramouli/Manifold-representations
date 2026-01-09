"""
Representation Extraction script

Purpose:
- Freeze trained models
- Extract layerwise representation
- Save clean Geomentry dataset for analysis

NO Training happens here.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.seed import seed_everything
from data.dataset import MNISTDataset
from models.mlp import MLP, EncoderOnlyMLP

# HELPERS
def build_model(cfg, objective):
    if objective in ["supervised", "random_labels"]:
        return MLP(**cfg["model"])
    
    else:
        return EncoderOnlyMLP(
            input_dim = cfg["model"]["input_dim"],
            hidden_dim = cfg["model"]["hidden_dim"],
            use_layer_norm = cfg["model"]["use_layer_norm"],
        )


def extract(cfg, objective, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    dataset = MNISTDataset(
        root = cfg["data"]["root"],
        train = False,
        objective = objective,
        subset_size = cfg["data"]["subset_size"],
        seed = cfg["experiment"]["seed"],
    )
    
    loader = DataLoader(
        dataset,
        batch_size = cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
    )
    
    
    # Model 
    model = build_model(cfg, objective).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    # Storage
    layer_buffers = None
    labels = []
    embeddings = []
    
    with torch.no_grad():
        for batch in loader:
            if objective == "contrasive":
                x = batch[0]
                y = None
            else:
                x, y = batch
                
            x = x.to(device)
            
            if objective in ['supervised', 'random_labels']:
                out, acts = model(x, return_activations=True)
                embeddings.append(out.cpu().numpy())
            
            else:
                z, acts = model(x, return_activations=True)
                embeddings.append(z.cpu().numpy())
                
            if layer_buffers is None:
                layer_buffers = [[] for _ in range(len(acts))]
                
            for i, a in enumerate(acts):
                layer_buffers[i].append(a.cpu().numpy())
                
            if y is not None:
                labels.append(y.numpy())

    # SAVE
    step = ckpt["global_step"]
    out_dir = Path(cfg["representation"]["output_dir"]) / "mnist" / objective / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, buf in enumerate(layer_buffers):
        np.save(out_dir / f"layer_{i}.npy", np.concatenate(buf, axis=0))

        np.save(out_dir / "embeddings.npy", np.concatenate(embeddings, axis=0))

        if labels:
            np.save(out_dir / "labels.npy", np.concatenate(labels, axis=0))
                
    
# Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["experiment"]["seed"], cfg["experiment"]["deterministic"])

    extract(cfg, args.objective, args.checkpoint) 