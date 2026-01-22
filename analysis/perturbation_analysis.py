"""
Perturbation smoothness analysis
"""

from pathlib import Path
import numpy as np
import json


REP_ROOT = Path("representation/mnist")
OUT_DIR = Path("analysis/results/perturbation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 0.05

results = {}

for objective_dir in REP_ROOT.iterdir():
    objective = objective_dir.name
    results[objective] = {}

    for step_dir in objective_dir.iterdir():
        step = step_dir.name

        X = np.load(step_dir / "embeddings.npy")
        noise = np.random.randn(*X.shape) * EPS

        drift = np.linalg.norm((X + noise) - X, axis=1).mean()
        results[objective][step] = float(drift)

with open(OUT_DIR / "perturbation_drift.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved perturbation results.")
