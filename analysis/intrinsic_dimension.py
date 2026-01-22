"""
Intrinsic dimensionality via PCA Participation Ratio
"""

from pathlib import Path
import numpy as np
import json

REP_ROOT = Path("representation/mnist")
OUT_DIR = Path("analysis/results/intrinsic_dimension")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def participation_ratio(X: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    return (eigvals.sum() ** 2) / (np.square(eigvals).sum() + 1e-8)


results = {}

for objective_dir in REP_ROOT.iterdir():
    objective = objective_dir.name
    results[objective] = {}

    for step_dir in objective_dir.iterdir():
        step = step_dir.name
        results[objective][step] = {}

        for layer_file in step_dir.glob("layer_*.npy"):
            layer = layer_file.stem
            X = np.load(layer_file)
            results[objective][step][layer] = participation_ratio(X)

with open(OUT_DIR / "intrinsic_dimension.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved intrinsic dimensionality results.")
