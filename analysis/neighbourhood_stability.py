"""
Neighborhood stability via kNN overlap
"""

from pathlib import Path
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors


REP_ROOT = Path("representation/mnist")
OUT_DIR = Path("analysis/results/neighborhood_stability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 10


def knn_indices(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    return nbrs.kneighbors(return_distance=False)[:, 1:]


results = {}

for objective_dir in REP_ROOT.iterdir():
    objective = objective_dir.name
    results[objective] = {}

    for step_dir in objective_dir.iterdir():
        step = step_dir.name

        input_space = np.load(step_dir / "layer_0.npy")
        repr_space = np.load(step_dir / "embeddings.npy")

        knn_input = knn_indices(input_space, K)
        knn_repr = knn_indices(repr_space, K)

        overlaps = []
        for i in range(len(knn_input)):
            overlap = len(set(knn_input[i]) & set(knn_repr[i])) / K
            overlaps.append(overlap)

        results[objective][step] = float(np.mean(overlaps))

with open(OUT_DIR / "neighborhood_overlap.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved neighborhood stability results.")
