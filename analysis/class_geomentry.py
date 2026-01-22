"""
Class geometry: intra / inter class variance + silhouette
"""

from pathlib import Path
import numpy as np
import json
from sklearn.metrics import silhouette_score

REP_ROOT = Path("representation/mnist")
OUT_DIR = Path("analysis/results/class_geometry")
OUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}

for objective_dir in REP_ROOT.iterdir():
    if not objective_dir.is_dir():
        continue  # Skip files
    
    objective = objective_dir.name
    results[objective] = {}

    for step_dir in objective_dir.iterdir():
        if not step_dir.is_dir():
            continue  # Skip files
        
        step = step_dir.name

        X = np.load(step_dir / "embeddings.npy")
        n_samples = X.shape[0]  # Get number of samples from X

        labels_path = step_dir / "labels.npy"
        if not labels_path.exists():
            continue  # contrastive / AE without labels

        y = np.load(labels_path)
        
        # Fix: Ensure y has the correct shape to match X
        if y.ndim > 1:
            # If y is 2D, reshape to match number of samples
            if y.shape[0] == n_samples:
                y = y.flatten()[:n_samples]  # Take first n_samples elements
            elif y.shape[1] == n_samples:
                y = y.T.flatten()[:n_samples]  # Transpose if needed
            else:
                # If shape doesn't match, try to reshape intelligently
                y = y.reshape(-1)[:n_samples]
        else:
            # If 1D, ensure it matches n_samples
            if len(y) != n_samples:
                y = y[:n_samples]  # Take first n_samples elements

        classes = np.unique(y)
        
        # Check if we have enough classes and samples for silhouette score
        if len(classes) < 2:
            continue  # Need at least 2 classes
        
        # Check if all classes have at least 2 samples
        min_samples_per_class = min([np.sum(y == c) for c in classes])
        if min_samples_per_class < 2:
            continue  # silhouette_score requires at least 2 samples per class

        intra = []
        for c in classes:
            Xc = X[y == c]
            if len(Xc) == 0:
                continue
            intra.append(np.mean(np.linalg.norm(Xc - Xc.mean(0), axis=1)))

        if len(intra) == 0:
            continue  # No valid classes found

        centroids = [X[y == c].mean(0) for c in classes]
        inter = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                inter.append(np.linalg.norm(centroids[i] - centroids[j]))

        if len(inter) == 0:
            continue  # Need at least 2 classes for inter-class distance

        # Calculate silhouette score (now safe since we've checked requirements)
        try:
            sil = silhouette_score(X, y)
        except ValueError as e:
            print(f"Warning: Could not calculate silhouette for {objective}/{step}: {e}")
            continue

        results[objective][step] = {
            "intra_class_variance": float(np.mean(intra)),
            "inter_class_distance": float(np.mean(inter)),
            "silhouette": float(sil),
        }

with open(OUT_DIR / "class_geometry.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved class geometry results.")