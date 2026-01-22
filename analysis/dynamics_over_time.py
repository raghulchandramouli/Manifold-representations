"""
Aggregate geometry metrics across training steps
"""

import json
from pathlib import Path

RESULTS_DIR = Path("analysis/results/dynamics_over_time")
OUT_FILE = RESULTS_DIR /"dynamics_summary.json"

summary = {}

for metric_dir in RESULTS_DIR.iterdir():
    if metric_dir.is_file():
        continue

    for file in metric_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        metric_name = metric_dir.name
        summary[metric_name] = data

with open(OUT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print("Saved dynamics summary.")
