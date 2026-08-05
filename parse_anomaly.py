import re
from pathlib import Path

import pandas as pd


ROOT = Path("out/anomaly")
METRICS = ["Accuracy", "Precision", "Recall", "F1"]

results = []

for metrics_path in ROOT.glob("*/*/*/STD_*/tol_*/cpd_metrics.txt"):
    # Expected path:
    # out/anomaly/{satellite}/{backbone}/{feat_type}/STD_{std}/tol_{tol}/cpd_metrics.txt
    satellite = metrics_path.parts[-6]
    backbone = metrics_path.parts[-5]
    feat_type = metrics_path.parts[-4]
    std = metrics_path.parts[-3].replace("STD_", "")
    tol = metrics_path.parts[-2].replace("tol_", "")

    text = metrics_path.read_text()

    row = {
        "satellite": satellite,
        "backbone": backbone,
        "feat_type": feat_type,
        "std": float(std),
        "tol": float(tol),
    }

    for metric in METRICS:
        match = re.search(rf"^{metric}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
        row[metric] = float(match.group(1)) if match else None

    results.append(row)

df = pd.DataFrame(results).sort_values(
    ["satellite", "backbone", "feat_type", "std", "tol"]
)

print(df.to_string(index=False))
df.to_csv("out/anomaly/anomaly_metrics_summary.csv", index=False)