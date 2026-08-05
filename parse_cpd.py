import re
from pathlib import Path

import pandas as pd


ROOT = Path("out/cpd")
METRICS = ["Accuracy", "Precision", "Recall", "F1"]

results = []

for metrics_path in ROOT.glob(
    "*/remoteclip-14/*/l2_penalty_1/tol_*/cpd_metrics.txt"
):
    parts = metrics_path.relative_to(ROOT).parts

    satellite = parts[0]
    backbone = parts[1]
    feat_type = parts[2]
    penalty = parts[3].replace("l2_penalty_", "")
    tol = parts[4].replace("tol_", "")

    text = metrics_path.read_text()

    row = {
        "satellite": satellite,
        "backbone": backbone,
        "feat_type": feat_type,
        "penalty": float(penalty),
        "tol": int(tol),
    }

    for metric in METRICS:
        match = re.search(
            rf"^{metric}:\s*([0-9.eE+-]+)",
            text,
            re.MULTILINE,
        )
        row[metric] = float(match.group(1)) if match else None

    results.append(row)

if not results:
    raise FileNotFoundError(
        "No files found matching:\n"
        "out/cpd/{satellite}/remoteclip-14/{feat_type}/"
        "l2_penalty_1/tol_{tol}/cpd_metrics.txt"
    )

df = pd.DataFrame(results).sort_values(
    ["satellite", "feat_type", "tol"]
)

output_path = ROOT / "cpd_metrics_summary.csv"
df.to_csv(output_path, index=False)

print(df.to_string(index=False))
print(f"\nSaved to: {output_path}")