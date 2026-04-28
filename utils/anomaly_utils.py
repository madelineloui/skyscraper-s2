import pandas as pd
from datetime import datetime

def anomaly_confusion(pred, gt, tol=2):
    # No GT changepoints
    if len(gt) == 0:
        if len(pred) == 0:
            return 0, 0, 1, 0   # TP, FP, TN, FN
        else:
            return 0, len(pred), 0, 0

    # GT exists
    matched = set()
    tp = 0

    for p in pred:
        for g in gt:
            if abs(p - g) <= tol and g not in matched:
                tp += 1
                matched.add(g)
                break

    fp = len(pred) - tp
    fn = len(gt) - tp
    tn = 0   # still undefined per changepoint

    return tp, fp, tn, fn

def anomaly_mae(pred, gt):
    if not pred or not gt:
        return None

    errs = []
    for g in gt:
        errs.append(min(abs(g - p) for p in pred))

    return sum(errs) / len(errs)

def true_anomaly_points(jpg_paths, event_start_date=None, event_end_date=None):
    dates = [
        datetime.strptime(p.name.split("_")[2], "%Y%m%d").date()
        for p in jpg_paths
    ]

    if pd.isna(event_start_date) or pd.isna(event_end_date):
        return []

    s = datetime.strptime(event_start_date, "%Y-%m-%d").date()
    e = datetime.strptime(event_end_date, "%Y-%m-%d").date()

    T = len(dates)

    # first image where event becomes visible
    start = next((i for i, d in enumerate(dates) if d >= s), None)

    # first image after event ends
    end = next((i for i, d in enumerate(dates) if d > e), None)

    points = []

    if start is not None and 0 < start < T - 1:
        points.append(start)

    if end is not None and 0 < end < T - 1:
        points.append(end)

    return points