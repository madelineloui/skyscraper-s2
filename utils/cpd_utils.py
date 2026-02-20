import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from datetime import datetime
import ruptures as rpt
from utils.backbones import prepare_image_for_backbone, extract_backbone_features


def get_feats(jpg_path, model, backbone_type, device):
    with torch.no_grad():
        img = Image.open(jpg_path).convert("RGB")

        # (1, C, H, W) float32 on device
        x = (
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        #print(x.shape)

        x = prepare_image_for_backbone(x, backbone_type)

        feats = extract_backbone_features(x, model, backbone_type)

        #print(jpg_path.name, feats.shape)

    return feats


def true_bkpts(jpg_paths, event_start_date=None, event_end_date=None):

    # parse jpg dates
    dates = [
        datetime.strptime(p.name.split("_")[2], "%Y%m%d").date()
        for p in jpg_paths
    ]
    
    T = len(dates)
    
    if isinstance(event_start_date, float) or isinstance(event_end_date, float):
        return [T]

    s = datetime.strptime(event_start_date, "%Y-%m-%d").date()
    e = datetime.strptime(event_end_date, "%Y-%m-%d").date()

    # first image where event visible
    start = next(i for i,d in enumerate(dates) if d >= s)

    # last image where event visible
    end = max(i for i,d in enumerate(dates) if d <= e)
        
    # build bkpts (ruptures-style ends)
    bkpts = []
    if start != 0:
        bkpts.append(start)
    if end + 1 != T:
        bkpts.append(end + 1)
    bkpts.append(T)
        
    return bkpts


def cpd_confusion(pred, gt, tol=2):
    # remove final endpoint
    pred = pred[:-1]
    gt = gt[:-1]

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
    tn = 0   # not defined per changepoint, only per sequence

    return tp, fp, tn, fn


def cpd_mae(pred, gt):
    pred = pred[:-1]
    gt = gt[:-1]

    if not pred or not gt:
        return None

    errs = []
    for g in gt:
        errs.append(min(abs(g - p) for p in pred))

    return sum(errs) / len(errs)