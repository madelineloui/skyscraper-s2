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
from tqdm import tqdm
from utils.backbones import load_backbone
from utils.cpd_utils import get_feats, true_bkpts, cpd_confusion, cpd_mae


def eval_cpd(jpg_paths, model, backbone_type, device, gt_bkpts, pen=2, pelt_model="l2", plot=False):
    
    # First extract features for each time point
    X = []
    for jpg_path in jpg_paths:
        feats = get_feats(jpg_path, model, backbone_type=backbone_type, device=device)        
        X.append(feats.reshape(-1).detach().cpu().numpy())   
    
    X = np.array(X)
    
    # Predict change points
    model = rpt.Pelt(model="l2").fit(X)
    pred_bkpts = model.predict(pen=pen)

    #print("Change points:", pred_bkpts)
    
    # Metrics
    tp, fp, tn, fn = cpd_confusion(pred_bkpts, gt_bkpts)
    mae = cpd_mae(pred_bkpts, gt_bkpts)
    
    # print('-- Performance --')
    # print(f'TP: {tp}')
    # print(f'FP: {fp}')
    # print(f'TN: {tn}')
    # print(f'FN: {fn}')
    # print(f'MAE: {mae}')
    
    if plot:
        plot_jpgs(jpg_paths, pred_bkpts, predict=True)
        plot_jpgs(jpg_paths, gt_bkpts)
        X_hw = X.reshape(-1, 16, 16, 1024)
        plot_feat_row(X_hw)
        
    return tp, fp, tn, fn, mae  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with annotations")
    ap.add_argument("--root", required=True, help="Root directory containing imagery/")
    ap.add_argument("--backbone", default="remoteclip-14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pen", type=int, default=2)
    ap.add_argument("--pelt_model", default='l2')
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f'Using {device}!')

    root = Path(args.root)
    imagery_root = root / "imagery"

    backbone_type = args.backbone
    model = load_backbone(backbone_type).to(device).eval()

    df = pd.read_csv(args.csv, dtype=str)

    article_ids = df["article_id"].tolist()
    print(f"{len(article_ids)} articles found!")

    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    mae_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        article_id = row['article_id']
        article_dir = imagery_root / str(article_id)

        jpg_paths = sorted(article_dir.rglob("*.jpg"), key=lambda p: str(p).lower())
        #print(f"\n=== {article_id} ({len(jpg_paths)} images) ===")

        #print(f'Start date: {row['event_start_date']}')
        #print(f'End date: {row['event_end_date']}')
        gt_bkpts = true_bkpts(jpg_paths, row['event_start_date'], row['event_end_date'])
        #print(f'True change points: {gt_bkpts}')

        tp, fp, tn, fn, mae = eval_cpd(jpg_paths, model, backbone_type=backbone_type, device=device, gt_bkpts=gt_bkpts, pen=args.pen, pelt_model=args.pelt_model, plot=False)
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
        mae_list.append(mae)
    
    # Performance summary
    TP = sum(tp_list)
    FP = sum(fp_list)
    TN = sum(tn_list)
    FN = sum(fn_list)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    avg_mae = np.mean([x for x in mae_list if x is not None])

    print("---- CPD Metrics ----")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print(f"FN: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")    
    

if __name__ == "__main__":
    main()
