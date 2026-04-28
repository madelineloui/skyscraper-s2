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
from utils.backbones import load_backbone, load_backbone_and_tokenizer_and_preprocess, get_feats, get_caption_sim
from utils.anomaly_utils import anomaly_confusion, anomaly_mae, true_anomaly_points

def eval_anomaly(
    jpg_paths, model, backbone_type, device, gt_bkpts,
    caption=None, tokenizer=None, preprocess=None,
    feat_type='cls', tol=2, std_thresh=2.0, plot=False
):
    X = []

    for jpg_path in jpg_paths:
        if caption is not None:
            feats = get_caption_sim(jpg_path, caption, model, preprocess, tokenizer, device)
        else:
            feats = get_feats(jpg_path, model, backbone_type=backbone_type, device=device)

        X.append(feats.reshape(-1).detach().cpu().numpy())

    X = np.array(X)
    #print(X.shape)

    # For caption/class similarity, make it 1D
    if caption is not None:
        X = X.reshape(-1, 1)

    # z-score normalize
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    #print(Z.shape)

    # anomaly score per timestep
    if Z.shape[1] == 1:
        scores = np.abs(Z[:, 0])
        pred_bkpts = np.where(scores > std_thresh)[0].tolist()
    else:
        # use frame-to-frame change instead of norm of embedding
        diffs = np.linalg.norm(X[1:] - X[:-1], axis=1)  # (T-1,)
        scores = (diffs - diffs.mean()) / (diffs.std() + 1e-6)
        pred_bkpts = (np.where(scores > std_thresh)[0] + 1).tolist()

    # remove first and last index, if included
    pred_bkpts = [i for i in pred_bkpts if 0 < i < len(X) - 1]

    # print("Anomaly scores:", scores)
    # print("Pred anomaly points:", pred_bkpts)
    # print("GT:", gt_bkpts)
    # print()
    # print()

    tp, fp, tn, fn = anomaly_confusion(pred_bkpts, gt_bkpts, tol=tol)
    mae = anomaly_mae(pred_bkpts, gt_bkpts)

    return tp, fp, tn, fn, mae 


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with annotations")
    ap.add_argument("--root", required=True, help="Root directory containing imagery/")
    ap.add_argument("--backbone", default="remoteclip-14")
    ap.add_argument("--feat_type", default='cls') #cls, patch, cap_sim
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--std_thresh", type=float, default=2)
    ap.add_argument("--output_dir", required=True, help="Directory to save metrics")
    
    args = ap.parse_args()
    
    print('== PARAMS ==')
    print(f'Backbone: {args.backbone}')
    print(f'STD Threshold: {args.std_thresh}')
    print(f'Feature type: {args.feat_type}')

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f'Using {device}!')

    root = Path(args.root)
    imagery_root = root / "imagery"

    backbone_type = args.backbone
    if args.feat_type in ['cap_sim', 'class_sim']:
        model, tokenizer, preprocess = load_backbone_and_tokenizer_and_preprocess(backbone_type)
        model.to(device).eval()
    else:
        model = load_backbone(backbone_type).to(device).eval()
        tokenizer = None
        preprocess = None

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
        # print(f"\n=== {article_id} ({len(jpg_paths)} images) ===")

        # print(f'Start date: {row['event_start_date']}')
        # print(f'End date: {row['event_end_date']}')
        gt_bkpts = true_anomaly_points(jpg_paths, row['event_start_date'], row['event_end_date'])
        # print(f'True anomaly points: {gt_bkpts}')

        if args.feat_type == 'cap_sim':
            caption = row['event_caption']
            tp, fp, tn, fn, mae = eval_anomaly(jpg_paths, model, backbone_type=backbone_type, device=device, gt_bkpts=gt_bkpts, caption=caption, tokenizer=tokenizer, preprocess=preprocess, feat_type=args.feat_type,  tol=args.tol, std_thresh=args.std_thresh, plot=False)
        elif args.feat_type == 'class_sim':
            caption = row['event_type']
            tp, fp, tn, fn, mae = eval_anomaly(jpg_paths, model, backbone_type=backbone_type, device=device, gt_bkpts=gt_bkpts, caption=caption, tokenizer=tokenizer, preprocess=preprocess, feat_type=args.feat_type,  tol=args.tol, std_thresh=args.std_thresh, plot=False)
        else:
            tp, fp, tn, fn, mae = eval_anomaly(jpg_paths, model, backbone_type=backbone_type, device=device, gt_bkpts=gt_bkpts, feat_type=args.feat_type,  tol=args.tol, std_thresh=args.std_thresh, plot=False)
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

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
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
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")  
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "cpd_metrics.txt"
    with open(out_file, "w") as f:
        f.write("---- Anomaly Metrics ----\n\n")

        # Write all argparse arguments automatically
        f.write("---- Config ----\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

        f.write("\n---- Results ----\n")
        f.write(f"TP: {TP}\n")
        f.write(f"FP: {FP}\n")
        f.write(f"TN: {TN}\n")
        f.write(f"FN: {FN}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")

    print(f"Metrics saved to {out_file}")

    print('\n\n\n')

if __name__ == "__main__":
    main()
