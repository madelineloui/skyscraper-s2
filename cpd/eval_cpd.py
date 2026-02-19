import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from utils.backbones import load_backbone, prepare_image_for_backbone, extract_backbone_features

def eval_cpd(jpg_paths, model, backbone_type, device):
    with torch.no_grad():
        for p in jpg_paths:
            img = Image.open(p).convert("RGB")

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
            print(x.shape)

            x = prepare_image_for_backbone(x, backbone_type)

            feats = extract_backbone_features(x, model, backbone_type)

            print(p.name, getattr(feats, "shape", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with annotations")
    ap.add_argument("--root", required=True, help="Root directory containing imagery/")
    ap.add_argument("--backbone", default="remoteclip-14")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    root = Path(args.root)
    imagery_root = root / "imagery"

    backbone_type = args.backbone
    model = load_backbone(backbone_type).to(device).eval()

    df = pd.read_csv(args.csv, dtype=str)
    if "article_id" not in df.columns:
        raise SystemExit("CSV missing required column: article_id")

    article_ids = df["article_id"].tolist()
    print(f"{len(article_ids)} articles found!")

    for article_id in article_ids:
        article_dir = imagery_root / str(article_id)

        if not article_dir.is_dir():
            print(f"[skip] {article_id} (missing: {article_dir})")
            continue

        jpg_paths = sorted(article_dir.rglob("*.jpg"), key=lambda p: str(p).lower())
        if not jpg_paths:
            print(f"[skip] {article_id} (no .jpg under {article_dir})")
            continue

        print(f"\n=== {article_id} ({len(jpg_paths)} jpg) ===")
        eval_cpd(jpg_paths, model, backbone_type=backbone_type, device=device)


if __name__ == "__main__":
    main()
