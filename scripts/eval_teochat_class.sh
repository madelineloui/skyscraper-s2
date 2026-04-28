#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

source activate skyscraper2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/gridsan/manderson/.cache/huggingface

python eval/eval_teochat.py \
  --json_path data/skyscraper_gdelt_sentinel/vqa/teochat_sentinel_event_classification_val.json \
  --data_root data/skyscraper_gdelt_sentinel \
  --teochat_path /home/gridsan/manderson/.cache/huggingface/hub/models--jirvin16--TEOChat/snapshots/a727ec6baabcaea1bf621d226f42126eda3cc7c2 \
  --output_csv out/vqa/classification_teochat.csv \
  --load_8bit