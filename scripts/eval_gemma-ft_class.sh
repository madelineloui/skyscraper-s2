#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

source activate skyscraper
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval/eval_gemma.py \
 --json_path data/skyscraper_gdelt_sentinel/vqa/teochat_sentinel_event_classification_val.json \
 --data_root data/skyscraper_gdelt_sentinel \
 --model_path /home/gridsan/manderson/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767 \
 --lora_ckpt /home/gridsan/manderson/skyscraper-s2/out/finetune/gemma/checkpoint-2320 \
 --output_csv out/vqa/classification_gemma-ft.csv \