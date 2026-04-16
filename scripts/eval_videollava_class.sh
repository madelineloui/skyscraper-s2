#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required modules
source /etc/profile
module load anaconda/2023a

source activate skyscraper
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval/eval_videollava.py \
 --json_path data/skyscraper_gdelt_sentinel/vqa/teochat_sentinel_event_classification_val.json \
 --data_root data/skyscraper_gdelt_sentinel \
 --model_path /home/gridsan/manderson/.cache/huggingface/hub/models--LanguageBind--Video-LLaVA-7B-hf/snapshots/4cf9d8cfc76a54f46a4cb43be5368b46b7f0d736 \
 --output_csv out/vqa/classification_videollava.csv