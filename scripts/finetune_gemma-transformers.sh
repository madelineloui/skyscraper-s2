#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G

module load miniforge/23.11.0-0
conda activate finetune

# Optional, but recommended if Gemma access/rate limits require auth:
# export HF_TOKEN=

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python train/finetune_transformers.py \
    --model_type gemma3 \
    --model_name_or_path google/gemma-3-4b-it \
    --data_name /home/mloui/data/skyscraper_gdelt_sentinel/train/train.json \
    --data_split train \
    --output_dir /home/mloui/orcd/pool/outputs/gemma \
    --cache_dir /home/mloui/orcd/pool/cache_dir \
    --bits 4 \
    --bf16 \
    --tf32 \
    --lora_enable \
    --lora_r 128 \
    --lora_alpha 256 \
    --freeze_backbone \
    --max_frames 8 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 250 \
    --save_total_limit 10 \
    --model_max_length 4096 \
    --gradient_checkpointing \
    --dataloader_num_workers 0 \
    --report_to none