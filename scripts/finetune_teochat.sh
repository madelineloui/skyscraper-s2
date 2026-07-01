#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=64G

# Loading the required modules
module load miniforge/23.11.0-0
conda activate teochat2

HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 deepspeed videollava/train/train.py \
    --bits 8 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path jirvin16/TEOChat \
    --version v1 \
    --data_name /home/mloui/data/skyscraper_gdelt_sentinel/train \
    --data_split train \
    --image_tower LanguageBind/LanguageBind_Image \
    --freeze_backbone True \
    --freeze_mm_mlp_adapter False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/mloui/orcd/pool/outputs/teochat \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --video_as_image_list True \
    --prompt_strategy interleave \
    --chronological_prefix True \
    --lazy_preprocess True \
    --report_to none \
    --cache_dir "cache_dir"
