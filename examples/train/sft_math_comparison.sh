#!/bin/bash
# SFT baseline for comparison experiments
# Uses the same model / hyperparameters as the GKD / GRPO scripts.

MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-1.7B

# ===== 选择数据集 (二选一, 注释掉不用的) =====
#
# --- MATH (EleutherAI/hendrycks_math, 7 个子集, ~7500 条) ---
DATASET='hf::EleutherAI/hendrycks_math:algebra/counting_and_probability/geometry/intermediate_algebra/number_theory/prealgebra/precalculus'
EXTRA_DATASET_ARGS=""
#
# --- DAPO-Math-17k (字节&清华, ~17000 条) ---
# DATASET='hf::BytedTsinghua-SIA/DAPO-Math-17k'
# EXTRA_DATASET_ARGS="--columns '{\"prompt\":\"messages\"}'"

# W&B: export WANDB_API_KEY before running
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-sft-baseline}"

# Hugging Face: use local Arrow cache, skip Hub network calls
export HF_DATASETS_CACHE="/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --report_to wandb \
    --use_hf true \
    --model "${MODEL}" \
    --tuner_type full \
    --dataset ${DATASET} \
    ${EXTRA_DATASET_ARGS} \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir output_sft \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --attn_impl flash_attn
