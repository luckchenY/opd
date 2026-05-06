#!/bin/bash
# Resume Confidence-Aware On-Policy Distillation (GKD-Confidence) from checkpoint

MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-1.7B
TEACHER_MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-4B

# 数据集
DATASET='hf::EleutherAI/hendrycks_math:algebra/counting_and_probability/geometry/intermediate_algebra/number_theory/prealgebra/precalculus'
EXTRA_DATASET_ARGS=""

ENTROPY_THRESHOLD=1.0

# 输出目录和检查点路径
OUTPUT_DIR=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_gkd_confidence/v0-20260331-012058
RESUME_CHECKPOINT=${OUTPUT_DIR}/checkpoint-2000

# W&B 配置
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-gkd-on-policy-confidence}"

# Hugging Face 缓存配置
export HF_DATASETS_CACHE="/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Resuming training from checkpoint: ${RESUME_CHECKPOINT}"
echo "Output directory: ${OUTPUT_DIR}"

NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type gkd_confidence \
    --report_to wandb \
    --use_hf true \
    --model "${MODEL}" \
    --teacher_model "${TEACHER_MODEL}" \
    --tuner_type full \
    --dataset ${DATASET} \
    ${EXTRA_DATASET_ARGS} \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --teacher_entropy_threshold ${ENTROPY_THRESHOLD} \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16000 \
    --max_completion_length 8192 \
    --output_dir ${OUTPUT_DIR} \
    --resume_from_checkpoint ${RESUME_CHECKPOINT} \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --top_k 50 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4
