#!/bin/bash
# Confidence-Aware On-Policy Distillation (GKD-Confidence)
#
# Difference from on_policy_distillation.sh:
#   --rlhf_type gkd_confidence
#       Only back-propagates through token positions where the teacher's
#       per-token entropy is BELOW teacher_entropy_threshold (in nats).
#       High-entropy positions (teacher is uncertain) are silently skipped.
#
# Entropy threshold guidance (natural-log / nats scale):
#   top-1 prob | approx entropy  | interpretation
#   -----------|-----------------|----------------------------------
#     >= 0.97  | ~0.2 nats       | very confident (deterministic)
#     ~0.90    | ~0.5 nats       | confident
#     ~0.75    | ~1.0 nats       | fairly confident   <- default
#     ~0.55    | ~1.5 nats       | mildly uncertain
#     ~0.35    | ~2.0 nats       | uncertain
#
# Reference: EOPD paper – https://arxiv.org/abs/2603.07079

MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-1.7B
TEACHER_MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-8B

# ===== 选择数据集 (二选一, 注释掉不用的) =====
#
# --- MATH (EleutherAI/hendrycks_math, 7 个子集, ~7500 条) ---
DATASET='hf::EleutherAI/hendrycks_math:algebra/counting_and_probability/geometry/intermediate_algebra/number_theory/prealgebra/precalculus'
EXTRA_DATASET_ARGS=""
#
# --- DAPO-Math-17k (字节&清华, ~17000 条) ---
# DATASET='hf::BytedTsinghua-SIA/DAPO-Math-17k'
# EXTRA_DATASET_ARGS="--columns '{\"prompt\":\"messages\"}'"

ENTROPY_THRESHOLD=0.248

# W&B: export WANDB_API_KEY before running (do NOT hardcode secrets in scripts)
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-gkd-on-policy-confidence}"

# Hugging Face: use local Arrow cache, skip Hub network calls
export HF_DATASETS_CACHE="/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# CUDA_VISIBLE_DEVICES=0,1,2,3 \


NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd_confidence \
    --report_to wandb \
    --temperature 1 \
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
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --max_completion_length 4096 \
    --output_dir output_gkd_confidence \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --top_k 50 \
    --top_p 1 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4
