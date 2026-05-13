#!/bin/bash
# Confidence-Aware On-Policy Distillation (GKD-Confidence)
#
# Difference from on_policy_distillation.sh:
#   --rlhf_type gkd_confidence
#       Per microbatch, ranks all valid tokens by the entropy ratio
#       r = H_teacher / H_student (ascending) and keeps only the
#       lowest-ratio entropy_ratio_keep_fraction of them.  Tokens where
#       the teacher is much more uncertain than the student (the student
#       is already sharper) are silently dropped, since distilling there
#       only adds noise.
#
# Keep-fraction guidance:
#   value | behaviour
#   ------|--------------------------------------------------------------
#   0.25  | aggressive — only the most teacher-confident 25% are kept
#   0.50  | default — keep the lowest-ratio 50% per microbatch
#   0.75  | relaxed — keep the lowest-ratio 75%
#   1.00  | no filtering (equivalent to plain GKD)
#
# Background: EOPD paper – https://arxiv.org/abs/2603.07079

MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen2.5-1.5B-Instruct
TEACHER_MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen2.5-14B-Instruct

# ===== 选择数据集 (二选一, 注释掉不用的) =====
#
# --- MATH (EleutherAI/hendrycks_math, 7 个子集, ~7500 条) ---
DATASET='hf::EleutherAI/hendrycks_math:algebra/counting_and_probability/geometry/intermediate_algebra/number_theory/prealgebra/precalculus'
EXTRA_DATASET_ARGS=""
#
# --- DAPO-Math-17k (字节&清华, ~17000 条) ---
# DATASET='hf::BytedTsinghua-SIA/DAPO-Math-17k'
# EXTRA_DATASET_ARGS="--columns '{\"prompt\":\"messages\"}'"

ENTROPY_RATIO_KEEP_FRACTION=0.5

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
    --gkd_logits_topk 128 \
    --use_hf true \
    --model "${MODEL}" \
    --teacher_model "${TEACHER_MODEL}" \
    --tuner_type full \
    --dataset ${DATASET} \
    ${EXTRA_DATASET_ARGS} \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --entropy_ratio_keep_fraction ${ENTROPY_RATIO_KEEP_FRACTION} \
    --torch_dtype bfloat16 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 16384 \
    --max_completion_length 8192 \
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
