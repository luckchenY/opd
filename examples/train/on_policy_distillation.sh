#!/bin/bash
# On-Policy Distillation (GKD)
# GKD 纯 on-policy: --lmbda 1 --beta 1 --seq_kd false
# 说明: https://swift.readthedocs.io/en/latest/Instruction/GKD.html

MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen2.5-1.5B-Instruct
TEACHER_MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen2.5-14B-Instruct

# 数据集
# --- MATH (EleutherAI/hendrycks_math, 7 个子集, ~7500 条) ---
DATASET='hf::EleutherAI/hendrycks_math:algebra/counting_and_probability/geometry/intermediate_algebra/number_theory/prealgebra/precalculus'
EXTRA_DATASET_ARGS=""
# --- DAPO-Math-17k (deduped) ---
# HF 发布版 DAPO 有 ~1.79M raw rows = ~17k 独立题目 × ~103 重复
# (见 tools/process_reward_weighted_distill.py 的 _extract_problem_text + 注释)
# 直接喂给 swift 等于把每题在一个 epoch 里 rollout ~100 次。
# 第一次跑前用 tools/build_dapo_dedup_jsonl.py 生成 dedup 后的 jsonl。
# DAPO_DEDUP_JSONL="/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/dapo_math_17k_dedup.jsonl"
# if [ ! -f "${DAPO_DEDUP_JSONL}" ]; then
#     echo "[prep] Building deduped DAPO jsonl at ${DAPO_DEDUP_JSONL} ..."
#     python "$(dirname "$0")/../../tools/build_dapo_dedup_jsonl.py" \
#         --out "${DAPO_DEDUP_JSONL}"
# fi
# DATASET="${DAPO_DEDUP_JSONL}"
# EXTRA_DATASET_ARGS=""

OUTPUT_DIR=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_gkd_on_policy/v0-$(date +%Y%m%d-%H%M%S)

# W&B: export WANDB_API_KEY before running (do NOT hardcode secrets in scripts)
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-gkd-on-policy}"

export HF_DATASETS_CACHE="/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Output directory: ${OUTPUT_DIR}"

NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type gkd \
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
    --temperature 1.0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 1 \
    --max_length 16384 \
    --max_completion_length 8192 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --log_completions true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --top_k 50 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3
