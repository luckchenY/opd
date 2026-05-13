#!/bin/bash
# TIP / Soft-OR On-Policy Distillation (arxiv 2604.14084v2, Eq. 5-7)
#
# Selects the top-rho fraction of on-policy tokens by Soft-OR score
#     s_t = h_hat + delta_hat - h_hat * delta_hat
# where h_t = H(P_S) / log|V| and delta_t = D_KL(P_S || P_T), each min-max
# normalized over the batch.
#
# KL direction:
#     学生在前的 KL：D_KL(P_S || P_T) = E_{x~P_S}[log P_S - log P_T]。
#     在 GKD 论文 (Agarwal et al. 2024) 的命名约定下，这是 reverse KL（β=1，
#     mode-seeking）；TIP/Soft-OR 论文把同一个东西叫 "forward KL"。数学上等价。
#
# Hyper-params used below:
#     --lmbda 1    : 100% 学生 on-policy 采样（每步都用 student model rollout）
#     --seq_kd false: 不做 sequence-level KD（lmbda=1 时也走不到这条分支）
#     --beta 1     : 注意 — gkd_softor trainer 把 KL 方向"写死"为 reverse KL，
#                    swift CLI 上的 --beta 实际上不影响损失计算；此处写 1 仅是
#                    为了让"记录下来的 GKD 配置"与实际计算保持一致，**改成其他值
#                    （如 0、0.5）也不会改变 loss**。要真正切换 KL 方向只能换回
#                    --rlhf_type gkd（父类 GKDTrainer 才会真的读 beta）。

MODEL=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_gkd_softor/v0-20260508-143250/v0-20260508-143310/checkpoint-2350
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

# rho — top-rho fraction of tokens kept. Paper reports 0.5 / 0.2 / 0.1.
SOFTOR_RETENTION_RATIO=0.2

OUTPUT_DIR=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_gkd_softor/v0-$(date +%Y%m%d-%H%M%S)

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-gkd-on-policy-softor}"

export HF_DATASETS_CACHE="/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Output directory: ${OUTPUT_DIR}"
echo "Soft-OR retention ratio (rho): ${SOFTOR_RETENTION_RATIO}"

NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type gkd_softor \
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
    --softor_retention_ratio ${SOFTOR_RETENTION_RATIO} \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 4 \
    --num_generations 8 \
    --save_strategy epoch \
    --save_total_limit 5 \
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
