#!/bin/bash
set -uo pipefail

# ========================================
# Hendrycks MATH (competition_math) 评测：一次跑到 MAX_LEN，再按多段 token 长度截断重打分
# 数据:    data/eval_datasets/hendrycks_math_train/all/test.jsonl
# adapter: evalscope 的 competition_math (subset=Level 1..5, reformat_subset)
#          - 本地只有 test 没有 train，所以把 few_shot_num 改成 0
# 注意:    7500 条全量评测很慢；如要抽样改 extra_eval_args 里的 limit
# ========================================
MODEL_PATH=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_grpo/v7-20260417-182219/checkpoint-44000
GPU_ID=1
# ========================================

PASS_K=4
TEMPERATURE=1
TOP_P=0.8
MAX_LEN=16384
TRUNCATE_LENGTHS=(2048 4096 8192 16384)
EVAL_LIMIT=500

DATA_DIR=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/eval_datasets/hendrycks_math_train/all

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT_DIR="${MODEL_PATH}/eval_results/hendrycks_math_${TIMESTAMP}"
EVAL_OUTPUT_DIR="${EVAL_ROOT_DIR}/max_len_${MAX_LEN}"
STATS_TSV="${EVAL_ROOT_DIR}/stats.tsv"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "${EVAL_ROOT_DIR}"
{
  echo -e "# model_path\t${MODEL_PATH}"
  echo -e "# gpu\t${GPU_ID}"
  echo -e "# repeats\t${PASS_K}\ttemperature\t${TEMPERATURE}\ttop_p\t${TOP_P}"
  echo -e "# bench\tcompetition_math\tlimit\t${EVAL_LIMIT}"
  echo -e "# strategy\trollout_once_at_${MAX_LEN}_then_truncate"
  echo -e "model_path\tmax_new_tokens\tdataset\tmetric\tvalue"
} > "${STATS_TSV}"

echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "Dataset:    Hendrycks MATH  (${DATA_DIR})"
echo "GPU:        ${GPU_ID}"
echo "Strategy:   rollout once at ${MAX_LEN}, then truncate"
echo "Truncate:   ${TRUNCATE_LENGTHS[*]}"
echo "Limit:      ${EVAL_LIMIT}"
echo "Stats:      ${STATS_TSV}"
echo "=========================================="

PORT=$((8000 + GPU_ID))

echo ""
echo "=========================================="
echo "Step 1: Rollout at max_new_tokens=${MAX_LEN}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift eval \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --eval_backend Native \
    --eval_dataset competition_math \
    --eval_dataset_args "{
        \"competition_math\": {
            \"local_path\": \"${DATA_DIR}\",
            \"few_shot_num\": 0,
            \"train_split\": null,
            \"aggregation\": \"mean_and_pass_at_k\"
        }
    }" \
    --extra_eval_args "{\"repeats\": ${PASS_K}, \"limit\": ${EVAL_LIMIT}}" \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --max_new_tokens ${MAX_LEN} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --stream false \
    --eval_output_dir "${EVAL_OUTPUT_DIR}"

if [ -d "${EVAL_OUTPUT_DIR}/native" ]; then
  find "${EVAL_OUTPUT_DIR}/native" -type d -name "predictions" -exec rm -rf {} + 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "Step 2: Truncate & re-score at ${TRUNCATE_LENGTHS[*]}"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "${SCRIPT_DIR}/truncate_score.py" \
    --eval_output_dir "${EVAL_OUTPUT_DIR}" \
    --stats_tsv "${STATS_TSV}" \
    --model_path "${MODEL_PATH}" \
    --pass_k "${PASS_K}" \
    --truncate_lengths "${TRUNCATE_LENGTHS[@]}"

echo ""
echo "=========================================="
echo "Done. Stats: ${STATS_TSV}"
cat "${STATS_TSV}"
echo "=========================================="
