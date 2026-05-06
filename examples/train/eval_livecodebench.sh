#!/bin/bash
set -uo pipefail

# ========================================
# LiveCodeBench v6 评测：一次跑到 MAX_LEN，再按多段 token 长度截断重跑测试
# 数据:    data/eval_datasets/livecodebench_v6/test.jsonl
# adapter: evalscope 的 live_code_bench（subset_list 覆盖成 ['default']）
# 重打分: 代码必须重新执行，所以用 truncate_score_code.py（按 group_id 对应回原 jsonl
#          取 evaluation_sample，再调 codegen_metrics 跑测试用例）
# ========================================
MODEL_PATH=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_grpo/v7-20260417-182219/checkpoint-44000
GPU_ID=1
# ========================================

PASS_K=4
TEMPERATURE=1
TOP_P=0.8
MAX_LEN=32768
TRUNCATE_LENGTHS=(4096 8192 16384 32768)
REVIEW_TIMEOUT=6
NUM_PROCESS=16

DATA_DIR=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/eval_datasets/livecodebench_v6
DATA_JSONL="${DATA_DIR}/test.jsonl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT_DIR="${MODEL_PATH}/eval_results/livecodebench_v6_${TIMESTAMP}"
EVAL_OUTPUT_DIR="${EVAL_ROOT_DIR}/max_len_${MAX_LEN}"
STATS_TSV="${EVAL_ROOT_DIR}/stats.tsv"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "${EVAL_ROOT_DIR}"
{
  echo -e "# model_path\t${MODEL_PATH}"
  echo -e "# gpu\t${GPU_ID}"
  echo -e "# repeats\t${PASS_K}\ttemperature\t${TEMPERATURE}\ttop_p\t${TOP_P}"
  echo -e "# bench\tlive_code_bench_v6\treview_timeout\t${REVIEW_TIMEOUT}"
  echo -e "# strategy\trollout_once_at_${MAX_LEN}_then_truncate_and_reexec"
  echo -e "model_path\tmax_new_tokens\tdataset\tmetric\tvalue"
} > "${STATS_TSV}"

echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "Dataset:    LiveCodeBench v6  (${DATA_DIR})"
echo "GPU:        ${GPU_ID}"
echo "Strategy:   rollout once at ${MAX_LEN}, then truncate & re-exec"
echo "Truncate:   ${TRUNCATE_LENGTHS[*]}"
echo "Timeout:    ${REVIEW_TIMEOUT}s / test case"
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
    --eval_dataset live_code_bench \
    --eval_dataset_args "{
        \"live_code_bench\": {
            \"local_path\": \"${DATA_DIR}\",
            \"subset_list\": [\"default\"],
            \"review_timeout\": ${REVIEW_TIMEOUT},
            \"aggregation\": \"mean_and_pass_at_k\"
        }
    }" \
    --extra_eval_args "{\"repeats\": ${PASS_K}}" \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --max_new_tokens ${MAX_LEN} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --stream false \
    --eval_output_dir "${EVAL_OUTPUT_DIR}"

# predictions 可以删，reviews 里已有完整 prediction 文本
if [ -d "${EVAL_OUTPUT_DIR}/native" ]; then
  find "${EVAL_OUTPUT_DIR}/native" -type d -name "predictions" -exec rm -rf {} + 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "Step 2: Truncate & re-execute at ${TRUNCATE_LENGTHS[*]}"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "${SCRIPT_DIR}/truncate_score_code.py" \
    --eval_output_dir "${EVAL_OUTPUT_DIR}" \
    --stats_tsv "${STATS_TSV}" \
    --model_path "${MODEL_PATH}" \
    --pass_k "${PASS_K}" \
    --truncate_lengths "${TRUNCATE_LENGTHS[@]}" \
    --data_jsonl "${DATA_JSONL}" \
    --review_timeout "${REVIEW_TIMEOUT}" \
    --num_process "${NUM_PROCESS}"

echo ""
echo "=========================================="
echo "Done. Stats: ${STATS_TSV}"
cat "${STATS_TSV}"
echo "=========================================="
