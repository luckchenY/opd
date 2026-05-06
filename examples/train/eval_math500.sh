#!/bin/bash
set -uo pipefail

# ========================================
# MATH-500 评测：每档 max_new_tokens 独立跑一次 swift eval
# 数据:    data/eval_datasets/math_500/test.jsonl
# adapter: evalscope 的 math_500 (subset=Level 1..5, reformat_subset)
# ========================================
MODEL_PATH=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_gkd_softor/v0-20260501-142325/v0-20260501-142406/checkpoint-469

GPU_ID=1
# ========================================

PASS_K=8
TEMPERATURE=1
TOP_P=0.8
MAX_NEW_TOKENS_SWEEP=(4096 8192 16384 32768)

DATA_DIR=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/eval_datasets/math_500

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT_DIR="${MODEL_PATH}/eval_results/math500_${TIMESTAMP}"
STATS_TSV="${EVAL_ROOT_DIR}/stats.tsv"

EVAL_MATH500_RUN_ID="eval_math500_$$_$(date +%s)_${GPU_ID}"
export EVAL_MATH500_RUN_ID

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "${EVAL_ROOT_DIR}"
{
  echo -e "# model_path\t${MODEL_PATH}"
  echo -e "# gpu\t${GPU_ID}"
  echo -e "# repeats\t${PASS_K}\ttemperature\t${TEMPERATURE}\ttop_p\t${TOP_P}"
  echo -e "# bench\tmath_500"
  echo -e "# strategy\tsweep_max_new_tokens"
  echo -e "model_path\tmax_new_tokens\tdataset\tmetric\tvalue"
} > "${STATS_TSV}"

PORT=$((8000 + GPU_ID))

echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "Dataset:    MATH-500  (${DATA_DIR})"
echo "GPU:        ${GPU_ID}        Port: ${PORT}"
echo "Sweep:      ${MAX_NEW_TOKENS_SWEEP[*]}"
echo "Stats:      ${STATS_TSV}"
echo "Run ID:     ${EVAL_MATH500_RUN_ID}"
echo "=========================================="

trap 'echo "[info]  trap: cleaning up run ${EVAL_MATH500_RUN_ID}"; _kill_run_residuals 2>/dev/null || true' EXIT INT TERM

# ------------------------------------------------------------------
# env 匹配兜底：只杀"本次 bash 脚本启动的 swift eval / vLLM 子孙"。
# 主清理走 setsid 进程组（见 run_one_eval），env 匹配只是兜底。
# ------------------------------------------------------------------
_pid_belongs_to_run() {
    local pid="$1"
    local env_file="/proc/${pid}/environ"
    [ -r "${env_file}" ] || return 1
    grep -qzxF "EVAL_MATH500_RUN_ID=${EVAL_MATH500_RUN_ID}" "${env_file}" 2>/dev/null || return 1
    return 0
}

_list_run_pids() {
    local pid
    for pid in /proc/[0-9]*; do
        pid=${pid##*/}
        [ "${pid}" = "$$" ] && continue
        _pid_belongs_to_run "${pid}" && echo "${pid}"
    done
}

_kill_run_residuals() {
    local pids
    pids=$(_list_run_pids)
    [ -z "${pids}" ] && return 0
    echo "[info]  killing run residuals on GPU ${GPU_ID}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    local i
    for i in $(seq 1 5); do
        pids=$(_list_run_pids)
        [ -z "${pids}" ] && return 0
        sleep 1
    done
    pids=$(_list_run_pids)
    if [ -n "${pids}" ]; then
        echo "[warn]  SIGKILL residuals: ${pids//$'\n'/ }"
        # shellcheck disable=SC2086
        kill -KILL ${pids} 2>/dev/null || true
    fi
    sleep 3
}

_wait_gpu_clean() {
    local wait_sec="${1:-60}"
    local i mem
    for i in $(seq 1 "${wait_sec}"); do
        mem=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if [ "${mem:-0}" -lt 500 ] && ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$"; then
            return 0
        fi
        sleep 1
    done
    echo "[warn]  GPU ${GPU_ID} still busy after ${wait_sec}s (mem=${mem}MiB); proceeding anyway"
}

# ------------------------------------------------------------------
# 跑一档 swift eval：
#   - setsid 独立 process group → 收尾按 pgid 一锅端，
#     能可靠打到 vLLM v1 EngineCore（它会 rewrite CUDA_VISIBLE_DEVICES，
#     基于 env 的匹配漏不了它，但不可靠）。
#   - 扫 log 的致命错误标志，命中立刻退出，不再等 HARD_TIMEOUT。
# ------------------------------------------------------------------
run_one_eval() {
    local max_new_tokens="$1"
    local out_dir="$2"
    local log_path="$3"

    mkdir -p "$(dirname "${log_path}")"

    _kill_run_residuals
    _wait_gpu_clean 60

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    setsid swift eval \
        --model "${MODEL_PATH}" \
        --port "${PORT}" \
        --eval_backend Native \
        --eval_dataset math_500 \
        --eval_dataset_args "{
            \"math_500\": {\"local_path\": \"${DATA_DIR}\", \"aggregation\": \"mean_and_pass_at_k\"}
        }" \
        --extra_eval_args "{\"repeats\": ${PASS_K}}" \
        --infer_backend vllm \
        --vllm_gpu_memory_utilization 0.9 \
        --max_new_tokens "${max_new_tokens}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --attn_impl flash_attn \
        --torch_dtype bfloat16 \
        --stream false \
        --eval_output_dir "${out_dir}" \
        >"${log_path}" 2>&1 &
    local swift_pid=$!
    local swift_pgid="${swift_pid}"   # setsid 之后 pid == sid == pgid
    echo "[info]  swift eval pid=${swift_pid} (pgid=${swift_pgid})  max_new_tokens=${max_new_tokens}  (tail -f ${log_path})"

    local DONE_MARK='End time of running main'
    local FATAL_RE='Engine core initialization failed|ValueError: Free memory on device|torch\.OutOfMemoryError|CUDA out of memory'
    local GRACE_SEC=30
    local HARD_TIMEOUT=21600   # 单档最多 6 小时（Level 5 需要更长时间）
    local start_ts=$(date +%s)
    local stage_status='timeout'

    while kill -0 "${swift_pid}" 2>/dev/null; do
        if [ -f "${log_path}" ]; then
            if grep -q "${DONE_MARK}" "${log_path}"; then
                echo "[info]  stage done in log, wait ${GRACE_SEC}s for clean exit..."
                for _ in $(seq 1 ${GRACE_SEC}); do
                    kill -0 "${swift_pid}" 2>/dev/null || break
                    sleep 1
                done
                stage_status='done'
                break
            fi
            if grep -qE "${FATAL_RE}" "${log_path}"; then
                echo "[warn]  fatal error detected in ${log_path##*/} — aborting stage ${max_new_tokens}"
                stage_status='crashed'
                break
            fi
        fi
        local now_ts=$(date +%s)
        if [ $((now_ts - start_ts)) -gt ${HARD_TIMEOUT} ]; then
            echo "[warn]  hard timeout ${HARD_TIMEOUT}s reached"
            break
        fi
        sleep 5
    done

    if kill -0 -- "-${swift_pgid}" 2>/dev/null; then
        echo "[info]  SIGTERM process group ${swift_pgid}"
        kill -TERM -- "-${swift_pgid}" 2>/dev/null || true
        local i
        for i in $(seq 1 5); do
            kill -0 -- "-${swift_pgid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 -- "-${swift_pgid}" 2>/dev/null; then
            echo "[warn]  SIGKILL process group ${swift_pgid}"
            kill -KILL -- "-${swift_pgid}" 2>/dev/null || true
        fi
    fi
    _kill_run_residuals
    wait "${swift_pid}" 2>/dev/null || true
    _wait_gpu_clean 30
    echo "[info]  stage max_new_tokens=${max_new_tokens} exited (status=${stage_status})"
}

# ------------------------------------------------------------------
# 聚合：从 reviews/<model_name>/<dataset>.jsonl 提 metric 写 stats.tsv
# ------------------------------------------------------------------
extract_metrics() {
    local max_new_tokens="$1"
    local out_dir="$2"
    python3 - "${MODEL_PATH}" "${max_new_tokens}" "${out_dir}" "${PASS_K}" "${STATS_TSV}" <<'PY'
import glob, json, os, sys
from collections import defaultdict

model_path, max_new_tokens, out_dir, pass_k, stats_tsv = sys.argv[1:]

# 查找所有 JSONL review 文件
jsonl_files = sorted(glob.glob(
    os.path.join(out_dir, 'native', '*', 'reviews', '*', '*.jsonl')
))

if not jsonl_files:
    print(f'[warn]  no jsonl report found under {out_dir}')
    sys.exit(0)

# 按数据集名称聚合结果
dataset_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'group_scores': defaultdict(list)})

for jsonl_path in jsonl_files:
    # 从文件名提取数据集名称，例如 "math_500_Level 2.jsonl" -> "math_500_Level 2"
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # 获取 group_id (来自同一个问题的不同重复)
                group_id = record.get('group_id', 0)
                sample_score = record.get('sample_score', {})
                score = sample_score.get('score', {})
                value = score.get('value', {})
                acc = value.get('acc', 0)

                dataset_results[base_name]['total'] += 1
                if acc > 0:
                    dataset_results[base_name]['correct'] += 1
                dataset_results[base_name]['group_scores'][group_id].append(acc)
            except (json.JSONDecodeError, KeyError) as e:
                continue

with open(stats_tsv, 'a', encoding='utf-8') as out:
    for ds_name, results in sorted(dataset_results.items()):
        total = results['total']
        correct = results['correct']
        mean_acc = correct / total if total > 0 else 0

        # 写入 mean_acc
        out.write(f'{model_path}\t{max_new_tokens}\t{ds_name}\tmean_acc\t{mean_acc:.4f}\n')
        print(f'  [{max_new_tokens}] {ds_name} mean_acc = {mean_acc:.4f}')

        # 计算 pass@k (只要该 group 有一个正确就算正确)
        pass_k = int(sys.argv[4])
        group_correct = 0
        group_total = 0
        for group_id, scores in results['group_scores'].items():
            group_total += 1
            if any(s > 0 for s in scores):
                group_correct += 1

        pass_at_k = group_correct / group_total if group_total > 0 else 0
        out.write(f'{model_path}\t{max_new_tokens}\t{ds_name}\tmean_acc_pass@{pass_k}\t{pass_at_k:.4f}\n')
        print(f'  [{max_new_tokens}] {ds_name} mean_acc_pass@{pass_k} = {pass_at_k:.4f}')
PY
}

# ------------------------------------------------------------------
# 主循环
# ------------------------------------------------------------------
FAILED=()
for L in "${MAX_NEW_TOKENS_SWEEP[@]}"; do
    echo ""
    echo "=========================================="
    echo "Sweep step: max_new_tokens=${L}"
    echo "=========================================="

    OUT_DIR="${EVAL_ROOT_DIR}/max_new_tokens_${L}"
    LOG_PATH="${EVAL_ROOT_DIR}/logs/stage_${L}.log"
    run_one_eval "${L}" "${OUT_DIR}" "${LOG_PATH}"

    if [ -d "${OUT_DIR}/native" ]; then
        find "${OUT_DIR}/native" -type d -name 'predictions' -exec rm -rf {} + 2>/dev/null || true
    fi

    if ! extract_metrics "${L}" "${OUT_DIR}"; then
        echo "[warn]  extract_metrics failed for ${L}"
        FAILED+=("${L}")
    fi
done

echo ""
echo "=========================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Done with failures at: ${FAILED[*]}"
else
    echo "Done."
fi
echo "Stats: ${STATS_TSV}"
cat "${STATS_TSV}"
echo "=========================================="
