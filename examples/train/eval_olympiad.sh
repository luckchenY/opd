#!/bin/bash
set -uo pipefail

# OlympiadBench（evalscope: olympiad_bench）
# 纯文本 LLM 默认只测 OE_TO 数学子集（可自动打分；不含 MM 图片 / TP 证明）
MODEL_PATH=${MODEL_PATH:-/data/chenyang/models/nomerge_segment}
#/data/chenyang/checkpoint/JustRL/global_step_200/merged_hf
#/data/chenyang/OPD/checkpoint/Justrl-consistency/global_step_200/merged_hf

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}

PASS_K=${PASS_K:-4}
TEMPERATURE=${TEMPERATURE:-1}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-50}
MAX_NEW_TOKENS_SWEEP=(${MAX_NEW_TOKENS_SWEEP:-8192})

DATA_DIR=${DATA_DIR:-/data/chenyang/opdswift/data/eval_datasets/olympiad_bench}

# 逗号分隔；默认 OE_TO 数学三子集（共 ~2322 题）
# 只要英文竞赛：SUBSET_LIST=OE_TO_maths_en_COMP
SUBSET_LIST=${SUBSET_LIST:-OE_TO_maths_en_COMP,OE_TO_maths_zh_CEE,OE_TO_maths_zh_COMP}

# Qwen3 系列
MODEL_TYPE=${MODEL_TYPE:-qwen2}
TEMPLATE=${TEMPLATE:-deepseek_r1}
ENABLE_THINKING=${ENABLE_THINKING:-true}

# ------------------------------------------------------------------
# 自动检测 GPU 列表
# ------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_LIST="${CUDA_VISIBLE_DEVICES}"
else
    GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
fi
if [ -z "${GPU_LIST}" ]; then
    echo "[error] 没有发现可用的 GPU"; exit 1
fi
GPU_IDS_ARRAY=( $(echo "${GPU_LIST}" | tr ',' ' ') )
NUM_GPUS=${#GPU_IDS_ARRAY[@]}
GPU_LIST=$(IFS=,; echo "${GPU_IDS_ARRAY[*]}")

VLLM_DP_SIZE=${NUM_GPUS}
VLLM_TP_SIZE=1
EVAL_NUM_PROC=$((16 * NUM_GPUS))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME_SAFE=$(basename "${MODEL_PATH}" | tr '/' '__')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT_DIR="${SCRIPT_DIR}/test_result/${MODEL_NAME_SAFE}/olympiad_${TIMESTAMP}"
STATS_TSV="${EVAL_ROOT_DIR}/stats.tsv"

CURRENT_SWIFT_PGID=""

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# 把 SUBSET_LIST 转成 JSON 数组字符串
export SUBSET_LIST
SUBSET_JSON=$(python3 - <<'PY'
import json, os
xs = [s.strip() for s in os.environ["SUBSET_LIST"].split(",") if s.strip()]
print(json.dumps(xs, ensure_ascii=False))
PY
)

if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
    echo "[WARN] OlympiadBench 数据缺失: ${DATA_DIR}" >&2
    echo "       请先跑: python tools/download_olympiad_bench.py" >&2
fi

mkdir -p "${EVAL_ROOT_DIR}"
{
  echo -e "# model_path\t${MODEL_PATH}"
  echo -e "# gpus\t${GPU_LIST}\tnum_gpus\t${NUM_GPUS}"
  echo -e "# vllm_parallel\tDP=${VLLM_DP_SIZE}\tTP=${VLLM_TP_SIZE}"
  echo -e "# repeats\t${PASS_K}\ttemperature\t${TEMPERATURE}\ttop_p\t${TOP_P}"
  echo -e "# bench\tolympiad_bench"
  echo -e "# subsets\t${SUBSET_LIST}"
  echo -e "# enable_thinking\t${ENABLE_THINKING}"
  echo -e "# strategy\tsweep_max_new_tokens"
  echo -e "model_path\tmax_new_tokens\tdataset\tmetric\tvalue"
} > "${STATS_TSV}"

PORT=$((8000 + GPU_IDS_ARRAY[0]))

echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "Type/Tpl:   ${MODEL_TYPE} / ${TEMPLATE}  thinking=${ENABLE_THINKING}"
echo "Dataset:    olympiad_bench  (${DATA_DIR})"
echo "Subsets:    ${SUBSET_LIST}"
echo "GPUs:       ${GPU_LIST}        (DP=${VLLM_DP_SIZE}, TP=${VLLM_TP_SIZE})"
echo "Port:       ${PORT}"
echo "Eval proc:  ${EVAL_NUM_PROC}"
echo "Sweep:      ${MAX_NEW_TOKENS_SWEEP[*]}"
echo "Stats:      ${STATS_TSV}"
echo "=========================================="

_kill_current_stage() {
    [ -z "${CURRENT_SWIFT_PGID}" ] && return 0
    local pgid="${CURRENT_SWIFT_PGID}"
    echo "[info]  sending SIGTERM to process group ${pgid}"
    kill -TERM -"${pgid}" 2>/dev/null || true
    local i
    for i in $(seq 1 8); do
        kill -0 -"${pgid}" 2>/dev/null || { CURRENT_SWIFT_PGID=""; return 0; }
        sleep 1
    done
    echo "[warn]  SIGKILL process group ${pgid}"
    kill -KILL -"${pgid}" 2>/dev/null || true
    sleep 2
    CURRENT_SWIFT_PGID=""
}

_wait_port_free() {
    local wait_sec="${1:-60}"
    local i
    for i in $(seq 1 "${wait_sec}"); do
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$" || return 0
        sleep 1
    done
    echo "[warn]  port ${PORT} still busy after ${wait_sec}s"
}

_on_exit() {
    echo "[info]  cleanup on exit"
    _kill_current_stage 2>/dev/null || true
}
_on_signal() {
    local sig="$1"
    echo "[info]  got ${sig}, aborting"
    _kill_current_stage 2>/dev/null || true
    trap - EXIT INT TERM
    exit 130
}
trap '_on_exit' EXIT
trap '_on_signal INT'  INT
trap '_on_signal TERM' TERM

run_one_eval() {
    local max_new_tokens="$1"
    local out_dir="$2"
    local log_path="$3"

    mkdir -p "$(dirname "${log_path}")"

    _kill_current_stage
    _wait_port_free 60

    setsid env \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift eval \
        --model "${MODEL_PATH}" \
        --port "${PORT}" \
        --model_type "${MODEL_TYPE}" \
        --template "${TEMPLATE}" \
        --enable_thinking "${ENABLE_THINKING}" \
        --eval_backend Native \
        --eval_dataset olympiad_bench \
        --eval_dataset_args "{
            \"olympiad_bench\": {
                \"local_path\": \"${DATA_DIR}\",
                \"aggregation\": \"mean_and_pass_at_k\",
                \"subset_list\": ${SUBSET_JSON}
            }
        }" \
        --extra_eval_args "{\"repeats\": ${PASS_K}}" \
        --infer_backend vllm \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
        --vllm_data_parallel_size "${VLLM_DP_SIZE}" \
        --eval_num_proc "${EVAL_NUM_PROC}" \
        --max_new_tokens "${max_new_tokens}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --attn_impl flash_attn \
        --torch_dtype bfloat16 \
        --stream false \
        --eval_output_dir "${out_dir}" \
        >"${log_path}" 2>&1 &
    local swift_pid=$!
    CURRENT_SWIFT_PGID=${swift_pid}
    echo "[info]  swift eval pid/pgid=${swift_pid}  max_new_tokens=${max_new_tokens}  (tail -f ${log_path})"

    local DONE_MARK='End time of running main'
    local FAIL_MARK_1='Free memory on device cuda:'
    local FAIL_MARK_2='EngineCore failed to start.'
    local FAIL_MARK_3='Engine core initialization failed.'
    local GRACE_SEC=30
    # OlympiadBench 题量大，单档最多 24h
    local HARD_TIMEOUT=${HARD_TIMEOUT:-86400}
    local STALL_TIMEOUT=${STALL_TIMEOUT:-600}
    local start_ts
    start_ts=$(date +%s)
    local hang=0

    while kill -0 "${swift_pid}" 2>/dev/null; do
        if [ -f "${log_path}" ] && tail -c 65536 "${log_path}" 2>/dev/null | grep -qF "${DONE_MARK}"; then
            echo "[info]  stage done in log, wait ${GRACE_SEC}s for clean exit..."
            for _ in $(seq 1 ${GRACE_SEC}); do
                kill -0 "${swift_pid}" 2>/dev/null || break
                sleep 1
            done
            break
        fi
        if [ -f "${log_path}" ] && tail -c 65536 "${log_path}" 2>/dev/null | grep -qF "${FAIL_MARK_1}"; then
            echo "[warn]  detected vLLM insufficient free memory; aborting this stage early"
            hang=1
            break
        fi
        if [ -f "${log_path}" ] && tail -c 65536 "${log_path}" 2>/dev/null | grep -qF "${FAIL_MARK_2}"; then
            echo "[warn]  detected vLLM EngineCore startup failure; aborting this stage early"
            hang=1
            break
        fi
        if [ -f "${log_path}" ] && tail -c 65536 "${log_path}" 2>/dev/null | grep -qF "${FAIL_MARK_3}"; then
            echo "[warn]  detected vLLM engine initialization failure; aborting this stage early"
            hang=1
            break
        fi

        local now_ts
        now_ts=$(date +%s)
        if [ -f "${log_path}" ]; then
            local log_mtime idle
            log_mtime=$(stat -c %Y "${log_path}" 2>/dev/null || echo "${now_ts}")
            idle=$(( now_ts - log_mtime ))
            if [ "${idle}" -gt "${STALL_TIMEOUT}" ]; then
                echo "[warn]  log stall ${idle}s (>${STALL_TIMEOUT}s), treating as hang"
                hang=1
                break
            fi
        fi
        if [ $((now_ts - start_ts)) -gt ${HARD_TIMEOUT} ]; then
            echo "[warn]  hard timeout ${HARD_TIMEOUT}s reached"
            hang=1
            break
        fi
        sleep 5
    done

    _kill_current_stage
    _wait_port_free 120
    echo "[info]  stage max_new_tokens=${max_new_tokens} exited (hang=${hang})"
}

extract_metrics() {
    local max_new_tokens="$1"
    local out_dir="$2"
    python3 - "${MODEL_PATH}" "${max_new_tokens}" "${out_dir}" "${PASS_K}" "${STATS_TSV}" <<'PY'
import glob, json, os, sys

model_path, max_new_tokens, out_dir, pass_k, stats_tsv = sys.argv[1:]
report_files = sorted(glob.glob(
    os.path.join(out_dir, 'native', '*', 'reports', '*', '*.json')
))
if not report_files:
    print(f'[warn]  no report found under {out_dir}')
    sys.exit(0)

wanted = ['mean_acc', f'mean_acc_pass@{pass_k}', 'acc']

def write_row(out, ds, name, value):
    out.write(f'{model_path}\t{max_new_tokens}\t{ds}\t{name}\t{value:.4f}\n')
    print(f'  [{max_new_tokens}] {ds} {name} = {value:.4f}')

with open(stats_tsv, 'a', encoding='utf-8') as out:
    for rp in report_files:
        with open(rp) as f:
            d = json.load(f)
        ds = d.get('dataset_name') or os.path.splitext(os.path.basename(rp))[0]
        metrics_by_name = {m['name']: m for m in d.get('metrics', [])}
        for name in wanted:
            m = metrics_by_name.get(name)
            if m is None:
                continue
            write_row(out, ds, name, float(m['score']))
            for cat in m.get('categories') or []:
                for sub in cat.get('subsets') or []:
                    sub_name = sub.get('name')
                    sub_score = sub.get('score')
                    if sub_name is None or sub_score is None:
                        continue
                    write_row(out, f'{ds}/{sub_name}', name, float(sub_score))
PY
}

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
