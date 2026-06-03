#!/bin/bash
set -uo pipefail

# ========================================
# 每次只需要改这一个参数（GPU 自动检测）
# ========================================
# 本地 checkpoint 路径，或 HuggingFace/ModelScope 模型名（base 模型直接填名字）
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
MODEL_PATH=/home/ubuntu/chenyang/opd/tools/consistency_opd_output/checkpoints/epoch_002

# GPU 选择：
#   1) 如果调用方设置了 CUDA_VISIBLE_DEVICES，就用它指定的卡（脚本不会改写它）
#   2) 否则用本机 nvidia-smi 看到的全部卡
# 多卡时采用 vLLM 数据并行（DP）：每卡一份完整模型副本，请求做负载均衡
# 单机评测里这是最快的并行方式（无 TP 通信开销，吞吐随卡数近线性增长）。
# ========================================

PASS_K=1
TEMPERATURE=1
TOP_P=0.95
TOP_K=50

# 每档分别做一次独立的 swift eval，跑完立即拿到该长度的官方指标（不做截断）
MAX_NEW_TOKENS_SWEEP=(16384)

DATA_DIR=/home/ubuntu/chenyang/opd/data/aime_datasets
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
# 解析为数组，兼容 "0,1,2" / "0 1 2" / 带空格 等情况
GPU_IDS_ARRAY=( $(echo "${GPU_LIST}" | tr ',' ' ') )
NUM_GPUS=${#GPU_IDS_ARRAY[@]}
# 重新规范化 GPU_LIST（去空格）
GPU_LIST=$(IFS=,; echo "${GPU_IDS_ARRAY[*]}")

# 数据并行 = 卡数；张量并行固定 1（模型够小，DP 吞吐最高）
VLLM_DP_SIZE=${NUM_GPUS}
VLLM_TP_SIZE=1
# 评测客户端并发也按卡数放大，确保所有 DP worker 都被喂饱
EVAL_NUM_PROC=$((16 * NUM_GPUS))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 将模型名中的 / 替换为 __，用作目录名（兼容本地路径和 HF 模型名）
MODEL_NAME_SAFE=$(basename "${MODEL_PATH}" | tr '/' '__')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT_DIR="${SCRIPT_DIR}/test_result/${MODEL_NAME_SAFE}/${TIMESTAMP}"
STATS_TSV="${EVAL_ROOT_DIR}/stats.tsv"

# 当前正在跑的 swift eval 进程组 ID（= setsid 后的 swift_pid）；空表示无
CURRENT_SWIFT_PGID=""

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "${EVAL_ROOT_DIR}"
{
  echo -e "# model_path\t${MODEL_PATH}"
  echo -e "# gpus\t${GPU_LIST}\tnum_gpus\t${NUM_GPUS}"
  echo -e "# vllm_parallel\tDP=${VLLM_DP_SIZE}\tTP=${VLLM_TP_SIZE}"
  echo -e "# repeats\t${PASS_K}\ttemperature\t${TEMPERATURE}\ttop_p\t${TOP_P}"
  echo -e "# strategy\tsweep_max_new_tokens"
  echo -e "model_path\tmax_new_tokens\tdataset\tmetric\tvalue"
} > "${STATS_TSV}"

# 端口随首张卡 id 偏移，避免不同 run 端口冲突
PORT=$((8000 + GPU_IDS_ARRAY[0]))

echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "GPUs:       ${GPU_LIST}        (DP=${VLLM_DP_SIZE}, TP=${VLLM_TP_SIZE})"
echo "Port:       ${PORT}"
echo "Eval proc:  ${EVAL_NUM_PROC}"
echo "Sweep:      ${MAX_NEW_TOKENS_SWEEP[*]}"
echo "Stats:      ${STATS_TSV}"
echo "=========================================="

# ------------------------------------------------------------------
# 进程组清理：用 setsid 让 swift eval 自成一个新 session（PGID = swift_pid）。
# 子孙进程（包括 vLLM EngineCore）继承 PGID；即使被 PID 1 收养，PGID 也不变。
# kill -TERM/-KILL -$PGID 直接由内核投递给整组，无需任何 /proc 扫描。
# ------------------------------------------------------------------

# 杀掉当前 stage 的整个进程组，并等待其退出
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

# 等端口释放（vLLM 退出后端口可能晚一两秒才关）
_wait_port_free() {
    local wait_sec="${1:-60}"
    local i
    for i in $(seq 1 "${wait_sec}"); do
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$" || return 0
        sleep 1
    done
    echo "[warn]  port ${PORT} still busy after ${wait_sec}s"
}

# 兜底：脚本被 Ctrl+C / kill / 正常退出时，清掉当前 stage 的进程组
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

# ------------------------------------------------------------------
# 封装：一次 swift eval，防 hang，结束后按 run_id 精确清理。
# 参数：$1=max_new_tokens  $2=eval_output_dir  $3=log_path
# ------------------------------------------------------------------
run_one_eval() {
    local max_new_tokens="$1"
    local out_dir="$2"
    local log_path="$3"

    mkdir -p "$(dirname "${log_path}")"

    # 进场前：确保上一 stage 的进程组已经退出
    _kill_current_stage
    _wait_port_free 60

    # setsid：让 swift eval 在全新 session 里运行，自动成为新进程组的 leader（PGID = PID）。
    # 子孙进程（含 vLLM EngineCore）继承该 PGID；即使被 PID 1 收养，PGID 也不变。
    # 后续只需 kill -TERM/-KILL -$PGID 即可整组清理，无需任何 /proc 扫描。
    setsid env \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift eval \
        --model "${MODEL_PATH}" \
        --port "${PORT}" \
        --model_type qwen3 \
        --template qwen3 \
        --eval_backend Native \
        --eval_dataset aime24 aime25 \
        --eval_dataset_args "{
            \"aime24\": {\"local_path\": \"${DATA_DIR}/aime24_local\", \"aggregation\": \"mean_and_pass_at_k\"},
            \"aime25\": {\"local_path\": \"${DATA_DIR}/aime25_local\", \"aggregation\": \"mean_and_pass_at_k\"},
            \"aime26\": {\"local_path\": \"${DATA_DIR}/aime26_local\", \"aggregation\": \"mean_and_pass_at_k\"}
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
        --enable_thinking false \
        --attn_impl flash_attn \
        --torch_dtype bfloat16 \
        --stream false \
        --eval_output_dir "${out_dir}" \
        >"${log_path}" 2>&1 &
    local swift_pid=$!
    CURRENT_SWIFT_PGID=${swift_pid}   # setsid 后 PID = PGID，整组用这个 id 清理
    echo "[info]  swift eval pid/pgid=${swift_pid}  max_new_tokens=${max_new_tokens}  (tail -f ${log_path})"

    local DONE_MARK='End time of running main'
    # vLLM / EngineCore 启动失败时通常会很快写出明确错误；提前识别可避免卡在 STALL_TIMEOUT 才被判 hang。
    local FAIL_MARK_1='Free memory on device cuda:'
    local FAIL_MARK_2='EngineCore failed to start.'
    local FAIL_MARK_3='Engine core initialization failed.'
    local GRACE_SEC=30
    local HARD_TIMEOUT=10800   # 整档最多跑 3 小时（兜底）
    local STALL_TIMEOUT=300    # 日志 5 分钟没更新就判定为 hang
    local start_ts
    start_ts=$(date +%s)
    local hang=0

    while kill -0 "${swift_pid}" 2>/dev/null; do
        # 1) 正常完成：只扫日志尾部，避免巨大文件每 5s 全量 grep
        if [ -f "${log_path}" ] && tail -c 65536 "${log_path}" 2>/dev/null | grep -qF "${DONE_MARK}"; then
            echo "[info]  stage done in log, wait ${GRACE_SEC}s for clean exit..."
            for _ in $(seq 1 ${GRACE_SEC}); do
                kill -0 "${swift_pid}" 2>/dev/null || break
                sleep 1
            done
            break
        fi

        # 1.5) 明确失败：vLLM 显存不足 / EngineCore 起不来时，直接结束本 stage（避免等到 stall 才 kill）
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

        # 2) 日志 stall：mtime 超过 STALL_TIMEOUT 没变就当卡死
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

        # 3) 硬超时兜底
        if [ $((now_ts - start_ts)) -gt ${HARD_TIMEOUT} ]; then
            echo "[warn]  hard timeout ${HARD_TIMEOUT}s reached"
            hang=1
            break
        fi
        sleep 5
    done

    # 收尾：杀掉整个进程组（含孤儿 EngineCore），然后等端口释放
    _kill_current_stage
    _wait_port_free 120
    echo "[info]  stage max_new_tokens=${max_new_tokens} exited"
}

# ------------------------------------------------------------------
# 聚合：从单档的 reports/<model_name>/<dataset>.json 提 metric 写 stats.tsv
# ------------------------------------------------------------------
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

with open(stats_tsv, 'a', encoding='utf-8') as out:
    for rp in report_files:
        with open(rp) as f:
            d = json.load(f)
        ds = d.get('dataset_name') or os.path.splitext(os.path.basename(rp))[0]
        metrics = {m['name']: m['score'] for m in d.get('metrics', [])}
        wanted = ['mean_acc', f'mean_acc_pass@{pass_k}']
        for name in wanted:
            v = metrics.get(name)
            if v is None:
                continue
            out.write(f'{model_path}\t{max_new_tokens}\t{ds}\t{name}\t{v:.4f}\n')
            print(f'  [{max_new_tokens}] {ds} {name} = {v:.4f}')
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

    # 清理巨大的 predictions 目录（reviews 保留方便排查；想省空间也可一并删）
    if [ -d "${OUT_DIR}/native" ]; then
        find "${OUT_DIR}/native" -type d -name 'predictions' -exec rm -rf {} + 2>/dev/null || true
    fi

    # 提本档指标到 stats.tsv
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
