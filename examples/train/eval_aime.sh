#!/bin/bash
set -uo pipefail

# ========================================
# 每次只需要改这两个参数
# ========================================
MODEL_PATH=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/output_gkd_softor/v0-20260501-142325/v0-20260501-142406/checkpoint-469
GPU_ID=0
# ========================================

PASS_K=8
TEMPERATURE=1
TOP_P=0.8
# 每档分别做一次独立的 swift eval，跑完立即拿到该长度的官方指标（不做截断）
MAX_NEW_TOKENS_SWEEP=(4096  8192 16384 32768)

DATA_DIR=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/data/aime_datasets

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_ROOT_DIR="${MODEL_PATH}/eval_results/${TIMESTAMP}"
STATS_TSV="${EVAL_ROOT_DIR}/stats.tsv"

# 唯一的 run id：写进子进程环境变量，用于识别"哪些进程是本次 bash 启动的"。
# 即使同一张 GPU 上同时跑其他人的 swift/vLLM，也不会被误杀。
EVAL_AIME_RUN_ID="eval_aime_$$_$(date +%s)_${GPU_ID}"
export EVAL_AIME_RUN_ID

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "${EVAL_ROOT_DIR}"
{
  echo -e "# model_path\t${MODEL_PATH}"
  echo -e "# gpu\t${GPU_ID}"
  echo -e "# repeats\t${PASS_K}\ttemperature\t${TEMPERATURE}\ttop_p\t${TOP_P}"
  echo -e "# strategy\tsweep_max_new_tokens"
  echo -e "model_path\tmax_new_tokens\tdataset\tmetric\tvalue"
} > "${STATS_TSV}"

PORT=$((8000 + GPU_ID))

echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "GPU:        ${GPU_ID}        Port: ${PORT}"
echo "Sweep:      ${MAX_NEW_TOKENS_SWEEP[*]}"
echo "Stats:      ${STATS_TSV}"
echo "Run ID:     ${EVAL_AIME_RUN_ID}"
echo "=========================================="

# 兜底：脚本被 Ctrl+C / kill / 正常退出时，清掉本次 run 的所有残留（孤儿 EngineCore 也在内）
# EXIT 只做清理；INT/TERM 先清理再立即退出（避免主循环继续跑 / 反复触发 handler）
_on_exit() {
    echo "[info]  cleanup on exit: run ${EVAL_AIME_RUN_ID}"
    _kill_run_residuals 2>/dev/null || true
}
_on_signal() {
    local sig="$1"
    echo "[info]  got ${sig}, aborting run ${EVAL_AIME_RUN_ID}"
    _kill_run_residuals 2>/dev/null || true
    trap - EXIT INT TERM
    exit 130
}
trap '_on_exit' EXIT
trap '_on_signal INT'  INT
trap '_on_signal TERM' TERM

# ------------------------------------------------------------------
# 三重判定识别"本次脚本产生的进程"（包括被 PID 1 收养的孤儿 VLLM::EngineCore）：
#   A) 进程环境变量里有 EVAL_AIME_RUN_ID=${EVAL_AIME_RUN_ID} （正常子进程）
#   B) pid 当前出现在 ${GPU_ID} 卡上 且 不在 GPU_BASELINE_PIDS 里（本 run 启动前已有的 pid）
#      —— 用来抓环境被清空的孤儿 EngineCore
# 两者取并集。别人 run 启动的进程不会被杀（不满足 A；若出现在 B，说明是本 run 启动后新增的在本卡上跑的进程，
# 同卡别人的进程不会被误杀 —— 因为启动前就在 baseline 里了）。
# ------------------------------------------------------------------

# 快照本卡上"本 run 启动前就存在的 pid 集合"。以后所有"GPU 新增 pid"都视为本 run 产物。
# 同时用 lsof 快照持有 /dev/nvidia${GPU_ID} fd 的进程——这能抓到孤儿 EngineCore 等
# nvidia-smi --query-compute-apps 报不出来但实际持有 fd 的进程。
_snapshot_gpu_baseline() {
    local smi_pids lsof_pids
    smi_pids=$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
        --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    lsof_pids=$(lsof "/dev/nvidia${GPU_ID}" 2>/dev/null | awk 'NR>1{print $2}' | sort -u)
    GPU_BASELINE_PIDS=$(printf '%s\n%s\n' "${smi_pids}" "${lsof_pids}" | sort -u)
    echo "[info]  GPU ${GPU_ID} baseline pids at start: ${GPU_BASELINE_PIDS//$'\n'/ } (these will NEVER be killed)"
}

_pid_env_has_run_id() {
    local pid="$1"
    local env_file="/proc/${pid}/environ"
    [ -r "${env_file}" ] || return 1
    # 过滤 bash 子 shell / 常见小工具：它们会继承 EVAL_AIME_RUN_ID（来自
    # $(...) 命令替换 / 管道），但不是我们要杀的 swift / vLLM 进程。
    # 不过滤的话，脚本每次扫 /proc 都会把自己的命令替换子 shell 当成
    # "本 run 残留"来杀，导致 _wait_stage_clean 永远不返回 0。
    local comm
    comm=$(cat "/proc/${pid}/comm" 2>/dev/null) || return 1
    case "${comm}" in
        bash|sh|dash|zsh|grep|awk|sed|sort|tr|tail|head|cat|stat|seq|sleep|date|kill|pgrep|pkill|nvidia-smi|ss|ps|tee|find|xargs|wc|readlink|basename|dirname)
            return 1 ;;
    esac
    grep -qzxF "EVAL_AIME_RUN_ID=${EVAL_AIME_RUN_ID}" "${env_file}" 2>/dev/null
}

_pid_in_baseline() {
    local pid="$1"
    [ -z "${GPU_BASELINE_PIDS:-}" ] && return 1
    echo "${GPU_BASELINE_PIDS}" | grep -qxF "${pid}"
}

# GPU 上"新增"的、不在 baseline 里的进程 —— 都是本 run 产物（含孤儿 EngineCore）
# 双通道：
#   1. nvidia-smi --query-compute-apps：抓到正常 CUDA 进程
#   2. lsof /dev/nvidia${GPU_ID}：抓到 nvidia-smi 认不出来但仍持有 fd 的孤儿进程
#      （vLLM EngineCore 被 PID 1 收养后常常只剩 fd，smi 看不到它但它确实占着显存）
_list_gpu_new_pids() {
    local smi_pids lsof_pids all_pids p
    smi_pids=$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
        --format=csv,noheader 2>/dev/null | tr -d ' ')
    lsof_pids=$(lsof "/dev/nvidia${GPU_ID}" 2>/dev/null | awk 'NR>1{print $2}')
    all_pids=$(printf '%s\n%s\n' "${smi_pids}" "${lsof_pids}" | tr -s ' ' '\n' | sort -u)
    for p in ${all_pids}; do
        [ -z "${p}" ] && continue
        # 只报告真实存在的进程（过滤 smi 返回的僵尸 PID）
        [ -d "/proc/${p}" ] || continue
        _pid_in_baseline "${p}" && continue
        echo "${p}"
    done
}

# 普通判定：环境变量有 run id 的进程（非 GPU 进程也算，比如 swift 主进程、evalscope worker）
_list_env_matched_pids() {
    local pid
    for pid in /proc/[0-9]*; do
        pid=${pid##*/}
        [ "${pid}" = "$$" ] && continue
        _pid_env_has_run_id "${pid}" && echo "${pid}"
    done
}

# 合并 A ∪ B 并去重
_list_run_pids() {
    {
        _list_env_matched_pids
        _list_gpu_new_pids
    } | sort -u
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

# 等待：本 run 进程全清 + 端口释放（不要求 GPU 整体显存空，因为可能有别人在同卡跑）
_wait_stage_clean() {
    # 等待本次 run 的残留进程清理、端口释放。
    # 默认等待时间设置更长，避免大模型/vLLM 退出释放显存与端口较慢时误判。
    local wait_sec="${1:-180}"
    local i pids
    for i in $(seq 1 "${wait_sec}"); do
        pids=$(_list_run_pids)
        if [ -z "${pids}" ] && ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$"; then
            return 0
        fi
        sleep 1
    done
    pids=$(_list_run_pids)
    if [ -n "${pids}" ]; then
        echo "[warn]  after ${wait_sec}s, still have residual pids: ${pids//$'\n'/ }; forcing SIGKILL"
        # shellcheck disable=SC2086
        kill -KILL ${pids} 2>/dev/null || true
        sleep 2
    fi
}

# ------------------------------------------------------------------
# 封装：一次 swift eval，防 hang，结束后按 run_id 精确清理。
# 参数：$1=max_new_tokens  $2=eval_output_dir  $3=log_path
# ------------------------------------------------------------------
run_one_eval() {
    local max_new_tokens="$1"
    local out_dir="$2"
    local log_path="$3"

    mkdir -p "$(dirname "${log_path}")"

    # 进场前：清掉上一 stage 的残留（只限本 run）
    _kill_run_residuals
    _wait_stage_clean 180

    # EVAL_AIME_RUN_ID 已 export，子进程自动继承；这里再显式传 CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift eval \
        --model "${MODEL_PATH}" \
        --port "${PORT}" \
        --eval_backend Native \
        --eval_dataset aime24 aime25 aime26 \
        --eval_dataset_args "{
            \"aime24\": {\"local_path\": \"${DATA_DIR}/aime24_local\", \"aggregation\": \"mean_and_pass_at_k\"},
            \"aime25\": {\"local_path\": \"${DATA_DIR}/aime25_local\", \"aggregation\": \"mean_and_pass_at_k\"},
            \"aime26\": {\"local_path\": \"${DATA_DIR}/aime26_local\", \"aggregation\": \"mean_and_pass_at_k\"}
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
    echo "[info]  swift eval pid=${swift_pid}  max_new_tokens=${max_new_tokens}  (tail -f ${log_path})"

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

    # 命中 hang 时，主动先 KILL 主 pid，避免 TERM 不动或 D 状态把后面的 wait 挂死
    if [ "${hang}" = "1" ] && kill -0 "${swift_pid}" 2>/dev/null; then
        kill -KILL "${swift_pid}" 2>/dev/null || true
    fi

    # 收尾：杀掉本次 run 的所有残留（孤儿 EngineCore 也在内）
    _kill_run_residuals

    # wait 加超时保险：后台等最多 30s，防止 D 状态把脚本钉死
    ( wait "${swift_pid}" 2>/dev/null ) &
    local wpid=$!
    for _ in $(seq 1 30); do
        kill -0 "${wpid}" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "${wpid}" 2>/dev/null || true

    _wait_stage_clean 120
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
# 先快照本卡上已经存在的 GPU 进程（别人开的任务）。这些 pid 永远不会被杀。
_snapshot_gpu_baseline

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
