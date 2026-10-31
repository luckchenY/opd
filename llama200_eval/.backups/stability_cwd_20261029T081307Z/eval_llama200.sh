#!/usr/bin/env bash
# Llama200 三项评测共用入口；每次新建结果目录，不自动复用旧缓存。
set -euo pipefail
set +m # 后台子进程继承当前进程组，setsid 不会额外 fork；$! 就是新 PGID。

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$(cd -- "$SCRIPT_DIR/.." && pwd)"}
# 路径只在这里配置；support.py 预检和 Swift 评测共用这些值。
PYTHON=${PYTHON:-/home/chenyang2/miniconda3/envs/swift/bin/python}
MODEL_PATH=${MODEL_PATH:-/data/chenyang/models/Llama200}
export AIME_DATA_DIR=${AIME_DATA_DIR:-"$PROJECT_DIR/data/data/aime_datasets"}
export OLYMPIAD_DATA_DIR=${OLYMPIAD_DATA_DIR:-"$PROJECT_DIR/data/eval_datasets/olympiad_bench"}
export GPQA_DATA_DIR=${GPQA_DATA_DIR:-"$PROJECT_DIR/data/data/gpqa_datasets"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,4,5,7}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="$(dirname -- "$PYTHON"):$PATH"
export VLLM_DENSE_DP_LB_FIX=${VLLM_DENSE_DP_LB_FIX:-auto}
PORT=${PORT:-18191}
HARD_TIMEOUT=${HARD_TIMEOUT:-172800} # 每项最多 48 小时。
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/test_results/$(date +%Y%m%d_%H%M%S)_$$"}

usage() {
    cat <<'HELP'
用法：bash eval_llama200.sh [all|aime|olympiad|gpqa] [--dry-run]
  all        依次运行 AIME24/25 → Olympiad 三个数学子集 → GPQA-Diamond。
  --dry-run  只检查模型、tokenizer、数据并显示命令，不启动评测。
路径配置：PROJECT_DIR、PYTHON、MODEL_PATH、AIME_DATA_DIR、OLYMPIAD_DATA_DIR、GPQA_DATA_DIR、OUTPUT_DIR。
运行配置：CUDA_VISIBLE_DEVICES、PORT、HARD_TIMEOUT。
输出：每项的 swift.log、command.sh、native/ 官方报告、verification.json。
HELP
}

STAGE=all
DRY_RUN=0
SELECTION_GIVEN=0
for arg in "$@"; do
    case "$arg" in
        all|aime|olympiad|gpqa)
            (( SELECTION_GIVEN == 0 )) || { usage >&2; exit 2; }
            STAGE=$arg; SELECTION_GIVEN=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ "$PORT" =~ ^[0-9]+$ && "$HARD_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || exit 2
(( PORT > 0 && PORT < 65536 )) || exit 2
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo 'CUDA_VISIBLE_DEVICES 必须是逗号分隔的 GPU 编号。' >&2; exit 2;
}
IFS=, read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
DP=${#GPU_IDS[@]}
[[ $(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l) -eq $DP ]] || {
    echo 'GPU 编号不能重复。' >&2; exit 2;
}
for required in "$PYTHON" swift setsid ss flock; do
    command -v "$required" >/dev/null || { echo "缺少依赖：$required" >&2; exit 1; }
done
exec 9>"$SCRIPT_DIR/.run.lock"
flock -n 9 || { echo '该入口已有评测在运行。' >&2; exit 1; }
MODEL_VIEW=$("$PYTHON" "$SCRIPT_DIR/support.py" prepare \
    --model "$MODEL_PATH" --view "$SCRIPT_DIR/.compat_model/Llama200" --stage "$STAGE")
echo "模型：$MODEL_PATH"
echo "结果目录：$OUTPUT_DIR"

# all 的阶段间只在 CPU 核验成功后继续；失败保留日志与所有已生成文件。
STAGES=("$STAGE")
[[ "$STAGE" != all ]] || STAGES=(aime olympiad gpqa)
PGID=''
cleanup() {
    if [[ -n "$PGID" ]]; then
        kill -TERM -- "-$PGID" 2>/dev/null || true
        for ((i=0; i<8; i++)); do
            kill -0 -- "-$PGID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$PGID" 2>/dev/null || true
        PGID=''
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_free() {
    local listeners
    listeners=$(ss -H -ltn "sport = :$PORT") || return 1
    [[ -z "$listeners" ]]
}
wait_port() {
    for ((i=0; i<60; i++)); do port_free && return 0; sleep 1; done
    echo "端口 $PORT 被占用；请检查已有服务或设置其他 PORT。" >&2
    return 1
}

if (( ! DRY_RUN )); then
    # mkdir 不覆盖既有目录，防止把两次运行或旧缓存混在一起。
    mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
    mkdir -- "$OUTPUT_DIR"
    OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
fi

run_stage() {
    local stage=$1 temperature=0.6 max_tokens=8192 client=16 max_seqs=32
    local dataset_args out="$OUTPUT_DIR/$1" log="$OUTPUT_DIR/$1/swift.log"
    local -a datasets
    case "$stage" in
        aime)
            datasets=(aime24 aime25) ;;
        olympiad)
            datasets=(olympiad_bench); temperature=1; client=256; max_seqs=16 ;;
        gpqa)
            datasets=(gpqa_diamond); max_tokens=32768 ;;
    esac
    dataset_args=$("$PYTHON" "$SCRIPT_DIR/support.py" dataset-args --stage "$stage")
    # 原评分和采样协议不变。仅用 vllm_max_model_len，避免 Swift 改写原 RoPE。
    local -a command=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY
        -u ALL_PROXY -u all_proxy LLAMA_EVAL_STAGE="$stage"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        swift eval --model "$MODEL_VIEW" --model_type llama
        --template deepseek_r1 --enable_thinking true --port "$PORT"
        --external_plugins "$SCRIPT_DIR/compat_plugins.py"
        --eval_backend Native --eval_dataset "${datasets[@]}" --eval_dataset_args "$dataset_args"
        --extra_eval_args '{"repeats":4,"seed":42,"judge_strategy":"rule","model_args":{"max_retries":0}}'
        --eval_generation_config "{\"temperature\":$temperature,\"top_p\":0.95,\"extra_body\":{\"top_k\":50},\"max_tokens\":$max_tokens,\"timeout\":3600,\"retries\":2,\"retry_interval\":10,\"stream\":false}"
        --infer_backend vllm --vllm_gpu_memory_utilization 0.7 --vllm_max_model_len 40960
        --vllm_max_num_seqs "$max_seqs" --vllm_data_parallel_size "$DP" --vllm_tensor_parallel_size 1
        --seed 42 --eval_num_proc "$client" --max_new_tokens "$max_tokens"
        --temperature "$temperature" --top_p 0.95 --top_k 50
        --attn_impl flash_attn --torch_dtype bfloat16 --stream false --eval_output_dir "$out")
    echo "[$stage] GPU=$CUDA_VISIBLE_DEVICES DP=$DP TP=1 client=$client max_seqs=$max_seqs"
    if (( DRY_RUN )); then printf '%q ' "${command[@]}"; printf '\n'; return; fi

    wait_port
    mkdir -- "$out"
    printf '#!/usr/bin/env bash\nexport PATH=%q CUDA_VISIBLE_DEVICES=%q CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_DENSE_DP_LB_FIX=%q\n' "$PATH" "$CUDA_VISIBLE_DEVICES" "$VLLM_DENSE_DP_LB_FIX" > "$out/command.sh"
    printf '%q ' "${command[@]}" >> "$out/command.sh"; printf '\n' >> "$out/command.sh"
    setsid "${command[@]}" >"$log" 2>&1 &
    local pid=$! start=$SECONDS done_at=0 status=0
    PGID=$pid
    echo "[$stage] 日志：$log"
    while kill -0 "$pid" 2>/dev/null; do
        if (( SECONDS - start >= HARD_TIMEOUT )); then
            echo "[$stage] 超过 $HARD_TIMEOUT 秒；已保留生成结果和日志。" >&2
            cleanup; wait "$pid" 2>/dev/null || true; return 124
        fi
        if tail -c 65536 "$log" | grep -Eq 'EngineCore failed to start\.|Engine core initialization failed\.|Free memory on device cuda:'; then
            echo "[$stage] 引擎启动失败，请查看日志。" >&2
            cleanup; wait "$pid" 2>/dev/null || true; return 1
        fi
        if tail -c 65536 "$log" | grep -Fq 'End time of running main'; then
            (( done_at != 0 )) || done_at=$SECONDS
            (( SECONDS - done_at < 30 )) || break
        fi
        sleep 5
    done
    cleanup
    wait "$pid" || status=$?
    wait_port
    # Swift 完成后有时仍挂着引擎；允许清理后的非零退出，但必须有完成标记和完整核验。
    if (( status != 0 )) && ! grep -Fq 'End time of running main' "$log"; then
        echo "[$stage] Swift 退出码 $status，请查看 $log。" >&2; return "$status"
    fi
    "$PYTHON" "$SCRIPT_DIR/verify_results.py" --stage "$stage" --output "$out" --model "$MODEL_VIEW"
}

for stage in "${STAGES[@]}"; do run_stage "$stage"; done
(( DRY_RUN )) || echo "全部所选评测完成：$OUTPUT_DIR"
