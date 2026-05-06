#!/bin/bash
# 在「数学域」上评测 checkpoint（与 GKD 训练脚本里用的 MATH / 竞赛数学相关）
#
# 说明：
# - 评测结果只打在终端，不写入模型目录（使用临时目录，结束后删除）。
# - swift 仍可能在 checkpoint 目录下追加 eval_result.jsonl / eval_result/*.jsonl；若需一并删除见脚本末尾注释。
# - 结束后会统计：错误题中有多少条的 finish_reason 为 length（达到 max_new_tokens 被截断）。
#
# 依赖：pip install 'ms-swift[eval]'（或 evalscope）

set -eo pipefail

# ========== 可调参数 ==========
# competition_math | math_500
MATH_BENCHMARK="${MATH_BENCHMARK:-competition_math}"

# 只评部分难度（competition_math / math_500 均支持 subset_list，见 EvalScope 文档）
# 留空表示全量。示例（export 后运行本脚本）：
#   export SUBSET_JSON='"subset_list": ["Level 4", "Level 5"]'
SUBSET_JSON="${SUBSET_JSON:-}"

# 抽样条数；留空为全量
EVAL_LIMIT="${EVAL_LIMIT:-}"

MAX_NEW_TOKENS=4096
TEMPERATURE=0
TOP_P=1.0
# ==============================

MODEL_PATH=/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-1.7B

EVAL_TMP="$(mktemp -d)"
trap 'rm -rf "${EVAL_TMP}"' EXIT

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [ -n "${SUBSET_JSON}" ]; then
    EVAL_DS_ARGS=$(printf '{"%s": {%s}}' "${MATH_BENCHMARK}" "${SUBSET_JSON}")
else
    EVAL_DS_ARGS=""
fi

EXTRA_EVAL=""
if [ -n "${EVAL_LIMIT}" ]; then
    EXTRA_EVAL=$(printf '{"limit": %s}' "${EVAL_LIMIT}")
fi

echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Benchmark: ${MATH_BENCHMARK}"
echo "max_new_tokens: ${MAX_NEW_TOKENS} (length 统计以此为准)"
echo "临时评测目录（结束后删除）: ${EVAL_TMP}"
echo "=========================================="

EVAL_CMD=(swift eval
    --model "${MODEL_PATH}"
    --eval_backend Native
    --eval_dataset "${MATH_BENCHMARK}"
    --infer_backend vllm
    --vllm_gpu_memory_utilization 0.9
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --attn_impl flash_attn
    --torch_dtype bfloat16
    --stream false
    --eval_output_dir "${EVAL_TMP}"
    --result_path "${EVAL_TMP}/swift_infer_samples.jsonl"
)

if [ -n "${EVAL_DS_ARGS}" ]; then
    EVAL_CMD+=(--eval_dataset_args "${EVAL_DS_ARGS}")
fi
if [ -n "${EXTRA_EVAL}" ]; then
    EVAL_CMD+=(--extra_eval_args "${EXTRA_EVAL}")
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
"${EVAL_CMD[@]}"

export EVAL_NATIVE_ROOT="${EVAL_TMP}/native"
export MAX_NEW_TOKENS

python3 <<'PY'
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

root = Path(os.environ["EVAL_NATIVE_ROOT"])
max_tok = int(os.environ.get("MAX_NEW_TOKENS", "0"))

def finish_reason_from_prediction(obj: Dict[str, Any]) -> Optional[str]:
    mo = obj.get("model_output") or {}
    choices = mo.get("choices") or []
    if not choices:
        return None
    ch0 = choices[0]
    fr = ch0.get("finish_reason")
    if fr is not None:
        return str(fr).lower()
    fr = ch0.get("stop_reason")
    if fr is not None:
        return str(fr).lower()
    msg = ch0.get("message") or {}
    fr = msg.get("finish_reason")
    if fr is not None:
        return str(fr).lower()
    return None

def usage_completion_tokens(obj: Dict[str, Any]) -> Optional[int]:
    mo = obj.get("model_output") or {}
    for u in (mo.get("usage"), obj.get("usage")):
        if not isinstance(u, dict):
            continue
        v = u.get("completion_tokens")
        if isinstance(v, int):
            return v
    return None

def is_wrong(sample_score: Dict[str, Any]) -> bool:
    try:
        acc = float(sample_score["score"]["value"]["acc"])
        return acc < 1.0 - 1e-9
    except Exception:
        return True

def is_length_limited(pred: Dict[str, Any]) -> bool:
    fr = finish_reason_from_prediction(pred)
    if fr == "length":
        return True
    ct = usage_completion_tokens(pred)
    if ct is not None and max_tok > 0 and ct >= max_tok:
        return True
    return False

total_wrong = 0
wrong_len = 0
files = 0

for rev_path in sorted(root.glob("**/reviews/**/*.jsonl")):
    reviews_dir = rev_path.parent.parent
    run_dir = reviews_dir.parent
    rel = rev_path.relative_to(reviews_dir)
    pred_path = run_dir / "predictions" / rel
    if not pred_path.is_file():
        print(f"[warn] 缺少 predictions，跳过统计: {pred_path}")
        continue
    files += 1
    preds = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    by_index = {}
    for p in preds:
        idx = p.get("index")
        by_index[idx] = p
        if isinstance(idx, int):
            by_index[str(idx)] = p

    with open(rev_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rev = json.loads(line)
            ss = rev.get("sample_score")
            if not isinstance(ss, dict):
                continue
            if not is_wrong(ss):
                continue
            total_wrong += 1
            idx = rev.get("index")
            pred = by_index.get(idx)
            if pred is None and isinstance(idx, int):
                pred = by_index.get(str(idx))
            if pred is None:
                continue
            if is_length_limited(pred):
                wrong_len += 1

print()
print("========== 错误样本 vs 长度截断 ==========")
if files == 0:
    print("[warn] 未找到 native/**/reviews/**/*.jsonl；若评测已成功，请检查 evalscope 输出目录结构是否变更。")
print(f"参与统计的 reviews 文件数: {files}")
print(f"判为错误的样本总数: {total_wrong}")
print(f"其中 finish_reason=length（或 completion_tokens>=max_new_tokens）的条数: {wrong_len}")
if total_wrong > 0:
    print(f"占错误样本比例: {wrong_len / total_wrong * 100:.2f}%")
print("==========================================")
PY

echo "脚本退出时会自动删除临时目录。"
echo "说明：swift 仍可能在模型目录下追加 eval_result.jsonl 或 eval_result/*.jsonl；若不需要可自行删除。"
