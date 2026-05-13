"""
process_reward_distill.py
=========================
Process-Reward Guided On-Policy Distillation Loop.

Pipeline overview
-----------------
Phase 0 (run once, result cached in sample_pool.jsonl):
  1. Sample N problems from hendrycks_math.
  2. Student rollout → discard problems the student already solves.
  3. Teacher rollout on remaining → record teacher_correct (do NOT discard).
  4. Keep all S✗ samples; each item carries teacher_correct ∈ {True, False}.
  5. Save held-out test set (test_set.jsonl, official MATH test split = 5000)
     and snapshot the initial pool as train_eval_set.jsonl for eval tracking.

Per-epoch loop (N_EPOCHS):
  Step 0  vLLM  : Held-out evaluations (in the isolated worker):
                    • test_set.jsonl     – full greedy eval (unbiased metric)
                    • train_eval_set     – incremental: skip already-solved
  Step 1  CPU   : Episode segmentation of each sample's student_response.
  Step 2  vLLM  : Same inference session (enable_prefix_caching=True):
                    a. Filter greedy  – check if student now solves the problem.
                    b. K-fold PR eval – K stochastic forced-answer completions
                       for each episode prefix.
                    c. Continuation   – top-K_CONTINUATION segments per sample,
                       gated by acc_rate ≥ ACC_RATE_THRESHOLD and pr > 0.
  Step 3  CPU   : Compute acc_rates + process_rewards; monotonic merge.
  Step 4  CPU   : Select training records in three branches:
                    • max(acc_rate) > 0   → PR-segment training
                    • max(acc_rate) == 0 and teacher_correct → distill teacher
                    • max(acc_rate) == 0 and teacher wrong   → drop
  Step 5  disk  : Write train_segments.jsonl.
  Step 6  proc  : swift rlhf --rlhf_type gkd  (subprocess, off-policy).
  Step 7  CPU   : Update sample pool (+ update train_eval_solved, eval_log).

Usage
-----
    python tools/process_reward_distill.py
    TRAIN_MODE=all N_EPOCHS=3 OUTPUT_DIR=/path/out python tools/process_reward_distill.py
"""

import gc
import json
import os
import re
import subprocess
import sys
import random
import warnings
from pathlib import Path

import pyarrow.ipc as ipc
import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

# ─── Model paths ─────────────────────────────────────────────────────────────

STUDENT_MODEL = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen"
    "/models/Qwen2.5-1.5B-Instruct"
)
TEACHER_MODEL = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen"
    "/models/Qwen2.5-14B-Instruct"
)

# ─── Dataset ─────────────────────────────────────────────────────────────────

DATASET = "hendrycks_math"
_CACHE_ROOT = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift"
    "/data/hf_datasets_cache"
)
_MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]
DATASET_CONFIGS = {
    "hendrycks_math": {
        "arrow_shards": [
            f"{_CACHE_ROOT}/EleutherAI___hendrycks_math/{subset}/0.0.0"
            "/21a5633873b6a120296cce3e2df9d5550074f4a3/hendrycks_math-train.arrow"
            for subset in _MATH_SUBSETS
        ],
        "test_shards": [
            f"{_CACHE_ROOT}/EleutherAI___hendrycks_math/{subset}/0.0.0"
            "/21a5633873b6a120296cce3e2df9d5550074f4a3/hendrycks_math-test.arrow"
            for subset in _MATH_SUBSETS
        ],
        "prompt_key":    "problem",
        "solution_key":  "solution",
    },
}

# Set N_SAMPLES=0 (default) to use the full dataset.
N_SAMPLES = int(os.environ.get("N_SAMPLES", "0"))   # 0 = all
SEED      = int(os.environ.get("SEED", "42"))

# ─── Episode segmentation ─────────────────────────────────────────────────────

SEGMENT_TRIGGER_PATTERNS = [
    r"\bwait\b",
    r"\balternatively\b",
    r"\blet\s+me\s+check\b",
    r"\blet\s+me\s+reconsider\b",
    r"\bhold\s+on\b",
    r"\bactually\b",
    r"\bhmm\b",
]
SEGMENT_MIN_PERIODS = 3

# ─── Process reward evaluation ───────────────────────────────────────────────

K_EVAL             = 4
EVAL_TEMPERATURE   = 0.7
MAX_ANSWER_TOKENS  = 384
MAX_NEW_TOKENS     = 8192
ROLLOUT_TEMPERATURE = 0.7

# Maximum prompt token budgets used to set vLLM max_model_len.
# Phase-0 prompts are just [system + problem] → 1024 is sufficient.
# Per-epoch prompts include the full student prefix → need up to MAX_NEW_TOKENS.
MAX_PROMPT_PHASE0   = 1024
MAX_PROMPT_EPOCH    = MAX_PROMPT_PHASE0 + MAX_NEW_TOKENS  # [sys][user] + full prefix
# Longest per-request length is the k-fold stage:
#   prompt = [sys][user] + full student prefix + EPISODE_EVAL_SUFFIX (~80 tok)
#   gen    = MAX_ANSWER_TOKENS
# Continuation is now dynamically capped at MAX_PROMPT_EPOCH, so it never exceeds this.
SUFFIX_TOKEN_BUDGET = 128                                 # slack for EPISODE_EVAL_SUFFIX
EPOCH_MAX_MODEL_LEN = MAX_PROMPT_EPOCH + MAX_ANSWER_TOKENS + SUFFIX_TOKEN_BUDGET

EPISODE_EVAL_SYSTEM = "Put your final answer within \\boxed{}."
EPISODE_EVAL_SUFFIX = (
    "\nBased on the reasoning above, stop thinking now and summarize the existing"
    " reasoning. If a definitive answer can be derived, give the final answer in"
    " \\boxed{answer}. If no answer has been reached yet or the result is still"
    " inconclusive, output \\boxed{no answer}."
)

# ─── vLLM ────────────────────────────────────────────────────────────────────

N_GPUS = torch.cuda.device_count()

# Number of pool samples processed per inference mini-batch inside run_epoch.
# Within each mini-batch the request order is:
#   filter(greedy) → kfold_ep0 × K → kfold_ep1 × K → … → continuations
# This keeps same-sample requests adjacent so APC cache hits are maximised.
# One LLM instance is reused across all mini-batches in the same epoch, so
# cached KV blocks remain valid for the whole epoch.
# Set to 0 to process the entire pool in one call (only safe for small pools).
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "200"))

# ─── Training ────────────────────────────────────────────────────────────────

TRAIN_MODE     = os.environ.get("TRAIN_MODE", "all")        # "neg_only" | "all"
N_EPOCHS       = int(os.environ.get("N_EPOCHS", "5"))

# Held-out test eval cadence and subsampling.
# The full MATH test split is 5000 problems — running it every epoch with
# max_tokens=MAX_NEW_TOKENS is the biggest vLLM cost in the per-epoch loop.
#   TEST_EVAL_EVERY   : run test eval every N epochs (last epoch is always run).
#   TEST_EVAL_SAMPLES : 0 = full set; otherwise a seeded subsample of this size
#                       is used on non-final epochs. The last epoch always uses
#                       the full set for a clean final metric.
# Defaults preserve previous behaviour (every epoch, full set).
TEST_EVAL_EVERY   = int(os.environ.get("TEST_EVAL_EVERY", "1"))
TEST_EVAL_SAMPLES = int(os.environ.get("TEST_EVAL_SAMPLES", "0"))

# Continuation fan-out controls.
#   TOP_K_CONTINUATION   : at most K continuations per sample per epoch
#   ACC_RATE_THRESHOLD   : require absolute acc_rate >= τ (not just positive delta)
# These two jointly bound pool growth and prevent "continuing from wrong prefixes".
TOP_K_CONTINUATION  = int(os.environ.get("TOP_K_CONTINUATION", "2"))
ACC_RATE_THRESHOLD  = float(os.environ.get("ACC_RATE_THRESHOLD", "0.3"))

GKD_TRAIN_GPUS  = os.environ.get("GKD_TRAIN_GPUS", ",".join(str(i) for i in range(N_GPUS)))
GKD_NPROC       = int(os.environ.get("GKD_NPROC", str(min(N_GPUS, 4))))
GKD_LR          = float(os.environ.get("GKD_LR", "1e-6"))
GKD_BATCH_SIZE  = int(os.environ.get("GKD_BATCH_SIZE", "1"))
GKD_GRAD_ACCUM  = int(os.environ.get("GKD_GRAD_ACCUM", "4"))
GKD_MAX_LEN     = int(os.environ.get("GKD_MAX_LEN", str(MAX_NEW_TOKENS)))
GKD_WARMUP      = float(os.environ.get("GKD_WARMUP", "0.05"))

# Outer LR schedule across epochs. Each epoch's swift-rlhf run gets a different
# peak LR; the inner per-run scheduler is `constant_with_warmup` so it just
# warms up to peak_i and stays flat (no inner cosine decay). This avoids
# double decay (outer-cosine × inner-cosine), so the only LR decay path is the
# outer one defined here. `cosine`/`linear` scales `GKD_LR` by
# (1, 0.9, 0.65, ...) over epochs; `constant` keeps a flat peak across epochs.
GKD_LR_SCHEDULE = os.environ.get("GKD_LR_SCHEDULE", "cosine")   # "cosine"|"linear"|"constant"

# Easy-sample replay: mix a random subset of student-correct (S✓) samples from
# phase 0 into each epoch's training data as an anchor against catastrophic
# forgetting on the easy distribution. 0 = disabled.
EASY_REPLAY_N   = int(os.environ.get("EASY_REPLAY_N", "0"))

# JSD beta: 0 = forward KL (mode-covering), 1 = reverse KL (mode-seeking),
# 0.5 = symmetric JSD (balanced; less mode collapse than beta=1).
GKD_BETA        = float(os.environ.get("GKD_BETA", "0.5"))

# ─── Paths ───────────────────────────────────────────────────────────────────

_DIR       = Path(__file__).parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(_DIR / "paragraph_distill_output")))
POOL_JSONL         = OUTPUT_DIR / "sample_pool.jsonl"
SEGMENTS_JSONL     = OUTPUT_DIR / "train_segments.jsonl"
CKPT_BASE_DIR      = OUTPUT_DIR / "checkpoints"
TEST_SET_JSONL     = OUTPUT_DIR / "test_set.jsonl"
TRAIN_EVAL_JSONL   = OUTPUT_DIR / "train_eval_set.jsonl"
TRAIN_EVAL_SOLVED  = OUTPUT_DIR / "train_eval_solved.json"
EVAL_LOG_JSONL     = OUTPUT_DIR / "eval_log.jsonl"
EASY_SAMPLES_JSONL = OUTPUT_DIR / "easy_samples.jsonl"

# ─── Answer helpers ──────────────────────────────────────────────────────────

def extract_boxed_answer(text):
    """Return the last \\boxed{...} content, handling one level of nesting."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            i = len(text)
    return results[-1].strip() if results else None


def normalize_answer(ans):
    if ans is None:
        return None
    ans = ans.strip().replace(" ", "").replace("\\,", "")
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return ans


def answers_match(pred, gold):
    return (
        normalize_answer(pred) == normalize_answer(gold)
        if pred is not None and gold is not None
        else False
    )


# ─── Dataset ─────────────────────────────────────────────────────────────────

def load_dataset_samples(split="train"):
    """
    Load problems from the Arrow shards for the given split.

    split="train":
        N_SAMPLES == 0  →  full train set.
        N_SAMPLES  > 0  →  random subset of that size (seeded by SEED).
    split="test":
        Always the full test set (N_SAMPLES does not apply).
    """
    random.seed(SEED)
    cfg        = DATASET_CONFIGS[DATASET]
    shards_key = "arrow_shards" if split == "train" else "test_shards"
    pool       = []
    print(f"Loading dataset '{DATASET}' split={split} "
          f"({len(cfg[shards_key])} shards) …", flush=True)
    for shard_path in cfg[shards_key]:
        reader = ipc.open_stream(shard_path)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            for i in range(len(batch)):
                pool.append({col: batch.column(col)[i].as_py()
                             for col in batch.schema.names})
    total = len(pool)

    if split == "train" and N_SAMPLES > 0 and N_SAMPLES < total:
        idx     = random.sample(range(total), N_SAMPLES)
        samples = [pool[i] for i in idx]
        print(f"  Sampled {len(samples)} / {total} (N_SAMPLES={N_SAMPLES})",
              flush=True)
    else:
        samples = pool
        print(f"  Using full {split} split: {total} samples", flush=True)

    del pool
    return samples


# ─── Episode segmentation (text-based) ───────────────────────────────────────

def segment_response_text(response_text):
    """
    Split response_text into episode segments at trigger words.

    Returns a list of (start_char, end_char) tuples.  The prefix text for
    episode i is  response_text[: end_char].

    A segment boundary is only inserted when a trigger word is encountered
    AND the current segment already contains at least SEGMENT_MIN_PERIODS
    period characters ('.').  This avoids spurious cuts in short sentences.
    """
    combined = re.compile("|".join(SEGMENT_TRIGGER_PATTERNS), re.IGNORECASE)

    episodes      = []
    current_start = 0

    for m in combined.finditer(response_text):
        trigger_pos = m.start()
        if trigger_pos <= current_start:
            continue
        segment_so_far = response_text[current_start:trigger_pos]
        if segment_so_far.count(".") >= SEGMENT_MIN_PERIODS:
            episodes.append((current_start, trigger_pos))
            current_start = trigger_pos

    # Final (or only) episode
    if current_start < len(response_text):
        episodes.append((current_start, len(response_text)))

    if not episodes:
        episodes.append((0, len(response_text)))

    return episodes


# ─── Monotonic run merging ────────────────────────────────────────────────────

def merge_monotonic_runs(points):
    """
    Collapse each maximal non-decreasing or non-increasing sub-sequence of
    process rewards into a single representative point (the last element of
    the run).

    points : list of dicts, each must have key "process_reward".
    Returns a filtered sub-list of points.

    Direction encoding (three-valued):
        +1  strictly increasing
         0  flat / equal  → never triggers a cut, extends current run
        -1  strictly decreasing

    A run boundary is created only on a strict sign flip (+1 → -1 or vice
    versa).
    """
    n = len(points)
    if n <= 1:
        return list(points)

    def _sign(a, b):
        d = b - a
        return 1 if d > 0 else (-1 if d < 0 else 0)

    signs = [
        _sign(points[i]["process_reward"], points[i + 1]["process_reward"])
        for i in range(n - 1)
    ]

    run_end_indices = []
    active_dir      = 0   # no strict direction yet

    for i in range(n - 1):
        s            = signs[i]
        is_last_gap  = (i == n - 2)

        if s != 0:
            if active_dir != 0 and s != active_dir:
                run_end_indices.append(i)   # end of previous run
            active_dir = s

        if is_last_gap:
            run_end_indices.append(n - 1)

    if not run_end_indices:           # all flat → single run
        run_end_indices = [n - 1]

    return [points[idx] for idx in run_end_indices]


# ─── JSONL helpers ────────────────────────────────────────────────────────────

def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records → {path}", flush=True)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_pool(pool, path=None):
    if path is None:
        path = POOL_JSONL
    write_jsonl(path, pool)


# ─── Prompt builders ──────────────────────────────────────────────────────────

def build_filter_prompt(problem, tokenizer):
    """
    Free-thinking prompt used for:
      • Phase-0 student/teacher initial rollout
      • Per-epoch greedy filter check

    Format: [sys: eval_system][user: problem]
    Shares the [sys][user] prefix with kfold/continuation prompts so APC
    hits across filter → kfold → continuation.
    """
    msgs = [
        {"role": "system", "content": EPISODE_EVAL_SYSTEM},
        {"role": "user",   "content": problem},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def build_episode_eval_prompt(problem, partial_resp, tokenizer):
    """
    K-fold forced-answer prompt.

    Format: [sys: eval_system][user: problem][partial_resp][SUFFIX]

    The SUFFIX instructs the model to stop thinking and give a boxed answer.
    If partial_resp ends inside a <think> block the block is closed first so
    the suffix is appended in the correct position.
    """
    resp = partial_resp
    if "<think>" in resp and "</think>" not in resp:
        resp = resp.rstrip() + "\n</think>"
    msgs = [
        {"role": "system", "content": EPISODE_EVAL_SYSTEM},
        {"role": "user",   "content": problem},
    ]
    base = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return base + resp + EPISODE_EVAL_SUFFIX


def build_continuation_prompt(problem, partial_resp, tokenizer):
    """
    Continuation prompt: let the (updated) student continue from a prefix.

    Format: [sys: eval_system][user: problem][partial_resp]

    No SUFFIX – the model continues its chain-of-thought freely.
    The [sys][user] prefix is identical to the K-fold eval prompt, so prefix
    caching maximises KV hits across kfold → continuation.
    """
    msgs = [
        {"role": "system", "content": EPISODE_EVAL_SYSTEM},
        {"role": "user",   "content": problem},
    ]
    base = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return base + partial_resp


# ─── vLLM lifecycle helpers ───────────────────────────────────────────────────

def _resolve_tensor_parallel_size(model_path, requested_tp):
    """Pick a legal TP <= requested_tp that divides every TP-shardable
    dimension vLLM checks at startup: num_attention_heads, num_kv_heads
    (if GQA), hidden_size, intermediate_size, vocab_size.

    Falls back to TP=1 if config can't be read.
    """
    requested_tp = max(1, int(requested_tp))
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"  [vLLM] warning: failed to read model config for TP ({e}); "
              f"falling back to tp=1", flush=True)
        return 1

    def _get_int(*names):
        for name in names:
            v = getattr(cfg, name, None)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    constraints = {
        "num_attention_heads": _get_int("num_attention_heads", "n_head", "num_heads"),
        "num_kv_heads":        _get_int("num_key_value_heads"),
        "hidden_size":         _get_int("hidden_size", "n_embd"),
        "intermediate_size":   _get_int("intermediate_size", "ffn_dim"),
        "vocab_size":          _get_int("vocab_size"),
    }
    constraints = {k: v for k, v in constraints.items() if v and v > 0}

    if not constraints:
        print(f"  [vLLM] warning: no usable shard constraints found in config; "
              f"falling back to tp=1", flush=True)
        return 1

    for tp in range(requested_tp, 0, -1):
        if all(v % tp == 0 for v in constraints.values()):
            if tp != requested_tp:
                bad = {k: v for k, v in constraints.items()
                       if v % requested_tp != 0}
                print(f"  [vLLM] adjusted tp {requested_tp} -> {tp}  "
                      f"(blocked by {bad})", flush=True)
            return tp

    # Should be unreachable because tp=1 always divides everything.
    return 1


def make_llm(model_path, gpu_util=0.85, max_model_len=None, apc=True):
    tp = _resolve_tensor_parallel_size(model_path, N_GPUS)
    if max_model_len is None:
        max_model_len = EPOCH_MAX_MODEL_LEN
    # Disable cudagraph + torch.compile path: vLLM v0.18 V1 engine + GQA
    # (num_kv_heads=2 in Qwen2.5-1.5B) + AOT compile cache has been observed
    # to silently corrupt logits → sampler picks token id 0 (`!`) every step.
    # Override via VLLM_ENFORCE_EAGER=0 if you want to re-enable graph capture.
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "1") != "0"
    print(f"  [vLLM] init  model={os.path.basename(model_path)}  "
          f"tp={tp}  apc={apc}  enforce_eager={enforce_eager}", flush=True)
    return LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        enable_prefix_caching=apc,
        enforce_eager=enforce_eager,
    )


def destroy_llm(llm):
    # Ask vLLM's executor to shut down cleanly before Python GC runs.
    try:
        llm.llm_engine.model_executor.shutdown()
    except Exception:
        pass
    del llm
    for _ in range(3):
        gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # Tear down the distributed process group that vLLM may have created,
    # which also releases its NCCL-allocated CUDA memory.
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    print("  [vLLM] engine destroyed.", flush=True)


def safe_generate(llm, prompts, sampling_params, tokenizer, max_model_len, label=""):
    """Wrapper around llm.generate() that skips prompts exceeding max_model_len.

    ``sampling_params`` may be a single SamplingParams (applied to every prompt)
    or a list of SamplingParams of the same length as ``prompts`` (per-prompt,
    used by the continuation stage to cap new-token budget dynamically).

    Returns a list of RequestOutput|None in 1:1 correspondence with *prompts*.
    None entries indicate prompts that were too long and skipped (warning logged).
    """
    per_prompt_params = isinstance(sampling_params, list)
    if per_prompt_params:
        assert len(sampling_params) == len(prompts), (
            "sampling_params list length must match prompts length")

    valid_indices = []
    for i, p in enumerate(prompts):
        # add_special_tokens=False: prompt strings already contain chat template tokens
        tok_len = len(tokenizer.encode(p, add_special_tokens=False))
        if tok_len >= max_model_len:
            warnings.warn(
                f"[safe_generate/{label}] prompt[{i}] skipped: "
                f"{tok_len} tokens >= max_model_len {max_model_len}",
                stacklevel=2,
            )
        else:
            valid_indices.append(i)

    n_skipped = len(prompts) - len(valid_indices)
    if n_skipped:
        print(f"  [WARNING] safe_generate({label}): "
              f"skipped {n_skipped}/{len(prompts)} over-length prompts", flush=True)

    if not valid_indices:
        return [None] * len(prompts)

    gen_prompts = [prompts[i] for i in valid_indices]
    gen_params = (
        [sampling_params[i] for i in valid_indices]
        if per_prompt_params else sampling_params
    )
    results = llm.generate(gen_prompts, gen_params)

    full_results = [None] * len(prompts)
    for i, r in zip(valid_indices, results):
        full_results[i] = r
    return full_results


# Worker script executed by run_isolated_rollout().
# Written to a temp file and run as a subprocess so that GPU memory is
# *completely* released when the subprocess exits (CUDA context destroyed).
_ROLLOUT_WORKER_SCRIPT = """\
import gc, json, sys, torch
from vllm import LLM, SamplingParams

def main():
    cfg_path, out_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as f:
        cfg = json.load(f)

    tp = cfg.get("tp", cfg["n_gpus"])
    llm = LLM(
        model=cfg["model"],
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=cfg["max_tokens"] + cfg["max_prompt"],
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,   # Phase-0 prompts share no prefixes
        enforce_eager=cfg.get("enforce_eager", True),
    )
    sp_kwargs = dict(n=1, temperature=cfg["temperature"],
                     max_tokens=cfg["max_tokens"])
    if cfg["temperature"] > 0:
        sp_kwargs.update(top_k=cfg["top_k"], top_p=cfg["top_p"])
    outputs = llm.generate(cfg["prompts"], SamplingParams(**sp_kwargs))
    texts = [out.outputs[0].text for out in outputs]

    with open(out_path, "w") as f:
        json.dump(texts, f)
    print(f"[worker] {len(texts)} texts → {out_path}", flush=True)

main()
"""


def run_isolated_rollout(model_path, prompts, temperature, max_tokens,
                         top_k=50, top_p=1.0):
    """
    Run a vLLM rollout in an **isolated subprocess**.

    Why: `del llm + torch.cuda.empty_cache()` does NOT fully release GPU
    memory – the CUDA context and memory pool remain allocated in the process.
    The only reliable way to free GPU memory is for the owning process to exit.

    This helper writes prompts to a temp JSON file, spawns a fresh Python
    subprocess that loads the model, generates, writes results, and exits.
    When the subprocess exits its entire CUDA context is destroyed, leaving
    clean GPU memory for the next model.
    """
    import tempfile

    tp = _resolve_tensor_parallel_size(model_path, N_GPUS)
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "1") != "0"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_cfg.json", delete=False
    ) as f:
        json.dump(
            {
                "model":         model_path,
                "prompts":       prompts,
                "temperature":   temperature,
                "max_tokens":    max_tokens,
                "top_k":         top_k,
                "top_p":         top_p,
                "n_gpus":        N_GPUS,
                "tp":            tp,
                "max_prompt":    MAX_PROMPT_PHASE0,
                "enforce_eager": enforce_eager,
            },
            f,
        )
        cfg_path = f.name

    out_path    = cfg_path.replace("_cfg.json", "_out.json")
    script_path = cfg_path.replace("_cfg.json", "_worker.py")

    with open(script_path, "w") as f:
        f.write(_ROLLOUT_WORKER_SCRIPT)

    print(f"  [isolated rollout] model={os.path.basename(model_path)}"
          f"  n={len(prompts)}  → subprocess", flush=True)
    try:
        subprocess.run(
            [sys.executable, script_path, cfg_path, out_path],
            check=True,
        )
        with open(out_path) as f:
            texts = json.load(f)
    finally:
        for p in [cfg_path, out_path, script_path]:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    print(f"  [isolated rollout] done, subprocess exited, GPU fully free.",
          flush=True)
    return texts


# ─── Checkpoint discovery ─────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir):
    """
    Locate the HuggingFace model saved by swift under output_dir.

    Swift creates a versioned run dir (e.g. v0-YYYYMMDD-HHMMSS/) and saves
    the model either directly there or in a checkpoint-N/ sub-directory.
    We look for config.json (present in every HF model) and return the
    directory of the most recently written one.
    """
    output_dir = Path(output_dir)
    candidates = sorted(
        output_dir.rglob("config.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(
            f"No HuggingFace checkpoint (config.json) found under {output_dir}"
        )
    ckpt = candidates[-1].parent
    print(f"  [ckpt] found → {ckpt}", flush=True)
    return str(ckpt)


# ─── GKD training subprocess ──────────────────────────────────────────────────

def _lr_for_epoch(base_lr, epoch_idx, n_epochs, schedule):
    """Outer LR schedule across epochs — see GKD_LR_SCHEDULE docstring."""
    if schedule == "constant" or n_epochs <= 1:
        return base_lr
    progress = epoch_idx / n_epochs       # 0 at first epoch, < 1 at last
    if schedule == "linear":
        return base_lr * max(0.0, 1.0 - progress)
    if schedule == "cosine":
        import math
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown GKD_LR_SCHEDULE: {schedule}")


def run_swift_gkd(segments_jsonl, student_ckpt, output_dir, lr=None, beta=None):
    """
    Run one epoch of GKD training via  `swift rlhf`  as a subprocess.

    Arguments
    ---------
    segments_jsonl : path to the JSONL file of training segments
    student_ckpt   : path to the current student checkpoint (or base model)
    output_dir     : directory where swift saves the new checkpoint

    Returns
    -------
    Path (str) of the saved checkpoint directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if lr is None:
        lr = GKD_LR
    if beta is None:
        beta = GKD_BETA

    hf_cache = (
        "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen"
        "/ms-swift/data/hf_datasets_cache"
    )

    cmd_parts = [
        f"NPROC_PER_NODE={GKD_NPROC}",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"CUDA_VISIBLE_DEVICES={GKD_TRAIN_GPUS}",
        f"HF_DATASETS_CACHE={hf_cache}",
        "HF_DATASETS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        "swift rlhf",
        f"  --rlhf_type gkd",
        f"  --model {student_ckpt}",
        f"  --teacher_model {TEACHER_MODEL}",
        f"  --dataset {segments_jsonl}",
        f"  --lmbda 0",             # fully off-policy: use dataset responses
        f"  --beta {beta}",         # JSD(S,T); 0=forward KL, 0.5=symmetric, 1=reverse KL
        f"  --seq_kd false",
        f"  --tuner_type full",
        f"  --torch_dtype bfloat16",
        f"  --num_train_epochs 1",
        f"  --per_device_train_batch_size {GKD_BATCH_SIZE}",
        f"  --gradient_accumulation_steps {GKD_GRAD_ACCUM}",
        f"  --learning_rate {lr}",
        f"  --max_length {GKD_MAX_LEN}",
        f"  --max_completion_length {GKD_MAX_LEN}",
        f"  --save_steps 9999999",  # save only at epoch end
        f"  --save_total_limit 1",
        f"  --save_only_model true",
        f"  --logging_steps 1",
        f"  --warmup_ratio {GKD_WARMUP}",
        f"  --lr_scheduler_type constant_with_warmup",  # inner: warmup-only, no decay
        f"  --output_dir {output_dir}",
        f"  --attn_impl flash_attn",
        f"  --gradient_checkpointing true",
        f"  --dataloader_num_workers 4",
        f"  --dataset_num_proc 1",
        f"  --report_to none",
        f"  --use_hf true",
    ]
    cmd = " \\\n".join(cmd_parts)

    print(f"\n{'─'*60}", flush=True)
    print(f"[GKD] Training command:", flush=True)
    print(cmd, flush=True)
    print(f"{'─'*60}\n", flush=True)

    result = subprocess.run(cmd, shell=True, check=True)

    ckpt_path = find_latest_checkpoint(output_dir)
    print(f"[GKD] Training complete → {ckpt_path}", flush=True)
    return ckpt_path


# ─── Phase 0: build initial S✗T✓ pool ────────────────────────────────────────

def phase0_build_pool():
    """
    One-time setup: student rollout → filter correct → teacher rollout →
    filter incorrect → save S✗T✓ pool to disk.

    If POOL_JSONL already exists it is loaded directly (resume support).
    """
    print(f"\n{'='*60}")
    print("Phase 0: Building initial S✗T✓ sample pool")
    print(f"{'='*60}", flush=True)

    if POOL_JSONL.exists():
        print(f"  Existing pool found at {POOL_JSONL}, loading …", flush=True)
        pool = load_jsonl(POOL_JSONL)
        print(f"  Loaded {len(pool)} samples.", flush=True)
        return pool

    cfg        = DATASET_CONFIGS[DATASET]
    samples    = load_dataset_samples()
    gt_answers = [extract_boxed_answer(s[cfg["solution_key"]]) for s in samples]
    n_total    = len(samples)

    # ── 0a: student rollout ─────────────────────────────────────────────────
    # Run in an isolated subprocess so the student model's GPU memory is
    # completely freed (CUDA context destroyed) before the teacher loads.
    print(f"\n  [0a] Student rollout  ({n_total} problems) …", flush=True)
    student_tok     = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    student_prompts = [
        build_filter_prompt(s[cfg["prompt_key"]], student_tok) for s in samples
    ]

    student_texts = run_isolated_rollout(
        STUDENT_MODEL, student_prompts,
        temperature=ROLLOUT_TEMPERATURE, max_tokens=MAX_NEW_TOKENS,
    )

    pending      = []
    easy_samples = []
    n_student_ok = 0
    for s, text, gt in zip(samples, student_texts, gt_answers):
        pred = extract_boxed_answer(text)
        if answers_match(pred, gt):
            n_student_ok += 1
            easy_samples.append({
                "problem":          s[cfg["prompt_key"]],
                "gt_answer":        gt,
                "student_response": text,
            })
            continue
        pending.append({
            "problem":          s[cfg["prompt_key"]],
            "gt_answer":        gt,
            "student_response": text,
        })
    print(f"  Student: {n_student_ok}/{n_total} correct → "
          f"{len(pending)} remain for teacher rollout "
          f"(S✓ samples saved for replay anchor)", flush=True)

    # ── 0b: teacher rollout ─────────────────────────────────────────────────
    # Again isolated – teacher (8B) needs clean GPU after student (1.7B) freed.
    print(f"\n  [0b] Teacher rollout  ({len(pending)} problems) …", flush=True)
    teacher_tok     = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    teacher_prompts = [
        build_filter_prompt(p["problem"], teacher_tok) for p in pending
    ]

    teacher_texts = run_isolated_rollout(
        TEACHER_MODEL, teacher_prompts,
        temperature=0, max_tokens=MAX_NEW_TOKENS,
    )

    pool         = []
    n_teacher_ok = 0
    for item, text in zip(pending, teacher_texts):
        pred = extract_boxed_answer(text)
        tc   = answers_match(pred, item["gt_answer"])
        if tc:
            n_teacher_ok += 1
        pool.append({**item,
                     "teacher_response": text,
                     "teacher_correct":  tc})
    print(f"  Teacher: {n_teacher_ok}/{len(pending)} correct "
          f"(all {len(pool)} S✗ samples kept; T✓ and T✗ both retained)",
          flush=True)

    save_pool(pool)

    # Save easy (S✓) samples for replay regularisation.
    if not EASY_SAMPLES_JSONL.exists():
        write_jsonl(EASY_SAMPLES_JSONL, easy_samples)

    # Save held-out test set (official MATH test split).
    if not TEST_SET_JSONL.exists():
        print(f"\n  Saving held-out test set …", flush=True)
        test_samples = load_dataset_samples(split="test")
        test_records = [
            {
                "problem":   s[cfg["prompt_key"]],
                "gt_answer": extract_boxed_answer(s[cfg["solution_key"]]),
            }
            for s in test_samples
        ]
        write_jsonl(TEST_SET_JSONL, test_records)

    # Snapshot initial pool as fixed train-eval reference (never mutated).
    if not TRAIN_EVAL_JSONL.exists():
        train_eval_records = [
            {"problem": item["problem"], "gt_answer": item["gt_answer"]}
            for item in pool
        ]
        write_jsonl(TRAIN_EVAL_JSONL, train_eval_records)
        with open(TRAIN_EVAL_SOLVED, "w") as f:
            json.dump([], f)

    return pool


# ─── Per-epoch: single vLLM session + training ───────────────────────────────

def _epoch_inference_worker(cfg_path, out_path):
    """
    Isolated-subprocess entry point for per-epoch vLLM inference (Steps 1–4).

    Invoked as:
        python process_reward_distill.py --_epoch-worker <cfg_path> <out_path>

    Reads the current pool from POOL_JSONL, runs all vLLM work, then writes
    a results JSON to out_path and exits.  Because this runs in a separate
    process, the CUDA context is completely destroyed upon exit, leaving the
    GPU fully free for the subsequent swift-rlhf training subprocess.

    Output JSON schema
    ------------------
    {
      "to_remove":  [int, ...],
      "merged_segs": {"<si>": [{"ep_idx":int,"process_reward":float,"end_char":int}, ...]},
      "continuations": [
          {"problem":str, "gt_answer":str, "teacher_response":str,
           "student_response":str},   # prefix + continuation already concatenated
          ...
      ],
      "n_raw": int,
      "n_merged": int
    }
    """
    with open(cfg_path) as f:
        cfg = json.load(f)
    current_ckpt  = cfg["current_ckpt"]
    epoch_idx     = cfg.get("epoch_idx", 0)
    n_epochs      = cfg.get("n_epochs", 1)
    is_last_epoch = (epoch_idx + 1 >= n_epochs)

    pool = load_jsonl(POOL_JSONL)
    if not pool:
        with open(out_path, "w") as f:
            json.dump({"to_remove": [], "merged_segs": {},
                       "empty_acc_rates": {}, "continuations": [],
                       "n_raw": 0, "n_merged": 0,
                       "eval": {}}, f)
        return

    student_tok = AutoTokenizer.from_pretrained(current_ckpt, trust_remote_code=True)

    # ── Step 1: episode segmentation (CPU) ──────────────────────────────────
    print("\n[Step 1] Episode segmentation …", flush=True)
    segmented = []
    for item in pool:
        episodes = segment_response_text(item["student_response"])
        segmented.append({**item, "episodes": episodes})

    n_ep_total = sum(len(s["episodes"]) for s in segmented)
    print(f"  {len(segmented)} samples, {n_ep_total} episodes total "
          f"(avg {n_ep_total / max(len(segmented), 1):.1f})", flush=True)

    # ── Steps 2–4: vLLM inference + CPU processing in mini-batches ─────────
    inf_bs    = INFERENCE_BATCH_SIZE if INFERENCE_BATCH_SIZE > 0 else len(segmented)
    n_batches = max(1, (len(segmented) + inf_bs - 1) // inf_bs)

    filter_params = SamplingParams(n=1, temperature=0, max_tokens=MAX_NEW_TOKENS)
    kfold_params  = SamplingParams(
        n=K_EVAL,
        temperature=EVAL_TEMPERATURE,
        max_tokens=MAX_ANSWER_TOKENS,
        top_k=50,
        top_p=1.0,
        stop=["<|im_end|>"],
    )
    # Continuation uses per-prompt SamplingParams: max_tokens is set dynamically
    # so that (prompt tokens) + (new tokens) stays within the same total budget
    # as phase-0 rollouts, MAX_PROMPT_PHASE0 + MAX_NEW_TOKENS. The longer the
    # prefix, the fewer new tokens are allowed.
    CONT_TOTAL_BUDGET = MAX_PROMPT_PHASE0 + MAX_NEW_TOKENS

    to_remove       = set()
    merged_segs     = {}
    empty_acc_rates = {}
    continuations   = []   # serialisable dicts: {problem, gt_answer, teacher_response, student_response}
    n_raw = n_merged_total = 0

    print(f"\n[Step 2–4] Inference  "
          f"(pool={len(segmented)}, inf_batch={inf_bs}, n_batches={n_batches})",
          flush=True)

    llm = make_llm(current_ckpt)

    # ── Eval: held-out test set + incremental train-eval set ────────────────
    def _run_eval(label, records, skip_idx=None):
        skip = set(skip_idx or [])
        prompts, meta = [], []
        for i, rec in enumerate(records):
            if i in skip:
                continue
            prompts.append(build_filter_prompt(rec["problem"], student_tok))
            meta.append(i)
        if not prompts:
            return len(skip), len(records), []
        outs = safe_generate(llm, prompts, filter_params, student_tok,
                             EPOCH_MAX_MODEL_LEN, label=label)
        newly = []
        for idx, out in zip(meta, outs):
            if out is None:
                continue
            if answers_match(extract_boxed_answer(out.outputs[0].text),
                             records[idx]["gt_answer"]):
                newly.append(idx)
        return len(skip) + len(newly), len(records), newly

    print("\n[Eval] Running held-out evaluations …", flush=True)
    test_records = load_jsonl(TEST_SET_JSONL) if TEST_SET_JSONL.exists() else []
    train_eval_records = (
        load_jsonl(TRAIN_EVAL_JSONL) if TRAIN_EVAL_JSONL.exists() else []
    )
    if TRAIN_EVAL_SOLVED.exists():
        with open(TRAIN_EVAL_SOLVED) as f:
            solved_train = set(json.load(f))
    else:
        solved_train = set()

    test_correct  = test_total  = 0
    train_correct = train_total = 0
    newly_solved_train = []

    should_test_eval = (
        TEST_EVAL_EVERY > 0
        and ((epoch_idx + 1) % TEST_EVAL_EVERY == 0 or is_last_epoch)
    )
    if test_records and should_test_eval:
        # Subsample for non-final epochs; always use full set on last epoch.
        if (TEST_EVAL_SAMPLES > 0 and not is_last_epoch
                and TEST_EVAL_SAMPLES < len(test_records)):
            rng = random.Random(SEED)   # deterministic across epochs
            test_records_use = rng.sample(test_records, TEST_EVAL_SAMPLES)
            tag = "  [subsampled]"
        else:
            test_records_use = test_records
            tag = ""
        test_correct, test_total, _ = _run_eval("eval_test", test_records_use)
        print(f"  test:  {test_correct}/{test_total} = "
              f"{test_correct / max(test_total, 1):.3f}{tag}", flush=True)
    elif test_records:
        print(f"  test:  skipped (TEST_EVAL_EVERY={TEST_EVAL_EVERY}, "
              f"epoch={epoch_idx + 1}/{n_epochs})", flush=True)
    if train_eval_records:
        train_correct, train_total, newly_solved_train = _run_eval(
            "eval_train", train_eval_records, skip_idx=solved_train
        )
        print(f"  train: {train_correct}/{train_total} = "
              f"{train_correct / max(train_total, 1):.3f}  "
              f"(newly solved this epoch: {len(newly_solved_train)})",
              flush=True)

    for b_idx in range(n_batches):
        b_start = b_idx * inf_bs
        b_end   = min(b_start + inf_bs, len(segmented))
        batch   = segmented[b_start:b_end]

        print(f"\n  [mini-batch {b_idx + 1}/{n_batches}]  "
              f"samples {b_start}–{b_end - 1}", flush=True)

        # 2a: filter ────────────────────────────────────────────────────────
        filter_prompts_b = [
            build_filter_prompt(item["problem"], student_tok) for item in batch
        ]
        filter_results_b = safe_generate(
            llm, filter_prompts_b, filter_params, student_tok,
            EPOCH_MAX_MODEL_LEN, label="filter",
        )

        newly_removed_b = 0
        for local_si, (item, out) in enumerate(zip(batch, filter_results_b)):
            si = b_start + local_si
            if out is None:
                continue
            if answers_match(extract_boxed_answer(out.outputs[0].text),
                             item["gt_answer"]):
                to_remove.add(si)
                newly_removed_b += 1

        # 2b: K-fold, short → long per sample ──────────────────────────────
        # Skip samples already removed by filter: their acc_rates / merged_segs
        # are never consumed downstream (Step 3/4/2c all gate on `si in to_remove`),
        # so running K-fold on them is pure waste.
        kfold_prompts_b = []
        kfold_meta_b    = []
        n_skipped_filtered = 0

        for local_si, item in enumerate(batch):
            si = b_start + local_si
            if si in to_remove:
                n_skipped_filtered += 1
                continue
            kfold_prompts_b.append(
                build_episode_eval_prompt(item["problem"], "", student_tok)
            )
            kfold_meta_b.append({"si": si, "ep_idx": -1})
            for ep_idx, (_sc, end_c) in enumerate(item["episodes"]):
                partial = item["student_response"][:end_c]
                kfold_prompts_b.append(
                    build_episode_eval_prompt(item["problem"], partial, student_tok)
                )
                kfold_meta_b.append({"si": si, "ep_idx": ep_idx})

        kfold_results_b = safe_generate(
            llm, kfold_prompts_b, kfold_params, student_tok,
            EPOCH_MAX_MODEL_LEN, label="kfold",
        )

        # 3: acc_rates + process rewards (CPU) ──────────────────────────────
        acc_rates_b = {}
        for meta, out in zip(kfold_meta_b, kfold_results_b):
            if out is None:
                continue
            si, ep_idx = meta["si"], meta["ep_idx"]
            gt = segmented[si]["gt_answer"]
            corrects = [
                answers_match(extract_boxed_answer(o.text), gt)
                for o in out.outputs
            ]
            acc_rates_b[(si, ep_idx)] = sum(corrects) / K_EVAL

        # 4: monotonic merge (CPU) ──────────────────────────────────────────
        for local_si, item in enumerate(batch):
            si = b_start + local_si
            if si in to_remove:
                continue
            empty_acc = acc_rates_b.get((si, -1), 0.0)
            empty_acc_rates[si] = round(empty_acc, 6)
            prev_acc  = empty_acc
            points    = []
            for ep_idx, (_sc, end_c) in enumerate(item["episodes"]):
                curr_acc = acc_rates_b.get((si, ep_idx), 0.0)
                points.append({
                    "ep_idx":         ep_idx,
                    "acc_rate":       round(curr_acc, 6),
                    "process_reward": round(curr_acc - prev_acc, 6),
                    "end_char":       end_c,
                })
                prev_acc = curr_acc
            merged = merge_monotonic_runs(points)
            merged_segs[si] = merged
            n_raw          += len(points)
            n_merged_total += len(merged)

        # 2c: continuations — top-K by acc_rate, thresholded ────────────────
        # For each sample pick at most TOP_K_CONTINUATION segments whose
        # absolute acc_rate >= ACC_RATE_THRESHOLD and whose process_reward > 0.
        # This keeps continuation starts "on the right basis" and bounds fan-out.
        cont_prompts_b = []
        cont_meta_b    = []
        cont_params_b  = []

        for local_si, item in enumerate(batch):
            si = b_start + local_si
            if si not in merged_segs or si in to_remove:
                continue
            candidates = [
                s for s in merged_segs[si]
                if s["process_reward"] > 0 and s["acc_rate"] >= ACC_RATE_THRESHOLD
            ]
            candidates.sort(key=lambda s: s["acc_rate"], reverse=True)
            candidates = candidates[:TOP_K_CONTINUATION]

            for seg in candidates:
                partial = item["student_response"][:seg["end_char"]]
                cont_prompt = build_continuation_prompt(
                    item["problem"], partial, student_tok
                )
                prompt_tok_len = len(student_tok.encode(
                    cont_prompt, add_special_tokens=False
                ))
                remaining = max(1, CONT_TOTAL_BUDGET - prompt_tok_len)
                cont_prompts_b.append(cont_prompt)
                cont_meta_b.append({"si": si, "seg": seg, "item": item})
                cont_params_b.append(SamplingParams(
                    n=1,
                    temperature=ROLLOUT_TEMPERATURE,
                    max_tokens=remaining,
                    top_k=50,
                    top_p=1.0,
                ))

        if cont_prompts_b:
            cont_results_b = safe_generate(
                llm, cont_prompts_b, cont_params_b, student_tok,
                EPOCH_MAX_MODEL_LEN, label="cont",
            )
            for meta_i, out_i in zip(cont_meta_b, cont_results_b):
                if out_i is None:
                    continue
                full_resp = (
                    meta_i["item"]["student_response"][: meta_i["seg"]["end_char"]]
                    + out_i.outputs[0].text
                )
                continuations.append({
                    "problem":          meta_i["item"]["problem"],
                    "gt_answer":        meta_i["item"]["gt_answer"],
                    "teacher_response": meta_i["item"].get("teacher_response", ""),
                    "teacher_correct":  meta_i["item"].get("teacher_correct", False),
                    "student_response": full_resp,
                })

        print(f"    filter_removed={newly_removed_b}  "
              f"kfold_skipped={n_skipped_filtered}  "
              f"kfold_prompts={len(kfold_prompts_b)}  "
              f"cont_prompts={len(cont_prompts_b)}  "
              f"to_remove_total={len(to_remove)}",
              flush=True)

    destroy_llm(llm)

    reduction = (1 - n_merged_total / max(n_raw, 1)) * 100
    print(f"\n[Step 4] Merge summary: raw={n_raw}  merged={n_merged_total}  "
          f"reduction={reduction:.1f}%", flush=True)
    print(f"  Newly learned: {len(to_remove)}", flush=True)
    print(f"  Continuations collected: {len(continuations)}", flush=True)

    results = {
        "to_remove":   sorted(to_remove),
        "merged_segs": {str(si): segs for si, segs in merged_segs.items()},
        "empty_acc_rates": {str(si): v for si, v in empty_acc_rates.items()},
        "continuations": continuations,
        "n_raw":   n_raw,
        "n_merged": n_merged_total,
        "eval": {
            "test_correct":         test_correct,
            "test_total":           test_total,
            "train_correct":        train_correct,
            "train_total":          train_total,
            "newly_solved_train":   newly_solved_train,
        },
    }
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[worker] Results written → {out_path}", flush=True)


def run_epoch(epoch_idx, pool, current_ckpt):
    """
    Execute one training epoch.

    Steps 1–4 (all CUDA work) run in an isolated subprocess so that the GPU
    is completely free when the swift-rlhf training subprocess starts.

    Returns (updated_pool, updated_ckpt_path).
    """
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch_idx + 1}/{N_EPOCHS}  |  pool={len(pool)}  |"
          f"  ckpt={os.path.basename(current_ckpt)}")
    print(f"{'='*60}", flush=True)

    if not pool:
        return pool, current_ckpt

    # Persist pool so the inference worker subprocess can read it.
    save_pool(pool)

    cfg_path = OUTPUT_DIR / f"epoch_{epoch_idx:03d}_worker_cfg.json"
    out_path = OUTPUT_DIR / f"epoch_{epoch_idx:03d}_worker_out.json"

    with open(cfg_path, "w") as f:
        json.dump({
            "current_ckpt": current_ckpt,
            "epoch_idx":    epoch_idx,
            "n_epochs":     N_EPOCHS,
        }, f)

    print(f"\n[Steps 1–4] Spawning isolated inference worker …", flush=True)
    subprocess.run(
        [sys.executable, __file__, "--_epoch-worker", str(cfg_path), str(out_path)],
        check=True,
    )
    # Worker process has exited → its CUDA context is fully destroyed.

    with open(out_path) as f:
        results = json.load(f)

    to_remove       = set(results["to_remove"])
    merged_segs     = {int(k): v for k, v in results["merged_segs"].items()}
    empty_acc_rates = {int(k): v for k, v in results.get("empty_acc_rates", {}).items()}
    continuations   = results["continuations"]
    n_raw           = results["n_raw"]
    n_merged        = results["n_merged"]
    eval_info       = results.get("eval", {}) or {}

    reduction = (1 - n_merged / max(n_raw, 1)) * 100
    print(f"\n[Step 4] Merge summary: raw={n_raw}  merged={n_merged}  "
          f"reduction={reduction:.1f}%", flush=True)
    print(f"  Newly learned: {len(to_remove)} samples removed from pool",
          flush=True)
    print(f"  Continuation samples collected: {len(continuations)}", flush=True)

    # Persist eval results for this epoch.
    if eval_info and eval_info.get("test_total", 0) + eval_info.get("train_total", 0) > 0:
        if TRAIN_EVAL_SOLVED.exists():
            with open(TRAIN_EVAL_SOLVED) as f:
                solved = set(json.load(f))
        else:
            solved = set()
        solved.update(eval_info.get("newly_solved_train", []))
        with open(TRAIN_EVAL_SOLVED, "w") as f:
            json.dump(sorted(solved), f)

        test_total  = eval_info.get("test_total", 0)
        train_total = eval_info.get("train_total", 0)
        log_entry = {
            "epoch":              epoch_idx + 1,
            "test_correct":       eval_info.get("test_correct", 0),
            "test_total":         test_total,
            "test_acc":           (eval_info.get("test_correct", 0) / test_total
                                   if test_total > 0 else None),
            "train_correct":      eval_info.get("train_correct", 0),
            "train_total":        train_total,
            "train_acc":          (eval_info.get("train_correct", 0) / train_total
                                   if train_total > 0 else None),
            "newly_solved_train": len(eval_info.get("newly_solved_train", [])),
        }
        EVAL_LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_LOG_JSONL, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "n/a"
        print(f"  [eval] epoch {epoch_idx + 1}  "
              f"test_acc={_fmt(log_entry['test_acc'])}  "
              f"train_acc={_fmt(log_entry['train_acc'])}  "
              f"newly_solved={log_entry['newly_solved_train']}", flush=True)

    # ── Step 5: build training records (three branches) ───────────────────
    # 1) max(acc_rate) > 0 and has PR-segment training data → segment training
    # 2) max(acc_rate) == 0 and teacher_correct            → distill teacher response
    # 3) max(acc_rate) == 0 and teacher wrong              → drop (truly hopeless)
    print(f"\n[Step 5] Building training records  (TRAIN_MODE={TRAIN_MODE}) …",
          flush=True)

    train_records = []
    n_pr_samples = n_teacher_samples = n_dropped_hopeless = 0

    for si, item in enumerate(pool):
        if si in to_remove:
            continue
        if si not in merged_segs:
            continue
        segs = merged_segs[si]
        empty_acc = empty_acc_rates.get(si, 0.0)
        max_acc   = max([empty_acc] + [s["acc_rate"] for s in segs])

        if max_acc > 0:
            added = 0
            for m_idx, seg in enumerate(segs):
                pr = seg["process_reward"]
                if TRAIN_MODE == "neg_only" and pr >= 0:
                    continue
                start_char   = segs[m_idx - 1]["end_char"] if m_idx > 0 else 0
                prior_text   = item["student_response"][:start_char]
                segment_text = item["student_response"][start_char:seg["end_char"]]
                if not segment_text.strip():
                    continue
                user_content = (
                    item["problem"] + "\n" + prior_text if prior_text else item["problem"]
                )
                train_records.append({
                    "messages": [
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": segment_text},
                    ],
                })
                added += 1
            if added:
                n_pr_samples += 1
        elif item.get("teacher_correct", False):
            teacher_resp = (item.get("teacher_response") or "").strip()
            if teacher_resp:
                train_records.append({
                    "messages": [
                        {"role": "user",      "content": item["problem"]},
                        {"role": "assistant", "content": teacher_resp},
                    ],
                })
                n_teacher_samples += 1
        else:
            n_dropped_hopeless += 1

    print(f"  PR-segment samples: {n_pr_samples}  "
          f"teacher-distill samples: {n_teacher_samples}  "
          f"hopeless dropped: {n_dropped_hopeless}", flush=True)

    # Easy-sample replay: append a random subset of S✓ samples to anchor the
    # easy distribution in each epoch's training data.
    if EASY_REPLAY_N > 0 and EASY_SAMPLES_JSONL.exists():
        easy_all = load_jsonl(EASY_SAMPLES_JSONL)
        if easy_all:
            rng = random.Random(SEED + epoch_idx)
            n_replay = min(EASY_REPLAY_N, len(easy_all))
            for rec in rng.sample(easy_all, n_replay):
                train_records.append({
                    "messages": [
                        {"role": "user",      "content": rec["problem"]},
                        {"role": "assistant", "content": rec["student_response"]},
                    ],
                })
            print(f"  Easy-replay samples: +{n_replay}  "
                  f"(source pool={len(easy_all)})", flush=True)

    print(f"  Training segments: {len(train_records)}", flush=True)

    # ── Step 6: GKD training subprocess ───────────────────────────────────
    # At this point the inference worker has exited and holds no GPU memory.
    new_ckpt = current_ckpt

    if train_records:
        write_jsonl(SEGMENTS_JSONL, train_records)
        ckpt_dir = CKPT_BASE_DIR / f"epoch_{epoch_idx + 1:03d}"
        lr_this_epoch = _lr_for_epoch(GKD_LR, epoch_idx, N_EPOCHS, GKD_LR_SCHEDULE)
        print(f"  [LR] epoch {epoch_idx + 1}/{N_EPOCHS}: "
              f"lr={lr_this_epoch:.3e}  (base={GKD_LR}, schedule={GKD_LR_SCHEDULE})",
              flush=True)
        new_ckpt = run_swift_gkd(
            str(SEGMENTS_JSONL), current_ckpt, str(ckpt_dir),
            lr=lr_this_epoch, beta=GKD_BETA,
        )
    else:
        print("  No training segments → skipping GKD.", flush=True)

    # ── Step 7: update pool ────────────────────────────────────────────────
    print(f"\n[Step 7] Updating sample pool …", flush=True)

    new_pool = [item for si, item in enumerate(pool) if si not in to_remove]

    n_cont_added = 0
    for cont in continuations:
        if answers_match(extract_boxed_answer(cont["student_response"]),
                         cont["gt_answer"]):
            continue
        new_pool.append({
            "problem":          cont["problem"],
            "gt_answer":        cont["gt_answer"],
            "student_response": cont["student_response"],
            "teacher_response": cont.get("teacher_response", ""),
            "teacher_correct":  cont.get("teacher_correct", False),
        })
        n_cont_added += 1

    print(f"  Pool: {len(pool)} → −{len(to_remove)} learned "
          f"+{n_cont_added} continuations → {len(new_pool)}", flush=True)
    save_pool(new_pool)

    return new_pool, new_ckpt


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("process_reward_distill.py")
    print(f"  STUDENT_MODEL = {STUDENT_MODEL}")
    print(f"  TEACHER_MODEL = {TEACHER_MODEL}")
    print(f"  TRAIN_MODE         = {TRAIN_MODE}")
    print(f"  N_EPOCHS           = {N_EPOCHS}")
    print(f"  N_GPUS             = {N_GPUS}")
    print(f"  K_EVAL             = {K_EVAL}")
    print(f"  TOP_K_CONTINUATION = {TOP_K_CONTINUATION}")
    print(f"  ACC_RATE_THRESHOLD = {ACC_RATE_THRESHOLD}")
    print(f"  GKD_LR             = {GKD_LR}")
    print(f"  GKD_LR_SCHEDULE    = {GKD_LR_SCHEDULE}")
    print(f"  GKD_BETA           = {GKD_BETA}")
    print(f"  EASY_REPLAY_N      = {EASY_REPLAY_N}")
    print(f"  N_SAMPLES          = {'all' if N_SAMPLES == 0 else N_SAMPLES}")
    print(f"  OUTPUT_DIR         = {OUTPUT_DIR}")
    print("=" * 60, flush=True)

    if TRAIN_MODE not in ("neg_only", "all"):
        print(f"ERROR: TRAIN_MODE must be 'neg_only' or 'all', got '{TRAIN_MODE}'",
              file=sys.stderr)
        sys.exit(1)

    pool         = phase0_build_pool()
    current_ckpt = STUDENT_MODEL

    for epoch in range(N_EPOCHS):
        if not pool:
            print("\nPool is empty – stopping early.", flush=True)
            break
        pool, current_ckpt = run_epoch(epoch, pool, current_ckpt)

    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"  Final checkpoint : {current_ckpt}")
    print(f"  Final pool size  : {len(pool)}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--_epoch-worker":
        _epoch_inference_worker(sys.argv[2], sys.argv[3])
    else:
        main()
