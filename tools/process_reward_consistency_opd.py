"""
process_reward_consistency_opd.py
=================================
Process-level supervision reliability filtering for on-policy OPD.

Core hypothesis
---------------
If a segment of student reasoning has higher process reward (PR) than another
segment in the SAME response, then the student and teacher distributions
should agree MORE on that segment (i.e. KL should be LOWER). When this
ordering is violated, the KL supervision on that segment is unreliable for
*this particular response* and should be down-weighted in the OPD loss.

We make the following deliberate design choices:

1. NO regression / linear fit between PR and KL.
   PR is noisy (K-fold std ~0.25 at K=8) and KL scales drift between reasoning
   phases. We only care about ORDERING, not magnitude.

2. Per-response, neighbor-only pairwise comparison (Kendall-Tau-ish).
   Within one response we sort segments by PR descending, then look at
   adjacent rank pairs (1-2, 2-3, ...). A pair is consistent iff the
   higher-PR segment has the lower KL. Violation strength is

       violation(A, B) = max(0, KL_A - KL_B) * max(0, PR_A - PR_B)

   so the violation is large only when BOTH the PR gap is big AND the KL
   ordering is meaningfully reversed. The segment inconsistency score is
   the sum of weighted violations the segment participates in.

3. Dataset-relative response filtering.
   Within one epoch, after rollout + PR eval + KL, we compute a per-response
   consistency ratio (fraction of neighbor pairs satisfying the hypothesis)
   and rank ALL responses globally. Only the TOP_PERCENT_RESPONSES% most
   consistent responses are used for training. The rest are routed as:

       student answered correctly (boxed match)  -> discard (already solved)
       student answered wrong                    -> retry next epoch, up to
                                                    MAX_RETRIES times, then
                                                    permanently dropped and
                                                    logged.

4. Segment masking, not deletion.
   Within each kept response, we mask up to the top MASK_TOP_PERCENT_SEGMENTS%
   most-inconsistent merged paragraphs — but ONLY those whose inconsistency
   score is strictly > 0 (i.e., that actually participated in a violation).
   A response whose neighbor pairs are all consistent has zero inconsistency
   on every paragraph and therefore NO mask is applied: the OPD loss falls
   back to its vanilla form on the full response.

   "Masking" means the per-token weight on those segments is set to 0 in
   the OPD loss; the tokens themselves remain in the input sequence as
   conditioning context. labels are kept != -100 so the forward pass goes
   through; the weight simply zeroes the loss contribution.

5. KL is computed on the teacher's TOP-K support set.
   Both during the offline KL phase (used for consistency scoring) AND in
   the training-time OPD loss. For each prefix c_t we pick S_t = TopK_q(c_t),
   gather the student's logits at those K indices, and softmax both
   distributions ONLY over S_t before computing KL. This follows the OPD
   top-K objective and is much cheaper than full-vocab KL.

6. Teacher top-K is CACHED to disk, not recomputed at train time.
   The KL phase (Step 4) already forwards the teacher via vLLM
   prompt_logprobs to obtain top-K (vocab_index, logprob) per response
   position. Instead of discarding them after computing the per-paragraph
   scalar normKL, we serialise them to a binary `.pt` cache keyed by
   rollout index. Step 6 (training) loads the cache and never instantiates
   the teacher on GPU at all. This is what lets us use arbitrarily large
   teachers (e.g. 32B) for OPD without exhausting per-rank GPU memory in
   DDP — the only train-time GPU cost is the student itself. The cache
   is deleted at the end of each epoch (after eval) so disk usage stays
   bounded to a single epoch's worth.

   NOTE: the cache field is named ``topk_logits`` for historical reasons
   but in vLLM mode it holds the top-K **logprobs**. Step 6 only ever
   passes the cache through ``log_softmax(cache / T)`` and gathers the
   student against the cached indices, for which raw logits and logprobs
   are mathematically identical (the global ``logsumexp`` shift cancels
   inside the softmax). Cache header carries ``schema_version=2`` and
   ``logprobs_in_cache=True`` to make the encoding explicit.

Per-epoch pipeline
------------------
  Step 1  vLLM (student) : rollout each work-set problem to a response
                           (generation only — no logprobs; same pattern as
                           process_reward_weighted_distill.py)
  Step 2  CPU            : segment each response (trigger words + period count)
  Step 3  vLLM (student) : K-fold PR eval per segment + empty-prefix baseline
                           (written into rollouts.jsonl for Step 4+)
  Step 4  vLLM (s + t)   : per-token KL on teacher Top-K support via vLLM
                           prompt_logprobs. Student vLLM (TP=1, first GPU)
                           and teacher vLLM (TP=N-1, remaining GPUs) run as
                           parallel subprocesses on disjoint GPU slices.
                           Each engine batches internally via vLLM's
                           continuous batcher; we just gather student
                           logprobs at teacher's top-K indices and reduce
                           to length-weighted normKL per MERGED paragraph
                           (monotonic-run merge over per-segment PR).
  Step 5  CPU            : (a) per-merged-paragraph inconsistency scores;
                           (b) per-response consistency ratio;
                           (c) global response ranking, top-X% selection;
                           (d) for kept responses, mask top-K% paragraphs;
                           (e) for dropped responses, route to discard or
                               retry-with-counter;
                           (f) write epoch_NNN_train.jsonl
  Step 6  torchrun       : OPD training with top-K-support KL and per-token
                           weights (masked tokens contribute zero loss)
  Step 7  vLLM (student) : post-training held-out + train-eval

The training KL is

    L = ( beta * KL(p_s || p_t) + (1 - beta) * KL(p_t || p_s) )   on S_t

with both p_s and p_t renormalised over the teacher's top-K support per
position.

Important env vars
------------------
  TOP_PERCENT_RESPONSES       Top fraction of responses (by consistency ratio)
                              kept for training each epoch. Default 30.
  MASK_TOP_PERCENT_SEGMENTS   Upper bound on the fraction of merged
                              paragraphs masked per kept response. Only
                              paragraphs with strictly positive inconsistency
                              are eligible; a response with no violations
                              gets zero masks (vanilla OPD on full response).
                              Default 20.
  MIN_MERGED_SEGS_PER_RESP    A response needs at least this many merged
                              paragraphs to be eligible for consistency
                              filtering. Default 3 (>= 2 neighbor pairs).
  MAX_RETRIES                 A wrong + low-consistency problem is retried
                              this many epochs before being permanently
                              dropped. Default 3.
  KL_TOPK                     Teacher top-K support set size for all KL
                              computations (offline + training loss).
                              Default 64.
  KL_GPUS_PER_WORKER          GPUs per Step-4 KL subprocess (its
                              CUDA_VISIBLE_DEVICES slice). Inside each worker
                              the teacher is sharded across the slice via HF
                              device_map="auto" and the student is replicated
                              on cuda:0. Workers (data-parallel) =
                              N_GPUS // KL_GPUS_PER_WORKER. Default 2.
  KL_MEM_FRACTION             Fraction of *free* memory on cuda:1+ (within each
                              worker's CUDA_VISIBLE_DEVICES slice) for teacher
                              weights. Default 0.85.
  KL_MEM_FRACTION_CUDA0       Same, but for the worker-local cuda:0 only (the
                              first GPU in that slice after remapping). Student
                              is replicated here and KL logits aggregate here,
                              so this defaults lower. Default 0.68.
                              Both are clamped to [0.5, 0.97].
  KL_FWD_CHUNK                Response positions processed per fp32 KL block.
                              Bounds the runtime peak to O(chunk × vocab),
                              independent of response length. Default 128.
  KL_ON_CPU                   If 1, run chunked KL on CPU (slower, saves GPU
                              headroom). Default 0 (KL on worker-local cuda:0).
  KL_SCORE_BATCH              Rollouts per batched HF forward (left-padded).
                              Default 2; set 1 if forward OOMs.
  KL_ATTN_IMPL                Attention kernel for the Step-4 HF forward.
                              Default: flash_attention_2 if importable, else
                              sdpa.
  OPD_REVERSE_KL_BETA         beta in  L = beta*KL(s||t) + (1-beta)*KL(t||s).
                              Default 0.7 (reverse-leaning, mode-seeking).
  OPD_TEMP                    Softmax temperature in the KL. Default 1.0.
  K_EVAL                      K-fold count per segment in PR eval. Default 8.
  N_EPOCHS                    Outer epochs. Default 5.
  GKD_LR / GKD_BATCH_SIZE / GKD_GRAD_ACCUM / GKD_MAX_LEN / GKD_WARMUP /
  GKD_LR_SCHEDULE             Step 6 training knobs (GKD_BATCH_SIZE default 4).
  TEST_EVAL_EVERY / TEST_EVAL_SAMPLES
                              Eval cadence.
  N_SAMPLES                   Pool problems (0 = full).
  POOL_DATASET                math | hendrycks_math (default) | dapo_math_17k
                              ("math" is an alias for "hendrycks_math").
  DAPO_SHARDS                 Optional ":"-separated arrow shard paths.
  OUTPUT_DIR                  Output dir. Default tools/consistency_opd_output.

Usage
-----
    python tools/process_reward_consistency_opd.py
    OUTPUT_DIR=$PWD/tools/consistency_opd_dapo POOL_DATASET=dapo_math_17k \\
        python tools/process_reward_consistency_opd.py
"""

import gc
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

# ─── Function-only import from the base distillation pipeline ───────────────
#
# We ONLY reuse FUNCTIONS from process_reward_distill — all parameters/constants
# (model paths, temperatures, token budgets, dataset config, vLLM helpers, etc.)
# are redefined locally below so that changes in process_reward_distill don't
# silently change this script's behaviour.
#
# Note: some imported functions still read a few module-level constants
# internally inside process_reward_distill (notably the chat-template
# strings inside build_episode_eval_prompt, and the trigger-word list inside
# segment_response_text). Those are treated as *function implementation
# details* that we deliberately reuse, not user-tunable parameters.
sys.path.insert(0, str(Path(__file__).parent))
from process_reward_distill import (  # noqa: E402
    extract_boxed_answer, answers_match,
    segment_response_text, merge_monotonic_runs,
    write_jsonl, load_jsonl,
    build_filter_prompt, build_episode_eval_prompt,
    make_llm, destroy_llm, safe_generate,
)

# ─── Models ──────────────────────────────────────────────────────────────────
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-1.7B")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-32B")
N_GPUS        = torch.cuda.device_count()

# ─── Reproducibility / pool sizing ───────────────────────────────────────────
SEED      = int(os.environ.get("SEED", "42"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "0"))   # 0 = use full local MATH pool

# ─── Dataset config (local; used by phase 0 and the local MATH loader) ───────
DATASET = "hendrycks_math"
DATASET_CONFIGS = {
    "hendrycks_math": {
        "prompt_key":   "problem",
        "solution_key": "solution",
    },
}

# ─── Generation budgets ──────────────────────────────────────────────────────
MAX_NEW_TOKENS       = int(os.environ.get("MAX_NEW_TOKENS", "8192"))
MAX_ANSWER_TOKENS    = int(os.environ.get("MAX_ANSWER_TOKENS", "384"))
ROLLOUT_TEMPERATURE  = float(os.environ.get("ROLLOUT_TEMPERATURE", "0.7"))
EVAL_TEMPERATURE     = float(os.environ.get("EVAL_TEMPERATURE",   "0.7"))

# Maximum prompt budget for the K-fold eval stage:
#   prompt = [sys] + [user] + full student prefix + EPISODE_EVAL_SUFFIX
#   gen    = MAX_ANSWER_TOKENS
# Plus a slack for chat-template and suffix tokens.
_MAX_PROMPT_PHASE0   = 1024
_MAX_PROMPT_EPOCH    = _MAX_PROMPT_PHASE0 + MAX_NEW_TOKENS
_SUFFIX_TOKEN_BUDGET = 128
EPOCH_MAX_MODEL_LEN  = int(os.environ.get(
    "EPOCH_MAX_MODEL_LEN",
    str(_MAX_PROMPT_EPOCH + MAX_ANSWER_TOKENS + _SUFFIX_TOKEN_BUDGET),
))

# vLLM mini-batching inside the student worker. 0 = process the entire
# work-set in a single call (only safe for small pools).
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "800"))

# K-fold count per segment endpoint in the PR eval stage. K=8 keeps per-PR
# std around 0.25; K=4 (the value used in process_reward_distill) is too
# noisy for the threshold/ordering signal we rely on here.
K_EVAL = int(os.environ.get("K_EVAL", "8"))


# ─── Consistency / masking knobs ─────────────────────────────────────────────
#
# Two-stage pipeline:
#
#   global ranking by response consistency ratio
#       └─► top TOP_PERCENT_RESPONSES%  → train (with segment masking)
#       └─► bottom (1-X)%:
#               student answered correctly → discard (already solved)
#               student answered wrong     → retry next epoch (up to MAX_RETRIES)
#
#   per-kept response:
#       └─► sort merged paragraphs by inconsistency score descending
#       └─► mask up to ceil(MASK_TOP_PERCENT_SEGMENTS% × n_segs) paragraphs,
#           but ONLY those whose inconsistency > 0. A response with zero
#           violations gets nothing masked → vanilla OPD on the full response.
#       └─► mask = per-token weight 0 in the OPD loss; sequence unchanged.

TOP_PERCENT_RESPONSES     = float(os.environ.get("TOP_PERCENT_RESPONSES", "30"))
MASK_TOP_PERCENT_SEGMENTS = float(os.environ.get("MASK_TOP_PERCENT_SEGMENTS", "20"))
MIN_MERGED_SEGS_PER_RESP  = int(os.environ.get("MIN_MERGED_SEGS_PER_RESP", "3"))
MAX_RETRIES               = int(os.environ.get("MAX_RETRIES", "3"))

if not (0.0 < TOP_PERCENT_RESPONSES <= 100.0):
    raise ValueError("TOP_PERCENT_RESPONSES must be in (0, 100]")
if not (0.0 < MASK_TOP_PERCENT_SEGMENTS <= 100.0):
    raise ValueError("MASK_TOP_PERCENT_SEGMENTS must be in (0, 100]")

# ─── KL knobs (used in BOTH the offline KL phase and the training loss) ──────
KL_TOPK             = int(os.environ.get("KL_TOPK", "64"))
OPD_REVERSE_KL_BETA = float(os.environ.get("OPD_REVERSE_KL_BETA", "0.7"))
OPD_TEMP            = float(os.environ.get("OPD_TEMP", "1.0"))
if not (0.0 <= OPD_REVERSE_KL_BETA <= 1.0):
    raise ValueError(f"OPD_REVERSE_KL_BETA must be in [0,1], got {OPD_REVERSE_KL_BETA}")

# ─── KL data-parallel knob ───────────────────────────────────────────────────
# Number of GPUs each Step-4 KL worker owns (its CUDA_VISIBLE_DEVICES slice).
# Step 4 now runs a forward-only HF pass (NOT vLLM): inside each worker the
# teacher is sharded across the slice via device_map="auto" (layer-split model
# parallelism) and the student (small) is replicated on cuda:0. Teacher
# top-K-support KL is computed on-GPU under no_grad. Data parallelism comes
# from launching
# ``N_GPUS // KL_GPUS_PER_WORKER`` such workers, each on a disjoint GPU slice.
# For a 32B teacher on 40GB cards, 2 GPUs per worker fit the sharded weights
# (~32GB/GPU) plus the student and activations.
KL_GPUS_PER_WORKER  = int(os.environ.get("KL_GPUS_PER_WORKER", "2"))
if KL_GPUS_PER_WORKER < 1:
    raise ValueError(f"KL_GPUS_PER_WORKER must be >= 1, got {KL_GPUS_PER_WORKER}")

# Student prompt_logprobs window in Step 4. Must be >= KL_TOPK; a wider
# window maximises the chance that teacher's top-K vocab indices are
# present in student's logprob dict (we gather student's lp at teacher's
# indices; any index that isn't returned is filled with a tiny fallback
# logprob and flagged in the per-record miss-rate).
KL_VLLM_STUDENT_LOGPROBS = int(os.environ.get("KL_VLLM_STUDENT_LOGPROBS", "256"))
if KL_VLLM_STUDENT_LOGPROBS < KL_TOPK:
    raise ValueError(
        f"KL_VLLM_STUDENT_LOGPROBS ({KL_VLLM_STUDENT_LOGPROBS}) must be "
        f">= KL_TOPK ({KL_TOPK})"
    )

# gpu_memory_utilization for the Step 4 vLLM engines. The default 0.80
# (was 0.85) leaves a ~10 GB headroom per card so vLLM can allocate the
# ephemeral [N, V] fp32 log_softmax tensor used by prompt_logprobs.
# Lower further if you still OOM in `_get_prompt_logprobs_dict`.
KL_VLLM_GPU_UTIL = float(os.environ.get("KL_VLLM_GPU_UTIL", "0.80"))

# Student vLLM (Step 1-3 rollout/K-fold, Step 7 eval) is ALWAYS data-parallel:
# one worker per visible GPU, TP=1 inside each worker.

# ─── Training knobs ──────────────────────────────────────────────────────────
N_EPOCHS        = int(os.environ.get("N_EPOCHS", "5"))
GKD_LR          = float(os.environ.get("GKD_LR", "1e-6"))
GKD_BATCH_SIZE  = int(os.environ.get("GKD_BATCH_SIZE", "1"))
GKD_GRAD_ACCUM  = int(os.environ.get("GKD_GRAD_ACCUM", "4"))
GKD_MAX_LEN     = int(os.environ.get("GKD_MAX_LEN", str(MAX_NEW_TOKENS + 1024)))
GKD_WARMUP      = float(os.environ.get("GKD_WARMUP", "0.05"))
GKD_LR_SCHEDULE = os.environ.get("GKD_LR_SCHEDULE", "cosine")
GKD_MASTER_PORT = int(os.environ.get("GKD_MASTER_PORT", "29600"))

TEST_EVAL_EVERY   = int(os.environ.get("TEST_EVAL_EVERY", "1"))
TEST_EVAL_SAMPLES = int(os.environ.get("TEST_EVAL_SAMPLES", "500"))

# ─── Pool dataset selection ──────────────────────────────────────────────────
# Accepted values:
#   "hendrycks_math" | "math"  → Hendrycks et al. MATH (7 subsets)
#                                Loaded LOCALLY via datasets.load_from_disk
#                                from LOCAL_MATH_DATA_DIR (the combined
#                                train+test set is split deterministically;
#                                LOCAL_MATH_TEST_HOLDOUT problems are held
#                                out for held-out eval).
#   "dapo_math_17k"           → DAPO-Math-17k (~17k, train only). Requires
#                                DAPO_SHARDS pointing to locally readable
#                                arrow shards.
POOL_DATASET = os.environ.get("POOL_DATASET", "hendrycks_math").lower()
if POOL_DATASET == "math":
    POOL_DATASET = "hendrycks_math"
if POOL_DATASET not in ("hendrycks_math", "dapo_math_17k"):
    raise ValueError(f"POOL_DATASET must be 'math', 'hendrycks_math' or "
                     f"'dapo_math_17k', got {POOL_DATASET!r}")

# Local MATH cache. Each subset under this folder is a HuggingFace
# `load_from_disk` directory (algebra/, counting_and_probability/, …).
# These directories hold the TRAIN split and are used as the training pool.
LOCAL_MATH_DATA_DIR = os.environ.get(
    "LOCAL_MATH_DATA_DIR",
    "/home/ubuntu/chenyang/opd/data/hendrycks_math",
)
# Directory containing the MATH TEST split, stored in the same per-subset
# `save_to_disk` layout as LOCAL_MATH_DATA_DIR.  When this directory exists
# and test_set.jsonl has not yet been built, the script will build it once
# from all test-split subsets.  Set to "" to skip automatic test-set building
# (you can then place test_set.jsonl manually).
LOCAL_MATH_TEST_DATA_DIR = os.environ.get(
    "LOCAL_MATH_TEST_DATA_DIR",
    "/home/ubuntu/chenyang/opd/data/hendrycks_math_test",
)
# How many problems to deterministically carve off the combined local MATH
# set for the held-out test split. Default 0: use all data for training;
# the proper test set comes from LOCAL_MATH_TEST_DATA_DIR instead.
LOCAL_MATH_TEST_HOLDOUT = int(os.environ.get("LOCAL_MATH_TEST_HOLDOUT", "0"))
_MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

# DAPO-Math-17k arrow shards (only consulted when POOL_DATASET=dapo_math_17k).
# Override via DAPO_SHARDS env var (colon-separated absolute paths).
DAPO_SHARDS = (
    os.environ.get("DAPO_SHARDS").split(":")
    if os.environ.get("DAPO_SHARDS") else []
)

# ─── Paths ───────────────────────────────────────────────────────────────────
_DIR       = Path(__file__).parent
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR", str(_DIR / "consistency_opd_output")))

POOL_JSONL             = OUTPUT_DIR / "pool.jsonl"
TEST_SET_JSONL         = OUTPUT_DIR / "test_set.jsonl"
TRAIN_EVAL_JSONL       = OUTPUT_DIR / "train_eval_set.jsonl"
TRAIN_EVAL_SOLVED      = OUTPUT_DIR / "train_eval_solved.json"
EVAL_LOG_JSONL         = OUTPUT_DIR / "eval_log.jsonl"
CKPT_BASE_DIR          = OUTPUT_DIR / "checkpoints"
RETRY_POOL_JSONL       = OUTPUT_DIR / "retry_pool.jsonl"
DROPPED_RETRIES_JSONL  = OUTPUT_DIR / "dropped_retries.jsonl"
DISCARDED_LOG_JSONL    = OUTPUT_DIR / "discarded_correct.jsonl"


# ─── Pool sample loaders (dataset-agnostic shim) ─────────────────────────────

def _extract_problem_text(row):
    val = row.get("prompt")
    if isinstance(val, list):
        for m in val:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c
        if val and isinstance(val[-1], dict):
            c = val[-1].get("content")
            if isinstance(c, str) and c.strip():
                return c
    if isinstance(val, str) and val.strip():
        return val
    val = row.get("messages")
    if isinstance(val, list):
        for m in val:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c
    val = row.get("problem")
    if isinstance(val, str) and val.strip():
        return val
    return None


def _extract_gt_answer(row):
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        for k in ("ground_truth", "answer", "solution"):
            v = rm.get(k)
            if v is not None:
                return v
    for k in ("ground_truth", "answer", "solution", "gt_answer"):
        v = row.get(k)
        if v is not None:
            return v
    return None


def _load_dapo_samples_normalized():
    from pyarrow import ipc

    pool = []
    print(f"Loading dataset 'dapo_math_17k' from {len(DAPO_SHARDS)} shards …",
          flush=True)
    for shard_path in DAPO_SHARDS:
        reader = ipc.open_stream(shard_path)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            for i in range(len(batch)):
                row = {col: batch.column(col)[i].as_py()
                       for col in batch.schema.names}
                problem = _extract_problem_text(row)
                gt      = _extract_gt_answer(row)
                if not problem or gt is None or not str(gt).strip():
                    continue
                pool.append({"problem": str(problem), "gt_answer": str(gt)})

    total = len(pool)
    if total == 0:
        raise RuntimeError("DAPO load returned 0 samples.")
    if N_SAMPLES > 0 and N_SAMPLES < total:
        rng = random.Random(SEED)
        idx = rng.sample(range(total), N_SAMPLES)
        samples = [pool[i] for i in idx]
        print(f"  Sampled {len(samples)} / {total} (N_SAMPLES={N_SAMPLES})",
              flush=True)
    else:
        samples = pool
        print(f"  Using full DAPO-Math-17k: {total} samples", flush=True)
    return samples


def _load_math_local_split(seed=SEED, test_holdout=None):
    """Load local MATH from LOCAL_MATH_DATA_DIR and produce a deterministic
    (train_pool, test_pool) split.

      • All 7 subsets are loaded via datasets.load_from_disk and concatenated
        in subset order, then shuffled with random.Random(seed) so the split
        is reproducible across runs.
      • The first `test_holdout` problems become the test set; the remainder
        becomes the training pool. Set test_holdout=0 to disable the split
        entirely (test_pool=[]).

    Each output record is {"problem": str, "gt_answer": str, "subset": str}.
    """
    from datasets import load_from_disk

    if test_holdout is None:
        test_holdout = LOCAL_MATH_TEST_HOLDOUT

    cfg = DATASET_CONFIGS[DATASET]
    pkey = cfg["prompt_key"]
    skey = cfg["solution_key"]

    if not os.path.isdir(LOCAL_MATH_DATA_DIR):
        raise FileNotFoundError(
            f"LOCAL_MATH_DATA_DIR does not exist: {LOCAL_MATH_DATA_DIR}\n"
            f"Set LOCAL_MATH_DATA_DIR to a folder containing per-subset "
            f"`load_from_disk` directories.")

    all_samples = []
    print(f"Loading local MATH from {LOCAL_MATH_DATA_DIR}  "
          f"({len(_MATH_SUBSETS)} subsets) …", flush=True)
    for subset in _MATH_SUBSETS:
        path = os.path.join(LOCAL_MATH_DATA_DIR, subset)
        if not os.path.isdir(path):
            print(f"  WARNING: subset folder missing: {path} (skipped)",
                  flush=True)
            continue
        ds = load_from_disk(path)
        before = len(all_samples)
        for row in ds:
            problem = row.get(pkey)
            sol     = row.get(skey)
            gt      = extract_boxed_answer(sol) if sol is not None else None
            if not problem or gt is None:
                continue
            all_samples.append({
                "problem":   str(problem),
                "gt_answer": str(gt),
                "subset":    subset,
            })
        print(f"  {subset:30s}: +{len(all_samples) - before}", flush=True)

    total = len(all_samples)
    if total == 0:
        raise RuntimeError(
            f"Local MATH load returned 0 samples from {LOCAL_MATH_DATA_DIR}.")

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    if test_holdout > 0 and test_holdout < total:
        test_pool  = all_samples[:test_holdout]
        train_pool = all_samples[test_holdout:]
    else:
        test_pool  = []
        train_pool = all_samples

    print(f"  Local MATH split: train={len(train_pool)}  test={len(test_pool)}  "
          f"(holdout={test_holdout}, seed={seed})", flush=True)
    return train_pool, test_pool


def _load_pool_samples_normalized():
    """Returns the training pool as list[{problem, gt_answer}]. For MATH the
    held-out test set is also cached on the module for test_set.jsonl setup.
    """
    global _LOCAL_MATH_TEST_CACHE
    if POOL_DATASET == "dapo_math_17k":
        _LOCAL_MATH_TEST_CACHE = []
        return _load_dapo_samples_normalized()

    train_pool, test_pool = _load_math_local_split()
    _LOCAL_MATH_TEST_CACHE = test_pool

    if N_SAMPLES > 0 and N_SAMPLES < len(train_pool):
        rng = random.Random(SEED + 1)  # different sub-stream from the test split
        idx = rng.sample(range(len(train_pool)), N_SAMPLES)
        train_pool = [train_pool[i] for i in idx]
        print(f"  Sub-sampled train pool: {len(train_pool)} "
              f"(N_SAMPLES={N_SAMPLES})", flush=True)
    return [{"problem": s["problem"], "gt_answer": s["gt_answer"]}
            for s in train_pool]


_LOCAL_MATH_TEST_CACHE = []   # populated by _load_pool_samples_normalized


# ─── Outer LR schedule ───────────────────────────────────────────────────────

def _lr_for_epoch(base_lr, epoch_idx, n_epochs, schedule):
    if schedule == "constant" or n_epochs <= 1:
        return base_lr
    progress = epoch_idx / n_epochs
    if schedule == "linear":
        return base_lr * max(0.0, 1.0 - progress)
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown GKD_LR_SCHEDULE: {schedule}")


# ─── Training pool (full dataset; no teacher pre-filter) ─────────────────────

def _load_math_test_split():
    """Load the MATH test split from LOCAL_MATH_TEST_DATA_DIR.

    Returns a list of {problem, gt_answer} dicts, or [] if the directory does
    not exist or contains no valid samples.
    """
    from datasets import load_from_disk

    if not LOCAL_MATH_TEST_DATA_DIR or not os.path.isdir(LOCAL_MATH_TEST_DATA_DIR):
        return []

    cfg   = DATASET_CONFIGS[DATASET]
    pkey  = cfg["prompt_key"]
    skey  = cfg["solution_key"]

    samples = []
    print(f"  Loading MATH test split from {LOCAL_MATH_TEST_DATA_DIR} …", flush=True)
    for subset in _MATH_SUBSETS:
        path = os.path.join(LOCAL_MATH_TEST_DATA_DIR, subset)
        if not os.path.isdir(path):
            continue
        ds = load_from_disk(path)
        before = len(samples)
        for row in ds:
            problem = row.get(pkey)
            sol     = row.get(skey)
            gt      = extract_boxed_answer(sol) if sol is not None else None
            if not problem or gt is None:
                continue
            samples.append({"problem": str(problem), "gt_answer": str(gt)})
        print(f"    {subset:30s}: +{len(samples) - before}", flush=True)

    print(f"  MATH test split: {len(samples)} problems total.", flush=True)
    return samples


def _ensure_eval_sidecars(pool):
    """Build test_set.jsonl from the MATH test split (once) if not yet present."""
    if TEST_SET_JSONL.exists():
        return

    if POOL_DATASET == "hendrycks_math":
        test_samples = _load_math_test_split()
        if test_samples:
            print(f"\n  Building test_set.jsonl from MATH test split "
                  f"({len(test_samples)} problems) …", flush=True)
            write_jsonl(TEST_SET_JSONL, test_samples)
            print(f"  Wrote {len(test_samples)} records → {TEST_SET_JSONL}", flush=True)
        else:
            print(f"  [eval] LOCAL_MATH_TEST_DATA_DIR not found or empty "
                  f"({LOCAL_MATH_TEST_DATA_DIR!r}) — Step 7 test eval will be skipped.\n"
                  f"  Download the MATH test split with tools/download_hendrycks_math.py "
                  f"--split test  or set LOCAL_MATH_TEST_DATA_DIR.", flush=True)
    else:
        print(f"  [{POOL_DATASET}] no bundled test split — "
              f"skipping test_set.jsonl (place an external one at {TEST_SET_JSONL} "
              f"to enable Step 7 test eval).", flush=True)


def load_training_pool():
    """Load the full training pool (no teacher greedy filter).

    Uses the global train split from the configured dataset. Cached to
    pool.jsonl for fast restarts; set REFRESH_POOL=1 to reload from source.
    """
    print(f"\n{'='*60}")
    print("Loading training pool (full dataset, no teacher filter)")
    print(f"{'='*60}", flush=True)

    refresh = os.environ.get("REFRESH_POOL", "0").strip() not in (
        "", "0", "false", "False",
    )

    if POOL_JSONL.exists() and not refresh:
        pool = load_jsonl(POOL_JSONL)
        print(f"  Loaded cached pool: {len(pool)} samples from {POOL_JSONL}",
              flush=True)
        _ensure_eval_sidecars(pool)
        return pool

    if refresh and POOL_JSONL.exists():
        print(f"  REFRESH_POOL=1 → reloading from {POOL_DATASET} …", flush=True)

    pool = _load_pool_samples_normalized()
    print(f"  Global train pool: {len(pool)} problems (no teacher filter)",
          flush=True)
    write_jsonl(POOL_JSONL, pool)
    _ensure_eval_sidecars(pool)
    return pool


# ─── Retry pool persistence ──────────────────────────────────────────────────
#
# retry_pool.jsonl is REWRITTEN each epoch with the current backlog. Each entry:
#     { "problem": ..., "gt_answer": ..., "retry_count": int,
#       "first_epoch": int, "last_epoch": int }
# When retry_count reaches MAX_RETRIES we drop the problem entirely and append
# its record (with reason) to dropped_retries.jsonl.
#
# We key retries on (problem, gt_answer). Pool order / pool_idx is NOT used:
# the retry pool floats on top of the static pool and merges in at the start
# of each epoch.

def _retry_key(rec):
    return (rec.get("problem", ""), rec.get("gt_answer", ""))


def _load_retry_pool():
    if not RETRY_POOL_JSONL.exists():
        return []
    return load_jsonl(RETRY_POOL_JSONL)


def _save_retry_pool(records):
    write_jsonl(RETRY_POOL_JSONL, records)


def _append_dropped_retries(records):
    if not records:
        return
    DROPPED_RETRIES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(DROPPED_RETRIES_JSONL, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_discarded(records):
    if not records:
        return
    DISCARDED_LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(DISCARDED_LOG_JSONL, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_workset_for_epoch(pool, retry_pool, epoch_idx):
    """Merge static pool + retry backlog into the rollout work-set for this
    epoch. Retry entries are deduplicated against the static pool by
    (problem, gt_answer) — if the problem is in both, we keep ONLY the retry
    record (which carries retry_count) so that the increment logic later
    sees the right counter.

    Each work-set entry is annotated with:
        retry_count : int       (0 for fresh pool problems)
        first_epoch : int       (epoch at which it first entered retry; -1 fresh)
        source      : "pool" | "retry"
    """
    retry_keys = {_retry_key(r): r for r in retry_pool}

    work = []
    for it in pool:
        k = _retry_key(it)
        if k in retry_keys:
            r = retry_keys[k]
            work.append({
                "problem":     it["problem"],
                "gt_answer":   it["gt_answer"],
                "retry_count": int(r.get("retry_count", 0)),
                "first_epoch": int(r.get("first_epoch", epoch_idx)),
                "source":      "retry",
            })
        else:
            work.append({
                "problem":     it["problem"],
                "gt_answer":   it["gt_answer"],
                "retry_count": 0,
                "first_epoch": -1,
                "source":      "pool",
            })

    # Defensive: retry entries that no longer exist in pool (shouldn't happen
    # for a frozen pool, but tolerate).
    pool_keys = {_retry_key(it) for it in pool}
    for k, r in retry_keys.items():
        if k not in pool_keys:
            work.append({
                "problem":     r["problem"],
                "gt_answer":   r["gt_answer"],
                "retry_count": int(r.get("retry_count", 0)),
                "first_epoch": int(r.get("first_epoch", epoch_idx)),
                "source":      "retry",
            })
    return work


# ─── Consistency scoring (CPU-only, runs in orchestrator after KL phase) ─────
#
# Inputs per response:
#   merged_paragraphs : list of dicts with keys
#       { idx, tok_start, tok_end, length, normKL, process_reward, ... }
# where `normKL` is the OPD-mix scalar (beta*rev + (1-beta)*fwd) used by the
# training loss — so "inconsistent" really means "unreliable for the loss".

def _dist_summary(values, ps=(50, 75, 90, 95, 99)):
    """Cheap pure-Python summary of a list of floats. Returns a dict with
    count, mean, min, max, and the requested percentiles (linear-interp).
    Empty inputs yield all-None numeric fields with count=0.
    """
    n = len(values)
    if n == 0:
        out = {"count": 0, "mean": None, "min": None, "max": None}
        out.update({f"p{p}": None for p in ps})
        return out
    s = sorted(values)
    mean = sum(s) / n
    out = {"count": n, "mean": round(mean, 6),
           "min": round(s[0], 6), "max": round(s[-1], 6)}
    for p in ps:
        k = (p / 100.0) * (n - 1)
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        out[f"p{p}"] = round(s[lo] * (1 - frac) + s[hi] * frac, 6)
    return out


def _print_dispatch_stats(stats):
    """Human-friendly summary of the per-epoch consistency dispatch.

    Crucial sanity check at the bottom:
      • frac_kept_paragraphs_zero ≈ 1.0  → almost every paragraph in the
        training set is already consistent; the masking machinery is mostly
        dead code and this run is effectively
            top-X% consistent responses + vanilla OPD.
      • mask_budget_utilisation  near 0  → same conclusion.
    Both are useful diagnostic signals.
    """
    def _fmt_dist(d):
        if not d or d.get("count", 0) == 0:
            return "n=0"
        return (f"n={d['count']}  mean={d['mean']:.4g}  "
                f"p50={d['p50']:.4g}  p75={d['p75']:.4g}  "
                f"p90={d['p90']:.4g}  p95={d['p95']:.4g}  "
                f"max={d['max']:.4g}")

    n_kept = stats["n_kept"]
    print("  [dispatch] headline counts")
    print(f"    n_kl_records           = {stats['n_kl_records']}")
    print(f"    n_eligible             = {stats['n_eligible']}")
    print(f"    n_kept (top-X% by cons)= {n_kept}")
    print(f"    n_retry_new            = {stats['n_retry_new']}")
    print(f"    n_dropped (max retries)= {stats['n_dropped']}")
    print(f"    n_discarded (S✓+lowC)  = {stats['n_discarded_correct']}")

    print("  [dispatch] masking budget vs actual")
    print(f"    paragraphs in kept     = {stats['n_paragraphs_total']}")
    print(f"    theoretical max masks  = {stats['n_paragraphs_mask_budget']}  "
          f"(= sum of ceil(n_segs × MASK%/100) over kept)")
    print(f"    actually masked        = {stats['n_paragraphs_masked']}  "
          f"(utilisation {stats['mask_budget_utilisation']:.1f}%)")
    print(f"    kept resps w/ ≥1 mask  = {stats['n_kept_with_actual_masks']}"
          f" / {n_kept}")
    print(f"    kept resps all-consistent (vanilla-OPD path) = "
          f"{stats['n_kept_all_consistent']} / {n_kept}")
    print(f"    frac kept paragraphs with inc=0 = "
          f"{stats['frac_kept_paragraphs_zero']:.3f}")

    print("  [dispatch] consistency-ratio distribution")
    print(f"    eligible: {_fmt_dist(stats['eligible_consistency'])}")
    print(f"    kept    : {_fmt_dist(stats['kept_consistency'])}")

    print("  [dispatch] inconsistency distribution (strictly > 0 only)")
    print(f"    all       : {_fmt_dist(stats['para_inc_all_pos'])}")
    print(f"    eligible  : {_fmt_dist(stats['para_inc_eligible_pos'])}")
    print(f"    kept      : {_fmt_dist(stats['para_inc_kept_pos'])}")

    # Loud diagnostic.
    if stats['n_paragraphs_masked'] == 0 and n_kept > 0:
        print("  [dispatch] WARNING: 0 paragraphs masked across "
              f"{n_kept} training responses — masking code is INACTIVE this "
              "epoch (only top-X% response filtering is doing work).")


def _violation(pr_a, kl_a, pr_b, kl_b):
    """Weighted neighbor violation. A is the higher-PR-rank segment.
    Returns the violation amount (>=0). Zero if either the PR gap is
    non-positive or the KL ordering already agrees with the hypothesis.
    """
    pr_gap = pr_a - pr_b
    kl_inv = kl_a - kl_b
    if pr_gap <= 0 or kl_inv <= 0:
        return 0.0
    return float(pr_gap) * float(kl_inv)


def _score_response_paragraphs(paras):
    """Given a list of merged paragraphs for ONE response (any order),
    compute:
      - per-paragraph inconsistency score (accumulated weighted violations)
      - response consistency ratio (fraction of consistent neighbor pairs)
      - n_pairs

    Sorting key is PR descending; ties broken by KL ascending so equal-PR
    segments do not produce spurious violations on themselves. If a
    response has fewer than 2 paragraphs there are no pairs — consistency
    is set to NaN and all scores to 0.
    """
    n = len(paras)
    inconsistency = [0.0] * n
    if n < 2:
        return inconsistency, float("nan"), 0

    order = sorted(
        range(n),
        key=lambda i: (-paras[i]["process_reward"], paras[i]["normKL"]),
    )

    n_pairs = 0
    n_consistent = 0
    for k in range(n - 1):
        i = order[k]
        j = order[k + 1]
        pr_a, kl_a = paras[i]["process_reward"], paras[i]["normKL"]
        pr_b, kl_b = paras[j]["process_reward"], paras[j]["normKL"]

        n_pairs += 1
        # Consistent iff higher-PR side does not have strictly larger KL.
        # (Ties on KL count as consistent — no violation either way.)
        if (pr_a - pr_b) <= 0:
            # Identical PR — no information about consistency. Skip.
            n_pairs -= 1
            continue
        if kl_a <= kl_b:
            n_consistent += 1
        v = _violation(pr_a, kl_a, pr_b, kl_b)
        inconsistency[i] += v
        inconsistency[j] += v

    consistency = (n_consistent / n_pairs) if n_pairs > 0 else float("nan")
    return inconsistency, consistency, n_pairs


# ─── Step 1–3: student worker (rollout + segmentation + K-fold PR eval) ──────
#
# Mirrors process_reward_weighted_distill._epoch_inference_worker for the
# vLLM loop (rollout → segment → K-fold), but writes consistency_opd rollout
# records (work_idx, retry metadata, response_token_ids for Step 4 KL).
#
# Output schema (one JSONL line per work-set item):
# {
#   "work_idx"     : int,                  # index in the work-set passed in
#   "problem"      : str,
#   "gt_answer"    : str,
#   "retry_count"  : int,
#   "first_epoch"  : int,
#   "source"       : "pool" | "retry",
#   "response_text": str,                  # decoded student response
#   "response_token_ids": [int, ...],      # from vLLM, EXACT
#   "student_correct"   : bool,            # rollout boxed match
#   "student_answer"    : str | null,
#   "segments"     : [{start_char, end_char}, ...],  # raw segments
#   "ep_accs"      : {ep_idx -> float},    # K-fold acc per segment endpoint
#   "empty_acc"    : float,                # K-fold acc with empty prefix
# }
# A "skipped" record (rollout failed or response empty) carries
# "skipped": true and the routing later treats it like a low-consistency
# wrong response so the problem stays on the retry queue.

def _epoch_student_worker(cfg_path, out_path):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(cfg_path) as f:
        cfg = json.load(f)
    current_ckpt   = cfg["current_ckpt"]
    workset_jsonl  = cfg["workset_jsonl"]
    rollouts_jsonl = cfg["rollouts_jsonl_out"]
    shard_idx      = int(cfg.get("shard_idx", 0))
    n_shards            = max(1, int(cfg.get("n_shards", 1)))

    workset_all = load_jsonl(workset_jsonl)
    workset = [(i, rec) for i, rec in enumerate(workset_all)
               if i % n_shards == shard_idx]
    if not workset_all or not workset:
        write_jsonl(rollouts_jsonl, [])
        with open(out_path, "w") as f:
            json.dump({"n_rollouts": 0,
                       "shard_idx":  shard_idx,
                       "n_shards":   n_shards}, f)
        return

    print(f"\n[student worker {shard_idx + 1}/{n_shards}] processing "
          f"{len(workset)}/{len(workset_all)} workset items", flush=True)

    workset_orig_idx = [i for (i, _) in workset]
    workset_items    = [rec for (_, rec) in workset]
    workset = workset_items

    student_tok = AutoTokenizer.from_pretrained(current_ckpt, trust_remote_code=True)

    rollout_params = SamplingParams(
        n=1, temperature=ROLLOUT_TEMPERATURE,
        top_k=50, top_p=1.0, max_tokens=MAX_NEW_TOKENS,
    )
    kfold_params = SamplingParams(
        n=K_EVAL, temperature=EVAL_TEMPERATURE,
        top_k=50, top_p=1.0,
        max_tokens=MAX_ANSWER_TOKENS, stop=["<|im_end|>"],
    )

    llm = make_llm(current_ckpt, max_model_len=EPOCH_MAX_MODEL_LEN,
                   tensor_parallel_size=1)

    inf_bs    = INFERENCE_BATCH_SIZE if INFERENCE_BATCH_SIZE > 0 else len(workset)
    n_batches = max(1, (len(workset) + inf_bs - 1) // inf_bs)

    print(f"\n[Student worker] workset={len(workset)}  inf_batch={inf_bs}  "
          f"n_batches={n_batches}", flush=True)

    rollouts  = []
    n_correct = 0
    n_empty   = 0

    for b_idx in range(n_batches):
        b_start = b_idx * inf_bs
        b_end   = min(b_start + inf_bs, len(workset))
        batch   = workset[b_start:b_end]

        print(f"\n  [batch {b_idx + 1}/{n_batches}]  items {b_start}–{b_end - 1}",
              flush=True)

        # ── Step 1: student rollout ──────────────────────────────────────
        rollout_prompts = [
            build_filter_prompt(item["problem"], student_tok) for item in batch
        ]
        rollout_outs = safe_generate(
            llm, rollout_prompts, rollout_params, student_tok,
            EPOCH_MAX_MODEL_LEN, label="rollout",
        )
        responses = [
            (out.outputs[0].text if out is not None and out.outputs else "")
            for out in rollout_outs
        ]
        response_ids = [
            (list(out.outputs[0].token_ids)
             if out is not None and out.outputs and out.outputs[0].text else [])
            for out in rollout_outs
        ]

        # ── Step 2: segmentation ────────────────────────────────────────
        episodes_per = [
            segment_response_text(r) if r.strip() else [] for r in responses
        ]

        # ── Step 3: K-fold PR eval (empty prefix + each episode endpoint) ─
        kfold_prompts, kfold_meta = [], []
        for li, (item, resp, episodes) in enumerate(
            zip(batch, responses, episodes_per)
        ):
            if not resp.strip() or not episodes:
                continue
            kfold_prompts.append(
                build_episode_eval_prompt(item["problem"], "", student_tok))
            kfold_meta.append((li, -1))
            for ep_idx, (_sc, end_c) in enumerate(episodes):
                kfold_prompts.append(build_episode_eval_prompt(
                    item["problem"], resp[:end_c], student_tok))
                kfold_meta.append((li, ep_idx))

        kfold_outs = (
            safe_generate(
                llm, kfold_prompts, kfold_params, student_tok,
                EPOCH_MAX_MODEL_LEN, label="kfold",
            )
            if kfold_prompts else []
        )

        acc_per_li = {}
        for (li, ep_idx), out in zip(kfold_meta, kfold_outs):
            if out is None:
                continue
            gt = batch[li]["gt_answer"]
            corrects = [
                answers_match(extract_boxed_answer(o.text), gt) for o in out.outputs
            ]
            acc_per_li.setdefault(li, {})[ep_idx] = sum(corrects) / K_EVAL

        # ── Assemble per-item rollout record ─────────────────────────────
        for li, (item, resp, episodes, rids) in enumerate(zip(
                batch, responses, episodes_per, response_ids)):
            orig_idx = workset_orig_idx[b_start + li]
            rec_base = {
                "work_idx":    orig_idx,
                "_orig_idx":   orig_idx,
                "problem":     item["problem"],
                "gt_answer":   item["gt_answer"],
                "retry_count": int(item.get("retry_count", 0)),
                "first_epoch": int(item.get("first_epoch", -1)),
                "source":      item.get("source", "pool"),
            }

            if not resp.strip() or not rids or not episodes:
                rec_base.update({
                    "response_text":       resp or "",
                    "response_token_ids":  rids or [],
                    "student_correct":     False,
                    "student_answer":      None,
                    "segments":            [],
                    "ep_accs":             {},
                    "empty_acc":           0.0,
                    "skipped":             True,
                    "skip_reason":         "empty_or_no_episodes",
                })
                n_empty += 1
                rollouts.append(rec_base)
                continue

            li_accs   = acc_per_li.get(li, {})
            empty_acc = float(li_accs.get(-1, 0.0))
            ep_accs   = {int(k): float(v) for k, v in li_accs.items() if k != -1}

            student_answer = extract_boxed_answer(resp)
            correct        = answers_match(student_answer, item["gt_answer"])
            if correct:
                n_correct += 1

            rec_base.update({
                "response_text":       resp,
                "response_token_ids":  rids,
                "student_correct":     bool(correct),
                "student_answer":      student_answer,
                "segments":            [
                    {"start_char": int(sc), "end_char": int(ec)}
                    for sc, ec in episodes
                ],
                "ep_accs":             ep_accs,
                "empty_acc":           empty_acc,
                "skipped":             False,
            })
            rollouts.append(rec_base)

        print(f"    rollouts={len(rollout_prompts)}  "
              f"kfold_prompts={len(kfold_prompts)}  "
              f"rollouts_so_far={len(rollouts)}  "
              f"correct={n_correct}  empty={n_empty}", flush=True)

    destroy_llm(llm)
    write_jsonl(rollouts_jsonl, rollouts)

    with open(out_path, "w") as f:
        json.dump({
            "n_rollouts":  len(rollouts),
            "n_correct":   n_correct,
            "n_empty":     n_empty,
            "shard_idx":   shard_idx,
            "n_shards":    n_shards,
        }, f)
    print(f"\n[Student worker {shard_idx + 1}/{n_shards}] rollouts written: "
          f"{len(rollouts)}  (correct={n_correct}, empty={n_empty})",
          flush=True)


# ─── Step 4: KL worker (HF forward, per-token top-K support KL) ──────────────
#
# For each rollout, we forward `prompt_ids + response_token_ids` through both
# student and teacher in bf16. At each response position we form
#   S_t = TopK_q(c_t),
# gather student logits at those K indices, log_softmax both over the K-dim,
# and compute per-token KL_fwd = KL(q_hat || pi_hat), KL_rev = KL(pi_hat||q_hat).
#
# Then for each MERGED paragraph (monotonic-run merge over per-segment PR) we
# compute length-weighted normKL_fwd, normKL_rev, and the OPD-mix scalar
#     normKL_mix = beta * normKL_rev + (1 - beta) * normKL_fwd
# (matching the training loss). This is what gets fed into the consistency
# scorer.
#
# The merged paragraph PR is the cumulative ΔAcc across the run, not the
# per-segment delta of any single sub-segment.

def _build_merged_paragraphs(segments, ep_accs, empty_acc):
    """Given raw segments [(start_char, end_char)] and K-fold accs per
    segment endpoint, compute per-segment PR and then collapse monotonic
    runs into merged paragraphs. Returns list of dicts:
        { idx, ep_start, ep_end, n_sub, tok_start_char, tok_end_char,
          char_len, process_reward, acc_end, acc_prior, sub_indices }
    Sub-segments are indexed by their position in `segments`. Token offsets
    are NOT included here — those come from the KL worker which has token
    boundaries from its own retokenisation.
    """
    if not segments:
        return []

    prev_acc = 0.0
    per_seg = []
    for ep_idx, seg in enumerate(segments):
        acc = float(ep_accs.get(ep_idx, prev_acc))
        per_seg.append({
            "ep_idx":         ep_idx,
            "start_char":     int(seg["start_char"]),
            "end_char":       int(seg["end_char"]),
            "char_len":       int(seg["end_char"] - seg["start_char"]),
            "acc_rate":       acc,
            "process_reward": acc - prev_acc,
        })
        prev_acc = acc

    if len(per_seg) <= 1:
        # Single segment → single paragraph; PR = acc - empty_acc (or 0 if missing)
        s = per_seg[0]
        merged_pr = s["acc_rate"] - float(empty_acc or 0.0)
        return [{
            "idx":            0,
            "ep_start":       1,
            "ep_end":         1,
            "n_sub":          1,
            "sub_indices":    [0],
            "start_char":     s["start_char"],
            "end_char":       s["end_char"],
            "char_len":       s["char_len"],
            "process_reward": float(merged_pr),
            "acc_end":        s["acc_rate"],
            "acc_prior":      float(empty_acc or 0.0),
        }]

    # Monotonic-run merge on the per-segment PR sequence.
    merged_points = merge_monotonic_runs(per_seg)

    # merge_monotonic_runs (imported from process_reward_distill) keeps the
    # LAST point of each run but loses the run boundaries; reconstruct them.
    kept_ep_idx = [p["ep_idx"] for p in merged_points]
    run_starts  = [0] + [kept_ep_idx[j] + 1 for j in range(len(kept_ep_idx) - 1)]

    out = []
    for mi, (run_start_idx, run_end_idx) in enumerate(zip(run_starts, kept_ep_idx)):
        run = per_seg[run_start_idx: run_end_idx + 1]
        char_start = run[0]["start_char"]
        char_end   = run[-1]["end_char"]
        char_len   = char_end - char_start
        acc_end    = run[-1]["acc_rate"]
        acc_prior  = (per_seg[run_start_idx - 1]["acc_rate"]
                      if run_start_idx > 0 else float(empty_acc or 0.0))
        merged_pr  = acc_end - acc_prior

        out.append({
            "idx":            mi,
            "ep_start":       run_start_idx + 1,
            "ep_end":         run_end_idx + 1,
            "n_sub":          len(run),
            "sub_indices":    [s["ep_idx"] for s in run],
            "start_char":     int(char_start),
            "end_char":       int(char_end),
            "char_len":       int(char_len),
            "process_reward": float(merged_pr),
            "acc_end":        float(acc_end),
            "acc_prior":      float(acc_prior),
        })
    return out


def _kl_prepare_or_skip(rec, orig_idx, student_tok):
    """Return (None, skip_dict) or (prepared_dict, None) for KL forward."""
    if rec.get("skipped"):
        out = dict(rec)
        out["merged_paragraphs"] = []
        out["_orig_idx"] = orig_idx
        return None, out

    response_ids = rec.get("response_token_ids") or []
    if not response_ids or not rec.get("segments"):
        out = dict(rec)
        out["merged_paragraphs"] = []
        out["skipped"] = True
        out["skip_reason"] = rec.get("skip_reason", "no_response_or_segments")
        out["_orig_idx"] = orig_idx
        return None, out

    prompt_text = build_filter_prompt(rec["problem"], student_tok)
    prompt_ids = student_tok.encode(prompt_text, add_special_tokens=False)
    full_ids_list = prompt_ids + list(response_ids)
    seq_len = len(full_ids_list)
    prompt_len = len(prompt_ids)
    resp_len = seq_len - prompt_len

    if resp_len <= 1:
        out = dict(rec)
        out["merged_paragraphs"] = []
        out["skipped"] = True
        out["skip_reason"] = "resp_too_short"
        out["_orig_idx"] = orig_idx
        return None, out

    return {
        "rec": rec,
        "orig_idx": orig_idx,
        "full_ids_list": full_ids_list,
        "prompt_len": prompt_len,
        "seq_len": seq_len,
        "resp_len": resp_len,
        "response_ids": response_ids,
    }, None


def _kl_pad_id(tok):
    if tok.pad_token_id is not None:
        return tok.pad_token_id
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return 0


def _kl_build_padded_batch(batch_items, pad_id):
    """Left-pad variable-length sequences to [B, max_len] for one HF forward."""
    max_len = max(it["seq_len"] for it in batch_items)
    bsz = len(batch_items)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    offsets = []
    for i, it in enumerate(batch_items):
        seq_len = it["seq_len"]
        off = max_len - seq_len
        input_ids[i, off:] = torch.tensor(it["full_ids_list"], dtype=torch.long)
        offsets.append(off)
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask, offsets


def _kl_chunked_topk_kl(s_resp, t_resp, temp, kl_chunk, topk):
    """Per-token KL on the teacher's TOP-K support set + top-K cache rows.

    Implements the OPD top-K objective (see module docstring and the two
    reference derivations). For each response position c_t we take the
    teacher's top-K support

        S_t = TopK_q(c_t),

    gather BOTH the teacher and the student logits at those K vocab indices,
    and ``log_softmax`` each ONLY over the K-subset, i.e. we renormalise both
    distributions over S_t:

        pi_hat(v) = softmax_{u in S_t}( s_logit_u / T )[v]   (student)
        q_hat(v)  = softmax_{u in S_t}( t_logit_u / T )[v]   (teacher)

    and then form the truncated per-token KLs

        kl_fwd[t] = KL( q_hat  || pi_hat )   (forward, teacher || student)
        kl_rev[t] = KL( pi_hat || q_hat  )   (reverse, student || teacher).

    This replaces the previous full-vocabulary KL: it compares the two
    distributions only on the teacher-supported local region S_t, is far
    cheaper than a full-vocab softmax, and matches the training-time loss
    in ``OPDTrainer._topk_kl_cached`` (which renormalises over the same
    cached S_t). The cache rows store the teacher's top-K vocab indices and
    the *raw* teacher logits at those indices; Step 6 re-applies
    ``log_softmax(cache / T)`` over the K-subset.
    """
    import torch.nn.functional as F

    resp_len = s_resp.size(0)
    vocab = s_resp.size(-1)
    kk = min(topk, vocab)
    kl_device = s_resp.device
    kl_fwd_list = [0.0] * resp_len
    kl_rev_list = [0.0] * resp_len
    cache_idx_rows = torch.zeros((resp_len, topk), dtype=torch.int32)
    cache_lgt_rows = torch.zeros((resp_len, topk), dtype=torch.bfloat16)

    chunk_eff = kl_chunk
    if kl_device.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(kl_device.index or 0)
        row_bytes = max(vocab, 1) * 4 * 4
        chunk_eff = max(1, min(kl_chunk, int((free_b * 0.15) // row_bytes)))

    for st in range(0, resp_len, chunk_eff):
        ed = min(st + chunk_eff, resp_len)
        t_bl = t_resp[st:ed].float()
        s_bl = s_resp[st:ed].float()
        # S_t = teacher top-K support on this block (TopK_q).
        tk = torch.topk(t_bl, k=kk, dim=-1)
        # Gather teacher + student logits on S_t and renormalise over S_t only.
        t_top = tk.values / temp                          # [b, kk]
        s_top = s_bl.gather(-1, tk.indices) / temp        # [b, kk]
        t_log = F.log_softmax(t_top, dim=-1)              # log q_hat  over S_t
        s_log = F.log_softmax(s_top, dim=-1)              # log pi_hat over S_t
        kl_fwd = F.kl_div(s_log, t_log, log_target=True,
                          reduction="none").sum(-1)        # KL(q_hat || pi_hat)
        kl_rev = F.kl_div(t_log, s_log, log_target=True,
                          reduction="none").sum(-1)        # KL(pi_hat || q_hat)
        kl_fwd_cpu = kl_fwd.clamp_min(0.0).tolist()
        kl_rev_cpu = kl_rev.clamp_min(0.0).tolist()
        for j in range(ed - st):
            kl_fwd_list[st + j] = float(kl_fwd_cpu[j])
            kl_rev_list[st + j] = float(kl_rev_cpu[j])
        cache_idx_rows[st:ed, :kk] = tk.indices.to(torch.int32).cpu()
        cache_lgt_rows[st:ed, :kk] = tk.values.to(torch.bfloat16).cpu()
        del t_bl, s_bl, t_top, s_top, t_log, s_log, kl_fwd, kl_rev

    return kl_fwd_list, kl_rev_list, cache_idx_rows, cache_lgt_rows


def _kl_record_from_token_kls(
    item, kl_fwd_list, kl_rev_list, cache_idx_rows, cache_lgt_rows,
    topk, beta, student_tok, teacher_cache,
):
    """Assemble a kl.jsonl record (+ teacher top-K cache entry) from
    precomputed per-response-token KL values.

    Inputs come from the HF forward pass in ``_epoch_kl_worker`` reduced on
    the teacher's top-K support S_t (see ``_kl_chunked_topk_kl``):
      • ``kl_fwd_list[t]`` = KL(q_hat ‖ pi_hat) at response position t,
      • ``kl_rev_list[t]`` = KL(pi_hat ‖ q_hat) at response position t,
        where both distributions are renormalised over S_t = TopK_q(c_t)
        (NO miss: the student logits are exact at every teacher top-K index).
      • ``cache_idx_rows`` / ``cache_lgt_rows`` are [resp_len, topk] tensors
        holding the teacher's top-K vocab indices and the *raw teacher
        logits* at those indices. Step 6 consumes the cache via
        ``log_softmax(cache / T)`` over the K-subset, so raw logits and
        logprobs are interchangeable (constant shifts cancel in softmax).

    The per-paragraph normKL aggregation below is identical to the previous
    vLLM-top-K implementation; only the source of the per-token KL changed.
    """
    rec = item["rec"]
    orig_idx = item["orig_idx"]
    prompt_len = item["prompt_len"]
    resp_len = item["resp_len"]
    response_ids = item["response_ids"]

    if teacher_cache is not None:
        teacher_cache[str(orig_idx)] = {
            "topk_indices": cache_idx_rows.to(torch.int32),
            "topk_logits":  cache_lgt_rows.to(torch.bfloat16),
            "prompt_len":   int(prompt_len),
            "resp_len":     int(resp_len),
            "topk":         int(topk),
            "vocab_clip":   -1,
        }

    tok_char_starts = [0] * resp_len
    tok_char_ends   = [0] * resp_len
    pos = 0
    for t in range(resp_len):
        piece = student_tok.decode([response_ids[t]])
        tok_char_starts[t] = pos
        pos += len(piece)
        tok_char_ends[t] = pos

    ep_accs = {int(k): float(v) for k, v in rec.get("ep_accs", {}).items()}
    empty_acc = float(rec.get("empty_acc", 0.0))
    merged = _build_merged_paragraphs(rec["segments"], ep_accs, empty_acc)

    for para in merged:
        cs, ce = para["start_char"], para["end_char"]
        tok_indices = []
        for t in range(resp_len):
            if tok_char_starts[t] >= ce:
                break
            if tok_char_ends[t] <= cs:
                continue
            tok_indices.append(t)
        if not tok_indices:
            para["normKL_fwd"] = 0.0
            para["normKL_rev"] = 0.0
            para["normKL"]     = 0.0
            para["tok_start"]  = -1
            para["tok_end"]    = -1
            continue
        fwd = sum(kl_fwd_list[t] for t in tok_indices) / len(tok_indices)
        rev = sum(kl_rev_list[t] for t in tok_indices) / len(tok_indices)
        mix = beta * rev + (1.0 - beta) * fwd
        para["normKL_fwd"] = float(max(0.0, fwd))
        para["normKL_rev"] = float(max(0.0, rev))
        para["normKL"]     = float(max(0.0, mix))
        para["tok_start"]  = int(tok_indices[0])
        para["tok_end"]    = int(tok_indices[-1] + 1)

    rec_out = dict(rec)
    rec_out["prompt_len"]        = int(prompt_len)
    rec_out["resp_len"]          = int(resp_len)
    rec_out["merged_paragraphs"] = merged
    rec_out["kl_topk"]           = int(topk)
    rec_out["opd_beta_used"]     = float(beta)
    rec_out["_orig_idx"]         = orig_idx
    # Student logits are gathered exactly at the teacher's top-K indices, so
    # there is no top-K coverage miss (kept for schema stability).
    rec_out["_kl_miss_rate"]     = 0.0
    return rec_out


def _epoch_kl_worker(cfg_path, out_path):
    """HF forward-only per-token KL on teacher top-K support, one GPU slice
    per worker.

    Parallelism (this worker owns ``CUDA_VISIBLE_DEVICES`` = its slice):
      • teacher (large): sharded across the slice via ``device_map="auto"``
        (layer-split model parallelism). ``max_memory`` reserves headroom
        on cuda:0 for the student + activations.
      • student (small): replicated on cuda:0.
    Both run forward-only under ``no_grad`` (NO HF Trainer — there is no
    backward pass here). Data parallelism comes from the orchestrator
    launching ``N_GPUS // KL_GPUS_PER_WORKER`` of these workers on disjoint
    slices, each scoring its modulo-sharded subset of rollouts.

    Per response position we take the teacher's top-K support S_t = TopK_q,
    gather BOTH models' logits at those K indices, renormalise each over S_t
    and compute, on-GPU and chunked:
      • kl_fwd = KL(q_hat ‖ pi_hat),  kl_rev = KL(pi_hat ‖ q_hat),
    then aggregate to length-weighted ``normKL`` per merged paragraph for
    the consistency step. The same teacher top-K logits are cached for the
    Step-6 distillation loss (schema-compatible with the trainer), so the
    offline KL and the training loss compare distributions on identical S_t.

    Retokenisation:
      • prompt: tokenize(build_filter_prompt(problem)) — matches training
      • response: use ``response_token_ids`` directly from the rollout
        (EXACT, on-policy).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    with open(cfg_path) as f:
        cfg = json.load(f)
    student_ckpt      = cfg["student_ckpt"]
    teacher_path      = cfg["teacher_model"]
    rollouts_jsonl    = cfg["rollouts_jsonl"]
    kl_jsonl_out      = cfg["kl_jsonl_out"]
    teacher_cache_out = cfg.get("teacher_cache_out")
    beta              = float(cfg.get("opd_reverse_kl_beta", OPD_REVERSE_KL_BETA))
    temp              = float(cfg.get("opd_temp", OPD_TEMP))
    topk              = int(cfg.get("kl_topk", KL_TOPK))
    shard_idx         = int(cfg.get("shard_idx", 0))
    n_shards          = max(1, int(cfg.get("n_shards", 1)))

    rollouts_all = load_jsonl(rollouts_jsonl)
    # Modulo sharding for load balance: skipped records resolve nearly
    # instantly, so contiguous sharding would imbalance the workers.
    rollouts = [(i, rec) for i, rec in enumerate(rollouts_all)
                if i % n_shards == shard_idx]
    if not rollouts:
        write_jsonl(kl_jsonl_out, [])
        with open(out_path, "w") as f:
            json.dump({"n_with_kl": 0,
                       "shard_idx": shard_idx,
                       "n_shards":  n_shards}, f)
        return

    student_tok = AutoTokenizer.from_pretrained(student_ckpt, trust_remote_code=True)

    print(f"\n[KL worker {shard_idx + 1}/{n_shards}] processing "
          f"{len(rollouts)}/{len(rollouts_all)} rollouts  "
          f"top-K={topk}  beta={beta}  temp={temp}  mode=HF-teacher-topK",
          flush=True)

    # ── Skip resolution / pending prep ─────────────────────────────────
    kl_records = []
    n_skipped  = 0
    pending    = []
    for orig_idx, rec in rollouts:
        prepared, skip_out = _kl_prepare_or_skip(rec, orig_idx, student_tok)
        if skip_out is not None:
            n_skipped += 1
            kl_records.append(skip_out)
            continue
        pending.append(prepared)

    if not pending:
        write_jsonl(kl_jsonl_out, kl_records)
        with open(out_path, "w") as f:
            json.dump({"n_with_kl":  0,
                       "n_skipped":  n_skipped,
                       "n_total":    len(kl_records),
                       "n_cached":   0,
                       "shard_idx":  shard_idx,
                       "n_shards":   n_shards}, f)
        print(f"\n[KL worker {shard_idx + 1}/{n_shards}] no pending records "
              f"(all skipped: {n_skipped})", flush=True)
        return

    # ── Model loading: student first (replicated), then teacher sharded ──
    #
    # Memory strategy — robust across heterogeneous GPUs (40GB / 80GB H100 /
    # mixed) AND against *compute-time* OOM:
    #   1. Load the small student on cuda:0 FIRST, so it claims its footprint
    #      before we measure free memory.
    #   2. Size the teacher device_map budget from each GPU's *actual free
    #      bytes* × KL_MEM_FRACTION_CUDA0 on worker-local cuda:0 (student +
    #      KL aggregation) and KL_MEM_FRACTION on cuda:1+ — NOT from total
    #      memory. Using free bytes is size-agnostic and subtracts the student
    #      already on cuda:0. Response logits are kept on CPU after forward.
    # Note device_map="auto" only bounds *weight placement*; the headroom
    # above + the chunked bf16→fp32 KL math below (peak O(KL_FWD_CHUNK × V),
    # independent of resp_len) are what actually keep the forward from OOMing.
    if not torch.cuda.is_available():
        raise RuntimeError("KL worker requires CUDA")
    n_vis = torch.cuda.device_count()  # GPUs in this worker's slice
    student_device = "cuda:0"

    # Pick an attention kernel: flash-attn if importable, else SDPA (always
    # available and fine for forward-only scoring).
    attn_impl = os.environ.get("KL_ATTN_IMPL", "").strip()
    if not attn_impl:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"

    print(f"  [KL worker {shard_idx + 1}/{n_shards}] loading student "
          f"({os.path.basename(student_ckpt)}) replicated on {student_device}",
          flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        student_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(student_device)
    student.config.use_cache = False
    student.eval()

    # Teacher budget from real free memory per device × fraction. mem_get_info
    # is per worker-local index: cuda:0 is the *first GPU in this worker's
    # CUDA_VISIBLE_DEVICES slice* (not necessarily physical GPU 0). Student
    # lives on cuda:0; give teacher a smaller cap there and more on cuda:1+.
    def _kl_mem_frac(key: str, default: float) -> float:
        return min(max(float(os.environ.get(key, str(default))), 0.3), 0.97)

    frac_cuda0 = _kl_mem_frac("KL_MEM_FRACTION_CUDA0", 0.3)
    frac_other = _kl_mem_frac("KL_MEM_FRACTION", 0.85)
    torch.cuda.synchronize()
    max_memory  = {}
    budget_desc = []
    for i in range(n_vis):
        frac = frac_cuda0 if i == 0 else frac_other
        free_b, total_b = torch.cuda.mem_get_info(i)
        cap_mib = int((free_b * frac) // (1024 ** 2))
        max_memory[i] = f"{max(512, cap_mib)}MiB"
        budget_desc.append(
            f"cuda:{i} frac={frac:.2f} free={free_b / 1024 ** 3:.1f}/"
            f"{total_b / 1024 ** 3:.1f}GiB→cap={cap_mib / 1024:.1f}GiB")

    print(f"  [KL worker {shard_idx + 1}/{n_shards}] loading teacher "
          f"({os.path.basename(teacher_path)}) sharded over {n_vis} GPU(s) "
          f"device_map=auto  frac_cuda0={frac_cuda0}  frac_other={frac_other}  "
          f"attn={attn_impl}\n"
          f"      budget: {'  |  '.join(budget_desc)}", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map="auto",
        max_memory=max_memory if n_vis > 1 else None,
        low_cpu_mem_usage=True,
    )
    teacher.config.use_cache = False
    teacher.eval()
    teacher_in_device = teacher.get_input_embeddings().weight.device

    kl_chunk     = max(1, int(os.environ.get("KL_FWD_CHUNK", "128")))
    kl_on_cpu    = os.environ.get("KL_ON_CPU", "0").strip().lower() in (
        "1", "true", "yes", "on")
    score_batch  = max(1, int(os.environ.get("KL_SCORE_BATCH", "2")))
    pad_id       = _kl_pad_id(student_tok)
    kl_store     = "cpu" if kl_on_cpu else student_device

    print(f"  [KL worker {shard_idx + 1}/{n_shards}] scoring: "
          f"batch={score_batch}  kl_device={kl_store}  kl_chunk={kl_chunk}",
          flush=True)

    teacher_cache = {} if teacher_cache_out else None
    n_with_kl  = 0
    processed  = 0
    sum_normkl = 0.0

    for batch_start in range(0, len(pending), score_batch):
        batch_items = pending[batch_start: batch_start + score_batch]
        if len(batch_items) == 1:
            input_ids = torch.tensor(
                [batch_items[0]["full_ids_list"]], dtype=torch.long)
            attention_mask = None
            offsets = [0]
        else:
            input_ids, attention_mask, offsets = _kl_build_padded_batch(
                batch_items, pad_id)

        fwd_kw_s = {"input_ids": input_ids.to(student_device)}
        fwd_kw_t = {"input_ids": input_ids.to(teacher_in_device)}
        if attention_mask is not None:
            am_s = attention_mask.to(student_device)
            fwd_kw_s["attention_mask"] = am_s
            fwd_kw_t["attention_mask"] = attention_mask.to(teacher_in_device)

        with torch.no_grad():
            s_logits_b = student(**fwd_kw_s).logits          # [B, seq, Vs]
            t_logits_b = teacher(**fwd_kw_t).logits          # [B, seq, Vt]

        for bi, it in enumerate(batch_items):
            off = offsets[bi]
            lo = off + it["prompt_len"] - 1
            hi = lo + it["resp_len"]
            s_row = s_logits_b[bi]
            t_row = t_logits_b[bi]
            vocab = min(s_row.size(-1), t_row.size(-1))
            if kl_on_cpu:
                s_resp = s_row[lo:hi, :vocab].cpu()
                t_resp = t_row[lo:hi, :vocab].cpu()
            else:
                s_resp = s_row[lo:hi, :vocab]
                t_resp = t_row[lo:hi, :vocab].to(student_device)

            kl_fwd_list, kl_rev_list, cache_idx_rows, cache_lgt_rows = (
                _kl_chunked_topk_kl(
                    s_resp, t_resp, temp, kl_chunk, topk))
            del s_resp, t_resp

            rec_out = _kl_record_from_token_kls(
                it, kl_fwd_list, kl_rev_list, cache_idx_rows, cache_lgt_rows,
                topk, beta, student_tok, teacher_cache,
            )
            kl_records.append(rec_out)
            n_with_kl += 1
            processed += 1

            merged = rec_out["merged_paragraphs"]
            if merged:
                sum_normkl += sum(p["normKL"] for p in merged) / len(merged)

            if processed == 1 or processed % 50 == 0:
                n_para = len(merged)
                mu_mix = ((sum(p["normKL"] for p in merged) / n_para)
                          if n_para else 0.0)
                print(f"  [shard {shard_idx + 1}/{n_shards} | "
                      f"{processed:4d}/{len(pending)}]  "
                      f"resp_len={rec_out['resp_len']:4d}  paras={n_para:2d}  "
                      f"mean_normKL_mix={mu_mix:.4f}", flush=True)

        del s_logits_b, t_logits_b, input_ids
        if attention_mask is not None:
            del attention_mask
        if not kl_on_cpu:
            torch.cuda.empty_cache()

    write_jsonl(kl_jsonl_out, kl_records)

    n_cached = 0
    if teacher_cache is not None and teacher_cache_out:
        torch.save({
            "schema_version":    2,
            "kl_topk":           int(topk),
            "vocab_clip":        -1,
            "records":           teacher_cache,
            # Raw teacher logits (full-vocab forward), NOT logprobs. Step 6
            # renormalises via log_softmax(cache / T) over the K-subset, for
            # which logits and logprobs are interchangeable.
            "logprobs_in_cache": False,
        }, teacher_cache_out)
        n_cached = len(teacher_cache)
        print(f"  [KL worker {shard_idx + 1}/{n_shards}] teacher topk cache: "
              f"{n_cached} records → {teacher_cache_out}", flush=True)

    with open(out_path, "w") as f:
        json.dump({
            "n_with_kl":  n_with_kl,
            "n_skipped":  n_skipped,
            "n_total":    len(kl_records),
            "n_cached":   n_cached,
            "shard_idx":  shard_idx,
            "n_shards":   n_shards,
        }, f)
    mean_normkl = (sum_normkl / n_with_kl) if n_with_kl else 0.0
    print(f"\n[KL worker {shard_idx + 1}/{n_shards}] kl written: "
          f"{len(kl_records)}  (with_kl={n_with_kl}, skipped={n_skipped}, "
          f"teacher_cached={n_cached}, mean_normKL={mean_normkl:.4f})",
          flush=True)


# ─── Step 5: consistency + ranking + retry/discard + train.jsonl ─────────────
#
# This step runs in the orchestrator (CPU only, in-process — no subprocess
# needed). It consumes the KL worker's output and produces the train.jsonl
# the train worker consumes.

def _consistency_dispatch(kl_records, epoch_idx):
    """Returns:
        train_records   : list (to feed the train worker)
        new_retry_recs  : list (carry over to next epoch)
        dropped_recs    : list (max-retries-exhausted; permanently dropped)
        discarded_recs  : list (low-consistency BUT student correct; dropped)
        stats           : dict (summary numbers for logging)

    Selection logic:
      1. Per-response: compute per-paragraph inconsistency scores +
         response consistency ratio over neighbor PR-rank pairs.
      2. Eligible response = has >= MIN_MERGED_SEGS_PER_RESP paragraphs and
         has at least one defined PR pair (NaN consistency means we can't
         judge; those responses are NOT eligible).
      3. Global ranking of eligible responses by consistency descending;
         keep the top TOP_PERCENT_RESPONSES%.
      4. Kept responses are turned into train_records with mask info.
      5. Not-kept responses (and ineligible / skipped / empty ones):
            student_correct == True  → discarded
            student_correct == False → retry_count += 1
                                       if retry_count >= MAX_RETRIES: drop
                                       else: append to new_retry_recs
    """
    # ── Phase a: per-response scoring ─────────────────────────────────────
    annotated = []
    eligible_responses = []
    for rec in kl_records:
        merged = rec.get("merged_paragraphs") or []
        if rec.get("skipped") or len(merged) < max(2, MIN_MERGED_SEGS_PER_RESP):
            annotated.append({
                "rec":           rec,
                "eligible":      False,
                "consistency":   float("nan"),
                "n_pairs":       0,
                "inconsistency": [0.0] * len(merged),
            })
            continue
        inc, cons, n_pairs = _score_response_paragraphs(merged)
        if math.isnan(cons) or n_pairs == 0:
            annotated.append({
                "rec":           rec,
                "eligible":      False,
                "consistency":   cons,
                "n_pairs":       n_pairs,
                "inconsistency": inc,
            })
            continue
        annotated.append({
            "rec":           rec,
            "eligible":      True,
            "consistency":   cons,
            "n_pairs":       n_pairs,
            "inconsistency": inc,
        })
        eligible_responses.append(annotated[-1])

    # ── Phase b: global ranking (only over eligible responses) ────────────
    n_eligible = len(eligible_responses)
    n_keep = max(1, int(math.ceil(n_eligible * TOP_PERCENT_RESPONSES / 100.0))) \
             if n_eligible > 0 else 0

    # Higher consistency = better. Tie-break by more pairs (more robust signal).
    eligible_responses.sort(
        key=lambda x: (x["consistency"], x["n_pairs"]),
        reverse=True,
    )
    kept = set(id(a) for a in eligible_responses[:n_keep])

    # ── Phase c: build records ────────────────────────────────────────────
    train_records  = []
    new_retry_recs = []
    dropped_recs   = []
    discarded_recs = []

    n_masked_segs_total       = 0
    n_kept_paras_total        = 0
    n_kept_theoretical_masks  = 0   # ceil(n_segs × MASK%) summed over kept

    # Distribution trackers — used to verify the masking machinery actually
    # fires. If kept responses turn out to be uniformly inconsistency=0,
    # masking is dead code and we have effectively just "top-X% consistent
    # responses + vanilla OPD".
    para_inc_all       = []   # inconsistency of every paragraph (all annotated)
    para_inc_eligible  = []   # ... eligible responses only
    para_inc_kept      = []   # ... kept responses only
    eligible_cons      = []   # consistency ratio of every eligible response
    kept_cons          = []   # consistency ratio of every kept response
    kept_n_pos_inc     = []   # per-kept-response: # paragraphs with inc > 0
    kept_n_paragraphs  = []   # per-kept-response: # paragraphs
    n_kept_all_zero    = 0    # kept responses with inc == 0 everywhere
    n_kept_with_masks  = 0    # kept responses that actually got >=1 mask
    n_vanilla_fallback = 0

    for a in annotated:
        rec    = a["rec"]
        merged = rec.get("merged_paragraphs") or []
        is_kept = (a["eligible"] and id(a) in kept)

        # Distribution trackers (always-update arms).
        if merged:
            para_inc_all.extend(a["inconsistency"])
        if a["eligible"]:
            para_inc_eligible.extend(a["inconsistency"])
            if not math.isnan(a["consistency"]):
                eligible_cons.append(a["consistency"])

        if is_kept:
            # Mask up to top MASK_TOP_PERCENT_SEGMENTS% of paragraphs by
            # inconsistency score — but ONLY paragraphs that actually
            # violated something (score > 0). A response whose neighbor
            # pairs are all consistent has zero inconsistency everywhere,
            # so no paragraph is masked and the OPD loss falls back to its
            # vanilla form on the full response.
            n_segs   = len(merged)
            n_mask_t = int(math.ceil(n_segs * MASK_TOP_PERCENT_SEGMENTS / 100.0))
            n_mask_t = min(n_segs, n_mask_t)
            scored   = sorted(
                range(n_segs),
                key=lambda idx: a["inconsistency"][idx],
                reverse=True,
            )
            mask_indices = {
                idx for idx in scored[:n_mask_t]
                if a["inconsistency"][idx] > 0.0
            }

            n_masked_segs_total      += len(mask_indices)
            n_kept_paras_total       += n_segs
            n_kept_theoretical_masks += n_mask_t  # budget regardless of >0 filter

            # Per-kept-response distribution.
            inc_vec     = a["inconsistency"]
            n_pos_inc   = sum(1 for v in inc_vec if v > 0.0)
            para_inc_kept.extend(inc_vec)
            kept_cons.append(a["consistency"])
            kept_n_pos_inc.append(n_pos_inc)
            kept_n_paragraphs.append(n_segs)
            if n_pos_inc == 0:
                n_kept_all_zero += 1
            if len(mask_indices) > 0:
                n_kept_with_masks += 1

            # Build per-token weights aligned with response_token_ids.
            response_ids = rec.get("response_token_ids") or []
            resp_len     = len(response_ids)
            weights = [1.0] * resp_len
            mask_para_summaries = []
            for idx in range(n_segs):
                para = merged[idx]
                summary = {
                    "para_idx":      idx,
                    "ep_start":      para["ep_start"],
                    "ep_end":        para["ep_end"],
                    "process_reward": round(para["process_reward"], 6),
                    "normKL":        round(para.get("normKL", 0.0), 6),
                    "inconsistency": round(a["inconsistency"][idx], 6),
                    "masked":        bool(idx in mask_indices),
                }
                mask_para_summaries.append(summary)
                if idx in mask_indices:
                    ts = max(0, int(para.get("tok_start", -1)))
                    te = min(resp_len, int(para.get("tok_end", -1)))
                    if ts >= 0 and te > ts:
                        for j in range(ts, te):
                            weights[j] = 0.0

            # If everything ended up zero-weighted (degenerate), drop.
            if all(w == 0.0 for w in weights):
                # Treat as ineligible; route by correctness.
                pass
            else:
                train_records.append({
                    "type":               "consistency_opd",
                    "problem":            rec["problem"],
                    "gt_answer":          rec["gt_answer"],
                    "response_token_ids": list(response_ids),
                    "response_weights":   weights,
                    # cache_id keys this record into the epoch-level teacher
                    # top-K cache (built in Step 4). Trainer reads logits by
                    # cache_id and never re-forwards the teacher.
                    "cache_id":           str(rec.get("_orig_idx", "")),
                    "n_paragraphs":       n_segs,
                    "n_masked":           len(mask_indices),
                    "consistency":        round(a["consistency"], 6),
                    "n_pairs":            a["n_pairs"],
                    "paragraphs":         mask_para_summaries,
                    "retry_count":        int(rec.get("retry_count", 0)),
                    "source":             rec.get("source", "pool"),
                })
                continue

        # Too few paragraphs → vanilla OPD (not retry/discard).
        is_ineligible_too_few_paras = (
            not a["eligible"]
            and not rec.get("skipped")
            and bool(rec.get("response_token_ids"))
            and len(merged) < max(2, MIN_MERGED_SEGS_PER_RESP)
        )
        if is_ineligible_too_few_paras:
            response_ids = list(rec["response_token_ids"])
            train_records.append({
                "type":               "vanilla_opd_fallback",
                "problem":            rec["problem"],
                "gt_answer":          rec["gt_answer"],
                "response_token_ids": response_ids,
                "response_weights":   [1.0] * len(response_ids),
                "cache_id":           str(rec.get("_orig_idx", "")),
                "n_paragraphs":       len(merged),
                "n_masked":           0,
                "consistency":        None,
                "n_pairs":            0,
                "paragraphs":         [],
                "retry_count":        int(rec.get("retry_count", 0)),
                "source":             rec.get("source", "pool"),
                "fallback_reason":    "too_few_merged_paragraphs",
            })
            n_vanilla_fallback += 1
            continue

        # Not kept (or degenerate): route by student correctness.
        problem  = rec.get("problem")
        gt       = rec.get("gt_answer")
        correct  = bool(rec.get("student_correct", False))
        rc_prev  = int(rec.get("retry_count", 0))
        first_ep = int(rec.get("first_epoch", -1))

        if correct:
            discarded_recs.append({
                "problem":     problem,
                "gt_answer":   gt,
                "epoch":       epoch_idx,
                "reason":      "low_consistency_but_student_correct",
                "consistency": (round(a["consistency"], 6)
                                if not math.isnan(a["consistency"]) else None),
            })
            continue

        rc_new = rc_prev + 1
        retry_payload = {
            "problem":      problem,
            "gt_answer":    gt,
            "retry_count":  rc_new,
            "first_epoch":  first_ep if first_ep >= 0 else epoch_idx,
            "last_epoch":   epoch_idx,
            "last_consistency": (round(a["consistency"], 6)
                                 if not math.isnan(a["consistency"]) else None),
            "n_merged":     len(merged),
        }
        if rc_new >= MAX_RETRIES:
            retry_payload["reason"] = "max_retries_exhausted"
            dropped_recs.append(retry_payload)
        else:
            new_retry_recs.append(retry_payload)

    # ── Phase d: distribution summaries ───────────────────────────────────
    n_train_total = len(train_records)
    n_kept        = n_train_total - n_vanilla_fallback
    util_pct = (100.0 * n_masked_segs_total / n_kept_theoretical_masks
                if n_kept_theoretical_masks > 0 else 0.0)

    # Per-paragraph inconsistency — overall, and within kept responses,
    # restricted to STRICTLY POSITIVE values (since 0 is the boring case).
    def _pos(values):
        return [v for v in values if v > 0.0]

    stats = {
        # Headline counts
        "n_kl_records":            len(kl_records),
        "n_eligible":              n_eligible,
        "n_kept":                  n_kept,
        "n_vanilla_fallback":      n_vanilla_fallback,
        "n_train_total":           n_train_total,
        "n_retry_new":             len(new_retry_recs),
        "n_dropped":               len(dropped_recs),
        "n_discarded_correct":     len(discarded_recs),

        # Masking budget vs actual
        "n_paragraphs_total":          n_kept_paras_total,
        "n_paragraphs_masked":         n_masked_segs_total,
        "n_paragraphs_mask_budget":    n_kept_theoretical_masks,
        "mask_budget_utilisation":     round(util_pct, 2),

        # Sanity-check distributions: response level (kept)
        "n_kept_with_any_violation":   n_kept - n_kept_all_zero,
        "n_kept_all_consistent":       n_kept_all_zero,
        "n_kept_with_actual_masks":    n_kept_with_masks,
        "kept_consistency":            _dist_summary(kept_cons),
        "eligible_consistency":        _dist_summary(eligible_cons),

        # Sanity-check distributions: paragraph level
        "para_inc_all":                _dist_summary(para_inc_all),
        "para_inc_all_pos":            _dist_summary(_pos(para_inc_all)),
        "para_inc_eligible_pos":       _dist_summary(_pos(para_inc_eligible)),
        "para_inc_kept_pos":           _dist_summary(_pos(para_inc_kept)),
        "frac_kept_paragraphs_zero":   round(
            (sum(1 for v in para_inc_kept if v == 0.0) /
             len(para_inc_kept)) if para_inc_kept else 0.0,
            4,
        ),
    }
    return train_records, new_retry_recs, dropped_recs, discarded_recs, stats


# ─── Step 6: training subprocess (OPD with top-K KL + per-token weights) ─────

def _copy_args_json(src_dir, dst_dir):
    src_args = Path(src_dir) / "args.json"
    if src_args.exists():
        import shutil
        dst_args = Path(dst_dir) / "args.json"
        shutil.copy2(str(src_args), str(dst_args))
        return True
    return False


def _train_worker(cfg_path):
    """OPD training with:
      • per-token weight tensor that ZEROES out masked segments
      • KL computed on the teacher's top-K support set per response position
      • β-mix between reverse and forward KL (default β=0.7, leaning reverse)

    A "consistency_opd" record carries:
      • problem
      • response_token_ids       (from vLLM rollout, EXACT)
      • response_weights         (same length as response_token_ids;
                                  1.0 for trained tokens, 0.0 for masked)
    The encoder concatenates prompt_ids (chat-templated) + response_ids and
    sets labels = [-100 for prompt] + response_ids (masked positions keep
    their token id; weights handle the loss zeroing).
    """
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    with open(cfg_path) as f:
        cfg = json.load(f)
    student_ckpt    = cfg["student_ckpt"]
    train_jsonl     = cfg["train_jsonl"]
    output_dir      = cfg["output_dir"]
    epoch_idx       = cfg["epoch_idx"]
    n_epochs        = cfg["n_epochs"]
    lr              = cfg["lr"]
    max_length      = cfg["max_length"]
    batch_size      = cfg["batch_size"]
    grad_accum      = cfg["grad_accum"]
    warmup_ratio    = cfg["warmup_ratio"]
    opd_beta        = float(cfg.get("opd_reverse_kl_beta", OPD_REVERSE_KL_BETA))
    opd_temp        = float(cfg.get("opd_temp", OPD_TEMP))
    kl_topk         = int(cfg.get("kl_topk", KL_TOPK))
    teacher_cache_in = cfg.get("teacher_cache_in")  # path to .pt built in Step 4

    is_main    = int(os.environ.get("RANK", "0")) == 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    tokenizer = AutoTokenizer.from_pretrained(student_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

    # ── Load teacher top-K cache (built in Step 4). Each rank loads its own
    # copy. With ~1.5 MB per response × ~5K kept responses ≈ 7-8 GB / rank
    # CPU RAM in typical runs — affordable on most servers.
    if not teacher_cache_in or not Path(teacher_cache_in).exists():
        raise RuntimeError(
            f"teacher top-K cache missing at {teacher_cache_in!r}. "
            "Step 4 must produce the cache before Step 6 can run.")
    if is_main:
        print(f"  [train worker] loading teacher topk cache from {teacher_cache_in}",
              flush=True)
    _tcache_blob = torch.load(teacher_cache_in, map_location="cpu")
    teacher_cache = _tcache_blob.get("records", {})
    cache_topk    = int(_tcache_blob.get("kl_topk", kl_topk))
    cache_vclip   = int(_tcache_blob.get("vocab_clip", -1))
    if cache_topk != kl_topk:
        if is_main:
            print(f"  [train worker] WARNING: cache topk={cache_topk} "
                  f"!= configured KL_TOPK={kl_topk}; using cache topk",
                  flush=True)
        kl_topk = cache_topk
    if is_main:
        print(f"  [train worker] cache: {len(teacher_cache)} records  "
              f"K={cache_topk}  vocab_clip={cache_vclip}", flush=True)

    class ConsistencyDataset(Dataset):
        def __init__(self, path):
            raw = load_jsonl(path)
            self.examples = []
            n_skip_other = 0
            n_skip_long  = 0
            n_skip_empty = 0
            for rec in raw:
                ex = self._encode(rec)
                if ex is None:
                    n_skip_other += 1
                elif ex == "too_long":
                    n_skip_long += 1
                elif ex == "all_zero":
                    n_skip_empty += 1
                else:
                    self.examples.append(ex)
            if is_main:
                # Token-level mask diagnostic: of all RESPONSE tokens
                # (label != -100), how many are zero-weighted?
                tot_resp_tokens   = 0
                tot_masked_tokens = 0
                for ex in self.examples:
                    for lab, w in zip(ex["labels"], ex["weights"]):
                        if lab == -100:
                            continue
                        tot_resp_tokens += 1
                        if w == 0.0:
                            tot_masked_tokens += 1
                pct_masked = (100.0 * tot_masked_tokens / tot_resp_tokens
                              if tot_resp_tokens > 0 else 0.0)
                print(f"  [dataset] kept {len(self.examples)} / {len(raw)}  "
                      f"(skip_long={n_skip_long}, skip_empty={n_skip_empty}, "
                      f"skip_other={n_skip_other})", flush=True)
                print(f"  [dataset] response-token mask: "
                      f"{tot_masked_tokens} / {tot_resp_tokens} "
                      f"({pct_masked:.2f}% of labeled tokens are weight=0)",
                      flush=True)
                if tot_masked_tokens == 0 and tot_resp_tokens > 0:
                    print("  [dataset] WARNING: 0 masked tokens this epoch — "
                          "OPD loss is uniform across all response tokens "
                          "(masking inactive).", flush=True)

        def _encode(self, rec):
            problem      = rec.get("problem", "")
            response_ids = rec.get("response_token_ids") or []
            weights_resp = rec.get("response_weights") or []
            cache_id     = str(rec.get("cache_id", ""))
            if not problem or not response_ids or not weights_resp:
                return None
            if len(weights_resp) != len(response_ids):
                return None
            tk = teacher_cache.get(cache_id)
            if tk is None:
                # Step 4 didn't cache this rollout (e.g. it was skipped or
                # the cache key mismatches). Without teacher topk we can't
                # compute the KL loss → drop the example.
                return None
            t_idx = tk["topk_indices"]                 # [resp_len_cache, K] int32 vocab idx
            # NOTE: field name is historical — in vLLM cache mode (schema v2)
            # this tensor stores top-K logprobs, not raw logits. The OPD KL
            # math (log_softmax over the K-subset) is identical either way.
            t_lgt = tk["topk_logits"]                  # [resp_len_cache, K] bfloat16 (logprobs)
            resp_len_cache = int(tk.get("resp_len", t_idx.shape[0]))
            if resp_len_cache != len(response_ids):
                # Sanity: cache must align 1:1 with response_token_ids.
                return None

            prompt_text = build_filter_prompt(problem, tokenizer)
            prompt_ids  = tokenizer.encode(prompt_text, add_special_tokens=False)

            # The im_end terminator is appended for the model to see during
            # forward, but it has NO supervision (label=-100, weight=0) — the
            # teacher cache only covers the response tokens themselves, and
            # OPD on the EOS position is not informative anyway.
            resp_ids    = list(response_ids) + [im_end_id]

            total_len = len(prompt_ids) + len(resp_ids)
            if total_len > max_length:
                return "too_long"

            input_ids = prompt_ids + resp_ids
            labels    = (
                [-100] * len(prompt_ids)
                + list(response_ids)
                + [-100]  # im_end: no supervision
            )
            weights   = (
                [0.0] * len(prompt_ids)
                + [float(w) for w in weights_resp]
                + [0.0]   # im_end: no loss
            )

            if all(w == 0.0 for w in weights):
                return "all_zero"
            return {
                "input_ids": input_ids,
                "labels":    labels,
                "weights":   weights,
                "prompt_len": int(len(prompt_ids)),
                "resp_len":   int(len(response_ids)),
                "teacher_topk_indices": t_idx,    # tensor [resp_len, K] vocab idx
                "teacher_topk_logits":  t_lgt,    # tensor [resp_len, K] bf16 logprobs
                                                  # (legacy field name; see cache note above)
            }

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return self.examples[i]

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, lbl, wts, attn = [], [], [], []
        B = len(batch)
        K = int(batch[0]["teacher_topk_indices"].shape[1])
        sh_max = max(1, max_len - 1)  # shifted length used by compute_loss

        # Pre-allocate per-batch teacher buffers in shifted coords:
        #   shifted position p in [0..max_len-2] predicts label at p+1.
        #   For a sample with prompt_len P and resp_len L, teacher_topk[i]
        #   (i in [0..L-1]) aligns with shifted position p = P - 1 + i,
        #   which predicts response_ids[i].
        teacher_idx_buf = torch.zeros(B, sh_max, K, dtype=torch.long)
        teacher_lgt_buf = torch.zeros(B, sh_max, K, dtype=torch.float32)

        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            pad = max_len - n
            ids.append(b["input_ids"] + [pad_id] * pad)
            lbl.append(b["labels"]    + [-100]   * pad)
            wts.append(b["weights"]   + [0.0]    * pad)
            attn.append([1] * n + [0] * pad)

            P  = int(b["prompt_len"])
            L  = int(b["resp_len"])
            start = max(0, P - 1)
            end   = min(sh_max, start + L)
            valid = end - start
            if valid > 0:
                teacher_idx_buf[i, start:end, :] = (
                    b["teacher_topk_indices"][:valid].to(torch.long)
                )
                teacher_lgt_buf[i, start:end, :] = (
                    b["teacher_topk_logits"][:valid].to(torch.float32)
                )
        return {
            "input_ids":            torch.tensor(ids,  dtype=torch.long),
            "attention_mask":       torch.tensor(attn, dtype=torch.long),
            "labels":               torch.tensor(lbl,  dtype=torch.long),
            "weights":              torch.tensor(wts,  dtype=torch.float32),
            "teacher_topk_indices": teacher_idx_buf,
            "teacher_topk_logits":  teacher_lgt_buf,
        }

    class OPDConsistencyTrainer(Trainer):
        """OPD loss with top-K teacher support (CACHED) and per-token weights.

        Teacher forward is performed ONCE per epoch during Step 4 (vLLM
        prompt_logprobs scoring) and cached to disk. At training time we
        never load the teacher: each batch carries
            teacher_topk_indices  [B, T-1, K]  : vocab indices on teacher support
            teacher_topk_logits   [B, T-1, K]  : teacher top-K logprobs at those
                                                 indices (field name is legacy;
                                                 vLLM cache stores logprobs —
                                                 see _epoch_kl_worker docstring)
        aligned with the shifted prediction positions.

        For an active (label != -100) position with student logits s_t ∈ R^V:
            S_t  = teacher_topk_indices[t]              (K vocab indices)
            s_t' = s_t[S_t] / T                          (gather + temp scale)
            t_t' = teacher_topk_logits[t] / T            (cached logprobs / T)
            ŝ_t  = softmax(s_t')
            t̂_t  = softmax(t_t')
            L    = β·KL(ŝ‖t̂) + (1−β)·KL(t̂‖ŝ), weighted by w_t.

        Math note: ``log_softmax(logprobs / T) ≡ log_softmax(logits / T)``
        on the K-subset because the global ``logsumexp`` shift cancels in
        softmax, so the loss is unchanged whether the cache stores raw
        teacher logits (schema v1, legacy HF path) or vLLM logprobs
        (schema v2, current).
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.opd_beta = OPD_REVERSE_KL_BETA
            self.opd_temp = OPD_TEMP
            self.kl_topk  = KL_TOPK

        def _topk_kl_cached(self, s_logits_flat, t_idx_flat, t_lgt_flat,
                            wts_flat, w_sum):
            """Compute weighted KL using student logits + cached teacher topk.

            Args:
                s_logits_flat:  [N, V]  student logits at active positions
                t_idx_flat:     [N, K]  teacher top-K vocab indices
                t_lgt_flat:     [N, K]  teacher top-K teacher score values at
                                        those indices (will be /T'd before
                                        log_softmax). In schema v2 (vLLM)
                                        these are logprobs; in legacy v1 they
                                        were raw logits — both yield the same
                                        log_softmax over the K-subset.
                wts_flat:       [N]     per-position weights
                w_sum:          scalar  sum of weights, clamped >= 1
            """
            T = self.opd_temp
            beta = self.opd_beta
            chunk = 1024
            n = s_logits_flat.size(0)
            total = s_logits_flat.new_zeros(())
            for st in range(0, n, chunk):
                ed = min(st + chunk, n)
                s_top = s_logits_flat[st:ed].gather(-1, t_idx_flat[st:ed]) / T
                t_top = t_lgt_flat[st:ed].to(s_top.dtype) / T

                s_log = F.log_softmax(s_top, dim=-1)
                t_log = F.log_softmax(t_top, dim=-1)

                if beta >= 1.0:
                    kl = F.kl_div(input=t_log, target=s_log,
                                  log_target=True, reduction="none").sum(-1)
                elif beta <= 0.0:
                    kl = F.kl_div(input=s_log, target=t_log,
                                  log_target=True, reduction="none").sum(-1)
                else:
                    rkl = F.kl_div(input=t_log, target=s_log,
                                   log_target=True, reduction="none").sum(-1)
                    fkl = F.kl_div(input=s_log, target=t_log,
                                   log_target=True, reduction="none").sum(-1)
                    kl = beta * rkl + (1.0 - beta) * fkl

                total = total + (kl * wts_flat[st:ed]).sum()
            return total / w_sum

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            wts        = inputs.pop("weights")
            t_idx_full = inputs.pop("teacher_topk_indices")   # [B, T-1, K]
            t_lgt_full = inputs.pop("teacher_topk_logits")    # [B, T-1, K]
            labels     = inputs["labels"]
            ids        = inputs["input_ids"]
            attn       = inputs["attention_mask"]

            out    = model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1, :].contiguous()
            sh_lbl = labels[:, 1:].contiguous()
            sh_wts = wts[:, 1:].contiguous().to(logits.dtype)

            mask = sh_lbl != -100
            if mask.sum() == 0:
                loss = logits.new_zeros(())
                if return_outputs:
                    return loss, out
                return loss

            # Move teacher buffers onto the same device as the model output.
            t_idx_full = t_idx_full.to(logits.device, non_blocking=True)
            t_lgt_full = t_lgt_full.to(logits.device, non_blocking=True)

            logits_flat = logits[mask]                    # [N_act, V]
            wts_flat    = sh_wts[mask]                    # [N_act]
            t_idx_flat  = t_idx_full[mask]                # [N_act, K]
            t_lgt_flat  = t_lgt_full[mask]                # [N_act, K]
            w_sum       = wts_flat.sum().clamp_min(1.0)

            # Vocab safety: clamp teacher indices to student's vocab. The
            # vLLM cache (schema v2) does not pre-clip vocab — vocab_clip is
            # -1 — because student & teacher share the same tokenizer in
            # standard configs. This guard only fires on mismatched
            # tokenizers / checkpoints.
            Vs = logits_flat.size(-1)
            if t_idx_flat.max().item() >= Vs:
                t_idx_flat = t_idx_flat.clamp_max(Vs - 1)

            loss = self._topk_kl_cached(
                logits_flat, t_idx_flat, t_lgt_flat, wts_flat, w_sum
            )

            if return_outputs:
                return loss, out
            return loss

    if is_main:
        if opd_beta >= 0.99:
            _label = "pure reverse-KL"
        elif opd_beta <= 0.01:
            _label = "pure forward-KL"
        elif abs(opd_beta - 0.5) < 1e-6:
            _label = "balanced (≈JSD)"
        else:
            _label = f"{opd_beta:.2f} reverse + {1.0 - opd_beta:.2f} forward"
        print(f"\n[train worker] epoch {epoch_idx + 1}/{n_epochs}  "
              f"mode=opd  lr={lr:.3e}  max_len={max_length}  "
              f"bs={batch_size}  ga={grad_accum}", flush=True)
        print(f"  [opd] β={opd_beta} ({_label})  T={opd_temp}  topK={kl_topk}",
              flush=True)
        print(f"  student_ckpt = {student_ckpt}", flush=True)
        print(f"  teacher      = (cached top-K from Step 4 — NOT loaded on GPU)",
              flush=True)
        print(f"  cache        = {teacher_cache_in}  "
              f"({len(teacher_cache)} records)", flush=True)
        print(f"  train_jsonl  = {train_jsonl}", flush=True)
        print(f"  output_dir   = {output_dir}", flush=True)

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"

    student = AutoModelForCausalLM.from_pretrained(
        student_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    student.config.use_cache = False

    train_dataset = ConsistencyDataset(train_jsonl)
    if len(train_dataset) == 0:
        if is_main:
            print("  [train worker] no training records — saving unchanged ckpt",
                  flush=True)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            _copy_args_json(student_ckpt, output_dir)
        return

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=SEED + epoch_idx,
    )

    import inspect as _inspect
    _trainer_params = _inspect.signature(Trainer.__init__).parameters
    _tok_kwargs = (
        {"processing_class": tokenizer}
        if "processing_class" in _trainer_params
        else {"tokenizer": tokenizer}
    )
    trainer = OPDConsistencyTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
        **_tok_kwargs,
    )
    trainer.opd_beta = opd_beta
    trainer.opd_temp = opd_temp
    trainer.kl_topk  = kl_topk

    # No teacher model on GPU — KL is computed from the pre-computed top-K
    # cache loaded in the dataset above. This is what lets us train with
    # arbitrarily large teachers (e.g. 32B) without consuming GPU memory at
    # train time.

    trainer.train()

    trainer.save_model(output_dir)
    if is_main:
        tokenizer.save_pretrained(output_dir)
        _copy_args_json(student_ckpt, output_dir)
        print(f"  [train worker] saved → {output_dir}", flush=True)

    del trainer, student
    # Free the cache from RAM (teacher_cache holds tensors with ref-cycles
    # through the trainer closure).
    teacher_cache.clear()
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()


# ─── Step 7: post-training held-out eval ─────────────────────────────────────

def _epoch_eval_worker(cfg_path, out_path):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(cfg_path) as f:
        cfg = json.load(f)
    eval_ckpt     = cfg["eval_ckpt"]
    epoch_idx     = cfg["epoch_idx"]
    n_epochs      = cfg["n_epochs"]
    shard_idx     = int(cfg.get("shard_idx", 0))
    n_shards      = max(1, int(cfg.get("n_shards", 1)))
    is_last_epoch = (epoch_idx + 1 >= n_epochs)

    student_tok   = AutoTokenizer.from_pretrained(eval_ckpt, trust_remote_code=True)
    filter_params = SamplingParams(n=1, temperature=0, max_tokens=MAX_NEW_TOKENS)

    llm = make_llm(eval_ckpt, max_model_len=EPOCH_MAX_MODEL_LEN,
                   tensor_parallel_size=1)

    def _run_eval(label, indexed_records, skip_idx=None):
        skip = set(skip_idx or [])
        rec_by_idx = {gidx: rec for gidx, rec in indexed_records}
        prompts, meta = [], []
        for gidx, rec in indexed_records:
            if gidx in skip:
                continue
            prompts.append(build_filter_prompt(rec["problem"], student_tok))
            meta.append(gidx)
        if not prompts:
            return 0, len(indexed_records), []
        outs = safe_generate(llm, prompts, filter_params, student_tok,
                             EPOCH_MAX_MODEL_LEN, label=label)
        newly = []
        n_correct = 0
        for gidx, out in zip(meta, outs):
            if out is None:
                continue
            if answers_match(extract_boxed_answer(out.outputs[0].text),
                             rec_by_idx[gidx]["gt_answer"]):
                newly.append(gidx)
                n_correct += 1
        return n_correct, len(indexed_records), newly

    print(f"\n[Step 7 shard {shard_idx + 1}/{n_shards}] Held-out evaluations …",
          flush=True)
    test_records = load_jsonl(TEST_SET_JSONL) if TEST_SET_JSONL.exists() else []

    should_test_eval = (
        TEST_EVAL_EVERY > 0
        and ((epoch_idx + 1) % TEST_EVAL_EVERY == 0 or is_last_epoch)
    )
    test_correct_shard = test_total_shard = 0

    if test_records and should_test_eval:
        if TEST_EVAL_SAMPLES > 0 and TEST_EVAL_SAMPLES < len(test_records):
            rng = random.Random(SEED + epoch_idx)
            indices = rng.sample(range(len(test_records)), TEST_EVAL_SAMPLES)
            test_records_use = [(i, test_records[i]) for i in indices]
            tag = f"  [subsampled {TEST_EVAL_SAMPLES}/{len(test_records)}]"
        else:
            test_records_use = list(enumerate(test_records))
            tag = ""
        my_test = [(gidx, rec) for k, (gidx, rec) in enumerate(test_records_use)
                   if k % n_shards == shard_idx]
        test_correct_shard, test_total_shard, _ = _run_eval(
            "eval_test", my_test,
        )
        print(f"  [shard {shard_idx + 1}/{n_shards}] test:  "
              f"{test_correct_shard}/{test_total_shard}{tag}", flush=True)
    elif test_records:
        print(f"  test:  skipped (TEST_EVAL_EVERY={TEST_EVAL_EVERY}, "
              f"epoch={epoch_idx + 1}/{n_epochs})", flush=True)
    else:
        print(f"  test:  no test_set.jsonl found, skipping.", flush=True)

    destroy_llm(llm)

    eval_info = {
        "test_correct":  test_correct_shard,
        "test_total":    test_total_shard,
        "shard_idx":     shard_idx,
        "n_shards":      n_shards,
    }
    with open(out_path, "w") as f:
        json.dump(eval_info, f)
    print(f"[eval worker {shard_idx + 1}/{n_shards}] Done → {out_path}",
          flush=True)


# ─── Orchestrator ────────────────────────────────────────────────────────────

def run_epoch(epoch_idx, current_ckpt, pool):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch_idx + 1}/{N_EPOCHS}  |  "
          f"ckpt={os.path.basename(current_ckpt)}")
    print(f"{'='*60}", flush=True)

    epoch_tag       = f"epoch_{epoch_idx:03d}"
    workset_jsonl   = OUTPUT_DIR / f"{epoch_tag}_workset.jsonl"
    rollouts_jsonl  = OUTPUT_DIR / f"{epoch_tag}_rollouts.jsonl"
    kl_jsonl        = OUTPUT_DIR / f"{epoch_tag}_kl.jsonl"
    train_jsonl     = OUTPUT_DIR / f"{epoch_tag}_train.jsonl"
    stats_json      = OUTPUT_DIR / f"{epoch_tag}_dispatch_stats.json"
    teacher_cache_pt = OUTPUT_DIR / f"{epoch_tag}_teacher_topk_cache.pt"

    student_cfg     = OUTPUT_DIR / f"{epoch_tag}_student_cfg.json"
    student_out     = OUTPUT_DIR / f"{epoch_tag}_student_out.json"
    kl_cfg          = OUTPUT_DIR / f"{epoch_tag}_kl_cfg.json"
    kl_out          = OUTPUT_DIR / f"{epoch_tag}_kl_out.json"

    # ── Build work-set: pool + retry backlog ──────────────────────────────
    retry_pool = _load_retry_pool()
    workset = _build_workset_for_epoch(pool, retry_pool, epoch_idx)
    print(f"\n[orchestrator] workset = {len(workset)}  "
          f"(pool={len(pool)}, retry_backlog={len(retry_pool)})", flush=True)

    if not workset_jsonl.exists():
        write_jsonl(workset_jsonl, workset)

    # ── Steps 1–3: student worker (rollout + segments + K-fold) ──────────
    # One vLLM subprocess per visible GPU (DP, tp=1), same inference pattern
    # as process_reward_weighted_distill._epoch_inference_worker.
    if rollouts_jsonl.exists() and student_out.exists():
        print(f"\n[Steps 1-3] Cached rollouts found — skipping.", flush=True)
    else:
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if parent_cvd:
            physical_ids = [s.strip() for s in parent_cvd.split(",") if s.strip()]
        else:
            physical_ids = [str(i) for i in range(N_GPUS)]
        n_visible = len(physical_ids)
        gpw       = 1
        n_workers = max(1, n_visible)

        print(f"\n[Steps 1-3] Spawning {n_workers} student worker(s)  "
              f"(visible_gpus={physical_ids}, DP tp=1 per worker) …",
              flush=True)

        shard_cfgs, shard_outs, shard_rollouts = [], [], []
        procs = []
        for w in range(n_workers):
            worker_phys = physical_ids[w * gpw : (w + 1) * gpw]
            shard_cfg = OUTPUT_DIR / f"{epoch_tag}_student_cfg.shard{w}.json"
            shard_out = OUTPUT_DIR / f"{epoch_tag}_student_out.shard{w}.json"
            shard_rollouts_jsonl = (
                OUTPUT_DIR / f"{epoch_tag}_rollouts.shard{w}.jsonl"
            )
            with open(shard_cfg, "w") as f:
                json.dump({
                    "current_ckpt":       current_ckpt,
                    "workset_jsonl":      str(workset_jsonl),
                    "rollouts_jsonl_out": str(shard_rollouts_jsonl),
                    "shard_idx":          w,
                    "n_shards":           n_workers,
                }, f)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(worker_phys)
            p = subprocess.Popen(
                [sys.executable, __file__, "--_student-worker",
                 str(shard_cfg), str(shard_out)],
                env=env,
            )
            procs.append((w, p, worker_phys))
            shard_cfgs.append(shard_cfg)
            shard_outs.append(shard_out)
            shard_rollouts.append(shard_rollouts_jsonl)
            print(f"  • student worker {w + 1}/{n_workers}  "
                  f"pid={p.pid}  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
                  flush=True)

        failed = []
        for w, p, gpus in procs:
            rc = p.wait()
            if rc != 0:
                failed.append((w, gpus, rc))
        if failed:
            for w, gpus, rc in failed:
                print(f"[Steps 1-3] student worker {w + 1} on GPUs {gpus} "
                      f"exited rc={rc}", flush=True)
            raise RuntimeError(f"{len(failed)} student worker(s) failed")

        merged_rollouts = []
        for shard_path in shard_rollouts:
            merged_rollouts.extend(load_jsonl(shard_path))
        merged_rollouts.sort(key=lambda r: int(r.get("_orig_idx", 0)))
        write_jsonl(rollouts_jsonl, merged_rollouts)

        agg = {"n_rollouts": 0, "n_correct": 0, "n_empty": 0,
               "n_workers": n_workers, "per_shard": []}
        for w, shard_out in enumerate(shard_outs):
            with open(shard_out) as f:
                info = json.load(f)
            agg["n_rollouts"] += int(info.get("n_rollouts", 0))
            agg["n_correct"]  += int(info.get("n_correct", 0))
            agg["n_empty"]    += int(info.get("n_empty", 0))
            agg["per_shard"].append({"shard": w, **info})
        with open(student_out, "w") as f:
            json.dump(agg, f, indent=2)

        for p in shard_rollouts:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    with open(student_out) as f:
        st_info = json.load(f)
    print(f"\n[orchestrator] student workers: "
          f"n_rollouts={st_info.get('n_rollouts')}  "
          f"n_correct={st_info.get('n_correct')}  "
          f"n_empty={st_info.get('n_empty')}  "
          f"n_workers={st_info.get('n_workers', 1)}", flush=True)

    # ── Step 4: KL worker (HF student + teacher, top-K support) ──────────
    # Data-parallel across GPUs: spawn N concurrent workers, each owning a
    # subset of rollouts and KL_GPUS_PER_WORKER GPUs (CUDA_VISIBLE_DEVICES
    # is remapped per-worker). Each worker writes a shard kl.jsonl; we merge
    # them back in original order using the _orig_idx tag emitted by the
    # worker.
    if kl_jsonl.exists() and kl_out.exists():
        print(f"\n[Step 4] Cached KL found — skipping.", flush=True)
    else:
        # Resolve physical GPU IDs from the parent process's
        # CUDA_VISIBLE_DEVICES. If unset, fall back to enumerate(N_GPUS).
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if parent_cvd:
            physical_ids = [s.strip() for s in parent_cvd.split(",") if s.strip()]
        else:
            physical_ids = [str(i) for i in range(N_GPUS)]
        n_visible = len(physical_ids)
        gpw       = max(1, min(KL_GPUS_PER_WORKER, n_visible))
        n_workers = max(1, n_visible // gpw)

        print(f"\n[Step 4] Spawning {n_workers} KL worker(s)  "
              f"(visible_gpus={physical_ids}, gpus_per_worker={gpw}) …",
              flush=True)

        shard_cfgs, shard_outs, shard_kls, shard_caches = [], [], [], []
        procs = []
        for w in range(n_workers):
            worker_phys = physical_ids[w * gpw : (w + 1) * gpw]
            shard_cfg   = OUTPUT_DIR / f"{epoch_tag}_kl_cfg.shard{w}.json"
            shard_out   = OUTPUT_DIR / f"{epoch_tag}_kl_out.shard{w}.json"
            shard_kl    = OUTPUT_DIR / f"{epoch_tag}_kl.shard{w}.jsonl"
            shard_cache = OUTPUT_DIR / f"{epoch_tag}_teacher_topk_cache.shard{w}.pt"
            with open(shard_cfg, "w") as f:
                json.dump({
                    "student_ckpt":         current_ckpt,
                    "teacher_model":        TEACHER_MODEL,
                    "rollouts_jsonl":       str(rollouts_jsonl),
                    "kl_jsonl_out":         str(shard_kl),
                    "teacher_cache_out":    str(shard_cache),
                    "opd_reverse_kl_beta":  OPD_REVERSE_KL_BETA,
                    "opd_temp":             OPD_TEMP,
                    "kl_topk":              KL_TOPK,
                    "shard_idx":            w,
                    "n_shards":             n_workers,
                }, f)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(worker_phys)
            p = subprocess.Popen(
                [sys.executable, __file__, "--_kl-worker",
                 str(shard_cfg), str(shard_out)],
                env=env,
            )
            procs.append((w, p, worker_phys))
            shard_cfgs.append(shard_cfg)
            shard_outs.append(shard_out)
            shard_kls.append(shard_kl)
            shard_caches.append(shard_cache)
            print(f"  • worker {w + 1}/{n_workers}  "
                  f"pid={p.pid}  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
                  flush=True)

        # Wait all workers — fail fast on non-zero exit.
        failed = []
        for w, p, gpus in procs:
            rc = p.wait()
            if rc != 0:
                failed.append((w, gpus, rc))
        if failed:
            for w, gpus, rc in failed:
                print(f"[Step 4] KL worker {w + 1} on GPUs {gpus} "
                      f"exited rc={rc}", flush=True)
            raise RuntimeError(f"{len(failed)} KL worker(s) failed")

        # Merge KL shards in original rollouts order via _orig_idx (kept on
        # records to also key the teacher cache for the trainer).
        merged_kl = []
        for shard_kl in shard_kls:
            merged_kl.extend(load_jsonl(shard_kl))
        merged_kl.sort(key=lambda r: int(r.get("_orig_idx", 0)))
        write_jsonl(kl_jsonl, merged_kl)

        # Merge teacher top-K cache shards into one .pt with a flat
        # {orig_idx_str: {...}} record dict. Each shard contributes a
        # disjoint subset of indices (modulo sharding), so a plain dict
        # update is safe. The merged cache is schema v2; each shard now
        # stores raw teacher logits (logprobs_in_cache=False) from the HF
        # full-vocab forward in `_epoch_kl_worker`. See
        # `_kl_record_from_token_kls` and `OPDConsistencyTrainer` for the
        # field semantics.
        merged_cache = {}
        cache_topk   = KL_TOPK
        cache_vclip  = -1
        logprobs_in_cache = False
        for shard_cache in shard_caches:
            if not shard_cache.exists():
                continue
            blob = torch.load(shard_cache, map_location="cpu")
            merged_cache.update(blob.get("records", {}))
            cache_topk        = int(blob.get("kl_topk", cache_topk))
            shard_vclip       = int(blob.get("vocab_clip", -1))
            if shard_vclip > 0:
                cache_vclip = shard_vclip
            logprobs_in_cache = bool(blob.get("logprobs_in_cache",
                                              logprobs_in_cache))
        torch.save({
            "schema_version":    2,
            "kl_topk":           cache_topk,
            "vocab_clip":        cache_vclip,
            "records":           merged_cache,
            "logprobs_in_cache": logprobs_in_cache,
        }, teacher_cache_pt)

        # Best-effort cleanup of shard files (kl + cache).
        for p in shard_kls + shard_caches:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

        # Aggregate per-shard out json into a single summary.
        agg = {"n_with_kl": 0, "n_skipped": 0, "n_total": 0,
               "n_cached":  len(merged_cache),
               "n_workers": n_workers, "per_shard": []}
        for w, shard_out in enumerate(shard_outs):
            with open(shard_out) as f:
                info = json.load(f)
            agg["n_with_kl"] += int(info.get("n_with_kl", 0))
            agg["n_skipped"] += int(info.get("n_skipped", 0))
            agg["n_total"]   += int(info.get("n_total", 0))
            agg["per_shard"].append({"shard": w, **info})
        with open(kl_out, "w") as f:
            json.dump(agg, f, indent=2)
    with open(kl_out) as f:
        kl_info = json.load(f)
    print(f"\n[orchestrator] KL workers: "
          f"n_with_kl={kl_info.get('n_with_kl')}  "
          f"n_skipped={kl_info.get('n_skipped')}  "
          f"n_total={kl_info.get('n_total')}  "
          f"n_cached={kl_info.get('n_cached', 0)}  "
          f"n_workers={kl_info.get('n_workers', 1)}", flush=True)

    # ── Step 5: CPU dispatch (consistency, ranking, retry list, train.jsonl)
    if train_jsonl.exists() and stats_json.exists():
        print(f"\n[Step 5] Cached dispatch found — skipping.", flush=True)
        with open(stats_json) as f:
            dispatch_stats = json.load(f)
    else:
        print(f"\n[Step 5] CPU dispatch (consistency + ranking + masking) …",
              flush=True)
        kl_records = load_jsonl(kl_jsonl)
        train_records, new_retry, dropped, discarded, dispatch_stats = \
            _consistency_dispatch(kl_records, epoch_idx)

        # Shuffle so DDP shards see a mixed order.
        random.Random(SEED + epoch_idx + 1).shuffle(train_records)

        write_jsonl(train_jsonl, train_records)
        with open(stats_json, "w") as f:
            json.dump(dispatch_stats, f, indent=2)

        _save_retry_pool(new_retry)
        _append_dropped_retries(dropped)
        _append_discarded(discarded)

        _print_dispatch_stats(dispatch_stats)
    n_train_records = dispatch_stats.get(
        "n_train_total",
        dispatch_stats.get("n_kept", 0)
        + dispatch_stats.get("n_vanilla_fallback", 0),
    )

    if n_train_records == 0:
        print("  [orchestrator] no training records this epoch → skipping training",
              flush=True)
        return current_ckpt

    # ── Step 6: training subprocess (torchrun for DDP) ───────────────────
    ckpt_dir       = CKPT_BASE_DIR / f"epoch_{epoch_idx + 1:03d}"
    train_cfg_path = OUTPUT_DIR / f"{epoch_tag}_train_cfg.json"

    if (ckpt_dir / "config.json").exists():
        print(f"\n[Step 6] Checkpoint {ckpt_dir} already exists — skipping.",
              flush=True)
    else:
        lr_this_epoch = _lr_for_epoch(GKD_LR, epoch_idx, N_EPOCHS, GKD_LR_SCHEDULE)
        print(f"\n  [LR] epoch {epoch_idx + 1}/{N_EPOCHS}: lr={lr_this_epoch:.3e}  "
              f"(base={GKD_LR}, schedule={GKD_LR_SCHEDULE})", flush=True)

        with open(train_cfg_path, "w") as f:
            json.dump({
                "student_ckpt":         current_ckpt,
                # teacher_model intentionally NOT passed to the train worker:
                # the teacher's top-K logits are pre-computed in Step 4 and
                # consumed via teacher_cache_in below.
                "teacher_cache_in":     str(teacher_cache_pt),
                "train_jsonl":          str(train_jsonl),
                "output_dir":           str(ckpt_dir),
                "epoch_idx":            epoch_idx,
                "n_epochs":             N_EPOCHS,
                "lr":                   lr_this_epoch,
                "max_length":           GKD_MAX_LEN,
                "batch_size":           GKD_BATCH_SIZE,
                "grad_accum":           GKD_GRAD_ACCUM,
                "warmup_ratio":         GKD_WARMUP,
                "opd_reverse_kl_beta":  OPD_REVERSE_KL_BETA,
                "opd_temp":             OPD_TEMP,
                "kl_topk":              KL_TOPK,
            }, f)

        master_port = GKD_MASTER_PORT + (epoch_idx % 100)
        nproc       = max(1, N_GPUS)
        if nproc == 1:
            cmd = [sys.executable, __file__, "--_train-worker", str(train_cfg_path)]
        else:
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc}",
                f"--master_port={master_port}",
                __file__, "--_train-worker", str(train_cfg_path),
            ]
        print(f"\n[Step 6] Spawning training worker  "
              f"(nproc_per_node={nproc}, master_port={master_port}) …", flush=True)
        subprocess.run(cmd, check=True)
        print(f"[orchestrator] Training done → {ckpt_dir}", flush=True)

    # ── Step 7: post-training eval ───────────────────────────────────────
    eval_cfg_path = OUTPUT_DIR / f"{epoch_tag}_eval_cfg.json"
    eval_out_path = OUTPUT_DIR / f"{epoch_tag}_eval_out.json"

    if eval_out_path.exists():
        print(f"\n[Step 7] Cached eval found — skipping.", flush=True)
        with open(eval_out_path) as f:
            eval_info = json.load(f)
    else:
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if parent_cvd:
            physical_ids = [s.strip() for s in parent_cvd.split(",") if s.strip()]
        else:
            physical_ids = [str(i) for i in range(N_GPUS)]
        n_visible = len(physical_ids)
        gpw       = 1
        n_workers = max(1, n_visible)

        print(f"\n[Step 7] Spawning {n_workers} eval worker(s)  "
              f"(visible_gpus={physical_ids}, DP tp=1 per worker) …",
              flush=True)

        shard_outs = []
        procs = []
        for w in range(n_workers):
            worker_phys = physical_ids[w * gpw : (w + 1) * gpw]
            shard_cfg = OUTPUT_DIR / f"{epoch_tag}_eval_cfg.shard{w}.json"
            shard_out = OUTPUT_DIR / f"{epoch_tag}_eval_out.shard{w}.json"
            with open(shard_cfg, "w") as f:
                json.dump({
                    "eval_ckpt": str(ckpt_dir),
                    "epoch_idx": epoch_idx,
                    "n_epochs":  N_EPOCHS,
                    "shard_idx": w,
                    "n_shards":  n_workers,
                }, f)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(worker_phys)
            p = subprocess.Popen(
                [sys.executable, __file__, "--_eval-worker",
                 str(shard_cfg), str(shard_out)],
                env=env,
            )
            procs.append((w, p, worker_phys))
            shard_outs.append(shard_out)
            print(f"  • eval worker {w + 1}/{n_workers}  "
                  f"pid={p.pid}  CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
                  flush=True)

        failed = []
        for w, p, gpus in procs:
            rc = p.wait()
            if rc != 0:
                failed.append((w, gpus, rc))
        if failed:
            for w, gpus, rc in failed:
                print(f"[Step 7] eval worker {w + 1} on GPUs {gpus} "
                      f"exited rc={rc}", flush=True)
            raise RuntimeError(f"{len(failed)} eval worker(s) failed")

        agg_test_correct  = 0
        agg_test_total    = 0
        for shard_out in shard_outs:
            with open(shard_out) as f:
                info = json.load(f)
            agg_test_correct  += int(info.get("test_correct", 0))
            agg_test_total    += int(info.get("test_total", 0))

        eval_info = {
            "test_correct":  agg_test_correct,
            "test_total":    agg_test_total,
            "n_workers":     n_workers,
        }
        with open(eval_out_path, "w") as f:
            json.dump(eval_info, f, indent=2)

    if eval_info and eval_info.get("test_total", 0) > 0:
        test_total  = eval_info.get("test_total", 0)
        log_entry = {
            "epoch":           epoch_idx + 1,
            "test_correct":    eval_info.get("test_correct", 0),
            "test_total":      test_total,
            "test_acc":        (eval_info.get("test_correct", 0) / test_total
                                if test_total > 0 else None),
            "n_train_records": n_train_records,
            "dispatch":        dispatch_stats,
        }
        EVAL_LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_LOG_JSONL, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "n/a"
        print(f"  [eval] epoch {epoch_idx + 1}  "
              f"test_acc={_fmt(log_entry['test_acc'])}", flush=True)

    # ── Cleanup: delete the (large) teacher top-K cache for this epoch ───
    # The cache exists only to bridge KL phase → training; once training is
    # done and the eval has run, it's no longer needed and would otherwise
    # accumulate to N_EPOCHS × ~10-20 GB on disk.
    if teacher_cache_pt.exists():
        try:
            sz_gb = teacher_cache_pt.stat().st_size / (1024 ** 3)
            teacher_cache_pt.unlink()
            print(f"  [cleanup] removed teacher cache "
                  f"({sz_gb:.2f} GB) → {teacher_cache_pt}", flush=True)
        except OSError as e:
            print(f"  [cleanup] WARN: failed to remove teacher cache: {e}",
                  flush=True)

    return str(ckpt_dir)


# ─── Resume helper / main ────────────────────────────────────────────────────

def _find_latest_checkpoint():
    if not CKPT_BASE_DIR.exists():
        return -1, None
    best_ep, best_path = -1, None
    for d in sorted(CKPT_BASE_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("epoch_"):
            continue
        try:
            ep_num = int(name[len("epoch_"):])
        except ValueError:
            continue
        if (d / "config.json").exists() and ep_num > best_ep:
            best_ep, best_path = ep_num, d
    if best_ep < 0:
        return -1, None
    return best_ep - 1, str(best_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("process_reward_consistency_opd.py")
    print(f"  STUDENT_MODEL              = {STUDENT_MODEL}")
    print(f"  TEACHER_MODEL              = {TEACHER_MODEL}")
    print(f"  POOL_DATASET               = {POOL_DATASET}")
    print(f"  TOP_PERCENT_RESPONSES      = {TOP_PERCENT_RESPONSES}")
    print(f"  MASK_TOP_PERCENT_SEGMENTS  = {MASK_TOP_PERCENT_SEGMENTS}  "
          f"(only paragraphs with inconsistency > 0 are actually masked)")
    print(f"  MIN_MERGED_SEGS_PER_RESP   = {MIN_MERGED_SEGS_PER_RESP}")
    print(f"  MAX_RETRIES                = {MAX_RETRIES}")
    print(f"  K_EVAL                     = {K_EVAL}")
    print(f"  KL_TOPK                    = {KL_TOPK}  "
          f"(teacher top-K cached for Step 6; consistency KL on teacher top-K support S_t)")
    print(f"  KL_GPUS_PER_WORKER         = {KL_GPUS_PER_WORKER}  "
          f"(Step 4 HF: teacher device_map-sharded over the slice; "
          f"student replicated on cuda:0; DP across {max(1, N_GPUS // KL_GPUS_PER_WORKER)} workers)")
    print(f"  student vLLM (Step 1-3, 7) = DP only  "
          f"(workers={N_GPUS}, tp=1 per worker)")
    print(f"  OPD_REVERSE_KL_BETA        = {OPD_REVERSE_KL_BETA}")
    print(f"  OPD_TEMP                   = {OPD_TEMP}")
    print(f"  N_EPOCHS                   = {N_EPOCHS}")
    print(f"  N_GPUS                     = {N_GPUS}")
    print(f"  GKD_LR                     = {GKD_LR}  (schedule={GKD_LR_SCHEDULE})")
    print(f"  GKD_BATCH_SIZE             = {GKD_BATCH_SIZE}  ga={GKD_GRAD_ACCUM}")
    print(f"  GKD_MAX_LEN                = {GKD_MAX_LEN}")
    print(f"  N_SAMPLES                  = {'all' if N_SAMPLES == 0 else N_SAMPLES}")
    print(f"  OUTPUT_DIR                 = {OUTPUT_DIR}")
    print("=" * 60, flush=True)

    pool = load_training_pool()
    if not pool:
        print("\nPool is empty — nothing to train on. Exiting.", flush=True)
        return

    last_done_idx, resumed_ckpt = _find_latest_checkpoint()
    if last_done_idx >= 0:
        current_ckpt = resumed_ckpt
        start_epoch  = last_done_idx + 1
        print(f"\n[resume] Detected completed checkpoint: {resumed_ckpt}")
        print(f"[resume] Resuming from epoch {start_epoch + 1}/{N_EPOCHS}",
              flush=True)
    else:
        current_ckpt = STUDENT_MODEL
        start_epoch  = 0
        print(f"\n[resume] No existing checkpoint found — starting from scratch.",
              flush=True)

    for epoch in range(start_epoch, N_EPOCHS):
        current_ckpt = run_epoch(epoch, current_ckpt, pool)

    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"  Final checkpoint : {current_ckpt}")
    print(f"  Pool size        : {len(pool)}  (global train split)")
    if RETRY_POOL_JSONL.exists():
        rp = _load_retry_pool()
        print(f"  Retry backlog    : {len(rp)}  → {RETRY_POOL_JSONL}")
    if DROPPED_RETRIES_JSONL.exists():
        n_drop = sum(1 for _ in open(DROPPED_RETRIES_JSONL, encoding="utf-8"))
        print(f"  Dropped retries  : {n_drop}  → {DROPPED_RETRIES_JSONL}")
    if DISCARDED_LOG_JSONL.exists():
        n_disc = sum(1 for _ in open(DISCARDED_LOG_JSONL, encoding="utf-8"))
        print(f"  Discarded (S✓ but low-consistency): {n_disc}  → "
              f"{DISCARDED_LOG_JSONL}")
    print(f"{'='*60}", flush=True)


# ─── CLI dispatch (subprocesses) ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--_student-worker":
        _epoch_student_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--_kl-worker":
        _epoch_kl_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--_eval-worker":
        _epoch_eval_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--_train-worker":
        _train_worker(sys.argv[2])
    else:
        main()
