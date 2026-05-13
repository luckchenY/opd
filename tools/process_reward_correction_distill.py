"""
process_reward_correction_distill.py
====================================
Segment-level teacher rewrite distillation (replacement for
process_reward_weighted_distill.py).

Why a new script
----------------
process_reward_weighted_distill.py up-weights JSD on "negative-PR" student
tokens, asking the student to match the teacher's logits exactly there. But the
teacher (Qwen3-8B) never generates the wrong prefix the student wrote, so the
teacher's logits at those positions are off-distribution noise. The student
ends up imitating a surface "Wait, let me reconsider..." pattern without
learning to actually recover, which produces verbose self-doubting loops that
run out of token budget before reaching \\boxed{}.

This script gives the student an *explicit* correction trajectory instead:

  1. Student rolls out a response. K-fold PR is computed per segment (same
     machinery as the weighted-distill script).
  2. For each problem with at least one negative-PR segment, splice the
     response at the chosen segment (default: the *first* place the student
     went wrong) and have the *teacher continue from that splice point all the
     way to a boxed answer*.
  3. Keep only continuations that reach the correct answer.
  4. Training record = student_prefix + correct_teacher_continuation. The wrong
     prefix is conditioning context (label = -100), NOT a training target. The
     first CORRECTION_WINDOW tokens of the continuation get a higher per-token
     weight so the model is most strongly supervised on the *moment of correction*.

Two mutually-exclusive training modes
-------------------------------------
TRAINING_MODE=sft (default)
  Loss = weighted cross-entropy on labeled tokens. Teacher is NOT loaded into
  the train subprocess — cheap and simple.

TRAINING_MODE=opd
  Loss = weighted KL between student and teacher distributions on labeled
  tokens. Teacher IS loaded into the train subprocess (extra ~16GB / rank for
  Qwen3-8B in bf16). The KL is a linear mix of reverse and forward KL:
      loss = β · KL(P_student ‖ P_teacher) + (1-β) · KL(P_teacher ‖ P_student)
  controlled by OPD_REVERSE_KL_BETA.  β=1.0 → pure reverse-KL (mode-seeking,
  decisive); β=0.0 → pure forward-KL (mode-covering, often verbose); β=0.5 →
  balanced (close to JSD).

Background SFT mixin (both modes)
---------------------------------
Each epoch, BACKGROUND_RATIO * |pool| random pool problems NOT covered by
correction records contribute a plain training record using the cached Phase-0
teacher rollout (full clean solution), with per-token weight BACKGROUND_WEIGHT.
This regularizes against forgetting easy problems and against over-fitting to
the narrow "hard-but-recoverable" correction slice.

There is NO fallback to teacher imitation on problems where the teacher cannot
recover from the student prefix — those splices are dropped.

Per-epoch loop
--------------
  Step 1  vLLM (student)  : fresh student rollout on every pool problem
  Step 2  CPU             : episode segmentation
  Step 3  vLLM (student)  : K-fold PR eval per episode boundary (incl. empty)
  Step 4  CPU             : pick a splice point per problem (negative PR)
  Step 5  vLLM (teacher)  : teacher continues from each splice point; keep the
                            shortest correct continuation;
                            also emit background SFT records
  Step 6  CPU             : write epoch_NNN_train.jsonl
  Step 7  proc (torchrun) : weighted SFT or OPD training, save ckpt
  Step 8  vLLM (student)  : post-training held-out evaluation

Important env vars
------------------
  CORRECTION_WEIGHT      weight on the first CORRECTION_WINDOW tokens of the
                         teacher correction. Default 2.0.
  CORRECTION_WINDOW      length of the high-weight correction zone (in tokens).
                         Default 256.
  SPLICE_AT              first|worst — which (merged) negative-PR group to
                         splice at. Default worst.
  MERGE_SEGMENTS         Collapse monotonic runs into one merged group before
                         filtering, so PR reflects the group's cumulative drop
                         instead of a noisy single-segment delta. Default true.
  MIN_NEG_PR             A merged group is eligible only if its PR (cumulative
                         drop across the run) < -MIN_NEG_PR. Default 0.30.
  K_EVAL                 K-fold sample count per segment. Default 8 here
                         (process_reward_distill ships 4 — too noisy for
                         threshold-based filtering).
  K_CORRECTIONS          teacher samples per splice. Default 4.
  CORRECTION_TEMPERATURE teacher sampling temperature. Default 0.3.
  BACKGROUND_RATIO       fraction-of-pool background SFT records / epoch. 0.3.
  BACKGROUND_WEIGHT      per-token weight for background records. 0.5.
  TRAINING_MODE          sft | opd. Default sft.
  OPD_REVERSE_KL_BETA    β in the reverse/forward KL mix (opd mode). Default 0.7.
  OPD_TEMP               softmax temperature in the KL (opd mode). Default 1.0.
  N_EPOCHS               outer epochs. Default 5.
  GKD_LR / GKD_BATCH_SIZE / GKD_GRAD_ACCUM / GKD_MAX_LEN / GKD_WARMUP /
  GKD_LR_SCHEDULE        standard training knobs.
  TEST_EVAL_EVERY / TEST_EVAL_SAMPLES   eval cadence.
  N_SAMPLES              pool problems (0 = full).
  POOL_DATASET           hendrycks_math (default) | dapo_math_17k.
                         hendrycks_math : MATH train ~7500, has test split.
                         dapo_math_17k  : DAPO-Math-17k (~17000, train only).
                                          MATH-test eval is skipped — run
                                          AIME via evalscope externally.
  DAPO_SHARDS            Optional ":" -separated absolute paths to DAPO arrow
                         shards if your local cache lives elsewhere.
  OUTPUT_DIR             output dir. Default tools/correction_distill_output.

Usage
-----
    # Plain SFT on MATH (default)
    python tools/process_reward_correction_distill.py

    # Plain SFT on DAPO-Math-17k
    POOL_DATASET=dapo_math_17k \\
    OUTPUT_DIR=$PWD/tools/correction_distill_output_sft_dapo \\
        python tools/process_reward_correction_distill.py

    # OPD with β-leaning-reverse KL
    TRAINING_MODE=opd OPD_REVERSE_KL_BETA=0.7 \\
        python tools/process_reward_correction_distill.py
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

GPU_IDS = [4,5,6,7] 

# ─── Import shared helpers from the original distillation script ─────────────
sys.path.insert(0, str(Path(__file__).parent))
from process_reward_distill import (  # noqa: E402
    DATASET, DATASET_CONFIGS, STUDENT_MODEL, TEACHER_MODEL, N_GPUS,
    K_EVAL, EVAL_TEMPERATURE, MAX_ANSWER_TOKENS, MAX_NEW_TOKENS,
    ROLLOUT_TEMPERATURE, EPOCH_MAX_MODEL_LEN,
    EPISODE_EVAL_SYSTEM, INFERENCE_BATCH_SIZE,
    SEED, N_SAMPLES,
    extract_boxed_answer, answers_match,
    load_dataset_samples, segment_response_text, merge_monotonic_runs,
    write_jsonl, load_jsonl,
    build_filter_prompt, build_episode_eval_prompt, build_continuation_prompt,
    make_llm, destroy_llm, safe_generate, run_isolated_rollout,
)

# Override K_EVAL: process_reward_distill defaults to 4, which gives a per-
# segment-acc std of ~sqrt(0.25/4)≈0.25 and per-segment-PR std of ~0.35 — way
# too noisy to use threshold-based splice filtering on. Default to 8 here
# (per-segment std ~0.18, per-segment-PR std ~0.25).
K_EVAL = int(os.environ.get("K_EVAL", "8"))


# ─── New / overridden config knobs ───────────────────────────────────────────

CORRECTION_WEIGHT       = float(os.environ.get("CORRECTION_WEIGHT", "2.0"))
CORRECTION_WINDOW       = int(os.environ.get("CORRECTION_WINDOW", "0"))
K_CORRECTIONS           = int(os.environ.get("K_CORRECTIONS", "2"))
CORRECTION_TEMPERATURE  = float(os.environ.get("CORRECTION_TEMPERATURE", "0.3"))

# ── Splice selection: how to find "where did the student go wrong" ──────────
#
# Two-stage pipeline:
#
#     all merged segments ──→ MIN_NEG_PR filter ──→ candidate set
#                                  (gate / yes-no)
#                                                       │
#                                                       ▼
#                                              SPLICE_AT picker ──→ splice point
#                                                  (which-one)
#
#     MIN_NEG_PR  =  "Is this group bad ENOUGH to count as a real error?"
#                    Decides which merged groups are even ELIGIBLE.
#     SPLICE_AT   =  "Among the eligible groups, which one to splice at?"
#                    Decides WHICH ELIGIBLE GROUP to use.
#
# These do completely different jobs and you usually tune them independently.
#
#   MERGE_SEGMENTS  Collapse consecutive segments whose acc_rate moves
#                   monotonically into one merged group, then recompute that
#                   group's PR as group_acc - prev_group_acc (the cumulative
#                   drop of the whole sustained run). Without merging, per-
#                   segment PR is dominated by K-fold sampling noise.
#                   Default true.
#
#   MIN_NEG_PR      Threshold on the merged-group PR. A group is ELIGIBLE iff
#                       group_PR < -MIN_NEG_PR.
#                   Default 0.30 — i.e. only groups whose cumulative drop is
#                   ≥ 30 percentage points get into the candidate set.
#
#   SPLICE_AT       Picker over the eligible groups:
#                     "first" → earliest passing group (teaches "stop early")
#                     "worst" → group with most-negative PR (cleanest signal,
#                               and matches our SPLICE_AT default)
#                   Default "worst".
#
# Worked example. After K-fold + merge, a problem has 5 merged groups with PRs
#     A=+0.10  B=-0.20  C=-0.05  D=-0.45  E=+0.15.
# With MIN_NEG_PR=0.30, only D survives the gate (-0.45 < -0.30).
#     candidate set = {D}
# SPLICE_AT picks D regardless of first/worst (only one candidate).
#
# If we relaxed MIN_NEG_PR to 0.10, B and D both pass:
#     candidate set = {B, D}
#     SPLICE_AT=first → B (earliest);  SPLICE_AT=worst → D (most-negative).
#
# Edge cases:
#   MIN_NEG_PR=0     → no filter; every PR<0 group is eligible.
#   MIN_NEG_PR≥1     → no group ever passes; every problem is dropped from the
#                      correction set and falls into background SFT.
SPLICE_AT               = os.environ.get("SPLICE_AT", "worst").lower()  # first|worst
MERGE_SEGMENTS          = os.environ.get("MERGE_SEGMENTS", "true").lower() == "true"
MIN_NEG_PR              = float(os.environ.get("MIN_NEG_PR", "0.30"))

# Background SFT mixin: each epoch, randomly draw this fraction-of-pool many
# pool problems (NOT already covered by a correction record) and emit plain-SFT
# records using the cached Phase-0 teacher rollout. Purpose is regularization:
# expose the student to easy problems it already handles + diverse teacher
# style, to prevent forgetting and overfit to the narrow "student-failed-AND-
# teacher-recovered" slice. Set BACKGROUND_RATIO=0 to disable.
BACKGROUND_RATIO        = float(os.environ.get("BACKGROUND_RATIO", "0.3"))
BACKGROUND_WEIGHT       = float(os.environ.get("BACKGROUND_WEIGHT", "0.5"))

# Training mode — pick one:
#   sft : weighted cross-entropy on labeled tokens (cheap, no teacher loaded).
#   opd : weighted divergence between student and teacher distributions on the
#         labeled tokens (requires teacher loaded into the train subprocess).
#         The divergence is a linear mix of reverse-KL and forward-KL controlled
#         by OPD_REVERSE_KL_BETA below.
TRAINING_MODE       = os.environ.get("TRAINING_MODE", "sft").lower()  # sft | opd

# OPD knobs (only used when TRAINING_MODE=opd).
#   OPD_REVERSE_KL_BETA  Mix coefficient β in
#                          loss = β · KL(P_student ‖ P_teacher)         (reverse, mode-seeking)
#                               + (1-β) · KL(P_teacher ‖ P_student)     (forward, mode-covering)
#                        β=1.0 → pure reverse-KL (sharp / decisive student);
#                        β=0.0 → pure forward-KL (mode-cover, often verbose);
#                        β=0.5 → balanced (close to JSD).
#                        Default 0.7 (lean toward reverse, since student ≪ teacher).
#   OPD_TEMP             Softmax temperature applied before computing the KLs.
OPD_REVERSE_KL_BETA = float(os.environ.get("OPD_REVERSE_KL_BETA", "0.7"))
OPD_TEMP            = float(os.environ.get("OPD_TEMP", "1.0"))

if TRAINING_MODE not in ("sft", "opd"):
    raise ValueError(f"TRAINING_MODE must be 'sft' or 'opd', got {TRAINING_MODE!r}")
if not (0.0 <= OPD_REVERSE_KL_BETA <= 1.0):
    raise ValueError(f"OPD_REVERSE_KL_BETA must be in [0,1], got {OPD_REVERSE_KL_BETA}")

N_EPOCHS        = int(os.environ.get("N_EPOCHS", "5"))
GKD_LR          = float(os.environ.get("GKD_LR", "1e-6"))
GKD_BATCH_SIZE  = int(os.environ.get("GKD_BATCH_SIZE", "1"))
GKD_GRAD_ACCUM  = int(os.environ.get("GKD_GRAD_ACCUM", "4"))
GKD_MAX_LEN     = int(os.environ.get("GKD_MAX_LEN", str(MAX_NEW_TOKENS + 1024)))
GKD_WARMUP      = float(os.environ.get("GKD_WARMUP", "0.05"))
GKD_LR_SCHEDULE = os.environ.get("GKD_LR_SCHEDULE", "cosine")

# Base port for torchrun's TCP rendezvous. The actual port used per epoch is
# GKD_MASTER_PORT + (epoch_idx % 100), so two parallel runs on the same host
# (e.g. an SFT vs OPD ablation) MUST use different GKD_MASTER_PORT bases —
# otherwise torchrun fails with EADDRINUSE on localhost.
GKD_MASTER_PORT = int(os.environ.get("GKD_MASTER_PORT", "29500"))

TEST_EVAL_EVERY   = int(os.environ.get("TEST_EVAL_EVERY", "1"))
TEST_EVAL_SAMPLES = int(os.environ.get("TEST_EVAL_SAMPLES", "500"))


# ─── Pool dataset selection ──────────────────────────────────────────────────
# POOL_DATASET = hendrycks_math (default, the imported MATH config from
#                process_reward_distill.py — has both train + test split)
#              | dapo_math_17k  (BytedTsinghua-SIA/DAPO-Math-17k, ~17k harder
#                problems, train only, no test split — eval_worker will skip
#                MATH-test eval and you should run AIME-style benchmarks via
#                evalscope on the saved checkpoints separately).
POOL_DATASET = os.environ.get("POOL_DATASET", "hendrycks_math").lower()
if POOL_DATASET not in ("hendrycks_math", "dapo_math_17k"):
    raise ValueError(f"POOL_DATASET must be 'hendrycks_math' or "
                     f"'dapo_math_17k', got {POOL_DATASET!r}")

# Shard paths for DAPO-Math-17k (used when POOL_DATASET=dapo_math_17k).
# Override via env var DAPO_SHARDS (colon-separated absolute paths) if your
# local cache lives elsewhere.
_DEFAULT_DAPO_SHARDS = [
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/"
    "data/hf_datasets_cache/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0/"
    "65877096c24ffa7abc4e4fa5edb95cf3413a5674/"
    "dapo-math-17k-train-00000-of-00002.arrow",
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/"
    "data/hf_datasets_cache/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0/"
    "65877096c24ffa7abc4e4fa5edb95cf3413a5674/"
    "dapo-math-17k-train-00001-of-00002.arrow",
]
DAPO_SHARDS = (
    os.environ.get("DAPO_SHARDS").split(":")
    if os.environ.get("DAPO_SHARDS") else _DEFAULT_DAPO_SHARDS
)


# ─── Paths ───────────────────────────────────────────────────────────────────

_DIR       = Path(__file__).parent
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR", str(_DIR / "correction_distill_output_sft")))
POOL_JSONL         = OUTPUT_DIR / "pool.jsonl"
TEST_SET_JSONL     = OUTPUT_DIR / "test_set.jsonl"
TRAIN_EVAL_JSONL   = OUTPUT_DIR / "train_eval_set.jsonl"
TRAIN_EVAL_SOLVED  = OUTPUT_DIR / "train_eval_solved.json"
EVAL_LOG_JSONL     = OUTPUT_DIR / "eval_log.jsonl"
CKPT_BASE_DIR      = OUTPUT_DIR / "checkpoints"


# ─── Pool sample loaders ─────────────────────────────────────────────────────
#
# The rest of the pipeline expects a list of dicts with at least
#   {"problem": str, "gt_answer": str}.
# `_load_pool_samples_normalized()` dispatches on POOL_DATASET and returns this.

def _extract_problem_text(row):
    """Best-effort extraction of the problem statement from a DAPO-style row."""
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
    """Best-effort extraction of the ground-truth answer from a DAPO-style row."""
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
    """Load DAPO-Math-17k from cached Arrow shards and normalize to
    list[{"problem": str, "gt_answer": str}]. No test split — DAPO is train-only.
    """
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

    # Defensive: if extraction failed, dump schema+first row to help debug.
    if total == 0:
        try:
            reader = ipc.open_stream(DAPO_SHARDS[0])
            b = reader.read_next_batch()
            print("  WARNING: extracted 0 samples from DAPO. "
                  f"First-shard schema = {b.schema.names}", flush=True)
            print("  First-row contents:")
            for col in b.schema.names:
                v = b.column(col)[0].as_py()
                s = repr(v); s = s[:300] + "..." if len(s) > 300 else s
                print(f"    {col}: {s}", flush=True)
        except Exception as e:
            print(f"  (debug print failed: {e})", flush=True)
        raise RuntimeError(
            "DAPO load returned 0 samples. Fix _extract_problem_text / "
            "_extract_gt_answer based on the schema printed above.")

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


def _load_pool_samples_normalized():
    """Returns list[{"problem": str, "gt_answer": str}] for the chosen
    POOL_DATASET. The rest of phase 0 (teacher rollout, T✓ filter,
    pool.jsonl write) is dataset-agnostic from here on.
    """
    if POOL_DATASET == "dapo_math_17k":
        return _load_dapo_samples_normalized()

    # MATH path: use the imported helper, then normalize keys + extract boxed gt.
    cfg     = DATASET_CONFIGS[DATASET]
    samples = load_dataset_samples()
    out = []
    for s in samples:
        problem = s.get(cfg["prompt_key"])
        sol     = s.get(cfg["solution_key"])
        gt      = extract_boxed_answer(sol) if sol is not None else None
        if not problem or gt is None:
            continue
        out.append({"problem": str(problem), "gt_answer": str(gt)})
    return out


# ─── Outer LR schedule (same as weighted-distill) ────────────────────────────

def _lr_for_epoch(base_lr, epoch_idx, n_epochs, schedule):
    if schedule == "constant" or n_epochs <= 1:
        return base_lr
    progress = epoch_idx / n_epochs
    if schedule == "linear":
        return base_lr * max(0.0, 1.0 - progress)
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown GKD_LR_SCHEDULE: {schedule}")


# ─── Phase 0: build static T✓ pool (identical to weighted-distill) ───────────

def phase0_build_pool():
    print(f"\n{'='*60}")
    print("Phase 0: Building static T✓ sample pool")
    print(f"{'='*60}", flush=True)

    if POOL_JSONL.exists():
        print(f"  Existing pool found at {POOL_JSONL}, loading …", flush=True)
        pool = load_jsonl(POOL_JSONL)
        print(f"  Loaded {len(pool)} samples.", flush=True)

        # If the pool was built by an older version that did NOT save
        # teacher_rollout, re-run teacher rollouts for these problems so the
        # background SFT path has data to draw from. Pool is enriched
        # in-place. Pool problems do not change, so cached splits/eval files
        # remain valid.
        if pool and "teacher_rollout" not in pool[0]:
            print(f"  [phase0] Pool lacks 'teacher_rollout' — re-running teacher "
                  f"rollouts on the existing pool ({len(pool)} problems) …",
                  flush=True)
            from transformers import AutoTokenizer
            teacher_tok = AutoTokenizer.from_pretrained(
                TEACHER_MODEL, trust_remote_code=True)
            prompts = [build_filter_prompt(it["problem"], teacher_tok)
                       for it in pool]
            txts = run_isolated_rollout(
                TEACHER_MODEL, prompts,
                temperature=0, max_tokens=MAX_NEW_TOKENS,
            )
            for it, t in zip(pool, txts):
                it["teacher_rollout"] = t
            write_jsonl(POOL_JSONL, pool)
            print(f"  [phase0] Pool enriched with teacher_rollout and saved.",
                  flush=True)
        return pool

    samples = _load_pool_samples_normalized()
    n_total = len(samples)

    print(f"\n  [0] Teacher rollout  ({n_total} problems) …", flush=True)
    from transformers import AutoTokenizer
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    teacher_prompts = [
        build_filter_prompt(s["problem"], teacher_tok) for s in samples
    ]
    teacher_texts = run_isolated_rollout(
        TEACHER_MODEL, teacher_prompts,
        temperature=0, max_tokens=MAX_NEW_TOKENS,
    )

    pool = []
    n_teacher_ok = 0
    for s, text in zip(samples, teacher_texts):
        gt = s["gt_answer"]
        pred = extract_boxed_answer(text)
        if answers_match(pred, gt):
            n_teacher_ok += 1
            pool.append({
                "problem":         s["problem"],
                "gt_answer":       gt,
                "teacher_rollout": text,   # cached for background SFT
            })
    print(f"  Teacher: {n_teacher_ok}/{n_total} correct  →  pool size = {len(pool)}",
          flush=True)
    write_jsonl(POOL_JSONL, pool)

    # Test split: only MATH (hendrycks_math) has one. DAPO is train-only —
    # leave TEST_SET_JSONL absent and the eval_worker will skip MATH-test eval.
    # For DAPO runs you should evaluate AIME-style benchmarks separately via
    # evalscope on the saved checkpoints.
    if POOL_DATASET == "hendrycks_math" and not TEST_SET_JSONL.exists():
        print(f"\n  Saving held-out test set …", flush=True)
        cfg = DATASET_CONFIGS[DATASET]
        test_samples = load_dataset_samples(split="test")
        test_records = [
            {"problem":   s[cfg["prompt_key"]],
             "gt_answer": extract_boxed_answer(s[cfg["solution_key"]])}
            for s in test_samples
        ]
        write_jsonl(TEST_SET_JSONL, test_records)
    elif POOL_DATASET != "hendrycks_math":
        print(f"  [phase0] {POOL_DATASET} has no test split — "
              f"skipping test_set.jsonl (run AIME eval externally).", flush=True)

    if not TRAIN_EVAL_JSONL.exists():
        train_eval_records = [
            {"problem": item["problem"], "gt_answer": item["gt_answer"]}
            for item in pool
        ]
        write_jsonl(TRAIN_EVAL_JSONL, train_eval_records)
        with open(TRAIN_EVAL_SOLVED, "w") as f:
            json.dump([], f)

    return pool


# ─── Splice selection ────────────────────────────────────────────────────────

def _select_splice(episodes, ep_accs, empty_acc, mode,
                   min_neg_pr=0.0, merge_segments=True):
    """Pick a splice point on the (merged) segments.

    Pipeline:
      1. Build per-segment points {end_char, acc_rate, process_reward} from the
         raw K-fold accuracies. process_reward = acc_i - acc_{i-1}.
      2. (Optional) Collapse monotonic runs via merge_monotonic_runs. After
         merging, recompute each merged group's process_reward as
         (group_acc - prev_group_acc), so the PR of a merged group reflects the
         *full* sustained drop across the run instead of one noisy step.
      3. Filter merged groups by  process_reward < -min_neg_pr.
      4. Pick the chosen group: 'first' = earliest passing, 'worst' = most-
         negative PR.

    Returns (group_idx, splice_char) where splice_char is the *character offset*
    in the student response at which the teacher should resume (= end_char of
    the merged group immediately before the chosen one, or 0 if the chosen
    group is the very first). Returns (None, None) if no group passes.
    """
    if not episodes:
        return None, None

    points = []
    prev_acc = empty_acc
    for ep_idx, (_sc, end_c) in enumerate(episodes):
        curr_acc = ep_accs.get(ep_idx, 0.0)
        points.append({
            "end_char":       end_c,
            "acc_rate":       curr_acc,
            "process_reward": curr_acc - prev_acc,
        })
        prev_acc = curr_acc

    if merge_segments and len(points) > 1:
        points = merge_monotonic_runs(points)
        prev_acc = empty_acc
        for p in points:
            p["process_reward"] = p["acc_rate"] - prev_acc
            prev_acc = p["acc_rate"]

    candidates = []   # list of (idx_in_points, point)
    for idx, p in enumerate(points):
        if p["process_reward"] >= -min_neg_pr:
            continue
        candidates.append((idx, p))

    if not candidates:
        return None, None

    if mode == "worst":
        candidates.sort(key=lambda c: c[1]["process_reward"])  # most negative

    chosen_idx, chosen_p = candidates[0]
    splice_char = 0 if chosen_idx == 0 else points[chosen_idx - 1]["end_char"]
    return chosen_idx, splice_char


# ─── Step 1–4: student worker (rollout + K-fold + splice selection) ──────────

def _epoch_student_worker(cfg_path, out_path):
    """vLLM-only subprocess. Loads the student, rolls out + K-fold PR-evals each
    pool problem, picks the splice point per problem, and emits a splices JSONL
    that the teacher worker will consume.
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(cfg_path) as f:
        cfg = json.load(f)
    current_ckpt      = cfg["current_ckpt"]
    splices_jsonl_out = cfg["splices_jsonl_out"]

    pool = load_jsonl(POOL_JSONL)
    if not pool:
        write_jsonl(splices_jsonl_out, [])
        with open(out_path, "w") as f:
            json.dump({"n_splice_records": 0}, f)
        return

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

    llm = make_llm(current_ckpt)

    inf_bs    = INFERENCE_BATCH_SIZE if INFERENCE_BATCH_SIZE > 0 else len(pool)
    n_batches = max(1, (len(pool) + inf_bs - 1) // inf_bs)

    splice_records = []
    n_with_neg, n_no_neg, n_skipped_empty = 0, 0, 0

    print(f"\n[Student worker] pool={len(pool)}  inf_batch={inf_bs}  "
          f"n_batches={n_batches}", flush=True)

    for b_idx in range(n_batches):
        b_start = b_idx * inf_bs
        b_end   = min(b_start + inf_bs, len(pool))
        batch   = pool[b_start:b_end]

        print(f"\n  [batch {b_idx + 1}/{n_batches}]  samples {b_start}–{b_end - 1}",
              flush=True)

        # Step 1: student rollout
        rollout_prompts = [
            build_filter_prompt(item["problem"], student_tok) for item in batch
        ]
        rollout_outs = safe_generate(
            llm, rollout_prompts, rollout_params, student_tok,
            EPOCH_MAX_MODEL_LEN, label="rollout",
        )
        responses = [
            (out.outputs[0].text if out is not None else "") for out in rollout_outs
        ]

        # Step 2: segmentation
        episodes_per = [
            segment_response_text(r) if r.strip() else [] for r in responses
        ]

        # Step 3: K-fold PR eval (incl. empty prefix baseline)
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

        kfold_outs = safe_generate(
            llm, kfold_prompts, kfold_params, student_tok,
            EPOCH_MAX_MODEL_LEN, label="kfold",
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

        # Step 4: splice selection
        for li, (item, resp, episodes) in enumerate(
            zip(batch, responses, episodes_per)
        ):
            if not resp.strip() or not episodes:
                n_skipped_empty += 1
                continue
            li_accs   = acc_per_li.get(li, {})
            empty_acc = li_accs.get(-1, 0.0)
            ep_accs   = {k: v for k, v in li_accs.items() if k != -1}

            ep_idx, splice_char = _select_splice(
                episodes, ep_accs, empty_acc, SPLICE_AT,
                min_neg_pr=MIN_NEG_PR,
                merge_segments=MERGE_SEGMENTS)
            if ep_idx is None:
                n_no_neg += 1
                continue
            n_with_neg += 1
            splice_records.append({
                "pool_idx":        b_start + li,   # absolute index into pool.jsonl
                "problem":         item["problem"],
                "gt_answer":       item["gt_answer"],
                "student_prefix":  resp[:splice_char],
                "splice_ep_idx":   ep_idx,
                "splice_char":     splice_char,
                "empty_acc":       round(empty_acc, 6),
                "neg_seg_acc":     round(ep_accs.get(ep_idx, 0.0), 6),
            })

        print(f"    splices_so_far={len(splice_records)}  "
              f"with_neg={n_with_neg}  no_neg={n_no_neg}  "
              f"empty={n_skipped_empty}", flush=True)

    destroy_llm(llm)

    write_jsonl(splices_jsonl_out, splice_records)
    with open(out_path, "w") as f:
        json.dump({
            "n_splice_records": len(splice_records),
            "n_with_neg":       n_with_neg,
            "n_no_neg":         n_no_neg,
            "n_skipped_empty":  n_skipped_empty,
        }, f)
    print(f"\n[Student worker] splices written: {len(splice_records)}  "
          f"(no_neg={n_no_neg}, empty={n_skipped_empty})", flush=True)


# ─── Step 5–6: teacher worker (continue + filter by correctness) ─────────────

def _epoch_teacher_worker(cfg_path, out_path):
    """vLLM-only subprocess.

    Produces two kinds of records into the per-epoch train JSONL:

      • "correction": teacher continued from the student's wrong prefix and
        reached the correct answer.  Prefix is included in input but masked
        from the loss; the first CORRECTION_WINDOW tokens of the correction
        get an elevated weight CORRECTION_WEIGHT.  If teacher cannot recover
        from a given prefix, the splice is dropped entirely (no fallback).
      • "background": random subset (BACKGROUND_RATIO of pool) of pool
        problems NOT covered by correction.  Plain SFT on the cached Phase-0
        teacher rollout, weight BACKGROUND_WEIGHT (regularizer).
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(cfg_path) as f:
        cfg = json.load(f)
    splices_jsonl   = cfg["splices_jsonl"]
    train_jsonl_out = cfg["train_jsonl_out"]
    epoch_idx       = cfg.get("epoch_idx", 0)

    splices = load_jsonl(splices_jsonl)
    pool    = load_jsonl(POOL_JSONL)

    pool_has_rollout = bool(pool) and bool(pool[0].get("teacher_rollout", "").strip())
    if BACKGROUND_RATIO > 0 and not pool_has_rollout:
        print("  [Teacher worker] WARNING: pool has no teacher_rollout — "
              "background SFT disabled this epoch.", flush=True)

    train_records = []
    n_correction = 0
    n_background = 0
    covered_pool_idx = set()

    if splices:
        teacher_tok = AutoTokenizer.from_pretrained(
            TEACHER_MODEL, trust_remote_code=True)
        correction_params = SamplingParams(
            n=K_CORRECTIONS,
            temperature=CORRECTION_TEMPERATURE,
            top_k=50, top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
        )
        llm = make_llm(TEACHER_MODEL)

        prompts = [
            build_continuation_prompt(rec["problem"], rec["student_prefix"],
                                      teacher_tok)
            for rec in splices
        ]
        print(f"\n[Teacher worker] generating corrections for {len(prompts)} "
              f"splice points  (K={K_CORRECTIONS}, T={CORRECTION_TEMPERATURE})",
              flush=True)
        outs = safe_generate(
            llm, prompts, correction_params, teacher_tok,
            EPOCH_MAX_MODEL_LEN, label="correction",
        )

        for rec, out in zip(splices, outs):
            pool_idx = rec.get("pool_idx", -1)
            if 0 <= pool_idx:
                covered_pool_idx.add(pool_idx)
            gt = rec["gt_answer"]

            candidates = []
            if out is not None:
                for o in out.outputs:
                    text = o.text
                    if text.strip() and answers_match(extract_boxed_answer(text),
                                                      gt):
                        candidates.append(text)

            if not candidates:
                # Teacher could not recover from this prefix — drop the splice.
                # We do NOT fall back to teacher's full rollout here: that would
                # be plain teacher imitation on exactly the problems where the
                # student's distribution sits farthest from the teacher's, which
                # tends to drag the student toward verbose 8B style without any
                # "self-correct" signal.
                continue

            # Prefer the shortest correct continuation (more decisive).
            candidates.sort(key=len)
            train_records.append({
                "type":               "correction",
                "pool_idx":           pool_idx,
                "problem":            rec["problem"],
                "gt_answer":          gt,
                "student_prefix":     rec["student_prefix"],
                "teacher_correction": candidates[0],
                "splice_ep_idx":      rec["splice_ep_idx"],
                "n_candidates_ok":    len(candidates),
            })
            n_correction += 1

        destroy_llm(llm)
    else:
        print("  [Teacher worker] no splices this epoch — only background SFT "
              "will be emitted.", flush=True)

    # ── Background SFT mixin ──────────────────────────────────────────────
    if BACKGROUND_RATIO > 0 and pool_has_rollout:
        n_bg_target = int(BACKGROUND_RATIO * len(pool))
        remaining = [i for i in range(len(pool))
                     if i not in covered_pool_idx
                     and pool[i].get("teacher_rollout", "").strip()]
        rng = random.Random(SEED + epoch_idx)
        n_bg = min(n_bg_target, len(remaining))
        bg_idx = rng.sample(remaining, n_bg) if n_bg > 0 else []
        for i in bg_idx:
            train_records.append({
                "type":             "background",
                "pool_idx":         i,
                "problem":          pool[i]["problem"],
                "gt_answer":        pool[i]["gt_answer"],
                "teacher_response": pool[i]["teacher_rollout"],
                "weight_scale":     BACKGROUND_WEIGHT,
            })
            n_background += 1

    # Shuffle so DDP shards see a mix of correction + background.
    random.Random(SEED + epoch_idx + 1).shuffle(train_records)

    write_jsonl(train_jsonl_out, train_records)
    with open(out_path, "w") as f:
        json.dump({
            "n_train_records": len(train_records),
            "n_correction":    n_correction,
            "n_background":    n_background,
            "n_splices_total": len(splices),
        }, f)
    print(f"\n[Teacher worker] train records: total={len(train_records)}  "
          f"(correction={n_correction}, background={n_background})", flush=True)


# ─── Step 7: training subprocess (weighted-CE SFT, no teacher loaded) ────────

def _copy_args_json(src_dir, dst_dir):
    src_args = Path(src_dir) / "args.json"
    if src_args.exists():
        import shutil
        dst_args = Path(dst_dir) / "args.json"
        shutil.copy2(str(src_args), str(dst_args))
        return True
    return False


def _train_worker(cfg_path):
    """Per-epoch training subprocess. Launched via torchrun for DDP across N_GPUS.

    Handles two record types from the train JSONL:
      • "correction": prompt + (masked student_prefix) + (labeled teacher_correction).
                      First CORRECTION_WINDOW tokens of correction get weight
                      CORRECTION_WEIGHT, remaining = 1.0.
      • "background": prompt + (labeled teacher_response).  Weight =
                      record["weight_scale"] (BACKGROUND_WEIGHT, default 0.5).

    Two mutually-exclusive training modes (cfg["training_mode"]):
      • "sft" : weighted cross-entropy on labeled tokens. No teacher loaded.
      • "opd" : weighted KL between student and teacher distributions on labeled
                tokens. Teacher is loaded into the train subprocess. The KL is
                a linear mix of reverse and forward KL controlled by
                cfg["opd_reverse_kl_beta"]:
                   loss = β · KL(P_s ‖ P_t)  +  (1-β) · KL(P_t ‖ P_s)
                β=1.0 → pure reverse-KL (mode-seeking, sharper student);
                β=0.0 → pure forward-KL (mode-covering, often verbose).
    """
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    with open(cfg_path) as f:
        cfg = json.load(f)
    student_ckpt      = cfg["student_ckpt"]
    teacher_path      = cfg.get("teacher_model", TEACHER_MODEL)
    train_jsonl       = cfg["train_jsonl"]
    output_dir        = cfg["output_dir"]
    epoch_idx         = cfg["epoch_idx"]
    n_epochs          = cfg["n_epochs"]
    lr                = cfg["lr"]
    max_length        = cfg["max_length"]
    batch_size        = cfg["batch_size"]
    grad_accum        = cfg["grad_accum"]
    warmup_ratio      = cfg["warmup_ratio"]
    correction_w      = cfg["correction_weight"]
    correction_win    = cfg["correction_window"]
    training_mode     = cfg.get("training_mode", "sft")
    opd_beta          = float(cfg.get("opd_reverse_kl_beta", 0.7))
    opd_temp          = float(cfg.get("opd_temp", 1.0))

    if training_mode not in ("sft", "opd"):
        raise ValueError(f"training_mode must be 'sft' or 'opd', got {training_mode!r}")

    is_main    = int(os.environ.get("RANK", "0")) == 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    tokenizer = AutoTokenizer.from_pretrained(student_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

    class CorrectionDataset(Dataset):
        def __init__(self, path):
            raw = load_jsonl(path)
            self.examples = []
            counts = {"correction": 0, "background": 0,
                      "skipped_other": 0, "skipped_too_long": 0}
            for rec in raw:
                ex = self._encode(rec)
                if ex is None:
                    counts["skipped_other"] += 1
                elif ex == "too_long":
                    counts["skipped_too_long"] += 1
                else:
                    self.examples.append(ex)
                    rtype = rec.get("type", "correction")
                    if rtype in counts:
                        counts[rtype] += 1
            if is_main:
                print(f"  [dataset] kept {len(self.examples)} / {len(raw)}  "
                      f"(correction={counts['correction']}, "
                      f"background={counts['background']}, "
                      f"skip_other={counts['skipped_other']}, "
                      f"skip_too_long={counts['skipped_too_long']})", flush=True)

        def _encode(self, rec):
            rtype   = rec.get("type", "correction")
            problem = rec["problem"]

            # Same chat template as rollout / teacher worker, so the model sees
            # the identical prompt format at train and inference time.
            prompt_text = build_filter_prompt(problem, tokenizer)
            prompt_ids  = tokenizer.encode(prompt_text, add_special_tokens=False)

            if rtype == "correction":
                prefix     = rec.get("student_prefix", "")
                correction = rec.get("teacher_correction", "")
                if not correction.strip():
                    return None
                prefix_ids = (tokenizer.encode(prefix, add_special_tokens=False)
                              if prefix else [])
                resp_ids   = tokenizer.encode(correction, add_special_tokens=False)
                resp_ids.append(im_end_id)

                total_len = len(prompt_ids) + len(prefix_ids) + len(resp_ids)
                if total_len > max_length:
                    return "too_long"

                input_ids = prompt_ids + prefix_ids + resp_ids
                labels    = ([-100] * (len(prompt_ids) + len(prefix_ids))
                             + list(resp_ids))
                weights   = [0.0] * (len(prompt_ids) + len(prefix_ids))
                for i in range(len(resp_ids)):
                    weights.append(correction_w if i < correction_win else 1.0)

            elif rtype == "background":
                response = rec.get("teacher_response", "")
                if not response.strip():
                    return None
                scale    = float(rec.get("weight_scale", 1.0))
                resp_ids = tokenizer.encode(response, add_special_tokens=False)
                resp_ids.append(im_end_id)

                total_len = len(prompt_ids) + len(resp_ids)
                if total_len > max_length:
                    return "too_long"

                input_ids = prompt_ids + resp_ids
                labels    = [-100] * len(prompt_ids) + list(resp_ids)
                weights   = [0.0]  * len(prompt_ids) + [scale] * len(resp_ids)

            else:
                return None

            assert len(input_ids) == len(labels) == len(weights)
            if all(l == -100 for l in labels):
                return None
            return {"input_ids": input_ids, "labels": labels, "weights": weights}

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, lbl, wts, attn = [], [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = max_len - n
            ids.append(b["input_ids"] + [pad_id] * pad)
            lbl.append(b["labels"]    + [-100]   * pad)
            wts.append(b["weights"]   + [0.0]    * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels":         torch.tensor(lbl,  dtype=torch.long),
            "weights":        torch.tensor(wts,  dtype=torch.float32),
        }

    class CorrectionTrainer(Trainer):
        """Two modes of compute_loss, mutually exclusive:

          • mode == "sft" : weighted cross-entropy on labeled tokens.
          • mode == "opd" : weighted (β · reverse-KL  +  (1-β) · forward-KL)
                            between student and teacher distributions on the
                            same labeled tokens.

        Per-token weights and label masking are identical in both modes; only
        the loss formula and whether teacher is loaded changes.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = None
            self.training_mode = "sft"
            self.opd_beta      = 0.7
            self.opd_temp      = 1.0

        def _weighted_kl(self, s_logits_flat, t_logits_flat, wts_flat, w_sum):
            """β · KL(P_s‖P_t) + (1-β) · KL(P_t‖P_s),  weighted, chunked."""
            # Vocab-size align (no-op when student/teacher share tokenizer).
            vs, vt = s_logits_flat.size(-1), t_logits_flat.size(-1)
            if vs != vt:
                if vs < vt:
                    s_logits_flat = F.pad(s_logits_flat, (0, vt - vs),
                                          value=float("-inf"))
                else:
                    t_logits_flat = F.pad(t_logits_flat, (0, vs - vt),
                                          value=float("-inf"))

            T          = self.opd_temp
            beta       = self.opd_beta
            chunk_size = 512
            n          = s_logits_flat.size(0)
            total      = s_logits_flat.new_zeros(())

            for st in range(0, n, chunk_size):
                ed = min(st + chunk_size, n)
                s_log = F.log_softmax(s_logits_flat[st:ed] / T, dim=-1)
                t_log = F.log_softmax(t_logits_flat[st:ed] / T, dim=-1)

                if beta >= 1.0:
                    # Pure reverse KL: KL(P_s ‖ P_t).
                    kl = F.kl_div(input=t_log, target=s_log,
                                  log_target=True, reduction="none").sum(-1)
                elif beta <= 0.0:
                    # Pure forward KL: KL(P_t ‖ P_s).
                    kl = F.kl_div(input=s_log, target=t_log,
                                  log_target=True, reduction="none").sum(-1)
                else:
                    # Linear mix.
                    rkl = F.kl_div(input=t_log, target=s_log,
                                   log_target=True, reduction="none").sum(-1)
                    fkl = F.kl_div(input=s_log, target=t_log,
                                   log_target=True, reduction="none").sum(-1)
                    kl  = beta * rkl + (1.0 - beta) * fkl

                total = total + (kl * wts_flat[st:ed]).sum()
            return total / w_sum

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            wts    = inputs.pop("weights")
            labels = inputs["labels"]
            ids    = inputs["input_ids"]
            attn   = inputs["attention_mask"]

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

            logits_flat = logits[mask]
            labels_flat = sh_lbl[mask]
            wts_flat    = sh_wts[mask]
            w_sum       = wts_flat.sum().clamp_min(1.0)

            if self.training_mode == "sft":
                ce   = F.cross_entropy(logits_flat, labels_flat,
                                       reduction="none")
                loss = (ce * wts_flat).sum() / w_sum
            else:  # opd
                with torch.no_grad():
                    t_out = self.teacher_model(input_ids=ids, attention_mask=attn,
                                               use_cache=False)
                t_logits      = t_out.logits[:, :-1, :].contiguous()
                t_logits_flat = t_logits[mask].to(logits_flat.dtype)
                loss = self._weighted_kl(logits_flat, t_logits_flat,
                                         wts_flat, w_sum)

            if return_outputs:
                return loss, out
            return loss

    if is_main:
        print(f"\n[train worker] epoch {epoch_idx + 1}/{n_epochs}  "
              f"mode={training_mode}  lr={lr:.3e}  max_len={max_length}  "
              f"bs={batch_size}  ga={grad_accum}  "
              f"corr_w={correction_w}  corr_win={correction_win}", flush=True)
        if training_mode == "opd":
            if opd_beta >= 0.99:
                _label = "pure reverse-KL"
            elif opd_beta <= 0.01:
                _label = "pure forward-KL"
            elif abs(opd_beta - 0.5) < 1e-6:
                _label = "balanced (≈JSD)"
            else:
                _label = f"{opd_beta:.2f} reverse + {1.0 - opd_beta:.2f} forward"
            print(f"  [opd] β={opd_beta}  T={opd_temp}  ({_label})", flush=True)
        print(f"  student_ckpt = {student_ckpt}", flush=True)
        print(f"  teacher      = {teacher_path}  "
              f"(loaded={'yes' if training_mode == 'opd' else 'no'})", flush=True)
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

    train_dataset = CorrectionDataset(train_jsonl)
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

    trainer = CorrectionTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
        tokenizer=tokenizer,
    )
    trainer.training_mode = training_mode
    trainer.opd_beta      = opd_beta
    trainer.opd_temp      = opd_temp

    # Load teacher AFTER trainer init so distributed env (LOCAL_RANK, cuda
    # device) is set up. Teacher lives on each rank's local GPU, frozen.
    if training_mode == "opd":
        if is_main:
            print(f"  [train worker] loading teacher (rank {local_rank}) for OPD …",
                  flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        teacher.config.use_cache = False
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        teacher.to(trainer.accelerator.device)
        trainer.teacher_model = teacher

    trainer.train()

    trainer.save_model(output_dir)
    if is_main:
        tokenizer.save_pretrained(output_dir)
        _copy_args_json(student_ckpt, output_dir)
        print(f"  [train worker] saved → {output_dir}", flush=True)

    # Best-effort cleanup before the process exits.
    trainer.teacher_model = None
    del trainer, student
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()


# ─── Step 8: post-training eval (same as weighted-distill) ───────────────────

def _epoch_eval_worker(cfg_path, out_path):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(cfg_path) as f:
        cfg = json.load(f)
    eval_ckpt     = cfg["eval_ckpt"]
    epoch_idx     = cfg["epoch_idx"]
    n_epochs      = cfg["n_epochs"]
    is_last_epoch = (epoch_idx + 1 >= n_epochs)

    student_tok   = AutoTokenizer.from_pretrained(eval_ckpt, trust_remote_code=True)
    filter_params = SamplingParams(n=1, temperature=0, max_tokens=MAX_NEW_TOKENS)

    llm = make_llm(eval_ckpt)

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

    print("\n[Step 8] Held-out evaluations (post-training) …", flush=True)
    test_records = load_jsonl(TEST_SET_JSONL) if TEST_SET_JSONL.exists() else []
    train_eval_records = (
        load_jsonl(TRAIN_EVAL_JSONL) if TRAIN_EVAL_JSONL.exists() else []
    )
    if TRAIN_EVAL_SOLVED.exists():
        with open(TRAIN_EVAL_SOLVED) as f:
            solved_train = set(json.load(f))
    else:
        solved_train = set()

    test_correct = test_total = 0
    train_correct = train_total = 0
    newly_solved_train = []

    should_test_eval = (
        TEST_EVAL_EVERY > 0
        and ((epoch_idx + 1) % TEST_EVAL_EVERY == 0 or is_last_epoch)
    )
    if test_records and should_test_eval:
        if (TEST_EVAL_SAMPLES > 0 and not is_last_epoch
                and TEST_EVAL_SAMPLES < len(test_records)):
            rng = random.Random(SEED)
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
            "eval_train", train_eval_records, skip_idx=solved_train,
        )
        print(f"  train: {train_correct}/{train_total} = "
              f"{train_correct / max(train_total, 1):.3f}  "
              f"(newly solved: {len(newly_solved_train)})", flush=True)

    destroy_llm(llm)

    eval_info = {
        "test_correct":       test_correct,
        "test_total":         test_total,
        "train_correct":      train_correct,
        "train_total":        train_total,
        "newly_solved_train": newly_solved_train,
    }
    with open(out_path, "w") as f:
        json.dump(eval_info, f)
    print(f"[eval worker] Done → {out_path}", flush=True)


# ─── Orchestrator ────────────────────────────────────────────────────────────

def run_epoch(epoch_idx, current_ckpt):
    """One outer epoch:
       1. student worker → splices.jsonl
       2. teacher worker → train.jsonl
       3. train worker   → ckpt
       4. eval worker    → eval log
    Returns the new student checkpoint path.
    """
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch_idx + 1}/{N_EPOCHS}  |  "
          f"ckpt={os.path.basename(current_ckpt)}")
    print(f"{'='*60}", flush=True)

    epoch_tag      = f"epoch_{epoch_idx:03d}"
    splices_jsonl  = OUTPUT_DIR / f"{epoch_tag}_splices.jsonl"
    train_jsonl    = OUTPUT_DIR / f"{epoch_tag}_train.jsonl"
    student_cfg    = OUTPUT_DIR / f"{epoch_tag}_student_cfg.json"
    student_out    = OUTPUT_DIR / f"{epoch_tag}_student_out.json"
    teacher_cfg    = OUTPUT_DIR / f"{epoch_tag}_teacher_cfg.json"
    teacher_out    = OUTPUT_DIR / f"{epoch_tag}_teacher_out.json"

    # ── Steps 1-4: student worker ──────────────────────────────────────────
    if student_out.exists() and splices_jsonl.exists():
        print(f"\n[Steps 1-4] Cached student output found — skipping.", flush=True)
    else:
        with open(student_cfg, "w") as f:
            json.dump({
                "current_ckpt":      current_ckpt,
                "splices_jsonl_out": str(splices_jsonl),
            }, f)
        print(f"\n[Steps 1-4] Spawning student worker …", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--_student-worker",
             str(student_cfg), str(student_out)],
            check=True,
        )
    with open(student_out) as f:
        st_info = json.load(f)
    print(f"\n[orchestrator] student worker: {st_info}", flush=True)

    # Note: even if n_splice_records == 0, the teacher worker still emits
    # background SFT records (when BACKGROUND_RATIO > 0), so we always proceed.
    # The teacher_out check below will catch the case where nothing was produced.

    # ── Steps 5-6: teacher worker ──────────────────────────────────────────
    if teacher_out.exists() and train_jsonl.exists():
        print(f"\n[Steps 5-6] Cached teacher output found — skipping.", flush=True)
    else:
        with open(teacher_cfg, "w") as f:
            json.dump({
                "splices_jsonl":   str(splices_jsonl),
                "train_jsonl_out": str(train_jsonl),
                "epoch_idx":       epoch_idx,
            }, f)
        print(f"\n[Steps 5-6] Spawning teacher worker …", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--_teacher-worker",
             str(teacher_cfg), str(teacher_out)],
            check=True,
        )
    with open(teacher_out) as f:
        tc_info = json.load(f)
    print(f"\n[orchestrator] teacher worker: {tc_info}", flush=True)
    n_train_records = tc_info.get("n_train_records", 0)

    if n_train_records == 0:
        print("  [orchestrator] no training records → skipping training",
              flush=True)
        return current_ckpt

    # ── Step 7: training subprocess (torchrun for DDP) ─────────────────────
    ckpt_dir       = CKPT_BASE_DIR / f"epoch_{epoch_idx + 1:03d}"
    train_cfg_path = OUTPUT_DIR / f"{epoch_tag}_train_cfg.json"

    if (ckpt_dir / "config.json").exists():
        print(f"\n[Step 7] Checkpoint {ckpt_dir} already exists — skipping.",
              flush=True)
    else:
        lr_this_epoch = _lr_for_epoch(GKD_LR, epoch_idx, N_EPOCHS, GKD_LR_SCHEDULE)
        print(f"\n  [LR] epoch {epoch_idx + 1}/{N_EPOCHS}: lr={lr_this_epoch:.3e}  "
              f"(base={GKD_LR}, schedule={GKD_LR_SCHEDULE})", flush=True)

        with open(train_cfg_path, "w") as f:
            json.dump({
                "student_ckpt":         current_ckpt,
                "teacher_model":        TEACHER_MODEL,
                "train_jsonl":          str(train_jsonl),
                "output_dir":           str(ckpt_dir),
                "epoch_idx":            epoch_idx,
                "n_epochs":             N_EPOCHS,
                "lr":                   lr_this_epoch,
                "max_length":           GKD_MAX_LEN,
                "batch_size":           GKD_BATCH_SIZE,
                "grad_accum":           GKD_GRAD_ACCUM,
                "warmup_ratio":         GKD_WARMUP,
                "correction_weight":    CORRECTION_WEIGHT,
                "correction_window":    CORRECTION_WINDOW,
                "training_mode":        TRAINING_MODE,
                "opd_reverse_kl_beta":  OPD_REVERSE_KL_BETA,
                "opd_temp":             OPD_TEMP,
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
        print(f"\n[Step 7] Spawning training worker  "
              f"(nproc_per_node={nproc}, master_port={master_port}) …", flush=True)
        subprocess.run(cmd, check=True)
        print(f"[orchestrator] Training done → {ckpt_dir}", flush=True)

    # ── Step 8: post-training eval ─────────────────────────────────────────
    eval_cfg_path = OUTPUT_DIR / f"{epoch_tag}_eval_cfg.json"
    eval_out_path = OUTPUT_DIR / f"{epoch_tag}_eval_out.json"

    if eval_out_path.exists():
        print(f"\n[Step 8] Cached eval found — skipping.", flush=True)
        with open(eval_out_path) as f:
            eval_info = json.load(f)
    else:
        with open(eval_cfg_path, "w") as f:
            json.dump({
                "eval_ckpt": str(ckpt_dir),
                "epoch_idx": epoch_idx,
                "n_epochs":  N_EPOCHS,
            }, f)
        print(f"\n[Step 8] Spawning eval worker …", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--_eval-worker",
             str(eval_cfg_path), str(eval_out_path)],
            check=True,
        )
        with open(eval_out_path) as f:
            eval_info = json.load(f)

    if eval_info and (eval_info.get("test_total", 0)
                      + eval_info.get("train_total", 0)) > 0:
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
            "n_train_records":    n_train_records,
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
    print("process_reward_correction_distill.py")
    print(f"  STUDENT_MODEL          = {STUDENT_MODEL}")
    print(f"  TEACHER_MODEL          = {TEACHER_MODEL}")
    print(f"  POOL_DATASET           = {POOL_DATASET}")
    print(f"  CORRECTION_WEIGHT      = {CORRECTION_WEIGHT}")
    print(f"  CORRECTION_WINDOW      = {CORRECTION_WINDOW}")
    print(f"  SPLICE_AT              = {SPLICE_AT}")
    print(f"  MERGE_SEGMENTS         = {MERGE_SEGMENTS}")
    print(f"  MIN_NEG_PR             = {MIN_NEG_PR}  "
          f"(merged-segment PR must be < -{MIN_NEG_PR:.2f})")
    print(f"  K_EVAL                 = {K_EVAL}")
    print(f"  K_CORRECTIONS          = {K_CORRECTIONS}")
    print(f"  CORRECTION_TEMPERATURE = {CORRECTION_TEMPERATURE}")
    print(f"  BACKGROUND_RATIO       = {BACKGROUND_RATIO}")
    print(f"  BACKGROUND_WEIGHT      = {BACKGROUND_WEIGHT}")
    print(f"  TRAINING_MODE          = {TRAINING_MODE}  "
          f"({'no teacher loaded' if TRAINING_MODE == 'sft' else 'teacher loaded for KL'})")
    if TRAINING_MODE == "opd":
        print(f"  OPD_REVERSE_KL_BETA    = {OPD_REVERSE_KL_BETA}")
        print(f"  OPD_TEMP               = {OPD_TEMP}")
    print(f"  N_EPOCHS               = {N_EPOCHS}")
    print(f"  N_GPUS                 = {N_GPUS}")
    print(f"  GKD_LR                 = {GKD_LR}  (schedule={GKD_LR_SCHEDULE})")
    print(f"  GKD_BATCH_SIZE         = {GKD_BATCH_SIZE}  ga={GKD_GRAD_ACCUM}")
    print(f"  GKD_MAX_LEN            = {GKD_MAX_LEN}")
    print(f"  N_SAMPLES              = {'all' if N_SAMPLES == 0 else N_SAMPLES}")
    print(f"  OUTPUT_DIR             = {OUTPUT_DIR}")
    print("=" * 60, flush=True)

    pool = phase0_build_pool()
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
        current_ckpt = run_epoch(epoch, current_ckpt)

    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"  Final checkpoint : {current_ckpt}")
    print(f"  Pool size        : {len(pool)}  (frozen since phase 0)")
    print(f"{'='*60}", flush=True)


# ─── CLI dispatch ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--_student-worker":
        _epoch_student_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--_teacher-worker":
        _epoch_teacher_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--_eval-worker":
        _epoch_eval_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--_train-worker":
        _train_worker(sys.argv[2])
    else:
        main()
