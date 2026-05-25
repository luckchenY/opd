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

Per-epoch pipeline
------------------
  Step 1  vLLM (student) : rollout each pool problem to a response
  Step 2  CPU            : segment each response (trigger words + period count)
  Step 3  vLLM (student) : K-fold PR eval per segment + empty-prefix baseline
                           (written into a single rollouts.jsonl file)
  Step 4  HF (s + t)     : per-token KL on teacher Top-K support, aggregated
                           into length-weighted normKL per MERGED paragraph
                           (monotonic-run merge over per-segment PR)
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
  OPD_REVERSE_KL_BETA         beta in  L = beta*KL(s||t) + (1-beta)*KL(t||s).
                              Default 0.7 (reverse-leaning, mode-seeking).
  OPD_TEMP                    Softmax temperature in the KL. Default 1.0.
  K_EVAL                      K-fold count per segment in PR eval. Default 8.
  N_EPOCHS                    Outer epochs. Default 5.
  GKD_LR / GKD_BATCH_SIZE / GKD_GRAD_ACCUM / GKD_MAX_LEN / GKD_WARMUP /
  GKD_LR_SCHEDULE             Standard training knobs.
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

GPU_IDS = [4, 5, 6, 7]

# ─── Shared helpers from the base distillation pipeline ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from process_reward_distill import (  # noqa: E402
    DATASET, DATASET_CONFIGS, STUDENT_MODEL, TEACHER_MODEL, N_GPUS,
    EVAL_TEMPERATURE, MAX_ANSWER_TOKENS, MAX_NEW_TOKENS,
    ROLLOUT_TEMPERATURE, EPOCH_MAX_MODEL_LEN,
    EPISODE_EVAL_SYSTEM, INFERENCE_BATCH_SIZE,
    SEED, N_SAMPLES,
    extract_boxed_answer, answers_match,
    load_dataset_samples, segment_response_text, merge_monotonic_runs,
    write_jsonl, load_jsonl,
    build_filter_prompt, build_episode_eval_prompt,
    make_llm, destroy_llm, safe_generate, run_isolated_rollout,
)

# Override K_EVAL: process_reward_distill defaults to 4. Threshold-based
# PR ordering is too noisy at K=4 (per-PR std ~0.35); 8 gives ~0.25.
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
#   "hendrycks_math" | "math"  → Hendrycks et al. MATH (7 subsets, train+test)
#                                ("math" is a convenience alias for the same data)
#   "dapo_math_17k"           → DAPO-Math-17k (~17k, train only)
POOL_DATASET = os.environ.get("POOL_DATASET", "hendrycks_math").lower()
if POOL_DATASET == "math":
    POOL_DATASET = "hendrycks_math"
if POOL_DATASET not in ("hendrycks_math", "dapo_math_17k"):
    raise ValueError(f"POOL_DATASET must be 'math', 'hendrycks_math' or "
                     f"'dapo_math_17k', got {POOL_DATASET!r}")

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


def _load_pool_samples_normalized():
    if POOL_DATASET == "dapo_math_17k":
        return _load_dapo_samples_normalized()
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


# ─── Phase 0: build static T✓ pool (teacher-greedy filter only) ──────────────

def phase0_build_pool():
    print(f"\n{'='*60}")
    print("Phase 0: Building static T✓ sample pool")
    print(f"{'='*60}", flush=True)

    if POOL_JSONL.exists():
        print(f"  Existing pool found at {POOL_JSONL}, loading …", flush=True)
        pool = load_jsonl(POOL_JSONL)
        print(f"  Loaded {len(pool)} samples.", flush=True)
        return pool

    samples = _load_pool_samples_normalized()
    n_total = len(samples)

    print(f"\n  [0] Teacher greedy filter ({n_total} problems) …", flush=True)
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
    for s, text in zip(samples, teacher_texts):
        gt = s["gt_answer"]
        pred = extract_boxed_answer(text)
        if answers_match(pred, gt):
            pool.append({"problem": s["problem"], "gt_answer": gt})
    print(f"  Teacher: {len(pool)}/{n_total} correct  →  pool size = {len(pool)}",
          flush=True)
    write_jsonl(POOL_JSONL, pool)

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

    workset = load_jsonl(workset_jsonl)
    if not workset:
        write_jsonl(rollouts_jsonl, [])
        with open(out_path, "w") as f:
            json.dump({"n_rollouts": 0}, f)
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

    inf_bs    = INFERENCE_BATCH_SIZE if INFERENCE_BATCH_SIZE > 0 else len(workset)
    n_batches = max(1, (len(workset) + inf_bs - 1) // inf_bs)

    print(f"\n[Student worker] workset={len(workset)}  inf_batch={inf_bs}  "
          f"n_batches={n_batches}", flush=True)

    rollouts = []
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

        responses    = []
        response_ids = []
        for out in rollout_outs:
            if out is None or not out.outputs:
                responses.append("")
                response_ids.append([])
                continue
            o = out.outputs[0]
            responses.append(o.text if o.text else "")
            response_ids.append(list(o.token_ids) if o.text else [])

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

        if kfold_prompts:
            kfold_outs = safe_generate(
                llm, kfold_prompts, kfold_params, student_tok,
                EPOCH_MAX_MODEL_LEN, label="kfold",
            )
        else:
            kfold_outs = []

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
            rec_base = {
                "work_idx":    b_start + li,
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

        print(f"    rollouts_so_far={len(rollouts)}  "
              f"correct={n_correct}  empty={n_empty}", flush=True)

    destroy_llm(llm)
    write_jsonl(rollouts_jsonl, rollouts)
    with open(out_path, "w") as f:
        json.dump({
            "n_rollouts":      len(rollouts),
            "n_correct":       n_correct,
            "n_empty":         n_empty,
        }, f)
    print(f"\n[Student worker] rollouts written: {len(rollouts)}  "
          f"(correct={n_correct}, empty={n_empty})", flush=True)


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


def _epoch_kl_worker(cfg_path, out_path):
    """HF-based per-token KL. Loads student + teacher; for each rollout
    re-tokenises (prompt + response) and computes top-K-support KL per
    response position. Emits per-merged-paragraph normKL_fwd / normKL_rev /
    normKL_mix into kl.jsonl.

    The retokenisation strategy:
      • prompt: tokenize(build_filter_prompt(problem)) — matches training
      • response: use response_token_ids directly from the rollout (EXACT,
                  on-policy)
    This guarantees the IDs we feed both models match what vLLM produced.
    """
    import torch.nn.functional as F
    from collections import defaultdict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(cfg_path) as f:
        cfg = json.load(f)
    student_ckpt   = cfg["student_ckpt"]
    teacher_path   = cfg["teacher_model"]
    rollouts_jsonl = cfg["rollouts_jsonl"]
    kl_jsonl_out   = cfg["kl_jsonl_out"]
    beta           = float(cfg.get("opd_reverse_kl_beta", OPD_REVERSE_KL_BETA))
    topk           = int(cfg.get("kl_topk", KL_TOPK))

    rollouts = load_jsonl(rollouts_jsonl)
    if not rollouts:
        write_jsonl(kl_jsonl_out, [])
        with open(out_path, "w") as f:
            json.dump({"n_with_kl": 0}, f)
        return

    print(f"\n[KL worker] processing {len(rollouts)} rollouts  "
          f"top-K={topk}  beta={beta}", flush=True)

    # ── GPU layout: student on GPU 0, teacher on the rest if available. ──
    visible = torch.cuda.device_count()
    if visible >= 2:
        s_gpus, t_gpus = [0], list(range(1, visible))
    else:
        s_gpus = t_gpus = [0]

    def _device_map(model_name, gpu_ids):
        if len(gpu_ids) == 1:
            return f"cuda:{gpu_ids[0]}"
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        with init_empty_weights():
            empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        max_mem = {}
        for i in gpu_ids:
            gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            max_mem[i] = f"{max(1, int(gib * 0.82))}GiB"
        max_mem["cpu"] = "32GiB"
        dmap = infer_auto_device_map(empty, max_memory=max_mem)
        del empty
        return dmap

    print(f"  Loading teacher ({os.path.basename(teacher_path)}) on GPUs {t_gpus} …",
          flush=True)
    t_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
        device_map=_device_map(teacher_path, t_gpus),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    print(f"  Loading student ({os.path.basename(student_ckpt)}) on GPUs {s_gpus} …",
          flush=True)
    s_model = AutoModelForCausalLM.from_pretrained(
        student_ckpt,
        torch_dtype=torch.bfloat16,
        device_map=_device_map(student_ckpt, s_gpus),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    s_dev = f"cuda:{s_gpus[0]}"
    t_dev = f"cuda:{t_gpus[0]}"

    student_tok = AutoTokenizer.from_pretrained(student_ckpt, trust_remote_code=True)

    def _forward(model, ids, device):
        ids = ids.to(device)
        with torch.no_grad():
            try:
                logits = model(input_ids=ids).logits[0].cpu()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                CHUNK = 1024
                seq_len = ids.shape[1]
                parts = []
                for st in range(0, seq_len, CHUNK):
                    ed = min(st + CHUNK, seq_len)
                    out = model(input_ids=ids[:, :ed]).logits[0, st:ed].cpu()
                    parts.append(out)
                    torch.cuda.empty_cache()
                logits = torch.cat(parts, dim=0)
        return logits

    kl_records = []
    n_with_kl  = 0
    n_skipped  = 0

    for ri, rec in enumerate(rollouts):
        # ── Skip records the student worker already flagged ──────────────
        if rec.get("skipped"):
            n_skipped += 1
            rec_out = dict(rec)
            rec_out["merged_paragraphs"] = []
            kl_records.append(rec_out)
            continue

        response_ids = rec.get("response_token_ids") or []
        if not response_ids or not rec.get("segments"):
            n_skipped += 1
            rec_out = dict(rec)
            rec_out["merged_paragraphs"] = []
            rec_out["skipped"]    = True
            rec_out["skip_reason"] = rec_out.get("skip_reason", "no_response_or_segments")
            kl_records.append(rec_out)
            continue

        # ── Build prompt_ids + response_ids exactly as training will. ────
        prompt_text = build_filter_prompt(rec["problem"], student_tok)
        prompt_ids  = student_tok.encode(prompt_text, add_special_tokens=False)

        full_ids   = torch.tensor(prompt_ids + list(response_ids),
                                  dtype=torch.long).unsqueeze(0)
        seq_len    = full_ids.shape[1]
        prompt_len = len(prompt_ids)
        resp_len   = seq_len - prompt_len

        if resp_len <= 1:
            n_skipped += 1
            rec_out = dict(rec)
            rec_out["merged_paragraphs"] = []
            rec_out["skipped"]    = True
            rec_out["skip_reason"] = "resp_too_short"
            kl_records.append(rec_out)
            continue

        # ── Forward both models, get response-position logits. ───────────
        s_logits_all = _forward(s_model, full_ids, s_dev)
        t_logits_all = _forward(t_model, full_ids, t_dev)
        s_resp = s_logits_all[prompt_len - 1: seq_len - 1].float()   # [resp_len, V_s]
        t_resp = t_logits_all[prompt_len - 1: seq_len - 1].float()   # [resp_len, V_t]
        del s_logits_all, t_logits_all

        # ── Align vocabs by clipping to the smaller vocab. Identical
        # tokenisers (Qwen/Qwen) → no-op. Mismatched → safe lower bound;
        # teacher's TopK may then exclude tokens above min(V_s, V_t), which
        # is the conservative thing to do.
        Vs = s_resp.size(-1)
        Vt = t_resp.size(-1)
        V = min(Vs, Vt)
        if Vs != V: s_resp = s_resp[:, :V]
        if Vt != V: t_resp = t_resp[:, :V]

        K = min(topk, V)
        t_topk = t_resp.topk(K, dim=-1)
        topk_idx      = t_topk.indices            # [resp_len, K]
        t_topk_logits = t_topk.values             # [resp_len, K]
        s_topk_logits = s_resp.gather(-1, topk_idx)

        t_log = F.log_softmax(t_topk_logits, dim=-1)
        s_log = F.log_softmax(s_topk_logits, dim=-1)
        t_p   = t_log.exp()
        s_p   = s_log.exp()

        kl_fwd = (t_p * (t_log - s_log)).sum(-1).tolist()   # KL(q_hat || pi_hat)
        kl_rev = (s_p * (s_log - t_log)).sum(-1).tolist()   # KL(pi_hat || q_hat)
        del s_resp, t_resp, t_topk, topk_idx, t_topk_logits, s_topk_logits
        del t_log, s_log, t_p, s_p

        # ── Map each response token to its character range so we can
        # aggregate per merged paragraph. ────────────────────────────────
        tok_char_starts = [0] * resp_len
        tok_char_ends   = [0] * resp_len
        pos = 0
        for t in range(resp_len):
            piece = student_tok.decode([response_ids[t]])
            tok_char_starts[t] = pos
            pos += len(piece)
            tok_char_ends[t]   = pos

        # ── Build merged paragraphs from raw segments + ep_accs ──────────
        ep_accs   = {int(k): float(v) for k, v in rec.get("ep_accs", {}).items()}
        empty_acc = float(rec.get("empty_acc", 0.0))
        merged    = _build_merged_paragraphs(rec["segments"], ep_accs, empty_acc)

        # ── Per-paragraph length-weighted KL ─────────────────────────────
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
            fwd = sum(kl_fwd[t] for t in tok_indices) / len(tok_indices)
            rev = sum(kl_rev[t] for t in tok_indices) / len(tok_indices)
            mix = beta * rev + (1.0 - beta) * fwd
            para["normKL_fwd"] = float(max(0.0, fwd))
            para["normKL_rev"] = float(max(0.0, rev))
            para["normKL"]     = float(max(0.0, mix))
            para["tok_start"]  = int(tok_indices[0])
            para["tok_end"]    = int(tok_indices[-1] + 1)

        # ── Per-response stats (helpful for logging only) ────────────────
        rec_out = dict(rec)
        rec_out["prompt_len"]        = int(prompt_len)
        rec_out["resp_len"]          = int(resp_len)
        rec_out["merged_paragraphs"] = merged
        rec_out["kl_topk"]           = int(K)
        rec_out["opd_beta_used"]     = float(beta)
        kl_records.append(rec_out)
        n_with_kl += 1

        if (ri + 1) % 25 == 0 or ri == 0:
            n_para = len(merged)
            mu_mix = (sum(p["normKL"] for p in merged) / n_para) if n_para else 0.0
            print(f"  [{ri+1:4d}/{len(rollouts)}]  "
                  f"resp_len={resp_len:4d}  paras={n_para:2d}  "
                  f"mean_normKL_mix={mu_mix:.4f}", flush=True)

    del s_model, t_model
    gc.collect()
    torch.cuda.empty_cache()

    write_jsonl(kl_jsonl_out, kl_records)
    with open(out_path, "w") as f:
        json.dump({
            "n_with_kl":  n_with_kl,
            "n_skipped":  n_skipped,
            "n_total":    len(kl_records),
        }, f)
    print(f"\n[KL worker] kl written: {len(kl_records)}  "
          f"(with_kl={n_with_kl}, skipped={n_skipped})", flush=True)


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
                    "n_paragraphs":       n_segs,
                    "n_masked":           len(mask_indices),
                    "consistency":        round(a["consistency"], 6),
                    "n_pairs":            a["n_pairs"],
                    "paragraphs":         mask_para_summaries,
                    "retry_count":        int(rec.get("retry_count", 0)),
                    "source":             rec.get("source", "pool"),
                })
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
    n_kept = len(train_records)
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
    student_ckpt   = cfg["student_ckpt"]
    teacher_path   = cfg["teacher_model"]
    train_jsonl    = cfg["train_jsonl"]
    output_dir     = cfg["output_dir"]
    epoch_idx      = cfg["epoch_idx"]
    n_epochs       = cfg["n_epochs"]
    lr             = cfg["lr"]
    max_length     = cfg["max_length"]
    batch_size     = cfg["batch_size"]
    grad_accum     = cfg["grad_accum"]
    warmup_ratio   = cfg["warmup_ratio"]
    opd_beta       = float(cfg.get("opd_reverse_kl_beta", OPD_REVERSE_KL_BETA))
    opd_temp       = float(cfg.get("opd_temp", OPD_TEMP))
    kl_topk        = int(cfg.get("kl_topk", KL_TOPK))

    is_main    = int(os.environ.get("RANK", "0")) == 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    tokenizer = AutoTokenizer.from_pretrained(student_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

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
            if not problem or not response_ids or not weights_resp:
                return None
            if len(weights_resp) != len(response_ids):
                return None

            prompt_text = build_filter_prompt(problem, tokenizer)
            prompt_ids  = tokenizer.encode(prompt_text, add_special_tokens=False)

            resp_ids    = list(response_ids) + [im_end_id]
            resp_wts    = list(weights_resp) + [1.0]

            total_len = len(prompt_ids) + len(resp_ids)
            if total_len > max_length:
                return "too_long"

            input_ids = prompt_ids + resp_ids
            labels    = [-100] * len(prompt_ids) + list(resp_ids)
            weights   = [0.0]  * len(prompt_ids) + [float(w) for w in resp_wts]

            if all(w == 0.0 for w in weights):
                return "all_zero"
            return {"input_ids": input_ids, "labels": labels, "weights": weights}

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return self.examples[i]

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

    class OPDConsistencyTrainer(Trainer):
        """OPD loss with top-K teacher support and per-token weights.

        For position t with student logits s_t ∈ R^V and teacher logits t_t,
        we form S_t = TopK_q(t_t) and let
            ŝ_t  = softmax( s_t[S_t] / T )
            t̂_t  = softmax( t_t[S_t] / T )
        Then
            loss_t = β · KL(ŝ_t ‖ t̂_t)  +  (1 − β) · KL(t̂_t ‖ ŝ_t)
            L      = (Σ_t w_t · loss_t) / max(1, Σ_t w_t)
        where w_t is the per-token weight from the dataset (0 on masked
        positions). Only positions with labels != -100 enter the sum.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = None
            self.opd_beta = OPD_REVERSE_KL_BETA
            self.opd_temp = OPD_TEMP
            self.kl_topk  = KL_TOPK

        def _topk_kl(self, s_logits_flat, t_logits_flat, wts_flat, w_sum):
            T = self.opd_temp
            beta = self.opd_beta
            K = self.kl_topk
            chunk_size = 512
            n = s_logits_flat.size(0)
            total = s_logits_flat.new_zeros(())

            # Vocab align (no-op when student/teacher share tokenizer).
            Vs = s_logits_flat.size(-1)
            Vt = t_logits_flat.size(-1)
            V = min(Vs, Vt)
            if Vs != V:
                s_logits_flat = s_logits_flat[:, :V]
            if Vt != V:
                t_logits_flat = t_logits_flat[:, :V]
            K = min(K, V)

            for st in range(0, n, chunk_size):
                ed = min(st + chunk_size, n)
                s_logits = s_logits_flat[st:ed] / T
                t_logits = t_logits_flat[st:ed] / T

                t_topk   = t_logits.topk(K, dim=-1)
                topk_idx = t_topk.indices                  # [b, K]
                t_top    = t_topk.values                   # [b, K]
                s_top    = s_logits.gather(-1, topk_idx)   # [b, K]

                t_log = F.log_softmax(t_top, dim=-1)
                s_log = F.log_softmax(s_top, dim=-1)

                # F.kl_div(input=log_q, target=log_p, log_target=True) ≈
                # sum_v exp(log_p) * (log_p - log_q)  =  KL(P || Q)
                if beta >= 1.0:
                    # KL(P_s || P_t) — reverse, mode-seeking
                    kl = F.kl_div(input=t_log, target=s_log,
                                  log_target=True, reduction="none").sum(-1)
                elif beta <= 0.0:
                    # KL(P_t || P_s) — forward, mode-covering
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
            wts_flat    = sh_wts[mask]
            w_sum       = wts_flat.sum().clamp_min(1.0)

            if self.teacher_model is None:
                raise RuntimeError("OPD requires teacher_model to be loaded.")

            with torch.no_grad():
                t_out = self.teacher_model(input_ids=ids, attention_mask=attn,
                                           use_cache=False)
            t_logits      = t_out.logits[:, :-1, :].contiguous()
            t_logits_flat = t_logits[mask].to(logits_flat.dtype)

            loss = self._topk_kl(logits_flat, t_logits_flat, wts_flat, w_sum)

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
        print(f"  teacher      = {teacher_path}  (loaded for KL)", flush=True)
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

    trainer = OPDConsistencyTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
        tokenizer=tokenizer,
    )
    trainer.opd_beta = opd_beta
    trainer.opd_temp = opd_temp
    trainer.kl_topk  = kl_topk

    # Load teacher AFTER trainer init so distributed env is wired up.
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

    trainer.teacher_model = None
    del trainer, student, teacher
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

    print("\n[Step 7] Held-out evaluations (post-training) …", flush=True)
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
    if rollouts_jsonl.exists() and student_out.exists():
        print(f"\n[Steps 1-3] Cached rollouts found — skipping.", flush=True)
    else:
        with open(student_cfg, "w") as f:
            json.dump({
                "current_ckpt":       current_ckpt,
                "workset_jsonl":      str(workset_jsonl),
                "rollouts_jsonl_out": str(rollouts_jsonl),
            }, f)
        print(f"\n[Steps 1-3] Spawning student worker …", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--_student-worker",
             str(student_cfg), str(student_out)],
            check=True,
        )
    with open(student_out) as f:
        st_info = json.load(f)
    print(f"\n[orchestrator] student worker: {st_info}", flush=True)

    # ── Step 4: KL worker (HF student + teacher, top-K support) ──────────
    if kl_jsonl.exists() and kl_out.exists():
        print(f"\n[Step 4] Cached KL found — skipping.", flush=True)
    else:
        with open(kl_cfg, "w") as f:
            json.dump({
                "student_ckpt":         current_ckpt,
                "teacher_model":        TEACHER_MODEL,
                "rollouts_jsonl":       str(rollouts_jsonl),
                "kl_jsonl_out":         str(kl_jsonl),
                "opd_reverse_kl_beta":  OPD_REVERSE_KL_BETA,
                "kl_topk":              KL_TOPK,
            }, f)
        print(f"\n[Step 4] Spawning KL worker …", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--_kl-worker",
             str(kl_cfg), str(kl_out)],
            check=True,
        )
    with open(kl_out) as f:
        kl_info = json.load(f)
    print(f"\n[orchestrator] KL worker: {kl_info}", flush=True)

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
    n_train_records = dispatch_stats.get("n_kept", 0)

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
        with open(eval_cfg_path, "w") as f:
            json.dump({
                "eval_ckpt": str(ckpt_dir),
                "epoch_idx": epoch_idx,
                "n_epochs":  N_EPOCHS,
            }, f)
        print(f"\n[Step 7] Spawning eval worker …", flush=True)
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
            "dispatch":           dispatch_stats,
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
          f"(teacher-defined support set, renormalised)")
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
        current_ckpt = run_epoch(epoch, current_ckpt, pool)

    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"  Final checkpoint : {current_ckpt}")
    print(f"  Pool size        : {len(pool)}  (frozen since phase 0)")
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
