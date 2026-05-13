"""
process_reward_weighted_distill.py
==================================
Static-pool, segment-weighted GKD distillation.

Differences from process_reward_distill.py
------------------------------------------
* Pool is built ONCE (Phase 0) by sampling N problems and filtering out those
  the teacher cannot solve. The pool is frozen — no removal of "learned"
  samples, no continuation-based growth, no easy-sample replay.
* No three-branch training-record construction. Each pool item produces ONE
  training record per epoch: full student response + per-segment weights.
* Loss is segment-weighted JSD between student and teacher distributions:
      weight_i = 1 + alpha * pr_i
  Positive-PR segments (student reasoning that raised accuracy) are
  up-weighted; negative-PR segments (that hurt accuracy) are down-weighted,
  so we don't reinforce teacher's behavior at student's bad prefixes.
* Training is a custom HF Trainer (this file's --_train-worker), not swift rlhf,
  because per-token loss weighting requires hooking into the loss directly.

Per-epoch loop
--------------
  Step 0  vLLM   : held-out evaluations (test_set + train_eval_set)
  Step 1  vLLM   : fresh student rollout on every pool problem
  Step 2  CPU    : episode segmentation
  Step 3  vLLM   : K-fold PR eval per episode boundary (incl. empty-prefix)
  Step 4  CPU    : compute PR + (optional) monotonic merge + per-segment weight,
                   write epoch_NNN_train.jsonl
  Step 5  proc   : torchrun --_train-worker → weighted JSD training, save ckpt

Important env vars
------------------
  PR_ALPHA           weight = 1 + alpha * pr.  Default 1.0.
  MERGE_SEGMENTS     true|false.  Default true (collapse monotonic runs).
  N_EPOCHS           Number of outer epochs.  Default 5.
  GKD_LR / GKD_BETA / GKD_BATCH_SIZE / GKD_GRAD_ACCUM / GKD_MAX_LEN
                     Standard GKD knobs.  Defaults match process_reward_distill.
  GKD_LR_SCHEDULE    cosine|linear|constant for outer-epoch peak-LR scaling.
  TEST_EVAL_EVERY / TEST_EVAL_SAMPLES
                     Held-out test cadence and subsampling.
  N_SAMPLES          Number of pool problems (0 = full set).
  POOL_DATASET       hendrycks_math (default) | dapo_math_17k.
                     hendrycks_math : MATH train ~7500, has test split.
                     dapo_math_17k  : DAPO-Math-17k (~17000, train only).
                                      MATH-test eval is skipped — run
                                      AIME via evalscope externally.
  DAPO_SHARDS        Optional ":" -separated absolute paths to DAPO arrow
                     shards if your local cache lives elsewhere.
  OUTPUT_DIR         Output directory.

Usage
-----
    # MATH (default)
    python tools/process_reward_weighted_distill.py

    # DAPO-Math-17k
    POOL_DATASET=dapo_math_17k \
    OUTPUT_DIR=$PWD/tools/weighted_distill_output_dapo \
        python tools/process_reward_weighted_distill.py

    PR_ALPHA=1.0 MERGE_SEGMENTS=true N_EPOCHS=3 \
        python tools/process_reward_weighted_distill.py
"""

import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# ─── Import shared helpers from the original script ──────────────────────────
# (Phase-0 helpers, segmentation, vLLM lifecycle, isolated rollout, etc.)
sys.path.insert(0, str(Path(__file__).parent))
from process_reward_distill import (  # noqa: E402
    DATASET, DATASET_CONFIGS, STUDENT_MODEL, TEACHER_MODEL, N_GPUS,
    K_EVAL, EVAL_TEMPERATURE, MAX_ANSWER_TOKENS, MAX_NEW_TOKENS,
    ROLLOUT_TEMPERATURE, MAX_PROMPT_PHASE0, EPOCH_MAX_MODEL_LEN,
    EPISODE_EVAL_SYSTEM, INFERENCE_BATCH_SIZE,
    SEED, N_SAMPLES,
    extract_boxed_answer, answers_match,
    load_dataset_samples, segment_response_text, merge_monotonic_runs,
    write_jsonl, load_jsonl,
    build_filter_prompt, build_episode_eval_prompt,
    make_llm, destroy_llm, safe_generate, run_isolated_rollout,
)

# ─── New config knobs ────────────────────────────────────────────────────────

PR_ALPHA        = float(os.environ.get("PR_ALPHA", "0.5"))
MERGE_SEGMENTS  = os.environ.get("MERGE_SEGMENTS", "true").lower() == "true"

N_EPOCHS        = int(os.environ.get("N_EPOCHS", "15"))
KEEP_LAST_CKPTS = int(os.environ.get("KEEP_LAST_CKPTS", "5"))

GKD_LR          = float(os.environ.get("GKD_LR", "1e-6"))
GKD_BETA        = float(os.environ.get("GKD_BETA", "0.5"))
GKD_BATCH_SIZE  = int(os.environ.get("GKD_BATCH_SIZE", "1"))
GKD_GRAD_ACCUM  = int(os.environ.get("GKD_GRAD_ACCUM", "4"))
GKD_MAX_LEN     = int(os.environ.get("GKD_MAX_LEN", str(MAX_NEW_TOKENS + 1024)))
GKD_TEMPERATURE = float(os.environ.get("GKD_TEMPERATURE", "1.0"))
GKD_WARMUP      = float(os.environ.get("GKD_WARMUP", "0.05"))
GKD_LR_SCHEDULE = os.environ.get("GKD_LR_SCHEDULE", "cosine")  # cosine|linear|constant

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
    "OUTPUT_DIR", str(_DIR / "weighted_distill_output")))
POOL_JSONL         = OUTPUT_DIR / "pool.jsonl"             # static, never mutated
TEST_SET_JSONL     = OUTPUT_DIR / "test_set.jsonl"
TRAIN_EVAL_JSONL   = OUTPUT_DIR / "train_eval_set.jsonl"
TRAIN_EVAL_SOLVED  = OUTPUT_DIR / "train_eval_solved.json"
EVAL_LOG_JSONL     = OUTPUT_DIR / "eval_log.jsonl"
CKPT_BASE_DIR      = OUTPUT_DIR / "checkpoints"


# ─── Outer LR schedule ───────────────────────────────────────────────────────

def _lr_for_epoch(base_lr, epoch_idx, n_epochs, schedule):
    """Outer LR schedule across epochs. The inner Trainer.train() uses
    `constant_with_warmup` (warmup to peak_i, then flat — no inner decay),
    so this outer schedule is the *only* LR-decay path. Avoids double
    decay (outer-cosine × inner-cosine)."""
    if schedule == "constant" or n_epochs <= 1:
        return base_lr
    progress = epoch_idx / n_epochs
    if schedule == "linear":
        return base_lr * max(0.0, 1.0 - progress)
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown GKD_LR_SCHEDULE: {schedule}")


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

    Note: HF's release of DAPO-Math-17k has 1,791,700 rows = ~17,398 unique
    problems × ~103 duplicates each. We dedupe by problem text so the pool
    reflects the ~17k unique problems advertised by the dataset name.
    """
    from pyarrow import ipc

    pool = []
    seen = set()
    n_raw = 0
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
                n_raw += 1
                key = str(problem).strip()
                if key in seen:
                    continue
                seen.add(key)
                pool.append({"problem": str(problem), "gt_answer": str(gt)})

    total = len(pool)
    print(f"  Read {n_raw} rows → {total} unique problems "
          f"(deduped {n_raw - total}).", flush=True)

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


# ─── Phase 0: build static pool (T✓ only) ────────────────────────────────────

def phase0_build_pool():
    """One-time: sample → teacher rollout → keep only T✓ → save POOL_JSONL.

    The pool is frozen for the entire run. Each pool entry = {problem, gt_answer}
    only — student responses are regenerated fresh each epoch.
    """
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

    # Teacher rollout in an isolated subprocess so its CUDA context is fully
    # destroyed afterwards (leaving GPU clean for the per-epoch worker).
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
                "problem":   s["problem"],
                "gt_answer": gt,
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

    # Snapshot pool as fixed train-eval reference.
    if not TRAIN_EVAL_JSONL.exists():
        train_eval_records = [
            {"problem": item["problem"], "gt_answer": item["gt_answer"]}
            for item in pool
        ]
        write_jsonl(TRAIN_EVAL_JSONL, train_eval_records)
        with open(TRAIN_EVAL_SOLVED, "w") as f:
            json.dump([], f)

    return pool


# ─── Per-segment weighting helpers ───────────────────────────────────────────

def compute_segments_for_response(episodes, acc_rates, empty_acc, do_merge):
    """Build per-segment list with end_char, acc_rate, process_reward, weight.

    `episodes`   : list of (start_char, end_char) from segment_response_text.
    `acc_rates`  : dict {ep_idx: acc_rate}. ep_idx == -1 not used here (passed in
                   separately via empty_acc).
    `empty_acc`  : acc_rate from the empty-prefix K-fold (baseline).
    `do_merge`   : if True, collapse monotonic runs and recompute PR as
                   cumulative change across each merged run.

    Returns list of dicts: [{"end_char", "acc_rate", "process_reward", "weight"}].
    """
    points = []
    prev_acc = empty_acc
    for ep_idx, (_sc, end_c) in enumerate(episodes):
        curr_acc = acc_rates.get(ep_idx, 0.0)
        points.append({
            "end_char":       end_c,
            "acc_rate":       round(curr_acc, 6),
            "process_reward": round(curr_acc - prev_acc, 6),
        })
        prev_acc = curr_acc

    if do_merge and len(points) > 1:
        points = merge_monotonic_runs(points)
        # After merging, recompute PR as the cumulative acc_rate change from the
        # previous merged point (or from empty_acc for the first one). This way
        # the merged segment's weight reflects total impact, not just the last
        # single-step change.
        prev_acc = empty_acc
        for p in points:
            p["process_reward"] = round(p["acc_rate"] - prev_acc, 6)
            prev_acc = p["acc_rate"]

    for p in points:
        p["weight"] = round(max(0.0, 1.0 + PR_ALPHA * p["process_reward"]), 6)
    return points


# ─── Inference subprocess (Step 0–4) ─────────────────────────────────────────

def _epoch_inference_worker(cfg_path, out_path):
    """vLLM-only subprocess. Runs student rollout, K-fold PR eval, builds
    weighted training JSONL. Exits cleanly so its CUDA context is fully freed
    before the training subprocess starts.
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(cfg_path) as f:
        cfg = json.load(f)
    current_ckpt    = cfg["current_ckpt"]
    train_jsonl_out = cfg["train_jsonl_out"]

    pool = load_jsonl(POOL_JSONL)
    student_tok = AutoTokenizer.from_pretrained(current_ckpt, trust_remote_code=True)

    if not pool:
        write_jsonl(train_jsonl_out, [])
        with open(out_path, "w") as f:
            json.dump({"n_train_records": 0}, f)
        return

    rollout_params = SamplingParams(
        n=1, temperature=ROLLOUT_TEMPERATURE,
        top_k=50, top_p=1.0, max_tokens=MAX_NEW_TOKENS,
    )
    kfold_params   = SamplingParams(
        n=K_EVAL, temperature=EVAL_TEMPERATURE,
        top_k=50, top_p=1.0,
        max_tokens=MAX_ANSWER_TOKENS, stop=["<|im_end|>"],
    )

    llm = make_llm(current_ckpt)

    # ── Steps 1–4 in mini-batches ────────────────────────────────────────
    inf_bs    = INFERENCE_BATCH_SIZE if INFERENCE_BATCH_SIZE > 0 else len(pool)
    n_batches = max(1, (len(pool) + inf_bs - 1) // inf_bs)

    train_records = []
    n_with_segments = 0
    n_skipped_empty = 0

    print(f"\n[Steps 1–4] Inference  "
          f"(pool={len(pool)}, inf_batch={inf_bs}, n_batches={n_batches})",
          flush=True)

    for b_idx in range(n_batches):
        b_start = b_idx * inf_bs
        b_end   = min(b_start + inf_bs, len(pool))
        batch   = pool[b_start:b_end]

        print(f"\n  [mini-batch {b_idx + 1}/{n_batches}]  "
              f"samples {b_start}–{b_end - 1}", flush=True)

        # ── Step 1: student rollout ─────────────────────────────────────
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

        # ── Step 2: segmentation (CPU) ──────────────────────────────────
        episodes_per = [segment_response_text(r) if r.strip() else []
                        for r in responses]

        # ── Step 3: K-fold PR eval (incl. empty prefix) ─────────────────
        kfold_prompts, kfold_meta = [], []
        for li, (item, resp, episodes) in enumerate(
            zip(batch, responses, episodes_per)
        ):
            if not resp.strip() or not episodes:
                continue
            # empty-prefix baseline
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

        # accumulate per-(li, ep_idx) accuracy
        acc_per_li = {}   # {li: {ep_idx_or_-1: acc_rate}}
        for (li, ep_idx), out in zip(kfold_meta, kfold_outs):
            if out is None:
                continue
            gt = batch[li]["gt_answer"]
            corrects = [
                answers_match(extract_boxed_answer(o.text), gt)
                for o in out.outputs
            ]
            acc_per_li.setdefault(li, {})[ep_idx] = sum(corrects) / K_EVAL

        # ── Step 4: build weighted segments + emit train records ────────
        for li, (item, resp, episodes) in enumerate(
            zip(batch, responses, episodes_per)
        ):
            if not resp.strip() or not episodes:
                n_skipped_empty += 1
                continue
            li_accs   = acc_per_li.get(li, {})
            empty_acc = li_accs.get(-1, 0.0)
            ep_accs   = {k: v for k, v in li_accs.items() if k != -1}

            segments = compute_segments_for_response(
                episodes, ep_accs, empty_acc, do_merge=MERGE_SEGMENTS,
            )
            train_records.append({
                "problem":          item["problem"],
                "gt_answer":        item["gt_answer"],
                "student_response": resp,
                "empty_acc":        round(empty_acc, 6),
                "segments":         segments,
            })
            n_with_segments += 1

        print(f"    rollouts={len(rollout_prompts)}  "
              f"kfold_prompts={len(kfold_prompts)}  "
              f"records_so_far={len(train_records)}  "
              f"skipped_empty={n_skipped_empty}", flush=True)

    destroy_llm(llm)

    write_jsonl(train_jsonl_out, train_records)
    print(f"\n[Step 4] Train records written: {len(train_records)}  "
          f"(skipped_empty={n_skipped_empty})", flush=True)

    with open(out_path, "w") as f:
        json.dump({"n_train_records": len(train_records)}, f)
    print(f"[inference worker] Done → {out_path}", flush=True)


# ─── Eval subprocess (post-training held-out evaluation) ─────────────────────

def _epoch_eval_worker(cfg_path, out_path):
    """vLLM-only subprocess. Evaluates the trained student checkpoint on
    held-out test and train-eval sets. Runs *after* training so the metrics
    reflect the freshly trained model. Exits cleanly.
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(cfg_path) as f:
        cfg = json.load(f)
    eval_ckpt  = cfg["eval_ckpt"]
    epoch_idx  = cfg["epoch_idx"]
    n_epochs   = cfg["n_epochs"]
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

    print("\n[Step 6] Held-out evaluations (post-training) …", flush=True)
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


# ─── Training subprocess (custom HF Trainer with weighted JSD) ───────────────

def _copy_args_json(src_dir, dst_dir):
    """Copy args.json from source checkpoint if it exists (for SWIFT compatibility)."""
    src_args = Path(src_dir) / "args.json"
    if src_args.exists():
        import shutil
        dst_args = Path(dst_dir) / "args.json"
        shutil.copy2(str(src_args), str(dst_args))
        return True
    return False


def _train_worker(cfg_path):
    """Per-epoch training subprocess. Launched via torchrun for DDP across N_GPUS.

    Loads student + teacher, runs HF Trainer with a custom compute_loss that
    applies per-token weights (carried in the data collator) to JSD between
    student and teacher distributions.
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
    beta           = cfg["beta"]
    temperature    = cfg["temperature"]
    max_length     = cfg["max_length"]
    batch_size     = cfg["batch_size"]
    grad_accum     = cfg["grad_accum"]
    warmup_ratio   = cfg["warmup_ratio"]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main    = int(os.environ.get("RANK", "0")) == 0

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(student_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

    # ── Dataset: pre-encode (input_ids, labels, weights) ────────────────
    class WeightedGKDDataset(Dataset):
        def __init__(self, path):
            raw = load_jsonl(path)
            self.examples = []
            n_skipped = 0
            for rec in raw:
                ex = self._encode(rec)
                if ex is None:
                    n_skipped += 1
                    continue
                self.examples.append(ex)
            if is_main:
                print(f"  [dataset] kept {len(self.examples)} / {len(raw)} "
                      f"(skipped {n_skipped} for prompt-too-long or empty)",
                      flush=True)

        def _encode(self, rec):
            problem  = rec["problem"]
            response = rec["student_response"]
            segments = rec["segments"]
            if not segments or not response:
                return None

            prompt_msgs = [
                {"role": "system", "content": EPISODE_EVAL_SYSTEM},
                {"role": "user",   "content": problem},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(prompt_ids) >= max_length:
                return None

            # Tokenize response with offset mapping → map each token to a segment.
            enc = tokenizer(response, return_offsets_mapping=True,
                            add_special_tokens=False)
            response_ids = list(enc["input_ids"])
            offsets      = enc["offset_mapping"]

            sorted_segs = sorted(segments, key=lambda s: s["end_char"])
            tok_weights = []
            seg_idx = 0
            for tok_start, _ in offsets:
                while (seg_idx < len(sorted_segs) - 1
                       and sorted_segs[seg_idx]["end_char"] <= tok_start):
                    seg_idx += 1
                tok_weights.append(float(sorted_segs[seg_idx]["weight"]))

            # Append im_end (termination token, content-independent → weight 1.0).
            response_ids.append(im_end_id)
            tok_weights.append(1.0)

            input_ids = prompt_ids + response_ids
            labels    = [-100] * len(prompt_ids) + response_ids
            weights   = [0.0]  * len(prompt_ids) + tok_weights

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels    = labels[:max_length]
                weights   = weights[:max_length]

            # If truncation removed every label token, drop the example.
            if all(l == -100 for l in labels):
                return None

            return {"input_ids": input_ids, "labels": labels, "weights": weights}

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    # ── Collator: pad input_ids/labels/weights to batch max length ──────
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

    # ── Weighted JSD loss (token-chunked to bound peak memory) ──────────
    def weighted_jsd(student_logits, teacher_logits, labels, weights,
                     beta, temperature, chunk_size=512):
        # Predict token at i+1 from logits at i.
        s_logits = student_logits[:, :-1, :].contiguous()
        t_logits = teacher_logits[:, :-1, :].contiguous()
        sh_lbl   = labels[:, 1:].contiguous()
        sh_wts   = weights[:, 1:].contiguous()

        mask = sh_lbl != -100
        s = s_logits[mask]
        t = t_logits[mask].to(s.dtype)
        w = sh_wts[mask].to(s.dtype)

        n = s.size(0)
        if n == 0:
            return s.new_zeros(())

        # Vocab alignment.  When student and teacher have *different* vocab
        # sizes (e.g. Qwen2.5-1.5B has 151936 vs Qwen2.5-14B has 152064), we
        # restrict both distributions to the COMMON prefix and let
        # ``log_softmax`` re-normalise.  Padding the smaller side with -inf
        # would propagate ``0 * -inf = NaN`` through ``F.kl_div`` (because
        # ``exp(s_log)*(s_log - m_log)`` is computed elementwise, including at
        # the padded -inf positions).
        vs, vt = s.size(-1), t.size(-1)
        if vs != vt:
            common_v = min(vs, vt)
            s = s[..., :common_v]
            t = t[..., :common_v]

        total = s.new_zeros(())

        if beta == 0:
            for st in range(0, n, chunk_size):
                ed = min(st + chunk_size, n)
                s_log = F.log_softmax(s[st:ed] / temperature, dim=-1)
                t_log = F.log_softmax(t[st:ed] / temperature, dim=-1)
                kl = F.kl_div(s_log, t_log, reduction="none",
                              log_target=True).sum(-1)
                total = total + (kl * w[st:ed]).sum()
        elif beta == 1:
            for st in range(0, n, chunk_size):
                ed = min(st + chunk_size, n)
                s_log = F.log_softmax(s[st:ed] / temperature, dim=-1)
                t_log = F.log_softmax(t[st:ed] / temperature, dim=-1)
                kl = F.kl_div(t_log, s_log, reduction="none",
                              log_target=True).sum(-1)
                total = total + (kl * w[st:ed]).sum()
        else:
            log_b   = math.log(beta)
            log_1mb = math.log(1.0 - beta)
            for st in range(0, n, chunk_size):
                ed = min(st + chunk_size, n)
                s_log = F.log_softmax(s[st:ed] / temperature, dim=-1)
                t_log = F.log_softmax(t[st:ed] / temperature, dim=-1)
                m_log = torch.logsumexp(
                    torch.stack([s_log + log_1mb, t_log + log_b]), dim=0)
                kl_t = F.kl_div(m_log, t_log, reduction="none",
                                log_target=True).sum(-1)
                kl_s = F.kl_div(m_log, s_log, reduction="none",
                                log_target=True).sum(-1)
                jsd = beta * kl_t + (1.0 - beta) * kl_s
                total = total + (jsd * w[st:ed]).sum()
        return total / max(n, 1)

    # ── Custom Trainer ──────────────────────────────────────────────────
    class WeightedGKDTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = None   # set after init
            self.beta_ = beta
            self.temperature_ = temperature

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            wts    = inputs.pop("weights")
            labels = inputs["labels"]
            ids    = inputs["input_ids"]
            attn   = inputs["attention_mask"]

            student_out = model(input_ids=ids, attention_mask=attn,
                                use_cache=False)
            with torch.no_grad():
                teacher_out = self.teacher_model(input_ids=ids,
                                                 attention_mask=attn,
                                                 use_cache=False)
            loss = weighted_jsd(student_out.logits, teacher_out.logits,
                                labels, wts, self.beta_, self.temperature_)
            # Safety net: if the loss went non-finite (NaN/Inf) for any reason
            # (e.g. extreme logits, vocab edge case), zero the loss for this
            # step so backward yields zero gradients instead of poisoning the
            # weights with NaN.  Logged once per occurrence.
            if not torch.isfinite(loss):
                if int(os.environ.get("RANK", "0")) == 0:
                    print(f"  [train worker] WARN: non-finite loss "
                          f"({loss.item()!r}) in step — zeroing this batch.",
                          flush=True)
                loss = student_out.logits.sum() * 0.0
            if return_outputs:
                return loss, student_out
            return loss

    # ── Build student / teacher / dataset / trainer ─────────────────────
    if is_main:
        print(f"\n[train worker] epoch {epoch_idx + 1}/{n_epochs}  "
              f"lr={lr:.3e}  beta={beta}  T={temperature}  "
              f"max_len={max_length}  bs={batch_size}  ga={grad_accum}",
              flush=True)
        print(f"  student_ckpt = {student_ckpt}", flush=True)
        print(f"  teacher      = {teacher_path}", flush=True)
        print(f"  train_jsonl  = {train_jsonl}", flush=True)
        print(f"  output_dir   = {output_dir}", flush=True)

    # Choose attention impl: prefer flash_attention_2 if available, else sdpa.
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

    train_dataset = WeightedGKDDataset(train_jsonl)
    if len(train_dataset) == 0:
        if is_main:
            print("  [train worker] no training records — skipping training",
                  flush=True)
        # Save the unchanged student so the orchestrator can pick it up.
        if is_main:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            student.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            # Copy args.json for SWIFT compatibility
            if _copy_args_json(student_ckpt, output_dir):
                print(f"  [train worker] copied args.json from {student_ckpt}",
                      flush=True)
        return

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="constant_with_warmup",  # inner: warmup-only, no decay
        warmup_ratio=warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",       # save manually after train()
        logging_steps=1,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,   # keep "weights" column
        dataloader_num_workers=2,
        seed=SEED + epoch_idx,
    )

    trainer = WeightedGKDTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
        tokenizer=tokenizer,
    )

    # Load teacher AFTER trainer init so distributed env (LOCAL_RANK, cuda
    # device) is set up. Teacher lives on each rank's local GPU, frozen.
    if is_main:
        print(f"  [train worker] loading teacher on rank {local_rank} …",
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

    # Save final student weights into output_dir directly (no "checkpoint-N").
    trainer.save_model(output_dir)
    if is_main:
        tokenizer.save_pretrained(output_dir)
        # Copy args.json for SWIFT compatibility
        if _copy_args_json(student_ckpt, output_dir):
            print(f"  [train worker] copied args.json from {student_ckpt}",
                  flush=True)
        print(f"  [train worker] saved → {output_dir}", flush=True)

    # Best-effort cleanup before the process exits.
    del trainer, student, teacher
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()


# ─── Orchestrator: run_epoch ─────────────────────────────────────────────────

def run_epoch(epoch_idx, current_ckpt):
    """One outer epoch:
       1. Spawn vLLM inference subprocess  → produces epoch_NNN_train.jsonl
       2. Spawn torchrun training subprocess → saves new ckpt
       3. Append eval log
    Returns the new student checkpoint path.
    """
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch_idx + 1}/{N_EPOCHS}  |  ckpt={os.path.basename(current_ckpt)}")
    print(f"{'='*60}", flush=True)

    epoch_tag      = f"epoch_{epoch_idx:03d}"
    inf_cfg_path   = OUTPUT_DIR / f"{epoch_tag}_inf_cfg.json"
    inf_out_path   = OUTPUT_DIR / f"{epoch_tag}_inf_out.json"
    train_jsonl    = OUTPUT_DIR / f"{epoch_tag}_train.jsonl"

    # ── Step 0–4: inference subprocess ──────────────────────────────────
    if inf_out_path.exists() and train_jsonl.exists():
        print(f"\n[Steps 0–4] Cached inference found — skipping re-inference.",
              flush=True)
    else:
        with open(inf_cfg_path, "w") as f:
            json.dump({
                "current_ckpt":    current_ckpt,
                "train_jsonl_out": str(train_jsonl),
            }, f)
        print(f"\n[Steps 0–4] Spawning inference worker …", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--_epoch-worker",
             str(inf_cfg_path), str(inf_out_path)],
            check=True,
        )

    with open(inf_out_path) as f:
        inf_results = json.load(f)
    n_train_records = inf_results.get("n_train_records", 0)
    print(f"\n[orchestrator] Inference done. n_train_records={n_train_records}",
          flush=True)

    if n_train_records == 0:
        print("  [orchestrator] no training records → skipping training",
              flush=True)
        return current_ckpt

    # ── Step 5: training subprocess (torchrun for DDP) ──────────────────
    ckpt_dir       = CKPT_BASE_DIR / f"epoch_{epoch_idx + 1:03d}"
    train_cfg_path = OUTPUT_DIR / f"{epoch_tag}_train_cfg.json"

    if (ckpt_dir / "config.json").exists():
        print(f"\n[Step 5] Checkpoint {ckpt_dir} already exists — skipping training.",
              flush=True)
    else:
        lr_this_epoch = _lr_for_epoch(GKD_LR, epoch_idx, N_EPOCHS, GKD_LR_SCHEDULE)
        print(f"\n  [LR] epoch {epoch_idx + 1}/{N_EPOCHS}: lr={lr_this_epoch:.3e}  "
              f"(base={GKD_LR}, schedule={GKD_LR_SCHEDULE})", flush=True)

        with open(train_cfg_path, "w") as f:
            json.dump({
                "student_ckpt":  current_ckpt,
                "teacher_model": TEACHER_MODEL,
                "train_jsonl":   str(train_jsonl),
                "output_dir":    str(ckpt_dir),
                "epoch_idx":     epoch_idx,
                "n_epochs":      N_EPOCHS,
                "lr":            lr_this_epoch,
                "beta":          GKD_BETA,
                "temperature":   GKD_TEMPERATURE,
                "max_length":    GKD_MAX_LEN,
                "batch_size":    GKD_BATCH_SIZE,
                "grad_accum":    GKD_GRAD_ACCUM,
                "warmup_ratio":  GKD_WARMUP,
            }, f)

        # Pick a free-ish master port per epoch to avoid collisions on reruns.
        master_port = 29500 + (epoch_idx % 100)
        nproc       = max(1, N_GPUS)

        print(f"\n[Step 5] Spawning torchrun training worker  "
              f"(nproc_per_node={nproc}, master_port={master_port}) …", flush=True)
        if nproc == 1:
            cmd = [sys.executable, __file__, "--_train-worker", str(train_cfg_path)]
        else:
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc}",
                f"--master_port={master_port}",
                __file__, "--_train-worker", str(train_cfg_path),
            ]
        subprocess.run(cmd, check=True)
        print(f"[orchestrator] Training done → {ckpt_dir}", flush=True)

    # ── Step 6: post-training eval subprocess ───────────────────────────
    eval_cfg_path = OUTPUT_DIR / f"{epoch_tag}_eval_cfg.json"
    eval_out_path = OUTPUT_DIR / f"{epoch_tag}_eval_out.json"

    if eval_out_path.exists():
        print(f"\n[Step 6] Cached eval found — skipping re-eval.", flush=True)
        with open(eval_out_path) as f:
            eval_info = json.load(f)
    else:
        with open(eval_cfg_path, "w") as f:
            json.dump({
                "eval_ckpt": str(ckpt_dir),
                "epoch_idx": epoch_idx,
                "n_epochs":  N_EPOCHS,
            }, f)
        print(f"\n[Step 6] Spawning eval worker …", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--_eval-worker",
             str(eval_cfg_path), str(eval_out_path)],
            check=True,
        )
        with open(eval_out_path) as f:
            eval_info = json.load(f)

    # Persist eval results to log.
    if eval_info and (eval_info.get("test_total", 0) + eval_info.get("train_total", 0)) > 0:
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


# ─── Main ────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint():
    """Scan CKPT_BASE_DIR for the highest epoch_NNN/ that contains a valid
    model checkpoint (config.json present).  Returns (epoch_idx_done, ckpt_path)
    where epoch_idx_done is the epoch index that *produced* the checkpoint
    (i.e. the next epoch to run is epoch_idx_done + 1).  Returns (-1, None)
    when no valid checkpoint is found.
    """
    if not CKPT_BASE_DIR.exists():
        return -1, None
    best_ep, best_path = -1, None
    for d in sorted(CKPT_BASE_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name  # e.g. "epoch_001"
        if not name.startswith("epoch_"):
            continue
        try:
            ep_num = int(name[len("epoch_"):])
        except ValueError:
            continue
        if (d / "config.json").exists():
            if ep_num > best_ep:
                best_ep, best_path = ep_num, d
    if best_ep < 0:
        return -1, None
    # checkpoint epoch_NNN was produced by epoch_idx = NNN - 1
    return best_ep - 1, str(best_path)


def _prune_old_checkpoints(keep_last):
    """Keep only the newest `keep_last` epoch_NNN checkpoints."""
    if keep_last <= 0 or not CKPT_BASE_DIR.exists():
        return

    ckpt_dirs = []
    for d in CKPT_BASE_DIR.iterdir():
        if not d.is_dir() or not d.name.startswith("epoch_"):
            continue
        try:
            ep_num = int(d.name[len("epoch_"):])
        except ValueError:
            continue
        ckpt_dirs.append((ep_num, d))

    if len(ckpt_dirs) <= keep_last:
        return

    ckpt_dirs.sort(key=lambda x: x[0], reverse=True)
    to_delete = ckpt_dirs[keep_last:]
    for ep_num, path in to_delete:
        shutil.rmtree(path, ignore_errors=False)
        print(f"  [ckpt cleanup] removed old checkpoint epoch_{ep_num:03d}: {path}",
              flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("process_reward_weighted_distill.py")
    print(f"  STUDENT_MODEL    = {STUDENT_MODEL}")
    print(f"  TEACHER_MODEL    = {TEACHER_MODEL}")
    print(f"  PR_ALPHA         = {PR_ALPHA}")
    print(f"  MERGE_SEGMENTS   = {MERGE_SEGMENTS}")
    print(f"  N_EPOCHS         = {N_EPOCHS}")
    print(f"  N_GPUS           = {N_GPUS}")
    print(f"  K_EVAL           = {K_EVAL}")
    print(f"  GKD_LR           = {GKD_LR}  (schedule={GKD_LR_SCHEDULE})")
    print(f"  GKD_BETA         = {GKD_BETA}")
    print(f"  GKD_TEMPERATURE  = {GKD_TEMPERATURE}")
    print(f"  GKD_BATCH_SIZE   = {GKD_BATCH_SIZE}  ga={GKD_GRAD_ACCUM}")
    print(f"  GKD_MAX_LEN      = {GKD_MAX_LEN}")
    print(f"  N_SAMPLES        = {'all' if N_SAMPLES == 0 else N_SAMPLES}")
    print(f"  POOL_DATASET     = {POOL_DATASET}")
    print(f"  OUTPUT_DIR       = {OUTPUT_DIR}")
    print("=" * 60, flush=True)

    pool = phase0_build_pool()

    if not pool:
        print("\nPool is empty — nothing to train on. Exiting.", flush=True)
        return

    # ── Resume: find the latest completed checkpoint ──────────────────────
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
        _prune_old_checkpoints(KEEP_LAST_CKPTS)

    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"  Final checkpoint : {current_ckpt}")
    print(f"  Pool size        : {len(pool)}  (frozen since phase 0)")
    print(f"  Keep last ckpts  : {KEEP_LAST_CKPTS}")
    print(f"{'='*60}", flush=True)


# ─── CLI dispatch ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--_epoch-worker":
        _epoch_inference_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--_eval-worker":
        _epoch_eval_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--_train-worker":
        _train_worker(sys.argv[2])
    else:
        main()
