"""
process_reward_kl_merged.py  —  Process Reward × Normalised KL (Monotonic Merge)
==================================================================================

Identical to process_reward_kl.py through Phase 4.  Phase 5 adds a
monotonic-run merging step before the scatter analysis:

  For each case, the per-episode sequence of process rewards is inspected
  for consecutive non-decreasing or non-increasing sub-sequences.
  Equal (flat) steps never trigger a cut — they extend whichever run is
  currently in progress.  A run boundary is created only when a strict
  sign flip occurs (strictly increasing → strictly decreasing, or vice
  versa).  Each run is collapsed into the *last* episode of the run.

  Example (rewards along episodes 3..8):
      raw:    0.10  0.20  0.20  0.10  0.10  0.25
              [  non-decreasing ↑  ][  non-increasing ↓  ][ ↑ ]
      merged: 0.20                  0.10                   0.25
              (ep5)                 (ep7)                  (ep8)

Phases 1–4 and Phase 6 are unchanged from process_reward_kl.py.

Usage:
    python tools/process_reward_kl_merged.py
    EVAL_MODEL=/path/to/model python tools/process_reward_kl_merged.py
"""

import bisect
import gc
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from scipy import stats as scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("WARNING: scipy not found; Pearson p-values will be NaN.", flush=True)

# ── config ────────────────────────────────────────────────────────────────────
STUDENT_MODEL = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen"
    "/models/Qwen3-1.7B"
)
TEACHER_MODEL = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen"
    "/models/Qwen3-8B"
)
EVAL_MODEL = os.environ.get("EVAL_MODEL", STUDENT_MODEL)

DATASET = "hendrycks_math"
_CACHE_ROOT = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift"
    "/data/hf_datasets_cache"
)
DATASET_CONFIGS = {
    "hendrycks_math": {
        "arrow_shards": [
            f"{_CACHE_ROOT}/EleutherAI___hendrycks_math/{subset}/0.0.0"
            "/21a5633873b6a120296cce3e2df9d5550074f4a3/hendrycks_math-train.arrow"
            for subset in [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
            ]
        ],
        "prompt_key": "problem",
        "solution_key": "solution",
    },
}

N_SAMPLES      = 300
MAX_NEW_TOKENS = 8192
SEED           = 42
TEMPERATURE    = 1.0
TOP_K          = 50
TOP_P          = 1.0

# ── episode evaluation ────────────────────────────────────────────────────────
K_EVAL            = 8
EVAL_TEMPERATURE  = 0.7
MAX_ANSWER_TOKENS = 384
MIN_EVAL_EPISODE  = 3

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

EPISODE_EVAL_SYSTEM = "Put your final answer within \\boxed{}."
EPISODE_EVAL_SUFFIX = (
    "\nBased on the reasoning above, stop thinking now and summarize the existing"
    " reasoning. If a definitive answer can be derived, give the final answer in"
    " \\boxed{answer}. If no answer has been reached yet or the result is still"
    " inconclusive, output \\boxed{no answer}."
)

DTYPE  = torch.bfloat16
N_GPUS = torch.cuda.device_count()
_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(_DIR, "process_reward_kl_merged.json")
OUTPUT_HTML = os.path.join(_DIR, "process_reward_kl_merged.html")

# ── category helpers ──────────────────────────────────────────────────────────
CATEGORY_KEYS = ["both_correct", "both_wrong", "student_only", "teacher_only"]
CATEGORY_LABELS = {
    "both_correct": "Both Correct (S✓ T✓)",
    "both_wrong":   "Both Wrong (S✗ T✗)",
    "student_only": "Student Only (S✓ T✗)",
    "teacher_only": "Teacher Only (S✗ T✓)",
}
CATEGORY_COLORS = {
    "both_correct": "#4caf50",
    "both_wrong":   "#f44336",
    "student_only": "#ff9800",
    "teacher_only": "#2196f3",
}


def get_category(s_correct, t_correct):
    if s_correct and t_correct:
        return "both_correct"
    if not s_correct and not t_correct:
        return "both_wrong"
    if s_correct and not t_correct:
        return "student_only"
    return "teacher_only"


# ── answer helpers ────────────────────────────────────────────────────────────

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


# ── dataset helpers ───────────────────────────────────────────────────────────

def build_prompt(problem_text, tokenizer):
    msgs = [{"role": "user", "content": problem_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return problem_text


def load_dataset_samples():
    random.seed(SEED)
    cfg = DATASET_CONFIGS[DATASET]
    shards = cfg["arrow_shards"]
    print(f"Loading dataset '{DATASET}' ({len(shards)} shards) …", flush=True)
    pool = []
    for shard_path in shards:
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
    idx = random.sample(range(total), min(N_SAMPLES, total))
    samples = [pool[i] for i in idx]
    del pool
    print(f"  Sampled {len(samples)} / {total} rows", flush=True)
    return samples


# ── vLLM generation helper ────────────────────────────────────────────────────

def _vllm_generate(model_path, tokenizer, prompt_strings, do_sample=True,
                   max_tokens=None, temperature=None, stop=None,
                   n_completions=1):
    """Batch generation via vLLM. Supports n_completions > 1 for K-fold eval."""
    tp = min(N_GPUS, 4)
    print(f"  Initializing vLLM ({os.path.basename(model_path)}, tp={tp}) …",
          flush=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 1024,
        gpu_memory_utilization=0.85,
    )

    if max_tokens is None:
        max_tokens = MAX_NEW_TOKENS
    if temperature is None:
        temperature = TEMPERATURE if do_sample else 0

    sp_kwargs = dict(n=n_completions, max_tokens=max_tokens, temperature=temperature)
    if temperature > 0:
        sp_kwargs.update(top_k=TOP_K, top_p=TOP_P)
    if stop:
        sp_kwargs["stop"] = stop

    params = SamplingParams(**sp_kwargs)
    print(f"  Generating {len(prompt_strings)} prompts"
          f"{'  ×  n=' + str(n_completions) if n_completions > 1 else ''} …",
          flush=True)
    outputs = llm.generate(prompt_strings, params)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


# ── model loading helpers ─────────────────────────────────────────────────────

def _make_device_map(model_name, gpu_ids):
    if len(gpu_ids) == 1:
        return f"cuda:{gpu_ids[0]}"
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    max_mem = {i: "75GiB" for i in gpu_ids}
    max_mem["cpu"] = "32GiB"
    device_map = infer_auto_device_map(empty_model, max_memory=max_mem)
    del empty_model
    return device_map


def _forward_logits(model, input_ids, device):
    """Forward-pass; return logits on CPU. Handles long sequences with chunking."""
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    with torch.no_grad():
        try:
            logits = model(input_ids=input_ids).logits[0].cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    OOM on seq_len={seq_len}, falling back to chunked forward …",
                  flush=True)
            CHUNK = 1024
            all_logits = []
            for start in range(0, seq_len, CHUNK):
                end = min(start + CHUNK, seq_len)
                chunk_ids = input_ids[:, :end]
                out = model(input_ids=chunk_ids).logits[0, start:end].cpu()
                all_logits.append(out)
                torch.cuda.empty_cache()
            logits = torch.cat(all_logits, dim=0)
    return logits


# ── episode segmentation ──────────────────────────────────────────────────────

def _add_segment(segments, tokens, start, end):
    seg_toks = tokens[start:end]
    length = len(seg_toks)
    if length == 0:
        return
    kl_fwd_mean = sum(t["kl_forward"] for t in seg_toks) / length
    kl_rev_mean = sum(t["kl_reverse"] for t in seg_toks) / length
    text = "".join(t["text"] for t in seg_toks)
    segments.append(dict(
        start=start,
        end=end,
        length=length,
        kl_forward_mean=round(kl_fwd_mean, 6),
        kl_reverse_mean=round(kl_rev_mean, 6),
        text_preview=text[:200],
    ))


def segment_by_triggers(tokens, trigger_patterns=None, min_periods=None):
    if trigger_patterns is None:
        trigger_patterns = SEGMENT_TRIGGER_PATTERNS
    if min_periods is None:
        min_periods = SEGMENT_MIN_PERIODS

    combined = re.compile("|".join(trigger_patterns), re.IGNORECASE)

    tok_offsets = []
    pos = 0
    for tok in tokens:
        tok_offsets.append(pos)
        pos += len(tok["text"])
    full_text = "".join(tok["text"] for tok in tokens)

    trigger_token_set = set()
    for m in combined.finditer(full_text):
        ti = bisect.bisect_right(tok_offsets, m.start()) - 1
        ti = max(0, ti)
        trigger_token_set.add(ti)

    segments = []
    current_start = 0
    period_count = 0

    for ti, tok in enumerate(tokens):
        period_count += tok["text"].count(".")
        if (ti in trigger_token_set
                and period_count >= min_periods
                and ti > current_start):
            _add_segment(segments, tokens, current_start, ti)
            current_start = ti
            period_count = tok["text"].count(".")

    _add_segment(segments, tokens, current_start, len(tokens))
    return segments


# ── Phase 1: student on-policy sampling ──────────────────────────────────────

def phase1_student(samples):
    print(f"\n{'='*60}")
    print(f"Phase 1: Student sampling  ({os.path.basename(STUDENT_MODEL)}, "
          f"n={N_SAMPLES})")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    cfg = DATASET_CONFIGS[DATASET]
    prompt_strings = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]
    gt_answers = [extract_boxed_answer(s[cfg["solution_key"]]) for s in samples]

    gen_outputs = _vllm_generate(STUDENT_MODEL, tokenizer, prompt_strings,
                                 do_sample=True)

    results = []
    for si, (sample, out, gt_answer) in enumerate(
            zip(samples, gen_outputs, gt_answers)):
        prompt_ids = list(out.prompt_token_ids)
        gen_ids    = list(out.outputs[0].token_ids)
        full_ids   = torch.tensor(prompt_ids + gen_ids, dtype=torch.long)
        prompt_len = len(prompt_ids)

        response_ids   = full_ids[prompt_len:]
        response_text  = tokenizer.decode(response_ids, skip_special_tokens=True)
        student_answer = extract_boxed_answer(response_text)
        correct        = answers_match(student_answer, gt_answer)

        results.append(dict(
            problem=sample[cfg["prompt_key"]],
            gt_solution=sample[cfg["solution_key"]],
            gt_answer=gt_answer,
            student_response=response_text,
            student_answer=student_answer,
            student_correct=correct,
            full_ids=full_ids,
            prompt_len=prompt_len,
        ))

        if (si + 1) % 50 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  gt={gt_answer}  "
                  f"pred={student_answer}  len={len(response_ids)}", flush=True)

    n_ok = sum(r["student_correct"] for r in results)
    print(f"  Student accuracy: {n_ok}/{len(results)} ({n_ok/len(results):.1%})",
          flush=True)
    return results, tokenizer


# ── Phase 2: teacher generation + 4-way categorisation ───────────────────────

def phase2_teacher(samples, student_results):
    print(f"\n{'='*60}")
    print(f"Phase 2: Teacher generation  ({os.path.basename(TEACHER_MODEL)}) "
          f"+ 4-way categorisation")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    cfg = DATASET_CONFIGS[DATASET]
    prompt_strings = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]

    gen_outputs = _vllm_generate(TEACHER_MODEL, tokenizer, prompt_strings,
                                 do_sample=False)

    for si, (sr, out) in enumerate(zip(student_results, gen_outputs)):
        gen_ids        = list(out.outputs[0].token_ids)
        response_text  = tokenizer.decode(gen_ids, skip_special_tokens=True)
        teacher_answer = extract_boxed_answer(response_text)
        correct        = answers_match(teacher_answer, sr["gt_answer"])

        sr["teacher_response"] = response_text
        sr["teacher_answer"]   = teacher_answer
        sr["teacher_correct"]  = correct
        sr["category"]         = get_category(sr["student_correct"], correct)

        if (si + 1) % 50 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  gt={sr['gt_answer']}  "
                  f"pred={teacher_answer}  → {sr['category']}", flush=True)

    n_ok = sum(r["teacher_correct"] for r in student_results)
    print(f"  Teacher accuracy: {n_ok}/{len(student_results)} "
          f"({n_ok/len(student_results):.1%})", flush=True)

    cat_counts = defaultdict(int)
    for r in student_results:
        cat_counts[r["category"]] += 1
    for cat in CATEGORY_KEYS:
        print(f"  {CATEGORY_LABELS[cat]:32s}: {cat_counts[cat]:3d}", flush=True)

    return student_results


# ── Phase 3: per-token KL for ALL samples (interleaved, no logit storage) ────

def phase3_kl(student_results, student_tokenizer):
    print(f"\n{'='*60}")
    print(f"Phase 3: Per-token KL + episode segmentation  "
          f"(all {len(student_results)} samples)")
    print(f"{'='*60}", flush=True)

    if N_GPUS >= 2:
        s_gpus, t_gpus = [0], [1]
        print("  Student → GPU 0  |  Teacher → GPU 1  (simultaneous)", flush=True)
    else:
        s_gpus = t_gpus = [0]
        print("  Single GPU: both models on GPU 0", flush=True)

    print("  Loading student model …", flush=True)
    s_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=DTYPE,
        device_map=_make_device_map(STUDENT_MODEL, s_gpus),
        trust_remote_code=True,
    ).eval()

    print("  Loading teacher model …", flush=True)
    t_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=DTYPE,
        device_map=_make_device_map(TEACHER_MODEL, t_gpus),
        trust_remote_code=True,
    ).eval()

    s_dev = f"cuda:{s_gpus[0]}"
    t_dev = f"cuda:{t_gpus[0]}"

    case_data = []
    skipped = 0

    for si, sr in enumerate(student_results):
        full_ids   = sr["full_ids"]
        prompt_len = sr["prompt_len"]
        seq_len    = full_ids.shape[0]
        resp_len   = seq_len - prompt_len

        if resp_len <= 1:
            skipped += 1
            continue

        ids = full_ids.unsqueeze(0)

        s_logits = _forward_logits(s_model, ids, s_dev)
        t_logits = _forward_logits(t_model, ids, t_dev)

        s_resp = s_logits[prompt_len - 1: seq_len - 1].float()
        t_resp = t_logits[prompt_len - 1: seq_len - 1].float()
        del s_logits, t_logits

        s_log_p = F.log_softmax(s_resp, dim=-1)
        t_log_p = F.log_softmax(t_resp, dim=-1)
        s_p = s_log_p.exp()
        t_p = t_log_p.exp()

        kl_fwd = (t_p * (t_log_p - s_log_p)).sum(-1).tolist()
        kl_rev = (s_p * (s_log_p - t_log_p)).sum(-1).tolist()
        del s_resp, t_resp, s_log_p, t_log_p, s_p, t_p

        resp_ids = full_ids[prompt_len: seq_len].tolist()
        response_text = student_tokenizer.decode(resp_ids)
        think_end_pos = response_text.find("</think>")
        think_boundary_token = None
        if think_end_pos >= 0:
            prefix = response_text[:think_end_pos + len("</think>")]
            think_boundary_token = len(student_tokenizer.encode(prefix))

        tokens = []
        for t in range(resp_len):
            tok_str = student_tokenizer.decode([resp_ids[t]])
            region = "answer"
            if think_boundary_token is not None and t < think_boundary_token:
                region = "thinking"
            tokens.append(dict(
                text=tok_str,
                token_id=resp_ids[t],
                kl_forward=round(kl_fwd[t], 6),
                kl_reverse=round(kl_rev[t], 6),
                region=region,
            ))

        segments = segment_by_triggers(tokens)

        case_data.append(dict(
            case_index=si,
            category=sr["category"],
            problem=sr["problem"],
            gt_answer=sr["gt_answer"],
            student_answer=sr["student_answer"],
            student_correct=sr["student_correct"],
            teacher_answer=sr["teacher_answer"],
            teacher_correct=sr["teacher_correct"],
            tokens=tokens,
            segments=segments,
        ))

        if (si + 1) % 25 == 0 or si == 0:
            n_segs = len(segments)
            fwd_mu = sum(kl_fwd) / len(kl_fwd) if kl_fwd else 0.0
            rev_mu = sum(kl_rev) / len(kl_rev) if kl_rev else 0.0
            print(f"  [{si+1:3d}/{N_SAMPLES}] cat={sr['category']:14s}  "
                  f"resp={resp_len:4d}  segs={n_segs:2d}  "
                  f"KL(T‖S)={fwd_mu:.4f}  KL(S‖T)={rev_mu:.4f}", flush=True)

    del s_model, t_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Models unloaded.  Processed {len(case_data)}, skipped {skipped}.",
          flush=True)
    return case_data


# ── Phase 4: K-fold episode evaluation ───────────────────────────────────────

def _build_episode_prefix(problem, partial_resp_text, eval_tokenizer):
    resp = partial_resp_text
    if "<think>" in resp and "</think>" not in resp:
        resp = resp.rstrip() + "\n</think>"
    msgs = [
        {"role": "system", "content": EPISODE_EVAL_SYSTEM},
        {"role": "user",   "content": problem},
    ]
    base = eval_tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return base + resp + EPISODE_EVAL_SUFFIX


def phase4_episode_eval(case_data):
    print(f"\n{'='*60}")
    print(f"Phase 4: K={K_EVAL}-fold episode evaluation  "
          f"(model: {os.path.basename(EVAL_MODEL)}, T={EVAL_TEMPERATURE})")
    print(f"{'='*60}", flush=True)

    eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL, trust_remote_code=True)

    eval_items = []
    for ci, case in enumerate(case_data):
        for ep_idx, seg in enumerate(case.get("segments", [])):
            ep_num       = ep_idx + 1
            partial_resp = "".join(
                tok["text"] for tok in case["tokens"][: seg["end"]]
            )
            eval_items.append(dict(
                case_idx=ci,
                ep_idx=ep_idx,
                ep_num=ep_num,
                ep_length=seg["length"],
                normKL_fwd=seg["kl_forward_mean"],
                normKL_rev=seg["kl_reverse_mean"],
                partial_resp=partial_resp,
                problem=case["problem"],
                gt_answer=case["gt_answer"],
            ))

    total_pairs = len(eval_items)
    print(f"  (case, episode) pairs: {total_pairs}  "
          f"(× K={K_EVAL} = {total_pairs * K_EVAL} completions)", flush=True)

    if not eval_items:
        print("  No eval items found.", flush=True)
        return []

    prefixes = [
        _build_episode_prefix(item["problem"], item["partial_resp"], eval_tokenizer)
        for item in eval_items
    ]

    tp = min(N_GPUS, 4)
    llm = LLM(
        model=EVAL_MODEL,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 1024,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        n=K_EVAL,
        max_tokens=MAX_ANSWER_TOKENS,
        temperature=EVAL_TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        stop=["<|im_end|>"],
    )
    print("  Running vLLM inference …", flush=True)
    outputs = llm.generate(prefixes, params)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    for item, out in zip(eval_items, outputs):
        gen_texts = [o.text for o in out.outputs]
        preds     = [extract_boxed_answer(t) for t in gen_texts]
        corrects  = [answers_match(p, item["gt_answer"]) for p in preds]
        item["acc_rate"]      = round(sum(corrects) / K_EVAL, 6)
        item["correct_count"] = int(sum(corrects))

    print(f"  Parsed {len(eval_items)} items.", flush=True)
    return eval_items


# ── Monotonic run merging ─────────────────────────────────────────────────────

def merge_monotonic_runs(points):
    """
    Collapse each maximal non-decreasing or non-increasing sub-sequence of
    process rewards into a single representative point (the last element of
    the run).

    Direction encoding (three-valued):
        +1  strictly increasing  (pr[i+1] >  pr[i])
         0  flat / equal         (pr[i+1] == pr[i])
        -1  strictly decreasing  (pr[i+1] <  pr[i])

    A run is cut ONLY when a strict sign flip occurs, i.e. when the gap
    transitions from strictly positive (+1) to strictly negative (-1) or
    vice versa.  Flat steps (0) never trigger a cut — they extend whichever
    run is in progress.

    Example
    -------
    rewards: [0.10, 0.20, 0.20, 0.10, 0.10, 0.25]
    signs:   [ +1,   0,   -1,    0,   +1 ]
             ↑ not cut (0 extends +1)
                    ↑ cut here (+1 → -1 flip)
                          ↑ not cut (0 extends -1)
                                 ↑ cut here (-1 → +1 flip)
    kept indices: 2, 4, 5
    kept rewards: [0.20, 0.10, 0.25]
    """
    n = len(points)
    if n <= 1:
        return list(points)

    def _sign(a, b):
        diff = b - a
        if diff > 0:
            return 1
        if diff < 0:
            return -1
        return 0

    signs = [
        _sign(points[i]["process_reward"], points[i + 1]["process_reward"])
        for i in range(n - 1)
    ]

    # Track the "active" strict direction (+1 or -1); 0 inherits the last one.
    run_end_indices = []
    active_dir = 0  # no strict direction established yet

    for i in range(n - 1):
        s = signs[i]
        is_last_gap = (i == n - 2)

        if s != 0:
            if active_dir != 0 and s != active_dir:
                # Strict flip: the current gap starts a new run, so i is the
                # last index of the previous run.
                run_end_indices.append(i)
            active_dir = s

        if is_last_gap:
            run_end_indices.append(n - 1)

    # Edge case: all flat — only one run, keep the last point.
    if not run_end_indices:
        run_end_indices = [n - 1]

    return [points[idx] for idx in run_end_indices]


# ── Phase 5: process rewards + monotonic merge + statistical analysis ─────────

def _pearsonr(x, y):
    if _SCIPY_OK and len(x) > 2:
        r, p = scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    if len(x) < 2:
        return 0.0, float("nan")
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = float(np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum()))
    r = float(num / den) if den > 1e-12 else 0.0
    return r, float("nan")


def _r2(y_true, y_pred):
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def phase5_analyze(case_data, eval_items):
    print(f"\n{'='*60}")
    print("Phase 5: Process rewards + monotonic merge + statistical analysis")
    print(f"{'='*60}", flush=True)

    # ── Step 1: compute process rewards (all episodes) ────────────────────────
    ep_map = defaultdict(dict)
    for item in eval_items:
        ep_map[item["case_idx"]][item["ep_num"]] = item

    for ci, case in enumerate(case_data):
        prev_acc = 0.0
        for ep_num in sorted(ep_map[ci].keys()):
            item = ep_map[ci][ep_num]
            item["process_reward"] = round(item["acc_rate"] - prev_acc, 6)
            prev_acc = item["acc_rate"]

    # ── Step 2: collect per-case sequences (ep >= MIN_EVAL_EPISODE) ───────────
    case_sequences = defaultdict(list)
    for item in eval_items:
        if item["ep_num"] < MIN_EVAL_EPISODE:
            continue
        if "process_reward" not in item:
            continue
        case_sequences[item["case_idx"]].append(dict(
            sample_idx=item["case_idx"],
            ep_num=item["ep_num"],
            normKL_fwd=round(max(0.0, item["normKL_fwd"]), 6),
            normKL_rev=round(max(0.0, item["normKL_rev"]), 6),
            acc_rate=item["acc_rate"],
            process_reward=item["process_reward"],
            ep_length=item["ep_length"],
        ))

    # ── Step 3: merge monotonic runs per case ─────────────────────────────────
    cat_points = defaultdict(list)
    total_raw = total_merged = 0

    for ci, seq in case_sequences.items():
        seq_sorted = sorted(seq, key=lambda x: x["ep_num"])
        cat = case_data[ci]["category"]
        total_raw += len(seq_sorted)
        merged = merge_monotonic_runs(seq_sorted)
        total_merged += len(merged)
        cat_points[cat].extend(merged)

    print(f"  Monotonic merge: {total_raw} raw → {total_merged} points "
          f"(reduction {(1 - total_merged/total_raw)*100:.1f}%)", flush=True)

    # ── Step 4: fit curves per category ──────────────────────────────────────
    results = {}
    for cat in CATEGORY_KEYS:
        points = cat_points.get(cat, [])
        n      = len(points)
        n_raw  = sum(
            len(case_sequences.get(ci, []))
            for ci, c in enumerate(case_data)
            if c["category"] == cat
        )
        n_samples = sum(1 for c in case_data if c["category"] == cat)

        print(f"  {CATEGORY_LABELS[cat]:32s}: {n_samples:3d} samples, "
              f"{n_raw:4d} raw → {n:4d} merged (ep ≥ {MIN_EVAL_EPISODE})",
              flush=True)

        cat_result = dict(n=n, n_raw=n_raw, n_samples=n_samples, points=points)

        if n < 3:
            cat_result["stats"] = {}
            results[cat] = cat_result
            continue

        y = np.array([p["process_reward"] for p in points])
        stats_out = {}

        for kl_dir in ("fwd", "rev"):
            x = np.array([p[f"normKL_{kl_dir}"] for p in points])

            r, p_val = _pearsonr(x, y)
            lin  = np.polyfit(x, y, 1)
            quad = np.polyfit(x, y, 2)

            r2_lin  = _r2(y, np.polyval(lin, x))
            r2_quad = _r2(y, np.polyval(quad, x))

            label = "KL(T‖S)" if kl_dir == "fwd" else "KL(S‖T)"
            print(f"    [{label}]  r={r:+.3f}  p={p_val:.5f}  "
                  f"R²_lin={r2_lin:.3f}  R²_quad={r2_quad:.3f}", flush=True)

            stats_out[kl_dir] = dict(
                pearson_r=round(r, 4),
                p_value=(round(p_val, 6) if not np.isnan(p_val) else None),
                linear_fit=[round(float(c), 6) for c in lin],
                quad_fit=[round(float(c), 6) for c in quad],
                r2_linear=round(r2_lin, 4),
                r2_quad=round(r2_quad, 4),
                x_min=round(float(x.min()), 4),
                x_max=round(float(x.max()), 4),
            )

        cat_result["stats"] = stats_out
        cat_result["mean_process_reward"] = round(float(y.mean()), 4)
        cat_result["std_process_reward"]  = round(float(y.std()),  4)
        results[cat] = cat_result

    return results


# ── Phase 6: JSON + HTML report ───────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Process Reward × KL Divergence (Monotonic Merge)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0f0f1a;color:#d8d8e8;
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
  line-height:1.6;padding:32px 24px;
}
.container{max-width:1400px;margin:0 auto}
h1{font-size:1.6rem;font-weight:700;letter-spacing:-.02em;margin-bottom:6px;color:#f0f0ff}
h2{font-size:1.05rem;font-weight:600;color:#c0c0ff;margin:28px 0 10px}
.subtitle{font-size:.85rem;color:#888;margin-bottom:24px}
.meta-grid{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}
.meta-chip{
  background:#181830;border:1px solid #2a2a4a;border-radius:8px;
  padding:7px 14px;font-size:.82rem;
}
.meta-chip .label{color:#888;font-size:.72rem;display:block;margin-bottom:1px}
.meta-chip .val{color:#7ecfff;font-weight:600}
.controls{
  background:#181830;border:1px solid #2a2a4a;border-radius:10px;
  padding:14px 20px;margin-bottom:22px;
  display:flex;gap:10px;align-items:center;flex-wrap:wrap;
}
.kl-btn{
  padding:6px 18px;border-radius:7px;cursor:pointer;font-size:.84rem;
  border:1px solid #3a3a6a;background:#22224a;color:#ccc;
  transition:background .15s,color .15s;
}
.kl-btn.active{background:#5b8def;border-color:#5b8def;color:#fff;font-weight:600}
.cat-bar{display:flex;gap:10px;margin-bottom:26px;flex-wrap:wrap}
.cat-chip{
  border-radius:8px;padding:7px 15px;font-size:.82rem;font-weight:600;
}
.plots-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:32px;
}
.plot-cell{
  background:#14142a;border:1px solid #252548;border-radius:12px;
  overflow:hidden;min-height:420px;
}
.stats-wrap{overflow-x:auto;margin-bottom:8px}
.stats-table{
  width:100%;border-collapse:collapse;font-size:.82rem;
}
.stats-table th{
  background:#181830;color:#aaa;padding:8px 14px;text-align:left;
  border-bottom:1px solid #252548;font-weight:500;white-space:nowrap;
}
.stats-table td{
  padding:7px 14px;border-bottom:1px solid #1a1a30;white-space:nowrap;
}
.stats-table tr:hover td{background:#1c1c32}
.pos{color:#4caf50;font-weight:700}
.neg{color:#f44336;font-weight:700}
.note{font-size:.76rem;color:#666;margin-top:8px}
</style>
</head>
<body>
<div class="container">
  <h1>Process Reward × Normalised KL Divergence
    <span style="font-size:.9rem;color:#888;font-weight:400;margin-left:10px">
      (monotonic-run merged)
    </span>
  </h1>
  <div class="subtitle">
    Consecutive episodes with monotonically increasing or decreasing process
    rewards are merged into one data point (last episode of each run retained).
    Each point therefore represents an inflection in the correctness trajectory.
  </div>

  <div id="meta" class="meta-grid"></div>

  <div class="controls">
    <span style="font-size:.82rem;color:#888;margin-right:4px">KL direction:</span>
    <button class="kl-btn active" id="btn-fwd" onclick="setDir('fwd')">
      Forward &nbsp; KL(T‖S)
    </button>
    <button class="kl-btn" id="btn-rev" onclick="setDir('rev')">
      Reverse &nbsp; KL(S‖T)
    </button>
    <span style="font-size:.76rem;color:#555;margin-left:8px">
      NormKL = mean KL over segment tokens &nbsp;|&nbsp;
      ProcessReward(i) = AccRate(i) − AccRate(i−1)
    </span>
  </div>

  <div class="cat-bar" id="cat-bar"></div>

  <h2>Scatter: NormKL vs Process Reward — merged inflection points (ep ≥ __MIN_EP__)</h2>
  <div class="plots-grid">
    <div class="plot-cell" id="plt-both_correct"></div>
    <div class="plot-cell" id="plt-both_wrong"></div>
    <div class="plot-cell" id="plt-student_only"></div>
    <div class="plot-cell" id="plt-teacher_only"></div>
  </div>

  <h2>Correlation summary</h2>
  <div class="stats-wrap"><div id="stats-tbl"></div></div>
  <div class="note">
    * Each point = last episode of a monotonic run &nbsp;|&nbsp;
    Solid = linear fit &nbsp;|&nbsp; Dashed = quadratic fit &nbsp;|&nbsp;
    Point colour = episode index &nbsp;|&nbsp; n_raw shown in parentheses
  </div>
</div>

<script>
const DATA   = __JSON_DATA__;
const LABELS = {
  both_correct:  'Both Correct (S\u2713 T\u2713)',
  both_wrong:    'Both Wrong (S\u2717 T\u2717)',
  student_only:  'Student Only (S\u2713 T\u2717)',
  teacher_only:  'Teacher Only (S\u2717 T\u2713)',
};
const COLORS = {
  both_correct: '#4caf50',
  both_wrong:   '#f44336',
  student_only: '#ff9800',
  teacher_only: '#2196f3',
};
const CATS = ['both_correct','both_wrong','student_only','teacher_only'];

let klDir = 'fwd';

// ── meta chips ──────────────────────────────────────────────────────────────
const metaEl = document.getElementById('meta');
const cfg = DATA.config;
[
  ['Student',      cfg.student_model],
  ['Teacher',      cfg.teacher_model],
  ['Evaluator',    cfg.eval_model],
  ['N samples',    cfg.n_samples],
  ['K eval',       cfg.k_eval],
  ['Eval T',       cfg.eval_temperature],
  ['Min episode',  cfg.min_eval_episode],
  ['Merge',        'monotonic runs'],
].forEach(([l, v]) => {
  metaEl.innerHTML +=
    `<div class="meta-chip"><span class="label">${l}</span>`+
    `<span class="val">${v}</span></div>`;
});

// ── category chips ───────────────────────────────────────────────────────────
const catBar = document.getElementById('cat-bar');
CATS.forEach(cat => {
  const d = DATA.categories[cat];
  catBar.innerHTML +=
    `<div class="cat-chip" style="background:${COLORS[cat]}22;`+
    `border:1px solid ${COLORS[cat]}55;color:${COLORS[cat]}">` +
    `${LABELS[cat]}: ${d.n_samples} samples / `+
    `${d.n} merged pts (raw ${d.n_raw})</div>`;
});

// ── toggle ───────────────────────────────────────────────────────────────────
function setDir(dir) {
  klDir = dir;
  document.getElementById('btn-fwd').className =
    'kl-btn' + (dir === 'fwd' ? ' active' : '');
  document.getElementById('btn-rev').className =
    'kl-btn' + (dir === 'rev' ? ' active' : '');
  renderPlots();
  renderStats();
}

function linspace(a, b, n) {
  return Array.from({length: n}, (_, i) => a + (b - a) * i / (n - 1));
}
function polyval(coeffs, xs) {
  return xs.map(x => coeffs.reduce((acc, c) => acc * x + c, 0));
}

// ── scatter plots ─────────────────────────────────────────────────────────────
function renderPlots() {
  CATS.forEach(cat => {
    const divId   = `plt-${cat}`;
    const catData = DATA.categories[cat];
    const st      = catData.stats && catData.stats[klDir];
    const pts     = catData.points || [];

    const x   = pts.map(p => Math.max(0, p[`normKL_${klDir}`]));
    const y   = pts.map(p => p.process_reward);
    const eps = pts.map(p => p.ep_num);

    const traces = [];

    traces.push({
      type: 'scatter', mode: 'markers',
      x, y,
      marker: {
        color: eps,
        colorscale: 'Viridis',
        size: 5, opacity: 0.6,
        colorbar: {
          title: {text: 'Ep', font: {size: 10, color: '#999'}},
          thickness: 10, len: 0.55,
          tickfont: {color: '#888', size: 9},
        },
      },
      text: pts.map(p =>
        `Case ${p.sample_idx+1}, Ep ${p.ep_num}<br>`+
        `acc=${p.acc_rate.toFixed(2)}, len=${p.ep_length}`),
      hovertemplate:
        '%{text}<br>KL=%{x:.4f}<br>PR=%{y:.4f}<extra></extra>',
      name: 'data',
    });

    if (st && pts.length >= 3) {
      const xs = linspace(st.x_min, st.x_max, 150);

      traces.push({
        type: 'scatter', mode: 'lines', x: xs,
        y: polyval(st.linear_fit, xs),
        line: {color: 'rgba(255,255,255,0.85)', width: 2},
        name: `Lin R\u00B2=${st.r2_linear.toFixed(3)}`,
        hoverinfo: 'skip',
      });
      traces.push({
        type: 'scatter', mode: 'lines', x: xs,
        y: polyval(st.quad_fit, xs),
        line: {color: 'rgba(255,220,50,0.85)', width: 2, dash: 'dash'},
        name: `Quad R\u00B2=${st.r2_quad.toFixed(3)}`,
        hoverinfo: 'skip',
      });
    }

    const rTag = st
      ? `r=${st.pearson_r >= 0 ? '+' : ''}${st.pearson_r.toFixed(3)}, `+
        `p=${st.p_value !== null ? st.p_value.toFixed(4) : 'n/a'}, `+
        `n=${catData.n} (raw ${catData.n_raw})`
      : `n=${catData.n} (raw ${catData.n_raw})`;

    const klLabel = klDir === 'fwd'
      ? 'NormKL = KL(T\u2016S) mean'
      : 'NormKL = KL(S\u2016T) mean';

    Plotly.react(divId, traces, {
      title: {
        text: `${LABELS[cat]}<br><sup style="font-size:9px">${rTag}</sup>`,
        font: {color: COLORS[cat], size: 13},
        x: 0.04,
      },
      paper_bgcolor: '#14142a',
      plot_bgcolor:  '#0d0d1a',
      font:  {color: '#d8d8e8', size: 11},
      xaxis: {title: klLabel,          gridcolor: '#252548', zerolinecolor: '#3a3a6a'},
      yaxis: {title: 'Process Reward', gridcolor: '#252548', zerolinecolor: '#3a3a6a'},
      legend: {
        bgcolor: 'rgba(0,0,0,0)', font: {size: 10},
        x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top',
      },
      margin: {l: 60, r: 30, t: 85, b: 60},
    }, {responsive: true});
  });
}

// ── stats table ───────────────────────────────────────────────────────────────
function renderStats() {
  const dirLabel = klDir === 'fwd' ? 'KL(T\u2016S)' : 'KL(S\u2016T)';
  let html = `<table class="stats-table"><thead><tr>
    <th>Category</th><th>Samples</th>
    <th>Raw pts</th><th>Merged pts</th>
    <th>Pearson r<br><small>${dirLabel}</small></th>
    <th>p-value</th>
    <th>R\u00B2 linear</th><th>R\u00B2 quadratic</th>
    <th>Mean PR</th><th>Std PR</th>
  </tr></thead><tbody>`;

  CATS.forEach(cat => {
    const d  = DATA.categories[cat];
    const st = d.stats && d.stats[klDir];
    const r  = st ? st.pearson_r : null;
    const rCls = r !== null ? (r >= 0 ? 'pos' : 'neg') : '';
    html += `<tr>
      <td style="color:${COLORS[cat]};font-weight:600">${LABELS[cat]}</td>
      <td>${d.n_samples}</td>
      <td>${d.n_raw}</td>
      <td>${d.n}</td>
      <td class="${rCls}">${r !== null
        ? (r >= 0 ? '+' : '') + r.toFixed(4) : '\u2014'}</td>
      <td>${st && st.p_value !== null
        ? st.p_value.toFixed(5) : '\u2014'}</td>
      <td>${st ? st.r2_linear.toFixed(4) : '\u2014'}</td>
      <td>${st ? st.r2_quad.toFixed(4)   : '\u2014'}</td>
      <td>${d.mean_process_reward !== undefined
        ? d.mean_process_reward.toFixed(4) : '\u2014'}</td>
      <td>${d.std_process_reward !== undefined
        ? d.std_process_reward.toFixed(4) : '\u2014'}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('stats-tbl').innerHTML = html;
}

renderPlots();
renderStats();
</script>
</body>
</html>
"""


def phase6_report(analysis_results, student_results):
    print(f"\n{'='*60}")
    print("Phase 6: Generating JSON + HTML report")
    print(f"{'='*60}", flush=True)

    cat_counts = defaultdict(int)
    for sr in student_results:
        cat_counts[sr["category"]] += 1

    report = dict(
        config=dict(
            student_model=os.path.basename(STUDENT_MODEL),
            teacher_model=os.path.basename(TEACHER_MODEL),
            eval_model=os.path.basename(EVAL_MODEL),
            n_samples=N_SAMPLES,
            k_eval=K_EVAL,
            eval_temperature=EVAL_TEMPERATURE,
            min_eval_episode=MIN_EVAL_EPISODE,
            monotonic_merge=True,
        ),
        category_counts={cat: cat_counts[cat] for cat in CATEGORY_KEYS},
        categories=analysis_results,
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved → {OUTPUT_JSON}", flush=True)

    html = HTML_TEMPLATE.replace("__JSON_DATA__", json.dumps(report, ensure_ascii=False))
    html = html.replace("__MIN_EP__", str(MIN_EVAL_EPISODE))
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved → {OUTPUT_HTML}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"GPUs available: {N_GPUS}", flush=True)
    for i in range(N_GPUS):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        print(f"  [{i}] {name}  ({mem:.0f} GB)", flush=True)

    print(f"\nStudent model : {STUDENT_MODEL}")
    print(f"Teacher model : {TEACHER_MODEL}")
    print(f"Eval model    : {EVAL_MODEL}")
    print(f"N_SAMPLES={N_SAMPLES}  K_EVAL={K_EVAL}  "
          f"EVAL_TEMPERATURE={EVAL_TEMPERATURE}  "
          f"MIN_EVAL_EPISODE={MIN_EVAL_EPISODE}  "
          f"merge=monotonic_runs", flush=True)

    samples = load_dataset_samples()

    student_results, student_tok = phase1_student(samples)
    student_results = phase2_teacher(samples, student_results)
    case_data       = phase3_kl(student_results, student_tok)
    eval_items      = phase4_episode_eval(case_data)
    analysis        = phase5_analyze(case_data, eval_items)
    phase6_report(analysis, student_results)

    # ── quick text summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary: Pearson r (process reward vs NormKL, after monotonic merge)")
    print(f"{'='*60}")
    for cat in CATEGORY_KEYS:
        res = analysis.get(cat, {})
        st_fwd = (res.get("stats") or {}).get("fwd", {})
        st_rev = (res.get("stats") or {}).get("rev", {})
        r_fwd  = st_fwd.get("pearson_r", float("nan"))
        r_rev  = st_rev.get("pearson_r", float("nan"))
        n_raw  = res.get("n_raw", 0)
        n_mrg  = res.get("n", 0)
        print(f"  {CATEGORY_LABELS[cat]:32s}  "
              f"KL(T‖S) r={r_fwd:+.4f}   KL(S‖T) r={r_rev:+.4f}"
              f"   [{n_raw}→{n_mrg}]")

    print(f"\nDone. Open {OUTPUT_HTML} in a browser.", flush=True)


if __name__ == "__main__":
    main()
