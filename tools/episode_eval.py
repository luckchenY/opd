"""
episode_eval.py  —  Episode-level student answer evaluation
============================================================

For cases where the student model answers incorrectly but the teacher answers
correctly, this script:

  1. Segments each student response into reasoning episodes (trigger words +
     accumulated periods, same logic as analyze_kl_heatmap.py).
  2. Starting from MIN_EVAL_EPISODE (default 3), truncates the student response
     at each episode boundary, closes any open <think> block, and asks the
     eval model to produce a final boxed answer from that partial reasoning.
  3. Compares each predicted answer against ground truth and reports per-episode
     correctness as JSON + an interactive HTML report.

The model used for episode-level evaluation (EVAL_MODEL) defaults to the
student model but can be overridden via environment variable, making it easy
to test different models without changing the script.

Flow:
  Phase 1 — Student on-policy sampling
  Phase 2 — Teacher greedy generation → select student-wrong / teacher-right cases
  Phase 3 — Per-token KL divergence + episode segmentation
  Phase 4 — Episode-level batch evaluation (vLLM, partial-response prefix)
  Phase 5 — JSON + HTML report

Usage:
    python tools/episode_eval.py
    EVAL_MODEL=/path/to/other_model python tools/episode_eval.py
"""

import bisect
import gc
import json
import os
import random
import re
from collections import defaultdict
from html import escape as html_escape

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# ── config ────────────────────────────────────────────────────────────────────
STUDENT_MODEL = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen"
    "/models/Qwen3-1.7B"
)
TEACHER_MODEL = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen"
    "/models/Qwen3-8B"
)
# Override with:  EVAL_MODEL=/path/to/model python tools/episode_eval.py
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

N_SAMPLES      = 50
MAX_NEW_TOKENS = 4096
N_CASES        = 5
SEED           = 42
TEMPERATURE    = 1.0
TOP_K          = 50
TOP_P          = 1.0

# ── episode evaluation ────────────────────────────────────────────────────────
# Start evaluating from this episode number (1-indexed, inclusive).
MIN_EVAL_EPISODE  = 3
# Max tokens for the answer-only generation (just needs to produce \boxed{...}).
MAX_ANSWER_TOKENS = 384
# Greedy decoding for reproducible answers.
EVAL_TEMPERATURE  = 0

# System message prepended when building episode-eval prompts.
# Deliberately omits "reason step by step" to avoid conflicting with the
# suffix instruction that tells the model to stop thinking immediately.
# Printed at startup so you can verify what the model actually sees.
EPISODE_EVAL_SYSTEM = (
    "Put your final answer within \\boxed{}."
)
# Appended after </think> to elicit a final boxed answer.
# Explicitly tells the model not to continue reasoning.
EPISODE_EVAL_SUFFIX = (
    "\nBased on the reasoning above, do not think further."
    " Directly give the final answer in \\boxed{answer}."
)

# ── episode segmentation ──────────────────────────────────────────────────────
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

DTYPE   = torch.bfloat16
N_GPUS  = torch.cuda.device_count()
_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(_DIR, "episode_eval.json")
OUTPUT_HTML = os.path.join(_DIR, "episode_eval.html")

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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# vLLM generation helper
# ─────────────────────────────────────────────────────────────────────────────

def _vllm_generate(model_path, tokenizer, prompt_strings, do_sample=True,
                   max_tokens=None, temperature=None, stop=None):
    """Batch generation via vLLM. Returns list of output text strings."""
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

    sp_kwargs = dict(max_tokens=max_tokens, temperature=temperature)
    if do_sample and temperature > 0:
        sp_kwargs.update(top_k=TOP_K, top_p=TOP_P)
    if stop:
        sp_kwargs["stop"] = stop

    params = SamplingParams(**sp_kwargs)
    print(f"  Generating {len(prompt_strings)} responses …", flush=True)
    outputs = llm.generate(prompt_strings, params)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Episode segmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Split token list into segments at trigger phrases, requiring min_periods
    sentence-ending periods since the last cut.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: student on-policy sampling
# ─────────────────────────────────────────────────────────────────────────────

def phase1_student(samples):
    print(f"\n{'='*60}")
    print(f"Phase 1: Student sampling ({os.path.basename(STUDENT_MODEL)}) "
          f"[vLLM, {N_GPUS} GPU(s)]")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    cfg = DATASET_CONFIGS[DATASET]
    prompt_strings = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]
    gt_answers = [extract_boxed_answer(s[cfg["solution_key"]]) for s in samples]

    gen_outputs = _vllm_generate(STUDENT_MODEL, tokenizer, prompt_strings, do_sample=True)

    results = []
    for si, (sample, out, gt_answer) in enumerate(zip(samples, gen_outputs, gt_answers)):
        prompt_ids = list(out.prompt_token_ids)
        gen_ids    = list(out.outputs[0].token_ids)
        full_ids   = torch.tensor(prompt_ids + gen_ids, dtype=torch.long)
        prompt_len = len(prompt_ids)

        response_ids  = full_ids[prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        student_answer = extract_boxed_answer(response_text)
        correct = answers_match(student_answer, gt_answer)

        results.append(dict(
            problem=sample[cfg["prompt_key"]],
            gt_solution=sample[cfg["solution_key"]],
            gt_answer=gt_answer,
            student_response=response_text,
            student_answer=student_answer,
            student_correct=correct,
            prompt_ids=full_ids[:prompt_len],
            response_ids=response_ids,
            full_ids=full_ids,
            prompt_len=prompt_len,
        ))

        if (si + 1) % 10 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  gt={gt_answer}  "
                  f"pred={student_answer}  len={len(response_ids)}", flush=True)

    n_ok = sum(r["student_correct"] for r in results)
    print(f"  Student accuracy: {n_ok}/{len(results)} ({n_ok/len(results):.1%})",
          flush=True)
    return results, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: teacher greedy generation
# ─────────────────────────────────────────────────────────────────────────────

def phase2_teacher(samples, student_results):
    print(f"\n{'='*60}")
    print(f"Phase 2: Teacher generation ({os.path.basename(TEACHER_MODEL)}) "
          f"[vLLM, {N_GPUS} GPU(s)]")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    cfg = DATASET_CONFIGS[DATASET]
    prompt_strings = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]

    gen_outputs = _vllm_generate(TEACHER_MODEL, tokenizer, prompt_strings,
                                 do_sample=False)

    for si, (sr, out) in enumerate(zip(student_results, gen_outputs)):
        prompt_len = len(out.prompt_token_ids)
        gen_ids    = list(out.outputs[0].token_ids)
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        teacher_answer = extract_boxed_answer(response_text)
        correct = answers_match(teacher_answer, sr["gt_answer"])

        sr["teacher_response"] = response_text
        sr["teacher_answer"]   = teacher_answer
        sr["teacher_correct"]  = correct

        if (si + 1) % 10 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  gt={sr['gt_answer']}  "
                  f"pred={teacher_answer}", flush=True)

    n_ok = sum(r["teacher_correct"] for r in student_results)
    print(f"  Teacher accuracy: {n_ok}/{len(student_results)} "
          f"({n_ok/len(student_results):.1%})", flush=True)

    candidates = [i for i, r in enumerate(student_results)
                  if not r["student_correct"] and r["teacher_correct"]]
    print(f"  Student-wrong & Teacher-right: {len(candidates)} cases", flush=True)

    selected = candidates[:N_CASES]
    if len(selected) < N_CASES:
        print(f"  WARNING: only {len(selected)} qualifying cases found "
              f"(wanted {N_CASES})")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: per-token KL + episode segmentation
# ─────────────────────────────────────────────────────────────────────────────

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


def phase3_kl(student_results, selected_indices, student_tokenizer):
    if not selected_indices:
        print("\n  No qualifying cases — nothing to compute.", flush=True)
        return []

    print(f"\n{'='*60}")
    print(f"Phase 3: Per-token KL + episode segmentation "
          f"({len(selected_indices)} cases, {N_GPUS} GPUs)")
    print(f"{'='*60}", flush=True)

    case_inputs = [
        dict(full_ids=student_results[si]["full_ids"],
             prompt_len=student_results[si]["prompt_len"])
        for si in selected_indices
    ]

    all_gpus = list(range(N_GPUS))
    if N_GPUS >= 2:
        s_gpus, t_gpus = all_gpus[:1], all_gpus[1:]
        load_simultaneously = True
        print(f"  Student → GPU {s_gpus}  |  Teacher → GPU {t_gpus}", flush=True)
    else:
        s_gpus = t_gpus = [0]
        load_simultaneously = False
        print("  Single GPU mode: sequential load/unload", flush=True)

    print("  Loading student model …", flush=True)
    s_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=DTYPE,
        device_map=_make_device_map(STUDENT_MODEL, s_gpus),
        trust_remote_code=True,
    ).eval()

    if load_simultaneously:
        print("  Loading teacher model …", flush=True)
        t_model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, torch_dtype=DTYPE,
            device_map=_make_device_map(TEACHER_MODEL, t_gpus),
            trust_remote_code=True,
        ).eval()

    s_dev = f"cuda:{s_gpus[0]}"
    student_logits_list = []
    for ci, ci_data in enumerate(case_inputs):
        ids = ci_data["full_ids"].unsqueeze(0)
        student_logits_list.append(_forward_logits(s_model, ids, s_dev))
        print(f"    Student forward case {ci}: seq_len={ids.shape[1]}", flush=True)

    if not load_simultaneously:
        del s_model
        gc.collect()
        torch.cuda.empty_cache()
        print("  Loading teacher model …", flush=True)
        t_model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, torch_dtype=DTYPE,
            device_map=_make_device_map(TEACHER_MODEL, t_gpus),
            trust_remote_code=True,
        ).eval()

    t_dev = f"cuda:{t_gpus[0]}"
    teacher_logits_list = []
    for ci, ci_data in enumerate(case_inputs):
        ids = ci_data["full_ids"].unsqueeze(0)
        teacher_logits_list.append(_forward_logits(t_model, ids, t_dev))
        print(f"    Teacher forward case {ci}: seq_len={ids.shape[1]}", flush=True)

    if load_simultaneously:
        del s_model
    del t_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Models unloaded.", flush=True)

    case_data = []
    for ci, si in enumerate(selected_indices):
        sr = student_results[si]
        prompt_len = sr["prompt_len"]
        full_ids   = sr["full_ids"]
        seq_len    = full_ids.shape[0]
        resp_len   = seq_len - prompt_len

        if resp_len <= 1:
            continue

        s_resp = student_logits_list[ci][prompt_len - 1: seq_len - 1]
        t_resp = teacher_logits_list[ci][prompt_len - 1: seq_len - 1]

        s_log_p = F.log_softmax(s_resp.float(), dim=-1)
        t_log_p = F.log_softmax(t_resp.float(), dim=-1)
        s_p = s_log_p.exp()
        t_p = t_log_p.exp()

        kl_fwd = (t_p * (t_log_p - s_log_p)).sum(-1).tolist()
        kl_rev = (s_p * (s_log_p - t_log_p)).sum(-1).tolist()
        t_ent  = (-(t_p * t_log_p).sum(-1)).tolist()

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
                teacher_entropy=round(t_ent[t], 6),
                region=region,
            ))

        segments = segment_by_triggers(tokens)
        for seg_idx, seg in enumerate(segments):
            for ti in range(seg["start"], seg["end"]):
                tokens[ti]["segment"] = seg_idx

        fwd_arr = np.array(kl_fwd)
        rev_arr = np.array(kl_rev)
        n_answer = sum(1 for tk in tokens if tk["region"] == "answer")
        print(f"  Case {ci}: resp_len={resp_len}  "
              f"(thinking={resp_len - n_answer}, answer={n_answer})  "
              f"segments={len(segments)}  "
              f"KL(T‖S) mean={fwd_arr.mean():.4f}  "
              f"KL(S‖T) mean={rev_arr.mean():.4f}", flush=True)

        case_data.append(dict(
            case_index=ci,
            sample_index=si,
            problem=sr["problem"],
            gt_answer=sr["gt_answer"],
            student_answer=sr["student_answer"],
            teacher_answer=sr["teacher_answer"],
            student_response=sr["student_response"],
            teacher_response=sr["teacher_response"],
            tokens=tokens,
            think_boundary_token=think_boundary_token,
            segments=segments,
        ))

    return case_data


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: episode-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _build_episode_prefix(problem, partial_resp_text, eval_tokenizer):
    """
    Construct the full vLLM prompt prefix for one (case, episode) pair:

        [system] Please reason step by step ...
        [user]   {problem}
        [assistant-prefix]  <think>
                            {partial reasoning up to episode boundary}
                            </think>
                            Based on the reasoning above, please provide the final answer.

    vLLM generates whatever comes next (expected: the boxed answer).
    """
    # Close any open <think> block
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
    print(f"Phase 4: Episode-level evaluation "
          f"(eval model: {os.path.basename(EVAL_MODEL)}, "
          f"episodes >= {MIN_EVAL_EPISODE})")
    print(f"{'='*60}", flush=True)

    eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL, trust_remote_code=True)

    # ── Print a sample prefix so the user can verify the format ──────────────
    print("\n── Sample eval prompt prefix (first 600 chars) ─────────────────")
    sample_prefix = _build_episode_prefix(
        "What is 2 + 2?",
        "<think>\nLet me think. 2 + 2 = 4.\n",
        eval_tokenizer,
    )
    print(repr(sample_prefix[:600]))
    print("────────────────────────────────────────────────────────────────\n")

    # ── Collect all (case, episode) pairs to evaluate ─────────────────────────
    eval_items = []
    for ci, case in enumerate(case_data):
        segs = case.get("segments", [])
        for ep_idx, seg in enumerate(segs):
            ep_num = ep_idx + 1
            if ep_num < MIN_EVAL_EPISODE:
                continue
            partial_resp = "".join(
                tok["text"] for tok in case["tokens"][: seg["end"]]
            )
            eval_items.append(dict(
                case_idx=ci,
                ep_idx=ep_idx,
                ep_num=ep_num,
                ep_length=seg["length"],
                partial_resp=partial_resp,
                problem=case["problem"],
                gt_answer=case["gt_answer"],
            ))

    if not eval_items:
        print("  No eval items — all cases have fewer than "
              f"{MIN_EVAL_EPISODE} segments.", flush=True)
        return []

    print(f"  Total (case, episode) pairs to evaluate: {len(eval_items)}",
          flush=True)

    # ── Build prefixes ─────────────────────────────────────────────────────────
    prefixes = [
        _build_episode_prefix(item["problem"], item["partial_resp"], eval_tokenizer)
        for item in eval_items
    ]

    # ── vLLM batch inference ───────────────────────────────────────────────────
    tp = min(N_GPUS, 4)
    llm = LLM(
        model=EVAL_MODEL,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 1024,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(
        max_tokens=MAX_ANSWER_TOKENS,
        temperature=EVAL_TEMPERATURE,
        stop=["<|im_end|>"],
    )
    print(f"  Running vLLM inference …", flush=True)
    outputs = llm.generate(prefixes, params)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ── Parse results ──────────────────────────────────────────────────────────
    for item, out in zip(eval_items, outputs):
        gen_text = out.outputs[0].text
        pred = extract_boxed_answer(gen_text)
        correct = answers_match(pred, item["gt_answer"])
        item["generated_text"]   = gen_text
        item["predicted_answer"] = pred
        item["correct"] = correct

        tag = "✓" if correct else "✗"
        print(f"    case={item['case_idx']+1}  ep={item['ep_num']}  "
              f"{tag}  pred={pred}  gt={item['gt_answer']}", flush=True)

    return eval_items


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: JSON + HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(case_data, eval_items):
    """Merge case_data and eval_items into a report-ready structure."""
    # Index eval results by (case_idx, ep_num)
    eval_by_case = defaultdict(list)
    for item in eval_items:
        eval_by_case[item["case_idx"]].append(item)

    cases_out = []
    for ci, case in enumerate(case_data):
        ep_results = sorted(eval_by_case.get(ci, []), key=lambda x: x["ep_num"])
        cases_out.append(dict(
            case_idx=ci,
            problem=case["problem"],
            gt_answer=case["gt_answer"],
            student_answer=case["student_answer"],
            teacher_answer=case["teacher_answer"],
            n_segments=len(case.get("segments", [])),
            segments=[
                dict(ep_num=k + 1,
                     length=s["length"],
                     text_preview=s["text_preview"])
                for k, s in enumerate(case.get("segments", []))
            ],
            episode_evals=[
                dict(
                    ep_num=e["ep_num"],
                    ep_length=e["ep_length"],
                    predicted_answer=e["predicted_answer"],
                    correct=e["correct"],
                    generated_text=e["generated_text"],
                )
                for e in ep_results
            ],
        ))

    # Determine global episode range for the summary matrix
    all_ep_nums = sorted({e["ep_num"] for e in eval_items}) if eval_items else []

    return dict(
        config=dict(
            student_model=os.path.basename(STUDENT_MODEL),
            teacher_model=os.path.basename(TEACHER_MODEL),
            eval_model=os.path.basename(EVAL_MODEL),
            min_eval_episode=MIN_EVAL_EPISODE,
            episode_eval_system=EPISODE_EVAL_SYSTEM,
            episode_eval_suffix=EPISODE_EVAL_SUFFIX,
        ),
        all_episode_nums=all_ep_nums,
        cases=cases_out,
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Episode-level Answer Evaluation</title>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body,{delimiters:[
          {left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false},
          {left:'\\\\(',right:'\\\\)',display:false},{left:'\\\\[',right:'\\\\]',display:true}
        ]});"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0f0f1a;color:#d8d8e8;
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
  line-height:1.6;padding:32px 24px;
}
.container{max-width:1100px;margin:0 auto}
h1{font-size:1.6rem;font-weight:700;letter-spacing:-.02em;margin-bottom:6px;color:#f0f0ff}
h2{font-size:1.1rem;font-weight:600;color:#c0c0ff;margin:28px 0 12px}
.subtitle{font-size:.85rem;color:#888;margin-bottom:28px}
.meta-grid{
  display:flex;flex-wrap:wrap;gap:12px;margin-bottom:28px;
}
.meta-chip{
  background:#181830;border:1px solid #2a2a4a;border-radius:8px;
  padding:8px 16px;font-size:.82rem;
}
.meta-chip .label{color:#888;font-size:.72rem;display:block;margin-bottom:2px}
.meta-chip .val{color:#7ecfff;font-weight:600}
/* ── summary matrix ── */
.matrix-wrap{overflow-x:auto;margin-bottom:32px}
table.matrix{
  border-collapse:collapse;font-size:.82rem;
  min-width:400px;
}
table.matrix th,table.matrix td{
  padding:6px 14px;border:1px solid #252548;text-align:center;
  white-space:nowrap;
}
table.matrix th{background:#181830;color:#aaa;font-weight:500}
table.matrix td.case-label{
  text-align:left;color:#ccc;background:#12122a;font-weight:500;
}
table.matrix td.cell-correct{background:#1a3a1a;color:#6fcf6f}
table.matrix td.cell-wrong{background:#3a1212;color:#cf6f6f}
table.matrix td.cell-orig-wrong{background:#3a1212;color:#cf6f6f;opacity:.7}
table.matrix td.cell-na{background:#111122;color:#444}
/* ── case card ── */
.case-card{
  background:#14142a;border:1px solid #252548;border-radius:12px;
  padding:24px 28px;margin-bottom:28px;
}
.case-header{
  display:flex;justify-content:space-between;align-items:baseline;
  margin-bottom:14px;border-bottom:1px solid #252548;padding-bottom:12px;
}
.case-title{font-size:1.05rem;font-weight:600;color:#c0c0ff}
.answers{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.answer-chip{
  font-size:.82rem;padding:4px 12px;border-radius:6px;
  font-family:'SF Mono','Fira Code',monospace;
}
.answer-gt{background:#1a3a1a;color:#6fcf6f;border:1px solid #2a5a2a}
.answer-student{background:#3a1a1a;color:#cf6f6f;border:1px solid #5a2a2a}
.answer-teacher{background:#1a2a3a;color:#6fafcf;border:1px solid #2a4a5a}
.problem-text{
  background:#1a1a30;border-radius:8px;padding:12px 16px;
  margin-bottom:16px;font-size:.88rem;color:#b8b8d8;
  border-left:3px solid #4a4a8a;max-height:180px;overflow-y:auto;
}
/* ── episode table ── */
table.ep-table{
  width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:8px;
}
table.ep-table th{
  background:#181830;color:#999;padding:7px 14px;text-align:left;
  border-bottom:1px solid #252548;font-weight:500;
}
table.ep-table td{
  padding:8px 14px;border-bottom:1px solid #1a1a30;vertical-align:top;
}
table.ep-table tr:hover td{background:#1c1c32}
.cell-correct-icon{color:#6fcf6f;font-weight:700}
.cell-wrong-icon{color:#cf6f6f;font-weight:700}
.gen-preview{
  font-size:.78rem;color:#8888aa;font-family:'SF Mono','Fira Code',monospace;
  max-width:420px;white-space:pre-wrap;word-break:break-word;
}
.ep-skip{color:#555;font-style:italic;font-size:.75rem}
/* ── toggle ── */
.resp-toggle{
  font-size:.78rem;color:#6688bb;cursor:pointer;
  text-decoration:underline;margin-top:10px;display:inline-block;
}
.resp-preview{
  display:none;background:#111128;border-radius:8px;padding:12px;
  margin-top:6px;font-size:.78rem;color:#aaa;
  max-height:260px;overflow-y:auto;white-space:pre-wrap;
  border:1px solid #1e1e3a;
}
.resp-preview.show{display:block}
</style>
</head>
<body>
<div class="container">
  <h1>Episode-level Answer Evaluation</h1>
  <div class="subtitle">
    Student wrong · Teacher right · Episode truncation starting from ep __MIN_EP__
  </div>

  <div id="meta" class="meta-grid"></div>
  <h2>Summary — correctness per (case, episode)</h2>
  <div class="matrix-wrap"><div id="matrix"></div></div>
  <h2>Per-case details</h2>
  <div id="cases"></div>
</div>

<script>
const DATA = __JSON_DATA__;

function hs(s){
  if(!s) return '';
  const d=document.createElement('div');d.textContent=s;return d.innerHTML;
}

// ── meta chips ──────────────────────────────────────────────────────────────
const metaEl = document.getElementById('meta');
[
  ['Student',  DATA.config.student_model],
  ['Teacher',  DATA.config.teacher_model],
  ['Evaluator',DATA.config.eval_model],
  ['Min episode', DATA.config.min_eval_episode],
  ['Cases', DATA.cases.length],
].forEach(([label, val]) => {
  metaEl.innerHTML += `<div class="meta-chip">
    <span class="label">${label}</span>
    <span class="val">${hs(String(val))}</span>
  </div>`;
});

// ── summary matrix ───────────────────────────────────────────────────────────
const allEps  = DATA.all_episode_nums;
const matrixEl = document.getElementById('matrix');
if (allEps.length === 0) {
  matrixEl.textContent = 'No episodes evaluated.';
} else {
  let html = '<table class="matrix"><thead><tr><th>Case</th><th>Full resp (student)</th>';
  allEps.forEach(ep => { html += `<th>Ep ${ep}</th>`; });
  html += '</tr></thead><tbody>';

  DATA.cases.forEach((c, ci) => {
    html += `<tr><td class="case-label">Case ${ci+1}<br>
      <span style="font-size:.7rem;color:#555">gt: ${hs(c.gt_answer||'?')}</span></td>`;
    // original student answer (always wrong in these cases)
    html += `<td class="cell-orig-wrong">✗ ${hs(c.student_answer||'—')}</td>`;
    // episode results
    const evalByEp = {};
    (c.episode_evals||[]).forEach(e => { evalByEp[e.ep_num] = e; });
    allEps.forEach(ep => {
      const e = evalByEp[ep];
      if (!e) {
        html += `<td class="cell-na">—</td>`;
      } else if (e.correct) {
        html += `<td class="cell-correct">✓ ${hs(e.predicted_answer||'')}</td>`;
      } else {
        html += `<td class="cell-wrong">✗ ${hs(e.predicted_answer||'?')}</td>`;
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  matrixEl.innerHTML = html;
}

// ── per-case details ─────────────────────────────────────────────────────────
const casesEl = document.getElementById('cases');
DATA.cases.forEach((c, ci) => {
  const evalByEp = {};
  (c.episode_evals||[]).forEach(e => { evalByEp[e.ep_num] = e; });

  const div = document.createElement('div');
  div.className = 'case-card';

  // header + answers
  div.innerHTML = `
    <div class="case-header">
      <span class="case-title">Case ${ci+1}
        <span style="font-size:.72rem;color:#666;font-weight:400;margin-left:8px">
          ${c.n_segments} episode(s) total
        </span>
      </span>
    </div>
    <div class="problem-text">${hs(c.problem)}</div>
    <div class="answers">
      <span class="answer-chip answer-gt">GT: ${hs(c.gt_answer||'?')}</span>
      <span class="answer-chip answer-student">Student (full): ${hs(c.student_answer||'N/A')}</span>
      <span class="answer-chip answer-teacher">Teacher: ${hs(c.teacher_answer||'N/A')}</span>
    </div>
  `;

  // episode results table
  const segs = c.segments || [];
  let tbl = `<table class="ep-table">
    <thead><tr>
      <th>Ep</th><th>Tokens</th><th>Region text preview</th>
      <th>Predicted</th><th>GT</th><th>✓?</th><th>Model output preview</th>
    </tr></thead><tbody>`;

  segs.forEach((seg, si) => {
    const ep_num = si + 1;
    if (ep_num < DATA.config.min_eval_episode) {
      tbl += `<tr>
        <td>${ep_num}</td>
        <td>${seg.length}</td>
        <td class="gen-preview">${hs(seg.text_preview.substring(0,80))}…</td>
        <td colspan="4" class="ep-skip">not evaluated (ep &lt; ${DATA.config.min_eval_episode})</td>
      </tr>`;
    } else {
      const e = evalByEp[ep_num];
      if (!e) {
        tbl += `<tr>
          <td>${ep_num}</td><td>${seg.length}</td>
          <td class="gen-preview">${hs(seg.text_preview.substring(0,80))}…</td>
          <td colspan="4" class="ep-skip">—</td>
        </tr>`;
      } else {
        const icon = e.correct
          ? '<span class="cell-correct-icon">✓</span>'
          : '<span class="cell-wrong-icon">✗</span>';
        const predStyle = e.correct
          ? 'color:#6fcf6f;font-weight:600'
          : 'color:#cf6f6f;font-weight:600';
        const genPreview = (e.generated_text||'').substring(0, 160);
        tbl += `<tr>
          <td>${ep_num}</td>
          <td>${e.ep_length}</td>
          <td class="gen-preview">${hs(seg.text_preview.substring(0,80))}…</td>
          <td style="${predStyle};font-family:'SF Mono','Fira Code',monospace">${hs(e.predicted_answer||'—')}</td>
          <td style="font-family:'SF Mono','Fira Code',monospace">${hs(c.gt_answer||'?')}</td>
          <td>${icon}</td>
          <td class="gen-preview">${hs(genPreview)}${genPreview.length>=160?'…':''}</td>
        </tr>`;
      }
    }
  });

  tbl += '</tbody></table>';
  div.innerHTML += tbl;

  // teacher response toggle
  div.innerHTML += `
    <span class="resp-toggle" onclick="this.nextElementSibling.classList.toggle('show')">
      ▸ Show teacher response
    </span>
    <div class="resp-preview">${hs(c.teacher_response||'')}</div>
  `;

  casesEl.appendChild(div);
});

// Attach teacher response to cases (teacher_response lives in case_data not output)
// (already injected above via DATA)
</script>
</body>
</html>
"""


def phase5_report(case_data, eval_items):
    print(f"\n{'='*60}")
    print("Phase 5: Generating report")
    print(f"{'='*60}", flush=True)

    # Add teacher_response to cases_out for the HTML toggle
    teacher_resp_map = {c["case_index"]: c.get("teacher_response", "") for c in case_data}
    report = _build_report(case_data, eval_items)
    for c in report["cases"]:
        c["teacher_response"] = teacher_resp_map.get(c["case_idx"], "")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved → {OUTPUT_JSON}", flush=True)

    html = HTML_TEMPLATE.replace("__JSON_DATA__", json.dumps(report, ensure_ascii=False))
    html = html.replace("__MIN_EP__", str(MIN_EVAL_EPISODE))
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved → {OUTPUT_HTML}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"GPUs available: {N_GPUS}", flush=True)
    for i in range(N_GPUS):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [{i}] {name}  ({mem:.0f} GB)", flush=True)

    print(f"\nStudent model  : {STUDENT_MODEL}")
    print(f"Teacher model  : {TEACHER_MODEL}")
    print(f"Eval model     : {EVAL_MODEL}")
    print(f"Min eval episode: {MIN_EVAL_EPISODE}", flush=True)

    samples = load_dataset_samples()

    student_results, student_tok = phase1_student(samples)
    selected = phase2_teacher(samples, student_results)
    case_data = phase3_kl(student_results, selected, student_tok)
    eval_items = phase4_episode_eval(case_data)
    phase5_report(case_data, eval_items)

    # ── Quick summary ──────────────────────────────────────────────────────────
    if eval_items:
        print(f"\n{'='*60}")
        print("Episode evaluation summary")
        print(f"{'='*60}")
        by_ep = defaultdict(list)
        for item in eval_items:
            by_ep[item["ep_num"]].append(item["correct"])
        for ep in sorted(by_ep):
            results = by_ep[ep]
            n_ok = sum(results)
            print(f"  Episode {ep:2d} : {n_ok}/{len(results)} correct "
                  f"({n_ok/len(results):.0%})")

    print(f"\nDone. Open {OUTPUT_HTML} in a browser.", flush=True)


if __name__ == "__main__":
    main()
