"""
KL Divergence Heatmap: Student vs Teacher on wrong-answer cases.

Finds cases where the student model answers incorrectly but the teacher model
answers correctly, then computes per-token KL divergence (forward & reverse)
and renders an interactive HTML heatmap with adjustable blending ratio.

Flow:
  Phase 1 — Student sampling (on-policy) + correctness check
  Phase 2 — Teacher greedy generation + correctness check → pick cases
  Phase 3 — Load both models, forward-pass student responses, compute KL
  Phase 4 — Render interactive HTML

Usage:
    python tools/analyze_kl_heatmap.py           # uses all visible GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/analyze_kl_heatmap.py
"""

import bisect
import gc
import json
import os
import random
import re
from html import escape as html_escape

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# ── config ──────────────────────────────────────────────────────────────────
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-1.7B")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-8B")

DATASET   = "hendrycks_math"
_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASET_CONFIGS = {
    "hendrycks_math": {
        "subsets": [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ],
        "prompt_key":   "problem",
        "solution_key": "solution",
    },
}

N_SAMPLES      = 500
MAX_NEW_TOKENS = 4096
N_CASES        = 5
SEED           = 42

# ── episode segmentation ─────────────────────────────────────────────────────
# A segment cut is made when a trigger phrase is encountered AND at least
# SEGMENT_MIN_PERIODS sentence-ending periods have accumulated since the last cut.
SEGMENT_TRIGGER_PATTERNS = [
    r'\bwait\b',
    r'\balternatively\b',
    r'\blet\s+me\s+check\b',
    r'\blet\s+me\s+reconsider\b',
    r'\bhold\s+on\b',
    r'\bactually\b',
    r'\bhmm\b',
]
SEGMENT_MIN_PERIODS = 3

TEMPERATURE = 1.0
TOP_K       = 50
TOP_P       = 1.0

K_EVAL              = 4
EVAL_TEMPERATURE    = 0.7
MAX_ANSWER_TOKENS   = 256
TEACHER_CONT_TOKENS = 128

EPISODE_EVAL_SYSTEM = "Put your final answer within \\boxed{}."
EPISODE_EVAL_SUFFIX = (
    "\nBased on the reasoning above, stop thinking now and summarize the existing"
    " reasoning. If a definitive answer can be derived, give the final answer in"
    " \\boxed{answer}. If no answer has been reached yet or the result is still"
    " inconclusive, output \\boxed{no answer}."
)

OUTPUT_HTML = os.path.join(os.path.dirname(__file__), "kl_heatmap.html")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "kl_heatmap_data.json")
DTYPE       = torch.bfloat16

N_GPUS = torch.cuda.device_count()
# ── end config ───────────────────────────────────────────────────────────────


# ── answer extraction ────────────────────────────────────────────────────────

def extract_boxed_answer(text):
    """Extract the last \\boxed{...} from *text*, handling one level of nesting."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
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
    return normalize_answer(pred) == normalize_answer(gold) \
        if pred is not None and gold is not None else False


# ── helpers ──────────────────────────────────────────────────────────────────

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
    cfg     = DATASET_CONFIGS[DATASET]
    subsets = cfg["subsets"]
    print(f"Loading dataset '{DATASET}' ({len(subsets)} subsets) …", flush=True)
    pool = []
    for subset in subsets:
        path = os.path.join(_DATA_ROOT, DATASET, subset)
        ds   = load_from_disk(path)
        pool.extend(ds)
    total   = len(pool)
    idx     = random.sample(range(total), min(N_SAMPLES, total))
    samples = [pool[i] for i in idx]
    del pool
    print(f"  Sampled {len(samples)} / {total} rows", flush=True)
    return samples


# ── vLLM generation helper ───────────────────────────────────────────────────

def _vllm_generate(model_path, tokenizer, prompt_strings, do_sample=True):
    """Use vLLM for fast parallel generation. Returns list[(full_ids_tensor, prompt_len)]."""
    tp =N_GPUS # tensor-parallel degree
    print(f"  Initializing vLLM (tp={tp}) …", flush=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 512,  # prompt + generation headroom
        gpu_memory_utilization=0.85,
    )

    if do_sample:
        params = SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
        )
    else:
        params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0)

    print(f"  Generating {len(prompt_strings)} responses …", flush=True)
    outputs = llm.generate(prompt_strings, params)

    results = []
    for out in outputs:
        prompt_ids = out.prompt_token_ids
        gen_ids = out.outputs[0].token_ids
        full_ids = torch.tensor(list(prompt_ids) + list(gen_ids), dtype=torch.long)
        prompt_len = len(prompt_ids)
        results.append((full_ids, prompt_len))

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ── episode segmentation helpers ─────────────────────────────────────────────

def _add_segment(segments, tokens, start, end):
    """Compute stats for tokens[start:end] and append to segments list."""
    seg_toks = tokens[start:end]
    length = len(seg_toks)
    if length == 0:
        return
    kl_fwd_mean = sum(t['kl_forward'] for t in seg_toks) / length
    kl_rev_mean = sum(t['kl_reverse'] for t in seg_toks) / length
    ent_mean = sum(t['teacher_entropy'] for t in seg_toks) / length
    text = ''.join(t['text'] for t in seg_toks)
    segments.append(dict(
        start=start,
        end=end,
        length=length,
        kl_forward_mean=round(kl_fwd_mean, 6),
        kl_reverse_mean=round(kl_rev_mean, 6),
        teacher_entropy_mean=round(ent_mean, 6),
        text_preview=text[:200],
    ))


def segment_by_triggers(tokens, trigger_patterns=None, min_periods=None):
    """
    Split a token list into segments at trigger phrases, but only when at least
    min_periods sentence-ending periods have accumulated since the last cut.

    Returns a list of dicts: start, end, length, kl_forward_mean,
    kl_reverse_mean, text_preview.
    Each dict covers tokens[start:end] (end is exclusive).
    """
    if trigger_patterns is None:
        trigger_patterns = SEGMENT_TRIGGER_PATTERNS
    if min_periods is None:
        min_periods = SEGMENT_MIN_PERIODS

    combined = re.compile('|'.join(trigger_patterns), re.IGNORECASE)

    # Build cumulative character offsets for each token (for regex matching)
    tok_offsets = []
    pos = 0
    for tok in tokens:
        tok_offsets.append(pos)
        pos += len(tok['text'])
    full_text = ''.join(tok['text'] for tok in tokens)

    # Map every trigger regex match → the token index it starts in
    trigger_token_set = set()
    for m in combined.finditer(full_text):
        # bisect_right gives insertion point; subtract 1 for the containing token
        ti = bisect.bisect_right(tok_offsets, m.start()) - 1
        ti = max(0, ti)
        trigger_token_set.add(ti)

    # Walk tokens; cut at trigger positions when enough periods have elapsed
    segments = []
    current_start = 0
    period_count = 0

    for ti, tok in enumerate(tokens):
        period_count += tok['text'].count('.')

        if (ti in trigger_token_set
                and period_count >= min_periods
                and ti > current_start):
            _add_segment(segments, tokens, current_start, ti)
            current_start = ti
            # Reset counter; keep any periods in the current (first) token of
            # the new segment so we don't double-count them later.
            period_count = tok['text'].count('.')

    _add_segment(segments, tokens, current_start, len(tokens))
    return segments


# ── Episode eval + monotonic merge helpers ────────────────────────────────────

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


def merge_monotonic_runs_indexed(segs):
    """Like merge_monotonic_runs but also returns kept indices into segs."""
    n = len(segs)
    if n <= 1:
        return list(segs), list(range(n))

    def _sign(a, b):
        d = b["process_reward"] - a["process_reward"]
        return 1 if d > 0 else (-1 if d < 0 else 0)

    signs = [_sign(segs[i], segs[i + 1]) for i in range(n - 1)]
    active_dir = 0
    kept_indices = []

    for i in range(n - 1):
        s = signs[i]
        if s != 0:
            if active_dir != 0 and s != active_dir:
                kept_indices.append(i)
            active_dir = s
        if i == n - 2:
            kept_indices.append(n - 1)

    if not kept_indices:
        kept_indices = [n - 1]

    return [segs[idx] for idx in kept_indices], kept_indices


def phase4_episode_eval(case_data):
    print(f"\n{'='*60}")
    print(f"Phase 4: K={K_EVAL}-fold episode evaluation  "
          f"(model: {os.path.basename(TEACHER_MODEL)}, T={EVAL_TEMPERATURE})")
    print(f"{'='*60}", flush=True)

    eval_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    eval_items = []
    for ci, case in enumerate(case_data):
        for ep_idx, seg in enumerate(case.get("segments", [])):
            partial_resp = "".join(tok["text"] for tok in case["tokens"][: seg["end"]])
            eval_items.append(dict(
                case_idx=ci,
                ep_idx=ep_idx,
                ep_num=ep_idx + 1,
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
        model=TEACHER_MODEL,
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
        item["acc_rate"] = round(sum(corrects) / K_EVAL, 4)

    print(f"  Evaluated {len(eval_items)} items.", flush=True)
    return eval_items


def phase5_merge_and_continuations(case_data, eval_items):
    print(f"\n{'='*60}")
    print("Phase 5: Monotonic merge + teacher continuations")
    print(f"{'='*60}", flush=True)

    # ── Attach acc_rate + process_reward to each segment ─────────────────
    ep_map = defaultdict(dict)
    for item in eval_items:
        ep_map[item["case_idx"]][item["ep_num"]] = item

    for ci, case in enumerate(case_data):
        prev_acc = 0.0
        for ep_idx, seg in enumerate(case["segments"]):
            item     = ep_map[ci].get(ep_idx + 1, {})
            acc      = item.get("acc_rate", 0.0)
            seg["acc_rate"]       = acc
            seg["process_reward"] = round(acc - prev_acc, 6)
            prev_acc = acc

    # ── Monotonic merge per case ──────────────────────────────────────────
    for ci, case in enumerate(case_data):
        segs = case["segments"]
        if not segs:
            case["merged_segments"] = []
            continue

        _, kept_indices = merge_monotonic_runs_indexed(segs)
        run_starts = [0] + [kept_indices[j] + 1 for j in range(len(kept_indices) - 1)]

        merged = []
        for mi, (kept_idx, run_start) in enumerate(zip(kept_indices, run_starts)):
            run_segs  = segs[run_start:kept_idx + 1]
            total_len = sum(s["length"] for s in run_segs) or 1
            kl_fwd = sum(s["kl_forward_mean"] * s["length"] for s in run_segs) / total_len
            kl_rev = sum(s["kl_reverse_mean"] * s["length"] for s in run_segs) / total_len
            ent    = sum(s["teacher_entropy_mean"] * s["length"] for s in run_segs) / total_len
            merged.append(dict(
                merged_idx=mi,
                orig_ep_start=run_start + 1,
                orig_ep_end=kept_idx + 1,
                start=segs[run_start]["start"],
                end=segs[kept_idx]["end"],
                length=total_len,
                kl_forward_mean=round(kl_fwd, 6),
                kl_reverse_mean=round(kl_rev, 6),
                teacher_entropy_mean=round(ent, 6),
                process_reward=segs[kept_idx]["process_reward"],
                acc_rate=segs[kept_idx]["acc_rate"],
                teacher_continuation="",
            ))

        case["merged_segments"] = merged

        # Re-annotate token merged_segment indices
        for mi, mseg in enumerate(merged):
            for ti in range(mseg["start"], mseg["end"]):
                if ti < len(case["tokens"]):
                    case["tokens"][ti]["merged_segment"] = mi

        raw_n = len(segs)
        mrg_n = len(merged)
        print(f"  Case {ci}: {raw_n} segs → {mrg_n} merged "
              f"({(1 - mrg_n / max(raw_n, 1)) * 100:.0f}% reduction)", flush=True)

    # ── Batch-generate teacher continuations ──────────────────────────────
    cont_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    cont_items = []
    for ci, case in enumerate(case_data):
        for mseg in case.get("merged_segments", []):
            partial = "".join(tok["text"] for tok in case["tokens"][:mseg["end"]])
            msgs = [{"role": "user", "content": case["problem"]}]
            # add_generation_prompt=False to avoid appending <think> token
            # then manually add assistant turn prefix so partial starts correctly
            base = cont_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            base = base + "<|im_start|>assistant\n"
            cont_items.append(dict(ci=ci, mi=mseg["merged_idx"], prompt=base + partial))

    if cont_items:
        tp = min(N_GPUS, 4)
        llm = LLM(
            model=TEACHER_MODEL,
            tensor_parallel_size=tp,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=MAX_NEW_TOKENS + 1024,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
        )
        params = SamplingParams(
            n=1,
            max_tokens=TEACHER_CONT_TOKENS,
            temperature=0.0,
            stop=["<|im_end|>"],
        )
        print(f"  Generating {len(cont_items)} teacher continuations …", flush=True)
        outputs = llm.generate([item["prompt"] for item in cont_items], params)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        for item, out in zip(cont_items, outputs):
            cont_text = out.outputs[0].text if out.outputs else ""
            case_data[item["ci"]]["merged_segments"][item["mi"]]["teacher_continuation"] = cont_text

    print(f"  Done. {len(cont_items)} continuations generated.", flush=True)
    return case_data


# ── Phase 1: student on-policy sampling ──────────────────────────────────────

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
    for si, (sample, (full_ids, prompt_len), gt_answer) in enumerate(
            zip(samples, gen_outputs, gt_answers)):
        response_ids = full_ids[prompt_len:]
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
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  gt={gt_answer}  pred={student_answer}  "
                  f"len={len(response_ids)}", flush=True)

    n_ok = sum(r["student_correct"] for r in results)
    print(f"  Student accuracy: {n_ok}/{len(results)} ({n_ok/len(results):.1%})", flush=True)
    return results, tokenizer


# ── Phase 2: teacher greedy generation ───────────────────────────────────────

def phase2_teacher(samples, student_results):
    print(f"\n{'='*60}")
    print(f"Phase 2: Teacher generation ({os.path.basename(TEACHER_MODEL)}) "
          f"[vLLM, {N_GPUS} GPU(s)]")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    cfg = DATASET_CONFIGS[DATASET]
    prompt_strings = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]

    gen_outputs = _vllm_generate(TEACHER_MODEL, tokenizer, prompt_strings, do_sample=False)

    for si, (sr, (full_ids, prompt_len)) in enumerate(zip(student_results, gen_outputs)):
        response_ids = full_ids[prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        teacher_answer = extract_boxed_answer(response_text)
        correct = answers_match(teacher_answer, sr["gt_answer"])

        sr["teacher_response"] = response_text
        sr["teacher_answer"] = teacher_answer
        sr["teacher_correct"] = correct

        if (si + 1) % 10 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  gt={sr['gt_answer']}  pred={teacher_answer}",
                  flush=True)

    n_ok = sum(r["teacher_correct"] for r in student_results)
    print(f"  Teacher accuracy: {n_ok}/{len(student_results)} ({n_ok/len(student_results):.1%})",
          flush=True)

    candidates = [i for i, r in enumerate(student_results)
                  if not r["student_correct"] and r["teacher_correct"]]
    print(f"  Student-wrong & Teacher-right: {len(candidates)} cases", flush=True)

    selected = candidates[:N_CASES]
    if len(selected) < N_CASES:
        print(f"  WARNING: only {len(selected)} qualifying cases found (wanted {N_CASES})")

    return selected


# ── Phase 3: per-token KL divergence ────────────────────────────────────────

def _make_device_map(model_name, gpu_ids):
    """Create a device_map that constrains a model to specific GPU IDs."""
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
    """Forward-pass, return logits on CPU. Handles long sequences with chunking."""
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
    print(f"Phase 3: Per-token KL divergence ({len(selected_indices)} cases, {N_GPUS} GPUs)")
    print(f"{'='*60}", flush=True)

    case_inputs = []
    for si in selected_indices:
        sr = student_results[si]
        case_inputs.append(dict(full_ids=sr["full_ids"], prompt_len=sr["prompt_len"]))

    # Split GPUs: student on first chunk, teacher on the rest
    # 1.7B student is small → 1 GPU; 8B teacher gets the remaining GPUs
    all_gpus = list(range(N_GPUS))
    if N_GPUS >= 2:
        s_gpus = all_gpus[:1]
        t_gpus = all_gpus[1:]
        load_simultaneously = True
        print(f"  Student → GPU {s_gpus}  |  Teacher → GPU {t_gpus}  (simultaneous)", flush=True)
    else:
        s_gpus = t_gpus = [0]
        load_simultaneously = False
        print(f"  Single GPU mode: sequential load/unload", flush=True)

    # ---- Load models (simultaneously if multi-GPU) ----
    print("  Loading student model …", flush=True)
    s_device_map = _make_device_map(STUDENT_MODEL, s_gpus)
    s_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=DTYPE, device_map=s_device_map, trust_remote_code=True,
    )
    s_model.eval()

    if load_simultaneously:
        print("  Loading teacher model …", flush=True)
        t_device_map = _make_device_map(TEACHER_MODEL, t_gpus)
        t_model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, torch_dtype=DTYPE, device_map=t_device_map, trust_remote_code=True,
        )
        t_model.eval()
        print("  Both models loaded simultaneously.", flush=True)

    s_dev = f"cuda:{s_gpus[0]}"
    student_logits_list = []
    for ci, ci_data in enumerate(case_inputs):
        ids = ci_data["full_ids"].unsqueeze(0)
        logits = _forward_logits(s_model, ids, s_dev)
        student_logits_list.append(logits)
        print(f"    Student forward case {ci}: seq_len={ids.shape[1]}", flush=True)

    if not load_simultaneously:
        del s_model
        gc.collect()
        torch.cuda.empty_cache()
        print("  Loading teacher model …", flush=True)
        t_device_map = _make_device_map(TEACHER_MODEL, t_gpus)
        t_model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, torch_dtype=DTYPE, device_map=t_device_map, trust_remote_code=True,
        )
        t_model.eval()

    t_dev = f"cuda:{t_gpus[0]}"
    teacher_logits_list = []
    for ci, ci_data in enumerate(case_inputs):
        ids = ci_data["full_ids"].unsqueeze(0)
        logits = _forward_logits(t_model, ids, t_dev)
        teacher_logits_list.append(logits)
        print(f"    Teacher forward case {ci}: seq_len={ids.shape[1]}", flush=True)

    if load_simultaneously:
        del s_model
    del t_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Models unloaded.", flush=True)

    # ---- Compute KL per token ----
    case_data = []
    for ci, si in enumerate(selected_indices):
        sr = student_results[si]
        prompt_len = sr["prompt_len"]
        full_ids = sr["full_ids"]
        seq_len = full_ids.shape[0]
        resp_len = seq_len - prompt_len

        if resp_len <= 1:
            continue

        s_resp = student_logits_list[ci][prompt_len - 1: seq_len - 1]  # [resp_len, V]
        t_resp = teacher_logits_list[ci][prompt_len - 1: seq_len - 1]

        s_log_p = F.log_softmax(s_resp.float(), dim=-1)
        t_log_p = F.log_softmax(t_resp.float(), dim=-1)
        s_p = s_log_p.exp()
        t_p = t_log_p.exp()

        kl_fwd = (t_p * (t_log_p - s_log_p)).sum(-1).tolist()   # KL(T‖S)
        kl_rev = (s_p * (s_log_p - t_log_p)).sum(-1).tolist()   # KL(S‖T)
        t_ent = (-(t_p * t_log_p).sum(-1)).tolist()

        resp_ids = full_ids[prompt_len: seq_len].tolist()

        # Find the </think> boundary to annotate thinking vs answer regions
        response_text = student_tokenizer.decode(resp_ids)
        think_end_pos = response_text.find("</think>")
        think_boundary_token = None
        if think_end_pos >= 0:
            prefix_to_think_end = response_text[:think_end_pos + len("</think>")]
            think_boundary_token = len(student_tokenizer.encode(prefix_to_think_end))

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

        # ---- Episode segmentation ----
        segments = segment_by_triggers(tokens)
        for seg_idx, seg in enumerate(segments):
            for ti in range(seg['start'], seg['end']):
                tokens[ti]['segment'] = seg_idx

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

        fwd_arr = np.array(kl_fwd)
        rev_arr = np.array(kl_rev)
        ent_arr = np.array(t_ent)
        n_answer = sum(1 for tk in tokens if tk["region"] == "answer")
        print(f"  Case {ci}: resp_len={resp_len}  "
              f"(thinking={resp_len - n_answer}, answer={n_answer})  "
              f"segments={len(segments)}  "
              f"KL(T‖S) mean={fwd_arr.mean():.4f} p90={np.percentile(fwd_arr, 90):.4f}  "
              f"KL(S‖T) mean={rev_arr.mean():.4f} p90={np.percentile(rev_arr, 90):.4f}  "
              f"Teacher-H mean={ent_arr.mean():.4f} p90={np.percentile(ent_arr, 90):.4f}",
              flush=True)
        if segments:
            seg_ent = [seg['teacher_entropy_mean'] for seg in segments]
            print(f"    Segment teacher-H: " +
                  "  ".join(f"ep{i+1}={v:.4f}" for i, v in enumerate(seg_ent)),
                  flush=True)

    return case_data


# ── Phase 4: HTML generation ────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KL Divergence Heatmap — Student vs Teacher</title>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body,{delimiters:[
          {left:'$$',right:'$$',display:true},
          {left:'$',right:'$',display:false},
          {left:'\\\\(',right:'\\\\)',display:false},
          {left:'\\\\[',right:'\\\\]',display:true}
        ]});"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0f0f1a;color:#d8d8e8;
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
  line-height:1.6;padding:32px 24px;
}
.container{max-width:1100px;margin:0 auto}
h1{
  font-size:1.6rem;font-weight:700;letter-spacing:-.02em;
  margin-bottom:6px;color:#f0f0ff;
}
.subtitle{font-size:.85rem;color:#888;margin-bottom:28px}
/* ── controls ── */
.controls{
  background:#181830;border:1px solid #2a2a4a;border-radius:10px;
  padding:20px 24px;margin-bottom:32px;display:flex;gap:32px;
  flex-wrap:wrap;align-items:center;
}
.ctrl-group{display:flex;flex-direction:column;gap:4px}
.ctrl-group label{font-size:.78rem;color:#999;text-transform:uppercase;letter-spacing:.06em}
.ctrl-group .value{font-size:1.05rem;font-weight:600;color:#7ecfff;font-variant-numeric:tabular-nums}
input[type=range]{
  width:260px;accent-color:#5b8def;cursor:pointer;
}
select{
  background:#22224a;color:#ccc;border:1px solid #3a3a6a;
  border-radius:6px;padding:4px 8px;font-size:.85rem;
}
.formula{
  font-size:.82rem;color:#7a7aa0;
  font-family:'SF Mono','Fira Code',monospace;
}
/* ── case card ── */
.case-card{
  background:#14142a;border:1px solid #252548;border-radius:12px;
  padding:24px 28px;margin-bottom:28px;
}
.case-header{
  display:flex;justify-content:space-between;align-items:baseline;
  margin-bottom:16px;border-bottom:1px solid #252548;padding-bottom:12px;
}
.case-title{font-size:1.1rem;font-weight:600;color:#c0c0ff}
.answers{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
.answer-chip{
  font-size:.82rem;padding:4px 12px;border-radius:6px;
  font-family:'SF Mono','Fira Code',monospace;
}
.answer-gt{background:#1a3a1a;color:#6fcf6f;border:1px solid #2a5a2a}
.answer-student{background:#3a1a1a;color:#cf6f6f;border:1px solid #5a2a2a}
.answer-teacher{background:#1a2a3a;color:#6fafcf;border:1px solid #2a4a5a}
/* ── problem text ── */
.problem-text{
  background:#1a1a30;border-radius:8px;padding:14px 18px;
  margin-bottom:18px;font-size:.88rem;color:#b8b8d8;
  border-left:3px solid #4a4a8a;
  max-height:200px;overflow-y:auto;
}
/* ── heatmap ── */
.heatmap-label{font-size:.78rem;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}
.heatmap{
  font-family:'SF Mono','Fira Code','Cascadia Code',monospace;
  font-size:.82rem;line-height:1.9;
  padding:16px;background:#0d0d1a;border-radius:8px;
  border:1px solid #1e1e3a;
  word-break:break-all;
  position:relative;
}
.tok{
  display:inline;padding:1px 0;border-radius:2px;
  cursor:pointer;transition:outline .15s;
  position:relative;
}
.tok:hover{outline:2px solid #7ecfff;outline-offset:1px;z-index:2}
.tok.thinking{opacity:0.45}
.tok.thinking.dim{display:none}
.think-divider{
  display:block;margin:8px 0;padding:4px 10px;
  font-size:.72rem;color:#8888aa;border-top:1px dashed #3a3a5a;
}
/* ── tooltip ── */
.tooltip{
  display:none;position:fixed;
  background:#1c1c3a;border:1px solid #3a3a6a;border-radius:8px;
  padding:10px 14px;font-size:.78rem;color:#ccc;
  z-index:100;pointer-events:none;
  box-shadow:0 8px 24px rgba(0,0,0,.5);
  min-width:200px;
  font-family:'SF Mono','Fira Code',monospace;
}
.tooltip .tt-token{font-size:.95rem;font-weight:700;color:#f0f0ff;margin-bottom:6px}
.tooltip .tt-row{display:flex;justify-content:space-between;gap:12px;padding:1px 0}
.tooltip .tt-label{color:#888}
.tooltip .tt-val{color:#7ecfff;font-weight:600}
/* ── legend ── */
.legend{
  display:flex;align-items:center;gap:8px;margin-top:10px;
  font-size:.75rem;color:#888;
}
.legend-bar{
  width:200px;height:14px;border-radius:4px;
  background:linear-gradient(to right,
    rgba(91,141,239,0.0),
    rgba(91,141,239,0.15),
    rgba(239,180,71,0.4),
    rgba(239,100,71,0.7),
    rgba(220,50,50,1.0));
}
/* ── response preview ── */
.resp-toggle{
  font-size:.78rem;color:#6688bb;cursor:pointer;
  text-decoration:underline;margin-top:10px;display:inline-block;
}
.resp-preview{
  display:none;background:#111128;border-radius:8px;padding:14px;
  margin-top:8px;font-size:.82rem;color:#aaa;
  max-height:300px;overflow-y:auto;white-space:pre-wrap;
  border:1px solid #1e1e3a;
}
.resp-preview.show{display:block}
/* ── segment divider (inside heatmap) ── */
.seg-divider{
  display:block;margin:5px 0 2px;padding:2px 10px;
  font-size:.68rem;color:#5599cc;
  border-top:1px solid #2a4a6a;
  letter-spacing:.04em;
}
/* ── teacher continuation ── */
.cont-wrap{display:block;margin:4px 0 8px;padding:0 10px}
.cont-toggle{
  font-size:.70rem;color:#7a9acc;cursor:pointer;
  text-decoration:underline dotted;letter-spacing:.02em;user-select:none;
}
.cont-toggle:hover{color:#aaccff}
.cont-text{
  display:none;margin-top:5px;padding:10px 14px;
  background:#131325;border-radius:6px;
  border:1px solid #2a3a5a;
  font-family:'SF Mono','Fira Code',monospace;
  font-size:.80rem;color:#c0d0e8;line-height:1.6;
  white-space:pre-wrap;max-height:280px;overflow-y:auto;
}
/* ── episode-level bar ── */
.episode-label{
  font-size:.78rem;color:#888;margin:16px 0 6px;
  text-transform:uppercase;letter-spacing:.05em;
}
.episode-view{
  display:flex;height:44px;border-radius:7px;overflow:hidden;
  border:1px solid #1e1e3a;margin-bottom:6px;
}
.ep-seg{
  height:100%;display:flex;align-items:center;justify-content:center;
  font-size:.68rem;color:rgba(255,255,255,.8);font-weight:600;
  cursor:pointer;transition:filter .15s;min-width:2px;overflow:hidden;
  border-right:1px solid rgba(0,0,0,.25);
  font-family:'SF Mono','Fira Code',monospace;
}
.ep-seg:hover{filter:brightness(1.5);z-index:2}
.ep-seg:last-child{border-right:none}
.episode-axis{
  display:flex;font-size:.66rem;color:#555;margin-bottom:4px;
}
.episode-axis span{text-align:center;overflow:hidden}
/* ── entropy bar ── */
.entropy-label{
  font-size:.78rem;color:#888;margin:10px 0 6px;
  text-transform:uppercase;letter-spacing:.05em;
}
.entropy-view{
  display:flex;height:28px;border-radius:7px;overflow:hidden;
  border:1px solid #1e1e3a;margin-bottom:4px;
}
.ent-seg{
  height:100%;display:flex;align-items:center;justify-content:center;
  font-size:.65rem;color:rgba(255,255,255,.8);font-weight:600;
  cursor:pointer;transition:filter .15s;min-width:2px;overflow:hidden;
  border-right:1px solid rgba(0,0,0,.25);
  font-family:'SF Mono','Fira Code',monospace;
}
.ent-seg:hover{filter:brightness(1.5);z-index:2}
.ent-seg:last-child{border-right:none}
</style>
</head>
<body>
<div class="container">
  <h1>KL Divergence Heatmap</h1>
  <div class="subtitle">
    Student: __STUDENT__ &nbsp;|&nbsp; Teacher: __TEACHER__ &nbsp;|&nbsp;
    Dataset: __DATASET__ &nbsp;|&nbsp; Cases: student wrong, teacher right
  </div>

  <div class="controls">
    <div class="ctrl-group">
      <label>Forward / Reverse ratio &nbsp; α</label>
      <div style="display:flex;align-items:center;gap:10px">
        <input type="range" id="alpha" min="0" max="1" step="0.01" value="0.5">
        <span class="value" id="alpha-val">0.50</span>
      </div>
      <div class="formula">mixed = α · KL(T‖S) + (1−α) · KL(S‖T)</div>
    </div>
    <div class="ctrl-group">
      <label>Color scale</label>
      <select id="scale">
        <option value="sqrt">√ (sqrt)</option>
        <option value="linear">Linear</option>
        <option value="log" selected>Log(1+x)</option>
      </select>
    </div>
    <div class="ctrl-group">
      <label>Normalize</label>
      <select id="norm">
        <option value="per_case" selected>Per case (local max)</option>
        <option value="global">Global max</option>
      </select>
    </div>
    <div class="ctrl-group">
      <label>Region</label>
      <select id="region">
        <option value="all" selected>Show all</option>
        <option value="answer_only">Answer only (hide thinking)</option>
        <option value="thinking_dim">Dim thinking</option>
      </select>
    </div>
  </div>

  <div id="cases"></div>
  <div class="tooltip" id="tooltip"></div>
</div>

<script>
const DATA = __JSON_DATA__;
const TEACHER_CONT_TOKENS = __TEACHER_CONT_TOKENS__;

function appendContinuation(container, text, mergedEpNum) {
  const wrap = document.createElement('div');
  wrap.className = 'cont-wrap';
  const toggle = document.createElement('span');
  toggle.className = 'cont-toggle';
  toggle.textContent = '▸ Teacher continuation — merged ep ' + mergedEpNum + ' (' + TEACHER_CONT_TOKENS + ' tok)';
  const pre = document.createElement('pre');
  pre.className = 'cont-text';
  toggle.onclick = () => {
    const show = pre.style.display !== 'block';
    pre.style.display = show ? 'block' : 'none';
    toggle.textContent = (show ? '▾' : '▸') +
      ' Teacher continuation — merged ep ' + mergedEpNum + ' (' + TEACHER_CONT_TOKENS + ' tok)';
  };
  pre.textContent = text;
  wrap.appendChild(toggle);
  wrap.appendChild(pre);
  container.appendChild(wrap);
}

const alphaSlider  = document.getElementById('alpha');
const alphaVal     = document.getElementById('alpha-val');
const scaleSelect  = document.getElementById('scale');
const normSelect   = document.getElementById('norm');
const regionSelect = document.getElementById('region');
const tooltip      = document.getElementById('tooltip');

function scaleFn(v, mode) {
  if (mode === 'sqrt')   return Math.sqrt(v);
  if (mode === 'log')    return Math.log1p(v);
  return v;
}

function klColor(ratio) {
  // ratio in [0,1] → blue→yellow→red
  const r = Math.min(1, Math.max(0, ratio));
  if (r < 0.25) {
    const t = r / 0.25;
    return `rgba(91,141,239,${(t * 0.2).toFixed(3)})`;
  } else if (r < 0.5) {
    const t = (r - 0.25) / 0.25;
    const red   = Math.round(91 + (239 - 91) * t);
    const green = Math.round(141 + (180 - 141) * t);
    const blue  = Math.round(239 + (71 - 239) * t);
    return `rgba(${red},${green},${blue},${(0.2 + t * 0.25).toFixed(3)})`;
  } else if (r < 0.75) {
    const t = (r - 0.5) / 0.25;
    const red   = Math.round(239 - (239 - 239) * t);
    const green = Math.round(180 - (180 - 100) * t);
    const blue  = 71;
    return `rgba(${red},${green},${blue},${(0.45 + t * 0.25).toFixed(3)})`;
  } else {
    const t = (r - 0.75) / 0.25;
    const red   = Math.round(239 - (239 - 220) * t);
    const green = Math.round(100 - (100 - 50) * t);
    const blue  = Math.round(71 - (71 - 50) * t);
    return `rgba(${red},${green},${blue},${(0.7 + t * 0.3).toFixed(3)})`;
  }
}

function entColor(ratio) {
  // ratio in [0,1] → dark-teal → cyan → lime → yellow  (entropy palette)
  const r = Math.min(1, Math.max(0, ratio));
  if (r < 0.33) {
    const t = r / 0.33;
    const green = Math.round(120 + (210 - 120) * t);
    const blue  = Math.round(160 + (200 - 160) * t);
    return `rgba(30,${green},${blue},${(0.15 + t * 0.3).toFixed(3)})`;
  } else if (r < 0.66) {
    const t = (r - 0.33) / 0.33;
    const red   = Math.round(30 + (160 - 30) * t);
    const green = Math.round(210 - (210 - 220) * t);
    const blue  = Math.round(200 - (200 - 60) * t);
    return `rgba(${red},${green},${blue},${(0.45 + t * 0.25).toFixed(3)})`;
  } else {
    const t = (r - 0.66) / 0.34;
    const red   = Math.round(160 + (230 - 160) * t);
    const green = Math.round(220 - (220 - 180) * t);
    const blue  = Math.round(60 - (60 - 30) * t);
    return `rgba(${red},${green},${blue},${(0.7 + t * 0.3).toFixed(3)})`;
  }
}

function renderAll() {
  const alpha = parseFloat(alphaSlider.value);
  const scale = scaleSelect.value;
  const normMode = normSelect.value;
  const regionMode = regionSelect.value;

  alphaVal.textContent = alpha.toFixed(2);

  // Compute mixed KL for all tokens
  DATA.forEach(c => {
    c.tokens.forEach(tok => {
      tok._mixed = alpha * tok.kl_forward + (1 - alpha) * tok.kl_reverse;
    });
  });

  // Global max for normalization
  let globalMax = 0;
  if (normMode === 'global') {
    DATA.forEach(c => c.tokens.forEach(tok => {
      const sv = scaleFn(tok._mixed, scale);
      if (sv > globalMax) globalMax = sv;
    }));
    if (globalMax === 0) globalMax = 1;
  }

  const container = document.getElementById('cases');
  container.innerHTML = '';

  DATA.forEach((c, ci) => {
    let localMax = 0;
    if (normMode === 'per_case') {
      c.tokens.forEach(tok => {
        const sv = scaleFn(tok._mixed, scale);
        if (sv > localMax) localMax = sv;
      });
      if (localMax === 0) localMax = 1;
    }
    const maxVal = normMode === 'global' ? globalMax : localMax;

    const card = document.createElement('div');
    card.className = 'case-card';

    // Stats (answer region only)
    const answerTokens = c.tokens.filter(t => t.region === 'answer');
    const allTokens = c.tokens;
    const statTokens = answerTokens.length > 0 ? answerTokens : allTokens;
    const klFwdMean = (statTokens.reduce((s,t) => s + t.kl_forward, 0) / statTokens.length).toFixed(4);
    const klRevMean = (statTokens.reduce((s,t) => s + t.kl_reverse, 0) / statTokens.length).toFixed(4);
    const mixedMean = (statTokens.reduce((s,t) => s + t._mixed, 0) / statTokens.length).toFixed(4);

    const nThink = allTokens.length - answerTokens.length;
    const nAnswer = answerTokens.length;

    card.innerHTML = `
      <div class="case-header">
        <span class="case-title">Case ${ci + 1}
          <span style="font-size:.72rem;color:#666;font-weight:400;margin-left:8px">
            thinking: ${nThink} tok &nbsp;|&nbsp; answer: ${nAnswer} tok
          </span>
        </span>
        <span style="font-size:.78rem;color:#888">
          Answer KL: fwd μ=${klFwdMean} &nbsp; rev μ=${klRevMean} &nbsp; mixed μ=${mixedMean}
        </span>
      </div>
      <div class="problem-text">${htmlSafe(c.problem)}</div>
      <div class="answers">
        <span class="answer-chip answer-gt">GT: ${htmlSafe(c.gt_answer || '?')}</span>
        <span class="answer-chip answer-student">Student: ${htmlSafe(c.student_answer || 'N/A')}</span>
        <span class="answer-chip answer-teacher">Teacher: ${htmlSafe(c.teacher_answer || 'N/A')}</span>
      </div>
      <div class="episode-label">Episode-level KL &nbsp;<span style="color:#555;font-size:.72rem;text-transform:none;letter-spacing:0">(width ∝ token count · color ∝ mean KL · hover for details)</span></div>
      <div class="episode-view" id="ep-${ci}"></div>
      <div class="episode-axis" id="ep-axis-${ci}"></div>
      <div class="entropy-label">Teacher Entropy per segment &nbsp;<span style="color:#555;font-size:.72rem;text-transform:none;letter-spacing:0">(width ∝ token count · color ∝ mean H · hover for details)</span></div>
      <div class="entropy-view" id="ent-${ci}"></div>
      <div class="heatmap-label" style="margin-top:14px">Student response — token-level KL heatmap (hover for details)</div>
      <div class="heatmap" id="hm-${ci}"></div>
      <div class="legend">
        <span>Low KL</span><div class="legend-bar"></div><span>High KL</span>
        <span style="margin-left:8px;color:#666">(max scaled = ${maxVal.toFixed(3)})</span>
      </div>
      <span class="resp-toggle" onclick="this.nextElementSibling.classList.toggle('show')">
        ▸ Show teacher response
      </span>
      <div class="resp-preview">${htmlSafe(c.teacher_response)}</div>
    `;
    container.appendChild(card);

    // ── Episode bar (merged segments) ────────────────────────────────────
    const msegs = c.merged_segments || [];
    if (msegs.length > 0) {
      const epView  = document.getElementById(`ep-${ci}`);
      const epAxis  = document.getElementById(`ep-axis-${ci}`);
      const entView = document.getElementById(`ent-${ci}`);
      epView.innerHTML = '';
      epAxis.innerHTML = '';
      entView.innerHTML = '';

      const totalLen = msegs.reduce((s, ms) => s + ms.length, 0);

      const segMixed = msegs.map(ms =>
        alpha * ms.kl_forward_mean + (1 - alpha) * ms.kl_reverse_mean);
      const segScaled = segMixed.map(v => scaleFn(v, scale));
      const segMax = Math.max(...segScaled, 1e-9);

      const segEnt = msegs.map(ms => ms.teacher_entropy_mean || 0);
      const segEntScaled = segEnt.map(v => scaleFn(v, scale));
      const segEntMax = Math.max(...segEntScaled, 1e-9);

      msegs.forEach((ms, mi) => {
        const widthPct = (ms.length / totalLen * 100).toFixed(3);
        const ratio    = segScaled[mi] / segMax;

        const el = document.createElement('div');
        el.className  = 'ep-seg';
        el.style.width = widthPct + '%';
        el.style.backgroundColor = klColor(ratio);
        if (ms.length > 25) el.textContent = `#${mi + 1}`;
        el.dataset.ci = ci;
        el.dataset.mi = mi;
        epView.appendChild(el);

        const lbl = document.createElement('span');
        lbl.style.width = widthPct + '%';
        lbl.style.paddingLeft = '2px';
        const epRange = ms.orig_ep_start === ms.orig_ep_end
          ? `ep${ms.orig_ep_start}` : `ep${ms.orig_ep_start}-${ms.orig_ep_end}`;
        lbl.title = `${epRange}: ${ms.length} tok · KL=${segMixed[mi].toFixed(3)}`;
        lbl.textContent = ms.length > 20 ? `${ms.length}t` : '';
        epAxis.appendChild(lbl);

        const entRatio = segEntScaled[mi] / segEntMax;
        const entEl = document.createElement('div');
        entEl.className = 'ent-seg';
        entEl.style.width = widthPct + '%';
        entEl.style.backgroundColor = entColor(entRatio);
        if (ms.length > 25) entEl.textContent = `#${mi + 1}`;
        entEl.dataset.ci = ci;
        entEl.dataset.mi = mi;
        entEl.dataset.isEnt = '1';
        entView.appendChild(entEl);
      });
    }

    // ── Token heatmap ────────────────────────────────────────────────────
    const hm = document.getElementById(`hm-${ci}`);
    let inThinking = true;
    let curMergedSeg = -1;
    c.tokens.forEach((tok, ti) => {
      if (tok.region === 'answer' && inThinking) {
        inThinking = false;
        const divider = document.createElement('span');
        divider.className = 'think-divider';
        divider.textContent = '── </think> ── answer below ──';
        hm.appendChild(divider);
      }

      const tokMSeg = tok.merged_segment !== undefined ? tok.merged_segment : 0;
      if (tokMSeg !== curMergedSeg && curMergedSeg >= 0) {
        const prevMs = (c.merged_segments || [])[curMergedSeg];
        const prStr = (prevMs && prevMs.process_reward !== undefined)
          ? '  PR ' + (prevMs.process_reward >= 0 ? '+' : '') + prevMs.process_reward.toFixed(3)
          : '';
        const segDiv = document.createElement('span');
        segDiv.className = 'seg-divider';
        segDiv.textContent = '── merged ep ' + (curMergedSeg + 1) + prStr + ' ──';
        hm.appendChild(segDiv);
        if (prevMs && prevMs.teacher_continuation) {
          appendContinuation(hm, prevMs.teacher_continuation, curMergedSeg + 1);
        }
      }
      curMergedSeg = tokMSeg;

      const sv = scaleFn(tok._mixed, scale);
      const ratio = sv / maxVal;
      const span = document.createElement('span');
      span.className = 'tok';
      if (tok.region === 'thinking') {
        span.classList.add('thinking');
        if (regionMode === 'answer_only') span.classList.add('dim');
      }
      span.style.backgroundColor = klColor(ratio);

      let display = tok.text;
      if (display === '\n' || display === '\r\n') display = '↵\n';
      span.textContent = display;

      span.dataset.ci = ci;
      span.dataset.ti = ti;
      hm.appendChild(span);
    });

    // Show divider + continuation for the last merged segment
    if (curMergedSeg >= 0) {
      const lastMs = (c.merged_segments || [])[curMergedSeg];
      const prStr = (lastMs && lastMs.process_reward !== undefined)
        ? '  PR ' + (lastMs.process_reward >= 0 ? '+' : '') + lastMs.process_reward.toFixed(3)
        : '';
      const segDiv = document.createElement('span');
      segDiv.className = 'seg-divider';
      segDiv.textContent = '── merged ep ' + (curMergedSeg + 1) + prStr + ' (end) ──';
      hm.appendChild(segDiv);
      if (lastMs && lastMs.teacher_continuation) {
        appendContinuation(hm, lastMs.teacher_continuation, curMergedSeg + 1);
      }
    }
  });
}

function htmlSafe(s) {
  if (!s) return '';
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

// Tooltip
document.addEventListener('mousemove', e => {
  const el = e.target;

  // ── Episode KL bar tooltip (merged segments) ─────────────────────────
  if (el.classList.contains('ep-seg')) {
    const ci = parseInt(el.dataset.ci);
    const mi = parseInt(el.dataset.mi);
    const ms = (DATA[ci].merged_segments || [])[mi];
    if (!ms) { tooltip.style.display = 'none'; return; }
    const alpha   = parseFloat(alphaSlider.value);
    const mixed   = (alpha * ms.kl_forward_mean + (1 - alpha) * ms.kl_reverse_mean).toFixed(4);
    const entMean = (ms.teacher_entropy_mean || 0).toFixed(4);
    const prStr   = ms.process_reward !== undefined
      ? (ms.process_reward >= 0 ? '+' : '') + ms.process_reward.toFixed(4) : '—';
    const epRange = ms.orig_ep_start === ms.orig_ep_end
      ? `Ep ${ms.orig_ep_start}` : `Ep ${ms.orig_ep_start}–${ms.orig_ep_end}`;
    tooltip.innerHTML = `
      <div class="tt-token">${epRange} (merged) — KL</div>
      <div class="tt-row"><span class="tt-label">Tokens</span><span class="tt-val">${ms.length}</span></div>
      <div class="tt-row"><span class="tt-label">KL(T‖S) mean</span><span class="tt-val">${ms.kl_forward_mean.toFixed(4)}</span></div>
      <div class="tt-row"><span class="tt-label">KL(S‖T) mean</span><span class="tt-val">${ms.kl_reverse_mean.toFixed(4)}</span></div>
      <div class="tt-row"><span class="tt-label">Mixed KL</span><span class="tt-val">${mixed}</span></div>
      <div class="tt-row"><span class="tt-label">Teacher H mean</span><span class="tt-val" style="color:#6fcfa0">${entMean}</span></div>
      <div class="tt-row"><span class="tt-label">Process Reward</span><span class="tt-val" style="color:${ms.process_reward >= 0 ? '#4caf50' : '#f44336'}">${prStr}</span></div>
      <div class="tt-row"><span class="tt-label">Acc Rate</span><span class="tt-val">${ms.acc_rate !== undefined ? ms.acc_rate.toFixed(4) : '—'}</span></div>
    `;
    tooltip.style.display = 'block';
    let left = e.clientX + 16, top = e.clientY + 16;
    if (left + 300 > window.innerWidth)  left = e.clientX - 310;
    if (top  + 260 > window.innerHeight) top  = e.clientY - 260;
    tooltip.style.left = left + 'px';
    tooltip.style.top  = top  + 'px';
    return;
  }

  // ── Entropy bar tooltip (merged segments) ────────────────────────────
  if (el.classList.contains('ent-seg')) {
    const ci = parseInt(el.dataset.ci);
    const mi = parseInt(el.dataset.mi);
    const ms = (DATA[ci].merged_segments || [])[mi];
    if (!ms) { tooltip.style.display = 'none'; return; }
    const alpha   = parseFloat(alphaSlider.value);
    const mixed   = (alpha * ms.kl_forward_mean + (1 - alpha) * ms.kl_reverse_mean).toFixed(4);
    const entMean = (ms.teacher_entropy_mean || 0).toFixed(4);
    const prStr   = ms.process_reward !== undefined
      ? (ms.process_reward >= 0 ? '+' : '') + ms.process_reward.toFixed(4) : '—';
    const epRange = ms.orig_ep_start === ms.orig_ep_end
      ? `Ep ${ms.orig_ep_start}` : `Ep ${ms.orig_ep_start}–${ms.orig_ep_end}`;
    tooltip.innerHTML = `
      <div class="tt-token" style="color:#6fcfa0">${epRange} (merged) — Teacher Entropy</div>
      <div class="tt-row"><span class="tt-label">Tokens</span><span class="tt-val">${ms.length}</span></div>
      <div class="tt-row"><span class="tt-label">Teacher H mean</span><span class="tt-val" style="color:#6fcfa0;font-size:1.05rem">${entMean}</span></div>
      <div class="tt-row"><span class="tt-label">Mixed KL (ref)</span><span class="tt-val" style="color:#888">${mixed}</span></div>
      <div class="tt-row"><span class="tt-label">Process Reward</span><span class="tt-val" style="color:${ms.process_reward >= 0 ? '#4caf50' : '#f44336'}">${prStr}</span></div>
    `;
    tooltip.style.display = 'block';
    let left = e.clientX + 16, top = e.clientY + 16;
    if (left + 300 > window.innerWidth)  left = e.clientX - 310;
    if (top  + 220 > window.innerHeight) top  = e.clientY - 220;
    tooltip.style.left = left + 'px';
    tooltip.style.top  = top  + 'px';
    return;
  }

  // ── Token tooltip ────────────────────────────────────────────────────
  if (!el.classList.contains('tok')) {
    tooltip.style.display = 'none';
    return;
  }
  const ci = parseInt(el.dataset.ci);
  const ti = parseInt(el.dataset.ti);
  const tok = DATA[ci].tokens[ti];
  const segIdx = tok.merged_segment !== undefined ? tok.merged_segment
               : (tok.segment !== undefined ? tok.segment : '—');

  tooltip.innerHTML = `
    <div class="tt-token">"${htmlSafe(tok.text)}"</div>
    <div class="tt-row"><span class="tt-label">Position</span><span class="tt-val">${ti}</span></div>
    <div class="tt-row"><span class="tt-label">Merged ep</span><span class="tt-val">${typeof segIdx === 'number' ? segIdx + 1 : segIdx}</span></div>
    <div class="tt-row"><span class="tt-label">Region</span><span class="tt-val">${tok.region || 'answer'}</span></div>
    <div class="tt-row"><span class="tt-label">Token ID</span><span class="tt-val">${tok.token_id}</span></div>
    <div class="tt-row"><span class="tt-label">KL(T‖S) fwd</span><span class="tt-val">${tok.kl_forward.toFixed(4)}</span></div>
    <div class="tt-row"><span class="tt-label">KL(S‖T) rev</span><span class="tt-val">${tok.kl_reverse.toFixed(4)}</span></div>
    <div class="tt-row"><span class="tt-label">Mixed KL</span><span class="tt-val">${tok._mixed.toFixed(4)}</span></div>
    <div class="tt-row"><span class="tt-label">Teacher H</span><span class="tt-val">${tok.teacher_entropy.toFixed(4)}</span></div>
  `;
  tooltip.style.display = 'block';

  let left = e.clientX + 16;
  let top = e.clientY + 16;
  if (left + 240 > window.innerWidth) left = e.clientX - 250;
  if (top + 200 > window.innerHeight) top = e.clientY - 200;
  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
});

alphaSlider.addEventListener('input', renderAll);
scaleSelect.addEventListener('change', renderAll);
normSelect.addEventListener('change', renderAll);
regionSelect.addEventListener('change', renderAll);

renderAll();
</script>
</body>
</html>"""


def phase6_html(case_data):
    if not case_data:
        print("  No case data — skipping HTML generation.", flush=True)
        return

    print(f"\n{'='*60}")
    print(f"Phase 6: Generating interactive HTML")
    print(f"{'='*60}", flush=True)

    json_blob = json.dumps(case_data, ensure_ascii=False)

    html = HTML_TEMPLATE
    html = html.replace('__JSON_DATA__', json_blob)
    html = html.replace('__STUDENT__', os.path.basename(STUDENT_MODEL))
    html = html.replace('__TEACHER__', os.path.basename(TEACHER_MODEL))
    html = html.replace('__DATASET__', DATASET)
    html = html.replace('__TEACHER_CONT_TOKENS__', str(TEACHER_CONT_TOKENS))

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML saved → {OUTPUT_HTML}", flush=True)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(case_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved → {OUTPUT_JSON}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"GPUs available: {N_GPUS}", flush=True)
    for i in range(N_GPUS):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [{i}] {name}  ({mem:.0f} GB)", flush=True)

    samples = load_dataset_samples()

    student_results, student_tok = phase1_student(samples)
    selected  = phase2_teacher(samples, student_results)
    case_data = phase3_kl(student_results, selected, student_tok)
    eval_items = phase4_episode_eval(case_data)
    case_data  = phase5_merge_and_continuations(case_data, eval_items)
    phase6_html(case_data)

    # ── Response length comparison ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Response length comparison (Student vs Teacher)")
    print(f"{'='*60}")
    s_lens = [len(r["response_ids"]) for r in student_results]
    t_lens = [len(student_tok.encode(r["teacher_response"])) for r in student_results
              if "teacher_response" in r]
    s_trunc = sum(1 for l in s_lens if l >= MAX_NEW_TOKENS)
    t_trunc = sum(1 for l in t_lens if l >= MAX_NEW_TOKENS)
    s_arr, t_arr = np.array(s_lens), np.array(t_lens)
    print(f"  Student: mean={s_arr.mean():.0f}  median={np.median(s_arr):.0f}  "
          f"p25={np.percentile(s_arr,25):.0f}  p75={np.percentile(s_arr,75):.0f}  "
          f"truncated={s_trunc}/{len(s_lens)}")
    print(f"  Teacher: mean={t_arr.mean():.0f}  median={np.median(t_arr):.0f}  "
          f"p25={np.percentile(t_arr,25):.0f}  p75={np.percentile(t_arr,75):.0f}  "
          f"truncated={t_trunc}/{len(t_lens)}")

    # Breakdown by correctness
    s_correct_lens = [s_lens[i] for i in range(len(student_results))
                      if student_results[i]["student_correct"]]
    s_wrong_lens = [s_lens[i] for i in range(len(student_results))
                    if not student_results[i]["student_correct"]]
    t_correct_lens = [t_lens[i] for i in range(len(student_results))
                      if student_results[i].get("teacher_correct")]
    t_wrong_lens = [t_lens[i] for i in range(len(student_results))
                    if not student_results[i].get("teacher_correct")]
    for label, lens in [("Student ✓", s_correct_lens), ("Student ✗", s_wrong_lens),
                        ("Teacher ✓", t_correct_lens), ("Teacher ✗", t_wrong_lens)]:
        if lens:
            a = np.array(lens)
            trunc = sum(1 for l in lens if l >= MAX_NEW_TOKENS)
            print(f"  {label:12s}: n={len(lens):3d}  mean={a.mean():6.0f}  "
                  f"median={np.median(a):6.0f}  truncated={trunc}")

    print(f"\nDone. Open {OUTPUT_HTML} in a browser to explore the heatmap.", flush=True)


if __name__ == "__main__":
    main()
