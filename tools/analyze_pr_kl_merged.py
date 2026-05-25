"""
analyze_pr_kl_merged.py — Process Reward × Normalised KL on MERGED paragraphs
==============================================================================

Closely follows ``analyze_kl_heatmap.py``'s segmentation + monotonic-merge
pipeline, but applied to a stratified 10%-per-subset sample of
hendrycks_math (not just 5 hand-picked cases).

For every MERGED segment ("paragraph") we report length-normalised KL
between teacher and student along with the paragraph's process reward,
and then study the relationship between them — overall and per
teacher/student correctness category.

KL is computed on the TEACHER'S TOP-K SUPPORT SET (not the full vocab).
For each prefix c_t we take S_t = TopK_q(c_t) and renormalise both
distributions to live on S_t:

    q_hat(v | c_t)  = q(v | c_t)  / Σ_{u∈S_t} q(u | c_t)
    pi_hat(v | c_t) = π(v | c_t)  / Σ_{u∈S_t} π(u | c_t)

then

    kl_fwd(c_t) = KL( q_hat  ‖ pi_hat )        on S_t   (teacher's view)
    kl_rev(c_t) = KL( pi_hat ‖ q_hat  )        on S_t   (L_top-K in the paper)

K is set via the ``KL_TOPK`` env var (default 50).  This is much cheaper
than full-vocab KL and follows the OPD top-K objective.

Both quantities are then reported AT THE MERGED-PARAGRAPH GRANULARITY
(not the raw trigger-cut sub-segment granularity):

  normKL_fwd  = (Σ_i  len_i · mean_kl_fwd_i) / Σ_i len_i
                where i indexes all raw sub-segments in the monotonic run.
  normKL_rev  = same for kl_rev.
  process_reward = AccRate(end_of_run) − AccRate(just_before_run_start).
                This is the TOTAL Δacc accumulated by the merged paragraph
                — i.e. ProcessReward summed across the run — so it matches
                the merged-paragraph KL boundaries.

This is the key correction relative to ``process_reward_kl_merged.py``,
which kept ``seg["kl_forward_mean"]`` and ``seg["process_reward"]`` from
only the LAST raw episode of each monotonic run.

Phases
------
1. Stratified ``--sample-frac`` (default 0.1) per subset
2. Student vLLM sampling + correctness check
3. Teacher vLLM greedy gen + 4-way categorisation
4. Per-token KL(T‖S), KL(S‖T) for every student response (both models loaded)
5. Trigger-based segmentation + K-fold episode evaluation → AccRate per ep
6. Monotonic-run merge with length-weighted KL and cumulative ΔAcc
7. Statistics + JSON + interactive Plotly HTML report

Usage
-----
    python tools/analyze_pr_kl_merged.py
    python tools/analyze_pr_kl_merged.py --sample-frac 0.05
    STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Qwen/Qwen3-8B \
        python tools/analyze_pr_kl_merged.py

Checkpoints and reports are tagged by teacher model (e.g. ``p10_qwen3-32b``)
so parallel 8B / 32B runs in the same ``tools/`` folder do not overwrite.
Override with ``--run-suffix mytag`` or env ``RUN_TAG=mytag``.
"""

import argparse
import bisect
import gc
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from scipy import stats as scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("WARNING: scipy not found; Pearson p-values will be NaN.", flush=True)


STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-1.7B")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-32B")
EVAL_MODEL    = os.environ.get("EVAL_MODEL",    TEACHER_MODEL)

DATASET   = "hendrycks_math"
_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
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

MAX_NEW_TOKENS = 8192
SEED           = 42
TEMPERATURE    = 1.0
TOP_K          = 50
TOP_P          = 1.0

# Teacher top-K support set for Phase 3 KL.
# All KL values in this script are computed over S_t = TopK_q(c_t) with both
# teacher and student distributions RENORMALISED inside S_t, following
# Section 3.1 of the OPD top-K paper:
#     S_t              = TopK_q(c_t)                                (teacher-defined support)
#     q_hat(v | c_t)   = q(v | c_t)      / sum_{u in S_t} q(u | c_t)
#     pi_hat(v | c_t)  = pi(v | c_t)     / sum_{u in S_t} pi(u | c_t)
#     L_top-K          = sum_{v in S_t} pi_hat(v) * log( pi_hat(v) / q_hat(v) )
# This is what "normKL_fwd / normKL_rev" mean throughout the rest of the script.
KL_TOPK = int(os.environ.get("KL_TOPK", "64"))

# ── episode evaluation ────────────────────────────────────────────────────────
K_EVAL            = 8
EVAL_TEMPERATURE  = 0.7
MAX_ANSWER_TOKENS = 384

# Phase 4 needs to fit:  chat_template + problem + partial_resp + suffix + output.
# partial_resp can be up to MAX_NEW_TOKENS tokens for the last episode, so we
# need a larger context than Phase 1/2's gen calls.
PHASE4_MAX_MODEL_LEN = MAX_NEW_TOKENS + MAX_ANSWER_TOKENS + 4096   # 12672

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


def _tag(sample_frac):
    return "full" if sample_frac >= 1.0 else f"p{int(round(sample_frac * 100))}"


def _model_slug(model_path):
    """Qwen/Qwen3-32B → qwen3-32b (safe for filenames)."""
    slug = model_path.rstrip("/").split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "model"


def _run_suffix(explicit=None):
    """
    Filename suffix separating parallel runs (8B vs 32B teacher, etc.).
    Priority: CLI --run-suffix > env RUN_TAG > TEACHER_MODEL slug.
    """
    if explicit and str(explicit).strip():
        raw = str(explicit).strip()
    else:
        raw = os.environ.get("RUN_TAG", "").strip()
    if raw:
        return re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-") or _model_slug(TEACHER_MODEL)
    suffix = _model_slug(TEACHER_MODEL)
    if EVAL_MODEL != TEACHER_MODEL:
        suffix = f"{suffix}_eval-{_model_slug(EVAL_MODEL)}"
    return suffix


def _run_tag(sample_frac, run_suffix=None):
    return f"{_tag(sample_frac)}_{_run_suffix(run_suffix)}"


def _output_paths(sample_frac, run_suffix=None):
    tag = _run_tag(sample_frac, run_suffix)
    return (
        os.path.join(_DIR, f"analyze_pr_kl_merged_{tag}.json"),
        os.path.join(_DIR, f"analyze_pr_kl_merged_{tag}.html"),
    )


_CKPT_DIR = os.path.join(_DIR, "analyze_pr_kl_merged_ckpts")


def _ckpt_paths(sample_frac, run_suffix=None):
    """Paths for per-phase resumable checkpoints."""
    tag = _run_tag(sample_frac, run_suffix)
    os.makedirs(_CKPT_DIR, exist_ok=True)
    return dict(
        # Phase 1 + 2 combined: student_results with teacher_response,
        # categories, and the student-side full_ids needed for Phase 3.
        gen=os.path.join(_CKPT_DIR, f"phase1_2_gen_{tag}.json"),
        # Phase 3 KL: case_data with per-token KL + segments. Heavy.
        kl=os.path.join(_CKPT_DIR, f"phase3_kl_{tag}.json"),
        # Phase 4: eval_items with acc_rate per (case, episode).
        eval=os.path.join(_CKPT_DIR, f"phase4_eval_{tag}.json"),
    )


# ── checkpoint (de)serialisation helpers ─────────────────────────────────────

def _student_results_to_jsonable(student_results):
    """Convert per-sample dicts (containing torch tensors) to JSON-safe form."""
    out = []
    for sr in student_results:
        sr_j = {k: v for k, v in sr.items() if k != "full_ids"}
        sr_j["full_ids"] = sr["full_ids"].tolist()
        out.append(sr_j)
    return out


def _student_results_from_jsonable(rows):
    out = []
    for sr_j in rows:
        sr = dict(sr_j)
        sr["full_ids"] = torch.tensor(sr_j["full_ids"], dtype=torch.long)
        out.append(sr)
    return out


def _save_json(path, obj):
    """Compact JSON dump (no indent — these files can be hundreds of MB)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_gen_ckpt(student_results, path):
    _save_json(path, _student_results_to_jsonable(student_results))
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"  Saved gen checkpoint → {path}  ({size_mb:.1f} MB)", flush=True)


def _load_gen_ckpt(path):
    rows = _load_json(path)
    print(f"  Loaded gen checkpoint ← {path}  ({len(rows)} samples)", flush=True)
    return _student_results_from_jsonable(rows)


def _save_kl_ckpt(case_data, path):
    _save_json(path, case_data)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"  Saved KL checkpoint  → {path}  ({size_mb:.1f} MB)", flush=True)


def _load_kl_ckpt(path):
    case_data = _load_json(path)
    print(f"  Loaded KL checkpoint  ← {path}  ({len(case_data)} cases)",
          flush=True)
    return case_data


def _save_eval_ckpt(eval_items, path):
    _save_json(path, eval_items)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"  Saved eval checkpoint → {path}  ({size_mb:.1f} MB)", flush=True)


def _load_eval_ckpt(path):
    items = _load_json(path)
    print(f"  Loaded eval checkpoint ← {path}  ({len(items)} items)", flush=True)
    return items


# ── adapter: reuse student gen from analyze_truncation_acc.py ────────────────

def load_student_from_truncation(path, student_tokenizer):
    """Adapter: load student gen results from analyze_truncation_acc.py's
    phase1 JSON (or final analysis JSON) and convert to the structure
    phase2_teacher / phase3_kl expect.

    The truncation_acc row schema is:
        problem, subset, gt_answer, student_response, student_answer,
        student_correct, response_len, truncated

    Phase 3 only needs `full_ids` and `prompt_len` from the student side,
    which we reconstruct by re-tokenising  chat_template(problem) + response.
    Resulting token IDs may differ slightly from the original vLLM sampling
    output at the byte/whitespace level, but the model forward pass and KL
    are mathematically valid regardless.
    """
    print(f"\n[reuse] Loading student gen from {path}", flush=True)
    rows = _load_json(path)
    print(f"  Loaded {len(rows)} student rows", flush=True)

    cfg = DATASET_CONFIGS[DATASET]
    if not rows:
        return []

    required = {"problem", "student_response", "gt_answer"}
    missing  = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"  Cannot reuse {path}: missing fields {missing}")

    results = []
    skipped = 0
    for r in rows:
        problem  = r["problem"]
        resp     = r.get("student_response") or ""
        if not resp:
            skipped += 1
            continue

        prompt_str = build_prompt(problem, student_tokenizer)
        prompt_ids = student_tokenizer.encode(prompt_str, add_special_tokens=False)
        resp_ids   = student_tokenizer.encode(resp,       add_special_tokens=False)
        full_ids   = torch.tensor(prompt_ids + resp_ids, dtype=torch.long)

        results.append(dict(
            problem=problem,
            subset=r.get("subset"),
            gt_solution=r.get("gt_solution"),  # may be absent — that's OK
            gt_answer=r["gt_answer"],
            student_response=resp,
            student_answer=r.get("student_answer"),
            student_correct=bool(r.get("student_correct", False)),
            full_ids=full_ids,
            prompt_len=len(prompt_ids),
        ))

    print(f"  Reconstructed {len(results)} samples "
          f"(skipped {skipped} empty)", flush=True)
    return results


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


def load_stratified_samples(sample_frac, seed=SEED):
    """
    Stratified per-subset sampling: take ``sample_frac`` of each subset
    independently. Mirrors ``analyze_truncation_acc.py``'s --sample-frac.
    """
    cfg     = DATASET_CONFIGS[DATASET]
    subsets = cfg["subsets"]
    rng     = random.Random(seed)

    label = "full" if sample_frac >= 1.0 else f"{sample_frac:.0%} stratified"
    print(f"Loading dataset '{DATASET}' ({len(subsets)} subsets, {label}) …",
          flush=True)
    samples = []
    for subset in subsets:
        path = os.path.join(_DATA_ROOT, DATASET, subset)
        ds   = load_from_disk(path)
        n    = len(ds)
        if sample_frac >= 1.0:
            indices = list(range(n))
        else:
            k = max(1, round(n * sample_frac))
            indices = sorted(rng.sample(range(n), k))
        for i in indices:
            row = dict(ds[i])
            row["_subset"] = subset
            samples.append(row)
        print(f"  {subset:30s}: {len(indices)}/{n}", flush=True)
    print(f"  Total: {len(samples)} samples", flush=True)
    return samples


# ── vLLM generation helper ────────────────────────────────────────────────────

def _vllm_generate(model_path, prompt_strings, do_sample=True,
                   max_tokens=None, temperature=None, stop=None,
                   n_completions=1):
    tp = N_GPUS
    print(f"  Initializing vLLM ({os.path.basename(model_path)}, tp={tp}) …",
          flush=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 1024,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
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
    suffix = f"  ×  n={n_completions}" if n_completions > 1 else ""
    print(f"  Generating {len(prompt_strings)} prompts{suffix} …", flush=True)
    outputs = llm.generate(prompt_strings, params)

    del llm
    _release_cuda()
    return outputs


# ── model loading helpers ─────────────────────────────────────────────────────

def _release_cuda():
    """Best-effort free of cached GPU memory between vLLM and HF phases."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _gpu_max_memory(gpu_ids, mem_frac=0.82):
    """Per-GPU cap for accelerate (must match real VRAM, not a generous default)."""
    max_mem = {}
    for i in gpu_ids:
        gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        max_mem[i] = f"{max(1, int(gib * mem_frac))}GiB"
    max_mem["cpu"] = "32GiB"
    return max_mem


def _make_device_map(model_name, gpu_ids):
    if len(gpu_ids) == 1:
        return f"cuda:{gpu_ids[0]}"
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    max_mem = _gpu_max_memory(gpu_ids)
    device_map = infer_auto_device_map(empty_model, max_memory=max_mem)
    del empty_model
    return device_map


def _load_causal_lm(model_name, gpu_ids):
    device_map = _make_device_map(model_name, gpu_ids)
    if len(gpu_ids) > 1:
        counts = defaultdict(int)
        for dev in device_map.values():
            counts[str(dev)] += 1
        print(f"    device_map for {os.path.basename(model_name)}: "
              f"{dict(sorted(counts.items()))}", flush=True)
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()


def _forward_logits(model, input_ids, device):
    """Forward-pass; return logits on CPU. Falls back to chunking on OOM."""
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
          f"n={len(samples)})")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    cfg = DATASET_CONFIGS[DATASET]
    prompt_strings = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]
    gt_answers     = [extract_boxed_answer(s[cfg["solution_key"]]) for s in samples]

    gen_outputs = _vllm_generate(STUDENT_MODEL, prompt_strings, do_sample=True)

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
            subset=sample.get("_subset"),
            gt_solution=sample[cfg["solution_key"]],
            gt_answer=gt_answer,
            student_response=response_text,
            student_answer=student_answer,
            student_correct=correct,
            full_ids=full_ids,
            prompt_len=prompt_len,
        ))

        if (si + 1) % 100 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:4d}/{len(samples)}] {tag}  gt={gt_answer}  "
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

    gen_outputs = _vllm_generate(TEACHER_MODEL, prompt_strings, do_sample=False)

    for si, (sr, out) in enumerate(zip(student_results, gen_outputs)):
        gen_ids        = list(out.outputs[0].token_ids)
        response_text  = tokenizer.decode(gen_ids, skip_special_tokens=True)
        teacher_answer = extract_boxed_answer(response_text)
        correct        = answers_match(teacher_answer, sr["gt_answer"])

        sr["teacher_response"] = response_text
        sr["teacher_answer"]   = teacher_answer
        sr["teacher_correct"]  = correct
        sr["category"]         = get_category(sr["student_correct"], correct)

        if (si + 1) % 100 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:4d}/{len(student_results)}] {tag}  "
                  f"gt={sr['gt_answer']}  pred={teacher_answer}  "
                  f"→ {sr['category']}", flush=True)

    n_ok = sum(r["teacher_correct"] for r in student_results)
    print(f"  Teacher accuracy: {n_ok}/{len(student_results)} "
          f"({n_ok/len(student_results):.1%})", flush=True)

    cat_counts = defaultdict(int)
    for r in student_results:
        cat_counts[r["category"]] += 1
    for cat in CATEGORY_KEYS:
        print(f"  {CATEGORY_LABELS[cat]:32s}: {cat_counts[cat]:4d}", flush=True)

    return student_results


# ── Phase 3: per-token KL for ALL samples (interleaved, no logit storage) ────

def phase3_kl(student_results, student_tokenizer):
    print(f"\n{'='*60}")
    print(f"Phase 3: Per-token KL + episode segmentation  "
          f"(all {len(student_results)} samples)")
    print(f"  KL on teacher Top-{KL_TOPK} support, renormalised in S_t",
          flush=True)
    print(f"{'='*60}", flush=True)

    _release_cuda()

    if N_GPUS >= 2:
        s_gpus, t_gpus = [0], list(range(1, N_GPUS))
        print(f"  Student → GPU {s_gpus}  |  Teacher → GPU {t_gpus}  (simultaneous)",
              flush=True)
        # Load the large teacher first so layer placement respects real VRAM
        # on GPUs 1..N-1 before the student occupies GPU 0.
        print("  Loading teacher model …", flush=True)
        t_model = _load_causal_lm(TEACHER_MODEL, t_gpus)
        print("  Loading student model …", flush=True)
        s_model = _load_causal_lm(STUDENT_MODEL, s_gpus)
    else:
        s_gpus = t_gpus = [0]
        print("  Single GPU: load teacher, then student (sequential share GPU 0)",
              flush=True)
        print("  Loading teacher model …", flush=True)
        t_model = _load_causal_lm(TEACHER_MODEL, t_gpus)
        print("  Loading student model …", flush=True)
        s_model = _load_causal_lm(STUDENT_MODEL, s_gpus)

    s_dev = f"cuda:{s_gpus[0]}"
    t_dev = f"cuda:{t_gpus[0]}"

    case_data = []
    skipped = 0
    n_total = len(student_results)

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

        # Slice the prediction window: for response token at position p,
        # the predicting logits are at position p-1.
        s_resp = s_logits[prompt_len - 1: seq_len - 1].float()  # [resp_len, V]
        t_resp = t_logits[prompt_len - 1: seq_len - 1].float()  # [resp_len, V]
        del s_logits, t_logits

        # ── Teacher top-K support set + renormalisation ───────────────────
        # S_t = TopK_q(c_t).  Gather student/teacher logits ONLY at the K
        # teacher-supported indices, then log_softmax over the K-dim — this
        # is equivalent to normalising the original probabilities by the
        # mass mass_S = sum_{u in S_t} p(u) (cf. eqs (3.2)/(3.3) of the paper).
        t_topk = t_resp.topk(KL_TOPK, dim=-1)
        topk_idx = t_topk.indices                          # [resp_len, K]
        t_topk_logits = t_topk.values                      # [resp_len, K]
        s_topk_logits = s_resp.gather(-1, topk_idx)        # [resp_len, K]

        t_log_p = F.log_softmax(t_topk_logits, dim=-1)
        s_log_p = F.log_softmax(s_topk_logits, dim=-1)
        t_p = t_log_p.exp()
        s_p = s_log_p.exp()

        # Forward  KL( q_hat || pi_hat )  on teacher's top-K support
        kl_fwd = (t_p * (t_log_p - s_log_p)).sum(-1).tolist()
        # Reverse  KL( pi_hat || q_hat )  on teacher's top-K support
        # — this is L_top-K from the paper.
        kl_rev = (s_p * (s_log_p - t_log_p)).sum(-1).tolist()
        del s_resp, t_resp, t_topk, topk_idx, t_topk_logits, s_topk_logits
        del s_log_p, t_log_p, s_p, t_p

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
            subset=sr.get("subset"),
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

        if (si + 1) % 50 == 0 or si == 0:
            n_segs = len(segments)
            fwd_mu = sum(kl_fwd) / len(kl_fwd) if kl_fwd else 0.0
            rev_mu = sum(kl_rev) / len(kl_rev) if kl_rev else 0.0
            print(f"  [{si+1:4d}/{n_total}] cat={sr['category']:14s}  "
                  f"resp={resp_len:4d}  segs={n_segs:2d}  "
                  f"KL(T‖S)={fwd_mu:.4f}  KL(S‖T)={rev_mu:.4f}", flush=True)

    del s_model, t_model
    _release_cuda()
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

    # Build all eval items (every (case, episode) pair, from episode 1)
    eval_items = []
    for ci, case in enumerate(case_data):
        for ep_idx, seg in enumerate(case.get("segments", [])):
            partial_resp = "".join(
                tok["text"] for tok in case["tokens"][: seg["end"]]
            )
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

    # Defensive filter: drop (case, episode) pairs whose tokenised prompt
    # would not leave room for MAX_ANSWER_TOKENS of output within
    # PHASE4_MAX_MODEL_LEN.  These rare long-tail prompts otherwise crash
    # vLLM with a max-context-length error.
    prompt_budget = PHASE4_MAX_MODEL_LEN - MAX_ANSWER_TOKENS
    kept_items, kept_prefixes, prompt_lens = [], [], []
    skipped = []
    for item, prefix in zip(eval_items, prefixes):
        tok_len = len(eval_tokenizer.encode(prefix, add_special_tokens=False))
        if tok_len > prompt_budget:
            skipped.append((item["case_idx"], item["ep_num"], tok_len))
            continue
        kept_items.append(item)
        kept_prefixes.append(prefix)
        prompt_lens.append(tok_len)
    if skipped:
        print(f"  Skipped {len(skipped)} / {len(eval_items)} eval pairs "
              f"(prompt > {prompt_budget} tok)", flush=True)
        for ci, ep, n in skipped[:5]:
            print(f"    case_idx={ci}  ep_num={ep}  prompt_len={n}", flush=True)
        if len(skipped) > 5:
            print(f"    … {len(skipped) - 5} more", flush=True)

    if not kept_items:
        print("  No eval items left after filtering.", flush=True)
        return []

    # Sort short→long for prefix-cache reuse (within-case order is preserved).
    order = sorted(range(len(kept_items)), key=lambda i: prompt_lens[i])
    eval_items_sorted = [kept_items[i]    for i in order]
    prefixes_sorted   = [kept_prefixes[i] for i in order]

    tp = min(N_GPUS, 4)
    llm = LLM(
        model=EVAL_MODEL,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=PHASE4_MAX_MODEL_LEN,
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
    print("  Running vLLM inference (sorted short→long for prefix cache) …",
          flush=True)
    outputs = llm.generate(prefixes_sorted, params)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    for item, out in zip(eval_items_sorted, outputs):
        gen_texts = [o.text for o in out.outputs]
        preds     = [extract_boxed_answer(t) for t in gen_texts]
        corrects  = [answers_match(p, item["gt_answer"]) for p in preds]
        item["acc_rate"]      = round(sum(corrects) / K_EVAL, 6)
        item["correct_count"] = int(sum(corrects))

    print(f"  Parsed {len(eval_items_sorted)} items.", flush=True)
    return eval_items_sorted


# ── Phase 5: monotonic merge with proper paragraph-level KL + cumulative PR ──

def merge_monotonic_runs_indexed(seg_prs):
    """
    Given a list of process_reward values along consecutive episodes
    (already sorted by ep_num), return the kept indices marking the END of
    each monotonic run.

    A run is cut ONLY at a STRICT sign flip (+1 → -1 or -1 → +1).  Flat
    gaps (sign == 0) extend whichever run is currently active.
    """
    n = len(seg_prs)
    if n <= 1:
        return list(range(n))

    def _sign(diff):
        if diff > 0: return 1
        if diff < 0: return -1
        return 0

    signs = [_sign(seg_prs[i + 1] - seg_prs[i]) for i in range(n - 1)]
    kept = []
    active_dir = 0
    for i in range(n - 1):
        s = signs[i]
        if s != 0:
            if active_dir != 0 and s != active_dir:
                kept.append(i)
            active_dir = s
        if i == n - 2:
            kept.append(n - 1)
    if not kept:
        kept = [n - 1]
    return kept


def phase5_merge(case_data, eval_items):
    """
    For each case:
      1. Attach per-episode acc_rate to its segment.
      2. Compute per-segment process_reward = acc_rate(i) - acc_rate(i-1)
         (with acc_rate(0) ≡ 0).
      3. Run monotonic-run merging on the PR sequence.
      4. For each merged paragraph, compute:
           - normKL_fwd / normKL_rev = LENGTH-WEIGHTED mean across all
                                       raw sub-segments in the run.
           - merged_process_reward   = ΔAcc across the whole run
                                       = acc_rate(run_end) − acc_rate(run_start − 1).
           - tokens_length, ep_start, ep_end, etc.
    Returns the merged paragraph list, and also attaches it onto each case.
    """
    print(f"\n{'='*60}")
    print("Phase 5: Monotonic merge (length-weighted KL + cumulative ΔAcc)")
    print(f"{'='*60}", flush=True)

    ep_map = defaultdict(dict)
    for item in eval_items:
        ep_map[item["case_idx"]][item["ep_idx"]] = item

    total_raw = total_merged = 0

    for ci, case in enumerate(case_data):
        segs = case["segments"]
        if not segs:
            case["merged_segments"] = []
            continue

        # ── Step 1+2: attach acc_rate and per-episode process_reward ──────
        prev_acc = 0.0
        for ep_idx, seg in enumerate(segs):
            item = ep_map[ci].get(ep_idx, {})
            acc  = item.get("acc_rate", prev_acc)  # fall back: no change
            seg["acc_rate"]       = acc
            seg["process_reward"] = round(acc - prev_acc, 6)
            prev_acc = acc

        total_raw += len(segs)

        # ── Step 3: monotonic-run merge ──────────────────────────────────
        prs = [s["process_reward"] for s in segs]
        kept_indices = merge_monotonic_runs_indexed(prs)
        run_starts   = [0] + [kept_indices[j] + 1 for j in range(len(kept_indices) - 1)]
        total_merged += len(kept_indices)

        # ── Step 4: build merged paragraphs ──────────────────────────────
        merged = []
        for mi, (run_start, run_end) in enumerate(zip(run_starts, kept_indices)):
            run_segs   = segs[run_start: run_end + 1]
            total_len  = sum(s["length"] for s in run_segs) or 1
            kl_fwd_avg = sum(s["kl_forward_mean"] * s["length"] for s in run_segs) / total_len
            kl_rev_avg = sum(s["kl_reverse_mean"] * s["length"] for s in run_segs) / total_len

            # Cumulative ΔAcc across the run (== sum of per-ep PRs)
            acc_end   = segs[run_end]["acc_rate"]
            acc_prior = segs[run_start - 1]["acc_rate"] if run_start > 0 else 0.0
            merged_pr = round(acc_end - acc_prior, 6)

            merged.append(dict(
                merged_idx=mi,
                ep_start=run_start + 1,        # 1-indexed inclusive
                ep_end=run_end + 1,
                n_subsegments=len(run_segs),
                tok_start=run_segs[0]["start"],
                tok_end=run_segs[-1]["end"],
                length=int(total_len),
                normKL_fwd=round(max(0.0, kl_fwd_avg), 6),
                normKL_rev=round(max(0.0, kl_rev_avg), 6),
                process_reward=merged_pr,
                acc_rate_end=round(acc_end, 6),
                acc_rate_prior=round(acc_prior, 6),
            ))

        case["merged_segments"] = merged

        if (ci + 1) % 50 == 0 or ci == 0:
            print(f"  Case {ci+1}: {len(segs):2d} raw → {len(merged):2d} merged  "
                  f"(category={case['category']})", flush=True)

    if total_raw:
        print(f"  Monotonic merge: {total_raw} raw → {total_merged} merged "
              f"(reduction {(1 - total_merged/total_raw)*100:.1f}%)", flush=True)
    return case_data


# ── Phase 6: statistical analysis of merged-paragraph PR vs KL ────────────────

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


def _spearmanr(x, y):
    if _SCIPY_OK and len(x) > 2:
        r, p = scipy_stats.spearmanr(x, y)
        return float(r), float(p)
    return float("nan"), float("nan")


def _r2(y_true, y_pred):
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def _category_stats(points):
    n = len(points)
    if n < 3:
        return {}
    y = np.array([p["process_reward"] for p in points])
    out = {}
    for kl_dir in ("fwd", "rev"):
        x = np.array([p[f"normKL_{kl_dir}"] for p in points])
        r,  p_val = _pearsonr(x, y)
        rho, sp_p = _spearmanr(x, y)
        lin  = np.polyfit(x, y, 1)
        quad = np.polyfit(x, y, 2)
        r2_lin  = _r2(y, np.polyval(lin, x))
        r2_quad = _r2(y, np.polyval(quad, x))
        out[kl_dir] = dict(
            pearson_r=round(r, 4),
            p_value=(round(p_val, 6) if not np.isnan(p_val) else None),
            spearman_r=round(rho, 4) if not np.isnan(rho) else None,
            spearman_p=round(sp_p, 6) if not np.isnan(sp_p) else None,
            linear_fit=[round(float(c), 6) for c in lin],
            quad_fit=[round(float(c), 6) for c in quad],
            r2_linear=round(r2_lin, 4),
            r2_quad=round(r2_quad, 4),
            x_min=round(float(x.min()), 4),
            x_max=round(float(x.max()), 4),
        )
    return out


def phase6_analyze(case_data):
    print(f"\n{'='*60}")
    print("Phase 6: Statistical analysis (merged-paragraph PR vs normKL)")
    print(f"{'='*60}", flush=True)

    # ── Flatten merged paragraphs into points ─────────────────────────────
    all_points = []
    cat_points = defaultdict(list)
    for ci, case in enumerate(case_data):
        cat = case["category"]
        for mseg in case.get("merged_segments", []):
            point = dict(
                sample_idx=ci,
                category=cat,
                subset=case.get("subset"),
                ep_start=mseg["ep_start"],
                ep_end=mseg["ep_end"],
                n_subsegments=mseg["n_subsegments"],
                length=mseg["length"],
                normKL_fwd=mseg["normKL_fwd"],
                normKL_rev=mseg["normKL_rev"],
                process_reward=mseg["process_reward"],
                acc_rate_end=mseg["acc_rate_end"],
            )
            all_points.append(point)
            cat_points[cat].append(point)

    print(f"  Total merged paragraphs: {len(all_points)}", flush=True)

    # ── Per-category + overall stats ──────────────────────────────────────
    results = dict(categories={}, overall={})

    for cat in CATEGORY_KEYS:
        points    = cat_points.get(cat, [])
        n_samples = sum(1 for c in case_data if c["category"] == cat)
        n         = len(points)
        cat_result = dict(n=n, n_samples=n_samples, points=points,
                          stats=_category_stats(points))
        if points:
            y = np.array([p["process_reward"] for p in points])
            cat_result["mean_process_reward"] = round(float(y.mean()), 4)
            cat_result["std_process_reward"]  = round(float(y.std()),  4)
        results["categories"][cat] = cat_result

        for kl_dir, label in [("fwd", "KL(T‖S)"), ("rev", "KL(S‖T)")]:
            st = cat_result["stats"].get(kl_dir)
            if st:
                print(f"  {CATEGORY_LABELS[cat]:32s}  n={n:4d}  [{label}]  "
                      f"r={st['pearson_r']:+.3f}  p={st['p_value']}  "
                      f"R²_lin={st['r2_linear']:.3f}  R²_quad={st['r2_quad']:.3f}",
                      flush=True)
            else:
                print(f"  {CATEGORY_LABELS[cat]:32s}  n={n:4d}  [{label}]  "
                      f"too few points", flush=True)

    overall = dict(
        n=len(all_points),
        points=all_points,
        stats=_category_stats(all_points),
    )
    if all_points:
        y = np.array([p["process_reward"] for p in all_points])
        overall["mean_process_reward"] = round(float(y.mean()), 4)
        overall["std_process_reward"]  = round(float(y.std()),  4)
    results["overall"] = overall

    print("  ── Overall (all categories combined) ──")
    for kl_dir, label in [("fwd", "KL(T‖S)"), ("rev", "KL(S‖T)")]:
        st = overall["stats"].get(kl_dir)
        if st:
            print(f"  {'OVERALL':32s}  n={overall['n']:4d}  [{label}]  "
                  f"r={st['pearson_r']:+.3f}  p={st['p_value']}  "
                  f"ρ={st['spearman_r']}  R²_lin={st['r2_linear']:.3f}  "
                  f"R²_quad={st['r2_quad']:.3f}", flush=True)

    return results


# ── Phase 7: JSON + HTML report ──────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Process Reward × Merged-Paragraph Normalised KL</title>
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
.plot-cell.wide{grid-column:1 / -1;min-height:480px}
.stats-wrap{overflow-x:auto;margin-bottom:8px}
.stats-table{width:100%;border-collapse:collapse;font-size:.82rem}
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
      (merged paragraphs — length-weighted KL · cumulative ΔAcc)
    </span>
  </h1>
  <div class="subtitle">
    Each data point = ONE merged monotonic-run paragraph from a student
    response.  normKL is the length-weighted mean of per-token KL across
    every raw sub-segment in the run.  process_reward is the cumulative
    Δacc_rate across the whole merged paragraph.
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
      KL computed on teacher Top-K support (renormalised in S_t).
      NormKL = length-weighted mean of per-token KL over the merged paragraph.
    </span>
  </div>

  <div class="cat-bar" id="cat-bar"></div>

  <h2>Scatter: NormKL vs Process Reward — all merged paragraphs (overall)</h2>
  <div class="plots-grid">
    <div class="plot-cell wide" id="plt-overall"></div>
  </div>

  <h2>Scatter: NormKL vs Process Reward — by category</h2>
  <div class="plots-grid">
    <div class="plot-cell" id="plt-both_correct"></div>
    <div class="plot-cell" id="plt-both_wrong"></div>
    <div class="plot-cell" id="plt-student_only"></div>
    <div class="plot-cell" id="plt-teacher_only"></div>
  </div>

  <h2>Correlation summary</h2>
  <div class="stats-wrap"><div id="stats-tbl"></div></div>
  <div class="note">
    Solid white = linear fit &nbsp;|&nbsp; Dashed yellow = quadratic fit &nbsp;|&nbsp;
    Point colour = end episode of merged paragraph
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
  ['Sample frac',  cfg.sample_frac],
  ['N samples',    cfg.n_samples],
  ['K eval',       cfg.k_eval],
  ['Eval T',       cfg.eval_temperature],
  ['KL support',   'teacher Top-' + (cfg.kl_topk || '?')],
  ['Merge',        'monotonic + length-weighted'],
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
    `${d.n} merged pts</div>`;
});

// ── toggle ───────────────────────────────────────────────────────────────────
function setDir(dir) {
  klDir = dir;
  document.getElementById('btn-fwd').className =
    'kl-btn' + (dir === 'fwd' ? ' active' : '');
  document.getElementById('btn-rev').className =
    'kl-btn' + (dir === 'rev' ? ' active' : '');
  renderAll();
}

function linspace(a, b, n) {
  return Array.from({length: n}, (_, i) => a + (b - a) * i / (n - 1));
}
function polyval(coeffs, xs) {
  return xs.map(x => coeffs.reduce((acc, c) => acc * x + c, 0));
}

function renderScatter(divId, title, color, n, pts, st) {
  const x   = pts.map(p => Math.max(0, p[`normKL_${klDir}`]));
  const y   = pts.map(p => p.process_reward);
  const eps = pts.map(p => p.ep_end);

  const traces = [];
  traces.push({
    type: 'scatter', mode: 'markers',
    x, y,
    marker: {
      color: eps,
      colorscale: 'Viridis',
      size: 5, opacity: 0.55,
      colorbar: {
        title: {text: 'End ep', font: {size: 10, color: '#999'}},
        thickness: 10, len: 0.55,
        tickfont: {color: '#888', size: 9},
      },
    },
    text: pts.map(p =>
      `Case ${p.sample_idx+1}, Ep ${p.ep_start}` +
      (p.ep_start !== p.ep_end ? `\u2013${p.ep_end}` : '') +
      `<br>${p.n_subsegments} sub-seg, ${p.length} tok<br>`+
      `acc_end=${p.acc_rate_end.toFixed(2)}`),
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
    ? `r=${st.pearson_r >= 0 ? '+' : ''}${st.pearson_r.toFixed(3)}, ` +
      `p=${st.p_value !== null ? st.p_value.toFixed(4) : 'n/a'}, n=${n}`
    : `n=${n}`;
  const klLabel = klDir === 'fwd'
    ? 'NormKL_fwd = mean KL(q\u0302 \u2016 \u03C0\u0302) on teacher Top-K  (paragraph)'
    : 'NormKL_rev = mean KL(\u03C0\u0302 \u2016 q\u0302) on teacher Top-K  (paragraph)';

  Plotly.react(divId, traces, {
    title: {
      text: `${title}<br><sup style="font-size:9px">${rTag}</sup>`,
      font: {color: color, size: 13},
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
}

function renderAll() {
  // Overall plot
  const ov = DATA.overall;
  renderScatter('plt-overall', 'Overall (all categories)',
                '#7ecfff', ov.n, ov.points, ov.stats && ov.stats[klDir]);

  // Per-category plots
  CATS.forEach(cat => {
    const d = DATA.categories[cat];
    renderScatter(`plt-${cat}`, LABELS[cat], COLORS[cat],
                  d.n, d.points || [], d.stats && d.stats[klDir]);
  });

  renderStats();
}

function renderStats() {
  const dirLabel = klDir === 'fwd' ? 'KL(T\u2016S)' : 'KL(S\u2016T)';
  let html = `<table class="stats-table"><thead><tr>
    <th>Category</th><th>Samples</th><th>Merged pts</th>
    <th>Pearson r<br><small>${dirLabel}</small></th>
    <th>p-value</th>
    <th>Spearman \u03C1</th>
    <th>R\u00B2 linear</th><th>R\u00B2 quadratic</th>
    <th>Mean PR</th><th>Std PR</th>
  </tr></thead><tbody>`;

  // Overall row
  const ov = DATA.overall;
  const stOv = ov.stats && ov.stats[klDir];
  const rOv  = stOv ? stOv.pearson_r : null;
  html += `<tr style="background:#1a1a30">
    <td style="color:#7ecfff;font-weight:700">OVERALL</td>
    <td>\u2014</td><td>${ov.n}</td>
    <td class="${rOv !== null ? (rOv >= 0 ? 'pos' : 'neg') : ''}">
      ${rOv !== null ? (rOv >= 0 ? '+' : '') + rOv.toFixed(4) : '\u2014'}</td>
    <td>${stOv && stOv.p_value !== null ? stOv.p_value.toFixed(5) : '\u2014'}</td>
    <td>${stOv && stOv.spearman_r !== null && stOv.spearman_r !== undefined
        ? (stOv.spearman_r >= 0 ? '+' : '') + stOv.spearman_r.toFixed(4) : '\u2014'}</td>
    <td>${stOv ? stOv.r2_linear.toFixed(4) : '\u2014'}</td>
    <td>${stOv ? stOv.r2_quad.toFixed(4)   : '\u2014'}</td>
    <td>${ov.mean_process_reward !== undefined ? ov.mean_process_reward.toFixed(4) : '\u2014'}</td>
    <td>${ov.std_process_reward !== undefined ? ov.std_process_reward.toFixed(4) : '\u2014'}</td>
  </tr>`;

  CATS.forEach(cat => {
    const d  = DATA.categories[cat];
    const st = d.stats && d.stats[klDir];
    const r  = st ? st.pearson_r : null;
    const rCls = r !== null ? (r >= 0 ? 'pos' : 'neg') : '';
    html += `<tr>
      <td style="color:${COLORS[cat]};font-weight:600">${LABELS[cat]}</td>
      <td>${d.n_samples}</td>
      <td>${d.n}</td>
      <td class="${rCls}">${r !== null
        ? (r >= 0 ? '+' : '') + r.toFixed(4) : '\u2014'}</td>
      <td>${st && st.p_value !== null
        ? st.p_value.toFixed(5) : '\u2014'}</td>
      <td>${st && st.spearman_r !== null && st.spearman_r !== undefined
          ? (st.spearman_r >= 0 ? '+' : '') + st.spearman_r.toFixed(4) : '\u2014'}</td>
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

renderAll();
</script>
</body>
</html>
"""


def phase7_report(analysis_results, n_samples, sample_frac,
                  output_json, output_html):
    print(f"\n{'='*60}")
    print("Phase 7: Generating JSON + HTML report")
    print(f"{'='*60}", flush=True)

    report = dict(
        config=dict(
            student_model=os.path.basename(STUDENT_MODEL),
            teacher_model=os.path.basename(TEACHER_MODEL),
            eval_model=os.path.basename(EVAL_MODEL),
            sample_frac=f"{sample_frac:.0%}",
            n_samples=n_samples,
            k_eval=K_EVAL,
            eval_temperature=EVAL_TEMPERATURE,
            kl_topk=KL_TOPK,
            kl_support="teacher TopK, renormalised in S_t",
            monotonic_merge="length-weighted KL + cumulative ΔAcc",
        ),
        overall=analysis_results["overall"],
        categories=analysis_results["categories"],
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved → {output_json}", flush=True)

    html = HTML_TEMPLATE.replace(
        "__JSON_DATA__", json.dumps(report, ensure_ascii=False)
    )
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved → {output_html}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process Reward × Normalised KL on merged paragraphs"
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.1,
        help="Per-subset sample fraction (default: 0.1; use 1.0 for full)",
    )
    parser.add_argument(
        "--run-suffix", default=None,
        help="Suffix in ckpt/output filenames (default: teacher model slug, "
             "e.g. qwen3-32b). Use to keep 8B and 32B runs separate in one folder.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed phases using checkpoints tagged by --sample-frac "
             "and teacher (--run-suffix / TEACHER_MODEL).",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint saving (useful for ad-hoc runs).",
    )
    parser.add_argument(
        "--reuse-student-from", default=None,
        help="Reuse student generations from an analyze_truncation_acc.py "
             "phase1 JSON (e.g. tools/truncation_acc_phase1_p10.json) instead "
             "of running Phase 1 from scratch. Phase 2 (teacher), Phase 3 (KL), "
             "Phase 4 (eval) still run as usual.",
    )
    args = parser.parse_args()

    if not (0 < args.sample_frac <= 1.0):
        parser.error("--sample-frac must be in (0, 1]")

    run_suffix = _run_suffix(args.run_suffix)
    run_tag    = _run_tag(args.sample_frac, args.run_suffix)
    output_json, output_html = _output_paths(args.sample_frac, args.run_suffix)
    ckpt = _ckpt_paths(args.sample_frac, args.run_suffix)

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
    print(f"Sample frac   : {args.sample_frac:.0%} per subset")
    print(f"Run tag       : {run_tag}  (ckpt/output suffix; from teacher model)")
    print(f"K_EVAL={K_EVAL}  EVAL_TEMPERATURE={EVAL_TEMPERATURE}  "
          f"KL_TOPK={KL_TOPK}  (teacher top-K support, renormalised)  "
          f"merge=monotonic + length-weighted",
          flush=True)
    print(f"Output JSON   : {output_json}")
    print(f"Output HTML   : {output_html}", flush=True)
    if not args.no_checkpoint:
        print(f"Checkpoint dir: {_CKPT_DIR}", flush=True)
        print(f"  gen → {ckpt['gen']}", flush=True)
        print(f"  kl  → {ckpt['kl']}", flush=True)
        print(f"  eval→ {ckpt['eval']}", flush=True)
    print(f"Resume mode   : {args.resume}", flush=True)

    save_ckpt = not args.no_checkpoint

    have_eval = args.resume and os.path.isfile(ckpt["eval"])
    have_kl   = args.resume and os.path.isfile(ckpt["kl"])
    have_gen  = args.resume and os.path.isfile(ckpt["gen"])

    samples = None
    student_results = None
    case_data = None
    eval_items = None

    # ── Phase 1 + 2 ──────────────────────────────────────────────────────────
    if have_gen or have_kl or have_eval:
        print(f"\n[resume] Skipping Phase 1+2 — checkpoint present.", flush=True)
    elif args.reuse_student_from:
        # Reuse student gen from analyze_truncation_acc.py; only run teacher gen.
        student_tokenizer = AutoTokenizer.from_pretrained(
            STUDENT_MODEL, trust_remote_code=True
        )
        student_results = load_student_from_truncation(
            args.reuse_student_from, student_tokenizer
        )
        # phase2_teacher takes the original "samples" list to rebuild prompts;
        # reconstruct a minimal samples list from the reused student_results.
        cfg = DATASET_CONFIGS[DATASET]
        samples = [
            {
                cfg["prompt_key"]:   sr["problem"],
                cfg["solution_key"]: sr.get("gt_solution") or "",
                "_subset":           sr.get("subset"),
            }
            for sr in student_results
        ]
        student_results = phase2_teacher(samples, student_results)
        if save_ckpt:
            _save_gen_ckpt(student_results, ckpt["gen"])
    else:
        samples = load_stratified_samples(args.sample_frac)
        student_results, _student_tok = phase1_student(samples)
        student_results = phase2_teacher(samples, student_results)
        if save_ckpt:
            _save_gen_ckpt(student_results, ckpt["gen"])

    # ── Phase 3 (KL) ─────────────────────────────────────────────────────────
    if have_kl or have_eval:
        print(f"\n[resume] Skipping Phase 3 — checkpoint present.", flush=True)
        case_data = _load_kl_ckpt(ckpt["kl"])
    else:
        if student_results is None:
            # Came in via --resume with only the gen checkpoint.
            student_results = _load_gen_ckpt(ckpt["gen"])
        _release_cuda()
        student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
        case_data   = phase3_kl(student_results, student_tok)
        if save_ckpt:
            _save_kl_ckpt(case_data, ckpt["kl"])
        # Free large student_results state now that KL is done.
        del student_results
        gc.collect()

    # ── Phase 4 (episode eval) ───────────────────────────────────────────────
    if have_eval:
        print(f"\n[resume] Skipping Phase 4 — checkpoint present.", flush=True)
        eval_items = _load_eval_ckpt(ckpt["eval"])
    else:
        eval_items = phase4_episode_eval(case_data)
        if save_ckpt:
            _save_eval_ckpt(eval_items, ckpt["eval"])

    # ── Phase 5–7 (cheap; always re-run) ─────────────────────────────────────
    case_data = phase5_merge(case_data, eval_items)
    analysis  = phase6_analyze(case_data)
    n_samples = len(case_data)
    phase7_report(analysis, n_samples, args.sample_frac,
                  output_json, output_html)

    # ── quick text summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary: Pearson r (merged-paragraph process reward vs NormKL)")
    print(f"{'='*60}")
    ov = analysis["overall"]
    for kl_dir, label in [("fwd", "KL(T‖S)"), ("rev", "KL(S‖T)")]:
        st = (ov.get("stats") or {}).get(kl_dir, {})
        r  = st.get("pearson_r", float("nan"))
        rho = st.get("spearman_r")
        print(f"  {'OVERALL':32s}  [{label}]  "
              f"r={r:+.4f}  ρ={rho}  n={ov.get('n')}")
    for cat in CATEGORY_KEYS:
        res = analysis["categories"].get(cat, {})
        st_fwd = (res.get("stats") or {}).get("fwd", {})
        st_rev = (res.get("stats") or {}).get("rev", {})
        r_fwd  = st_fwd.get("pearson_r", float("nan"))
        r_rev  = st_rev.get("pearson_r", float("nan"))
        print(f"  {CATEGORY_LABELS[cat]:32s}  "
              f"KL(T‖S) r={r_fwd:+.4f}   KL(S‖T) r={r_rev:+.4f}"
              f"   [n={res.get('n', 0)}]")

    print(f"\nDone. Open {output_html} in a browser.", flush=True)


if __name__ == "__main__":
    main()
