"""
Truncation-accuracy analysis on the full dataset (student model only).

For each problem, segments the student response into episodes at trigger phrases.
At each episode boundary, the eval model is prompted with the partial response
+ EPISODE_EVAL_SUFFIX; if ANY of K completions gives the correct answer the
problem is "intermediate-correct."

Target count: problems where intermediate-correct == True but final answer == wrong.

KV-cache strategy
-----------------
Each sample's episode prefixes are nested:
  ep1_prompt ⊂ ep2_prompt ⊂ ep3_prompt  (each is a prefix of the next)
Processing short→long for the same sample means every longer episode hits the
prefix cache for all tokens already computed in the shorter episode.
We therefore sort ALL (sample, episode) prompts globally by prompt length
(ascending) before submitting to vLLM, which preserves the within-sample
short→long order and maximises cache hits.

To avoid CPU OOM we submit in chunks of PROMPT_CHUNK prompts at a time
rather than all at once.

Flow:
  Phase 1 — Load full dataset
  Phase 2 — Student on-policy generation via vLLM
  Phase 3 — Episode eval: sort short→long, chunk-submit, early-stop in post
  Phase 4 — Statistics

Usage:
    python tools/analyze_truncation_acc.py
    STUDENT_MODEL=Qwen/Qwen3-4B EVAL_MODEL=Qwen/Qwen3-8B python tools/analyze_truncation_acc.py

    # Resume after interrupt (needs phase1 checkpoint with student_response):
    python tools/analyze_truncation_acc.py --resume

    # Stratified 10% per subset (faster smoke run):
    python tools/analyze_truncation_acc.py --sample-frac 0.1

    # Print statistics from an existing JSON (old or new format):
    python tools/analyze_truncation_acc.py --stats-only
"""

import argparse
import gc
import json
import os
import re
import random

import numpy as np

# ── config ──────────────────────────────────────────────────────────────────
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-1.7B")
EVAL_MODEL    = os.environ.get("EVAL_MODEL",    STUDENT_MODEL)

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

MAX_NEW_TOKENS    = 8192
SEED              = 42
TEMPERATURE       = 1.0
TOP_K             = 50
TOP_P             = 1.0

K_EVAL            = 4      # completions per episode prefix
EVAL_TEMPERATURE  = 0.7
MAX_ANSWER_TOKENS = 384

# Number of prompts per vLLM call — keeps CPU RAM under control.
PROMPT_CHUNK = 512

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

EPISODE_EVAL_SYSTEM = "Put your final answer within \\boxed{}."
EPISODE_EVAL_SUFFIX = (
    "\nBased on the reasoning above, stop thinking now and summarize the existing"
    " reasoning. If a definitive answer can be derived, give the final answer in"
    " \\boxed{answer}. If no answer has been reached yet or the result is still"
    " inconclusive, output \\boxed{no answer}."
)

_TOOLS_DIR = os.path.dirname(__file__)


def _sample_suffix(sample_frac):
    if sample_frac >= 1.0:
        return ""
    return f"_p{int(round(sample_frac * 100))}"


def _output_paths(sample_frac):
    suffix = _sample_suffix(sample_frac)
    return (
        os.path.join(_TOOLS_DIR, f"truncation_acc_analysis{suffix}.json"),
        os.path.join(_TOOLS_DIR, f"truncation_acc_phase1{suffix}.json"),
        os.path.join(_TOOLS_DIR, f"truncation_acc_phase2{suffix}.json"),
    )

# ── end config ───────────────────────────────────────────────────────────────


def _n_gpus():
    import torch
    return torch.cuda.device_count()


# ── answer utilities ─────────────────────────────────────────────────────────

def extract_boxed_answer(text):
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


# ── dataset ──────────────────────────────────────────────────────────────────

def build_prompt(problem_text, tokenizer):
    msgs = [{"role": "user", "content": problem_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return problem_text


def load_full_dataset(sample_frac=1.0, seed=SEED):
    from datasets import load_from_disk

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
        print(f"  {subset}: {len(indices)}/{n} samples", flush=True)
    print(f"  Total: {len(samples)} samples", flush=True)
    return samples


# ── episode segmentation ─────────────────────────────────────────────────────

def get_intermediate_prefixes(response_text):
    """
    Return partial response strings at each episode cut, ordered short → long.
    Each is response_text[:cut_pos].  The full response itself is excluded
    (that is the "final answer" path checked separately).
    """
    combined = re.compile('|'.join(SEGMENT_TRIGGER_PATTERNS), re.IGNORECASE)

    prefixes     = []
    last_cut     = 0
    period_count = 0

    for m in combined.finditer(response_text):
        pos = m.start()
        period_count += response_text[last_cut:pos].count('.')
        if period_count >= SEGMENT_MIN_PERIODS and pos > last_cut:
            prefixes.append(response_text[:pos])
            last_cut     = pos
            period_count = 0

    # Already short→long because cut positions increase monotonically
    return prefixes


# ── eval prefix builder (mirrors process_reward_kl.py Phase 4) ───────────────

def _build_eval_prefix(problem, partial_resp, eval_tokenizer):
    resp = partial_resp
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


# ── Phase 1: student generation ──────────────────────────────────────────────

def phase1_student(samples):
    import gc
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    n_gpus = _n_gpus()
    print(f"\n{'='*60}")
    print(f"Phase 1: Student generation — {os.path.basename(STUDENT_MODEL)}"
          f"  [{len(samples)} samples, {n_gpus} GPU(s)]")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    cfg = DATASET_CONFIGS[DATASET]

    prompt_strings = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]
    gt_answers     = [extract_boxed_answer(s[cfg["solution_key"]]) for s in samples]

    llm = LLM(
        model=STUDENT_MODEL,
        tensor_parallel_size=n_gpus,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 512,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    print("  Generating …", flush=True)
    outputs = llm.generate(prompt_strings, params)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    results = []
    for si, (sample, out, gt_answer) in enumerate(zip(samples, outputs, gt_answers)):
        gen_text  = out.outputs[0].text
        gen_len   = len(out.outputs[0].token_ids)
        final_ans = extract_boxed_answer(gen_text)
        correct   = answers_match(final_ans, gt_answer)
        truncated = (gen_len >= MAX_NEW_TOKENS)

        results.append(dict(
            problem          = sample[cfg["prompt_key"]],
            subset           = sample.get("_subset"),
            gt_answer        = gt_answer,
            student_response = gen_text,
            student_answer   = final_ans,
            student_correct  = correct,
            response_len     = gen_len,
            truncated        = truncated,
        ))

        if (si + 1) % 500 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:5d}/{len(samples)}] {tag}  trunc={truncated}"
                  f"  len={gen_len}  gt={gt_answer}  pred={final_ans}", flush=True)

    n_ok    = sum(r["student_correct"] for r in results)
    n_trunc = sum(r["truncated"]       for r in results)
    print(f"  Accuracy:   {n_ok}/{len(results)} ({n_ok/len(results):.1%})", flush=True)
    print(f"  Truncated:  {n_trunc}/{len(results)} ({n_trunc/len(results):.1%})", flush=True)
    return results


# ── Phase 2: episode eval — short→long sort + chunked submit ─────────────────

def _save_phase1_checkpoint(student_results, phase1_path):
    with open(phase1_path, "w", encoding="utf-8") as f:
        json.dump(student_results, f, ensure_ascii=False)
    print(f"  Phase1 checkpoint → {phase1_path}", flush=True)


def _load_phase1_checkpoint(phase1_path):
    with open(phase1_path, encoding="utf-8") as f:
        return json.load(f)


def _save_phase2_checkpoint(correct_map, chunk_start, phase2_path):
    payload = {
        "chunk_start": chunk_start,
        "correct_map": {f"{si},{ep}": v for (si, ep), v in correct_map.items()},
    }
    with open(phase2_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"  Phase2 checkpoint @ chunk {chunk_start}", flush=True)


def _load_phase2_checkpoint(phase2_path):
    with open(phase2_path, encoding="utf-8") as f:
        payload = json.load(f)
    correct_map = {
        (int(k.split(",")[0]), int(k.split(",")[1])): v
        for k, v in payload["correct_map"].items()
    }
    return correct_map, payload["chunk_start"]


def phase2_episode_eval(student_results, resume=False, phase2_path=None):
    import gc
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    n_gpus = _n_gpus()
    print(f"\n{'='*60}")
    print(f"Phase 2: Episode eval — {os.path.basename(EVAL_MODEL)}"
          f"  K={K_EVAL}  chunk={PROMPT_CHUNK}")
    print(f"{'='*60}", flush=True)

    eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL, trust_remote_code=True)

    # Initialise per-sample fields
    for sr in student_results:
        sr["intermediate_prefixes"] = get_intermediate_prefixes(sr["student_response"])
        sr["num_intermediate"]      = len(sr["intermediate_prefixes"])
        sr["intermediate_correct"]  = False
        sr["first_correct_ep"]      = None

    n_with_eps = sum(1 for r in student_results if r["num_intermediate"] > 0)
    print(f"  Samples with ≥1 intermediate episode: {n_with_eps}", flush=True)

    # Build all (sample_idx, ep_idx, prompt_str) items
    # Within a sample ep_idx 0 < 1 < 2 ... gives strictly increasing prompt lengths,
    # so sorting globally by prompt length preserves short→long within each sample.
    all_items = []
    for si, sr in enumerate(student_results):
        for ep_idx, partial in enumerate(sr["intermediate_prefixes"]):
            prompt = _build_eval_prefix(sr["problem"], partial, eval_tokenizer)
            all_items.append((si, ep_idx, prompt))

    if not all_items:
        print("  No intermediate episodes — nothing to eval.", flush=True)
        return student_results

    # Sort globally short→long: maximises prefix-cache hits for same-sample episodes
    all_items.sort(key=lambda x: len(x[2]))

    total_prompts = len(all_items)
    print(f"  Total eval prompts: {total_prompts} × K={K_EVAL} = {total_prompts * K_EVAL} completions",
          flush=True)
    print(f"  Processing in chunks of {PROMPT_CHUNK} prompts …", flush=True)

    # Load eval model once for all chunks
    llm = LLM(
        model=EVAL_MODEL,
        tensor_parallel_size=n_gpus,
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

    # (si, ep_idx) → any-correct bool
    correct_map: dict[tuple[int, int], bool] = {}
    start_chunk = 0
    if resume and phase2_path and os.path.isfile(phase2_path):
        correct_map, start_chunk = _load_phase2_checkpoint(phase2_path)
        print(f"  Resuming phase2 from prompt offset {start_chunk}"
              f"  ({len(correct_map)} eval pairs cached)", flush=True)

    for chunk_start in range(start_chunk, total_prompts, PROMPT_CHUNK):
        chunk = all_items[chunk_start: chunk_start + PROMPT_CHUNK]
        prompts = [item[2] for item in chunk]

        outputs = llm.generate(prompts, params)

        for (si, ep_idx, _), out in zip(chunk, outputs):
            gt    = student_results[si]["gt_answer"]
            preds = [extract_boxed_answer(o.text) for o in out.outputs]
            correct_map[(si, ep_idx)] = any(answers_match(p, gt) for p in preds)

        done = min(chunk_start + PROMPT_CHUNK, total_prompts)
        print(f"  [{done:6d}/{total_prompts}] chunks processed …", flush=True)
        if phase2_path:
            _save_phase2_checkpoint(correct_map, done, phase2_path)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Apply early stopping in post-processing:
    # for each sample, walk episodes short→long and stop at first correct hit
    for si, sr in enumerate(student_results):
        for ep_idx in range(sr["num_intermediate"]):
            if correct_map.get((si, ep_idx), False):
                sr["intermediate_correct"] = True
                sr["first_correct_ep"]     = ep_idx + 1   # 1-indexed
                break

    n_inter = sum(1 for r in student_results if r["intermediate_correct"])
    print(f"  Intermediate-correct: {n_inter}/{len(student_results)}", flush=True)
    return student_results


# ── Phase 3: statistics ──────────────────────────────────────────────────────

def phase3_statistics(student_results):
    print(f"\n{'='*60}")
    print("Phase 3: Statistics")
    print(f"{'='*60}", flush=True)

    total   = len(student_results)
    n_ok    = sum(r["student_correct"]      for r in student_results)
    n_wrong = total - n_ok
    n_trunc = sum(r["truncated"]            for r in student_results)
    n_inter = sum(r["intermediate_correct"] for r in student_results)

    icfw = [r for r in student_results
            if r["intermediate_correct"] and not r["student_correct"]]
    icfc = [r for r in student_results
            if r["intermediate_correct"] and     r["student_correct"]]
    no_icfw = [r for r in student_results
               if not r["intermediate_correct"] and not r["student_correct"]]

    pct = lambda n, d: f"{n/max(d,1):.2%}"

    print(f"\n  Dataset        : {DATASET}  ({total} total samples)")
    print(f"  Student model  : {STUDENT_MODEL}")
    print(f"  Eval model     : {EVAL_MODEL}  (K={K_EVAL})")
    print(f"  MAX_NEW_TOKENS : {MAX_NEW_TOKENS}")
    print()
    print(f"  Student accuracy               : {n_ok:5d} / {total}  ({pct(n_ok, total)})")
    print(f"  Truncated responses            : {n_trunc:5d} / {total}  ({pct(n_trunc, total)})")
    print(f"  Any intermediate correct       : {n_inter:5d} / {total}  ({pct(n_inter, total)})")
    print()
    print("  ── Core query ──")
    print(f"  Intermediate correct, final correct : "
          f"{len(icfc):5d} / {total}  ({pct(len(icfc), total)})")
    print(f"  Intermediate correct, final WRONG   : "
          f"{len(icfw):5d} / {total}  ({pct(len(icfw), total)})  ← target")
    print(f"    of all wrong-answer problems       : "
          f"{len(icfw):5d} / {n_wrong}  ({pct(len(icfw), n_wrong)})")
    print(f"  No intermediate correct, final wrong: "
          f"{len(no_icfw):5d} / {total}  ({pct(len(no_icfw), total)})")
    print()

    trunc_t   = sum(1 for r in icfw if r["truncated"])
    notrunc_t = len(icfw) - trunc_t
    print(f"  Of the {len(icfw)} 'intermediate-correct but final-wrong' problems:")
    print(f"    Truncated (hit {MAX_NEW_TOKENS} tokens) : {trunc_t}")
    print(f"    Not truncated                  : {notrunc_t}")
    print()

    if icfw:
        first_eps = [r["first_correct_ep"] for r in icfw if r["first_correct_ep"]]
        if first_eps:
            arr = np.array(first_eps)
            print(f"  First-correct episode (target group):")
            print(f"    mean={arr.mean():.1f}  median={np.median(arr):.0f}"
                  f"  min={arr.min()}  max={arr.max()}")
            for ep in [1, 2, 3, 4, 5]:
                cnt = int((arr <= ep).sum())
                print(f"    ep <= {ep}: {cnt} ({pct(cnt, len(arr))})")

    subsets = sorted({r["subset"] for r in student_results if r.get("subset")})
    if subsets:
        print("\n  Per-subset student accuracy:")
        for subset in subsets:
            rows = [r for r in student_results if r.get("subset") == subset]
            ok   = sum(r["student_correct"] for r in rows)
            print(f"    {subset:30s}: {ok:4d}/{len(rows):4d}  ({pct(ok, len(rows))})")

    return icfw


def _records_from_output_json(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    records = []
    for row in rows:
        rec = dict(
            problem              = row["problem"],
            subset               = row.get("subset"),
            gt_answer            = row["gt_answer"],
            student_answer       = row.get("student_answer"),
            student_correct      = row["student_correct"],
            response_len         = row.get("response_len", 0),
            truncated            = row.get("truncated", False),
            num_intermediate     = row.get("num_intermediate", 0),
            intermediate_correct = row["intermediate_correct"],
            first_correct_ep     = row.get("first_correct_ep"),
        )
        if "episode_results" in row and rec["first_correct_ep"] is None:
            for ep_idx, ep in enumerate(row["episode_results"], start=1):
                if ep.get("correct"):
                    rec["first_correct_ep"] = ep_idx
                    break
        records.append(rec)
    return records


def stats_only(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No results file: {path}")
    records = _records_from_output_json(path)
    print(f"Loaded {len(records)} records from {path}", flush=True)
    phase3_statistics(records)


def _write_output_json(student_results, output_json):
    save_data = [
        dict(
            problem              = r["problem"],
            subset               = r.get("subset"),
            gt_answer            = r["gt_answer"],
            student_answer       = r["student_answer"],
            student_correct      = r["student_correct"],
            response_len         = r["response_len"],
            truncated            = r["truncated"],
            num_intermediate     = r["num_intermediate"],
            intermediate_correct = r["intermediate_correct"],
            first_correct_ep     = r["first_correct_ep"],
        )
        for r in student_results
    ]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"  Results saved → {output_json}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Truncation-accuracy analysis")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip phase1 if checkpoint exists; resume phase2 from last chunk",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Print statistics from existing OUTPUT_JSON and exit",
    )
    parser.add_argument(
        "--phase1-only", action="store_true",
        help="Run student generation only and save phase1 checkpoint",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.1,
        help="Per-subset sample fraction (default: 0.1; use 1.0 for full dataset)",
    )
    parser.add_argument(
        "--input-json", default=None,
        help="JSON path for --stats-only (default: derived from --sample-frac)",
    )
    args = parser.parse_args()

    output_json, phase1_path, phase2_path = _output_paths(args.sample_frac)
    input_json = args.input_json or output_json

    if args.stats_only:
        stats_only(input_json)
        return

    if not (0 < args.sample_frac <= 1.0):
        parser.error("--sample-frac must be in (0, 1]")

    import torch

    random.seed(SEED)
    torch.manual_seed(SEED)

    n_gpus = _n_gpus()
    print(f"GPUs available: {n_gpus}", flush=True)
    print(f"Sample fraction: {args.sample_frac:.0%} per subset", flush=True)
    print(f"Output: {output_json}", flush=True)
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [{i}] {name}  ({mem:.0f} GB)", flush=True)

    if args.resume and os.path.isfile(phase1_path):
        print(f"\n[resume] Loading phase1 from {phase1_path}", flush=True)
        student_results = _load_phase1_checkpoint(phase1_path)
        print(f"  {len(student_results)} samples loaded", flush=True)
    else:
        if args.resume:
            print("\n[resume] No phase1 checkpoint — running phase1 from scratch",
                  flush=True)
        samples         = load_full_dataset(sample_frac=args.sample_frac)
        student_results = phase1_student(samples)
        _save_phase1_checkpoint(student_results, phase1_path)

    if args.phase1_only:
        print("Phase1-only complete.", flush=True)
        return

    student_results = phase2_episode_eval(
        student_results, resume=args.resume, phase2_path=phase2_path,
    )
    phase3_statistics(student_results)
    _write_output_json(student_results, output_json)

    if os.path.isfile(phase2_path):
        os.remove(phase2_path)
        print(f"  Removed checkpoint {phase2_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
