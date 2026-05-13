"""
Analyze per-token teacher entropy on student on-policy samples.

1. Load student model (Qwen3-1.7B), sample responses for N prompts
2. Unload student, load teacher model (Qwen3-8B)
3. Forward-pass student responses through teacher, compute per-token entropy
4. Report distribution + high-entropy token analysis

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/analyze_teacher_entropy.py
"""

import gc
import json
import os
import random
from collections import Counter

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── config ──────────────────────────────────────────────────────────────────
STUDENT_MODEL = "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-1.7B"
TEACHER_MODEL = "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/models/Qwen3-8B"

# Dataset selection: "dapo" or "hendrycks_math"
DATASET = "hendrycks_math"

_CACHE_ROOT = (
    "/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift"
    "/data/hf_datasets_cache"
)

DATASET_CONFIGS = {
    "dapo": {
        "arrow_shards": [
            f"{_CACHE_ROOT}/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0"
            "/65877096c24ffa7abc4e4fa5edb95cf3413a5674/dapo-math-17k-train-00000-of-00002.arrow",
            f"{_CACHE_ROOT}/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0"
            "/65877096c24ffa7abc4e4fa5edb95cf3413a5674/dapo-math-17k-train-00001-of-00002.arrow",
        ],
        "prompt_key": "prompt",       # list[{role, content}] — already in chat format
    },
    "hendrycks_math": {
        "arrow_shards": [
            f"{_CACHE_ROOT}/EleutherAI___hendrycks_math/{subset}/0.0.0"
            "/21a5633873b6a120296cce3e2df9d5550074f4a3/hendrycks_math-train.arrow"
            for subset in [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
            ]
        ],
        "prompt_key": "problem",      # plain string
    },
}

N_SAMPLES      = 100
MAX_NEW_TOKENS = 200
SEED           = 42

# Student sampling params (match training config)
TEMPERATURE    = 1.0
TOP_K          = 50
TOP_P          = 1.0

OUTPUT_JSON    = os.path.join(os.path.dirname(__file__), f"teacher_entropy_results_{DATASET}.json")
OUTPUT_PNG     = os.path.join(os.path.dirname(__file__), f"teacher_entropy_hist_{DATASET}.png")
DEVICE         = "cuda:0"
DTYPE          = torch.bfloat16
# ── end config ───────────────────────────────────────────────────────────────


def build_prompt(sample, tokenizer, dataset_name):
    """Convert a dataset sample to a tokenizer-ready prompt string."""
    cfg = DATASET_CONFIGS[dataset_name]
    raw = sample[cfg["prompt_key"]]

    if isinstance(raw, list):
        # DAPO-Math-17k: already list[{role, content}]
        msgs = raw
    else:
        # hendrycks_math: plain problem string → wrap as user message
        msgs = [{"role": "user", "content": raw}]

    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return "\n".join(m["content"] for m in msgs)


def entropy_from_logits(logits):
    """logits: [vocab] → scalar entropy in nats."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum().item()


def load_dataset_samples():
    """Read Arrow shards and return N_SAMPLES random prompts."""
    random.seed(SEED)
    cfg = DATASET_CONFIGS[DATASET]
    shards = cfg["arrow_shards"]

    print(f"Loading dataset '{DATASET}' ({len(shards)} shard(s)) …", flush=True)
    pool = []
    for shard_path in shards:
        reader = ipc.open_stream(shard_path)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            for i in range(len(batch)):
                pool.append({col: batch.column(col)[i].as_py() for col in batch.schema.names})

    total = len(pool)
    idx = random.sample(range(total), min(N_SAMPLES, total))
    samples = [pool[i] for i in idx]
    del pool
    print(f"  Sampled {len(samples)} / {total} rows", flush=True)
    return samples, idx


def student_generate(samples):
    """Load student model, sample responses, unload student. Returns list of (prompt_ids, response_ids)."""
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 1: Student sampling ({STUDENT_MODEL})", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print("  Student model loaded.", flush=True)

    results = []
    for si, sample in enumerate(samples):
        prompt_str = build_prompt(sample, tokenizer, DATASET)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_ids = inputs["input_ids"][0].cpu()         # [prompt_len]
        response_ids = out[0, prompt_len:].cpu()           # [gen_len]
        full_ids = out[0].cpu()                            # [prompt_len + gen_len]
        results.append({
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "full_ids": full_ids,
            "prompt_len": prompt_len,
        })

        if (si + 1) % 10 == 0 or si == 0:
            decoded_resp = tokenizer.decode(response_ids, skip_special_tokens=True)
            print(f"  [{si+1:3d}/{N_SAMPLES}] prompt_len={prompt_len}, "
                  f"resp_len={len(response_ids)}, resp_preview={decoded_resp[:80]!r}…",
                  flush=True)

    # Free student model GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Student model unloaded.\n", flush=True)

    return results, tokenizer


def teacher_forward(student_results, student_tokenizer):
    """Load teacher model, forward-pass student sequences, compute per-token entropy."""
    print(f"{'='*60}", flush=True)
    print(f"Phase 2: Teacher forward pass ({TEACHER_MODEL})", flush=True)
    print(f"{'='*60}", flush=True)

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    # Check vocab compatibility — both should be Qwen3 family
    if teacher_tokenizer.vocab_size != student_tokenizer.vocab_size:
        print(f"  WARNING: vocab size mismatch! student={student_tokenizer.vocab_size}, "
              f"teacher={teacher_tokenizer.vocab_size}", flush=True)
        print("  Will re-tokenize with teacher tokenizer.", flush=True)
        retokenize = True
    else:
        retokenize = False

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print("  Teacher model loaded.", flush=True)

    all_entropies = []
    all_token_entropy_pairs = []
    per_sample_stats = []

    for si, sr in enumerate(student_results):
        if retokenize:
            full_text = student_tokenizer.decode(sr["full_ids"], skip_special_tokens=False)
            inputs = teacher_tokenizer(full_text, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            # Re-estimate prompt boundary by tokenizing just the prompt
            prompt_text = student_tokenizer.decode(sr["prompt_ids"], skip_special_tokens=False)
            prompt_len = teacher_tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        else:
            input_ids = sr["full_ids"].unsqueeze(0).to(DEVICE)
            prompt_len = sr["prompt_len"]

        seq_len = input_ids.shape[1]
        resp_len = seq_len - prompt_len

        if resp_len <= 0:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # logits shape: [1, seq_len, vocab_size]
        # For token at position t, logits[:, t-1, :] is the prediction distribution
        # We want teacher's distribution at each response position:
        #   logits[:, prompt_len-1 : seq_len-1, :] predicts tokens at [prompt_len : seq_len]
        resp_logits = outputs.logits[0, prompt_len - 1: seq_len - 1, :]  # [resp_len, vocab]
        resp_token_ids = input_ids[0, prompt_len: seq_len]               # [resp_len]

        sample_entropies = []
        for t in range(resp_logits.shape[0]):
            ent = entropy_from_logits(resp_logits[t])
            sample_entropies.append(ent)

            tok_id = resp_token_ids[t].item()
            tok_str = teacher_tokenizer.decode([tok_id]) if retokenize else student_tokenizer.decode([tok_id])
            all_token_entropy_pairs.append((tok_str, ent, si, tok_id))

        all_entropies.extend(sample_entropies)

        h = np.array(sample_entropies)
        per_sample_stats.append({
            "sample_idx": si,
            "n_tokens": len(sample_entropies),
            "mean":     float(h.mean()) if len(h) > 0 else 0.0,
            "median":   float(np.median(h)) if len(h) > 0 else 0.0,
            "p25":      float(np.percentile(h, 25)) if len(h) > 0 else 0.0,
            "p75":      float(np.percentile(h, 75)) if len(h) > 0 else 0.0,
            "p90":      float(np.percentile(h, 90)) if len(h) > 0 else 0.0,
            "frac_below_0_5": float((h < 0.5).mean()) if len(h) > 0 else 0.0,
            "frac_below_1_0": float((h < 1.0).mean()) if len(h) > 0 else 0.0,
            "frac_below_1_5": float((h < 1.5).mean()) if len(h) > 0 else 0.0,
            "frac_below_2_0": float((h < 2.0).mean()) if len(h) > 0 else 0.0,
        })

        if (si + 1) % 10 == 0 or si == 0:
            print(f"  [{si+1:3d}/{N_SAMPLES}] resp_tokens={len(sample_entropies)}, "
                  f"mean_H={h.mean():.3f}, frac<1.0={per_sample_stats[-1]['frac_below_1_0']:.2%}",
                  flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    tok = teacher_tokenizer if retokenize else student_tokenizer
    return all_entropies, all_token_entropy_pairs, per_sample_stats, tok


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    samples, idx = load_dataset_samples()

    # Phase 1: student on-policy sampling
    student_results, student_tokenizer = student_generate(samples)

    # Phase 2: teacher forward pass on student samples
    all_entropies, all_token_entropy_pairs, per_sample_stats, tokenizer = teacher_forward(
        student_results, student_tokenizer
    )

    # ── aggregate stats ───────────────────────────────────────────────────────
    H = np.array(all_entropies)
    thresholds = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    agg = {
        "n_tokens_total": len(H),
        "mean":   float(H.mean()),
        "std":    float(H.std()),
        "median": float(np.median(H)),
        "p10":    float(np.percentile(H, 10)),
        "p25":    float(np.percentile(H, 25)),
        "p75":    float(np.percentile(H, 75)),
        "p90":    float(np.percentile(H, 90)),
        "p95":    float(np.percentile(H, 95)),
        "p99":    float(np.percentile(H, 99)),
        "threshold_fractions": {
            f"below_{t}": float((H < t).mean()) for t in thresholds
        },
    }

    print(f"\n{'='*60}")
    print("Aggregate entropy stats (teacher on student on-policy)")
    print(f"{'='*60}")
    print(f"  Dataset : {DATASET}")
    print(f"  Student : {STUDENT_MODEL}")
    print(f"  Teacher : {TEACHER_MODEL}")
    print(f"  Total tokens  : {agg['n_tokens_total']:,}")
    print(f"  Mean ± std    : {agg['mean']:.4f} ± {agg['std']:.4f} nats")
    print(f"  Median        : {agg['median']:.4f}")
    print(f"  p10/p25/p75/p90/p95/p99: "
          f"{agg['p10']:.3f} / {agg['p25']:.3f} / {agg['p75']:.3f} / "
          f"{agg['p90']:.3f} / {agg['p95']:.3f} / {agg['p99']:.3f}")
    print("  Fraction of tokens below threshold:")
    for t, frac in agg["threshold_fractions"].items():
        bar = "█" * int(frac * 40)
        print(f"    τ = {t.split('_')[1]:>4s} nats : {frac:6.1%}  {bar}")

    # ── high-entropy token analysis ──────────────────────────────────────────
    sorted_pairs = sorted(all_token_entropy_pairs, key=lambda x: x[1], reverse=True)
    TOP_K = 50
    print(f"\n=== Top-{TOP_K} highest-entropy tokens ===")
    print(f"{'Rank':>4s}  {'Entropy':>8s}  {'TokID':>7s}  {'Sample':>6s}  Token")
    print("-" * 60)
    for rank, (tok_str, ent, sample_i, tok_id) in enumerate(sorted_pairs[:TOP_K], 1):
        print(f"{rank:4d}  {ent:8.4f}  {tok_id:7d}  {sample_i:6d}  {repr(tok_str)}")

    p90_threshold = float(np.percentile(H, 90))
    high_ent_tokens = [t for t in all_token_entropy_pairs if t[1] >= p90_threshold]
    token_freq = Counter(tok_str for tok_str, _, _, _ in high_ent_tokens)
    print(f"\n=== Most frequent tokens in high-entropy positions (entropy >= {p90_threshold:.3f}, p90) ===")
    print(f"  Total high-entropy token positions: {len(high_ent_tokens)}")
    print(f"{'Rank':>4s}  {'Count':>5s}  {'Frac':>6s}  Token")
    print("-" * 45)
    for rank, (tok_str, cnt) in enumerate(token_freq.most_common(30), 1):
        frac = cnt / len(high_ent_tokens)
        print(f"{rank:4d}  {cnt:5d}  {frac:6.1%}  {repr(tok_str)}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    high_entropy_details = [
        {"token": tok_str, "entropy": ent, "sample_idx": si, "token_id": tid}
        for tok_str, ent, si, tid in sorted_pairs[:200]
    ]
    result = {
        "config": {
            "dataset": DATASET,
            "student_model": STUDENT_MODEL,
            "teacher_model": TEACHER_MODEL,
            "n_samples": N_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "seed": SEED,
        },
        "aggregate": agg,
        "per_sample": per_sample_stats,
        "all_entropies": H.tolist(),
        "high_entropy_tokens_top200": high_entropy_details,
        "high_entropy_token_freq": [
            {"token": tok_str, "count": cnt}
            for tok_str, cnt in token_freq.most_common(50)
        ],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUTPUT_JSON}")

    # ── matplotlib histogram ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Teacher entropy on student on-policy  |  {DATASET}  |  "
            f"S={os.path.basename(STUDENT_MODEL)}, T={os.path.basename(TEACHER_MODEL)}  |  "
            f"{N_SAMPLES} samples, {agg['n_tokens_total']:,} tokens",
            fontsize=11,
        )

        ax = axes[0]
        ax.hist(H, bins=120, color="#4c8fe8", edgecolor="none", alpha=0.85)
        for t, c in zip([0.5, 1.0, 1.5, 2.0], ["#e04a3f", "#f09030", "#40b060", "#9060c0"]):
            frac = float((H < t).mean())
            ax.axvline(t, color=c, lw=1.8, linestyle="--",
                       label=f"τ={t:.1f}  ({frac:.1%} below)")
        ax.set_xlabel("Per-token entropy (nats)")
        ax.set_ylabel("Token count")
        ax.set_title("Full distribution")
        ax.legend(fontsize=9)

        ax2 = axes[1]
        sorted_H = np.sort(H)
        cdf = np.arange(1, len(sorted_H) + 1) / len(sorted_H)
        ax2.plot(sorted_H, cdf, color="#4c8fe8", lw=1.5)
        for t, c in zip([0.5, 1.0, 1.5, 2.0], ["#e04a3f", "#f09030", "#40b060", "#9060c0"]):
            frac = float((H < t).mean())
            ax2.axvline(t, color=c, lw=1.8, linestyle="--", label=f"τ={t:.1f}  ({frac:.1%})")
            ax2.axhline(frac, color=c, lw=0.7, linestyle=":")
        ax2.set_xlabel("Entropy threshold τ (nats)")
        ax2.set_ylabel("Fraction of tokens kept (CDF)")
        ax2.set_title("CDF — fraction of tokens below τ")
        ax2.legend(fontsize=9)
        ax2.set_xlim(left=0)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(OUTPUT_PNG, dpi=150)
        print(f"Histogram saved → {OUTPUT_PNG}")
    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
