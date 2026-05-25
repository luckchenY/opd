"""
Response Length Analysis: Student vs Teacher on correctly-answered problems.

Samples problems from hendrycks_math, runs both models, then compares
response lengths broken down by correctness status.

Flow:
  Phase 1 — Student sampling (on-policy) + correctness check
  Phase 2 — Teacher greedy generation + correctness check
  Phase 3 — Statistics + optional HTML histogram

Usage:
    python tools/analyze_response_length.py
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/analyze_response_length.py
    STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Qwen/Qwen3-8B python tools/analyze_response_length.py
"""

import gc
import json
import os
import random

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
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
        "level_key":    "level",
    },
}

N_SAMPLES      = 10_000_000  # effectively unlimited — uses all available data
MAX_NEW_TOKENS = 4096
SEED           = 42
TEMPERATURE    = 1.0
TOP_K          = 50
TOP_P          = 1.0

OUTPUT_HTML = os.path.join(os.path.dirname(__file__), "response_length.html")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "response_length_data.json")

N_GPUS = torch.cuda.device_count()
# ── end config ───────────────────────────────────────────────────────────────


# ── answer extraction ────────────────────────────────────────────────────────

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
        for row in ds:
            pool.append(dict(row, _subset=subset))
    total = len(pool)
    idx   = random.sample(range(total), min(N_SAMPLES, total))
    samples = [pool[i] for i in idx]
    del pool
    print(f"  Sampled {len(samples)} / {total} rows", flush=True)
    return samples


def _vllm_generate(model_path, prompt_strings, do_sample=True):
    tp = N_GPUS
    print(f"  Initializing vLLM (tp={tp}) …", flush=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=MAX_NEW_TOKENS + 512,
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
        gen_ids  = out.outputs[0].token_ids
        gen_text = out.outputs[0].text
        results.append((list(gen_ids), gen_text))

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ── Phase 1: student sampling ─────────────────────────────────────────────────

def phase1_student(samples):
    print(f"\n{'='*60}")
    print(f"Phase 1: Student sampling ({os.path.basename(STUDENT_MODEL)}) "
          f"[vLLM, {N_GPUS} GPU(s)]")
    print(f"{'='*60}", flush=True)

    tokenizer   = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    cfg         = DATASET_CONFIGS[DATASET]
    prompts     = [build_prompt(s[cfg["prompt_key"]], tokenizer) for s in samples]
    gt_answers  = [extract_boxed_answer(s[cfg["solution_key"]]) for s in samples]

    gen_outputs = _vllm_generate(STUDENT_MODEL, prompts, do_sample=True)

    results = []
    for si, (sample, (gen_ids, gen_text), gt_answer) in enumerate(
            zip(samples, gen_outputs, gt_answers)):
        student_answer = extract_boxed_answer(gen_text)
        correct        = answers_match(student_answer, gt_answer)
        results.append(dict(
            problem       = sample[cfg["prompt_key"]],
            gt_solution   = sample[cfg["solution_key"]],
            gt_answer     = gt_answer,
            level         = sample.get(cfg.get("level_key", "level"), ""),
            subset        = sample.get("_subset", ""),
            student_response = gen_text,
            student_answer   = student_answer,
            student_correct  = correct,
            student_len      = len(gen_ids),
        ))
        if (si + 1) % 50 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  len={len(gen_ids)}  "
                  f"gt={gt_answer}  pred={student_answer}", flush=True)

    n_ok = sum(r["student_correct"] for r in results)
    print(f"  Student accuracy: {n_ok}/{len(results)} ({n_ok/len(results):.1%})", flush=True)
    return results, tokenizer


# ── Phase 2: teacher greedy generation ───────────────────────────────────────

def phase2_teacher(samples, student_results, student_tokenizer):
    print(f"\n{'='*60}")
    print(f"Phase 2: Teacher generation ({os.path.basename(TEACHER_MODEL)}) "
          f"[vLLM, {N_GPUS} GPU(s)]")
    print(f"{'='*60}", flush=True)

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    cfg     = DATASET_CONFIGS[DATASET]
    prompts = [build_prompt(s[cfg["prompt_key"]], teacher_tokenizer) for s in samples]

    gen_outputs = _vllm_generate(TEACHER_MODEL, prompts, do_sample=True)

    for si, (sr, (gen_ids, gen_text)) in enumerate(zip(student_results, gen_outputs)):
        teacher_answer = extract_boxed_answer(gen_text)
        correct        = answers_match(teacher_answer, sr["gt_answer"])
        sr["teacher_response"] = gen_text
        sr["teacher_answer"]   = teacher_answer
        sr["teacher_correct"]  = correct
        sr["teacher_len"]      = len(gen_ids)

        if (si + 1) % 50 == 0 or si == 0:
            tag = "✓" if correct else "✗"
            print(f"  [{si+1:3d}/{N_SAMPLES}] {tag}  len={len(gen_ids)}  "
                  f"gt={sr['gt_answer']}  pred={teacher_answer}", flush=True)

    n_ok = sum(r["teacher_correct"] for r in student_results)
    print(f"  Teacher accuracy: {n_ok}/{len(student_results)} ({n_ok/len(student_results):.1%})",
          flush=True)
    return student_results


# ── Phase 3: statistics ───────────────────────────────────────────────────────

def _stats(lens):
    if not lens:
        return {}
    a = np.array(lens, dtype=float)
    return dict(
        n       = len(lens),
        mean    = float(a.mean()),
        median  = float(np.median(a)),
        std     = float(a.std()),
        p25     = float(np.percentile(a, 25)),
        p75     = float(np.percentile(a, 75)),
        p90     = float(np.percentile(a, 90)),
        p95     = float(np.percentile(a, 95)),
        min     = float(a.min()),
        max     = float(a.max()),
        truncated = int(sum(1 for l in lens if l >= MAX_NEW_TOKENS)),
    )


def phase3_stats(results):
    print(f"\n{'='*60}")
    print("Phase 3: Response length statistics")
    print(f"{'='*60}", flush=True)

    groups = {
        "Student ✓": [r["student_len"] for r in results if r["student_correct"]],
        "Student ✗": [r["student_len"] for r in results if not r["student_correct"]],
        "Teacher ✓": [r["teacher_len"] for r in results if r["teacher_correct"]],
        "Teacher ✗": [r["teacher_len"] for r in results if not r["teacher_correct"]],
        # both correct / student wrong+teacher right
        "Both ✓":    [r["student_len"] for r in results
                      if r["student_correct"] and r["teacher_correct"]],
        "S✗ T✓":     [r["student_len"] for r in results
                      if not r["student_correct"] and r["teacher_correct"]],
    }

    stats = {}
    header = f"  {'Group':<14} {'n':>5} {'mean':>7} {'median':>7} {'std':>7} " \
             f"{'p25':>6} {'p75':>6} {'p90':>6} {'trunc':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, lens in groups.items():
        s = _stats(lens)
        stats[label] = s
        if s:
            print(f"  {label:<14} {s['n']:>5} {s['mean']:>7.0f} {s['median']:>7.0f} "
                  f"{s['std']:>7.0f} {s['p25']:>6.0f} {s['p75']:>6.0f} "
                  f"{s['p90']:>6.0f} {s['truncated']:>6}")
        else:
            print(f"  {label:<14}  (no data)")

    # Per-level breakdown
    levels = sorted(set(r.get("level", "") for r in results if r.get("level")))
    if levels:
        print(f"\n  Per-level breakdown (Student ✓ / Teacher ✓ mean length):")
        print(f"  {'Level':<10} {'S✓ n':>6} {'S✓ mean':>8} {'T✓ n':>6} {'T✓ mean':>8}")
        print("  " + "-" * 42)
        level_stats = {}
        for lv in levels:
            s_ok  = [r["student_len"] for r in results if r["student_correct"]  and r.get("level") == lv]
            t_ok  = [r["teacher_len"] for r in results if r["teacher_correct"]  and r.get("level") == lv]
            s_mu  = np.mean(s_ok) if s_ok else float("nan")
            t_mu  = np.mean(t_ok) if t_ok else float("nan")
            level_stats[lv] = dict(student_correct=_stats(s_ok), teacher_correct=_stats(t_ok))
            print(f"  {lv:<10} {len(s_ok):>6} {s_mu:>8.0f} {len(t_ok):>6} {t_mu:>8.0f}")
        stats["per_level"] = level_stats

    return stats


# ── Phase 4: HTML histogram ───────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Response Length — Student vs Teacher</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0f0f1a;color:#d8d8e8;
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
  line-height:1.6;padding:32px 24px;
}
.container{max-width:1100px;margin:0 auto}
h1{font-size:1.5rem;font-weight:700;margin-bottom:6px;color:#f0f0ff}
.subtitle{font-size:.85rem;color:#888;margin-bottom:28px}
.controls{
  background:#181830;border:1px solid #2a2a4a;border-radius:10px;
  padding:16px 22px;margin-bottom:28px;display:flex;gap:24px;
  flex-wrap:wrap;align-items:center;
}
.ctrl-group{display:flex;flex-direction:column;gap:4px}
.ctrl-group label{font-size:.75rem;color:#999;text-transform:uppercase;letter-spacing:.06em}
select,input[type=number]{
  background:#22224a;color:#ccc;border:1px solid #3a3a6a;
  border-radius:6px;padding:4px 8px;font-size:.85rem;
}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}
@media(max-width:720px){.grid{grid-template-columns:1fr}}
.card{
  background:#14142a;border:1px solid #252548;border-radius:10px;
  padding:20px 22px;
}
.card h2{font-size:1rem;font-weight:600;color:#c0c0ff;margin-bottom:14px}
.stat-table{width:100%;border-collapse:collapse;font-size:.83rem}
.stat-table th{color:#888;font-weight:500;text-align:left;padding:3px 8px;
  border-bottom:1px solid #252548;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
.stat-table td{padding:4px 8px;font-variant-numeric:tabular-nums}
.stat-table tr:nth-child(even) td{background:rgba(255,255,255,.02)}
.val{color:#7ecfff;font-weight:600}
.val-good{color:#6fcf6f}
.val-bad{color:#cf7070}
canvas{width:100%!important;display:block}
.chart-wrap{margin-top:16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.78rem;margin-bottom:10px}
.legend-item{display:flex;align-items:center;gap:6px}
.legend-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.full-card{
  background:#14142a;border:1px solid #252548;border-radius:10px;
  padding:20px 22px;margin-bottom:20px;
}
.full-card h2{font-size:1rem;font-weight:600;color:#c0c0ff;margin-bottom:14px}
</style>
</head>
<body>
<div class="container">
  <h1>Response Length Analysis</h1>
  <div class="subtitle">
    Student: __STUDENT__ &nbsp;|&nbsp; Teacher: __TEACHER__ &nbsp;|&nbsp;
    Dataset: __DATASET__ &nbsp;|&nbsp; N = __N_SAMPLES__ problems
  </div>

  <div class="controls">
    <div class="ctrl-group">
      <label>Bin width (tokens)</label>
      <select id="binWidth">
        <option value="64">64</option>
        <option value="128" selected>128</option>
        <option value="256">256</option>
        <option value="512">512</option>
      </select>
    </div>
    <div class="ctrl-group">
      <label>Density / Count</label>
      <select id="yMode">
        <option value="density" selected>Density (fraction)</option>
        <option value="count">Count</option>
      </select>
    </div>
    <div class="ctrl-group">
      <label>Groups</label>
      <select id="groupSel" multiple size="4" style="min-width:160px">
        <option value="s_ok"  selected>Student ✓</option>
        <option value="s_fail" selected>Student ✗</option>
        <option value="t_ok"  selected>Teacher ✓</option>
        <option value="t_fail" selected>Teacher ✗</option>
      </select>
    </div>
  </div>

  <div class="full-card">
    <h2>Response length distribution</h2>
    <div class="legend" id="legend"></div>
    <div class="chart-wrap"><canvas id="histCanvas" height="300"></canvas></div>
  </div>

  <div class="grid" id="stat-grid"></div>

  <div class="full-card" id="level-section" style="display:none">
    <h2>Per-difficulty-level breakdown</h2>
    <canvas id="levelCanvas" height="240"></canvas>
  </div>
</div>

<script>
const RAW    = __JSON_DATA__;
const STATS  = __STATS_DATA__;
const MAX_LEN = __MAX_NEW_TOKENS__;

const GROUPS = {
  s_ok:   {label:"Student ✓", color:"#5b9def", data: RAW.filter(r=>r.student_correct).map(r=>r.student_len)},
  s_fail: {label:"Student ✗", color:"#ef6b5b", data: RAW.filter(r=>!r.student_correct).map(r=>r.student_len)},
  t_ok:   {label:"Teacher ✓", color:"#5bef9d", data: RAW.filter(r=>r.teacher_correct).map(r=>r.teacher_len)},
  t_fail: {label:"Teacher ✗", color:"#efcf5b", data: RAW.filter(r=>!r.teacher_correct).map(r=>r.teacher_len)},
};

function buildHistogram(data, binW) {
  if (!data.length) return {bins:[], counts:[]};
  const maxV = Math.max(...data, MAX_LEN);
  const bins = [];
  for (let lo = 0; lo < maxV; lo += binW) bins.push(lo);
  const counts = new Array(bins.length).fill(0);
  data.forEach(v => {
    const i = Math.min(Math.floor(v / binW), bins.length - 1);
    counts[i]++;
  });
  return {bins, counts};
}

function drawHist() {
  const binW   = parseInt(document.getElementById('binWidth').value);
  const yMode  = document.getElementById('yMode').value;
  const sel    = Array.from(document.getElementById('groupSel').selectedOptions).map(o=>o.value);
  const canvas = document.getElementById('histCanvas');
  const dpr    = window.devicePixelRatio || 1;
  canvas.width  = canvas.offsetWidth  * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  const ctx    = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const pad = {top:20, right:30, bottom:40, left:56};
  ctx.clearRect(0, 0, W, H);

  const activeGroups = sel.filter(k => GROUPS[k].data.length > 0).map(k => {
    const {bins, counts} = buildHistogram(GROUPS[k].data, binW);
    const total = GROUPS[k].data.length || 1;
    const vals  = yMode === 'density' ? counts.map(c => c / total) : counts;
    return {key:k, label:GROUPS[k].label, color:GROUPS[k].color, bins, vals};
  });

  if (!activeGroups.length) return;

  const allBins = activeGroups[0].bins;
  const maxY    = Math.max(...activeGroups.flatMap(g => g.vals), 1e-9);

  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top  - pad.bottom;

  // axes
  ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + chartH);
  ctx.lineTo(pad.left + chartW, pad.top + chartH);
  ctx.stroke();

  // y gridlines
  const nGrid = 5;
  ctx.setLineDash([3,4]); ctx.strokeStyle = '#1e1e3a';
  for (let gi = 1; gi <= nGrid; gi++) {
    const yv = maxY * gi / nGrid;
    const ypx = pad.top + chartH - (yv / maxY) * chartH;
    ctx.beginPath(); ctx.moveTo(pad.left, ypx); ctx.lineTo(pad.left + chartW, ypx); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(yMode === 'density' ? (yv*100).toFixed(1)+'%' : Math.round(yv), pad.left - 6, ypx + 4);
  }
  ctx.setLineDash([]);

  // bars
  const barW = chartW / allBins.length;
  const groupW = barW * 0.9 / activeGroups.length;
  activeGroups.forEach((g, gi) => {
    ctx.fillStyle = g.color + '88';
    ctx.strokeStyle = g.color;
    ctx.lineWidth = 1;
    g.bins.forEach((bin, bi) => {
      const v = g.vals[bi];
      const x = pad.left + bi * barW + gi * groupW;
      const h = (v / maxY) * chartH;
      const y = pad.top + chartH - h;
      ctx.fillRect(x, y, groupW, h);
      ctx.strokeRect(x, y, groupW, h);
    });
  });

  // x-axis labels
  ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
  const step = Math.max(1, Math.ceil(allBins.length / 10));
  allBins.forEach((bin, bi) => {
    if (bi % step !== 0) return;
    const x = pad.left + bi * barW + barW / 2;
    ctx.fillText(bin, x, pad.top + chartH + 16);
  });

  // mean lines
  activeGroups.forEach(g => {
    const mean = GROUPS[g.key].data.reduce((a,b)=>a+b,0) / GROUPS[g.key].data.length;
    const x = pad.left + (mean / (allBins.length * binW)) * chartW;
    ctx.strokeStyle = g.color; ctx.lineWidth = 2; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + chartH); ctx.stroke();
    ctx.setLineDash([]);
  });

  // legend
  const legendEl = document.getElementById('legend');
  legendEl.innerHTML = '';
  activeGroups.forEach(g => {
    const mean = (GROUPS[g.key].data.reduce((a,b)=>a+b,0) / GROUPS[g.key].data.length).toFixed(0);
    legendEl.innerHTML += `<div class="legend-item">
      <div class="legend-dot" style="background:${g.color}"></div>
      <span>${g.label} (n=${GROUPS[g.key].data.length}, μ=${mean})</span>
    </div>`;
  });
}

function renderStatGrid() {
  const grid = document.getElementById('stat-grid');
  const entries = [
    {key:"Student ✓", color:"#5b9def", model:"student"},
    {key:"Student ✗", color:"#ef6b5b", model:"student"},
    {key:"Teacher ✓", color:"#5bef9d", model:"teacher"},
    {key:"Teacher ✗", color:"#efcf5b", model:"teacher"},
  ];
  entries.forEach(({key, color}) => {
    const s = STATS[key];
    if (!s || !s.n) return;
    const trPct = s.n ? (s.truncated / s.n * 100).toFixed(1) : '0.0';
    grid.innerHTML += `
    <div class="card">
      <h2 style="color:${color}">${key}</h2>
      <table class="stat-table">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Count (n)</td><td class="val">${s.n}</td></tr>
        <tr><td>Mean</td><td class="val">${s.mean.toFixed(1)}</td></tr>
        <tr><td>Median</td><td class="val">${s.median.toFixed(1)}</td></tr>
        <tr><td>Std dev</td><td class="val">${s.std.toFixed(1)}</td></tr>
        <tr><td>P25 / P75</td><td class="val">${s.p25.toFixed(0)} / ${s.p75.toFixed(0)}</td></tr>
        <tr><td>P90 / P95</td><td class="val">${s.p90.toFixed(0)} / ${s.p95.toFixed(0)}</td></tr>
        <tr><td>Min / Max</td><td class="val">${s.min.toFixed(0)} / ${s.max.toFixed(0)}</td></tr>
        <tr><td>Truncated (≥${MAX_LEN})</td>
            <td class="${parseFloat(trPct) > 5 ? 'val-bad' : 'val'}">${s.truncated} (${trPct}%)</td></tr>
      </table>
    </div>`;
  });
}

function renderLevelChart() {
  const perLevel = STATS.per_level;
  if (!perLevel) return;
  const levels = Object.keys(perLevel).sort();
  if (!levels.length) return;
  document.getElementById('level-section').style.display = '';

  const canvas = document.getElementById('levelCanvas');
  const dpr    = window.devicePixelRatio || 1;
  canvas.width  = canvas.offsetWidth  * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  const ctx    = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const pad = {top:20, right:30, bottom:50, left:60};
  ctx.clearRect(0, 0, W, H);

  const colors = ["#5b9def", "#5bef9d"];
  const labels = ["Student ✓ mean", "Teacher ✓ mean"];
  const series = [
    levels.map(lv => perLevel[lv].student_correct?.mean || 0),
    levels.map(lv => perLevel[lv].teacher_correct?.mean || 0),
  ];
  const maxY = Math.max(...series.flat(), 1);
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top  - pad.bottom;

  ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + chartH);
  ctx.lineTo(pad.left + chartW, pad.top + chartH);
  ctx.stroke();

  ctx.setLineDash([3,4]); ctx.strokeStyle = '#1e1e3a';
  for (let gi = 1; gi <= 5; gi++) {
    const yv  = maxY * gi / 5;
    const ypx = pad.top + chartH - (yv / maxY) * chartH;
    ctx.beginPath(); ctx.moveTo(pad.left, ypx); ctx.lineTo(pad.left + chartW, ypx); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(Math.round(yv), pad.left - 6, ypx + 4);
  }
  ctx.setLineDash([]);

  const barSlot = chartW / levels.length;
  const barW    = barSlot * 0.8 / series.length;
  series.forEach((vals, si) => {
    ctx.fillStyle   = colors[si] + '88';
    ctx.strokeStyle = colors[si];
    ctx.lineWidth = 1;
    vals.forEach((v, li) => {
      const h  = (v / maxY) * chartH;
      const x  = pad.left + li * barSlot + si * barW + barSlot * 0.1;
      const y  = pad.top + chartH - h;
      ctx.fillRect(x, y, barW, h);
      ctx.strokeRect(x, y, barW, h);
      if (v > 0) {
        ctx.fillStyle = colors[si]; ctx.font = '10px monospace'; ctx.textAlign = 'center';
        ctx.fillText(Math.round(v), x + barW / 2, y - 3);
        ctx.fillStyle = colors[si] + '88';
      }
    });
  });

  ctx.fillStyle = '#888'; ctx.font = '12px monospace'; ctx.textAlign = 'center';
  levels.forEach((lv, li) => {
    const x = pad.left + li * barSlot + barSlot / 2;
    ctx.fillText(lv, x, pad.top + chartH + 20);
  });

  const legendEl = document.querySelector('#level-section .legend') || (() => {
    const d = document.createElement('div'); d.className='legend';
    document.getElementById('level-section').insertBefore(d, document.getElementById('levelCanvas'));
    return d;
  })();
  legendEl.innerHTML = labels.map((l,i) =>
    `<div class="legend-item"><div class="legend-dot" style="background:${colors[i]}"></div><span>${l}</span></div>`
  ).join('');
}

document.getElementById('binWidth').addEventListener('change', drawHist);
document.getElementById('yMode').addEventListener('change', drawHist);
document.getElementById('groupSel').addEventListener('change', drawHist);
window.addEventListener('resize', () => { drawHist(); renderLevelChart(); });

renderStatGrid();
drawHist();
renderLevelChart();
</script>
</body>
</html>"""


def phase4_html(results, stats):
    print(f"\n{'='*60}")
    print("Phase 4: Generating HTML report")
    print(f"{'='*60}", flush=True)

    row_data = [
        dict(
            problem        = r["problem"],
            gt_answer      = r["gt_answer"],
            student_answer = r["student_answer"],
            teacher_answer = r["teacher_answer"],
            student_correct= r["student_correct"],
            teacher_correct= r["teacher_correct"],
            student_len    = r["student_len"],
            teacher_len    = r["teacher_len"],
            level          = r.get("level", ""),
            subset         = r.get("subset", ""),
        )
        for r in results
    ]

    html = HTML_TEMPLATE
    html = html.replace("__JSON_DATA__",        json.dumps(row_data, ensure_ascii=False))
    html = html.replace("__STATS_DATA__",       json.dumps(stats,    ensure_ascii=False))
    html = html.replace("__STUDENT__",          os.path.basename(STUDENT_MODEL))
    html = html.replace("__TEACHER__",          os.path.basename(TEACHER_MODEL))
    html = html.replace("__DATASET__",          DATASET)
    html = html.replace("__N_SAMPLES__",        str(len(results)))
    html = html.replace("__MAX_NEW_TOKENS__",   str(MAX_NEW_TOKENS))

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML saved → {OUTPUT_HTML}", flush=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": row_data, "stats": stats}, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved → {OUTPUT_JSON}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"GPUs available: {N_GPUS}", flush=True)
    for i in range(N_GPUS):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [{i}] {name}  ({mem:.0f} GB)", flush=True)

    samples = load_dataset_samples()

    results, student_tok = phase1_student(samples)
    results = phase2_teacher(samples, results, student_tok)
    stats   = phase3_stats(results)
    phase4_html(results, stats)

    print(f"\nDone. Open {OUTPUT_HTML} in a browser.", flush=True)


if __name__ == "__main__":
    main()
