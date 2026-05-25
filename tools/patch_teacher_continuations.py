"""
Patch teacher continuations in an existing kl_heatmap.html.

The original HTML was generated with a buggy prompt that forced the teacher
out of its thinking block, causing it to restart from scratch. This script:
  1. Reads the HTML and extracts the embedded JSON (const DATA = [...])
  2. Re-builds correct prompts: problem → <|im_start|>assistant\n + partial student tokens
     (no </think> insertion — let teacher continue naturally inside <think>)
  3. Runs vLLM to generate teacher continuations
  4. Patches the JSON and writes the updated HTML

Usage:
    python tools/patch_teacher_continuations.py [--html path/to/kl_heatmap.html]

Env vars:
    TEACHER_MODEL   (default: Qwen/Qwen3-8B)
    TEACHER_CONT_TOKENS  (default: 128)
"""

import argparse
import gc
import json
import os
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEACHER_MODEL      = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-8B")
TEACHER_CONT_TOKENS = int(os.environ.get("TEACHER_CONT_TOKENS", "512"))
N_GPUS             = torch.cuda.device_count()

DEFAULT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kl_heatmap.html")


def extract_data_from_html(html_text):
    """Extract the JSON array embedded as `const DATA = [...]` in the HTML."""
    m = re.search(r"const DATA = (\[.*?\]);\s*\n", html_text, re.DOTALL)
    if not m:
        raise ValueError("Could not find 'const DATA = [...]' in HTML")
    return json.loads(m.group(1))


def build_prompt(problem, partial, tokenizer):
    msgs = [{"role": "user", "content": problem}]
    base = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    base = base + "<|im_start|>assistant\n"
    return base + partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", default=DEFAULT_HTML, help="Path to kl_heatmap.html")
    args = parser.parse_args()

    html_path = args.html
    print(f"Reading {html_path} …", flush=True)
    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()

    case_data = extract_data_from_html(html_text)
    print(f"Loaded {len(case_data)} cases.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    cont_items = []  # {ci, mi, prompt}
    for ci, case in enumerate(case_data):
        merged_segs = case.get("merged_segments", [])
        tokens = case.get("tokens", [])
        problem = case.get("problem", "")
        for mseg in merged_segs:
            end = mseg.get("end", 0)
            partial = "".join(tok["text"] for tok in tokens[:end])
            prompt = build_prompt(problem, partial, tokenizer)
            cont_items.append({"ci": ci, "mi": mseg["merged_idx"], "prompt": prompt})

    print(f"Building {len(cont_items)} teacher continuation prompts …", flush=True)

    if not cont_items:
        print("No merged_segments found — nothing to do.")
        return

    tp = N_GPUS
    print(f"Launching vLLM with {tp} GPU(s) …", flush=True)
    llm = LLM(
        model=TEACHER_MODEL,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        n=1,
        max_tokens=TEACHER_CONT_TOKENS,
        temperature=0.0,
        stop=["<|im_end|>"],
    )

    print(f"Generating {len(cont_items)} continuations …", flush=True)
    outputs = llm.generate([item["prompt"] for item in cont_items], params)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Patch case_data in-place
    for item, out in zip(cont_items, outputs):
        cont_text = out.outputs[0].text if out.outputs else ""
        ci, mi = item["ci"], item["mi"]
        for mseg in case_data[ci]["merged_segments"]:
            if mseg["merged_idx"] == mi:
                mseg["teacher_continuation"] = cont_text
                break

    # Replace const DATA = [...] in HTML
    # Use a lambda so re.sub doesn't process \n in the JSON as escape sequences
    new_json = json.dumps(case_data, ensure_ascii=False)
    replacement = f"const DATA = {new_json};\n"
    new_html = re.sub(
        r"const DATA = \[.*?\];\s*\n",
        lambda _: replacement,
        html_text,
        flags=re.DOTALL,
    )

    out_path = html_path.replace(".html", "_patched.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_html)
    print(f"\nDone. Patched HTML → {out_path}", flush=True)


if __name__ == "__main__":
    main()
