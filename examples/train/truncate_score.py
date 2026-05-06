#!/usr/bin/env python3
"""从一次最长的 rollout 结果中，按不同 token 长度截断后重新打分（通用数学版）。

思路：
  模型输出通常是 `<think>…</think>\\boxed{…}`。一次 rollout 到 MAX_LEN 后，
  把同一个 prediction 字符串按 token 数截断当作"短 budget"下的模型输出，
  重新抽 `\\boxed{…}` 并与 target 做 math_equal 判等 —— 等价于在不重新推理的前提下
  拿到不同 think 长度下的 acc / pass@k。

适用数据集（subset 名任意）：AIME / MATH-500 / Hendrycks MATH / AMC 等。
不适用：代码类（如 LiveCodeBench）—— 见 truncate_score_code.py。

与旧版的差异：
  - glob 不再写死 `*_default.jsonl`，改为匹配所有 review jsonl；
  - 用 evalscope.metrics.math_parser 里的 extract_answer + math_equal 替换
    "正则剥非数字 + int()" 的 AIME 专用判等，支持分数/根式/sympy 等；
  - 按 subset（文件名）分组输出，并给出整体 micro-mean；
  - **修复**：reviews 里的 `sample_score.score.prediction` 只是 `</think>` 之后
    的最终答案正文（典型 1k~4k token），用它做截断永远截不到思考过程，导致所有
    长度档分数全等。改成从同目录 predictions/*.jsonl 拿 `model_output.choices[0]
    .message.content` 还原完整 raw（含 `<think>...</think>` 思考段），再按 token
    截断，才反映"短 budget 下被砍掉思考"的真实表现。
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


def load_math_utils():
    from evalscope.metrics.math_parser import extract_answer, math_equal
    return extract_answer, math_equal


def score(pred_text, target, extract_answer, math_equal):
    pred = extract_answer(pred_text or '')
    if not pred:
        return 0
    try:
        return 1 if math_equal(pred, str(target)) else 0
    except Exception:
        return 0


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def truncate_by_tokens(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def reconstruct_raw(content):
    """把 evalscope parse 过的 message.content 还原成 raw 生成文本。

    - list 形态（<think>...</think> 已闭合，被 reasoning_parser 拆成两段）：
        拼成 `<think>\n{reasoning}\n</think>\n\n{text}`
    - str 形态（<think> 没闭合，整坨带原始标签）：原样返回
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    reasoning = ''
    text = ''
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get('type') == 'reasoning' and p.get('reasoning'):
            reasoning = p['reasoning']
        elif p.get('type') == 'text' and p.get('text'):
            text = p['text']
    if reasoning:
        return f'<think>\n{reasoning}\n</think>\n\n{text}'
    return text


def build_prediction_index(pred_path):
    """predictions/*.jsonl -> {index: (raw_text, output_tokens, stop_reason)}"""
    idx = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mo = r.get('model_output') or {}
            choice = (mo.get('choices') or [{}])[0]
            content = (choice.get('message') or {}).get('content', '')
            raw = reconstruct_raw(content)
            ot = (mo.get('usage') or {}).get('output_tokens') or 0
            sr = choice.get('stop_reason') or choice.get('finish_reason')
            idx[str(r.get('index'))] = (raw, ot, sr)
    return idx


def parse_bench_subset(fname):
    """`{bench}_{subset}.jsonl` -> (bench, subset)；若解析不出，就整体当 bench，subset=''。"""
    stem = re.sub(r'\.jsonl$', '', fname)
    m = re.match(r'^([a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*?)_(.+)$', stem)
    if m:
        return m.group(1), m.group(2)
    return stem, ''


def process_review_file(review_path, pred_index, tokenizer,
                        truncate_lengths, pass_k,
                        extract_answer, math_equal):
    samples = []
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    results = {}
    for length in truncate_lengths:
        per_group = defaultdict(list)
        truncated_cnt = 0
        for s in samples:
            target = s.get('target')
            idx = str(s.get('index'))
            group_id = s.get('group_id') or s.get('sample_score', {}).get('group_id')
            if group_id is None:
                group_id = idx
            raw, ot, _sr = pred_index.get(idx, ('', 0, None))
            if ot and ot > length:
                truncated_cnt += 1
            truncated = truncate_by_tokens(raw, tokenizer, length)
            acc = score(truncated, target, extract_answer, math_equal)
            per_group[group_id].append(acc)

        num_groups = len(per_group)
        total_samples = sum(len(v) for v in per_group.values())
        total_acc = sum(sum(v) for v in per_group.values())

        if total_samples == 0 or num_groups == 0:
            results[length] = {
                'mean_acc': 0.0, f'pass@{pass_k}': 0.0,
                'num_groups': 0, 'num_samples': 0, 'truncated': 0,
            }
            continue

        mean_acc = total_acc / total_samples
        pass_at_k = sum(1 for v in per_group.values() if max(v) >= 1) / num_groups
        results[length] = {
            'mean_acc': mean_acc,
            f'pass@{pass_k}': pass_at_k,
            'num_groups': num_groups,
            'num_samples': total_samples,
            'truncated': truncated_cnt,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_output_dir', required=True)
    parser.add_argument('--stats_tsv', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--pass_k', type=int, default=8)
    parser.add_argument('--truncate_lengths', type=int, nargs='+', required=True)
    args = parser.parse_args()

    review_files = sorted(glob.glob(
        os.path.join(args.eval_output_dir, 'native', '*', 'reviews', '*', '*.jsonl')
    ))
    if not review_files:
        print(f'[ERROR] No review files in {args.eval_output_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'[INFO] Loading tokenizer from {args.model_path}')
    tokenizer = load_tokenizer(args.model_path)
    extract_answer, math_equal = load_math_utils()

    # 收集每个 length 的 per-subset 结果用于最后做 micro-mean
    overall = defaultdict(lambda: {'total_acc': 0, 'total_samples': 0,
                                   'pass_groups': 0, 'num_groups': 0,
                                   'truncated': 0})

    with open(args.stats_tsv, 'a', encoding='utf-8') as out:
        for review_path in review_files:
            fname = os.path.basename(review_path)
            bench, subset = parse_bench_subset(fname)
            tag = f'{bench}/{subset}' if subset else bench
            # 同目录下的 predictions 文件：reviews/<model>/x.jsonl -> predictions/<model>/x.jsonl
            pred_path = review_path.replace(
                os.sep + 'reviews' + os.sep,
                os.sep + 'predictions' + os.sep,
            )
            if not os.path.exists(pred_path):
                print(f'[WARN] no predictions file for {review_path}; skip',
                      file=sys.stderr)
                continue
            print(f'[INFO] Processing {tag} ({review_path})')
            pred_index = build_prediction_index(pred_path)
            results = process_review_file(
                review_path, pred_index, tokenizer,
                args.truncate_lengths, args.pass_k,
                extract_answer, math_equal,
            )
            for length in args.truncate_lengths:
                r = results[length]
                mean_acc = r['mean_acc']
                pk = r[f'pass@{args.pass_k}']
                n_g = r['num_groups']
                n_s = r['num_samples']
                tr = r['truncated']
                out.write(f'{args.model_path}\t{length}\t{tag}\tmean_acc\t{mean_acc:.4f}\n')
                out.write(f'{args.model_path}\t{length}\t{tag}\tpass@{args.pass_k}\t{pk:.4f}\n')
                print(f'  [{length}] {tag}  mean_acc={mean_acc:.4f}  pass@{args.pass_k}={pk:.4f}  '
                      f'(truncated={tr}/{n_s}, groups={n_g})')

                overall[length]['total_acc'] += mean_acc * n_s
                overall[length]['total_samples'] += n_s
                overall[length]['pass_groups'] += pk * n_g
                overall[length]['num_groups'] += n_g
                overall[length]['truncated'] += tr

        # micro-mean 汇总（跨 subset 按样本数加权）
        if len(review_files) > 1:
            for length in args.truncate_lengths:
                o = overall[length]
                if o['total_samples'] == 0 or o['num_groups'] == 0:
                    continue
                mean_acc = o['total_acc'] / o['total_samples']
                pk = o['pass_groups'] / o['num_groups']
                out.write(f'{args.model_path}\t{length}\tOVERALL\tmean_acc\t{mean_acc:.4f}\n')
                out.write(f'{args.model_path}\t{length}\tOVERALL\tpass@{args.pass_k}\t{pk:.4f}\n')
                print(f'  [{length}] OVERALL  mean_acc={mean_acc:.4f}  pass@{args.pass_k}={pk:.4f}  '
                      f'(truncated={o["truncated"]}/{o["total_samples"]})')

    print(f'[OK] Stats written to {args.stats_tsv}')


if __name__ == '__main__':
    main()
