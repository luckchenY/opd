#!/usr/bin/env python3
"""LiveCodeBench：从一次最长的 rollout 结果中，按不同 token 长度截断后重跑测试用例打分。

与数学题不同，代码题的重打分必须重新执行测试用例（不能只做字符串判等），
所以这个脚本会：
  1. 按 group_id 对应回原始 test.jsonl 取出 evaluation_sample（public + private 测试）
  2. 对每个 truncate 长度：对所有 prediction 按 token 截断 → extract_code_generation 抽代码
     → 调用 live_code_bench.evaluate_utils.codegen_metrics 跑测试 → 得 pass@1 / pass@k

用法（在 eval_livecodebench.sh 最后一步调用，与 truncate_score.py 对称）：
  python3 truncate_score_code.py \\
      --eval_output_dir <max_len_dir> \\
      --stats_tsv <stats.tsv> \\
      --model_path <model> \\
      --pass_k <k> \\
      --truncate_lengths 4096 8192 16384 32768 \\
      --data_jsonl data/eval_datasets/livecodebench_v6/test.jsonl

group_id 对应关系的前提：RemoteDataLoader 按 jsonl 原顺序加载 → 复制 repeats 次
→ （默认 start_date/end_date=None，filter 是 no-op）→ reindex, group_id = i // repeats。
所以 group_id == test.jsonl 里的行号（0-based）。
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def truncate_by_tokens(text, tokenizer, max_tokens):
    if not text:
        return ''
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def build_eval_samples_from_jsonl(data_jsonl):
    """读原始 test.jsonl，按行序返回 evaluation_sample 列表。"""
    from evalscope.benchmarks.live_code_bench.load_utils import transform
    eval_samples = []
    with open(data_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = transform(rec)
            eval_samples.append(rec['evaluation_sample'])
    return eval_samples


def score_length(group_to_preds, eval_samples, pass_k, timeout, num_process, debug):
    """对一批已截断的预测跑测试，返回 {'pass@1': x, f'pass@{pass_k}': y, 'num_problems': n}。"""
    from evalscope.benchmarks.live_code_bench.evaluate_utils import codegen_metrics

    problem_ids = sorted(group_to_preds.keys())
    references = [{'input_output': eval_samples[g]} for g in problem_ids]
    predictions = [group_to_preds[g] for g in problem_ids]  # list[list[code_str]]

    k_list = sorted({1, pass_k})
    try:
        metrics, _results, _meta = codegen_metrics(
            references, predictions,
            k_list=k_list,
            num_process_evaluate=num_process,
            timeout=timeout,
            debug=debug,
        )
    except Exception as e:
        print(f'[ERROR] codegen_metrics failed: {e}', file=sys.stderr)
        return {'pass@1': 0.0, f'pass@{pass_k}': 0.0, 'num_problems': len(problem_ids)}

    out = {
        'pass@1': float(metrics.get('pass@1', 0.0)) / 100.0,
        f'pass@{pass_k}': float(metrics.get(f'pass@{pass_k}', metrics.get('pass@1', 0.0))) / 100.0,
        'num_problems': len(problem_ids),
    }
    return out


def process_review_file(review_path, tokenizer, truncate_lengths, pass_k,
                         eval_samples, timeout, num_process, debug):
    from evalscope.benchmarks.live_code_bench.extract_utils import extract_code_generation

    samples = []
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    results = {}
    for length in truncate_lengths:
        group_to_preds = defaultdict(list)
        for s in samples:
            group_id = s.get('group_id') or s.get('sample_score', {}).get('group_id')
            if group_id is None:
                group_id = s.get('index')
            if group_id is None or group_id >= len(eval_samples):
                continue
            pred_text = (
                s.get('sample_score', {})
                 .get('score', {})
                 .get('prediction', '')
            )
            truncated = truncate_by_tokens(pred_text, tokenizer, length)
            code = extract_code_generation(truncated)
            group_to_preds[group_id].append(code)

        print(f'  [{length}] scoring {len(group_to_preds)} problems ...', flush=True)
        results[length] = score_length(
            group_to_preds, eval_samples, pass_k, timeout, num_process, debug,
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_output_dir', required=True)
    parser.add_argument('--stats_tsv', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--pass_k', type=int, default=1)
    parser.add_argument('--truncate_lengths', type=int, nargs='+', required=True)
    parser.add_argument('--data_jsonl', required=True, help='原始 livecodebench test.jsonl')
    parser.add_argument('--review_timeout', type=int, default=6)
    parser.add_argument('--num_process', type=int, default=16)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    review_files = sorted(glob.glob(
        os.path.join(args.eval_output_dir, 'native', '*', 'reviews', '*', 'live_code_bench_*.jsonl')
    ))
    if not review_files:
        print(f'[ERROR] No review files in {args.eval_output_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'[INFO] Loading eval samples from {args.data_jsonl}')
    eval_samples = build_eval_samples_from_jsonl(args.data_jsonl)
    print(f'[INFO] Loaded {len(eval_samples)} problems')

    print(f'[INFO] Loading tokenizer from {args.model_path}')
    tokenizer = load_tokenizer(args.model_path)

    with open(args.stats_tsv, 'a', encoding='utf-8') as out:
        for review_path in review_files:
            tag = os.path.basename(review_path).replace('.jsonl', '')
            print(f'[INFO] Processing {tag} ({review_path})')
            results = process_review_file(
                review_path, tokenizer, args.truncate_lengths, args.pass_k,
                eval_samples, args.review_timeout, args.num_process, args.debug,
            )
            for length in args.truncate_lengths:
                r = results[length]
                for metric_name in ['pass@1', f'pass@{args.pass_k}']:
                    val = r[metric_name]
                    out.write(f'{args.model_path}\t{length}\t{tag}\t{metric_name}\t{val:.4f}\n')
                    print(f'  [{length}] {tag} {metric_name} = {val:.4f}  (problems={r["num_problems"]})')

    print(f'[OK] Stats written to {args.stats_tsv}')


if __name__ == '__main__':
    main()
