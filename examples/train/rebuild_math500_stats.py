#!/usr/bin/env python3
"""从一个 math500 eval_results 目录里的 reviews JSONL 重建 stats.tsv

用法:
    python3 rebuild_math500_stats.py <eval_root_dir> [--pass_k 8] [--model_path ...]

说明:
    - 与 eval_math500.sh 中 extract_metrics 的逻辑一致：
        * 用 input 字段（题面）作为 group key 计算 pass@k（不靠 group_id）
        * 每档输出 5 个 Level + math_500_overall（500 题加权）+ math_500_macro（5 Level 算术平均）
    - 保留原 stats.tsv 中以 '#' 开头的元信息和列头行，只重写数据部分
    - 自动从子目录 max_new_tokens_<L> 中扫描所有 reviews JSONL
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


def collect_one_stage(out_dir):
    """扫一档（max_new_tokens_<L>/）下的所有 reviews JSONL，按 dataset 聚合。"""
    jsonl_files = sorted(glob.glob(
        os.path.join(out_dir, 'native', '*', 'reviews', '*', '*.jsonl')))
    dataset_results = defaultdict(
        lambda: {'correct': 0, 'total': 0, 'group_scores': defaultdict(list)})
    for jsonl_path in jsonl_files:
        base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inp = record.get('input')
                group_key = (inp if isinstance(inp, str)
                             else json.dumps(inp, sort_keys=True, ensure_ascii=False))
                acc = (record.get('sample_score', {}).get('score', {})
                       .get('value', {}).get('acc', 0))
                dataset_results[base_name]['total'] += 1
                if acc and acc > 0:
                    dataset_results[base_name]['correct'] += 1
                dataset_results[base_name]['group_scores'][group_key].append(acc)
    return dataset_results


def write_stage_rows(model_path, max_new_tokens, dataset_results, pass_k, lines_out):
    """把一档结果写入 lines_out（list of strings, 每行末尾不含 \n）。"""
    overall_correct = overall_total = 0
    overall_g_correct = overall_g_total = 0
    per_level_acc = []
    per_level_pak = []
    for ds_name, results in sorted(dataset_results.items()):
        total = results['total']
        correct = results['correct']
        mean_acc = correct / total if total > 0 else 0
        lines_out.append(
            f'{model_path}\t{max_new_tokens}\t{ds_name}\tmean_acc\t{mean_acc:.4f}')
        gc = gt = 0
        for _, scores in results['group_scores'].items():
            gt += 1
            if any(s and s > 0 for s in scores):
                gc += 1
        pak = gc / gt if gt > 0 else 0
        lines_out.append(
            f'{model_path}\t{max_new_tokens}\t{ds_name}\tmean_acc_pass@{pass_k}\t{pak:.4f}')
        overall_correct += correct
        overall_total += total
        overall_g_correct += gc
        overall_g_total += gt
        if total > 0:
            per_level_acc.append(mean_acc)
        if gt > 0:
            per_level_pak.append(pak)

    if overall_total > 0:
        v = overall_correct / overall_total
        lines_out.append(
            f'{model_path}\t{max_new_tokens}\tmath_500_overall\tmean_acc\t{v:.4f}')
    if overall_g_total > 0:
        v = overall_g_correct / overall_g_total
        lines_out.append(
            f'{model_path}\t{max_new_tokens}\tmath_500_overall\tmean_acc_pass@{pass_k}\t{v:.4f}')
    if per_level_acc:
        v = sum(per_level_acc) / len(per_level_acc)
        lines_out.append(
            f'{model_path}\t{max_new_tokens}\tmath_500_macro\tmean_acc\t{v:.4f}')
    if per_level_pak:
        v = sum(per_level_pak) / len(per_level_pak)
        lines_out.append(
            f'{model_path}\t{max_new_tokens}\tmath_500_macro\tmean_acc_pass@{pass_k}\t{v:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('eval_root', help='math500_<TIMESTAMP> 目录')
    ap.add_argument('--pass_k', type=int, default=None,
                    help='pass@k 的 k；默认从原 stats.tsv 的 # repeats 行解析')
    ap.add_argument('--model_path', default=None,
                    help='用于第 1 列；默认从原 stats.tsv 的 # model_path 行解析')
    ap.add_argument('--out', default=None,
                    help='输出路径；默认覆盖 <eval_root>/stats.tsv')
    args = ap.parse_args()

    eval_root = os.path.abspath(args.eval_root)
    if not os.path.isdir(eval_root):
        print(f'[error] not a directory: {eval_root}', file=sys.stderr)
        sys.exit(1)
    stats_tsv = os.path.join(eval_root, 'stats.tsv')

    header_lines = []
    parsed_model_path = None
    parsed_pass_k = None
    if os.path.isfile(stats_tsv):
        with open(stats_tsv, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#') or line.startswith('model_path\t'):
                    header_lines.append(line)
                    if line.startswith('# model_path\t'):
                        parsed_model_path = line.split('\t', 2)[1]
                    if line.startswith('# repeats\t'):
                        m = re.match(r'# repeats\t(\d+)', line)
                        if m:
                            parsed_pass_k = int(m.group(1))
                else:
                    break

    pass_k = args.pass_k if args.pass_k is not None else (parsed_pass_k or 8)
    model_path = args.model_path or parsed_model_path or ''
    if not model_path:
        print('[warn] model_path 未知；请用 --model_path 显式指定', file=sys.stderr)
        sys.exit(2)

    stage_dirs = sorted(
        [d for d in glob.glob(os.path.join(eval_root, 'max_new_tokens_*'))
         if os.path.isdir(d)],
        key=lambda p: int(re.search(r'max_new_tokens_(\d+)', p).group(1)))
    if not stage_dirs:
        print(f'[error] 在 {eval_root} 下没找到 max_new_tokens_* 目录', file=sys.stderr)
        sys.exit(3)

    if not header_lines:
        header_lines = [
            f'# model_path\t{model_path}',
            f'# bench\tmath_500',
            f'# strategy\tsweep_max_new_tokens',
            f'# repeats\t{pass_k}',
            'model_path\tmax_new_tokens\tdataset\tmetric\tvalue',
        ]

    data_rows = []
    for sd in stage_dirs:
        L = int(re.search(r'max_new_tokens_(\d+)', sd).group(1))
        ds_results = collect_one_stage(sd)
        if not ds_results:
            print(f'[warn] {sd} 下没有 reviews JSONL，跳过', file=sys.stderr)
            continue
        write_stage_rows(model_path, str(L), ds_results, pass_k, data_rows)

    out_path = args.out or stats_tsv
    with open(out_path, 'w', encoding='utf-8') as f:
        for ln in header_lines:
            f.write(ln + '\n')
        for ln in data_rows:
            f.write(ln + '\n')
    print(f'[ok] wrote {out_path}  ({len(data_rows)} data rows)')

    # 顺手把 overall / macro 速览打到 stdout
    print('\n--- summary ---')
    for ln in data_rows:
        if 'math_500_overall' in ln or 'math_500_macro' in ln:
            cols = ln.split('\t')
            print(f'  max_new_tokens={cols[1]:>6}  {cols[2]:<18}  {cols[3]:<22} = {cols[4]}')


if __name__ == '__main__':
    main()
