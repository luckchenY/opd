#!/usr/bin/env python3
"""Download AI-ModelScope/OlympiadBench for local evalscope Native eval.

Default output:
  data/eval_datasets/olympiad_bench/
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'data' / 'eval_datasets' / 'olympiad_bench',
        help='写出数据集目录',
    )
    parser.add_argument(
        '--hub',
        choices=['modelscope', 'hf'],
        default='modelscope',
        help='数据源：modelscope 或 huggingface',
    )
    parser.add_argument(
        '--dataset_id',
        default='AI-ModelScope/OlympiadBench',
        help='数据集 id',
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.hub == 'modelscope':
        from modelscope import snapshot_download
        print(f'[modelscope] downloading {args.dataset_id} -> {args.out_dir}')
        snapshot_download(
            args.dataset_id,
            repo_type='dataset',
            local_dir=str(args.out_dir),
        )
    else:
        from huggingface_hub import snapshot_download
        print(f'[hf] downloading {args.dataset_id} -> {args.out_dir}')
        snapshot_download(
            args.dataset_id,
            repo_type='dataset',
            local_dir=str(args.out_dir),
        )

    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(str(args.out_dir))
    print(f'configs ({len(configs)}): {configs[:8]}...' if len(configs) > 8 else f'configs: {configs}')
    want = [
        'OE_TO_maths_en_COMP',
        'OE_TO_maths_zh_CEE',
        'OE_TO_maths_zh_COMP',
    ]
    for name in want:
        if name not in configs:
            print(f'[warn] missing config {name}')
            continue
        ds = load_dataset(str(args.out_dir), name, split='train')
        print(f'  {name}: {len(ds)} rows')

    print('Done. Point eval_olympiad.sh DATA_DIR to:', args.out_dir.resolve())


if __name__ == '__main__':
    main()
