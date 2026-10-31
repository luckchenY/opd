#!/usr/bin/env python3
"""Download evalscope/amc_22-24 in the layout evalscope expects.

evalscope AMC adapter uses split_as_subset=True and loads via:
  datasets.load_dataset(<local_path>, split='amc23')

所以本地目录必须是带命名 split 的 HF/ModelScope 数据集仓库
（data/amc22-*.parquet 等），不能只是裸 amc23.jsonl
（jsonl 会被当成单一 train split）。

Default output:
  data/eval_datasets/amc_official/
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'data' / 'eval_datasets' / 'amc_official',
        help='写出官方布局数据集的目录',
    )
    parser.add_argument(
        '--hub',
        choices=['modelscope', 'hf'],
        default='modelscope',
        help='数据源：modelscope 或 huggingface',
    )
    parser.add_argument(
        '--dataset_id',
        default='evalscope/amc_22-24',
        help='数据集 id（两边同名）',
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

    # sanity check
    from datasets import load_dataset
    ds = load_dataset(str(args.out_dir))
    print('splits:', {k: len(v) for k, v in ds.items()})
    for name in ['amc22', 'amc23', 'amc24']:
        if name not in ds:
            raise SystemExit(f'missing split {name} under {args.out_dir}')
        print(f'  {name}: {len(ds[name])} rows')

    # cleanup empty legacy jsonl placeholder if present next door
    legacy = args.out_dir.parent / 'amc' / 'test.jsonl'
    if legacy.exists() and legacy.stat().st_size == 0:
        legacy.unlink()
        print(f'[cleanup] removed empty {legacy}')

    print('Done. Point eval_amc.sh DATA_DIR to:', args.out_dir.resolve())


if __name__ == '__main__':
    main()
