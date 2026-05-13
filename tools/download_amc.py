#!/usr/bin/env python3
"""Download evalscope/amc_22-24 and save as per-subset JSONL.

Output layout matches what evalscope 的 amc adapter 期望（split_as_subset=True）：
  <out_dir>/amc22.jsonl
  <out_dir>/amc23.jsonl
  <out_dir>/amc24.jsonl

这样 eval_amc.sh 里直接用默认 subset_list=['amc22','amc23','amc24'] 即可。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'data' / 'eval_datasets' / 'amc',
        help='写出 <split>.jsonl 的目录',
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

    subsets = ['amc22', 'amc23', 'amc24']

    if args.hub == 'modelscope':
        from modelscope import MsDataset
        for sub in subsets:
            print(f'[modelscope] loading {args.dataset_id} split={sub} ...')
            ds = MsDataset.load(dataset_name=args.dataset_id, split=sub)
            try:
                ds = ds.to_hf_dataset()
            except AttributeError:
                pass
            out_path = args.out_dir / f'{sub}.jsonl'
            ds.to_json(str(out_path), orient='records', lines=True, force_ascii=False)
            print(f'  -> {out_path}  ({len(ds)} rows)')
    else:
        from datasets import load_dataset
        for sub in subsets:
            print(f'[hf] loading {args.dataset_id} split={sub} ...')
            ds = load_dataset(args.dataset_id, split=sub)
            out_path = args.out_dir / f'{sub}.jsonl'
            ds.to_json(str(out_path), orient='records', lines=True, force_ascii=False)
            print(f'  -> {out_path}  ({len(ds)} rows)')

    # 老的占位空文件不要了，避免 eval 时误读
    stale = args.out_dir / 'test.jsonl'
    if stale.exists() and stale.stat().st_size == 0:
        stale.unlink()
        print(f'[cleanup] removed empty {stale}')

    print('Done.')


if __name__ == '__main__':
    main()
