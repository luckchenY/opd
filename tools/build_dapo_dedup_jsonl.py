"""
build_dapo_dedup_jsonl.py
=========================
Build a deduplicated DAPO-Math-17k JSONL for ms-swift consumption.

The HF release of DAPO-Math-17k contains ~1,791,700 rows = ~17,398 unique
problems duplicated ~103 times each. Loading it raw into `swift rlhf` would
cause each problem to be rolled out ~103 times per epoch — pure waste plus a
nasty drop in per-batch prompt diversity. We dedupe by user-prompt text and
emit a JSONL with a `messages` field that ms-swift's standard preprocessor can
consume directly (no `--columns` remap needed).

Output schema (one JSON per line):
    {"messages": [{"role": "user", "content": <problem>}]}

This mirrors the chat-format that `on_policy_distillation.sh` produces from
`--dataset 'hf::BytedTsinghua-SIA/DAPO-Math-17k' --columns '{"prompt":"messages"}'`,
just with the duplicates removed up front.

The extraction logic (`_extract_problem_text`) is copied from
`tools/process_reward_weighted_distill.py` so the two pipelines stay aligned.

Usage
-----
    python ms-swift/tools/build_dapo_dedup_jsonl.py \
        --out ms-swift/data/dapo_math_17k_dedup.jsonl
"""
import argparse
import json
from pathlib import Path


DEFAULT_SHARDS = [
    '/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/'
    'data/hf_datasets_cache/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0/'
    '65877096c24ffa7abc4e4fa5edb95cf3413a5674/'
    'dapo-math-17k-train-00000-of-00002.arrow',
    '/mnt/shared-storage-gpfs2/labutopia-shared/wanhaiyuan/yangchen/ms-swift/'
    'data/hf_datasets_cache/BytedTsinghua-SIA___dapo-math-17k/default/0.0.0/'
    '65877096c24ffa7abc4e4fa5edb95cf3413a5674/'
    'dapo-math-17k-train-00001-of-00002.arrow',
]


def extract_problem_text(row):
    """Best-effort extraction of the problem statement from a DAPO-style row.

    Mirrors `_extract_problem_text` in tools/process_reward_weighted_distill.py.
    """
    val = row.get('prompt')
    if isinstance(val, list):
        for m in val:
            if isinstance(m, dict) and m.get('role') == 'user':
                c = m.get('content')
                if isinstance(c, str) and c.strip():
                    return c
        if val and isinstance(val[-1], dict):
            c = val[-1].get('content')
            if isinstance(c, str) and c.strip():
                return c
    if isinstance(val, str) and val.strip():
        return val
    val = row.get('messages')
    if isinstance(val, list):
        for m in val:
            if isinstance(m, dict) and m.get('role') == 'user':
                c = m.get('content')
                if isinstance(c, str) and c.strip():
                    return c
    val = row.get('problem')
    if isinstance(val, str) and val.strip():
        return val
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', default=DEFAULT_SHARDS,
                    help='Absolute paths to DAPO arrow shards.')
    ap.add_argument('--out', required=True, help='Output JSONL path.')
    ap.add_argument('--max_unique', type=int, default=0,
                    help='If >0, stop after this many unique problems.')
    args = ap.parse_args()

    from pyarrow import ipc  # imported lazily so --help works without pyarrow

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_raw = 0
    n_kept = 0
    n_no_problem = 0

    with open(out_path, 'w') as fout:
        for shard in args.shards:
            print(f'Reading {shard} ...', flush=True)
            reader = ipc.open_stream(shard)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                cols = batch.schema.names
                for i in range(len(batch)):
                    row = {c: batch.column(c)[i].as_py() for c in cols}
                    problem = extract_problem_text(row)
                    if not problem:
                        n_no_problem += 1
                        continue
                    n_raw += 1
                    key = problem.strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(
                        {'messages': [{'role': 'user', 'content': problem}]},
                        ensure_ascii=False) + '\n')
                    n_kept += 1
                    if args.max_unique and n_kept >= args.max_unique:
                        print(f'Reached --max_unique={args.max_unique}, stopping.',
                              flush=True)
                        print(f'Done: {n_raw} raw -> {n_kept} unique '
                              f'(skipped {n_no_problem} rows without a problem). '
                              f'Wrote {out_path}', flush=True)
                        return

    print(f'Done: {n_raw} raw rows -> {n_kept} unique problems '
          f'(deduped {n_raw - n_kept}, skipped {n_no_problem} rows without a problem). '
          f'Wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
