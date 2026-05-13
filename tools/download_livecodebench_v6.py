#!/usr/bin/env python3
"""Download LiveCodeBench code_generation_lite v6 from Hugging Face and save as JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "eval_datasets" / "livecodebench_v6",
        help="Directory to write <split>.jsonl files",
    )
    parser.add_argument(
        "--version_tag",
        default="v6",
        help='HF dataset version_tag, e.g. "v6", "release_v5", "v4_v5"',
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("livecodebench/code_generation_lite", version_tag=args.version_tag)
    print("Loaded:", ds)
    for split_name, split in ds.items():
        out_path = args.out_dir / f"{split_name}.jsonl"
        split.to_json(str(out_path), orient="records", lines=True, force_ascii=False)
        print(f"Wrote {len(split)} rows -> {out_path}")

    sample = next(iter(ds.values()))[0]
    print("\nExample keys:", list(sample.keys()))


if __name__ == "__main__":
    main()
