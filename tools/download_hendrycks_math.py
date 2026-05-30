"""
Download EleutherAI/hendrycks_math (all 7 subsets) to the project data dir.

By default downloads only the train split (used as the training pool).
Pass --split test  to download the test split instead (used as test_set.jsonl
source for Step 7 evaluation in process_reward_consistency_opd.py).

Usage:
    python tools/download_hendrycks_math.py               # train split
    python tools/download_hendrycks_math.py --split test  # test split
"""

import argparse
import os
from datasets import load_dataset

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to download (default: train)")
    args = parser.parse_args()

    subdir = "hendrycks_math" if args.split == "train" else "hendrycks_math_test"

    for subset in SUBSETS:
        save_path = os.path.join(DATA_DIR, subdir, subset)
        if os.path.exists(save_path):
            print(f"  [skip] {subset} already exists at {save_path}")
            continue
        print(f"  Downloading {subset} ({args.split}) …", flush=True)
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split=args.split)
        os.makedirs(save_path, exist_ok=True)
        ds.save_to_disk(save_path)
        print(f"  Saved {len(ds)} samples → {save_path}", flush=True)

    print(f"\nDone. Data at: {os.path.join(DATA_DIR, subdir)}", flush=True)


if __name__ == "__main__":
    main()
