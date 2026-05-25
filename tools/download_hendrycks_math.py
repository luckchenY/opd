"""
Download EleutherAI/hendrycks_math (all 7 subsets) to the project data dir.

Usage:
    python tools/download_hendrycks_math.py
"""

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
    for subset in SUBSETS:
        save_path = os.path.join(DATA_DIR, "hendrycks_math", subset)
        if os.path.exists(save_path):
            print(f"  [skip] {subset} already exists at {save_path}")
            continue
        print(f"  Downloading {subset} …", flush=True)
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split="train")
        os.makedirs(save_path, exist_ok=True)
        ds.save_to_disk(save_path)
        print(f"  Saved {len(ds)} samples → {save_path}", flush=True)

    print(f"\nDone. Data at: {os.path.join(DATA_DIR, 'hendrycks_math')}", flush=True)


if __name__ == "__main__":
    main()
