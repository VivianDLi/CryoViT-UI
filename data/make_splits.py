"""Splits

Creates a master split file from all the annotation csvs.
Each tomogram is assigned 3 split IDs - LOO, 5-fold, and 10-fold.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import KFold
import argparse

SEED = 0

def split(csv_dir, df, sample, K):
    annotation_file = os.path.join(csv_dir, f"{sample}.csv")

    if not os.path.exists(annotation_file):
        raise RuntimeError(f"No annotations for {sample} were found")

    num_samples = df.shape[0]
    n_splits = num_samples if num_samples < K or K == 0 else K

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    X = [[0] for _ in range(num_samples)]
    split_id = [-1 for _ in range(num_samples)]

    for f, (_, test) in enumerate(kf.split(X)):
        for idx in test:
            split_id[idx] = f

    return split_id


def main():
    dfs = []

    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='sample name')
    parser.add_argument('csv_dir', type=str, help='path to directory with tomograms')
    args = parser.parse_args()

    annotation_file = os.path.join(args.csv_dir, f"{args.sample}.csv")
    df = pd.read_csv(annotation_file)
    split_id = split(args.csv_dir, df, args.sample, 10)
    df["split_id"] = split_id
    df["sample"] = args.sample
    
    dst_path = os.path.join(args.csv_dir, "splits.csv")
    df.to_csv(dst_path, mode='a', index=False, header=False)


if __name__ == "__main__":
    main()
