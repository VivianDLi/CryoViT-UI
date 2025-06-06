"""Script for adding annotations to tomograms and setting up splits for training."""

import os
import sys
from pathlib import Path
from PIL import Image
from typing import List
import logging

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import KFold

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def add_annotations(
    src_dir: Path,
    dst_dir: Path,
    annot_dir: Path,
    csv_file: Path,
    features: List[str],
) -> None:
    """Import annotations from a .png folder and add them to tomograms.

    Args:
        src_dir (Path): Path to the directory with tomograms.
        dst_dir (Path): Path to the directory where the tomograms with annotations will be saved.
        annot_dir (Path): Path to the directory with annotations.
        csv_file (Path): Path to a .csv file specifying z-limits and labeled slices.
        features (List[str]): List of feature names in order of decreasing annotation value to be added to the tomograms.
    """
    os.makedirs(dst_dir, exist_ok=True)
    annotation_df = pd.read_csv(csv_file)
    for i, row in tqdm(
        enumerate(annotation_df.itertuples()),
        desc="Inserting annotations",
        total=len(annotation_df),
    ):
        # Get the tomogram name and z-limits from the DataFrame
        tomo_name = row[1]
        z_min, z_max = row[2:4]
        slices = row[4:]
        dst_tomo = dst_dir / tomo_name

        # Load the tomogram data
        with h5py.File(src_dir / tomo_name, "r") as fh:
            data = fh["data"][()]  # d, w, h
        # Load annotations
        feature_labels = {feat: np.zeros_like(data, dtype=np.int8) for feat in features}
        for feat in features:
            feature_labels[feat][z_min:z_max] = -1

        # Add annotations to labels
        for idx in slices:
            annot_path = annot_dir / f"{tomo_name[:-4]}_{idx}.png"
            if annot_path.exists():
                annotation = np.asarray(Image.open(annot_path))
                for i, labels in enumerate(feature_labels.values()):
                    label = np.where(annotation == 254 - i, 1, 0)
                    label = ndimage.binary_fill_holes(label)
                    labels[idx] = label.astype(np.int8)
            else:
                logger.warning(f"Annotation file {annot_path} not found. Skipping.")
                continue

        # Save the tomogram with annotations
        with h5py.File(dst_tomo, "a") as fh:
            if "data" in fh:
                del fh["data"]
            fh.create_dataset("data", data=data, compression="gzip")
            for feat in features:
                if feat in fh:
                    del fh[feat]
                fh.create_dataset(feat, data=feature_labels[feat], compression="gzip")


def add_splits(
    splits_file: Path,
    csv_file: Path,
    sample: str = None,
    num_splits: int = 10,
    seed: int = 0,
) -> None:
    """Create splits for cross-validation.

    Args:
        splits_file (Path): Path to the file where the splits .csv file will be saved.
        csv_file (Path): Path to the .csv file with annotations.
        sample (str, optional): Sample name to be used in the splits. Defaults to None. If None, the sample name is extracted from the tomogram names.
        num_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
    """
    annotation_df = pd.read_csv(csv_file)
    n_samples = annotation_df.shape[0]
    n_splits = n_samples if n_samples < num_splits or num_splits == 0 else num_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = [[0] for _ in range(n_samples)]
    annotation_df["split_id"] = -1
    for fold_id, (_, test_ids) in enumerate(kf.split(X)):
        for idx in test_ids:
            annotation_df.at[idx, "split_id"] = fold_id
    annotation_df["sample"] = (
        annotation_df["tomo_name"][0].split("_")[1] if sample is None else sample
    )

    # Create splits file if it doesn't exist
    if not splits_file.exists():
        splits_df = pd.DataFrame(
            columns=["tomo_name", "z_min", "z_max", "split_id", "sample"]
        )
    else:
        splits_df = pd.read_csv(splits_file)
    # remove matching rows from the splits_df
    splits_df = splits_df[~splits_df["tomo_name"].isin(annotation_df["tomo_name"])]
    # append new rows to the splits_df
    splits_df = pd.concat([splits_df, annotation_df], ignore_index=True)
    splits_df.to_csv(splits_file, mode="w", index=False)


def generate_new_splits(
    splits_file: Path,
    dst_file: Path = None,
    num_splits: int = 10,
    seed: int = 0,
) -> None:
    """Generate new splits for cross-validation given old splits.

    Args:
        splits_file (Path): Path to the .csv file with existing splits.
        dst_file (Path, optional): Path to save the new splits. Defaults to None. If None, the old .csv will be replaced.
        num_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
    """
    splits_df = pd.read_csv(splits_file)
    n_samples = splits_df.shape[0]
    n_splits = n_samples if n_samples < num_splits or num_splits == 0 else num_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = [[0] for _ in range(n_samples)]
    splits_df["split_id"] = -1
    for fold_id, (_, test_ids) in enumerate(kf.split(X)):
        for idx in test_ids:
            splits_df.at[idx, "split_id"] = fold_id

    if dst_file is None:
        dst_file = splits_file
    splits_df.to_csv(dst_file, mode="w", index=False)
