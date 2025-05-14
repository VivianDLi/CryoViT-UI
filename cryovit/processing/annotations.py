"""Script for adding annotations to tomograms and setting up splits for training."""

import os
import sys
from pathlib import Path
from PIL import Image
from typing import List
import logging

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
    callback_fn: callable = None,
) -> None:
    """Import annotations from a .png folder and add them to tomograms.

    Args:
        src_dir (Path): Path to the directory with tomograms.
        dst_dir (Path): Path to the directory where the tomograms with annotations will be saved.
        annot_dir (Path): Path to the directory with annotations.
        csv_file (Path): Path to a .csv file specifying z-limits and labeled slices.
        features (List[str]): List of feature names in order of decreasing annotation value to be added to the tomograms.
        callback_fn (callable, optional): Callback function to be called after each tomogram is processed. Takes as arguments the current index and total index. Defaults to None.
    """
    os.makedirs(dst_dir, exist_ok=True)
    annotation_df = pd.read_csv(csv_file)
    for i, row in enumerate(annotation_df.itertuples()):
        # Get the tomogram name and z-limits from the DataFrame
        tomo_name = row[1]
        z_min, z_max = row[2:4]
        slices = row[4:]

        # Create a new directory for the tomogram
        dst_tomo_dir = dst_dir / tomo_name
        dst_tomo_dir.mkdir(parents=True, exist_ok=True)

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

        # Save the tomogram with annotations
        with h5py.File(dst_tomo_dir / tomo_name, "w") as fh:
            if "data" in fh:
                del fh["data"]
            fh.create_dataset("data", data=data, compression="gzip")
            for feat in features:
                if feat in fh:
                    del fh[feat]
                fh.create_dataset(feat, data=feature_labels[feat], compression="gzip")
        callback_fn(i, len(annotation_df)) if callback_fn else None


def add_splits(
    dst_dir: Path,
    csv_file: Path,
    sample: str = None,
    num_splits: int = 10,
    seed: int = 0,
    callback_fn: callable = None,
) -> None:
    """Create splits for cross-validation.

    Args:
        dst_dir (Path): Path to the directory where the splits .csv will be saved.
        csv_file (Path): Path to the .csv file with annotations.
        sample (str, optional): Sample name to be used in the splits. Defaults to None. If None, the sample name is extracted from the tomogram names.
        num_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        callback_fn (callable, optional): Callback function to be called after each tomogram is processed. Takes as arguments the current index and total index. Defaults to None.
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
        callback_fn(fold_id, n_splits) if callback_fn else None
    annotation_df["sample"] = (
        annotation_df["tomo_name"].str.split("_").str[1] if sample is None else sample
    )

    annotation_df.to_csv(dst_dir / "splits.csv", mode="a", index=False, header=False)


def generate_new_splits(
    splits_file: Path,
    dst_file: Path = None,
    num_splits: int = 10,
    seed: int = 0,
    callback_fn: callable = None,
) -> None:
    """Generate new splits for cross-validation given old splits.

    Args:
        splits_file (Path): Path to the .csv file with existing splits.
        dst_file (Path, optional): Path to save the new splits. Defaults to None. If None, the old .csv will be replaced.
        num_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        callback_fn (callable, optional): Callback function to be called after each tomogram is processed. Takes as arguments the current index and total index. Defaults to None.
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
        callback_fn(fold_id, n_splits) if callback_fn else None

    if dst_file is None:
        dst_file = splits_file
    splits_df.to_csv(dst_file, mode="w", index=False, header=False)
