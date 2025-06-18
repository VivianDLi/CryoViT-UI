"""Script for adding annotations to tomograms and setting up splits for training."""

import os
import shutil
from pathlib import Path
from PIL import Image
from typing import List

import mrcfile
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import KFold

#### Logging Setup ####

import logging

logger = logging.getLogger("cryovit.processing.annotations")
debug_logger = logging.getLogger("debug")


def generate_slices(
    src_dir: Path,
    dst_dir: Path,
    csv_file: Path,
):
    """Extract slices from tomograms using a .csv file with z-limits and slice indices and save them as .png files."""
    # Clear the destination directory if it exists
    if dst_dir.exists() and dst_dir.is_dir():
        try:
            shutil.rmtree(dst_dir)
        except OSError as e:
            logger.error(f"Error clearing destination directory {dst_dir}: {e}")
            debug_logger.error(
                f"Error clearing destination directory {dst_dir}: {e}", exc_info=True
            )
    os.makedirs(dst_dir, exist_ok=True)

    annotation_df = pd.read_csv(csv_file)
    for row in tqdm(
        annotation_df.itertuples(),
        desc="Extracting slices",
        total=len(annotation_df),
    ):
        # Get the tomogram name and z-limits from the DataFrame
        tomo_name = row[1]
        z_min, z_max = row[2:4]
        slices = row[4:]
        src_file = src_dir / tomo_name
        dst_file = dst_dir / tomo_name

        # Load the tomogram data
        try:
            with h5py.File(src_file, "r") as fh:
                data = fh["data"][()]  # d, w, h
        except OSError as e:
            logger.error(
                f"Error reading file {src_file}: {e}. Trying to load with mrcfile."
            )
            debug_logger.error(f"Error reading file {src_file}: {e}.", exc_info=True)
            data = None
        try:
            data = data if data is not None else mrcfile.read(src_file)
        except OSError as e:
            logger.error(f"Error reading file {src_file} with mrcfile: {e}. Skipping.")
            debug_logger.error(
                f"Error reading file {src_file} with mrcfile: {e}", exc_info=True
            )
            continue

        # Save slices as images
        for idx in slices:
            if z_min <= idx < z_max:
                out_path = dst_file.parent / f"{dst_file.stem}_{idx}.png"
                img = data[idx]
                # Normalize and convert to uint8
                img = ((img + 1) * 0.5 * 255 / np.max(img)).astype("uint8")
                img = Image.fromarray(img)
                img.save(out_path)
            else:
                logger.warning(
                    f"Slice index {idx} out of bounds for {tomo_name}. Skipping."
                )


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
        src_file = src_dir / tomo_name
        dst_file = dst_dir / tomo_name
        annot_file = annot_dir / tomo_name

        # Load the tomogram data
        try:
            with h5py.File(src_file, "r") as fh:
                data = fh["data"][()]  # d, w, h
        except OSError as e:
            logger.error(
                f"Error reading file {src_file}: {e}. Trying to load with mrcfile."
            )
            debug_logger.error(f"Error reading file {src_file}: {e}.", exc_info=True)
            data = None
        try:
            data = data if data is not None else mrcfile.read(src_file)
        except OSError as e:
            logger.error(f"Error reading file {src_file} with mrcfile: {e}. Skipping.")
            debug_logger.error(
                f"Error reading file {src_file} with mrcfile: {e}", exc_info=True
            )
            continue

        # Load annotations
        feature_labels = {feat: np.zeros_like(data, dtype=np.int8) for feat in features}
        for feat in features:
            feature_labels[feat][z_min:z_max] = -1

        # Add annotations to labels
        for idx in slices:
            annot_path = annot_file.parent / f"{annot_file.stem}_{idx}.png"
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
        with h5py.File(dst_file.with_suffix(".hdf"), "w") as fh:
            fh.create_dataset("data", data=data)
            for feat in features:
                fh.create_dataset(feat, data=feature_labels[feat], compression="gzip")


def generate_training_splits(
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
