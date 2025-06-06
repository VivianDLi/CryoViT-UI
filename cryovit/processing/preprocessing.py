"""Script for pre-processing raw tomogram data based on configuration files."""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import mrcfile
from h5py import File
import numpy as np
import torch

from cryovit.config import tomogram_exts

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def pool(data: np.ndarray, bin_size: int) -> np.ndarray:
    """Pool tomogram data to reduce its size.

    Args:
        data (np.ndarray): The tomogram data to be pooled.
        bin_size (int): The size of the binning window."""
    pooler = torch.nn.AvgPool1d(kernel_size=bin_size, stride=bin_size, ceil_mode=True)

    # pooler works on last dimension and tomogram is (D, H, W)
    data = torch.Tensor(data).permute(1, 2, 0)  # (D, H, W) => (H, W, D)
    data = pooler(data).permute(2, 0, 1).numpy()  # (H, W, D) => (D, H, W)

    return data


def normalize_data(data: np.ndarray, clip: bool) -> np.ndarray:
    """Normalize tomogram data and optionally clip values to +/- 3 std devs.

    Args:
        data (np.ndarray): The tomogram data to be normalized.
        clip (bool): Whether to clip the normalized values to +/- 3 std devs."""
    # normalize the tomogram to the range [-1, 1]
    data = (data - np.mean(data)) / np.std(data)
    if clip:
        # clip to +/- 3 std devs.
        data = np.clip(data, -3.0, 3.0) / 3.0

    return data


def resize_data(data: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize tomogram data to a specified size.

    Args:
        data (np.ndarray): The tomogram data to be resized.
        size (Tuple[int, int]): The target size (height, width) for the tomogram."""
    # Add channel dimension for interpolation, treating slices as batch
    data = np.expand_dims(data, axis=1)  # (D, H, W) => (D, 1, H, W)
    torch_data = torch.from_numpy(data)
    # Resize the tomogram to the target size
    data = torch.nn.functional.interpolate(
        torch_data,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return data.squeeze(1).numpy()  # (D, 1, H, W) => (D, H, W)


def run_preprocess(
    src_file: Path,
    dst_file: Path = None,
    bin_size: int = 2,
    resize_image: Tuple[int, int] = None,
    normalize: bool = True,
    clip: bool = True,
) -> None:
    """Pre-process raw tomogram data.

    Args:
        src_file (Path): Path to the tomogram to process.
        dst_file (Path, optional): Path to where to save the processed tomogram. Defaults to None. If None, the original tomogram is replaced.
        bin_size (int, optional): Number of tomogram slice to combine. Defaults to 2.
        normalize (bool, optional): Whether to normalize tomogram values. Defaults to True.
        clip (bool, optional): Whether to clip normalized values to +/- 3 std devs. Defaults to True.
    """
    # load tomogram
    try:  # try loading with h5py
        with File(src_file, "r") as fh:
            data = (
                fh["MDF"]["images"]["0"]["image"][()] if "MDF" in fh else fh["data"][()]
            )
    except OSError as e:
        logger.error(f"Error reading file {src_file}: {e}")
        # try loading with mrcfile
        data = mrcfile.read(src_file)
    # pool tomogram
    if bin_size > 1:
        data = pool(data, bin_size)
    # resize tomogram
    if resize_image is not None:
        original_size = data.shape[1:3]  # (D, H, W) => (H, W)
        data = resize_data(data, resize_image)
        # Save original size, new size, and scale factor in a text file
        scale_factor = (
            resize_image[0] / original_size[0],
            resize_image[1] / original_size[1],
        )
    # normalize tomogram
    if normalize:
        data = normalize_data(data, clip)
    # Save preprocessing parameters in a text file
    with open(dst_file.parent / (dst_file.stem + "_preprocessing.txt"), "w+") as f:
        f.write(f"Normalize: {normalize}\n")
        f.write(f"Clip: {clip}\n")
        f.write(f"Bin size: {bin_size}\n")
        if resize_image is not None:
            f.write(f"Original size: {original_size}\n")
            f.write(f"Resize image: {resize_image}\n")
            f.write(f"Scale factor: {scale_factor}\n")
    # Save processed data as hdf5 file
    with File(dst_file, "w") as fh:
        if "data" in fh:
            del fh["data"]
        fh.create_dataset("data", data=data)
