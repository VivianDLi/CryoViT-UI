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
    src_dir: Path,
    dst_dir: Path = None,
    bin_size: int = 2,
    resize_image: Tuple[int, int] = None,
    normalize: bool = True,
    clip: bool = True,
) -> None:
    """Pre-process raw tomogram data.

    Args:
        src_dir (Path): Path to the directory with tomograms.
        dst_dir (Path, optional): Path to the directory for processed tomograms. Defaults to None. If None, the original tomograms are replaced.
        bin_size (int, optional): Number of tomogram slice to combine. Defaults to 2.
        normalize (bool, optional): Whether to normalize tomogram values. Defaults to True.
        clip (bool, optional): Whether to clip normalized values to +/- 3 std devs. Defaults to True.
    """
    files = list(p.resolve() for p in src_dir.glob("*") if p.suffix in tomogram_exts)
    logger.info(f"Found {len(files)} files in {src_dir}.")
    for file_name in tqdm(files, desc="Pre-processing tomograms"):
        # load tomogram
        try:  # try loading with h5py
            with File(file_name, "r") as fh:
                data = (
                    fh["MDF"]["images"]["0"]["image"][()]
                    if "MDF" in fh
                    else fh["data"][()]
                )
        except OSError as e:
            logger.error(f"Error reading file {file_name}: {e}")
            # try loading with mrcfile
            data = mrcfile.read(file_name)
        # pool tomogram
        if bin_size > 1:
            data = pool(data, bin_size)
        # resize tomogram
        if resize_image is not None:
            data = resize_data(data, resize_image)
        # normalize tomogram
        if normalize:
            data = normalize_data(data, clip)
        # save tomogram
        if dst_dir is None:
            dst_dir = file_name
        else:
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = dst_dir / file_name.name
        # Save processed data as hdf5 file
        with File(dst_path.parent / (dst_path.stem + ".hdf"), "w") as fh:
            if "data" in fh:
                del fh["data"]
            fh.create_dataset("data", data=data)
