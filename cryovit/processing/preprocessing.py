"""Script for pre-processing raw tomogram data based on configuration files."""

import os
import logging
from pathlib import Path

from h5py import File
import numpy as np
import torch


def pool(data: np.ndarray, bin_size: int) -> np.ndarray:
    pooler = torch.nn.AvgPool3d(kernel_size=bin_size, stride=bin_size, ceil_mode=True)

    # pooler expects (C, D, H, W) input and tomogram is (D, H, W)
    data = torch.Tensor(np.expand_dims(data, axis=0))
    # pool
    data = pooler(data).squeeze().numpy()

    return data


def normalize(data: np.ndarray, clip: bool) -> np.ndarray:
    # normalize the tomogram to the range [-1, 1]
    data = (data - np.mean(data)) / np.std(data)
    if clip:
        # clip to +/- 3 std devs.
        data = np.clip(data, -3.0, 3.0) / 3.0

    return data


def run_preprocess(
    src_dir: Path,
    dst_dir: Path = None,
    bin_size: int = 2,
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
    files = (
        p.resolve() for p in src_dir.glob("*") if p.suffix in {".rec", ".mrc", ".hdf"}
    )
    logging.info(f"Found {len(list(files))} files in {src_dir}.")
    for file_name in files:
        logging.debug(f"Processing {file_name}.")
        # load tomogram
        with File(file_name, "r") as fh:
            data = fh["MDF"]["images"]["0"]["image"][()]
        # pool tomogram
        if bin_size > 1:
            data = pool(data, bin_size)
        # normalize tomogram
        if normalize:
            data = normalize(data, clip)
        # save tomogram
        if dst_dir is None:
            dst_dir = file_name
        else:
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = dst_dir / file_name.name
        with File(dst_path, "a") as fh:
            fh.create_dataset("data", data=data)
