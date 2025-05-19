"""Script for pre-processing raw tomogram data based on configuration files."""

import os
import sys
import logging
from pathlib import Path

from tqdm import tqdm
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
    pooler = torch.nn.AvgPool3d(kernel_size=bin_size, stride=bin_size, ceil_mode=True)

    # pooler expects (C, D, H, W) input and tomogram is (D, H, W)
    data = torch.Tensor(np.expand_dims(data, axis=0))
    # pool
    data = pooler(data).squeeze().numpy()

    return data


def normalize_data(data: np.ndarray, clip: bool) -> np.ndarray:
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
    files = list(p.resolve() for p in src_dir.glob("*") if p.suffix in tomogram_exts)
    logger.info(f"Found {len(files)} files in {src_dir}.")
    for file_name in tqdm(files, desc="Pre-processing tomograms"):
        logger.debug(f"Processing {file_name}.")
        # load tomogram
        with File(file_name, "r") as fh:
            data = (
                fh["MDF"]["images"]["0"]["image"][()] if "MDF" in fh else fh["data"][()]
            )
        # pool tomogram
        if bin_size > 1:
            data = pool(data, bin_size)
        # normalize tomogram
        if normalize:
            data = normalize_data(data, clip)
        # save tomogram
        if dst_dir is None:
            dst_dir = file_name
        else:
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = dst_dir / file_name.name
        with File(dst_path, "a") as fh:
            if "data" in fh:
                del fh["data"]
            fh.create_dataset("data", data=data)
