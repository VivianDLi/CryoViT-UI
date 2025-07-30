"""Script to organize downloaded datasets into a standard format."""

from pathlib import Path
from typing import List, Optional
import shutil

import h5py
import mrcfile
import numpy as np
import torch

from cryovit_gui.config import tomogram_exts

#### Logging Setup ####

import logging

logger = logging.getLogger("cryovit.processing.dataset")
debug_logger = logging.getLogger("debug")

def load_data(src_file: Path) -> np.ndarray:
    try:
        with h5py.File(src_file, "r") as fh:
            if "data" in fh:
                data = fh["data"][()]  # d, h, w
            else:
                data = fh["MDF"]["images"]["0"]["image"][()]
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
        return -1
    return data

def preprocess_dataset(raw_dir: Path, target_dir: Path, bin_size: int, resize: Optional[List[int]], normalize: bool, clip: bool, scale: bool) -> None:
    tomogram_files = [
        str(f.relative_to(raw_dir)) for f in raw_dir.glob("**") if f.suffix in tomogram_exts
    ]
    pooler = torch.nn.AvgPool3d(kernel_size=bin_size, stride=bin_size, ceil_mode=True)
    
    for file in tomogram_files:
        src_path = raw_dir / file
        dst_path = target_dir / file
        dst_path = dst_path.with_suffix(".hdf")
        dst_path.mkdir(parents=True, exist_ok=True)
        log_file = target_dir / "resize.log"
        
        data = load_data(src_path)
        if data == -1:
            continue
        
        # Do pooling
        if bin_size != 1:
            data = np.expand_dims(data, [0, 1])
            data = torch.tensor(data)
            data = pooler(data)
            data = data.squeeze().numpy()
        
        # Do resizing
        if resize is not None:
            original_size = data.shape[-2:]
            # Check for original size
            if resize != original_size:
                data = np.expand_dims(data, axis=1) # D, 1, W, H
                data = torch.tensor(data)
                data = torch.nn.functional.interpolate(data, size=resize, mode="bilinear", align_corners=False)
                data = data.squeeze(1).numpy()
                
                # Record rescaling parameters
                scale_factor = (resize[0] / original_size[0], resize[1] / original_size[1])
                with open(log_file, "w+") as f:
                    f.write(f"{file}: {original_size} -> {resize} ({scale_factor})\n")
        
        # Do normalization, clipping, and scaling
        if normalize:
            data = (data - np.mean(data)) / np.std(data)
            if clip:
                data = np.clip(data, -3.0, 3.0)
                if scale:
                    data = data / 3.0
        

def get_all_tomogram_files(src_dir: Path, search_query: str) -> List[Path]:
    """Get all tomogram files in the source directory."""
    # Get all tomogram files in a Reconstructions folder
    tomogram_files = [
        f for f in src_dir.glob(search_query) if f.suffix in tomogram_exts
    ]
    return tomogram_files


def create_dataset(
    src_files: List[Path],
    data_dir: Path,
    sample_name: str,
) -> None:
    """Create a dataset structure for the tomograms.

    Args:
        src_files (List[Path]): The paths to the downloaded tomogram files to copy.
        data_dir (Path): The base directory where the dataset will be moved.
        sample_name (str): The name of the sample to create a subdirectory for.
        search_query (str): The search pattern to find tomogram files. Default is "**/Tomograms/**".
    """
    # Create the destination directory if it doesn't exist
    dst_dir = data_dir / sample_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy each tomogram file to the destination directory
    for tomogram_file in src_files:
        dst_file = dst_dir / tomogram_file.name
        shutil.copyfile(tomogram_file, dst_file)
