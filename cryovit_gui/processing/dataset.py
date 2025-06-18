"""Script to organize downloaded datasets into a standard format."""

from pathlib import Path
from typing import List
import shutil

from cryovit_gui.config import tomogram_exts


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
