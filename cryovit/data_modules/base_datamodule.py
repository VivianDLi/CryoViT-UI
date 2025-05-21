"""Module defining base data loading functionality for CryoVIT experiments."""

from pathlib import Path
from typing import Callable
from typing import Dict

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cryovit.config import tomogram_exts
from cryovit.datasets import TomoDataset


class BaseDataModule(LightningDataModule):
    """Base module defining common functions for creating data loaders."""

    def __init__(
        self,
        split_file: Path | None,
        dataloader_fn: Callable,
        dataset_params: Dict = {},
    ) -> None:
        """Initializes the BaseDataModule with dataset parameters, a dataloader function, and a path to the split file.

        Args:
            split_file (Union[Path, None]): The path to the CSV file containing data splits. None only to use full directory in inference.
            dataloader_fn (Callable): Function to create a DataLoader from a dataset.
            dataset_params (Dict, optional): Dictionary of parameters to pass to the dataset class..
        """
        super().__init__()
        self.dataset_params = dataset_params
        self.dataloader_fn = dataloader_fn
        if split_file and str(split_file) != "None":
            self._load_splits(split_file)
        else:
            self._create_splits(self.dataset_params["data_root"])

    def _load_splits(self, split_file: Path) -> None:
        if not split_file.exists():
            raise RuntimeError(f"split file {split_file} not found")

        self.record_df = pd.read_csv(split_file)

    def _create_splits(self, data_dir: Path) -> None:
        # Assumes the data_root is the sample directory
        dataset_files = [f for f in data_dir.glob("*") if f.suffix in tomogram_exts]
        self.record_df = pd.DataFrame(
            {
                "tomo_name": [f.name for f in dataset_files],
                "sample": [f.parent.name for f in dataset_files],
            }
        )
        self.dataset_params["data_root"] = data_dir.parent

    def train_dataloader(self) -> DataLoader:
        """Creates DataLoader for training data.

        Returns:
            DataLoader: A DataLoader instance for training data.
        """
        dataset = TomoDataset(
            records=self.train_df(),
            train=True,
            **self.dataset_params,
        )

        return self.dataloader_fn(dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Creates DataLoader for validation data.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """
        dataset = TomoDataset(
            records=self.val_df(),
            train=False,
            **self.dataset_params,
        )
        return self.dataloader_fn(dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Creates DataLoader for testing data.

        Returns:
            DataLoader: A DataLoader instance for testing data.
        """
        dataset = TomoDataset(
            records=self.test_df(),
            train=False,
            **self.dataset_params,
        )
        return self.dataloader_fn(dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Creates DataLoader for inference data.

        Returns:
            DataLoader: A DataLoader instance for inference data.
        """
        dataset = TomoDataset(
            records=self.record_df,
            train=False,
            **self.dataset_params,
        )
        return self.dataloader_fn(dataset, shuffle=False)

    def train_df(self) -> pd.DataFrame:
        """Abstract method to generate train splits."""
        raise NotImplementedError

    def val_df(self) -> pd.DataFrame:
        """Abstract method to generate validation splits."""
        raise NotImplementedError

    def test_df(self) -> pd.DataFrame:
        """Abstract method to generate test splits."""
        raise NotImplementedError
