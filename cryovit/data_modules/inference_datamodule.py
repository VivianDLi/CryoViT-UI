"""Implementation of the multi sample data module."""

from pathlib import Path

import pandas as pd

from cryovit.config import tomogram_exts
from cryovit.data_modules.base_datamodule import BaseDataModule


class InferenceDataModule(BaseDataModule):
    """Data module for CryoVIT experiments involving multiple samples."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Creates a data module for multiple samples.

        Args:
            sample (List[Sample]): List of samples used for training.
            split_id (Optional[int]): Optional split ID to be excluded from training and used for eval.
            test_samples (List[Sample]): List of samples used for testing.
        """
        super(InferenceDataModule, self).__init__(**kwargs)

    def _load_splits(self, split_file: Path | None) -> None:
        """Overrides the base class method to load all files if a split file is not provided."""
        if split_file is None or not split_file.exists():
            self.record_df = pd.DataFrame(
                {
                    "tomo_name": [
                        f.name
                        for f in self.dataset_params["data_dir"].glob("*")
                        if f.suffix in tomogram_exts
                    ]
                }
            )
        else:
            self.record_df = pd.read_csv(split_file)

    def train_df(self) -> pd.DataFrame:
        return self.record_df

    def val_df(self) -> pd.DataFrame:
        return self.record_df

    def test_df(self) -> pd.DataFrame:
        return self.record_df
