"""Implementation of the multi sample data module."""

import pandas as pd

from cryovit.data_modules.base_datamodule import BaseDataModule


class InferenceDataModule(BaseDataModule):
    """Data module for CryoVIT experiments involving multiple samples."""

    def __init__(self, **kwargs) -> None:
        """Creates a data module for inferring on multiple samples."""
        super(InferenceDataModule, self).__init__(**kwargs)

    def train_df(self) -> pd.DataFrame:
        return self.record_df

    def val_df(self) -> pd.DataFrame:
        return self.record_df

    def test_df(self) -> pd.DataFrame:
        return self.record_df
