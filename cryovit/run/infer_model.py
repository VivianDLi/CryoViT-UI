"""Script for segmenting with pre-trained models based on configuration files."""

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

from cryovit.config import ModelArch, InferModelConfig

torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


class TomoPredictionWriter(Callback):
    """Callback to add label predictions to tomograms."""

    def __init__(self, results_dir: Path, label_key: str) -> None:
        """Creates a callback to save predictions on the test data.

        Args:
            results_dir (Path): Directory in which the predictions should be saved.
            label_key (str): Key for the label in the dataset.
        """
        self.results_dir = results_dir
        self.label_key = label_key
        os.makedirs(self.results_dir, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handles the end of a batch to write predictions to tomograms.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (BaseModel): The model instance.
            outputs (torch.Tensor): Predictions from the model.
            batch (Any): The batch of data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        data = batch["data"].cpu().numpy()
        outputs = outputs.cpu().numpy().astype(np.float32)
        preds = np.where(outputs > 0.5, 1, 0).astype(np.uint8)  # binary classification
        # Save predictions to disk
        with h5py.File(self.results_dir / batch["tomo_name"], "a") as fh:
            if "data" in fh:
                del fh["data"]
            fh.create_dataset("data", data=data)
            pred_key = "predictions/" + self.label_key
            if pred_key in fh:
                del fh[pred_key]
            fh.create_dataset(pred_key, data=preds, compression="gzip")


def build_datamodules(cfg: InferModelConfig) -> List[LightningDataModule]:
    """Creates a data module for the model based on the configuration.

    Args:
        index (int): Index of the current run.
        cfg (TrainModelConfig): Configuration object specifying the dataset and model settings.

    Returns:
        LightningDataModule: A data module instance configured as specified.
    """
    dataloaders = []
    for model_config in cfg.models:
        match model_config.model_type:
            case "CRYOVIT":
                input_key = "dino_features"
            case _:
                input_key = "data"

        dataset_params = {
            "input_key": input_key,
            "label_key": model_config.label_key,
            "data_root": cfg.exp_paths.tomo_dir,
            "aux_keys": cfg.aux_keys,
        }

        dataloaders.append(
            instantiate(cfg.dataset)(
                split_file=cfg.exp_paths.split_file,
                dataloader_fn=instantiate(cfg.dataloader),
                dataset_params=dataset_params,
            )
        )
    return dataloaders


def run_trainer(cfg: InferModelConfig) -> None:
    """Sets up and runs the inference process using the specified configuration.

    Args:
        cfg (TrainModelConfig): Configuration object containing all settings for the inference process.
    """
    datamodules = build_datamodules(cfg)
    pred_writers = [
        TomoPredictionWriter(
            cfg.exp_paths.exp_dir,
            m_cfg.name + "_" + m_cfg.label_key,
        )
        for m_cfg in cfg.models
    ]
    model_configs = instantiate(cfg.models)
    for model_config, datamodule, pred_writer in zip(
        model_configs, datamodules, pred_writers
    ):
        model = model_config.model
        # Load the model weights
        model.load_state_dict(torch.load(model_config.model_weights))

        # Add tomogram writer callback
        trainer = instantiate(cfg.trainer)
        trainer.callbacks.append(pred_writer)

        trainer.predict(model, datamodule, return_predictions=False)
