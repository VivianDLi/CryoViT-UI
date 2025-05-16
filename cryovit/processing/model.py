"""Script for training, evaluating, and running inference with a model for GUI use."""

import os
import sys
import platform
from pathlib import Path
import shutil
import json
from typing import Any, List, Optional, Tuple
import logging

from tqdm import tqdm
import h5py
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
import pandas as pd
import torch
from torch.utils.data import DataLoader as torchDataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import BasePredictionWriter

from cryovit.datasets import TomoDataset, VITDataset
from cryovit.models.base_model import BaseModel
from cryovit.run.dino_features import (
    dino_model,
    dino_features,
    dataloader_params,
)
from cryovit.config import (
    DataLoader,
    InterfaceModelConfig,
    CryoVIT,
    UNet3D,
    Trainer,
    TrainerFit,
    MultiSample,
    ModelArch,
)

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class TomoPredictionWriter(BasePredictionWriter):
    """Callback to add label predictions to tomograms."""

    def __init__(self, results_dir: Path, label_key: str) -> None:
        self.results_dir = results_dir
        self.label_key = label_key
        os.makedirs(self.results_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: torch.Tensor,
        batch_indices: Optional[List[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handles the end of a batch to write predictions to tomograms.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (BaseModel): The model instance.
            prediction (torch.Tensor): Predictions from the model.
            batch_indices (Optional[List[int]]): Indices of the batch.
            batch (Any): The batch of data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        # Save predictions to disk
        with h5py.File(self.results_dir / batch["tomo_name"], "w+") as fh:
            if "data" in fh:
                del fh["data"]
            fh.create_dataset("data", data=batch["data"])
            pred_key = "predictions/" + self.label_key
            if pred_key in fh:
                del fh[pred_key]
            fh.create_dataset(pred_key, data=prediction, compression="gzip")


def get_available_models(model_dir: Path) -> List[str]:
    """Get a list of available models in the specified directory.

    Args:
        model_dir (Path): Directory containing the model weight files.

    Returns:
        List[str]: List of model names (without file extension) found in the directory.
    """
    return [f.stem for f in model_dir.glob("*.pt") if f.is_file()]


def get_model_configs(
    model_dir: Path, model_names: List[str]
) -> List[InterfaceModelConfig]:
    """Get model information for a list of model names.

    Args:
        model_dir (Path): Directory containing the model configuration files.
        model_names (List[str]): List of model names to retrieve configurations for.

    Returns:
        List[InterfaceModelConfig]: List of model configurations for the specified model names.
    """
    configs = []
    for model_name in model_names:
        with open(model_dir / f"{model_name}.json", "r") as f:
            model_config = InterfaceModelConfig(**json.load(f))
        configs.append(model_config)
    return configs


def save_model(
    model_config: InterfaceModelConfig,
    model_dir: Path,
    model_name: str = None,
    model: BaseModel = None,
) -> None:
    model_name = model_name if model_name else model_config.model_name
    if model:
        # Save model weights
        torch.save(model.state_dict(), model_dir / f"{model_name}.pt")
        torch.cuda.empty_cache()
    # Save model configuration
    with open(model_dir / f"{model_name}.json", "w") as f:
        json.dump(model_config, f, default=lambda x: x.value)


def load_model(
    model_dir: Path, model_name: str
) -> Tuple[BaseModel, InterfaceModelConfig]:
    # Load the model configuration
    with open(model_dir / f"{model_name}.json", "r") as f:
        model_config = InterfaceModelConfig(**json.load(f))
    # Create model instance
    model = load_base_model(model_config)
    # Load model weights
    try:
        model_weights = model_dir / f"{model_name}.pt"
        model.load_state_dict(torch.load(model_weights))
    except RuntimeError as e:
        logging.error(f"Error loading model weights from {model_weights}: {e}")

    return model, model_config


def load_base_model(model_config: InterfaceModelConfig) -> BaseModel:
    """Load a base model based on the provided configuration.

    Args:
        model_config (InterfaceModelConfig): The model configuration.

    Returns:
        BaseModel: The loaded model.
    """
    match model_config.model_type:
        case ModelArch.CRYOVIT:
            model_config = CryoVIT(**model_config.model_params)
        case ModelArch.UNET3D:
            model_config = UNet3D(**model_config.model_params)

        case _:
            logger.error(f"Unknown model type: {model_config.model_type}")
    return instantiate(model_config)


def get_dino_features(
    dino_dir: Path,
    data_dir: Path,
    batch_size: int,
    dst_dir: Path = None,
    csv_file: Path = None,
):
    """Compute DINOv2 features for a set of tomograms.

    Args:
        dino_dir (Path): Path to the directory where DINOv2 is/will be saved.
        data_dir (Path): Path to the directory containing the tomograms.
        batch_size (int): Batch size for processing the tomograms.
        dst_dir (Path, optional): Path to the directory to save tomograms with DINOv2 features. Defaults to None. If None, features will be added to the original tomograms.
        csv_file (Path, optional): Path to the .csv file for specifying which tomograms to compute. Defaults to None. If None, all tomograms in the directory will be used.
    """
    torch.set_float32_matmul_precision("high")  # ensures tensor cores are used

    if dst_dir is None:
        # overwrite original tomograms
        dst_dir = data_dir
        replace = True
    else:
        # copy tomograms to a new directory
        os.makedirs(dst_dir, exist_ok=True)
        replace = False

    # Setup the dataset and dataloader
    if csv_file:
        # read files from .csv
        records = pd.read_csv(csv_file)["tomo_name"]
    else:
        # use all files in the directory
        records = pd.Series(
            [
                f.name
                for f in data_dir.glob("*")
                if f.suffix in {".rec", ".mrc", ".hdf"}
            ]
        )
    dataset = VITDataset(records, root=data_dir)
    dataloader = torchDataLoader(dataset, **dataloader_params)
    # Load the DINOv2 model
    torch.hub.set_dir(dino_dir)
    model = torch.hub.load(*dino_model, verbose=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    # Get Dinov2 features
    print(len(dataloader))
    for i, x in tqdm(
        enumerate(dataloader),
        desc=f"Computing features for {data_dir.name}",
        total=len(dataloader),
    ):
        features = dino_features(x, model, batch_size)
        if not replace:
            # copy tomograms to a new directory
            shutil.copy(data_dir / records[i], dst_dir / records[i])
        with h5py.File(dst_dir / records[i], "a") as fh:
            if "dino_features" in fh:
                del fh["dino_features"]
            fh.create_dataset("dino_features", data=features)


def train_model(
    model: BaseModel,
    model_config: InterfaceModelConfig,
    trainer_config: TrainerFit,
    data_dir: Path,
    split_file: Path,
    batch_size: int = None,
    split_id: int = 0,
    seed: int = 0,
) -> float:
    """Train a specified model on a set of tomograms.

    Args:
        model (BaseModel): The model to train.
        model_config (InterfaceModelConfig): Information about the model.
        trainer_config (TrainerFit): The training configuration (i.e., gpu, devices, bit-precision, epochs, etc.).
        data_dir (Path): Path to the directory containing the tomograms to train on.
        split_file (Path): Path to the .csv file specifying which tomograms and splits to train on.
        batch_size (int, optional): The size of batches. Defaults to None (i.e., 1).
        split_id (int, optional): The split to use for validation. Defaults to 0.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.

    Returns:
        float: _description_
    """
    seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")  # ensures tensor cores are used
    # Setup the dataloader
    match model_config.model_type:
        case "CryoViT":
            input_key = "dino_features"
        case _:
            input_key = "data"
    dataset_params = {
        "input_key": input_key,
        "label_key": model_config.label_key,
        "data_root": data_dir,
        "aux_keys": ["data"],
    }
    datamodule = instantiate(
        MultiSample(sample=tuple(model_config.samples), split_id=split_id)
    )(
        split_file=split_file,
        dataloader_fn=instantiate(DataLoader(batch_size=batch_size)),
        dataset_params=dataset_params,
    )
    # Setup the trainer and model
    trainer = instantiate(trainer_config)
    if platform.system() != "Windows":
        model.forward = torch.compile(model.forward)
    # Train model
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    metrics = trainer.validate(model, dataloaders=datamodule.test_dataloader())
    model_config.metrics = metrics[0]


def run_inference(
    model: BaseModel,
    model_config: InterfaceModelConfig,
    data_dir: Path,
    batch_size: int = None,
    dst_dir: Path = None,
    csv_file: Path = None,
) -> None:
    """Run inference on a set of tomograms using the specified model.

    Args:
        model (BaseModel): The model to use for inference.
        model_config (InterfaceModelConfig): Information about the model.
        data_dir (Path): Path to the directory containing the tomograms to infer on.
        batch_size (int, optional): The size of batches. Defaults to None (i.e., 1).
        dst_dir (Path, optional): Path to the directory to save predictions. Defaults to None. If None, predictions will be added to the original tomograms.
        csv_file (Path, optional): Path to the .csv file specifying which tomograms to infer on. Defaults to None. If None, all tomograms in the directory will be used.
    """
    # Setup the dataloader
    if csv_file:
        # read files from .csv
        records = pd.read_csv(csv_file)["tomo_name"]
    else:
        # use all files in the directory
        records = pd.DataFrame(
            {
                "tomo_name": [
                    f.name
                    for f in data_dir.glob("*")
                    if f.suffix in {".rec", ".mrc", ".hdf"}
                ]
            }
        )
    records["sample"] = data_dir.name
    match model_config.model_type:
        case ModelArch.CRYOVIT:
            input_key = "dino_features"
        case _:
            input_key = "data"
    dataset = TomoDataset(
        records=records,
        data_root=data_dir,
        input_key=input_key,
        train=False,
        aux_keys=["data"],
    )
    dataloader = instantiate(DataLoader(batch_size=batch_size))(dataset, shuffle=False)
    # Setup trainer
    pred_writer = TomoPredictionWriter(
        dst_dir,
        model_config.name + "_" + model_config.label_key,
        write_interval="batch",
    )
    trainer = instantiate(Trainer(callbacks=[pred_writer]))
    trainer.predict(model, dataloader, return_predictions=False)
