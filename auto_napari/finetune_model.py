import os
from typing import Tuple
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download


#### Get Trained Model Weights ####
def get_pretrained_model(
    model_name: str = "nnInteractive_v1.0", download_dir: str = "/nninter/temp"
) -> Module:
    """
    Download the pretrained model weights from Hugging Face and load the model.

    Args:
        model_name (str): Name of the model to download.
        download_dir (Path): Directory to save the downloaded model.

    Returns:
        Model: The loaded model with pretrained weights.
    """
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    # Download the model weights
    snapshot_download(
        repo_id=model_name, local_dir=download_dir, local_dir_use_symlinks=False
    )
    model_path = os.path.join(download_dir, model_name)
    # Load the model
    session = nnInteractiveInferenceSession(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(model_path)

    return session.network


def get_nn_dataset(
    dataset_path: str, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset_path (str): Path to the dataset.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    # dataset.json has "ignore" label for ignoring unlabelled slices (must be highest integer label)
    # nnUNETv2_train FINETUNING DATASET CONFIG FOLD -pretrained_weights PATH_TO_CHECKPOINT

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(
        nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id)
    )
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, "dataset.json"))
    nnunet_trainer = nnunet_trainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device,
    )
    # nnUNetDataLoader create train, val, test dataloaders
    return nnunet_trainer
    pass


def run_finetuner(
    model: Module,
    train_data: str,
    val_data: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> Module:
    """
    Fine-tune the model with the provided training and validation data.

    Args:
        model (Module): The model to be fine-tuned.
        train_data (str): Path to the training data.
        val_data (str): Path to the validation data.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Module: The fine-tuned model.
    """
    from nnInteractive.trainer.nnInteractiveTrainer import nnUNetTrainer

    trainer = nnUNetTrainer(
        plans={},
        configuration="",
        fold=0,
        dataset_json={},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return trainer.train()


if __name__ == "__main__":
    # Example usage
    model = get_pretrained_model()
    fine_tuned_model = run_finetuner(
        model,
        train_data="path/to/train/data",
        val_data="path/to/val/data",
        epochs=5,
        batch_size=16,
        learning_rate=1e-5,
    )
    print("Fine-tuning complete.")


"""Script for setting up and training CryoVIT models based on configuration files."""

import logging
import os

import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from pytorch_lightning import seed_everything

from cryovit.config import ExpPaths
from cryovit.config import Sample
from cryovit.config import TrainModelConfig


seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


def set_wandb_config(cfg: TrainModelConfig) -> None:
    """Sets the W&B logger configuration based on the training parameters.

    Args:
        cfg (TrainModelConfig): Configuration object containing model and dataset settings.
    """
    if isinstance(cfg.dataset.sample, Sample):
        sample = cfg.dataset.sample.name
    else:
        sample = "_".join(sorted([s.name for s in cfg.dataset.sample]))

    config = {
        "model": cfg.model._target_.split(".")[-1],
        "experiment": cfg.exp_name,
        "split_id": cfg.dataset.split_id,
        "sample": sample,
    }

    for logger in cfg.trainer.logger:
        if logger._target_.split(".")[-1] == "WandbLogger":
            logger.config.update(config)


def setup_params(exp_paths: ExpPaths, cfg: TrainModelConfig) -> None:
    """Configures experiment paths based on the dataset type and experiment settings.

    Args:
        exp_paths (ExpPaths): Object containing path settings for the experiment.
        cfg (TrainModelConfig): Configuration object for the training model.
    """
    dataset_type = HydraConfig.get().runtime.choices.dataset

    match dataset_type:
        case "single" | "loo" | "fractional":
            exp_paths.exp_dir /= cfg.dataset.sample.name

        case "multi":
            samples = sorted([s.name for s in cfg.dataset.sample])
            exp_paths.exp_dir /= "_".join(samples)

    match dataset_type:
        case "single" | "multi" | "loo":
            split_id = cfg.dataset.split_id
            split_dir = "" if split_id is None else f"split_{split_id}"
            exp_paths.exp_dir /= split_dir

        case "fractional":
            split_id = cfg.dataset.split_id
            exp_paths.exp_dir /= f"split_{split_id}"

            if not 1 <= split_id <= 10:
                raise ValueError(f"split_id: {split_id} must be between 1 and 10")

    os.makedirs(exp_paths.exp_dir, exist_ok=True)


def build_datamodule(cfg: TrainModelConfig) -> LightningDataModule:
    """Creates a data module for the model based on the configuration.

    Args:
        cfg (TrainModelConfig): Configuration object specifying the dataset and model settings.

    Returns:
        LightningDataModule: A data module instance configured as specified.
    """
    model_type = HydraConfig.get().runtime.choices.model

    match model_type:
        case "cryovit":
            input_key = "dino_features"
        case _:
            input_key = "data"

    dataset_params = {
        "input_key": input_key,
        "label_key": cfg.label_key,
        "data_root": cfg.exp_paths.tomo_dir,
        "aux_keys": cfg.aux_keys,
    }

    return instantiate(cfg.dataset)(
        split_file=cfg.exp_paths.split_file,
        dataloader_fn=instantiate(cfg.dataloader),
        dataset_params=dataset_params,
    )


def run_trainer(cfg: TrainModelConfig) -> None:
    """Sets up and runs the training process using the specified configuration.

    Args:
        cfg (TrainModelConfig): Configuration object containing all settings for the training process.
    """
    exp_paths = cfg.exp_paths
    setup_params(exp_paths, cfg)
    set_wandb_config(cfg)

    datamodule = build_datamodule(cfg)
    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)
    model.forward = torch.compile(model.forward)

    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.test_dataloader(),
    )

    torch.save(model.state_dict(), (exp_paths.exp_dir / "weights.pt"))
    torch.cuda.empty_cache()
    wandb.finish(quiet=True)
