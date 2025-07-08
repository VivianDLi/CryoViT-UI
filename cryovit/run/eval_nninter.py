"""Script for evaluating CryoVIT models based on configuration files."""

import os
from collections import defaultdict
from pathlib import Path

import h5py
import pandas as pd
import torch
import numpy as np
from scipy.ndimage import center_of_mass
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from huggingface_hub import snapshot_download

from cryovit.config import EvalModelConfig, ExpPaths
from cryovit.models.metrics import DiceMetric


def validate_paths(exp_paths: ExpPaths, cfg: EvalModelConfig) -> None:
    """Validates the existence of necessary paths for model evaluation.

    Args:
        exp_paths (ExpPaths): Configuration of experiment paths.
        cfg (EvalModelConfig): Evaluation configuration.
    """
    dataset_type = HydraConfig.get().runtime.choices.dataset

    match dataset_type:
        case "single" | "loo" | "fractional":
            exp_paths.exp_dir /= cfg.dataset.sample.name

        case "multi":
            samples = sorted([s.name for s in cfg.dataset.sample])
            exp_paths.exp_dir /= "_".join(samples)


def build_datamodule(cfg: EvalModelConfig) -> LightningDataModule:
    """Constructs and returns a data module for model evaluation based on configuration settings.

    Args:
        cfg (EvalModelConfig): Configuration object specifying the dataset and model settings.

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


def save_results(
    data: np.ndarray,
    labels: np.ndarray,
    results: np.ndarray,
    result_dir: Path,
    sample: str,
    tomo_name: str,
) -> None:
    tomo_dir = result_dir / sample
    os.makedirs(tomo_dir, exist_ok=True)

    with h5py.File(tomo_dir / tomo_name, "w") as fh:
        fh.create_dataset("data", data=data)
        fh.create_dataset("preds", data=results, compression="gzip")
        fh.create_dataset("label", data=labels, compression="gzip")


def get_scores(
    session: nnInteractiveInferenceSession,
    model_dir: Path,
    result_dir: Path,
    cfg: EvalModelConfig,
):
    """Evaluates the model"""
    session.initialize_from_trained_model_folder(model_dir)
    datamodule = build_datamodule(cfg)

    scores = defaultdict(list)
    metric = DiceMetric(threshold=0.5)
    for data in datamodule.test_dataloader():
        img = data["input"].numpy()
        labels = data["label"].numpy()
        if len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
            labels = labels[np.newaxis, :, :, :]
        session.set_image(img)
        ## Define Output Buffer ##
        target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)
        session.set_target_buffer(target_tensor)
        ## Add interaction based on the center of labeled slices ##
        int_point = center_of_mass(labels, labels=labels, index=1)[1:]
        print("after point")
        session.add_point_interaction(int_point, include_interaction=True)
        print("after interaction")
        ## Get Results ##
        results = session.target_buffer.clone()
        print("after results")
        ## Save Results ##
        save_results(
            img, labels, results.numpy(), result_dir, data["sample"], data["tomo_name"]
        )
        print("after save")
        ## Add to Scores ##
        scores["sample"].append(data["sample"])
        scores["tomo_name"].append(data["tomo_name"])
        score = metric(results, torch.from_numpy(labels))
        scores["TEST_DiceMetric"].append(score.item())
        ## Start next image ##
        session.reset_interactions()

    return pd.DataFrame.from_dict(scores)


def run_trainer(cfg: EvalModelConfig) -> None:
    """Sets up and executes the model evaluation using the specified configuration.

    Args:
        cfg (EvalModelConfig): Configuration object containing all settings for the evaluation process.
    """
    dataset_type = HydraConfig.get().runtime.choices.dataset
    exp_paths = cfg.exp_paths
    validate_paths(exp_paths, cfg)

    split_id = cfg.dataset.split_id
    split_dir = "" if split_id is None else f"split_{split_id}"

    session = nnInteractiveInferenceSession(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )

    match dataset_type:
        case "single":
            result_dir = exp_paths.exp_dir.parent / "results"
            model_dir = exp_paths.exp_dir.parent / "nn_model"

        case "multi":
            result_dir = exp_paths.exp_dir / "results" / split_dir
            model_dir = exp_paths.exp_dir / "nn_model"

        case "loo" | "fractional":
            result_dir = exp_paths.exp_dir.parent / "results" / split_dir
            model_dir = exp_paths.exp_dir.parent / "nn_model"

    # --- Download Trained Model Weights (~400MB) ---
    REPO_ID = "nnInteractive/nnInteractive"
    MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future

    download_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{MODEL_NAME}/*"],
        local_dir=model_dir
    )

    model_path = os.path.join(model_dir, MODEL_NAME)
    result_df = get_scores(session, model_path, result_dir, cfg)
    result_file = result_dir / "nnInteractive.csv"
    result_df.to_csv(result_file, index=False)
    print(result_df.describe())
