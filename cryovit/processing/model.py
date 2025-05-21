"""Script for training, evaluating, and running inference with a model for GUI use."""

import sys
from pathlib import Path
import json
from typing import Any, List, Tuple
import logging

from cryovit.config import (
    InterfaceModelConfig,
    Model,
    CryoVIT,
    UNet3D,
    ModelArch,
)

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Custom classes and functions for handling ModelArch enums
class InterfaceEncoder(json.JSONEncoder):
    """Custom JSON encoder for InterfaceModelConfig."""

    def default(self, obj: InterfaceModelConfig) -> Any:
        """Convert InterfaceModelConfig to JSON serializable format."""
        return obj.to_json()


def as_config(json_dct: dict) -> InterfaceModelConfig:
    """Convert dictionary to InterfaceModelConfig."""
    kwargs = {}
    for key, value in json_dct.items():
        match key:
            case "model_type":
                kwargs[key] = ModelArch[value]
            case "model_params":
                kwargs[key] = {k: v for k, v in value.items()}
            case "model_weights":
                kwargs[key] = Path(value).resolve()
            case _:
                kwargs[key] = value
    return InterfaceModelConfig(**kwargs)


def get_available_models(model_dir: Path) -> List[str]:
    """Get a list of available models in the specified directory.

    Args:
        model_dir (Path): Directory containing the model config files.

    Returns:
        List[str]: List of model names (without file extension) found in the directory.
    """
    return [str(f.parent.name) for f in model_dir.glob("**/config.json") if f.is_file()]


def get_model_configs(
    model_dir: Path, model_names: List[str]
) -> List[InterfaceModelConfig]:
    """Get model information for the UI for a list of model names.

    Args:
        model_dir (Path): Directory containing the model configuration files.
        model_names (List[str]): List of model names to retrieve configurations for.

    Returns:
        List[InterfaceModelConfig]: List of model UI configurations for the specified model names.
    """
    configs = []
    for model_name in model_names:
        # Check if the model directory exists
        model_path = model_dir / model_name / "config.json"
        if not model_path.exists():
            logger.error(f"Model config {model_path} does not exist.")
            continue
        with open(model_path, "r") as f:
            config_dict = json.load(f)
            model_config = as_config(config_dict)
        configs.append(model_config)
    return configs


def save_model_config(model_dir: Path, model_config: InterfaceModelConfig):
    """Save a model UI configuration to disk as a JSON file."""
    model_name = model_config.name
    # Check if the model directory exists, if not create it
    config_dir = model_dir / model_name
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / model_name / "config.json", "w+") as f:
        json.dump(model_config, f, cls=InterfaceEncoder)


def load_base_model_config(model_config: InterfaceModelConfig) -> Tuple[str, Model]:
    """Load a base model for training based on the provided configuration.

    Args:
        model_config (InterfaceModelConfig): The model configuration.

    Returns:
        Tuple[str, Model]: A tuple containing:
            - str: The name of the model.
            - Model: The loaded model.
    """
    match model_config.model_type:
        case ModelArch.CRYOVIT:
            model_config = CryoVIT(**model_config.model_params)
            model_name = "cryovit"
        case ModelArch.UNET3D:
            model_config = UNet3D(**model_config.model_params)
            model_name = "unet3d"
        case _:
            logger.error(f"Unknown model type: {model_config.model_type}")
    return model_name, model_config
