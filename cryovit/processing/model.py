"""Script for training, evaluating, and running inference with a model for GUI use."""

import sys
from pathlib import Path
import json
from typing import Any, List
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
class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder for Enum types."""

    def default(self, obj: Any) -> Any:
        if type(obj) in ModelArch.values():
            return {"__enum__": obj.name}
        return super().default(obj)


def as_enum(dct: dict) -> ModelArch:
    if "__enum__" in dct:
        name = dct["__enum__"]
        return ModelArch[name]
    return dct


def get_available_models(model_dir: Path) -> List[str]:
    """Get a list of available models in the specified directory.

    Args:
        model_dir (Path): Directory containing the model config files.

    Returns:
        List[str]: List of model names (without file extension) found in the directory.
    """
    return [f.parent for f in model_dir.glob("*/config.json") if f.is_file()]


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
        with open(model_dir / model_name / "config.json", "r") as f:
            model_config = InterfaceModelConfig(**json.loads(f, object_hook=as_enum))
        configs.append(model_config)
    return configs


def save_model_config(model_dir: Path, model_config: InterfaceModelConfig) -> None:
    model_name = model_config.name
    with open(model_dir / model_name / "config.json", "w+") as f:
        json.dumps(model_config, f, cls=EnumEncoder)


def load_base_model_config(model_config: InterfaceModelConfig) -> Model:
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
    return model_config
