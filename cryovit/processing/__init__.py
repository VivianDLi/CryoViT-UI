"""Implementation of pre- and post-processing pipelines."""

from cryovit.processing.preprocessing import run_preprocess
from cryovit.processing.annotations import (
    add_annotations,
    add_splits,
    generate_new_splits,
)
from cryovit.processing.model import (
    get_available_models,
    get_model_configs,
    save_model,
    load_model,
    load_base_model,
    get_dino_features,
    train_model,
    run_inference,
)

chimera_script_path = __file__.replace("__init__.py", "chimera_slices.py")

__all__ = [
    "run_preprocess",
    "add_annotations",
    "add_splits",
    "generate_new_splits",
    "get_available_models",
    "get_model_configs",
    "save_model",
    "load_model",
    "load_base_model",
    "get_dino_features",
    "train_model",
    "run_inference",
    "chimera_script_path",
]
