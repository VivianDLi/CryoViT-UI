"""Dataclasses for configuring settings and options in the CryoViT GUI for processing steps."""

from dataclasses import dataclass, field, fields
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Any, Dict, List, Union

import logging

logger = logging.getLogger(__name__)

#### Enums and Constants ####


class ModelArch(Enum):
    CRYOVIT = "CryoViT"
    UNET3D = "UNet3D"


models = [model.name for model in ModelArch]

tomogram_exts = [".rec", ".mrc", ".hdf"]

#### Interface Config Dataclasses ####

ignored_config_keys = [
    "aux_keys",
    "dataloader",
    "cryovit_root",
    "test_samples",
    "split_id",
    "logger",
    "callbacks",
    "losses",
    "metrics",
    "models",
    "all_samples",
    "_target_",
    "_partial_",
]


@dataclass
class InterfaceModelConfig:
    """Metadata for a model for GUI use."""

    name: str
    label_key: str
    model_type: ModelArch
    model_weights: Path
    model_params: Dict[str, Union[str, int, float]]
    samples: List[str]

    def to_json(self) -> dict:
        """Convert the model configuration to a JSON serializable format."""
        json_dict = {
            "name": self.name,
            "label_key": self.label_key,
            "model_type": self.model_type.name,
            "model_weights": str(self.model_weights.resolve()),
            "model_params": dict(self.model_params),
            "samples": list(map(str, self.samples)),
        }
        return json_dict


#### Settings Dataclasses ####


class ConfigInputType(Flag):
    """Enum for different input types of configs."""

    FILE = auto()
    DIRECTORY = auto()
    TEXT = auto()
    NUMBER = auto()
    BOOL = auto()
    STR_LIST = auto()
    INT_LIST = auto()


@dataclass
class ConfigField:
    """Dataclass to hold a single config field."""

    name: str
    input_type: ConfigInputType
    value: str = None
    default: str = None
    description: str = ""
    required: bool = False

    def get_value(self) -> Any:
        """Get the value of the setting field."""
        if self.value is None:
            if self.required and self.default is None:
                raise ValueError(f"Value for {self.name} is required but not set.")
            else:
                self.value = self.default
        # Double check types
        match self.input_type:
            case ConfigInputType.FILE | ConfigInputType.DIRECTORY:
                return Path(self.value).resolve()
            case ConfigInputType.TEXT:
                return self.value
            case ConfigInputType.NUMBER:
                try:
                    return int(self.value)
                except ValueError:
                    raise ValueError(f"Value for {self.name} must be a number.")
            case ConfigInputType.BOOL:
                return self.value.lower() in ["true", "1", "yes"]
            case ConfigInputType.STR_LIST:
                return list(map(str.strip, self.value.split(","))) if self.value else []
            case ConfigInputType.INT_LIST:
                return (
                    list(map(int, map(str.strip, self.value.split(","))))
                    if self.value
                    else []
                )
        return self.value

    def get_value_as_str(self) -> str:
        """Get the value of the setting field as a string."""
        if self.value is None:
            if self.required and self.default is None:
                raise ValueError(f"Value for {self.name} is required but not set.")
            else:
                self.value = self.default
        return self.value if self.value is not None else None

    def set_value(self, value: str) -> None:
        """Set the value of the setting field."""
        self.value = str(value).strip()


@dataclass
class ConfigGroup:
    """Dataclass to hold a group of configs."""

    def get_all_fields(self) -> List[str]:
        """Get a list of fields for the config."""
        results = []
        for f in fields(self):
            field = getattr(self, f.name)
            if isinstance(field, ConfigField):
                results.append(f.name)
            elif isinstance(field, ConfigGroup):
                # Recursively get subfields
                subfields = getattr(self, f.name).get_available_settings()
                results.extend([f"{f.name}/{subfield}" for subfield in subfields])
            else:
                logger.warning("Unknown type for field %s: %s", f.name, f.type)
        return sorted(results)

    def get_field(self, key: str, as_str: bool = False):
        """Get a setting from the settings data, reading the key like a path."""
        paths = key.split("/")
        if len(paths) == 1:  # base case
            result = getattr(self, paths[0], None)
            if result is None or (
                not isinstance(result, ConfigField)
                and not isinstance(result, ConfigGroup)
            ):
                logger.error(f"Setting {paths[0]} was not found.")
                return None
            if isinstance(result, ConfigField):
                return result.get_value_as_str() if as_str else result.get_value()
            else:
                return result
        parent_key, subkey = paths[0], "/".join(paths[1:])
        if not hasattr(self, parent_key):
            logger.error(f"Parent setting {parent_key} does not exist.")
            return None
        return getattr(self, parent_key).get_field(subkey, as_str=as_str)

    def set_field(self, key: str, value: str) -> None:
        """Set a setting in the settings data, reading the key like a path."""
        paths = key.split("/")
        if len(paths) == 1:  # base case
            setting_field = getattr(self, paths[0])
            setting_field.set_value(value)
            return
        parent_key, subkey = paths[0], "/".join(paths[1:])
        getattr(self, parent_key).set_field(subkey, value)


@dataclass
class GeneralSettings(ConfigGroup):
    """Dataclass to hold general settings relating to the GUI application."""

    data_directory: ConfigField = ConfigField(
        "Data Directory",
        input_type=ConfigInputType.DIRECTORY,
        description="Directory where data is stored. This is used as the default location for file/folder selection dialogs.",
    )


@dataclass
class PresetSettings(ConfigGroup):
    """Dataclass to hold preset settings."""

    current_preset: ConfigField = ConfigField(
        "Current Preset",
        input_type=ConfigInputType.TEXT,
        description="Name of the current settings preset. This is used to load and save various setting configurations.",
    )
    available_presets: ConfigField = ConfigField(
        "Available Presets",
        input_type=ConfigInputType.STR_LIST,
        default="",
        required=True,
        description="List of available presets. This is used to load and save various setting configurations.",
    )


@dataclass
class PreprocessingSettings(ConfigGroup):
    """Dataclass to hold preprocessing settings."""

    bin_size: ConfigField = ConfigField(
        "Bin Size",
        input_type=ConfigInputType.NUMBER,
        default="2",
        required=True,
        description="Bin size for downsampling the tomogram data z-axis.",
    )
    resize_image: ConfigField = ConfigField(
        "Resize Image",
        input_type=ConfigInputType.INT_LIST,
        description="Resize the tomograms to a specific size.",
    )
    normalize: ConfigField = ConfigField(
        "Normalize",
        input_type=ConfigInputType.BOOL,
        default="True",
        required=True,
        description="Whether to normalize the tomogram data.",
    )
    clip: ConfigField = ConfigField(
        "Clip",
        input_type=ConfigInputType.BOOL,
        default="True",
        description="Whether to clip the tomogram data to 3 standard deviations.",
    )


@dataclass
class ModelSettings(ConfigGroup):
    """Dataclass to hold model settings."""

    model_directory: ConfigField = ConfigField(
        "Model Directory",
        input_type=ConfigInputType.DIRECTORY,
        default="./models",
        required=True,
        description="Directory where the model weights are stored.",
    )


@dataclass
class DinoSettings(ConfigGroup):
    """Dataclass to hold DINO settings."""

    model_directory: ConfigField = ConfigField(
        "DINO Model Directory",
        input_type=ConfigInputType.DIRECTORY,
        default="./DINOv2",
        required=True,
        description="Directory where the DINO model weights are stored.",
    )
    feature_directory: ConfigField = ConfigField(
        "DINO Feature Directory",
        input_type=ConfigInputType.DIRECTORY,
        default="./DINOv2/features",
        required=True,
        description="Directory where the computed DINO model features are stored.",
    )
    batch_size: ConfigField = ConfigField(
        "DINO Batch Size",
        input_type=ConfigInputType.NUMBER,
        default="128",
        description="Batch size for DINO model inference.",
    )


@dataclass
class SegmentationSettings(ConfigGroup):
    """Dataclass to hold segmentation settings."""

    batch_size: ConfigField = ConfigField(
        "Segmentation Batch Size",
        input_type=ConfigInputType.NUMBER,
        default="1",
        description="Batch size for segmentation model inference.",
    )


@dataclass
class AnnotationSettings(ConfigGroup):
    """Dataclass to hold annotation settings."""

    chimera_path: ConfigField = ConfigField(
        "Chimera Path",
        input_type=ConfigInputType.DIRECTORY,
        default="",
        required=True,
        description="Path to the ChimeraX executable for annotation.",
    )
    num_slices: ConfigField = ConfigField(
        "Number of Slices",
        input_type=ConfigInputType.NUMBER,
        default="5",
        description="Number of slices to extract from the tomogram for annotation.",
    )


@dataclass
class TrainingSettings(ConfigGroup):
    """Dataclass to hold training settings."""

    num_splits = ConfigField(
        "Number of Splits",
        input_type=ConfigInputType.NUMBER,
        default="10",
        description="Number of splits for cross-validation during training.",
    )
    batch_size: ConfigField = ConfigField(
        "Batch Size",
        input_type=ConfigInputType.NUMBER,
        default="1",
        description="Batch size for training the model.",
    )
    random_seed: ConfigField = ConfigField(
        "Random Seed",
        input_type=ConfigInputType.NUMBER,
        default="42",
        description="Random seed for reproducibility in training.",
    )


@dataclass
class Settings(ConfigGroup):
    """Dataclass to hold general settings relating to the GUI application."""

    general: GeneralSettings = field(default_factory=GeneralSettings)
    preset: PresetSettings = field(default_factory=PresetSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    dino: DinoSettings = field(default_factory=DinoSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    annotation: AnnotationSettings = field(default_factory=AnnotationSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
