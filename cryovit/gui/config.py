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


class SettingsInputType(Flag):
    """Enum for different input types of settings."""

    FILE = auto()
    DIRECTORY = auto()
    TEXT = auto()
    NUMBER = auto()
    BOOL = auto()
    STR_LIST = auto()
    INT_LIST = auto()


@dataclass
class SettingField:
    """Dataclass to hold a single setting field."""

    name: str
    input_type: SettingsInputType
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
            case SettingsInputType.FILE | SettingsInputType.DIRECTORY:
                return Path(self.value).resolve()
            case SettingsInputType.TEXT:
                return self.value
            case SettingsInputType.NUMBER:
                try:
                    return int(self.value)
                except ValueError:
                    raise ValueError(f"Value for {self.name} must be a number.")
            case SettingsInputType.BOOL:
                return self.value.lower() in ["true", "1", "yes"]
            case SettingsInputType.STR_LIST:
                return list(map(str.strip, self.value.split(","))) if self.value else []
            case SettingsInputType.INT_LIST:
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
class BaseSetting:
    """Base dataclass for storing and retrieving settings."""

    def get_available_settings(self) -> List[str]:
        """Get a list of available settings."""
        results = []
        for f in fields(self):
            field = getattr(self, f.name)
            if isinstance(field, SettingField):
                results.append(f.name)
            elif isinstance(field, BaseSetting):
                # Recursively get subfields
                subfields = getattr(self, f.name).get_available_settings()
                results.extend([f"{f.name}/{subfield}" for subfield in subfields])
            else:
                logger.warning("Unknown type for field %s: %s", f.name, f.type)
        return sorted(results)

    def get_setting(self, key: str, as_str: bool = False):
        """Get a setting from the settings data, reading the key like a path."""
        paths = key.split("/")
        if len(paths) == 1:  # base case
            result = getattr(self, paths[0], None)
            if result is None or (
                not isinstance(result, SettingField)
                and not isinstance(result, BaseSetting)
            ):
                logger.error(f"Setting {paths[0]} was not found.")
                return None
            if isinstance(result, SettingField):
                return result.get_value_as_str() if as_str else result.get_value()
            else:
                return result
        parent_key, subkey = paths[0], "/".join(paths[1:])
        if not hasattr(self, parent_key):
            logger.error(f"Parent setting {parent_key} does not exist.")
            return None
        return getattr(self, parent_key).get_setting(subkey, as_str=as_str)

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting in the settings data, reading the key like a path."""
        paths = key.split("/")
        if len(paths) == 1:  # base case
            setting_field = getattr(self, paths[0])
            setting_field.set_value(value)
            return
        parent_key, subkey = paths[0], "/".join(paths[1:])
        getattr(self, parent_key).set_setting(subkey, value)


@dataclass
class GeneralSettings(BaseSetting):
    """Dataclass to hold general settings relating to the GUI application."""

    data_directory: SettingField = SettingField(
        "Data Directory",
        input_type=SettingsInputType.DIRECTORY,
        description="Directory where data is stored. This is used as the default location for file/folder selection dialogs.",
    )


@dataclass
class PresetSettings(BaseSetting):
    """Dataclass to hold preset settings."""

    current_preset: SettingField = SettingField(
        "Current Preset",
        input_type=SettingsInputType.TEXT,
        description="Name of the current settings preset. This is used to load and save various setting configurations.",
    )
    available_presets: SettingField = SettingField(
        "Available Presets",
        input_type=SettingsInputType.STR_LIST,
        default="",
        required=True,
        description="List of available presets. This is used to load and save various setting configurations.",
    )


@dataclass
class PreprocessingSettings(BaseSetting):
    """Dataclass to hold preprocessing settings."""

    bin_size: SettingField = SettingField(
        "Bin Size",
        input_type=SettingsInputType.NUMBER,
        default="2",
        required=True,
        description="Bin size for downsampling the tomogram data z-axis.",
    )
    resize_image: SettingField = SettingField(
        "Resize Image",
        input_type=SettingsInputType.INT_LIST,
        description="Resize the tomograms to a specific size.",
    )
    normalize: SettingField = SettingField(
        "Normalize",
        input_type=SettingsInputType.BOOL,
        default="True",
        required=True,
        description="Whether to normalize the tomogram data.",
    )
    clip: SettingField = SettingField(
        "Clip",
        input_type=SettingsInputType.BOOL,
        default="True",
        description="Whether to clip the tomogram data to 3 standard deviations.",
    )


@dataclass
class ModelSettings(BaseSetting):
    """Dataclass to hold model settings."""

    model_directory: SettingField = SettingField(
        "Model Directory",
        input_type=SettingsInputType.DIRECTORY,
        default="./models",
        required=True,
        description="Directory where the model weights are stored.",
    )


@dataclass
class DinoSettings(BaseSetting):
    """Dataclass to hold DINO settings."""

    model_directory: SettingField = SettingField(
        "DINO Model Directory",
        input_type=SettingsInputType.DIRECTORY,
        default="./DINOv2",
        required=True,
        description="Directory where the DINO model weights are stored.",
    )
    feature_directory: SettingField = SettingField(
        "DINO Feature Directory",
        input_type=SettingsInputType.DIRECTORY,
        default="./DINOv2/features",
        required=True,
        description="Directory where the computed DINO model features are stored.",
    )
    batch_size: SettingField = SettingField(
        "DINO Batch Size",
        input_type=SettingsInputType.NUMBER,
        default="128",
        description="Batch size for DINO model inference.",
    )


@dataclass
class SegmentationSettings(BaseSetting):
    """Dataclass to hold segmentation settings."""

    batch_size: SettingField = SettingField(
        "Segmentation Batch Size",
        input_type=SettingsInputType.NUMBER,
        default="1",
        description="Batch size for segmentation model inference.",
    )


@dataclass
class AnnotationSettings(BaseSetting):
    """Dataclass to hold annotation settings."""

    chimera_path: SettingField = SettingField(
        "Chimera Path",
        input_type=SettingsInputType.DIRECTORY,
        default="",
        required=True,
        description="Path to the ChimeraX executable for annotation.",
    )
    num_slices: SettingField = SettingField(
        "Number of Slices",
        input_type=SettingsInputType.NUMBER,
        default="5",
        description="Number of slices to extract from the tomogram for annotation.",
    )


@dataclass
class TrainingSettings(BaseSetting):
    """Dataclass to hold training settings."""

    num_splits = SettingField(
        "Number of Splits",
        input_type=SettingsInputType.NUMBER,
        default="10",
        description="Number of splits for cross-validation during training.",
    )
    batch_size: SettingField = SettingField(
        "Batch Size",
        input_type=SettingsInputType.NUMBER,
        default="1",
        description="Batch size for training the model.",
    )
    random_seed: SettingField = SettingField(
        "Random Seed",
        input_type=SettingsInputType.NUMBER,
        default="42",
        description="Random seed for reproducibility in training.",
    )


@dataclass
class Settings(BaseSetting):
    """Dataclass to hold general settings relating to the GUI application."""

    general: GeneralSettings = field(default_factory=GeneralSettings)
    preset: PresetSettings = field(default_factory=PresetSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    dino: DinoSettings = field(default_factory=DinoSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    annotation: AnnotationSettings = field(default_factory=AnnotationSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
