"""Dataclasses to hold general settings for the CryoViT GUI application."""

from dataclasses import dataclass, field
from pathlib import Path

from cryovit_gui.config import ConfigInputType, ConfigField, ConfigGroup


@dataclass
class GeneralSettings(ConfigGroup):
    """Dataclass to hold general settings relating to the GUI application."""

    name: str = "General Settings"
    data_directory: ConfigField = ConfigField(
        "Data Directory",
        input_type=ConfigInputType.DIRECTORY,
        description="Directory where data is stored. This is used as the default location for file/folder selection dialogs.",
    )


@dataclass
class PresetSettings(ConfigGroup):
    """Dataclass to hold preset settings."""

    name: str = "Preset Settings"
    current_preset: ConfigField = ConfigField(
        "Current Preset",
        input_type=ConfigInputType.TEXT,
        description="Name of the current settings preset. This is used to load and save various setting configurations.",
    )
    available_presets: ConfigField = ConfigField(
        "Available Presets",
        input_type=ConfigInputType.STR_LIST,
        default=[],
        required=True,
        description="List of available presets. This is used to load and save various setting configurations.",
    )


@dataclass
class AnnotationSettings(ConfigGroup):
    """Dataclass to hold annotation settings."""

    name: str = "Annotation Settings"
    chimera_path: ConfigField = ConfigField(
        "Chimera Path",
        input_type=ConfigInputType.DIRECTORY,
        default=Path("C:/Program Files/ChimeraX 1.9/bin/ChimeraX.exe"),
        required=True,
        description="Path to the ChimeraX executable for annotation.",
    )
    num_slices: ConfigField = ConfigField(
        "Number of Slices",
        input_type=ConfigInputType.NUMBER,
        default=5,
        description="Number of slices to extract from the tomogram for annotation.",
    )


@dataclass
class DinoSettings(ConfigGroup):
    """Dataclass to hold DINO settings."""

    name: str = "DINO Settings"
    model_directory: ConfigField = ConfigField(
        "DINO Model Directory",
        input_type=ConfigInputType.DIRECTORY,
        default=Path("./DINOv2"),
        required=True,
        description="Directory where the DINO model weights are stored.",
    )


@dataclass
class Settings(ConfigGroup):
    """Dataclass to hold general settings relating to the GUI application."""

    name: str = "Settings"
    general: GeneralSettings = field(default_factory=GeneralSettings)
    preset: PresetSettings = field(default_factory=PresetSettings)
    annotation: AnnotationSettings = field(default_factory=AnnotationSettings)
    dino: DinoSettings = field(default_factory=DinoSettings)
