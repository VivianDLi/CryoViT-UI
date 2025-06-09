"""Dataclasses to hold inference configuration for the CryoViT GUI application."""

from dataclasses import dataclass, field

from cryovit_gui.config import ConfigInputType, ConfigField, ConfigGroup


@dataclass
class ModelConfig(ConfigGroup):
    """Dataclass to hold model configuration."""

    model_directory: ConfigField = ConfigField(
        "Model Directory",
        input_type=ConfigInputType.DIRECTORY,
        default="./models",
        required=True,
        description="Directory where the model weights are stored.",
    )


@dataclass
class DinoConfig(ConfigGroup):
    """Dataclass to hold DINO settings."""

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
        default=128,
        description="Batch size for DINO model inference.",
    )


@dataclass
class SegmentationConfig(ConfigGroup):
    """Dataclass to hold segmentation configuration."""

    batch_size: ConfigField = ConfigField(
        "Segmentation Batch Size",
        input_type=ConfigInputType.NUMBER,
        default=1,
        description="Batch size for segmentation model inference.",
    )


@dataclass
class InferenceConfig(ConfigGroup):
    """Dataclass to hold inference configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
