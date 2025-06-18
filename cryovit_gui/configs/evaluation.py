"""Dataclasses to hold evaluation configuration for the CryoViT GUI application."""

from dataclasses import dataclass, field
from pathlib import Path

from cryovit_gui.config import ConfigInputType, ConfigField, ConfigGroup


@dataclass
class DinoConfig(ConfigGroup):
    """Dataclass to hold DINO configuration."""

    name: str = "DINO Configuration"
    feature_directory: ConfigField = ConfigField(
        "DINO Feature Directory",
        input_type=ConfigInputType.DIRECTORY,
        default=Path("./DINOv2/features"),
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
class MetricsConfig(ConfigGroup):
    """Dataclass to hold evaluation metrics configuration."""

    name: str = "Metrics Configuration"
    num_splits: ConfigField = ConfigField(
        "Number of Splits",
        input_type=ConfigInputType.NUMBER,
        default=10,
        description="Number of splits for cross-validation during evaluation.",
    )


@dataclass
class EvaluationConfig(ConfigGroup):
    """Dataclass to hold evaluation configuration."""

    name: str = "Evaluation Configuration"
    dino: DinoConfig = field(default_factory=DinoConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
