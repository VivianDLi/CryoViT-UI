"""Dataclasses to hold training configuration for the CryoViT GUI application."""

from dataclasses import dataclass, field

from cryovit_gui.config import ConfigInputType, ConfigField, ConfigGroup


@dataclass
class DinoConfig(ConfigGroup):
    """Dataclass to hold DINO configuration."""

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
class TrainerConfig(ConfigGroup):
    """Dataclass to hold trainer configuration."""

    num_splits = ConfigField(
        "Number of Splits",
        input_type=ConfigInputType.NUMBER,
        default=10,
        description="Number of splits for cross-validation during training.",
    )
    batch_size: ConfigField = ConfigField(
        "Batch Size",
        input_type=ConfigInputType.NUMBER,
        default=1,
        description="Batch size for training the model.",
    )
    random_seed: ConfigField = ConfigField(
        "Random Seed",
        input_type=ConfigInputType.NUMBER,
        default=42,
        description="Random seed for reproducibility in training.",
    )


@dataclass
class TrainingConfig(ConfigGroup):
    """Dataclass to hold training configuration."""

    dino: DinoConfig = field(default_factory=DinoConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
