"""Dataclasses to hold pre-processing configuration for the CryoViT GUI application."""

from dataclasses import dataclass

from cryovit_gui.config import ConfigInputType, ConfigField, ConfigGroup


@dataclass
class PreprocessingConfig(ConfigGroup):
    """Dataclass to hold preprocessing settings."""

    name: str = "Preprocessing Configuration"
    bin_size: ConfigField = ConfigField(
        "Bin Size",
        input_type=ConfigInputType.NUMBER,
        default=2,
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
        default=True,
        required=True,
        description="Whether to normalize the tomogram data.",
    )
    clip: ConfigField = ConfigField(
        "Clip",
        input_type=ConfigInputType.BOOL,
        default=True,
        description="Whether to clip the tomogram data to 3 standard deviations.",
    )
