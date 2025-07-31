"""Dataclasses for configuring settings and options in the CryoViT GUI for processing steps."""

from dataclasses import dataclass, fields
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union

#### Logging Setup ####

import logging

logger = logging.getLogger("cryovit.config")
debug_logger = logging.getLogger("debug")

#### Types, Enums and Constants ####


class Colors(Enum):
    """Enum for different colors used in the GUI."""

    RED = (249, 65, 68)
    DARK_ORANGE = (243, 114, 44)
    ORANGE = (248, 150, 30)
    YELLOW = (249, 199, 79)
    GREEN = (144, 190, 109)
    WHITE = (220, 220, 220)
    GRAY = (150, 150, 150)
    BLUE = (30, 136, 229)


class ConfigInputType(Flag):
    """Enum for different input types of configs."""

    FILE = auto()
    DIRECTORY = auto()
    TEXT = auto()
    NUMBER = auto()
    BOOL = auto()
    STR_LIST = auto()
    INT_LIST = auto()


ConfigKey = List[str]


class SampleData(TypedDict):
    tomogram_files: List[Path]
    annotated: List[bool]
    selected: List[bool]
    exported: List[bool]


FileData = Dict[str, SampleData]

tomogram_exts = [".rec", ".mrc", ".hdf"]
required_directories = ["tomograms", "csv", "slices"]
optional_directories = ["annotations", "tomo_annot"]

# CryoViT commands
preprocess_command = "cryovit.preprocess"
train_command = "cryovit.train_model"
evaluate_command = "cryovit.evaluate_model"
inference_command = "cryovit.segment_model"

#### Config Dataclasses ####


@dataclass(order=True)
class ConfigKey:
    """Dataclass to hold a key for a configuration field."""

    key: List[str]

    def __str__(self) -> str:
        """Return the string representation of the key."""
        return "/".join(self.key)

    def __repr__(self) -> str:
        """Return the string representation of the key."""
        return "/".join(self.key)

    def __iter__(self):
        """Return an iterator over the key."""
        return iter(self.key)

    def add_parent(self, parent: str) -> None:
        """Add a parent key to the current key."""
        self.key.insert(0, parent)

    def pop_first(self) -> str:
        """Pop the first key from the current key."""
        if self.key:
            return ConfigKey([self.key.pop(0)])
        else:
            logger.error("Cannot pop from an empty ConfigKey.")
            return ""


@dataclass
class ConfigField:
    """Dataclass to hold a single configuration field."""

    name: str
    input_type: ConfigInputType
    value: Any = None
    default: Any = None
    description: str = ""
    required: bool = False
    _parent = None

    def get_parent(self) -> Union["ConfigGroup", None]:
        """Get the parent group for this configuration field."""
        return self._parent

    def set_parent(self, parent: "ConfigGroup") -> None:
        """Set the parent group for this configuration field."""
        self._parent = parent

    def get_type(self) -> type:
        """Get the type of the setting field."""
        match self.input_type:
            case ConfigInputType.FILE | ConfigInputType.DIRECTORY:
                return Path
            case ConfigInputType.TEXT:
                return str
            case ConfigInputType.NUMBER:
                return int
            case ConfigInputType.BOOL:
                return bool
            case ConfigInputType.STR_LIST | ConfigInputType.INT_LIST:
                return list
            case _:
                logger.error(
                    f"Unknown input type: {self.input_type}. Defaulting to str."
                )
                return str

    def get_value(self) -> Any:
        """Get the value of the setting field."""
        # Get target type
        target_type = self.get_type()
        # Check for defaults and required
        if self.value is None:
            if self.required and self.default is None:
                logger.error(f"Value for {self.name} is required but not set.")
            elif self.required and not isinstance(self.default, target_type):
                logger.error(
                    f"Default value for {self.name} must be of type {target_type.__name__}."
                )
            else:
                self.value = self.default
        # Check type
        try:
            if self.value is None:
                return None
            self.value = target_type(self.value)
        except ValueError as e:
            logger.error(
                f"Value for {self.name} must be of type {target_type.__name__}: {e}."
            )
            debug_logger.error(
                f"Value for {self.name} must be of type {target_type.__name__}: {e}.",
                exc_info=True,
            )
        except:
            logger.error(f"Unexpected error while getting value for {self.name}.")
            debug_logger.error(
                f"Unexpected error while getting value for {self.name}.", exc_info=True
            )
        finally:
            return self.value

    def get_value_as_str(self) -> str:
        """Get the value of the setting field as a string."""
        value = self.get_value()
        if isinstance(value, list):
            return ", ".join(map(str, value))
        return str(value)

    def set_value(self, value: Any, from_str: bool = False) -> None:
        """Set the value of the setting field."""
        # Parse value from string if needed (i.e., cannot be converted with base types)
        if (
            from_str
            and self.input_type in ConfigInputType.STR_LIST | ConfigInputType.INT_LIST
        ):
            if not value:  # empty string
                self.value = None
                return
            if self.input_type == ConfigInputType.INT_LIST:
                map_type = int
            else:
                map_type = str
            try:
                value_list = list(map(str.strip, value.split(",")))
                # Convert to list of appropriate type
                value = map(map_type, value_list)
            except ValueError as e:
                logger.error(
                    f"Value for {self.name} must be a comma-separated list of {map_type.__name__}: {e}."
                )
                debug_logger.error(
                    f"Value for {self.name} must be a comma-separated list of {map_type.__name__}: {e}.",
                    exc_info=True,
                )
                return

        # Get target type
        target_type = self.get_type()
        # Check type
        try:
            self.value = target_type(value)
        except ValueError as e:
            logger.error(
                f"Value for {self.name} must be of type {target_type.__name__}: {e}."
            )
            debug_logger.error(
                f"Value for {self.name} must be of type {target_type.__name__}: {e}.",
                exc_info=True,
            )
            return
        except:
            logger.error(f"Unexpected error while setting value for {self.name}.")
            debug_logger.error(
                f"Unexpected error while setting value for {self.name}.", exc_info=True
            )
            return


@dataclass
class ConfigGroup:
    """Dataclass to hold a group of configurations."""

    name: str
    _parent = None

    def __post_init__(self):
        """Post-initialization to set parent for all fields."""
        for f in fields(self):
            field = getattr(self, f.name)
            if isinstance(field, ConfigField) or isinstance(field, ConfigGroup):
                field.set_parent(self)

    def get_parent(self) -> Union["ConfigGroup", None]:
        """Get the parent group for this configuration group."""
        return self._parent

    def set_parent(self, parent: "ConfigGroup") -> None:
        """Set the parent group for this configuration group."""
        self._parent = parent

    def get_fields(self, recursive: bool = True) -> List[ConfigKey]:
        """Get a list of all immediate children of the group. If recursive, get a list of keys for the config, where each key is a list of subkeys forming a path."""
        if not recursive:
            return sorted(
                [
                    ConfigKey([f.name])
                    for f in fields(self)
                    if f.name not in ["name", "parent"]
                ]
            )
        results = []
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            field = getattr(self, f.name)
            if isinstance(field, ConfigField):
                results.append(ConfigKey([f.name]))
            elif isinstance(field, ConfigGroup):
                # Recursively get subfields
                keys = field.get_fields(recursive=True)
                for key in keys:
                    key.add_parent(f.name)
                results.extend(keys)
        return sorted(results)

    def get_field(self, key: ConfigKey) -> Union[ConfigField, "ConfigGroup"] | None:
        """Get a config dataclass from the config data, reading the key like a hierarchical path."""
        parent_config = self
        for subkey in key:
            result = getattr(parent_config, subkey, None)
            if result is None or not (
                isinstance(result, ConfigField) or isinstance(result, ConfigGroup)
            ):
                logger.error(f"Setting {key} was not found.")
                return None
            else:
                parent_config = result
        if isinstance(parent_config, (ConfigField, ConfigGroup)):
            return parent_config
        else:
            logger.warning(f"Setting {key} is not a valid ConfigField or ConfigGroup.")
            return None

    def set_field(self, key: ConfigKey, value: Any, from_str: bool = False) -> None:
        """Set a ConfigField value in the config data, reading the key like a path."""
        parent_config = self
        for subkey in key:
            field = getattr(self, subkey, None)
            if field is None:
                logger.error(f"Setting {key} was not found.")
                return
            else:
                parent_config = field
        if isinstance(parent_config, ConfigField):
            parent_config.set_value(value, from_str=from_str)
        else:
            logger.warning(
                f"Setting {key} is not a valid ConfigField and cannot be set."
            )

    def save_config(self, path: Path) -> None:
        """Save the configuration to a file."""
        try:
            with open(path, "w") as f:
                for field in self.get_fields(recursive=True):
                    config_field = self.get_field(field)
                    if isinstance(config_field, ConfigField):
                        f.write(f"{field}: {config_field.get_value_as_str()}\n")
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            debug_logger.error(f"Failed to save config to {path}: {e}", exc_info=True)

    def generate_commands(self) -> List[str]:
        """Generate a list of commands based on the current configuration."""
        commands = []
        for key in self.get_fields(recursive=True):
            field = self.get_field(key)
            command_name = ".".join(key)
            command_value = field.get_value_as_str()
            commands.append(f"{command_name}={command_value}")
        return commands
