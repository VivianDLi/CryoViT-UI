"""Define an item model supporting settings for the CryoViT GUI application."""

from PyQt6.QtCore import QSettings, QModelIndex

from cryovit_gui.models.config_model import ConfigModel
from cryovit_gui.config import ConfigGroup, ConfigField, ConfigKey

#### Logging Setup ####

import logging

logger = logging.getLogger("cryovit.models.settings")


class SettingsModel(ConfigModel):
    """
    Model for managing settings in the CryoViT GUI application.

    This model extends the ConfigModel to provide functionality specific to application settings.
    It allows for easy retrieval and modification of settings data.
    """

    def __init__(self, config: ConfigGroup, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def _reset_settings(self, parent: QModelIndex = None) -> None:
        if parent is None or not parent.isValid():
            # At, root, recursively reset child settings
            parent_config = self._config
        else:
            parent_config = parent.internalPointer()
        for i, config_key in enumerate(parent_config.get_fields(recursive=False)):
            config = parent_config.get_field(config_key)
            if isinstance(config, ConfigField):
                config.set_value(config.default)
                index = self.index(i, 1, parent)
                self.dataChanged.emit(index, index)
            elif isinstance(config, ConfigGroup):
                index = self.index(i, 0, parent)
                self._reset_settings(index)
            else:
                logger.warning(
                    f"Unsupported type {type(config)} for resetting settings. Skipping."
                )

    def reset_settings(self) -> None:
        """Reset all settings to their default values."""
        # Change python data
        self._reset_settings()

        # Reset saved settings on disk
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        settings.clear()
        settings.sync()

        logger.info("Settings have been reset to default values.")

    def save_settings(
        self, name: str = "", force: bool = False, log: bool = True
    ) -> None:
        """Save the settings to the QSettings object."""
        settings = QSettings(
            "Stanford University_Wah Chiu", "CryoViT" if not name else "CryoViT_" + name
        )
        for key in self._config.get_fields(recursive=True):
            value = self._config.get_field(key).get_value_as_str()
            settings.setValue(str(key), value)
        if force:
            settings.sync()
        if log:
            logger.info(
                f"Settings have been saved to disk{(' to preset: ' + name) if name else ''}."
            )

    def _load_settings(
        self, settings, parent: QModelIndex = None, parent_key: ConfigKey = None
    ) -> None:
        if parent is None or not parent.isValid():
            # At root, recursively load child settings
            parent_config = self._config
            parent = QModelIndex()
        else:
            parent_config = parent.internalPointer()
        for i, config_key in enumerate(parent_config.get_fields(recursive=False)):
            config = parent_config.get_field(config_key)
            if parent_key is not None:
                config_key.add_parent(str(parent_key))
            if isinstance(config, ConfigField):
                value = settings.value(str(config_key), None)
                if value is not None:
                    config.set_value(value, from_str=True)
                    index = self.index(i, 1, parent)
                    self.dataChanged.emit(index, index)
                else:
                    logger.warning(
                        f"Setting {config_key} not found in disk settings. Skipping load."
                    )
            elif isinstance(config, ConfigGroup):
                index = self.index(i, 0, parent)
                self._load_settings(settings, index, config_key)
            else:
                logger.warning(
                    f"Unsupported type {type(config)} for loading settings. Skipping."
                )

    def load_settings(self, name: str = "", log: bool = True) -> None:
        """Load the settings from the QSettings object."""
        settings = QSettings(
            "Stanford University_Wah Chiu", "CryoViT" if not name else "CryoViT_" + name
        )
        self._load_settings(settings)
        if log:
            logger.info(
                f"Settings have been loaded from disk{(' with preset: ' + name) if name else ''}."
            )
