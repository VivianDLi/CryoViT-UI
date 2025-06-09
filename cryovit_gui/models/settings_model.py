"""Define an item model supporting settings for the CryoViT GUI application."""

from PyQt6.QtCore import QSettings

from cryovit_gui.models.config_model import ConfigModel
from cryovit_gui.config import ConfigGroup

## Setup logging ##
import logging

logger = logging.getLogger("cryovit.models.settings")


class SettingsModel(ConfigModel):
    """
    Model for managing settings in the CryoViT GUI application.

    This model extends the ConfigModel to provide functionality specific to application settings.
    It allows for easy retrieval and modification of settings data.
    """

    def __init__(self, config: ConfigGroup):
        super().__init__(config)

    def reset_settings(self) -> None:
        """Reset all settings to their default values."""
        # Change python data
        for key in self._data.get_fields(recursive=True):
            self._data.set_field(key, self._data.get_field(key).default)
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, 0))

        # Reset saved settings on disk
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        settings.clear()
        settings.sync()

        logger.info("Settings have been reset to default values.")

    def save_settings(self, name: str = "CryoViT") -> None:
        """Save the settings to the QSettings object."""
        settings = QSettings("Stanford University_Wah Chiu", name)
        for key in self._data.get_fields(recursive=True):
            value = self._data.get_field(key).get_value_as_str()
            settings.setValue(str(key), value)

        logger.info("Settings have been saved to disk.")

    def load_settings(self, name: str = "CryoViT") -> None:
        """Load the settings from the QSettings object."""
        settings = QSettings("Stanford University_Wah Chiu", name)
        for key in self._data.get_fields(recursive=True):
            value = settings.value(str(key), None)
            if value is not None:
                self._data.set_field(key, value, from_str=True)
            else:
                logger.warning(
                    f"Setting {key} not found in disk settings. Skipping load."
                )
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, 0))

        logger.info("Settings have been loaded from disk.")
