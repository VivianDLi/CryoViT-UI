"""UI dialog window for managing presets in CryoViT."""

from PyQt6.QtWidgets import QDialog, QMessageBox, QDialogButtonBox

import cryovit_gui.resources
from cryovit_gui.layouts.presetdialog import Ui_Dialog
from cryovit_gui.config import ConfigKey
from cryovit_gui.models import SettingsModel

#### Setup logging ####
import logging

logger = logging.getLogger("cryovit.gui.preset_dialog")
debug_logger = logging.getLogger("debug")


class PresetDialog(QDialog, Ui_Dialog):
    """A dialog for adding, removing, saving, and loading presets."""

    def __init__(
        self,
        parent,
        title: str,
        available_presets: list[str],
        current_preset: str | None = None,
        load_preset: bool = False,
    ):
        """Initialize the PresetDialog.

        Args:
            parent: The parent Qt widget of the dialog.
            title: The title of the dialog window.
            available_presets: The list of available presets loaded from existing settings.
            current_preset (str, optional): The current preset to select. Defaults to None.
            load_preset (bool, optional): Whether to load the selected preset (True) or save it (False). Defaults to False.
        """
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(title)
        self.results = [current_preset, available_presets]
        # UI setup
        self.presetAdd.clicked.connect(self._add_preset)
        self.presetRemove.clicked.connect(self._remove_preset)
        self.presetSelect.addItems(available_presets)
        self.presetSelect.currentTextChanged.connect(self._set_preset)
        # Remove the ability to add new presents if loading a preset
        if load_preset:
            self.presetName.returnPressed.connect(self._remove_preset)
            self.presetAdd.setVisible(False)
        else:
            self.presetName.returnPressed.connect(self._add_preset)

        if current_preset is not None:
            index = self.presetSelect.findText(current_preset)
            if index != -1:
                self.presetSelect.setCurrentIndex(index)
            else:
                logger.warning(
                    f"Current preset '{current_preset}' not found in available presets. Adding to available presets."
                )
                self.presetSelect.addItem(current_preset)
                self.presetSelect.setCurrentText(current_preset)
                self.results[1].append(current_preset)
        else:
            self.presetSelect.setCurrentIndex(0)

    def showEvent(self, event):
        super().showEvent(event)
        # Disable using 'Enter' to close the dialog
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setDefault(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setDefault(False)

    def _add_preset(self):
        """Add a new preset to the list. If the preset already exists, a warning is shown."""
        try:
            preset_name = self.presetName.text()
            available_presets = self.results[1]
            if preset_name in available_presets:
                QMessageBox.warning(
                    self, "Warning", f"Preset '{preset_name}' already exists."
                )
                self.presetSelect.setCurrentText(preset_name)
                self.presetName.clear()
                return
            self.presetSelect.addItem(preset_name)
            available_presets.append(preset_name)
            self.presetSelect.setCurrentText(preset_name)
            self.presetName.clear()
        except Exception as e:
            logger.error(f"Error adding preset: {e}")
            debug_logger.error(f"Error adding preset: {e}.", exc_info=True)

    def _remove_preset(self):
        """Remove the selected preset from the list. If no preset is specified, nothing happens. If the preset isn't found, a warning is shown."""
        try:
            preset_name = self.presetName.text()
            available_presets = self.results[1]
            index = self.presetSelect.findText(preset_name)
            if index == -1:
                QMessageBox.warning(
                    self, "Warning", f"Preset '{preset_name}' not found."
                )
                return
            self.presetSelect.removeItem(index)
            available_presets.remove(preset_name)
            self.presetName.clear()
        except Exception as e:
            logger.error(f"Error removing preset: {e}")
            debug_logger.error(f"Error removing preset: {e}", exc_info=True)

    def _set_preset(self, preset_name: str, *args):
        self.results[0] = preset_name
