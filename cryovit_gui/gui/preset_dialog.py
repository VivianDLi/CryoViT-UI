"""UI dialog window for managing presets in CryoViT."""

from typing import List

from PyQt6.QtWidgets import QDialog, QMessageBox, QDialogButtonBox

import cryovit_gui.resources
from cryovit_gui.layouts.presetdialog import Ui_Dialog


class PresetDialog(QDialog, Ui_Dialog):
    """A dialog for adding, removing, saving, and loading presets."""

    def __init__(
        self,
        parent,
        title,
        *presets,
        current_preset: str = None,
        load_preset: bool = False,
    ):
        """Initialize the PresetDialog.

        Args:
            parent: The parent Qt widget of the dialog.
            title: The title of the dialog window.
            *presets: The list of available presets loaded from existing settings.
            current_preset (str, optional): The current preset to select. Defaults to None.
            load_preset (bool, optional): Whether to load the selected preset (True) or save it (False). Defaults to False.
        """
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(title)
        self.result = None
        # UI setup
        self.presetSelect.addItems(presets)
        if current_preset:
            self.result = current_preset
            index = self.presetSelect.findText(current_preset)
            if index != -1:
                self.presetSelect.setCurrentIndex(index)
            else:
                parent.log(
                    "warning",
                    f"Preset '{current_preset}' not found in the list of presets.",
                )
        else:
            self.presetSelect.setCurrentIndex(0)
            self.result = self.presetSelect.itemText(0)
        self.presetName.returnPressed.connect(self._add_preset)
        self.presetName.returnPressed.disconnect(self._remove_preset)
        self.presetAdd.clicked.connect(self._add_preset)
        self.presetRemove.clicked.connect(self._remove_preset)
        # Remove the ability to add new presents if loading a preset
        if load_preset:
            self.presetName.returnPressed.disconnect(self._add_preset)
            self.presetName.returnPressed.connect(self._remove_preset)
            self.presetAdd.setVisible(False)

    def showEvent(self, event):
        super().showEvent(event)
        # Disable using 'Enter' to close the dialog
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setDefault(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setDefault(False)

    def get_presets(self) -> List[str]:
        """Get the list of presets."""
        return [self.presetSelect.itemText(i) for i in range(self.presetSelect.count())]

    def _add_preset(self):
        """Add a new preset to the list. If the preset already exists, a warning is shown."""
        try:
            preset_name = self.presetName.text()
            if not preset_name:
                return
            if preset_name in [
                self.presetSelect.itemText(i) for i in range(self.presetSelect.count())
            ]:
                QMessageBox.warning(
                    self, "Warning", f"Preset '{preset_name}' already exists."
                )
                self.presetSelect.setCurrentText(preset_name)
                self.presetName.clear()
                return
            self.presetSelect.addItem(preset_name)
            self.presetSelect.setCurrentText(preset_name)
            self.presetName.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error adding preset: {e}")
            return

    def _remove_preset(self):
        """Remove the selected preset from the list. If no preset is specified, nothing happens. If the preset isn't found, a warning is shown."""
        try:
            preset_name = self.presetName.text()
            if not preset_name:
                return
            index = self.presetSelect.findText(preset_name)
            if index == -1:
                QMessageBox.warning(
                    self, "Warning", f"Preset '{preset_name}' not found."
                )
                return
            self.presetSelect.removeItem(index)
            self.presetName.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing preset: {e}")
            return

    def accept(self):
        """Override accept to set the result to the selected preset and close the dialog."""
        index = self.presetSelect.currentIndex()
        self.result = self.presetSelect.itemText(index) if index != -1 else None
        super().accept()

    def reject(self):
        """Override reject to set the result to None and close the dialog."""
        self.result = None
        super().reject()
