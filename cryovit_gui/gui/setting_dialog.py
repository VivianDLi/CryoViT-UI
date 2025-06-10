"""UI dialog window for viewing, editing, saving, and loading settings."""

from PyQt6.QtWidgets import QDialog

import cryovit_gui.resources
from cryovit_gui.layouts.settingswindow import Ui_SettingsWindow
from cryovit_gui.models import SettingsModel


class SettingsWindow(QDialog, Ui_SettingsWindow):
    """A freestanding dialog window for viewing and editing current settings (e.g., file/folder locations, DINOv2 feature settings)."""

    def __init__(self, parent, model: SettingsModel):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Settings")
        self.settingsView.setModel(model)
