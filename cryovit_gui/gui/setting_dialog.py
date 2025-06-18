"""UI dialog window for viewing, editing, saving, and loading settings."""

from PyQt6.QtWidgets import QDialog, QHeaderView

import cryovit_gui.resources
from cryovit_gui.layouts.settingswindow import Ui_SettingsWindow
from cryovit_gui.models import SettingsModel, ConfigDelegate


class SettingsWindow(QDialog, Ui_SettingsWindow):
    """A freestanding dialog window for viewing and editing current settings (e.g., file/folder locations, DINOv2 feature settings)."""

    def __init__(self, parent, model: SettingsModel):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Settings")
        self.model = model
        self.settingsView.setModel(self.model)
        self.settingsView.setItemDelegate(ConfigDelegate(self.settingsView))
        self.settingsView.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.defaultButton.clicked.connect(self.model.reset_settings)
