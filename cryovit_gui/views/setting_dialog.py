"""UI dialog window for viewing, editing, saving, and loading settings."""

from dataclasses import fields
from functools import partial
from typing import List
import logging

from PyQt6.QtWidgets import (
    QDialog,
    QMessageBox,
    QFrame,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QLabel,
    QLineEdit,
    QSpinBox,
    QCheckBox,
)
from PyQt6.QtCore import QSettings, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

import cryovit_gui.resources
from cryovit_gui.layouts.settingswindow import Ui_SettingsWindow
from config import BaseSetting, Settings, SettingField, SettingsInputType
from utils import select_file_folder_dialog
from ..gui.clickable_line import ClickableLineEdit

logger = logging.getLogger(__name__)


class SettingsWindow(QDialog, Ui_SettingsWindow):
    """A freestanding dialog window for viewing and editing current settings (e.g., file/folder locations, DINOv2 feature settings)."""

    @staticmethod
    def reset_settings():
        """Reset the settings to the default values."""
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        settings.clear()
        settings.sync()

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Settings")
        # Load settings
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        self.settings = Settings()
        for key in settings.allKeys():
            if key in self.settings.get_available_settings():
                self.set_setting(key, settings.value(key))
        # Create the settings UI dynamically from the dataclass
        self.variables = {}
        self.createSettingsUI(self.settings, self)

    def createSettingsUI(
        self,
        parent: BaseSetting,
        parent_widget: QGroupBox | QScrollArea,
        parent_key: str = None,
    ):
        """Recursively create SettingsWindow UI from Settings dataclass."""
        for f in fields(parent):
            settings_path = parent_key + "/" + f.name if parent_key else f.name
            field = getattr(parent, f.name)
            self.variables[settings_path] = self._generate_variable_name()
            if isinstance(field, SettingField):
                # Get the parent form layout (skipping the first level)
                if parent_key is None:
                    continue
                parent_form: QFormLayout = parent_widget.findChild(
                    QFormLayout, f"formLayout_{self.variables[parent_key]}"
                )
                # Create label for the field
                label = QLabel(parent=parent_widget, text=field.name + ":")
                label.setToolTip(field.description)
                data = self._create_UI_element(parent_widget, field)
                # Prevent garbage collection of objects
                setattr(self, f"label_{self.variables[settings_path]}", label)
                setattr(self, f"data_{self.variables[settings_path]}", data)
                parent_form.addRow(label, data)
            elif isinstance(field, BaseSetting):
                # Create a group box with a vertical layout and initial form layout
                g_box = QGroupBox(parent=parent_widget, title=f.name.capitalize())
                g_vbox = QVBoxLayout(g_box)
                g_vbox.setContentsMargins(0, 0, 0, 0)
                g_vbox.setObjectName(f"verticalLayout_{self.variables[settings_path]}")
                form_frame = QFrame(parent=g_box)
                form_frame.setFrameShape(QFrame.Shape.StyledPanel)
                form_frame.setFrameShadow(QFrame.Shadow.Raised)
                group_form = QFormLayout(form_frame)
                group_form.setObjectName(f"formLayout_{self.variables[settings_path]}")
                parent_layout = (
                    self.verticalLayoutScroll
                    if parent_key is None
                    else parent_widget.findChild(
                        QVBoxLayout, f"verticalLayout_{self.variables[parent_key]}"
                    )
                )
                g_vbox.addWidget(form_frame)
                parent_layout.addWidget(g_box)
                # Prevent garbage collection of objects
                setattr(self, f"groupBox_{self.variables[settings_path]}", g_box)
                setattr(self, f"verticalLayout_{self.variables[settings_path]}", g_vbox)
                setattr(self, f"frame_{self.variables[settings_path]}", form_frame)
                setattr(self, f"formLayout_{self.variables[settings_path]}", group_form)
                # Recursively create settings UI for the field
                self.createSettingsUI(
                    field,
                    g_box,
                    settings_path,
                )
            else:
                logger.warning(
                    f"Unsupported type: {f.type} for {settings_path}. Ignoring this field."
                )

    def get_available_settings(self) -> List[str]:
        """Get a list of available settings."""
        return self.settings.get_available_settings()

    def get_setting(self, key: str, as_str: bool = False):
        """Get a setting from the settings data, reading the key like a path."""
        return self.settings.get_setting(key, as_str=as_str)

    def set_setting(self, key: str, value):
        """Set a setting in the settings data."""
        self.settings.set_setting(key, value)

    def validate_settings(self, parent: BaseSetting, parent_name: str = None):
        """Recursively validate and save settings inputted in the UI to the settings object."""
        try:
            for f in fields(parent):
                settings_path = parent_name + "/" + f.name if parent_name else f.name
                field = getattr(parent, f.name)
                if isinstance(field, SettingField):
                    data = self._get_from_UI_element(field, settings_path)
                    self.set_setting(settings_path, data)
                elif isinstance(field, BaseSetting):
                    # If the field is a BaseSetting, recursively validate its settings
                    self.validate_settings(field, settings_path)
                else:
                    logger.warning(
                        f"Unsupported type: {f.type} for {settings_path}. Ignoring this field."
                    )
            return True
        except Exception as e:
            logger.error(
                f"Error validating settings: {e}. Please ensure all fields are filled out correctly."
            )
            return False

    def save_settings(self, key: str = None):
        """Save the settings to the QSettings object. If a key is specified, only that setting is saved.

        Args:
            key (str, optional): The key to save. If None, all settings are saved. Defaults to None.
        """
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        if key:
            settings.setValue(key, self.get_setting(key, as_str=True))
        else:
            for field in self.settings.get_available_settings():
                settings.setValue(field, self.get_setting(field, as_str=True))

    def accept(self):
        """Override accept to save the settings and close the dialog."""
        if self.validate_settings(self.settings):
            self.save_settings()
            super().accept()
        else:
            QMessageBox.warning(
                self,
                "Invalid Settings",
                "Please esnure all fields are filled out correctly.",
            )

    def _generate_variable_name(self) -> str:
        """Generate a variable name for the setting field."""
        return str(len(self.variables))

    def _create_UI_element(self, parent_widget, settings_field: SettingField):
        """Create a UI element for the setting field."""
        # Create value widget based on the field type
        match settings_field.input_type:
            case SettingsInputType.FILE:
                data = ClickableLineEdit(
                    parent=parent_widget,
                    text=settings_field.get_value_as_str(),
                )
                data.clicked.connect(
                    partial(
                        self._file_directory_prompt,
                        data,
                        settings_field.name,
                        False,
                    )
                )
            case SettingsInputType.DIRECTORY:
                data = ClickableLineEdit(
                    parent=parent_widget,
                    text=settings_field.get_value_as_str(),
                )
                data.clicked.connect(
                    partial(
                        self._file_directory_prompt,
                        data,
                        settings_field.name,
                        True,
                    )
                )
            case SettingsInputType.TEXT:
                data = QLineEdit(
                    parent=parent_widget,
                    text=settings_field.get_value_as_str(),
                )
            case SettingsInputType.NUMBER:
                data = QSpinBox(parent=parent_widget)
                data.setRange(0, 10000)
                data.setSingleStep(1)
                data.setValue(settings_field.get_value())
                data.setKeyboardTracking(False)
            case SettingsInputType.BOOL:
                data = QCheckBox(parent=parent_widget)
                data.setChecked(settings_field.get_value())
            case SettingsInputType.STR_LIST | SettingsInputType.INT_LIST:
                data = QLineEdit(
                    parent=parent_widget,
                    text=settings_field.get_value_as_str(),
                )
                if settings_field.input_type == SettingsInputType.INT_LIST:
                    # Add a validator to ensure only integers are entered
                    reg = QRegularExpression(r"^(\d+,?)+$")
                    data.setValidator(QRegularExpressionValidator(reg, parent_widget))
            case _:
                logger.warning(
                    f"Unsupported input type: {settings_field.input_type} for {settings_field.name}. Ignoring this field."
                )
        return data

    def _get_from_UI_element(
        self, settings_field: SettingField, settings_path: str
    ) -> str:
        """Get the value from the UI element."""
        match settings_field.input_type:
            case (
                SettingsInputType.FILE
                | SettingsInputType.DIRECTORY
                | SettingsInputType.TEXT
                | SettingsInputType.STR_LIST
                | SettingsInputType.INT_LIST
            ):
                return getattr(self, f"data_{self.variables[settings_path]}").text()
            case SettingsInputType.NUMBER:
                return str(
                    getattr(self, f"data_{self.variables[settings_path]}").value()
                )
            case SettingsInputType.BOOL:
                return str(
                    getattr(self, f"data_{self.variables[settings_path]}").isChecked()
                )
            case _:
                logger.warning(
                    f"Unsupported input type: {settings_field.input_type} for {settings_field.name}. Ignoring this field."
                )
                return ""

    def _file_directory_prompt(
        self,
        data: ClickableLineEdit | QLineEdit,
        name: str,
        is_folder: bool,
    ):
        """Open a file or directory selection dialog and set the selected path to the data widget."""
        start_dir = data.text() if data.text() else ""
        selected_path = select_file_folder_dialog(
            self, f"Select {name}", is_folder, False, start_dir=start_dir
        )
        data.setText(selected_path)
