"""Subwindow for viewing and editing current settings (e.g., file/folder locations, DINOv2 feature settings)."""

import dataclasses
from typing import List, cast

from PyQt6.QtWidgets import (
    QDialog,
    QMessageBox,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QCheckBox,
)
from PyQt6.QtCore import QSettings

import cryovit.gui.resources
from cryovit.gui.layouts.settingswindow import Ui_SettingsWindow
from cryovit.gui.layouts.presetdialog import Ui_Dialog


class PresetDialog(QDialog, Ui_Dialog):
    def __init__(self, parent, title, *presets, current_preset: str = None):
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
            self.presetSelect.setCurrentIndex(0)
            self.result = self.presetSelect.itemText(0)
        self.presetSelect.currentIndexChanged.connect(self._set_result_from_index)
        self.presetName.returnPressed.connect(self._add_preset)
        self.presetAdd.clicked.connect(self._add_preset)
        self.presetRemove.clicked.connect(self._remove_preset)

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

    def _set_result_from_index(self, index):
        """Set the result of the dialog to the selected preset."""
        self.result = self.presetSelect.itemText(index) if index != -1 else None


class SettingsWindow(QDialog, Ui_SettingsWindow):
    """A freestanding "window" for viewing and editing current settings (e.g., file/folder locations, DINOv2 feature settings)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Settings")
        # Disable using 'Enter' to close the dialog
        # Load settings
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        self.settings = Settings()
        for key in settings.allKeys():
            self.set_setting(key, settings.value(key))

    def createSettingsUI(self):
        for field in dataclasses.fields(self.settings):
            group_box = QGroupBox(parent=self, title=field.name.capitalize())
            group_box.setObjectName(f"groupBox_{field.name}")
            form_layout = QFormLayout(parent=group_box)
            form_layout.setObjectName(f"formLayout_{field.name}")
            for i, subfield in enumerate(
                dataclasses.fields(getattr(self.settings, field.name))
            ):
                label = QLabel(parent=group_box, text=subfield.name.capitalize())
                match subfield.type.__qualname__:
                    case str.__qualname__:
                        data = QLineEdit(
                            parent=group_box,
                            text=self.get_setting(field.name + "/" + subfield.name),
                        )
                        data.setObjectName(f"lineEdit_{field.name}_{subfield.name}")
                        data.editingFinished.connect(
                            lambda: self.set_setting(
                                field.name + "/" + subfield.name, data.text()
                            )
                        )
                    case int.__qualname__:
                        data = QSpinBox(parent=group_box)
                        data.setValue(
                            self.get_setting(field.name + "/" + subfield.name)
                        )
                        data.setObjectName(f"spinBox_{field.name}_{subfield.name}")
                        data.valueChanged.connect(
                            lambda: self.set_setting(
                                field.name + "/" + subfield.name, data.value()
                            )
                        )
                    case bool.__qualname__:
                        data = QCheckBox(parent=group_box)
                        data.setChecked(
                            self.get_setting(field.name + "/" + subfield.name)
                        )
                        data.setObjectName(f"checkBox_{field.name}_{subfield.name}")
                        data.toggled.connect(
                            lambda: self.set_setting(
                                field.name + "/" + subfield.name, data.isChecked()
                            )
                        )
                    case List.__qualname__:
                        data = QLineEdit(
                            parent=group_box,
                            text=", ".join(
                                map(
                                    str.strip,
                                    self.settings.get_setting(
                                        field.name + "/" + subfield.name
                                    ),
                                )
                            ),
                        )
                        data.setObjectName(f"lineEdit_{field.name}_{subfield.name}")
                        data.editingFinished.connect(
                            lambda: self.set_setting(
                                field.name + "/" + subfield.name,
                                list(map(str.strip, data.text().split(","))),
                            )
                        )
                    case _:
                        self.parent.log(
                            "warning",
                            f"Unsupported type: {subfield.type} for {field.name}/{subfield.name}. Ignoring this field.",
                        )
                form_layout.setWidget(i, QFormLayout.ItemRole.LabelRole, label)

    def showEvent(self, event):
        super().showEvent(event)
        # Disable using 'Enter' to close the dialog
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setDefault(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setDefault(False)

    def get_available_settings(self) -> List[str]:
        """Get a list of available settings."""
        return self.settings.get_available_settings()

    def get_setting(self, key: str):
        """Get a setting from the settings data, reading the key like a path."""
        return self.settings.get_setting(key)

    def set_setting(self, key: str, value):
        """Set a setting in the settings data."""
        self.settings.set_setting(key, value)

    def save_settings(self, key: str = None):
        """Save the settings to the QSettings object. If a key is specified, only that setting is saved."""
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        if key:
            settings.setValue(key, self.settings.get_setting(key))
        else:
            for field in self.settings.get_available_settings():
                settings.setValue(field, self.settings.get_setting(field))

    def accept(self):
        """Override accept to save the settings and close the dialog."""
        self.save_settings()
        super().accept()


## Settings Dataclasses


@dataclasses.dataclass
class BaseSetting:
    def get_available_settings(self) -> List[str]:
        """Get a list of available settings."""
        results = []
        for f in dataclasses.fields(self):
            if not isinstance(getattr(self, f.name), BaseSetting):
                results.append(f.name)
                continue
            subfields = getattr(self, f.name).get_available_settings()
            results.extend([f"{f.name}/{subfield}" for subfield in subfields])
        return sorted(results)

    def get_setting(self, key: str):
        """Get a setting from the settings data, reading the key like a path."""
        paths = key.split("/")
        if len(paths) == 1:
            fields = {field.name: field.type for field in dataclasses.fields(self)}
            result_type = fields[paths[0]]
            result = getattr(self, paths[0])
            # Check for lists/non-base types
            if isinstance(result, List):
                return cast(List, result)
            else:
                # Basic types
                return result_type(result)
        parent_dir, child_dir = paths[0], paths[1:]
        result = getattr(self, parent_dir).get_setting("/".join(child_dir))
        return result

    def set_setting(self, key: str, value) -> None:
        """Set a setting in the settings data."""
        paths = key.split("/")
        if len(paths) == 1:
            setattr(self, paths[0], value)
            return
        parent_dir, child_dir = paths[0], paths[1:]
        getattr(self, parent_dir).set_setting("/".join(child_dir), value)


@dataclasses.dataclass
class GeneralSettings(BaseSetting):
    """Dataclass to hold general settings relating to the GUI application."""

    data_dir: str = ""
    dino_dir: str = ""
    features_dir: str = ""


@dataclasses.dataclass
class PresetSettings(BaseSetting):
    """Dataclass to hold preset settings."""

    current_preset: str = ""
    available_presets: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PreprocessingSettings(BaseSetting):
    """Dataclass to hold preprocessing settings."""

    bin_size: int = 2
    normalize: bool = True
    clip: bool = True


@dataclasses.dataclass
class ModelSettings(BaseSetting):
    model_dir: str = ""


@dataclasses.dataclass
class SegmentationSettings(BaseSetting):
    batch_size: int = 1
    csv_file: str = ""


@dataclasses.dataclass
class AnnotationSettings(BaseSetting):
    chimerax_path: str = ""
    num_slices: int = 5


@dataclasses.dataclass
class TrainingSettings(BaseSetting):
    splits_file: str = ""
    splits: int = 10
    batch_size: int = 1
    split_id: int = 0
    split_seed: int = 42


@dataclasses.dataclass
class Settings(BaseSetting):
    """Dataclass to hold general settings relating to the GUI application."""

    general: GeneralSettings = dataclasses.field(default_factory=GeneralSettings)
    preset: PresetSettings = dataclasses.field(default_factory=PresetSettings)
    preprocessing: PreprocessingSettings = dataclasses.field(
        default_factory=PreprocessingSettings
    )
    model: ModelSettings = dataclasses.field(default_factory=ModelSettings)
    segmentation: SegmentationSettings = dataclasses.field(
        default_factory=SegmentationSettings
    )
    annotation: AnnotationSettings = dataclasses.field(
        default_factory=AnnotationSettings
    )
    training: TrainingSettings = dataclasses.field(default_factory=TrainingSettings)
