"""Classes for viewing, editing, saving, and loading UI settings."""

import dataclasses
from typing import List, cast

from PyQt6.QtWidgets import (
    QDialog,
    QMessageBox,
    QDialogButtonBox,
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
from PyQt6.QtCore import QSettings

import cryovit.gui.resources
from cryovit.gui.layouts.settingswindow import Ui_SettingsWindow
from cryovit.gui.layouts.presetdialog import Ui_Dialog


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


class SettingsWindow(QDialog, Ui_SettingsWindow):
    """A freestanding "window" for viewing and editing current settings (e.g., file/folder locations, DINOv2 feature settings)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Settings")
        # Load settings
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        self.settings = Settings()
        for key in settings.allKeys():
            if key in [f.name for f in dataclasses.fields(self.settings)]:
                self.set_setting(key, settings.value(key))
        # Create the settings UI dynamically from the dataclass
        self.createSettingsUI(self.settings, self)

    def createSettingsUI(
        self,
        parent: "BaseSetting",
        parent_widget: QGroupBox | QScrollArea,
        parent_name: str = None,
    ):
        """Recursively create settings Qt widgets from Settings dataclasses."""
        for field in dataclasses.fields(parent):
            # Save widgets as variables to prevent garbage collection
            variable_name = (
                parent_name + "__" + field.name if parent_name else field.name
            )  # double underscore to avoid replacing parts of the name later
            if isinstance(getattr(parent, field.name), BaseSetting):
                # Create a group box for the field with a vertical layout and initial form layout
                group_box = QGroupBox(
                    parent=parent_widget, title=field.name.capitalize()
                )
                group_box.setObjectName(f"groupBox_{variable_name}")
                group_vbox = QVBoxLayout(group_box)
                group_vbox.setContentsMargins(0, 0, 0, 0)
                group_vbox.setObjectName(f"verticalLayout_{variable_name}")
                form_frame = QFrame(parent=group_box)
                form_frame.setFrameShape(QFrame.Shape.StyledPanel)
                form_frame.setFrameShadow(QFrame.Shadow.Raised)
                form_frame.setObjectName(f"frame_{variable_name}")
                group_form = QFormLayout(form_frame)
                group_form.setObjectName(f"formLayout_{variable_name}")
                parent_layout = (
                    self.verticalLayoutScroll
                    if parent_name is None
                    else parent_widget.findChild(
                        QVBoxLayout, f"verticalLayout_{parent_name}"
                    )
                )
                group_vbox.addWidget(form_frame)
                parent_layout.addWidget(group_box)
                # Prevent garbage collection of objects
                setattr(self, f"groupBox_{variable_name}", group_box)
                setattr(self, f"verticalLayout_{variable_name}", group_vbox)
                setattr(self, f"frame_{variable_name}", form_frame)
                setattr(self, f"formLayout_{variable_name}", group_form)
                # Recursively create settings UI for the field
                self.createSettingsUI(
                    getattr(parent, field.name), group_box, variable_name
                )
                continue
            # Get the parent form layout (skipping the first level)
            if parent_name is None:
                continue
            parent_form: QFormLayout = parent_widget.findChild(
                QFormLayout, f"formLayout_{parent_name}"
            )
            settings_path = variable_name.replace("__", "/")
            label = QLabel(
                parent=parent_widget,
                text=" ".join(map(str.capitalize, field.name.split("_"))) + ":",
            )
            # Create value widget based on the field type
            match field.type.__qualname__:
                case str.__qualname__:
                    data = QLineEdit(
                        parent=parent_widget,
                        text=self.get_setting(settings_path),
                    )
                case int.__qualname__:
                    data = QSpinBox(parent=parent_widget)
                    data.setValue(self.get_setting(settings_path))
                    data.setKeyboardTracking(False)
                    data.setRange(0, 10000)
                    data.setSingleStep(1)
                case bool.__qualname__:
                    data = QCheckBox(parent=parent_widget)
                    data.setChecked(self.get_setting(settings_path))
                case List.__qualname__:
                    data = QLineEdit(
                        parent=parent_widget,
                        text=", ".join(
                            map(
                                str.strip,
                                self.settings.get_setting(settings_path),
                            )
                        ),
                    )
                case _:
                    self.parent.log(
                        "warning",
                        f"Unsupported type: {field.type} for {settings_path}. Ignoring this field.",
                    )
            # Prevent garbage collection of objects
            setattr(self, f"label_{variable_name}", label)
            setattr(self, f"data_{variable_name}", data)
            parent_form.addRow(label, data)

    def showEvent(self, event):
        super().showEvent(event)
        # Disable using 'Enter' to close the dialog
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setDefault(False)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setDefault(False)
        # Update setting values
        self.update_UI(self.settings)

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
        """Save the settings to the QSettings object. If a key is specified, only that setting is saved.

        Args:
            key (str, optional): The key to save. If None, all settings are saved. Defaults to None.
        """
        settings = QSettings("Stanford University_Wah Chiu", "CryoViT")
        if key:
            settings.setValue(key, self.settings.get_setting(key))
        else:
            for field in self.settings.get_available_settings():
                settings.setValue(field, self.settings.get_setting(field))

    def validate_settings(self, parent: "BaseSetting", parent_name: str = None):
        """Recursively validate and save settings inputted in the UI to the settings object.

        Args:
            parent (BaseSetting): The parent settings dataclass to validate to.
            parent_name (str, optional): The name of the parent settings object. Defaults to None to signify the base settings class.
        """
        try:
            for field in dataclasses.fields(parent):
                variable_name = (
                    parent_name + "__" + field.name if parent_name else field.name
                )
                if isinstance(getattr(parent, field.name), BaseSetting):
                    # Recursively validate settings for the settings group
                    self.validate_settings(getattr(parent, field.name), variable_name)
                    continue
                # Get values from the UI and add to settings
                settings_path = variable_name.replace("__", "/")
                match field.type.__qualname__:
                    case str.__qualname__:
                        data = str(getattr(self, f"data_{variable_name}").text())
                    case int.__qualname__:
                        data = int(getattr(self, f"data_{variable_name}").value())
                    case bool.__qualname__:
                        data = bool(getattr(self, f"data_{variable_name}").isChecked())
                    case List.__qualname__:
                        data = list(
                            map(
                                str.strip,
                                getattr(self, f"data_{variable_name}")
                                .text()
                                .split(","),
                            )
                        )
                    case _:
                        self.parent.log(
                            "warning",
                            f"Unsupported type: {field.type} for {settings_path}. Ignoring this field.",
                        )
                self.settings.set_setting(settings_path, data)
            return True
        except Exception as e:
            self.parent.log(
                "error",
                f"Error validating settings: {e}\n",
            )
            return False

    def update_UI(self, parent: "BaseSetting", parent_name: str = None):
        """Recursively update the UI with the current settings."""
        try:
            for field in dataclasses.fields(parent):
                variable_name = (
                    parent_name + "__" + field.name if parent_name else field.name
                )
                if isinstance(getattr(parent, field.name), BaseSetting):
                    self.update_UI(getattr(parent, field.name), variable_name)
                    continue
                # Get values from the UI and add to settings
                settings_path = variable_name.replace("__", "/")
                value = self.settings.get_setting(settings_path)
                match field.type.__qualname__:
                    case str.__qualname__:
                        data = getattr(self, f"data_{variable_name}")
                        data.setText(str(value))
                    case int.__qualname__:
                        data = getattr(self, f"data_{variable_name}")
                        data.setValue(int(value))
                    case bool.__qualname__:
                        data = getattr(self, f"data_{variable_name}")
                        data.setChecked(bool(value))
                    case List.__qualname__:
                        data = getattr(self, f"data_{variable_name}")
                        data.setText(", ".join(map(str.strip, value)))
                    case _:
                        self.parent.log(
                            "warning",
                            f"Unsupported type: {field.type} for {settings_path}. Ignoring this field.",
                        )
        except Exception as e:
            self.parent.log(
                "error",
                f"Error resetting UI settings: {e}\n",
            )

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

    def reject(self):
        """Override reject to revert the UI settings and close the dialog."""
        self.update_UI(self.settings)
        super().reject()


## Settings Dataclasses


@dataclasses.dataclass
class BaseSetting:
    """Base dataclass for storing settings."""

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
        """Set a setting in the settings data, reading the key like a path."""
        paths = key.split("/")
        if len(paths) == 1:
            setattr(self, paths[0], value)
            return
        parent_dir, child_dir = paths[0], paths[1:]
        getattr(self, parent_dir).set_setting("/".join(child_dir), value)


@dataclasses.dataclass
class GeneralSettings(BaseSetting):
    """Dataclass to hold general settings relating to the GUI application."""

    data_directory: str = ""


@dataclasses.dataclass
class PresetSettings(BaseSetting):
    """Dataclass to hold preset settings."""

    current_preset: str = ""
    available_presets: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PreprocessingSettings(BaseSetting):
    """Dataclass to hold preprocessing settings."""

    bin_size: int = 2
    resize_image: List[str] = dataclasses.field(default_factory=lambda: ["512", "512"])
    normalize: bool = True
    clip: bool = True


@dataclasses.dataclass
class ModelSettings(BaseSetting):
    """Dataclass to hold model settings."""

    model_directory: str = ""


@dataclasses.dataclass
class DinoSettings(BaseSetting):
    """Dataclass to hold DINO settings."""

    model_directory: str = ""
    features_directory: str = ""
    batch_size: int = 128


@dataclasses.dataclass
class SegmentationSettings(BaseSetting):
    """Dataclass to hold segmentation settings."""

    batch_size: int = 1
    csv_file: str = ""


@dataclasses.dataclass
class AnnotationSettings(BaseSetting):
    """Dataclass to hold annotation settings."""

    chimera_path: str = ""
    num_slices: int = 5


@dataclasses.dataclass
class TrainingSettings(BaseSetting):
    """Dataclass to hold training settings."""

    splits_file: str = ""
    number_of_splits: int = 10
    batch_size: int = 1
    random_seed: int = 42


@dataclasses.dataclass
class Settings(BaseSetting):
    """Dataclass to hold general settings relating to the GUI application."""

    general: GeneralSettings = dataclasses.field(default_factory=GeneralSettings)
    preset: PresetSettings = dataclasses.field(default_factory=PresetSettings)
    preprocessing: PreprocessingSettings = dataclasses.field(
        default_factory=PreprocessingSettings
    )
    model: ModelSettings = dataclasses.field(default_factory=ModelSettings)
    dino: DinoSettings = dataclasses.field(default_factory=DinoSettings)
    segmentation: SegmentationSettings = dataclasses.field(
        default_factory=SegmentationSettings
    )
    annotation: AnnotationSettings = dataclasses.field(
        default_factory=AnnotationSettings
    )
    training: TrainingSettings = dataclasses.field(default_factory=TrainingSettings)
