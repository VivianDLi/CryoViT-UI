"""Subwindow for configuring model parameters and training settings."""

from typing import Dict, Union
from PyQt6.QtWidgets import (
    QDialog,
    QMessageBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QInputDialog,
)

from cryovit.config import InterfaceModelConfig, models, ModelArch, TrainerFit
from cryovit.gui.layouts.modeldialog import Ui_ModelDialog


class ModelDialog(QDialog, Ui_ModelDialog):
    """Subwindow for configuring model parameters and training settings."""

    def __init__(
        self,
        parent,
        model_config: InterfaceModelConfig,
        trainer_config: TrainerFit = None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Model Config:")
        self.config = model_config
        self.trainer_config = trainer_config
        self.param_dict = []
        # Setup UI
        self.archCombo.addItems(models)
        self.paramAdd.clicked.connect(self.add_param)
        self.paramRemove.clicked.connect(self.remove_param)
        if self.trainer_config:
            self.trainerConfigGroup.setVisible(True)
        else:
            self.trainerConfigGroup.setVisible(False)
        # Update UI elements with current model configuration
        self.update_UI(self.config, self.trainer_config)

    def add_param(self):
        name, ok = QInputDialog.getText(
            self,
            "Add Parameter",
            "Enter the name of the parameter:",
            QLineEdit.EchoMode.Normal,
        )
        if not ok or not name:
            return
        value_type, ok = QInputDialog.getItem(
            self,
            "Add Parameter",
            "Enter the type of the parameter:",
            ["String", "Integer", "Float"],
            editable=False,
        )
        if not ok or not value_type:
            return
        match value_type:
            case "String":
                value = ""
            case "Integer":
                value = 0
            case "Float":
                value = 0.0
            case _:
                value = ""
        self._add_param(name, value)

    def add_params(self, params: Dict[str, Union[str, int, float]]):
        for name, value in params.items():
            self._add_param(name, value)

    def _add_param(self, name: str, value: Union[str, int, float]):
        self.param_dict.append({"name": name, "value": value})
        index = len(self.param_dict) - 1
        param_name = QLineEdit(name)
        param_name.editingFinished.connect(
            lambda: self.param_dict[index].update({"name": param_name.text()})
        )
        match type(value).__qualname__:
            case str.__qualname__:
                param_value = QLineEdit(str(value))
                param_value.editingFinished.connect(
                    lambda: self.param_dict[index].update({"value": param_value.text()})
                )
            case int.__qualname__:
                param_value = QSpinBox()
                param_value.setValue(int(value))
                param_value.valueChanged.connect(
                    lambda: self.param_dict[index].update(
                        {"value": param_value.value()}
                    )
                )
            case float.__qualname__:
                param_value = QDoubleSpinBox()
                param_value.setValue(float(value))
                param_value.valueChanged.connect(
                    lambda: self.param_dict[index].update(
                        {"value": param_value.value()}
                    )
                )
            case _:
                self.parent.log(
                    "warning",
                    f"Unsupported type for parameter value: {type(value)}. Ignoring this parameter.",
                )
                self.param_dict.pop(index)
                return
        # Prevent garbage collection
        setattr(self, f"paramLabel_{index}", param_name)
        setattr(self, f"paramValue_{index}", param_value)
        self.paramLayout.insertRow(
            self.paramLayout.rowCount() - 1, param_name, param_value
        )

    def remove_param(self):
        name, ok = QInputDialog.getText(
            self,
            "Remove Parameter",
            "Enter the name of the parameter to remove:",
            QLineEdit.EchoMode.Normal,
        )
        if not ok or not name:
            return
        param_names = [p["name"] for p in self.param_dict]
        if name in param_names:
            index = param_names.index(name)
            self.param_dict.pop(index)
            self.paramLayout.removeRow(index)
            delattr(self, f"paramLabel_{index}")
            delattr(self, f"paramValue_{index}")

    def validate_config(self):
        try:
            self.config.name = self.nameDisplay.text()
            self.label_key = self.labelDisplay.text()
            self.config.model_type = ModelArch[self.archCombo.currentText()]
            self.config.model_params = {
                p["name"]: p["value"] for p in self.param_dict if p["name"]
            }
            self.config.samples = (
                list(map(str.strip, self.samplesDisplay.text().split(",")))
                if self.samplesDisplay.text()
                else []
            )
            if not self.config.samples:
                self.parent.log("error", "Samples cannot be empty.")
                return False

            if self.trainer_config:
                self.trainer_config.accelerator = self.accelCombo.currentText()
                self.trainer_config.devices = self.devicesDisplay.text()
                self.trainer_config.precision = self.precisionCombo.currentText()
                self.trainer_config.max_epochs = self.epochSpin.value()
                self.trainer_config.log_every_n_steps = self.loggingSpin.value()
            return True
        except Exception as e:
            self.parent.log("error", f"Error validating configuration: {e}")
            return False

    def update_UI(
        self,
        model_config: InterfaceModelConfig,
        trainer_config: Union[TrainerFit, None],
    ):
        """Update the UI with the new model configuration."""
        self.config = model_config
        self.trainer_config = trainer_config
        self.nameDisplay.setText(self.config.name)
        self.labelDisplay.setText(self.config.label_key)
        self.archCombo.setCurrentText(self.config.model_type.name)
        self.add_params(self.config.model_params)
        self.samplesDisplay.setText(
            ", ".join(self.config.samples) if self.config.samples else ""
        )
        if self.trainer_config:
            self.accelCombo.setCurrentText(self.trainer_config.accelerator)
            self.devicesDisplay.setText(str(self.trainer_config.devices))
            self.precisionCombo.setCurrentText(self.trainer_config.precision)
            self.epochSpin.setValue(self.trainer_config.max_epochs)
            self.loggingSpin.setValue(self.trainer_config.log_every_n_steps)

    def accept(self):
        """Override accept to validate and save the model configuration."""
        if self.validate_config():
            super().accept()
        else:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Please ensure all required fields are filled out correctly.",
            )
