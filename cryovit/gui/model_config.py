"""Subwindow for configuring model parameters and training settings."""

from PyQt6.QtWidgets import QDialog, QMessageBox, QLineEdit, QInputDialog

from cryovit.config import InterfaceModelConfig, models, ModelArch, TrainerFit
from cryovit.gui.layouts.modeldialog import Ui_ModelDialog


class ModelDialog(QDialog, Ui_ModelDialog):
    """Subwindow for configuring model parameters and training settings."""

    def __init__(self, parent=None, trainer_config: TrainerFit = None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Model Details:")
        self.config = InterfaceModelConfig()
        self.param_dict = []
        self.metrics_dict = []
        # Setup UI
        self.archCombo.addItems(models)
        self.paramAdd.clicked.connect(self.add_param)
        self.paramRemove.clicked.connect(self.remove_param)
        self.metricsAdd.clicked.connect(self.add_metrics)
        self.metricsRemove.clicked.connect(self.remove_metrics)
        self.trainer_config = trainer_config
        if self.trainer_config:
            self.trainerConfigGroup.setVisible(True)
        else:
            self.trainerConfigGroup.setVisible(False)

    def add_param(self):
        param_name = QLineEdit()
        param_value = QLineEdit()
        self.param_dict.append({"name": "", "value": ""})
        index = len(self.param_dict) - 1
        param_name.editingFinished.connect(
            lambda: self.param_dict[index].update({"name": param_name.text()})
        )
        param_value.editingFinished.connect(
            lambda: self.param_dict[index].update({"value": param_value.text()})
        )
        self.metricsFrame.insertRow(
            self.metricsFrame.rowCount() - 1, param_name, param_value
        )

    def remove_param(self):
        name, _ = QInputDialog.getText(
            self,
            "Remove Parameter",
            "Enter the name of the parameter to remove:",
            QLineEdit.EchoMode.Normal,
        )
        if name in [p["name"] for p in self.param_dict]:
            index = [i for i, p in enumerate(self.param_dict) if p["name"] == name][0]
            self.param_dict.pop(index)
            self.metricsFrame.removeRow(index)

    def add_metrics(self):
        metric_name = QLineEdit()
        metric_value = QLineEdit()
        self.metrics_dict.append({"name": "", "value": ""})
        index = len(self.metrics_dict) - 1
        metric_name.editingFinished.connect(
            lambda: self.metrics_dict[index].update({"name": metric_name.text()})
        )
        metric_value.editingFinished.connect(
            lambda: self.metrics_dict[index].update({"value": metric_value.text()})
        )
        self.metricsFrame.insertRow(
            self.metricsFrame.rowCount() - 1, metric_name, metric_value
        )

    def remove_metrics(self):
        name, _ = QInputDialog.getText(
            self,
            "Remove Metric",
            "Enter the name of the metric to remove:",
            QLineEdit.EchoMode.Normal,
        )
        if name in [m["name"] for m in self.metrics_dict]:
            index = [i for i, m in enumerate(self.metrics_dict) if m["name"] == name][0]
            self.metrics_dict.pop(index)
            self.metricsFrame.removeRow(index)

    def validate_config(self):
        try:
            self.config.name = self.nameDisplay.text()
            self.label_key = self.labelDisplay.text()
            self.config.model_type = ModelArch[self.archCombo.currentText()]
            self.config.params = {
                p["name"]: p["value"] for p in self.param_dict if p["name"]
            }
            self.config.samples = list(map(str.strip, self.samplesDisplay.text().split(","))) if self.samplesDisplay.text() else []
            if not self.config.samples:
                raise ValueError("Samples cannot be empty.")
            self.config.metrics = {
                m["name"]: m["value"] for m in self.metrics_dict if m["name"]
            }

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
