"""Main application window for CryoVit segmentation and training."""

import os
import sys
import platform
import traceback
from typing import Any, List, Tuple
from pathlib import Path
import shutil
import psutil
from functools import partial
import pyperclip

import pandas as pd

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QInputDialog,
    QMessageBox,
)
from PyQt6.QtCore import (
    Qt,
    QUrl,
    QTimer,
    QProcess,
    QRunnable,
    QThreadPool,
    QObject,
    pyqtSignal,
)
from PyQt6.QtGui import QDesktopServices, QGuiApplication

import cryovit_gui.resources
from cryovit_gui.layouts.mainwindow import Ui_MainWindow
from cryovit_gui.gui import PresetDialog, SettingsWindow
from cryovit_gui.config import (
    ConfigKey,
    FileData,
    tomogram_exts,
    preprocess_command,
    train_command,
    evaluate_command,
    inference_command,
)
from cryovit_gui.utils import TextEditLogger, select_file_folder_dialog
from cryovit_gui.configs import (
    PreprocessingConfig,
    TrainingConfig,
    EvaluationConfig,
    InferenceConfig,
    Settings,
)
from cryovit_gui.models import (
    ConfigModel,
    SettingsModel,
    FileModel,
    SampleModel,
    TomogramModel,
    ConfigDelegate,
)
from cryovit_gui.processing import (
    add_annotations,
    create_dataset,
    generate_slices,
    generate_training_splits,
    get_all_tomogram_files,
    chimera_script_path,
)

#### Logging Setup ####

import logging
import json

logging.config.dictConfig(json.load("logging.conf"))
logger = logging.getLogger("cryovit")
debug_logger = logging.getLogger("debug")


def filter_maker(level):
    level = getattr(logging, level)

    def filter(record):
        return record.levelno <= level

    return filter


class WorkerSignals(QObject):
    """Signals for background threads to communicate with the GUI."""

    finish = pyqtSignal(
        Any
    )  # signals when the thread is finished, containing any return value
    error = pyqtSignal(tuple)  # signals when the thread errors


class Worker(QRunnable):
    """Thread for running long processing tasks in the background."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.result = self.fn(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finish.emit(self.result if hasattr(self, "result") else None)


class MainWindow(QMainWindow, Ui_MainWindow):
    """The main UI window for CryoViT. Handles the main application logic and UI interactions."""

    # Signals for background and external threads and processes
    def _handle_thread_exception(
        self, info: Tuple[type[BaseException], BaseException, str], *args
    ):
        """Handle exceptions in background threads.
        Args:
            info (Tuple[type[BaseException], BaseException, str]): Tuple containing the exception type, value, and traceback.
        """
        exctype, value, traceback_info = info
        logger.error(f"Error in thread: {exctype}: {value}.\n{traceback_info}")

    def _handle_process_exception(self, error: QProcess.ProcessError, *args):
        """Handle exceptions in QProcess.
        Args:
            error (QProcess.ProcessError): The error that occurred in the process.
        """
        if error == QProcess.ProcessError.FailedToStart:
            logger.error("Process failed to start. Please check the command and path.")
        elif error == QProcess.ProcessError.Crashed:
            logger.error("Process crashed unexpectedly.")
        elif error == QProcess.ProcessError.Timedout:
            logger.error("Process timed out.")
        else:
            logger.error(f"Process encountered an error: {error}")

    # UI initialization
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        # Setup thread pool and processes
        # Limit threads based on memory, assuming 5 GB per thread
        thread_count = max(1, psutil.virtual_memory().available // (5 * 10 ^ 9))
        self.threadpool: QThreadPool = QThreadPool(thread_count=thread_count)
        self.progress_dict = {}  # progress bar update dict
        self.chimera_process = None

        # Setup data models
        preprocessing_config = PreprocessingConfig()
        training_config = TrainingConfig()
        evaluation_config = EvaluationConfig()
        inference_config = InferenceConfig()
        settings_config = Settings()

        self.preprocessing_model = ConfigModel(preprocessing_config)
        self.training_model = ConfigModel(training_config)
        self.evaluation_model = ConfigModel(evaluation_config)
        self.inference_model = ConfigModel(inference_config)
        self.settings_model = SettingsModel(settings_config)
        try:
            self.settings_model.load_settings()
        except ValueError as e:
            logger.warning(f"Error loading settings: {e}\nResetting to defaults.")
            self.settings_model.reset_settings()

        # Setup uninitialized models (wait for user input)
        self.file_model = None
        self.sample_model = None
        self.tomogram_model = None

        # Setup UI elements
        try:
            self.setup_preprocessing()
            self.setup_annotation()
            self.setup_training()
            self.setup_evaluation()
            self.setup_inference()
            self.setup_settings()
            self.setup_menu()
            self.setup_console()
        except Exception as e:
            logger.critical(f"Error setting up UI: {e}")
            sys.exit(1)
        logger.info("Welcome to CryoViT!")

    def setup_preprocessing(self):
        """Setup the preprocessing tab in the main window."""
        self.processView.setModel(self.preprocessing_model)
        self.processView.setItemDelegate(ConfigDelegate(self.processView))
        self.processButton.clicked.connect(self.run_preprocessing)

    def run_preprocessing(self, *args):
        """Generate CLI command to run preprocessing from a terminal."""
        # Get config from view model
        commands = ["-m", preprocess_command]
        commands += self.processView.model().generate_commands()
        command = "python " + " ".join(commands)
        # Copy to clipboard
        pyperclip.copy(command)
        logger.info(f"Copied pre-processing command to clipboard:\n{command}")

    def setup_annotation(self):
        """Setup the annotation tab in the main window."""
        # Setup root directory selection
        self.directoryButton.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.projectDirectory,
                "project root directory",
                True,
                False,
            )
        )
        self.projectDirectory.editingFinished.connect(self._update_root_directory)

        # Setup annotation
        self.selectAllButton.clicked.connect(partial(self._select_tomograms, "all"))
        self.deselectAllButton.clicked.connect(partial(self._select_tomograms, "none"))
        self.resetDefaultButton.clicked.connect(
            partial(self._select_tomograms, "default")
        )
        self.chimeraButton.clicked.connect(self.run_chimerax)
        self.slicesButton.clicked.connect(self.run_slice_generation)
        self.featuresButton.clicked.connect(self._import_features)
        self.featuresDisplay.editingFinished.connect(self._update_features)
        self.annotButton.clicked.connect(self.run_add_annotations)

        # Setup sync timer
        self.syncTimer = QTimer(self)
        self.syncTimer.timeout.connect(self._sync_files)
        self.syncTimer.start(1000)  # Sync every second

        # Setup training split generation
        self.splitsButton.clicked.connect(self.run_training_splits_generation)

    def _update_root_directory(self, *args):
        root_dir = Path(self.projectDirectory.text()).resolve()
        # Check for existing file model
        if self.file_model is not None:
            self.file_model.set_root(root_dir)
        else:
            # First initialization
            self.file_model = FileModel(root_dir)
            self.sample_model = SampleModel(self.file_model)
            self.tomogram_model = TomogramModel(self.file_model)

            self.sampleView.setModel(self.sample_model)
            self.sampleView.setSelectionMode(Qt.ItemSelectionMode.SingleSelection)
            self.sampleView.setSelectionBehavior(Qt.SelectionBehavior.SelectRows)
            self.sampleView.selectionModel().currentChanged.connect(
                lambda current, _: self.tomogram_model.setSample(
                    self.sample_model.data(current, Qt.ItemDataRole.UserRole).value()
                )
            )
            self.fileView.setModel(self.tomogram_model)

    def _select_tomograms(self, action: str, *args):
        if (
            self.file_model is None
            or self.tomogram_model is None
            or self.tomogram_model.sample is None
        ):
            logger.warning(
                "No file model initialized. Please set the project root directory first."
            )
            return
        match action:
            case "all":
                for row in range(self.tomogram_model.rowCount()):
                    index = self.tomogram_model.index(row, 0)
                    if index.isValid():
                        self.tomogram_model.setData(
                            index, True, Qt.ItemDataRole.EditRole
                        )
            case "none":
                for row in range(self.tomogram_model.rowCount()):
                    index = self.tomogram_model.index(row, 0)
                    if index.isValid():
                        self.tomogram_model.setData(
                            index, False, Qt.ItemDataRole.EditRole
                        )
            case _:
                data = self.file_model.read_data()
                annotated = data[self.tomogram_model.sample].annotated
                for row in range(self.tomogram_model.rowCount()):
                    index = self.tomogram_model.index(row, 0)
                    if index.isValid():
                        self.tomogram_model.setData(
                            index, annotated[row], Qt.ItemDataRole.EditRole
                        )

    def _import_features(self, *args):
        feature_file, ok = select_file_folder_dialog(
            self,
            "Select Features File",
            "JSON files (*.json)",
            False,
            False,
            start_dir=str(
                self.settings_model.get_config_by_key(
                    ConfigKey(["general", "data_directory"])
                ).get_value()
            ),
        )
        if not ok or not feature_file:
            return
        try:
            feature_data = json.load(feature_file)
            feature_data = sorted(
                feature_data, key=lambda x: x["png_index"], order="desc"
            )
            features = [f["name"] for f in feature_data]
            self.featuresDisplay.setText(", ".join(features))
            self.featuresDisplay.editingFinished.emit()
        except json.JSONDecodeError as e:
            logger.error(f"Error loading features file: {e}")
            debug_logger.error(f"Error loading features file: {e}", exc_info=True)
            return

    def _update_features(self, *args):
        str_features = self.featuresDisplay.text()
        self.features = [f.strip() for f in str_features.split(",")]

    def _sync_files(self, *args):
        """Update the file model and views based on the current root directory."""
        if self.file_model is not None:
            self.syncIcon = (
                None  # TODO: Implement sync icon timer with self._update_files()
            )
            data_worker = Worker(self.file_model.read_data)
            data_worker.signals.finish.connect(self._update_files)
            data_worker.signals.error.connect(
                partial(self._handle_thread_exception, "Update Files")
            )
            self.threadpool.start(data_worker)

    def _update_files(self, data: FileData, *args):
        if self.file_model.update_data(data, update_selection=False):
            logger.info("File model updated with new data.")

    def run_chimerax(self, *args):
        """Launch ChimeraX externally to select z-limits and tomogram slices to label (and create .csv file)."""
        # Check for settings
        chimera_path = self.settings_model.get_config(
            ConfigKey(["annotation", "chimera_path"])
        ).get_value()
        if not chimera_path and platform.system().lower() != "linux":
            logger.warning("ChimeraX path not set. Please set it in the settings.")
            return
        num_slices = self.settings_model.get_config_by_key(
            ConfigKey(["annotation", "num_slices"])
        ).get_value()

        # Check for directories
        if self.file_model is None or not self.file_model.validate_root():
            logger.warning("No root directory specified.")
            return
        sample = self.sample_model.sample
        if not sample:
            logger.warning("No sample selected.")
            return
        tomogram_dir = self.file_model.get_directory("tomograms")
        csv_dir = self.file_model.get_directory("csv")
        slices_dir = self.file_model.get_directory("slices")
        if not all([tomogram_dir, csv_dir, slices_dir]):
            logger.warning(
                "One or more required directories (tomograms, csv, slices) are not available."
            )
            return

        # Get tomograms to process
        tomograms = [
            self.tomogram_model.data(
                self.tomogram_model.index(row, 0), Qt.ItemDataRole.DisplayRole
            ).value()
            for row in range(self.tomogram_model.rowCount())
            if self.tomogram_model.data(
                self.tomogram_model.index(row, 0), Qt.ItemDataRole.CheckStateRole
            )
            == Qt.CheckState.Checked
        ]

        # Get ChimeraX command
        process, command = self._create_chimerax_command(
            chimera_path,
            tomogram_dir,
            sample,
            tomograms=tomograms,
            dst_dir=slices_dir,
            csv_dir=csv_dir,
            num_slices=num_slices,
        )
        if process is None or command is None:
            logger.error("Failed to create ChimeraX command.")
            return
        else:
            # Copy start command to clipboard
            pyperclip.copy(command)
            # Setup QProcess to run ChimeraX
            self.chimera_process = QProcess()
            self.chimera_process.finished.connect(self._sync_files)
            self.chimera_process.errorOccured.connect(
                partial(self._handle_process_exception, "ChimeraX Process")
            )
            self.chimera_process.start(process)

    def _create_chimerax_command(
        self,
        chimera_path: Path,
        tomogram_dir: Path,
        sample: str,
        tomograms: List[str] = None,
        dst_dir: Path = None,
        csv_dir: Path = None,
        num_slices: int = 5,
    ) -> Tuple[str, str]:
        if self.chimera_process is not None:
            return None, None  # ChimeraX process already running
        # Create script args
        commands = [
            "open",
            chimera_script_path,
            ";",
            "start slice labels",
            str(tomogram_dir.resolve()),
            sample,
            "num_slices",
            str(num_slices),
        ]
        # Check for optional arguments
        if tomograms:
            commands.extend(["tomograms", "(" + ",".join(tomograms) + ")"])
        if dst_dir:
            commands.extend(["dst_dir", str(dst_dir.resolve())])
        if csv_dir:
            commands.extend(["csv_dir", str(csv_dir.resolve())])
        command = " ".join(commands)
        # Validate ChimeraX path based on OS
        match platform.system().lower():
            case "windows":  # Needs to be run from a specific path
                # Check for valid path
                if (
                    not os.path.isfile(chimera_path)
                    or not chimera_path.name.lower() == "chimerax.exe"
                ):
                    logger.warning(
                        f"Invalid ChimeraX path: {chimera_path}. This should be the path to the ChimeraX.exe executable typically found in 'C:/Program Files/ChimeraX/bin/ChimeraX.exe'. Please set it in the settings."
                    )
                    process = None
            case "linux":  # Has chimerax from command line
                process = "chimerax"
            case "darwin":  # Needs to be run from a specific path
                if not os.path.isfile(chimera_path):
                    logger.warning(
                        f"Invalid ChimeraX path: {chimera_path}. This should be in chimerax_install_dir/Contents/MacOS/ChimeraX where 'chimerax_install_dir' is typically '/Applications/ChimeraX.app'. Please set it in the settings."
                    )
                    process = None
            case _:
                logger.warning(f"Unsupported OS type {platform.system()}.")
                process = None
        return process, command

    def run_slice_generation(self, *args):
        # Check for directories
        if self.file_model is None or not self.file_model.validate_root():
            logger.warning("No root directory specified.")
            return
        sample = self.sample_model.sample
        if not sample:
            logger.warning("No sample selected.")
            return
        tomogram_dir = self.file_model.get_directory("tomograms")
        csv_dir = self.file_model.get_directory("csv")
        slices_dir = self.file_model.get_directory("slices")
        if not all([tomogram_dir, csv_dir, slices_dir]):
            logger.warning(
                "One or more required directories (tomograms, csv, slices) are not available."
            )
            return
        csv_file = csv_dir / f"{sample}.csv"
        if not csv_file.exists():
            logger.warning(
                f"No CSV file found for sample {sample} in {csv_file}. Please run the ChimeraX annotation first."
            )
            return

        # Check completion status
        sample_df = pd.read_csv(csv_file)
        annotated = len(sample_df["tomo_name"])
        total = self.tomogram_model.rowCount()

        if annotated < total:
            continue_slices = QMessageBox.question(
                self,
                "Incomplete Annotations",
                f"Only {annotated} out of {total} tomograms for sample {sample} have been annotated. Are you sure you want to generate slices for this sample?",
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                defaultbutton=QMessageBox.StandardButton.Yes,
            )
            if continue_slices == QMessageBox.StandardButton.No:
                return

        # Setup thread
        slice_worker = Worker(
            partial(generate_slices, tomogram_dir, slices_dir, csv_file)
        )
        slice_worker.signals.finish.connect(
            partial(logger.info, "Slice generation complete.")
        )
        slice_worker.signals.error.connect(
            partial(self._handle_thread_exception, "Generate Slices")
        )
        self.threadpool.start(slice_worker)

    def run_add_annotations(self, *args):
        # Check for settings
        if not self.features:
            logger.warning(
                "No features selected. Please add annotated features in the 'Annotation' tab."
            )
            return

        # Check for directories
        if self.file_model is None or not self.file_model.validate_root():
            logger.warning("No root directory specified.")
            return
        sample = self.sample_model.sample
        if not sample:
            logger.warning("No sample selected.")
            return
        tomogram_dir = self.file_model.get_directory("tomograms")
        csv_dir = self.file_model.get_directory("csv")
        slices_dir = self.file_model.get_directory("slices")
        if not all([tomogram_dir, csv_dir, slices_dir]):
            logger.warning(
                "One or more required directories (tomograms, csv, slices) are not available."
            )
            return
        csv_file = csv_dir / f"{sample}.csv"
        if not csv_file.exists():
            logger.warning(
                f"No CSV file found for sample {sample} in {csv_file}. Please run the ChimeraX annotation first."
            )
            return

        # Check completion status
        annotations_present = [
            self.sample_model.data(
                self.sample_model.index(row, 3), Qt.ItemDataRole.UserRole
            ).value()
            for row in range(self.sample_model.rowCount())
            if self.sample_model.data(
                self.sample_model.index(row, 0), Qt.ItemDataRole.UserRole
            ).value()
            == sample
        ]
        annotated = sum(annotations_present)
        if annotated == 0:
            QMessageBox.warning(
                self,
                "Annotations Required",
                "No annotations found for this sample. Please make sure exported annotations are present in the Annotations folder.",
                buttons=QMessageBox.StandardButton.Ok,
            )
            return

        # Setup thread
        annotations_worker = Worker(
            partial(
                add_annotations,
                tomogram_dir,
                tomogram_dir,
                slices_dir,
                csv_file,
                self.features,
            )
        )
        annotations_worker.signals.finish.connect(
            partial(logger.info, "Add annotations complete.")
        )
        annotations_worker.signals.error.connect(
            partial(self._handle_thread_exception, "Add Annotations")
        )
        self.threadpool.start(annotations_worker)

    def run_training_splits_generation(self, *args):
        """Generate training splits .csv file."""
        # Check for settings
        num_splits = self.training_model.get_config_by_key(
            ["trainer", "num_splits"]
        ).get_value()
        seed = self.training_model.get_config_by_key(
            ["trainer", "random_seed"]
        ).get_value()
        if not self.features:
            logger.warning(
                "No labeled features inputed. Please add annotated features above."
            )
            return

        # Check for directories
        if self.file_model is None or not self.file_model.validate_root():
            logger.warning("No root directory specified.")
            return
        sample = self.sample_model.sample
        if not sample:
            logger.warning("No sample selected.")
            return
        csv_dir = self.file_model.get_directory("csv")
        if not csv_dir:
            logger.warning("Csv directory is not available.")
            return
        csv_file = csv_dir / f"{sample}.csv"
        if not csv_file.exists():
            logger.warning(
                f"No CSV file found for sample {sample} in {csv_file}. Please run the ChimeraX annotation first."
            )
            return

        # Get splits file
        splits_file = select_file_folder_dialog(
            self,
            "Select Splits File:",
            False,
            False,
            "CSV files (*.csv)",
            start_dir=str(csv_dir),
        )
        if not splits_file or not os.path.isfile(splits_file):
            logger.warning("No valid splits file selected.")
            return
        else:
            splits_file = Path(splits_file).resolve()

        # Setup thread
        splits_worker = Worker(
            partial(
                generate_training_splits(
                    splits_file,
                    csv_file,
                    sample=sample,
                    num_splits=num_splits,
                    seed=seed,
                )
            )
        )
        splits_worker.signals.finish.connect(
            partial(logger.info, "Training splits generation complete.")
        )
        splits_worker.signals.error.connect(
            partial(self._handle_thread_exception, "Training Splits")
        )
        self.threadpool.start(splits_worker)

    def setup_training(self):
        """Setup the training tab in the main window."""
        self.trainView.setModel(self.training_model)
        self.trainView.setItemDelegate(ConfigDelegate(self.trainView))
        self.trainButton.clicked.connect(self.run_training)

    def run_training(self, *args):
        """Generate CLI command to run training from a terminal."""
        # Get config from view model
        commands = ["-m", train_command]
        commands += self.trainView.model().generate_commands()
        command = "python " + " ".join(commands)
        # Copy to clipboard
        pyperclip.copy(command)
        logger.info(f"Copied training command to clipboard:\n{command}")

    def setup_evaluation(self):
        """Setup the evaluation tab in the main window."""
        self.evalView.setModel(self.evaluation_model)
        self.evalView.setItemDelegate(ConfigDelegate(self.evalView))
        self.evalButton.clicked.connect(self.run_evaluation)

    def run_evaluation(self, *args):
        """Generate CLI command to run evaluation from a terminal."""
        # Get config from view model
        commands = ["-m", evaluate_command]
        commands += self.evalView.model().generate_commands()
        command = "python " + " ".join(commands)
        # Copy to clipboard
        pyperclip.copy(command)
        logger.info(f"Copied evaluation command to clipboard:\n{command}")

    def setup_inference(self):
        """Setup the inference tab in the main window."""
        self.inferenceView.setModel(self.inference_model)
        self.inferenceView.setItemDelegate(ConfigDelegate(self.inferenceView))
        self.inferenceButton.clicked.connect(self.run_inference)

    def run_inference(self, *args):
        """Generate CLI command to run inference from a terminal."""
        # Get config from view model
        commands = ["-m", inference_command]
        commands += self.inferenceView.model().generate_commands()
        command = "python " + " ".join(commands)
        # Copy to clipboard
        pyperclip.copy(command)
        logger.info(f"Copied inference command to clipboard:\n{command}")

    def _file_directory_prompt(
        self, text_field, name: str, is_folder: bool, is_multiple: bool, *args
    ):
        """Open a file or directory selection dialog."""
        text_field.setText(
            select_file_folder_dialog(
                self,
                f"Select {name} {'directory' if is_folder else 'file'}:",
                is_folder,
                is_multiple,
                start_dir=str(
                    self.settings_model.get_config_by_key(["general", "data_directory"])
                ).get_value_as_str(),
            )
        )

    def setup_menu(self):
        """Setup the menu bar for loading and saving presets."""
        self.actionLoad_Preset.triggered.connect(self.load_preset)
        self.actionSave_Preset.triggered.connect(partial(self.save_preset, True))
        self.actionSaveAs_Preset.triggered.connect(partial(self.save_preset, False))
        self.actionSettings.triggered.connect(self.open_settings)

        self.actionDirectory_Setup.triggered.connect(self.setup_directory)

        self.actionGithub.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/VivianDLi/CryoViT")
            )
        )

    def load_preset(self, *args):
        """Opens a prompt to optionally load previous settings from a saved name."""
        available_presets = self.settings_model.get_config_by_key(
            ConfigKey(["preset", "available_presets"])
        ).get_value()
        if not available_presets:
            # If no presets are available, cancel
            logger.warning("No available presets to load. Cancelling.")
            return

        preset_dialog = PresetDialog(
            self,
            "Load preset",
            self.settings_model.copy(),
            load_preset=True,
        )
        result = preset_dialog.exec()
        if result == preset_dialog.DialogCode.Accepted:
            self.settings_model = preset_dialog.model
            logger.info(
                f"Loaded preset: {self.settings_model.get_config_by_key(ConfigKey(['preset', 'current_preset'])).get_value()}"
            )

    def save_preset(self, replace: bool, *args):
        """Opens a prompt to save the current settings to a preset name."""
        if replace:
            # Save over the current preset without prompting for a name
            current_preset = self.settings_model.get_config_by_key(
                ConfigKey(["preset", "current_preset"])
            ).get_value()
            if not current_preset:
                logger.warning(
                    "No preset name specified. Load an existing preset or create a new one."
                )
                return
        else:
            preset_dialog = PresetDialog(
                self,
                "Save preset",
                self.settings_model.copy(),
                load_preset=False,
            )
            result = preset_dialog.exec()
            if result == preset_dialog.DialogCode.Accepted:
                self.settings_model = preset_dialog.model
                current_preset = self.settings_model.get_config_by_key(
                    ConfigKey(["preset", "current_preset"])
                ).get_value()
            else:
                return  # User cancelled the dialog
        self.settings_model.save_settings("CryoViT_" + current_preset)
        logger.info(
            f"Saved preset: {self.settings_model.get_config_by_key(ConfigKey(['preset', 'current_preset'])).get_value()}"
        )

    def open_settings(self, *args):
        """Open the settings window."""
        window = SettingsWindow(self, self.settings_model.copy())
        result = window.exec()
        if result == self.settings.DialogCode.Accepted:
            self.settings_model = window.settingsView.model()
            self.settings_model.save_settings()
            logger.info("Settings updated successfully.")
        else:
            logger.info("Settings update cancelled.")

    def setup_directory(self, *args):
        """Open file dialogs to setup a data directory for downloaded datasets."""
        # Get the directory where datasets are downloaded
        src_dir = select_file_folder_dialog(
            self,
            "Select downloaded dataset base directory:",
            True,
            False,
            start_dir=str(self.settings.get_setting("general/data_directory")),
        )
        if not src_dir or not os.path.isdir(src_dir):
            logger.warning("No valid directory selected.")
            return
        src_dir = Path(src_dir).resolve()
        search_query, ok = QInputDialog.getText(
            self,
            "Dataset Search Query",
            "Enter the search query for finding tomograms in the downloaded dataset:\n\n"
            + "Note: This is a regex string, where '**' matches to any combination of directories and '*' matches to immediate children\n"
            + "(e.g., '**/Tomograms/**/*' finds all files inside any child of a folder named 'Tomograms' anywhere in the base directory,\n"
            + "and '**/Tomograms/*' only finds the files inside the folder named 'Tomograms').\n"
            + f"Only files with extensions in {tomogram_exts} will be moved to the new directory.",
            text="**/Tomograms/**/*",
        )
        if not ok or not search_query:
            logger.warning(
                "No search query specified. Defaulting to '**/Tomograms/**'."
            )
            search_query = "**/Tomograms/**/*"
        # Get the tomogram files to move
        tomogram_files = get_all_tomogram_files(src_dir, search_query)
        continue_moving = QMessageBox.question(
            self,
            "Move Files?",
            f"Found {len(tomogram_files)} tomograms. Do you want to move them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.Yes,
        )
        if continue_moving == QMessageBox.StandardButton.No:
            return

        # Get the directory where raw tomograms are stored
        data_dir = select_file_folder_dialog(
            self,
            "Select raw tomogram base directory:",
            True,
            False,
            start_dir=str(self.settings.get_setting("general/data_directory")),
        )
        if not data_dir or not os.path.isdir(data_dir):
            logger.warning("No valid directory selected.")
            return
        data_dir = Path(data_dir).resolve()
        sample_name, ok = QInputDialog.getText(
            self, "Dataset Sample Name", "Enter sample name for the downloaded dataset:"
        )
        if not ok or not sample_name:
            logger.warning("No sample name specified.")
            return
        delete_dir = QMessageBox.question(
            self,
            "Delete Directory?",
            "Do you want to delete the existing directory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.No,
        )
        # Create the dataset structure
        logger.info(f"Moving {len(tomogram_files)} tomograms from {src_dir}.")
        create_dataset(tomogram_files, data_dir, sample_name)
        logger.info(f"Dataset structure created at {data_dir / sample_name}.")
        # Clean up the source directory if requested
        if delete_dir == QMessageBox.StandardButton.Yes:
            shutil.rmtree(src_dir)
            logger.info(f"Deleted {src_dir}.")

    def setup_console(self):
        """Setup the UI console to display stdout and stderr messages."""
        console_handler = TextEditLogger(self.consoleText)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(console_handler)

    def closeEvent(self, event):
        """Override close event to save settings."""
        self.syncTimer.stop()
        self.settings_model.save_settings()
        event.accept()


if __name__ == "__main__":
    # Setup application
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
    )

    app = QApplication(sys.argv)
    app.setApplicationName("CryoViT")
    app.setOrganizationName("Stanford University")
    app.setOrganizationDomain("stanford.edu")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    app.exec()
