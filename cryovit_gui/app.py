"""Main application window for CryoVit segmentation and training."""

import os
from pathlib import Path
import shutil
import psutil
import sys
from functools import partial
from dataclasses import is_dataclass, fields
import platform
import traceback
from typing import List, Tuple
import pyperclip

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QScrollArea,
    QLabel,
    QFormLayout,
    QWidget,
    QInputDialog,
    QMessageBox,
)
from PyQt6.QtCore import (
    QSettings,
    QUrl,
    Qt,
    QProcess,
    QRunnable,
    QThreadPool,
    QObject,
    pyqtSignal,
)
from PyQt6.QtGui import QDesktopServices, QGuiApplication

import cryovit_gui.resources
from cryovit_gui.layouts.mainwindow import Ui_MainWindow
from cryovit_gui.gui import (
    PresetDialog,
    MultiSelectComboBox,
)
from cryovit_gui.views import ModelDialog, SettingsWindow
from utils import (
    EmittingStream,
    select_file_folder_dialog,
)
from config import (
    InterfaceModelConfig,
    ModelArch,
    models,
    ignored_config_keys,
)
from cryovit_gui.configs import *
from cryovit_gui.models import *
from cryovit_gui.processing import *

## Setup logging ##
import logging
import json

logging.config.dictConfig(json.load("logging.conf"))
logger = logging.getLogger("cryovit")
debug_logger = logging.getLogger("debug")


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


def _catch_exceptions(desc: str, concurrent: bool = False):
    """Decorator to catch exceptions when running any UI function.

    Args:
        desc (str): Description of the function being run.
        concurrent (bool): Whether the function is running in a thread.
    """

    def inner(func):
        def wrapper(self, *args, **kwargs):
            try:
                if (
                    concurrent and self.threadpool.activeThreadCount() > 0
                ):  # if another thread is running
                    self.log(
                        "warning"
                        f"Cannot run {desc}: Another process is already running."
                    )
                func(self, *args, **kwargs)
                if (
                    concurrent and self.threadpool.activeThreadCount() > 0
                ):  # set busy cursor
                    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                if (
                    self.chimera_process is not None or self.dino_process is not None
                ):  # if external processes are running
                    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            except Exception as e:
                logger.error(f"Error running {desc}: {e}")
                debug_logger.error(f"Error running {desc}: {e}", exc_info=True)

        return wrapper

    return inner


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

    def _on_thread_finish(self, name: str, *args):
        """Handle progress bar updates, busy cursor, and logging when a background thread finishes.

        Args:
            name (str): Name of the process that finished.
        """
        if name in self.progress_dict:
            self.progress_dict[name]["count"] += 1
            count, total = (
                self.progress_dict[name]["count"],
                self.progress_dict[name]["total"],
            )
            if total > 0:  # Avoid no sample processes
                self._update_progress_bar(count, total)
            if count >= total:
                QGuiApplication.restoreOverrideCursor()
                logger.info(f"{name} complete.")
        else:
            QGuiApplication.restoreOverrideCursor()
            logger.info(f"{name} complete.")

    def _handle_stdout(self, process_name: str, *args):
        """Handle standard output from an external QProcess.
        Args:
            process_name (str): Name of the process.
        """
        data = getattr(self, process_name).readAllStandardOutput()
        stdout = bytes(data).decode("utf-8")
        self.log("info", stdout, end="", use_timestamp=False)

    def _handle_stderr(self, process_name: str, *args):
        """Handle standard error from an external QProcess.
        Args:
            process_name (str): Name of the process.
        """
        data = getattr(self, process_name).readAllStandardError()
        stderr = bytes(data).decode("utf-8")
        self.log("error", stderr, end="", use_timestamp=False)

    def _handle_state_change(
        self, process_name: str, state: QProcess.ProcessState, *args
    ):
        """Handle state changes of an external QProcess.
        Args:
            process_name (str): Name of the process.
            state (QProcess.ProcessState): Current state of the process.
        """
        match state:
            case QProcess.ProcessState.NotRunning:
                debug_logger.debug(f"{process_name} process not running.")
            case QProcess.ProcessState.Starting:
                debug_logger.debug(f"{process_name} process starting.")
            case QProcess.ProcessState.Running:
                debug_logger.debug(f"{process_name} process running.")
            case _:
                debug_logger.warning(f"{process_name} process unknown state.")

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
        settings_config = SettingsConfig()

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
        self.processButton.clicked.connect(self.run_preprocessing)

    def run_preprocessing(self, *args):
        """Generate CLI command to run preprocessing from a terminal."""
        # Get config from view model
        commands = ["-m", preprocess_command]
        commands += self.processView.model().generate_commands()
        preprocess_command = "python " + " ".join(infer_commands)
        # Copy to clipboard
        pyperclip.copy(preprocess_command)
        logger.info(
            f"Copied pre-processing command to clipboard:\n{preprocess_command}"
        )

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
            self.file_model = FileModel(root_dir)
            self.sample_model = SampleModel(self.file_model)

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
        self.file_model.update_data(data, update_selection=False)

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
            self.chimera_process.readyReadStandardOutput.connect(
                partial(self._handle_stdout, self.chimera_process)
            )
            self.chimera_process.readyReadStandardError.connect(
                partial(self._handle_stderr, self.chimera_process)
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
                    self.log(
                        "error",
                        f"Invalid ChimeraX path: {chimera_path}. This should be the path to the ChimeraX.exe executable typically found in 'C:/Program Files/ChimeraX/bin/ChimeraX.exe'. Please set it in the settings.",
                    )
                    process = None
            case "linux":  # Has chimerax from command line
                process = "chimerax"
            case "darwin":  # Needs to be run from a specific path
                if not os.path.isfile(chimera_path):
                    self.log(
                        "error",
                        f"Invalid ChimeraX path: {chimera_path}. This should be in chimerax_install_dir/Contents/MacOS/ChimeraX where 'chimerax_install_dir' is typically '/Applications/ChimeraX.app'. Please set it in the settings.",
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

        # Get tomograms progress
        annotated = self.sample_model.data()
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

        # Check completion status
        annotated = sum([])
        total = self.tomogram_model.rowCount()

        # Setup thread

    def run_add_annotations(self, *args):
        pass

    def run_training_splits_generation(self, *args):
        """Generate training splits .csv file."""
        # Get kwargs from settings
        num_splits = self.settings.get_setting("training/num_splits")
        seed = self.settings.get_setting("training/random_seed")
        if not self.features:
            self.log(
                "warning",
                "No labeled features inputed. Please add annotated features above.",
            )
            return
        # Get directories from settings
        if self.dataDirectoryTrain.text():
            src_dir = Path(self.dataDirectoryTrain.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if self.annoDirectory.text():
            annot_dir = Path(self.annoDirectory.text()).resolve()
        else:
            self.log("warning", "No annotation directory specified.")
            return
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text()).resolve()
        else:
            self.log("warning", "No CSV directory specified.")
            return
        # Check for samples
        samples = self.sampleSelectCombo.getCurrentData()
        if len(samples) == 0:  # add end of src_dir as sample
            samples = [src_dir.name]
            src_dir = src_dir.parent
            self.dataDirectoryTrain.setText(str(src_dir))
            self.sampleSelectCombo.setCurrentData(samples)
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
            self.log(
                "warning",
                "No valid splits file selected.",
            )
            return
        else:
            splits_file = Path(splits_file).resolve()

        # Setup thread callbacks
        finish_callback = partial(self._on_thread_finish, "Annotations")
        self.progress_dict["Annotations"] = {"count": 0, "total": len(samples) * 2}
        # For multi-sample training splits to use multithreading
        for sample in samples:
            if not os.path.isdir(src_dir / sample):
                self.log(
                    "error",
                    f"Invalid raw directory: {src_dir / sample}, skipping sample.",
                )
                continue
            annotation_worker = Worker(
                partial(
                    add_annotations,
                    src_dir=src_dir / sample,
                    dst_dir=src_dir / sample,
                    annot_dir=annot_dir / sample,
                    csv_file=csv_dir / f"{sample}.csv",
                    features=self.features,
                )
            )
            split_worker = Worker(
                partial(
                    add_splits,
                    splits_file=splits_file,
                    csv_file=csv_dir / f"{sample}.csv",
                    sample=sample,
                    num_splits=num_splits,
                    seed=seed,
                )
            )
            annotation_worker.signals.finish.connect(finish_callback)
            split_worker.signals.finish.connect(finish_callback)
            annotation_worker.signals.error.connect(self._handle_thread_exception)
            split_worker.signals.error.connect(self._handle_thread_exception)
            # Run both in parallel with all samples for multithreading
            self.threadpool.start(annotation_worker)
            self.threadpool.start(split_worker)

    @_catch_exceptions("feature extraction", concurrent=True)
    def run_feature_extraction(self, *args):
        """Calculate DINO features for tomograms."""
        # Check for existing processes
        if self.dino_process is not None:
            self.log(
                "warning",
                "Cannot run feature extraction: Another process is already running.",
            )
            return
        # Get directories from settings
        if self.dataDirectoryTrain.text():
            src_dir = Path(self.dataDirectoryTrain.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text()).resolve()
        else:
            self.log("warning", "No CSV directory specified.")
            return
        dino_dir = self.settings.get_setting("dino/model_directory")
        features_dir = self.settings.get_setting("dino/features_directory")
        if not dino_dir:
            self.log(
                "warning",
                f"Missing DINO directory: {dino_dir}. This is where the DINOv2 model will be saved. Please set it in the settings.",
            )
            return
        if not features_dir:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the features directory in the settings.",
            )
            features_dir = src_dir
        dino_batch_size = self.settings.get_setting("dino/batch_size")

        # Setup DINOv2 command
        dino_config = DinoFeaturesConfig(
            dino_dir=dino_dir,
            tomo_dir=src_dir,
            csv_dir=csv_dir,
            feature_dir=features_dir,
            batch_size=dino_batch_size,
            sample=self.sampleSelectCombo.getCurrentData(),
        )
        dino_commands = ["-m", "cryovit.dino_features"]
        dino_commands += self._create_command_from_config(
            dino_config, ignored_config_keys
        )
        dino_commands += ["hydra.mode=RUN"]
        dino_command = "python " + " ".join(dino_commands)
        # Copy to clipboard
        pyperclip.copy(dino_command)
        self.log("info", f"Copied DINO command to clipboard:\n{dino_command}")

    @_catch_exceptions("segmentation", concurrent=True)
    def run_segmentation(self, *args):
        """Segments a tomogram folder using multiple models."""
        # Check for existing processes
        if self.dino_process is not None or self.segment_process is not None:
            self.log(
                "warning",
                "Cannot run training: Another process is already running.",
            )
            return
        # Get directories from settings
        if self.dataDirectory.text():
            src_dir = Path(self.dataDirectory.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if self.replaceCheckboxSeg.isChecked():
            if self.replaceDirectorySeg.text():
                dst_dir = Path(self.replaceDirectorySeg.text()).resolve()
            else:
                self.log("warning", "No replace directory specified.")
                return
        else:
            self.replaceDirectorySeg.setText(str(src_dir))
            dst_dir = src_dir
        model_dir = self.settings.get_setting("model/model_directory")
        features_dir = self.settings.get_setting("dino/features_directory")
        if not model_dir:
            self.log(
                "warning",
                f"Missing model directory: {model_dir}. Please set it in the settings.",
            )
            return
        if not features_dir:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the features directory in the settings.",
            )
            features_dir = src_dir
        batch_size = self.settings.get_setting("segmentation/batch_size")

        ## Setup segmentation command
        # Get model list
        model_names = ",".join(
            [
                '"' + str(page) + '"'
                for page in self._models
                if self.modelTabs.indexOf(self._models[page]["widget"]) != -1
            ]
        )
        exp_paths = ExpPaths(exp_dir=dst_dir, tomo_dir=features_dir, split_file=None)
        dataset_config = Inference()
        infer_config = InferModelConfig(
            models=[],
            trainer=TrainerInfer(),
            dataset=dataset_config,
            exp_paths=exp_paths,
        )
        infer_commands = ["-m", "cryovit.infer_model"]
        infer_commands += ["dataset=inference"]
        infer_commands += self._create_command_from_config(
            infer_config,
            ignored_config_keys + ["trainer"],
        )
        # Add in additional settings
        infer_commands += [
            # f"dataloader.batch_size={batch_size}",
            f"models=[{model_names}]",
            "hydra.mode=RUN",
        ]
        infer_command = "python " + " ".join(infer_commands)
        # Copy to clipboard
        pyperclip.copy(infer_command)
        self.log("info", f"Copied Segmentation command to clipboard:\n{infer_command}")

    @_catch_exceptions("training", concurrent=True)
    def run_training(self, *args):
        """Train a selected model with selected training samples."""
        # Check for existing processes
        if self.dino_process is not None or self.train_process is not None:
            self.log(
                "warning",
                "Cannot run training: Another process is already running.",
            )
            return
        if not self.train_model_config:
            self.log(
                "warning",
                "No model selected. Please select a model in the 'Training' section in the 'Train Model' tab.",
            )
            return
        # Get directories from settings
        if self.dataDirectoryTrain.text():
            src_dir = Path(self.dataDirectoryTrain.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text()).resolve()
        else:
            self.log("warning", "No CSV directory specified.")
            return
        model_dir = self.settings.get_setting("model/model_directory")
        features_dir = self.settings.get_setting("dino/features_directory")
        if not model_dir:
            self.log(
                "warning",
                f"Missing model directory: {model_dir}. Please set it in the settings.",
            )
            return
        if not features_dir:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the features directory in the settings.",
            )
            features_dir = src_dir

        splits_file = select_file_folder_dialog(
            self,
            "Select Splits File:",
            False,
            False,
            "CSV files (*.csv)",
            start_dir=str(csv_dir),
        )
        if not splits_file or not os.path.isfile(splits_file):
            self.log(
                "warning",
                "No valid splits file selected.",
            )
            return
        else:
            splits_file = Path(splits_file).resolve()
        batch_size = self.settings.get_setting("training/batch_size")
        seed = self.settings.get_setting("training/random_seed")

        # Setup training command
        model_name, model_config = self.train_model_config
        if len(self.train_model_config.samples) > 1:
            dataset_config = MultiSample(sample=self.train_model_config.samples)
            dataset_name = "multi"
        else:
            dataset_config = SingleSample(sample=self.train_model_config.samples[0])
            dataset_name = "single"
        exp_paths = ExpPaths(
            exp_dir=model_dir / self.train_model_config.name,
            tomo_dir=features_dir,
            split_file=splits_file,
        )
        train_config = TrainModelConfig(
            exp_name=self.train_model_config.name,
            label_key=self.train_model_config.label_key,
            save_pretrained=True,
            random_seed=seed,
            model=model_config,
            trainer=self.trainer_config,
            dataset=dataset_config,
            exp_paths=exp_paths,
        )
        train_commands = ["-m", "cryovit.train_model"]
        # Add in dataclass config
        train_commands += [
            f"model={model_name}",
            f"dataset={dataset_name}",
        ]
        train_commands += self._create_command_from_config(
            train_config,
            ignored_config_keys,
        )
        # Add in additional settings
        train_commands += [
            # f"dataloader.batch_size={batch_size}",
            "trainer.logger=[]",
            "hydra.mode=RUN",
        ]
        # Save model config
        self.train_model_config.model_weights = (
            Path(exp_paths.exp_dir) / self.train_model_config.name / "weights.pt"
        )

        train_command = "python " + " ".join(train_commands)
        # Copy to clipboard
        pyperclip.copy(train_command)
        self.log("info", f"Copied Training command to clipboard:\n{train_command}")

    def setup_console(self):
        """Setup the UI console to display stdout and stderr messages."""
        sys.stdout = EmittingStream(textWritten=partial(self.log, "info"))
        sys.stderr = EmittingStream(textWritten=partial(self.log, "error"))

    def setup_checkboxes(self):
        """Setup replace checkboxes for selecting destination directories."""
        # Update from current checkbox state
        self._show_hide_widgets(
            not self.replaceCheckboxProc.isChecked(),
            self.replaceDirectoryLabelProc,
            self.replaceDirectoryProc,
            self.replaceSelectProc,
        )
        self._show_hide_widgets(
            not self.replaceCheckboxSeg.isChecked(),
            self.replaceDirectoryLabelSeg,
            self.replaceDirectorySeg,
            self.replaceSelectSeg,
        )
        # Setup checkboxes for processing options
        self.replaceCheckboxProc.checkStateChanged.connect(
            lambda state: self._show_hide_widgets(
                state == Qt.CheckState.Unchecked,
                self.replaceDirectoryLabelProc,
                self.replaceDirectoryProc,
                self.replaceSelectProc,
            )
        )
        self.replaceCheckboxSeg.checkStateChanged.connect(
            lambda state: self._show_hide_widgets(
                state == Qt.CheckState.Unchecked,
                self.replaceDirectoryLabelSeg,
                self.replaceDirectorySeg,
                self.replaceSelectSeg,
            )
        )

    def setup_sample_select(self):
        """Setup the sample select combobox for selecting training samples."""
        old_combo = self.sampleSelectCombo
        self.sampleSelectCombo = MultiSelectComboBox(parent=self.sampleSelectFrame)
        self.sampleSelectCombo.setSizePolicy(old_combo.sizePolicy())
        self.sampleSelectCombo.setObjectName(old_combo.objectName())
        self.sampleSelectCombo.setToolTip(old_combo.toolTip())
        self.sampleSelectCombo.setPlaceholderText(old_combo.placeholderText())
        # Set initial available samples from existing enum
        self.sampleSelectCombo.addItems(samples)
        self.sampleSelectCombo.currentTextChanged.connect(
            self._update_train_model_samples
        )
        self.sampleSelectLayout.replaceWidget(old_combo, self.sampleSelectCombo)
        old_combo.deleteLater()

        self.sampleAdd.clicked.connect(self._add_sample)

    @_catch_exceptions("update sample select")
    def _add_sample(self, *args):
        """Add a new sample to the sample list."""
        self.sampleSelectCombo.addNewItem()

    def setup_folder_selects(self):
        """Setup tool buttons for opening folder and file select dialogs."""
        # Setup folder select buttons
        self.rawSelect.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.rawDirectory,
                "raw tomogram",
                True,
                False,
            )
        )
        self.dataSelect.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.dataDirectory,
                "processed tomogram",
                True,
                False,
            )
        )
        self.dataSelectTrain.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.dataDirectoryTrain,
                "processed tomogram",
                True,
                False,
            )
        )
        self.csvSelect.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.csvDirectory,
                "csv",
                True,
                False,
            )
        )
        self.sliceSelect.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.sliceDirectory,
                "slices",
                True,
                False,
            )
        )
        self.annoSelect.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.annoDirectory,
                "annotation",
                True,
                False,
            )
        )
        self.replaceSelectProc.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.replaceDirectoryProc,
                "new processed",
                True,
                False,
            )
        )
        self.replaceSelectSeg.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.replaceDirectorySeg,
                "new result",
                True,
                False,
            )
        )

        # Setup folder displays
        self.rawDirectory.editingFinished.connect(
            partial(self._update_file_directory_field, self.rawDirectory, True)
        )
        self.dataDirectory.editingFinished.connect(
            partial(self._update_file_directory_field, self.dataDirectory, True)
        )
        self.dataDirectoryTrain.editingFinished.connect(
            partial(self._update_file_directory_field, self.dataDirectoryTrain, True)
        )
        self.csvDirectory.editingFinished.connect(
            partial(self._update_file_directory_field, self.csvDirectory, True)
        )
        self.sliceDirectory.editingFinished.connect(
            partial(self._update_file_directory_field, self.sliceDirectory, True)
        )
        self.annoDirectory.editingFinished.connect(
            partial(self._update_file_directory_field, self.annoDirectory, True)
        )
        self.replaceDirectoryProc.editingFinished.connect(
            partial(
                self._update_file_directory_field,
                self.replaceDirectoryProc,
                True,
            )
        )
        self.replaceDirectorySeg.editingFinished.connect(
            partial(
                self._update_file_directory_field,
                self.replaceDirectorySeg,
                True,
            )
        )

    @_catch_exceptions("select file/directory prompt")
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
                start_dir=str(self.settings.get_setting("general/data_directory")),
            )
        )

    @_catch_exceptions("validate file/directory field")
    def _update_file_directory_field(
        self, text_field: QLineEdit, is_folder: bool, *args
    ):
        """Validate and update a file or directory field to ensure it is a valid directory."""
        current_text = text_field.text()
        valid_text = (
            text_field.text()
            if (
                (os.path.isdir(text_field.text()) and is_folder)
                or (os.path.isfile(text_field.text()) and not is_folder)
            )
            else None
        )
        if not valid_text:
            self.warning(f"Invalid folder path: {current_text}")
            return
        else:
            text_field.setText(valid_text)
            text_field.editingFinished.emit()

    def setup_feature_select(self):
        """Setup the feature selection for training."""
        self.features = []
        self.featuresAdd.clicked.connect(self._add_feature)
        self.featuresDisplay.editingFinished.connect(self._update_features)

    @_catch_exceptions("add training feature")
    def _add_feature(self, *args):
        """Adds a new feature to the available training feature list."""
        feature_name, _ = QInputDialog.getText(
            self, "Add Feature", "Enter feature name:"
        )
        if feature_name:
            self.features.append(feature_name)
            self.featuresDisplay.setText(", ".join(self.features))
            self._update_features()

    @_catch_exceptions("update training features")
    def _update_features(self, *args):
        """Update the training features based on the text field."""
        current_text = self.featuresDisplay.text()
        if current_text:
            self.features = [feature.strip() for feature in current_text.split(",")]
            self.featuresDisplay.setText(", ".join(self.features))
        else:
            self.features = []

        self.labelCombo.clear()
        self.labelCombo.addItems(self.features)

    def setup_training_config(self):
        """Setup creating and editing the train model and trainer configurations."""
        self.train_model = None
        self.train_model_config = None
        self.trainer_config = TrainerFit()

        self.modelComboTrain.currentTextChanged.connect(self._update_train_model_arch)
        self.modelComboTrain.addItems(models)
        self.modelComboTrain.setCurrentIndex(0)
        self.modelSettings.clicked.connect(self._open_train_config)

        if self.features:
            self.labelCombo.addItems(self.features)
        else:
            self.log(
                "warning",
                "No features specified. Please add features in the 'Annotations' section above.",
            )
        self.labelCombo.currentTextChanged.connect(self._update_train_model_label)
        self.labelCombo.setCurrentIndex(0)

    @_catch_exceptions("update training model architecture")
    def _update_train_model_arch(self, text: str, *args):
        """Update the training model architecture based on the selected model."""
        if not self.train_model_config:
            samples = self.sampleSelectCombo.getCurrentData()
            self.train_model_config = InterfaceModelConfig(
                text.lower(),
                "",
                ModelArch[text],
                "",
                {},
                samples,
            )
        self.train_model_config.model_type = ModelArch[text]

    @_catch_exceptions("update training model samples")
    def _update_train_model_samples(self, text: str, *args):
        """Update the training model samples based on the selected samples."""
        samples = list(map(str.strip, text.split(",")))
        if not self.train_model_config:
            self.train_model_config = InterfaceModelConfig(
                self.modelComboTrain.currentText().lower(),
                "",
                ModelArch[text],
                "",
                {},
                samples,
            )
        self.train_model_config.samples = samples

    @_catch_exceptions("update training model label")
    def _update_train_model_label(self, text: str, *args):
        """Update the training model classiification label based on the selected label."""
        if not self.train_model_config:
            self.log(
                "warning",
                "No model selected. Please select a model in the 'Training' section.",
            )
            return
        self.train_model_config.label_key = text

    @_catch_exceptions("open training config")
    def _open_train_config(self, *args):
        """Open the training model configuration dialog."""
        if not self.train_model_config:
            self.log(
                "warning",
                "No model selected. Please select a model in the 'Training' section.",
            )
            return
        config_dialog = ModelDialog(
            self, self.train_model_config, trainer_config=self.trainer_config
        )
        config_dialog.exec()
        if config_dialog.result() == config_dialog.DialogCode.Accepted:
            self.train_model_config = config_dialog.config
            self.train_model_config.model_weights = (
                self.settings.get_setting("model/model_directory")
                / self.train_model_config.name
                / "weights.pt"
            )
            self.trainer_config = config_dialog.trainer_config
            if self.train_model_config.samples:
                self.sampleSelectCombo.setCurrentData(self.train_model_config.samples)
            self.log("success", "Training configuration updated.")

    def setup_run_buttons(self):
        """Setup the run buttons for processing and training."""
        self.processButtonSeg.clicked.connect(self.run_preprocessing)
        self.chimeraButton.clicked.connect(self.run_chimerax)
        self.splitsButton.clicked.connect(self.run_generate_training_splits)
        self.splitsButtonNew.clicked.connect(self.run_new_training_splits)
        self.featureButtonTrain.clicked.connect(self.run_feature_extraction)
        self.segmentButton.clicked.connect(self.run_segmentation)
        self.trainButton.clicked.connect(self.run_training)

    def setup_menu(self):
        """Setup the menu bar for loading and saving presets."""
        self.actionLoad_Preset.triggered.connect(self._load_preset)
        self.actionSave_Preset.triggered.connect(partial(self._save_preset, True))
        self.actionSaveAs_Preset.triggered.connect(partial(self._save_preset, False))
        self.actionSettings.triggered.connect(self._open_settings)

        self.actionDirectory_Setup.triggered.connect(self._setup_directory)
        self.actionGenerate_New_Training_Splits.triggered.connect(
            self.run_new_training_splits
        )

        self.actionGithub.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/VivianDLi/CryoViT")
            )
        )

    @_catch_exceptions("load preset settings")
    def _load_preset(self, *args):
        """Opens a prompt to optionally load previous settings from a saved name."""
        available_presets = (
            sorted(self.settings.get_setting("preset/available_presets")) or []
        )
        if not available_presets:
            self.log("warning", "No available presets to load.")
            return
        preset_dialog = PresetDialog(
            self,
            "Load preset",
            *available_presets,
            current_preset=self.settings.get_setting("preset/current_preset"),
            load_preset=True,
        )
        result = preset_dialog.exec()
        name = preset_dialog.result
        if result == preset_dialog.DialogCode.Accepted:
            self.settings.set_setting(
                "preset/available_presets", preset_dialog.get_presets()
            )
            if name:
                temp_settings = QSettings(
                    "Stanford University_Wah Chiu", f"CryoViT_{name}"
                )
                for key in temp_settings.allKeys():
                    self.settings.set_setting(key, temp_settings.value(key))
                self.settings.set_setting("preset/current_preset", name)
                self.log("success", f"Loaded preset: {name}")
            else:
                self.settings.set_setting("preset/current_preset", "")
                self.log("warning", "No preset name specified.")
            # Remove unused presets
            unused_settings = set(available_presets) - set(
                self.settings.get_setting("preset/available_presets")
            )
            for unused in unused_settings:
                temp_settings = QSettings(
                    "Stanford University_Wah Chiu", f"CryoViT_{unused}"
                )
                temp_settings.clear()

    @_catch_exceptions("save preset settings")
    def _save_preset(self, replace: bool, *args):
        """Opens a prompt to save the current settings to a preset name."""
        name = None
        if replace:
            name = self.settings.get_setting("preset/current_preset")
            if not name:
                self.log(
                    "warning",
                    "No preset name specified. Load an existing preset or create a new one.",
                )
                return
        else:
            current_presets = (
                self.settings.get_setting("preset/available_presets") or []
            )
            preset_dialog = PresetDialog(
                self,
                "Save preset",
                *current_presets,
                current_preset=self.settings.get_setting("preset/current_preset"),
            )
            result = preset_dialog.exec()
            if result == preset_dialog.DialogCode.Accepted:
                name = preset_dialog.result
                self.settings.set_setting(
                    "preset/available_presets",
                    preset_dialog.get_presets(),
                )
                if not name:
                    self.log("warning", "No preset name specified.")
                    return
                if name in current_presets:
                    self.log(
                        "warning",
                        f"Preset '{name}' already exists. Overwriting.",
                    )
                # Remove unused settings
                unused_settings = set(current_presets) - set(
                    self.settings.get_setting("preset/available_presets")
                )
                for unused in unused_settings:
                    temp_settings = QSettings(
                        "Stanford University_Wah Chiu", f"CryoViT_{unused}"
                    )
                    temp_settings.clear()
        if name:
            temp_settings = QSettings("Stanford University_Wah Chiu", f"CryoViT_{name}")
            for key in [
                key
                for key in self.settings.get_available_settings()
                if not key.startswith("preset/")
            ]:
                temp_settings.setValue(key, self.settings.get_setting(key, as_str=True))
            self.settings.set_setting("preset/current_preset", name)
            self.log("success", f"Saved preset: {name}")

    @_catch_exceptions("open settings")
    def _open_settings(self, *args):
        """Open the settings window."""
        result = self.settings.exec()
        if result == self.settings.DialogCode.Accepted:
            self.setup_model_select()
            self.log("success", "Settings saved.")

    @_catch_exceptions("setup directory")
    def _setup_directory(self, *args):
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
            self.log("warning", "No valid directory selected.")
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
            self.log(
                "warning", "No search query specified. Defaulting to '**/Tomograms/**'."
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
            self.log("warning", "No valid directory selected.")
            return
        data_dir = Path(data_dir).resolve()
        sample_name, ok = QInputDialog.getText(
            self, "Dataset Sample Name", "Enter sample name for the downloaded dataset:"
        )
        if not ok or not sample_name:
            self.log("warning", "No sample name specified.")
            return
        delete_dir = QMessageBox.question(
            self,
            "Delete Directory?",
            "Do you want to delete the existing directory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.No,
        )
        # Create the dataset structure
        self.log("info", f"Moving {len(tomogram_files)} tomograms from {src_dir}.")
        create_dataset(tomogram_files, data_dir, sample_name)
        self.log("success", f"Dataset structure created at {data_dir / sample_name}.")
        # Clean up the source directory if requested
        if delete_dir == QMessageBox.StandardButton.Yes:
            shutil.rmtree(src_dir)
            self.log("success", f"Deleted {src_dir}.")

    def _show_hide_widgets(self, visible: bool, *widgets):
        """Show or hide multiple widgets."""
        for widget in widgets:
            widget.setVisible(visible)

    def _update_progress_bar(self, index: int, total: int):
        """Update the progress bar with the current index (assume 0-indexed) and total."""
        self.progressBar.setValue(round((index / total) * self.progressBar.maximum()))

    def log(self, mode: str, text: str, end="\n", use_timestamp: bool = True):
        """Write text to the console with a timestamp and different colors for normal output and errors."""
        import time

        # Ignore newlines but print whitespaces (no timestamp)
        if not text.strip():
            if text == "\n":
                return
            else:
                self.consoleText.insertPlainText(text)
                return
        # Get timestamp
        timestamp_str = "{}> ".format(time.strftime("[%H:%M:%S]"))
        # Format text with colors and timestamp
        if use_timestamp:
            full_text = timestamp_str + text + end
        else:
            full_text = text + end
        # Format text with colors
        match mode:
            case "error":
                full_text = '<font color="#FF4500">{}</font>'.format(full_text)
            case "warning":
                full_text = '<font color="#FFA500">{}</font>'.format(full_text)
            case "success":
                full_text = '<font color="#32CD32">{}</font>'.format(full_text)
            case "debug":
                full_text = '<font color="#1E90FF">{}</font>'.format(full_text)
            case _:
                full_text = '<font color="white">{}</font>'.format(full_text)
        # Add line breaks in HTML
        full_text = full_text.replace("\n", "<br>")
        # Write to console ouptut
        keep_scrolling = (
            self.consoleText.verticalScrollBar().value()
            == self.consoleText.verticalScrollBar().maximum()
        )
        self.consoleText.insertHtml(full_text)
        if keep_scrolling:
            self.consoleText.verticalScrollBar().setValue(
                self.consoleText.verticalScrollBar().maximum()
            )

    def closeEvent(self, event):
        """Override close event to save settings."""
        self.syncTimer.stop()
        self.settings_model.save_settings()
        event.accept()


def filter_maker(level):
    level = getattr(logging, level)

    def filter(record):
        return record.levelno <= level

    return filter


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
