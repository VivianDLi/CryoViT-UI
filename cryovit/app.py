"""Main application window for CryoVit segmentation and training."""

import os
from pathlib import Path
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

from cryovit.config import (
    DinoFeaturesConfig,
    ExpPaths,
    MultiSample,
    SingleSample,
    Inference,
    TrainModelConfig,
    InferModelConfig,
    InterfaceModelConfig,
    ModelArch,
    samples,
    models,
)
import cryovit.gui.resources
from cryovit.gui.layouts.mainwindow import Ui_MainWindow
from cryovit.gui.model_config import ModelDialog
from cryovit.gui.settings import PresetDialog, SettingsWindow
from cryovit.gui.utils import (
    EmittingStream,
    MultiSelectComboBox,
    select_file_folder_dialog,
)
from cryovit.config import TrainerFit, TrainerInfer
from cryovit.processing import *


class WorkerSignals(QObject):
    """Signals for background threads to communicate with the GUI."""

    finish = pyqtSignal()  # signals when the thread is finished
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
            self.fn(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finish.emit()


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
                self.log("error", f"Error running {desc}: {e}")

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
        self.log(
            "error",
            f"Error in thread: {exctype}: {value}.\n{traceback_info}",
        )

    def _on_thread_finish(self, name: str, *args):
        """Handle progress bar updates, busy cursor, and logging when a background thread finishes.

        Args:
            name (str): Name of the process that finished.
        """
        self.progress_dict[name]["count"] += 1
        count, total = (
            self.progress_dict[name]["count"],
            self.progress_dict[name]["total"],
        )
        self._update_progress_bar(count, total)
        if count >= total:
            QGuiApplication.restoreOverrideCursor()
            self.log("success", f"{name} complete.")

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
                self.log("debug", f"{process_name} process not running.")
            case QProcess.ProcessState.Starting:
                self.log("debug", f"{process_name} process starting.")
            case QProcess.ProcessState.Running:
                self.log("debug", f"{process_name} process running.")
            case _:
                self.log("warning", f"{process_name} process unknown state.")

    # UI initialization
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        # Setup settings
        self.settings = SettingsWindow(self)
        # Setup thread pool and processes
        self.threadpool: QThreadPool = (
            QThreadPool.globalInstance()
            if QThreadPool.globalInstance()
            else QThreadPool()
        )
        self.progress_dict = {}  # progress bar update dict
        self.chimera_process = None
        self.dino_process = None
        self.segment_process = None
        self.train_process = None

        # Setup UI elements
        try:
            self.setup_checkboxes()
            self.setup_sample_select()
            self.setup_model_select()
            self.setup_feature_select()
            self.setup_training_config()
            self.setup_folder_selects()
            self.setup_run_buttons()
            self.setup_menu()
            self.setup_console()
        except Exception as e:
            sys.stderr.write(f"Error setting up UI: {e}")
            sys.exit(1)
        self.log("success", "Welcome to CryoViT!")

    @_catch_exceptions("preprocessing", concurrent=True)
    def run_preprocessing(self, is_train: bool, *args):
        """Background thread to bin, normalize, and clip tomograms.

        Args:
            is_train (bool): Whether to use widgets in the train tab to determine arguments.
        """
        # Get optional kwargs from settings
        kwargs = {
            "bin_size": self.settings.get_setting("preprocessing/bin_size"),
            "normalize": self.settings.get_setting("preprocessing/normalize"),
            "clip": self.settings.get_setting("preprocessing/clip"),
        }
        # Get directories from settings
        if is_train:
            raw_dir = self.rawDirectoryTrain
            replace = self.replaceCheckboxProcTrain.isChecked()
            replace_dir = self.replaceDirectoryProcTrain
        else:
            raw_dir = self.rawDirectory
            replace = self.replaceCheckboxProc.isChecked()
            replace_dir = self.replaceDirectoryProc
        if raw_dir.text():
            src_dir = Path(raw_dir.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if replace:
            replace_dir.setText(str(src_dir))
        if replace_dir.text():
            dst_dir = Path(replace_dir.text()).resolve()
        else:
            self.log("warning", "No replace directory specified.")
            return

        # Setup thread callbacks
        finish_callback = partial(self._on_thread_finish, "Preprocessing")
        self.progress_dict["Preprocessing"] = {"count": 0, "total": 0}
        # Check for samples
        samples = self.sampleSelectCombo.getCurrentData()
        if is_train and samples:
            # For multi-sample train pre-processing to use multithreading
            self.progress_dict["Preprocessing"]["total"] = len(samples)
            for sample in samples:
                if not os.path.isdir(src_dir / sample):
                    self.log(
                        "warning",
                        f"Invalid raw directory: {src_dir / sample}, skipping sample.",
                    )
                    continue
                worker = Worker(
                    run_preprocess,
                    src_dir / sample,
                    dst_dir / sample,
                    **kwargs,
                )
                worker.signals.finish.connect(finish_callback)
                worker.signals.error.connect(self._handle_thread_exception)
                self.threadpool.start(worker)
        else:
            # Single sample train pre-processing and segmentation pre-processing are both 1 folder
            if not os.path.isdir(src_dir):
                self.log(
                    "error",
                    f"Invalid raw directory: {src_dir}.",
                )
                return
            worker = Worker(run_preprocess, src_dir, dst_dir, **kwargs)
            worker.signals.finish.connect(finish_callback)
            worker.signals.error.connect(self._handle_thread_exception)
            self.threadpool.start(worker)

    @_catch_exceptions("ChimeraX")
    def run_chimerax(self, *args):
        """Launch ChimeraX externally to select z-limits and tomogram slices to label (and create .csv file)."""
        # Get path to ChimeraX executable (only matters on Windows or Mac)
        chimerax_path = self.settings.get_setting("annotation/chimera_path")
        if not chimerax_path and platform.system().lower() != "linux":
            self.log(
                "warning",
                "ChimeraX path not set. Please set it in the settings.",
            )
            return
        else:
            chimerax_path = Path(chimerax_path if chimerax_path else "").resolve()

        # Get directories from settings
        samples = self.sampleSelectCombo.getCurrentData()
        if self.dataDirectoryTrain.text():
            src_dir = Path(self.dataDirectoryTrain.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if self.sliceDirectory.text():
            dst_dir = Path(self.sliceDirectory.text()).resolve()
        else:
            dst_dir = (
                src_dir.resolve() / "slices"
                if samples
                else src_dir.parent.resolve() / "slices"
            )
            self.sliceDirectory.setText(str(dst_dir))
            self.log(
                "warning",
                f"No slice directory specified. Slices will be saved in {src_dir.resolve() / 'slices/sample' if samples else src_dir.parent.resolve() / 'slices'}.",
            )
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text()).resolve()
        else:
            csv_dir = (
                src_dir.parent.resolve() / "csv"
                if samples
                else src_dir.parent.parent.resolve() / "csv"
            )
            self.csvDirectory.setText(str(csv_dir))
            self.log(
                "warning",
                f"No CSV directory specified. CSV files will be saved in {src_dir.parent.resolve() / 'csv' if samples else src_dir.parent.parent.resolve() / 'csv'}.",
            )
        num_slices = self.settings.get_setting("annotation/num_slices")
        # Check for no samples
        if not samples:
            self._next_chimera_process(
                0, chimerax_path, src_dir, None, dst_dir, csv_dir, num_slices
            )
        else:
            # Sequentially process each sample
            self._next_chimera_process(
                0,
                chimerax_path,
                src_dir,
                samples,
                dst_dir,
                csv_dir,
                num_slices,
            )

    def _next_chimera_process(
        self,
        index: int,
        chimerax_path: Path,
        src_dir: Path,
        samples: List[str] | None,
        dst_dir: Path,
        csv_dir: Path,
        num_slices: int,
    ):
        """Launches ChimeraX and sets up the next sample to launch when closed.

        Args:
            index (int): Index of the current sample.
            chimerax_path (Path): Path to the ChimeraX executable.
            src_dir (Path): Source directory for the tomograms.
            samples (List[str] | None): List of samples to process. If None, processing a single directory.
            dst_dir (Path): Destination directory for the slices.
            csv_dir (Path): Destination directory for the CSV files.
            num_slices (int): Number of slices to process.
        """
        self.chimera_process = None
        if samples:
            # Update progress bar and check for completion
            self._update_progress_bar(index + 1, len(samples))
            if index >= len(samples):
                self.log("success", "ChimeraX processing complete.")
                QGuiApplication.restoreOverrideCursor()
                return
            sample = samples[index]
            is_dir = os.path.isdir(src_dir / sample)
        else:
            # Check for completion
            if index >= 1:
                self.log("success", "ChimeraX processing complete.")
                QGuiApplication.restoreOverrideCursor()
                return
            sample = None
            is_dir = os.path.isdir(src_dir)
        if not is_dir:
            self.log(
                "error",
                f"Invalid raw directory: {src_dir / sample if sample else src_dir}, skipping sample.",
            )
            self._next_chimera_process(
                index + 1,
                chimerax_path,
                src_dir,
                samples,
                dst_dir,
                csv_dir,
                num_slices,
            )
            return
        # Launch ChimeraX
        command = self._validate_chimerax_process(
            chimerax_path,
            src_dir,
            sample,
            dst_dir=dst_dir,
            csv_dir=csv_dir,
            num_slices=num_slices,
        )
        if command is not None:  # No errors in command generation
            self.chimera_process = QProcess()
            self.chimera_process.readyReadStandardOutput.connect(
                partial(self._handle_stdout, self.chimera_process)
            )
            self.chimera_process.readyReadStandardError.connect(
                partial(self._handle_stderr, self.chimera_process)
            )
            self.chimera_process.finished.connect(
                partial(
                    self._next_chimera_process,
                    index + 1,
                    chimerax_path,
                    src_dir,
                    samples,
                    dst_dir,
                    csv_dir,
                    num_slices,
                )
            )
            self.chimera_process.start(command)
        else:
            self.log(
                "error",
                f"ChimeraX process for src_dir: {src_dir / sample if sample else src_dir} failed.",
            )
            self._next_chimera_process(
                index + 1,
                chimerax_path,
                src_dir,
                samples,
                dst_dir,
                csv_dir,
                num_slices,
            )
            return

    def _validate_chimerax_process(
        self,
        chimerax_path: Path,
        src_dir: Path,
        sample: str = None,
        dst_dir: Path = None,
        csv_dir: Path = None,
        num_slices: int = 5,
    ) -> str | None:
        """Generate the start command for ChimeraX to start labeling in the clipboard and return the command to launch ChimeraX.

        Args:
            chimerax_path (Path): Path to the ChimeraX executable.
            src_dir (Path): Source directory for the tomograms.
            sample (str, optional): Sample name. Defaults to None (i.e., single directory).
            dst_dir (Path, optional): Destination directory for the slices. Defaults to None.
            csv_dir (Path, optional): Destination directory for the CSV files. Defaults to None.
            num_slices (int, optional): Number of slices to process. Defaults to 5.
        """
        if self.chimera_process is not None:
            return None
        # Create script args
        commands = [
            "open",
            chimera_script_path,
            ";",
            "start slice labels",
            "'" + str(src_dir.resolve()) + "'",
        ]
        if sample:
            commands.append("'" + str(sample) + "'")
        if dst_dir:
            commands.extend(["dst_dir", "'" + str(dst_dir.resolve()) + "'"])
        if csv_dir:
            commands.extend(["csv_dir", "'" + str(csv_dir.resolve()) + "'"])
        commands.extend(["num_slices", str(num_slices)])
        # Command to run chimera_slices and start slice labels
        command = " ".join(commands)
        # Copy command to clipboard
        pyperclip.copy(command)
        # Check for OS type
        match platform.system().lower():
            case "windows":  # Needs to be run from a specific path
                # Check for valid path
                if (
                    not os.path.isfile(chimerax_path)
                    or not chimerax_path.name.lower() == "chimerax.exe"
                ):
                    self.log(
                        "error",
                        f"Invalid ChimeraX path: {chimerax_path}. This should be the path to the ChimeraX.exe executable typically found in 'C:/Program Files/ChimeraX/bin/ChimeraX.exe'. Please set it in the settings.",
                    )
                    return None
            case "linux":  # Has chimerax from command line
                chimerax_path = "chimerax"
            case "darwin":  # Needs to be run from a specific path
                if (
                    not os.path.isfile(chimerax_path)
                    or not chimerax_path.name.lower() == "chimerax.app"
                ):
                    self.log(
                        "error",
                        f"Invalid ChimeraX path: {chimerax_path}. This should be the path to the ChimeraX.app application typically found in '/Applications/ChimeraX.app'. Please set it in the settings.",
                    )
                    return None
            case _:
                self.log("error", f"Unsupported OS type {platform.system()}.")
                return None
        return str(chimerax_path)

    @_catch_exceptions("generate training splits", concurrent=True)
    def run_generate_training_splits(self, *args):
        """Add annotations to tomograms and generate training splits .csv file."""
        # Get directories from settings
        samples = self.sampleSelectCombo.getCurrentData()
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
        num_splits = self.settings.get_setting("training/number_of_splits")
        seed = self.settings.get_setting("training/random_seed")
        if not self.features:
            self.log(
                "warning",
                "No labeled features inputed. Please add annotated features above.",
            )
            return

        # Setup thread callbacks
        finish_callback = partial(self._on_thread_finish, "Annotations")
        self.progress_dict["Annotations"] = {"count": 0, "total": 0}
        # Check for no samples (single directory)
        if not samples:
            if not os.path.isdir(src_dir):
                self.log(
                    "error",
                    f"Invalid raw directory: {src_dir}.",
                )
                return
            self.progress_dict["Annotations"]["total"] = 2
            sample = src_dir.name
            annotation_worker = Worker(
                partial(
                    add_annotations,
                    src_dir=src_dir,
                    dst_dir=src_dir,
                    annot_dir=annot_dir,
                    csv_file=csv_dir / f"{sample}.csv",
                    features=self.features,
                )
            )
            split_worker = Worker(
                partial(
                    add_splits,
                    dst_dir=csv_dir,
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
            # Run adding annotations and generating splits in parallel
            self.threadpool.start(annotation_worker)
            self.threadpool.start(split_worker)
        else:
            self.progress_dict["Annotations"]["total"] = len(samples) * 2
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
                        dst_dir=csv_dir,
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

    @_catch_exceptions("generate new training splits", concurrent=True)
    def run_new_training_splits(self, *args):
        """Generate new training splits .csv file from existing splits."""
        # Get directory from settings
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text()).resolve()
        else:
            self.log(
                "warning",
                f"No CSV directory specified. Please specify a CSV directory.",
            )
            return
        num_splits = self.settings.get_setting("training/number_of_splits")
        seed = self.settings.get_setting("training/random_seed")

        # Get splits file or set it to default value
        splits_file = self.settings.get_setting("training/splits_file")
        if splits_file:
            splits_file = Path(splits_file).resolve()
        else:
            splits_file = csv_dir / "splits.csv"
            self.settings.set_setting("training/splits_file", str(splits_file))
            self.log(
                "warning",
                f"No splits file specified in Settings. Using {splits_file}.",
            )

        # Check to see if new splits replaces old splits
        dst_name, ok = QInputDialog.getText(
            self, "New Split Name", "Enter new split name:"
        )
        if ok and dst_name in [p.stem for p in csv_dir.glob("*.csv")]:
            result = QMessageBox.warning(
                self,
                "Warning!",
                f"This will overwriting existing splits {dst_name}. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.No:
                self.log("warning", "New split generation cancelled.")
                return
        elif not ok:
            self.log("warning", f"No split specified. Please specify a split name.")
            return

        # Start thread to generate new splits
        finish_callback = partial(self._on_thread_finish, "New Splits")
        worker = Worker(
            generate_new_splits,
            splits_file,
            dst_file=csv_dir / f"{dst_name}.csv",
            num_splits=num_splits,
            seed=seed,
        )
        worker.signals.finish.connect(finish_callback)
        worker.signals.error.connect(self._handle_thread_exception)
        self.threadpool.start(worker)

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
        dino_dir = self.settings.get_setting("dino/model_directory")
        if dino_dir:
            dino_dir = Path(dino_dir).resolve()
        else:
            self.log(
                "warning",
                f"Missing DINO directory: {dino_dir}. This is where the DINOv2 model will be saved. Please set it in the settings.",
            )
            return
        if self.dataDirectoryTrain.text():
            data_dir = Path(self.dataDirectoryTrain.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text()).resolve()
        else:
            self.log("warning", "No CSV directory specified.")
            return
        if features_dir:
            features_dir = Path(features_dir).resolve()
        else:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the features directory in the settings.",
            )
            features_dir = data_dir
        dino_batch_size = self.settings.get_setting("dino/batch_size")

        # Setup DINOv2 command
        dino_config = DinoFeaturesConfig(
            dino_dir=dino_dir,
            tomo_dir=data_dir,
            csv_dir=csv_dir,
            feature_dir=features_dir,
            batch_size=dino_batch_size,
            sample=self.sampleSelectCombo.getCurrentData(),
        )
        dino_commands = ["-m", "cryovit.dino_features"]
        dino_commands += self._create_command_from_config(
            dino_config, ["all_samples", "cryovit_root"]
        )
        dino_commands += ["hydra.mode=RUN"]

        # Setup QProcesses
        self.dino_process = QProcess()
        self.dino_process.readyReadStandardOutput.connect(
            partial(self._handle_stdout, "dino_process")
        )
        self.dino_process.readyReadStandardError.connect(
            partial(self._handle_stderr, "dino_process")
        )
        self.dino_process.stateChanged.connect(
            partial(self._handle_state_change, "dino_process")
        )
        self.dino_process.finished.connect(self._dino_process_finish)
        self.log("info", f"Running DINO features:")
        self.log("debug", f"Command: {dino_commands}")
        self.dino_process.start("python", dino_commands)

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
        # Check for required settings
        model_dir = self.settings.get_setting("model/model_directory")
        if model_dir:
            model_dir = Path(model_dir).resolve()
        else:
            self.log(
                "warning",
                f"Missing model directory: {model_dir}. Please set it in the settings.",
            )
            return
        if self.dataDirectory.text():
            data_dir = Path(self.dataDirectory.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return

        # Get directories from settings
        replace_seg = self.replaceCheckboxSeg.isChecked()
        replace_seg_dir = self.replaceDirectorySeg.text()
        dst_dir = Path(data_dir if replace_seg else replace_seg_dir).resolve()
        features_dir = self.settings.get_setting("dino/features_directory")
        if features_dir:
            features_dir = Path(features_dir).resolve()
        else:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the features directory in the settings.",
            )
            features_dir = data_dir
        batch_size = self.settings.get_setting("segmentation/batch_size")
        if self.settings.get_setting("segmentation/csv_file"):
            csv_file = self.settings.get_setting("segmentation/csv_file")
            csv_file = Path(csv_file).resolve()
        else:
            csv_file = None

        # Setup segmentation command
        # Get model list
        model_names = ",".join(
            [
                '"' + str(page) + '"'
                for page in self._models
                if self.modelTabs.indexOf(self._models[page]["widget"]) != -1
            ]
        )
        exp_paths = ExpPaths(
            exp_dir=dst_dir, tomo_dir=features_dir, split_file=csv_file
        )
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
            [
                "aux_keys",
                "dataloader",
                "trainer",
                "cryovit_root",
                "logger",
                "callbacks",
                "losses",
                "metrics",
                "models",
                "_target_",
                "_partial_",
            ],
        )
        # Add in additional settings
        infer_commands += [
            # f"dataloader.batch_size={batch_size}",
            f"models=[{model_names}]",
            "hydra.mode=RUN",
        ]

        # Setup QProcesses
        self.segment_process = QProcess()
        self.segment_process.readyReadStandardOutput.connect(
            partial(self._handle_stdout, "segment_process")
        )
        self.segment_process.readyReadStandardError.connect(
            partial(self._handle_stderr, "segment_process")
        )
        self.segment_process.stateChanged.connect(
            partial(self._handle_state_change, "segment_process")
        )
        self.segment_process.finished.connect(self._segment_process_finish)
        self.log("info", f"Running segmentation:")
        self.log("debug", f"Command: {infer_commands}")
        self.segment_process.start("python", infer_commands)

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
        # Check for required settings
        model_dir = self.settings.get_setting("model/model_directory")
        if model_dir:
            model_dir = Path(model_dir).resolve()
        else:
            self.log(
                "warning",
                f"Missing model directory: {model_dir}. Please set it in the settings.",
            )
            return
        if self.dataDirectoryTrain.text():
            data_dir = Path(self.dataDirectoryTrain.text()).resolve()
        else:
            self.log("warning", "No data directory specified.")
            return
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text()).resolve()
        else:
            self.log("warning", "No CSV directory specified.")
            return
        if not self.train_model_config:
            self.log(
                "warning",
                "No model selected. Please select a model in the 'Training' section in the 'Train Model' tab.",
            )
            return

        # Get directories
        split_file = self.settings.get_setting("training/splits_file")
        if split_file:
            split_file = Path(split_file).resolve()
        else:
            split_file = csv_dir / "splits.csv"
            self.settings.set_setting("training/splits_file", str(split_file))
        features_dir = self.settings.get_setting("dino/features_directory")
        if features_dir:
            features_dir = Path(features_dir).resolve()
        else:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the features directory in the settings.",
            )
            features_dir = data_dir
        batch_size = self.settings.get_setting("training/batch_size")
        seed = self.settings.get_setting("training/random_seed")

        # Setup training command
        model_name, model_config = load_base_model_config(self.train_model_config)
        if len(self.train_model_config.samples) > 1:
            dataset_config = MultiSample(sample=self.train_model_config.samples)
            dataset_name = "multi"
        else:
            dataset_config = SingleSample(sample=self.train_model_config.samples[0])
            dataset_name = "single"
        exp_paths = ExpPaths(
            exp_dir=model_dir / self.train_model_config.name,
            tomo_dir=features_dir,
            split_file=split_file,
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
            [
                "aux_keys",
                "dataloader",
                "cryovit_root",
                "test_samples",
                "split_id",
                "logger",
                "callbacks",
                "losses",
                "metrics",
                "_target_",
                "_partial_",
            ],
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
        save_model_config(model_dir, self.train_model_config)

        # Setup QProcesses
        self.train_process = QProcess()
        self.train_process.readyReadStandardOutput.connect(
            partial(self._handle_stdout, "train_process")
        )
        self.train_process.readyReadStandardError.connect(
            partial(self._handle_stderr, "train_process")
        )
        self.train_process.stateChanged.connect(
            partial(self._handle_state_change, "train_process")
        )
        self.train_process.finished.connect(self._train_process_finish)
        self.log("info", f"Running training:")
        self.log("debug", f"Command: {train_commands}")
        self.train_process.start("python", train_commands)

    def _create_command_from_config(self, config, excluded_keys: str = []) -> List[str]:
        """Create a command recursively from the config dataclass."""
        dino_commands = []
        for f in fields(config):
            if f.name in excluded_keys:
                continue
            if is_dataclass(
                getattr(config, f.name)
            ):  # Top-level settings still need to be set later
                dino_commands += [
                    f.name + "." + cmd
                    for cmd in self._create_command_from_config(
                        getattr(config, f.name), excluded_keys
                    )
                ]
            elif isinstance(getattr(config, f.name), list) or isinstance(
                getattr(config, f.name), tuple
            ):
                list_commands = ", ".join(map(str, getattr(config, f.name)))
                dino_commands += [f"{f.name}=[{list_commands}]"]
            elif isinstance(getattr(config, f.name), Path):
                dino_commands += [
                    f"{f.name}='{str(getattr(config, f.name).resolve())}'"
                ]
            else:
                dino_commands += [f"{f.name}={getattr(config, f.name)}"]
        return dino_commands

    def _dino_process_finish(self, *args):
        self.log("success", "DINO features complete. Running model...")
        self.dino_process = None
        QGuiApplication.restoreOverrideCursor()

    def _segment_process_finish(self, *args):
        self.log("success", "Segmentation complete.")
        self.segment_process = None
        QGuiApplication.restoreOverrideCursor()

    def _train_process_finish(self, *args):
        self.log("success", "Training complete.")
        self.train_process = None
        QGuiApplication.restoreOverrideCursor()

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
            not self.replaceCheckboxProcTrain.isChecked(),
            self.replaceDirectoryLabelProcTrain,
            self.replaceDirectoryProcTrain,
            self.replaceSelectProcTrain,
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
        self.replaceCheckboxProcTrain.checkStateChanged.connect(
            lambda state: self._show_hide_widgets(
                state == Qt.CheckState.Unchecked,
                self.replaceDirectoryLabelProcTrain,
                self.replaceDirectoryProcTrain,
                self.replaceSelectProcTrain,
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

    def setup_model_select(self):
        """Setup model architecture selection for training."""
        # Setup model cache
        self._models = {}
        # Get available models from settings
        model_dir = self.settings.get_setting("model/model_directory")
        if not model_dir or not os.path.isdir(model_dir):
            self.log(
                "error",
                f"Invalid model directory: {model_dir}. Please set it in the settings.",
            )
            return
        available_models = get_available_models(Path(model_dir))
        self.modelCombo.addItems(available_models)
        self.modelCombo.currentIndexChanged.connect(self._update_current_model_info)
        self.modelCombo.setCurrentIndex(0)
        # Setup model buttons
        self.addModelButton.clicked.connect(self._add_model)
        self.removeModelButton.clicked.connect(self._remove_model)
        self.importModelButton.clicked.connect(self._import_model)

    @_catch_exceptions("update current model info")
    def _update_current_model_info(self, *args):
        """Update the current model info based on the selected model."""
        model_name = self.modelCombo.currentText()
        # Check for cache
        if model_name in self._models:
            self.selectedModelArea.setWidget(
                self._models[model_name]["widget"].widget()
            )
            return
        model_dir = self.settings.get_setting("model/model_directory")
        if model_dir:
            model_dir = Path(model_dir).resolve()
        else:
            self.log(
                "error",
                f"Invalid model directory: {model_dir}. Please set it in the settings.",
            )
            return
        model_config = get_model_configs(model_dir, [model_name])[0]
        self.selectedModelArea.setWidget(
            self._create_model_scroll(model_name, model_config).widget()
        )

    def _create_model_scroll(
        self, model_name: str, model_config: InterfaceModelConfig
    ) -> QWidget:
        """Update the model scroll area with the current model config."""
        scroll = QScrollArea()
        contents = QWidget()
        form_layout = QFormLayout(contents)
        form_layout.setWidget(
            0,
            QFormLayout.ItemRole.LabelRole,
            QLabel(parent=contents, text="Model Name:"),
        )
        form_layout.setWidget(
            0,
            QFormLayout.ItemRole.FieldRole,
            QLabel(parent=contents, text=model_config.name),
        )
        form_layout.setWidget(
            1,
            QFormLayout.ItemRole.LabelRole,
            QLabel(parent=contents, text="Model Architecture:"),
        )
        form_layout.setWidget(
            1,
            QFormLayout.ItemRole.FieldRole,
            QLabel(parent=contents, text=model_config.model_type.value),
        )
        form_layout.setWidget(
            2,
            QFormLayout.ItemRole.LabelRole,
            QLabel(parent=contents, text="Dataset Samples:"),
        )
        form_layout.setWidget(
            2,
            QFormLayout.ItemRole.FieldRole,
            QLabel(
                parent=contents,
                text=(
                    "".join(model_config.samples)
                    if len(model_config.samples) < 2
                    else ", ".join(model_config.samples)
                ),
            ),
        )
        form_layout.setWidget(
            3,
            QFormLayout.ItemRole.LabelRole,
            QLabel(parent=contents, text="Segmentation Label:"),
        )
        form_layout.setWidget(
            3,
            QFormLayout.ItemRole.FieldRole,
            QLabel(parent=contents, text=model_config.label_key),
        )
        scroll.setWidget(contents)
        # Update cache
        self._models[model_name] = {
            "widget": scroll,
            "config": model_config,
        }
        # Add scroll to self
        setattr(self, f"{model_name}Contents", contents)
        setattr(self, f"{model_name}Scroll", scroll)
        return scroll

    @_catch_exceptions("add segmentation model")
    def _add_model(self, *args):
        """Add a new model to the model list."""
        model_name = self.modelCombo.currentText()
        # Check for cache
        if model_name in self._models:
            index = self.modelTabs.addTab(
                self._models[model_name]["widget"], model_name
            )
            self.modelTabs.setCurrentIndex(index)
            return
        model_dir = self.settings.get_setting("model/model_directory")
        if model_dir:
            model_dir = Path(model_dir).resolve()
        else:
            self.log(
                "error",
                f"Invalid model directory: {model_dir}. Please set it in the settings.",
            )
            return
        model_config = get_model_configs(model_dir, [model_name])[0]
        index = self.modelTabs.addTab(
            self._create_model_scroll(model_name, model_config), model_name
        )
        self.modelTabs.setCurrentIndex(index)
        self.log("success", f"Added model: {model_name}")

    @_catch_exceptions("remove segmentation model")
    def _remove_model(self, *args):
        """Remove the selected model from the model list."""
        model_name = self.modelCombo.currentText()
        if model_name:
            self.modelTabs.removeTab(self.modelTabs.currentIndex())
            self.log("success", f"Removed model: {model_name}")
        else:
            self.log("warning", "No model selected to remove.")

    @_catch_exceptions("import segmentation model")
    def _import_model(self, *args):
        """Import a new model(s) from a file."""
        model_dir = self.settings.get_setting("model/model_directory")
        if model_dir:
            model_dir = Path(model_dir).resolve()
        else:
            self.log(
                "error",
                f"Invalid model directory: {model_dir}. Please set it in the settings.",
            )
            return
        model_paths = select_file_folder_dialog(
            self,
            "Select model file(s):",
            False,
            True,
            file_types=[".pt", ".pth"],
            start_dir=str(model_dir),
        )
        # Check for empty selection
        if not model_paths:
            return
        for model_path in model_paths:
            # Ask user for config
            model_name = os.path.basename(model_path)
            model_config = InterfaceModelConfig(
                model_name, "", ModelArch.CRYOVIT, Path(model_path), {}, []
            )
            config_dialog = ModelDialog(self, model_config)
            result = config_dialog.exec()
            if result == config_dialog.DialogCode.Rejected:
                self.log("info", f"Skipping import for model {model_name}.")
                continue
            model_config = config_dialog.config
            # Save config to disk
            save_model_config(model_dir, model_config)
            # Add model to list
            self.modelCombo.addItem(model_name)
            self.modelCombo.setCurrentText(model_name)
            index = self.modelTabs.addTab(
                self._create_model_scroll(model_name, model_config), model_name
            )
            self.modelTabs.setCurrentIndex(index)
            self.log("success", f"Imported model: {model_name}")
        else:
            self.log("warning", "No model file selected.")

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
        self.rawSelectTrain.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.rawDirectoryTrain,
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
        self.replaceSelectProcTrain.clicked.connect(
            partial(
                self._file_directory_prompt,
                self.replaceDirectoryProcTrain,
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
        self.rawDirectoryTrain.editingFinished.connect(
            partial(self._update_file_directory_field, self.rawDirectoryTrain, True)
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
        self.replaceDirectoryProcTrain.editingFinished.connect(
            partial(
                self._update_file_directory_field,
                self.replaceDirectoryProcTrain,
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
                start_dir=(self.settings.get_setting("general/data_directory")),
            )
        )

    @_catch_exceptions("validate file/directory field")
    def _update_file_directory_field(self, text_field, is_folder: bool, *args):
        """Validate and update a file or directory field to ensure it is a valid directory."""
        current_text = text_field.text()
        valid_text = (
            text_field.text()
            if (os.path.isdir(text_field.text()) and is_folder)
            or (os.path.isfile(text_field.text()) and not is_folder)
            else ""
        )
        text_field.setText(valid_text)
        if not valid_text:
            self.log("error", f"Invalid folder path: {current_text}")

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
                Path(self.settings.get_setting("model/model_directory"))
                / self.train_model_config.name
                / "weights.pt"
            )
            self.trainer_config = config_dialog.trainer_config
            if self.train_model_config.samples:
                self.sampleSelectCombo.setCurrentData(self.train_model_config.samples)
            self.log("success", "Training configuration updated.")

    def setup_run_buttons(self):
        """Setup the run buttons for processing and training."""
        self.processButtonSeg.clicked.connect(partial(self.run_preprocessing, False))
        self.processButtonTrain.clicked.connect(partial(self.run_preprocessing, True))
        self.chimeraButton.clicked.connect(self.run_chimerax)
        self.splitsButton.clicked.connect(self.run_generate_training_splits)
        self.splitsButtonNew.clicked.connect(self.run_new_training_splits)
        self.featureButtonSeg.clicked.connect(self.run_feature_extraction)
        self.featureButtonTrain.clicked.connect(self.run_feature_extraction)
        self.segmentButton.clicked.connect(self.run_segmentation)
        self.trainButton.clicked.connect(self.run_training)

    def setup_menu(self):
        """Setup the menu bar for loading and saving presets."""
        self.actionLoad_Preset.triggered.connect(self._load_preset)
        self.actionSave_Preset.triggered.connect(partial(self._save_preset, True))
        self.actionSaveAs_Preset.triggered.connect(partial(self._save_preset, False))

        self.actionSettings.triggered.connect(self._open_settings)
        self.actionGithub.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/VivianDLi/CryoViT")
            )
        )

    @_catch_exceptions("load preset settings")
    def _load_preset(self, *args):
        """Opens a prompt to optionally load previous settings from a saved name."""
        available_presets = (
            sorted(
                [name for name in self.settings.get_setting("preset/available_presets")]
            )
            if self.settings.get_setting("preset/available_presets")
            else []
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
                self.settings.get_setting("preset/available_presets")
                if self.settings.get_setting("preset/available_presets")
                else []
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
                temp_settings.setValue(key, self.settings.get_setting(key))
            self.settings.set_setting("preset/current_preset", name)
            self.log("success", f"Saved preset: {name}")

    @_catch_exceptions("open settings")
    def _open_settings(self, *args):
        """Open the settings window."""
        result = self.settings.exec()
        if result == self.settings.DialogCode.Accepted:
            self.setup_model_select()
            self.log("success", "Settings saved.")

    def _show_hide_widgets(self, visible: bool, *widgets):
        """Show or hide multiple widgets."""
        for widget in widgets:
            widget.setVisible(visible)

    def _update_progress_bar(self, index: int, total: int):
        """Update the progress bar with the current index (assume 0-indexed) and total."""
        self.progressBar.setValue((index // total) * self.progressBar.maximum())

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
        self.settings.save_settings()
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
