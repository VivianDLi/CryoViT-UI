"""Main application window for CryoVit segmentation and training."""

import os
from pathlib import Path
import sys
from functools import partial
import platform
import traceback

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
from PyQt6.QtGui import QDesktopServices

from cryovit.config import InterfaceModelConfig, ModelArch, models
import cryovit.gui.resources
from cryovit.gui.layouts.mainwindow import Ui_MainWindow
from cryovit.gui.model_config import ModelDialog, TrainerFit
from cryovit.gui.settings import PresetDialog, SettingsWindow
from cryovit.gui.utils import (
    EmittingStream,
    MultiSelectComboBox,
    select_file_folder_dialog,
)
from cryovit.processing import *


class WorkerSignals(QObject):
    """Signals for worker threads."""

    finish = pyqtSignal()
    progress = pyqtSignal(int, int)
    error = pyqtSignal(tuple)


class Worker(QRunnable):
    """Worker thread for running tasks in the background."""

    def __init__(self, fn, *args, has_progress: bool = False, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.has_progress = has_progress
        self.signals = WorkerSignals()

    def run(self):
        """Run the function in the worker thread."""
        try:
            if self.has_progress:
                self.fn(
                    *self.args, **self.kwargs, callback_fn=self.signals.progress.emit
                )
            else:
                self.fn(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finish.emit()


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow, Ui_MainWindow):
    def _catch_exceptions(desc: str, concurrent: bool = False):
        """Decorator to catch exceptions when running any function."""

        def inner(func):
            def wrapper(self, *args, **kwargs):
                try:
                    if concurrent and self.running:  # if not running in a thread
                        self.log(
                            "warning"
                            f"Cannot run {desc}: Another process is already running."
                        )
                    return func(self, *args, **kwargs)
                except Exception as e:
                    self.log("error", f"Error running {desc}: {e}")

            return wrapper

        return inner

    def _handle_thread_exception(self, info, *args):
        """Handle exceptions in worker threads."""
        exctype, value, traceback_info = info
        self.log(
            "error",
            f"Error in thread: {exctype}: {value}.\n{traceback_info}",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        # Setup settings
        self.settings = SettingsWindow(self)
        # Setup thread pool
        self.running = False  # Flag to check if a process is running
        self.threadpool = QThreadPool()

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
        except Exception as e:
            sys.stderr.write(f"Error setting up UI: {e}")
            sys.exit(1)

        self.setup_console()
        self.log("success", "Welcome to CryoViT!")

    @_catch_exceptions("preprocessing", concurrent=True)
    def run_preprocessing(self, is_train: bool, *args):
        self.running = True

        def on_finish():
            self.running = False
            self.log("success", "Preprocessing complete.")

        # Get optional kwargs from settings
        kwargs = {}
        if self.settings.get_setting("preprocessing/bin_size"):
            kwargs["bin_size"] = self.settings.get_setting("preprocessing/bin_size")
        if self.settings.get_setting("preprocessing/normalize"):
            kwargs["normalize"] = self.settings.get_setting("preprocessing/normalize")
        if self.settings.get_setting("preprocessing/clip"):
            kwargs["clip"] = self.settings.get_setting("preprocessing/clip")
        # Check for valid directories
        if is_train:
            raw_dir = self.rawDirectoryTrain.text()
            replace = self.replaceCheckboxProcTrain.isChecked()
            replace_dir = self.replaceDirectoryProcTrain
        else:
            raw_dir = self.rawDirectory.text()
            replace = self.replaceCheckboxProc.isChecked()
            replace_dir = self.replaceDirectoryProc
        if not os.path.isdir(raw_dir):
            self.log(
                "error",
                f"Invalid raw directory: {raw_dir}",
            )
            self.running = False
            return
        if replace:
            replace_dir.setText(raw_dir)
        src_dir = Path(raw_dir)
        dst_dir = Path(replace_dir.text())

        samples = self.sampleSelectCombo.getCurrentData()

        if is_train and samples:
            self.preproc_completed = 0

            def update_preproc_completed():
                self.preproc_completed += 1
                self._update_progress_bar(self.preproc_completed - 1, len(samples))
                if self.preproc_completed == len(samples):
                    on_finish()

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
                    has_progress=True if len(samples) == 1 else False,
                    **kwargs,
                )
                if len(samples) == 1:
                    worker.signals.progress.connect(self._update_progress_bar)
                    worker.signals.finish.connect(on_finish)
                else:
                    worker.signals.finish.connect(update_preproc_completed)
                worker.signals.error.connect(self._handle_thread_exception)
                self.threadpool.start(worker)
            return
        worker = Worker(run_preprocess, src_dir, dst_dir, has_progress=True, **kwargs)
        worker.signals.progress.connect(self._update_progress_bar)
        worker.signals.finish.connect(on_finish)
        worker.signals.error.connect(self._handle_thread_exception)
        self.threadpool.start(worker)

    @_catch_exceptions("ChimeraX")
    def run_chimerax(self, *args):
        chimerax_path = self.settings.get_setting("annotation/chimerax_path")
        if not chimerax_path and platform.system().lower() != "linux":
            self.log(
                "error",
                "ChimeraX path not set. Please set it in the settings.",
            )
            return

        # Get arguments from settings
        samples = self.sampleSelectCombo.getCurrentData()
        raw_dir = self.rawDirectoryTrain.text()
        replace_proc = self.replaceCheckboxProcTrain.isChecked()
        replace_proc_dir = self.replaceDirectoryProcTrain.text()
        src_dir = Path(raw_dir if replace_proc else replace_proc_dir)
        if self.sliceDirectory.text():
            dst_dir = Path(self.sliceDirectory.text())
        else:
            dst_dir = (
                src_dir.resolve() / "slices"
                if samples
                else src_dir.parent.resolve() / "slices"
            )
            self.log(
                "warning",
                f"No slice directory specified. Slices will be saved in {src_dir.resolve() / 'slices/sample' if samples else src_dir.parent.resolve() / 'slices'}.",
            )
        if self.csvDirectory.text():
            csv_dir = Path(self.csvDirectory.text())
        else:
            csv_dir = (
                src_dir.parent.resolve() / "csv"
                if samples
                else src_dir.parent.parent.resolve() / "csv"
            )
            self.log(
                "warning",
                f"No CSV directory specified. CSV files will be saved in {src_dir.parent.resolve() / 'csv' if samples else src_dir.parent.parent.resolve() / 'csv'}.",
            )

        num_slices = self.settings.get_setting("annotation/num_slices")
        # Check for no samples
        if not samples:
            if not os.path.isdir(src_dir):
                self.log(
                    "error",
                    f"Invalid raw directory: {src_dir}",
                )
                return
            self.running = True
            self._launch_chimerax(
                chimerax_path,
                src_dir,
                dst_dir=dst_dir,
                csv_dir=csv_dir,
                num_slices=num_slices,
            )
            if self.chimera_process:
                self.chimera_process.waitForFinished(-1)
                self.chimera_process = None
        else:
            self.running = True
            for i in range(len(samples)):
                # Check for valid directories
                if not os.path.isdir(src_dir / samples[i]):
                    self.log(
                        "error",
                        f"Invalid raw directory: {src_dir / samples[i]}",
                    )
                    continue
                self._launch_chimerax(
                    chimerax_path if chimerax_path else "",
                    src_dir,
                    samples[i],
                    dst_dir=dst_dir,
                    csv_dir=csv_dir,
                    num_slices=num_slices,
                )
                if self.chimera_process:
                    self.chimera_process.waitForFinished(-1)
                    self._update_progress_bar(i, len(samples))
                    self.chimera_process = None
                else:
                    self.log(
                        "warning",
                        f"ChimeraX process for src_dir: {src_dir / samples[i]} failed.",
                    )
                    continue
        self.running = False
        self.log("success", "ChimeraX processing complete.")

    def _launch_chimerax(
        self,
        chimerax_path: str,
        src_dir: Path,
        sample: str = None,
        dst_dir: Path = None,
        csv_dir: Path = None,
        num_slices: int = 5,
    ):
        import subprocess

        self.chimera_process = QProcess()
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
        # Check for OS type
        match platform.system().lower():
            case "windows":  # Needs to be run from a specific path
                # Check for valid path
                if not os.path.isfile(
                    chimerax_path
                ) or not chimerax_path.lower().endswith("chimerax.exe"):
                    self.log(
                        "error",
                        f"Invalid ChimeraX path: {chimerax_path}. This should be the path to the ChimeraX.exe executable typically found in 'C:/Program Files/ChimeraX/bin/ChimeraX.exe'. Please set it in the settings.",
                    )
                    return None
                # Copy command to clipboard
                subprocess.check_call("echo " + command + " | clip", shell=True)
                # Launch ChimeraX normally
                self.chimera_process.start(chimerax_path)
            case "linux":  # Has chimerax from command line
                # Copy command to clipboard
                subprocess.check_call("echo " + command + " | xsel -ib", shell=True)
                self.chimera_process.start("chimerax")
            case "darwin":  # Needs to be run from a specific path
                if not os.path.isfile(
                    chimerax_path
                ) or not chimerax_path.lower().endswith("chimerax.app"):
                    self.log(
                        "error",
                        f"Invalid ChimeraX path: {chimerax_path}. This should be the path to the ChimeraX.app application typically found in '/Applications/ChimeraX.app'. Please set it in the settings.",
                    )
                    return None
                # Copy command to clipboard
                subprocess.check_call("echo " + command + " | pbcopy", shell=True)
                self.chimera_process.start(chimerax_path)
            case _:
                self.log("error", f"Unsupported OS type {platform.system()}.")
        return self.chimera_process

    @_catch_exceptions("generate training splits", concurrent=True)
    def run_generate_training_splits(self, *args):
        self.running = True

        def on_finish():
            self.running = False
            self.log("success", "Generate training splits complete.")

        # Get directory from settings
        raw_dir = self.rawDirectory.text()
        replace_proc = self.replaceCheckboxProc.isChecked()
        replace_proc_dir = self.replaceDirectoryProc
        src_dir = Path(replace_proc_dir.text() if replace_proc else raw_dir)
        dst_dir = src_dir
        samples = self.sampleSelectCombo.getCurrentData()
        src_dirs = [src_dir / sample for sample in samples] if samples else [src_dir]
        dst_dirs = [dst_dir / sample for sample in samples] if samples else [dst_dir]
        annot_dir = Path(self.annoDirectory.text())
        annot_dirs = (
            [annot_dir / sample for sample in samples] if samples else [annot_dir]
        )
        csv_dir = Path(self.csvDirectory.text())
        csv_files = [csv_dir / f"{sample}.csv" for sample in samples]
        num_splits = self.settings.get_setting("training/splits")
        seed = self.settings.get_setting("training/split_seed")

        self.splits_completed = 0

        def update_samples_completed():
            self.splits_completed += 1
            self._update_progress_bar(self.splits_completed - 1, len(samples))
            if self.splits_completed == len(samples):
                on_finish()

        for i in range(len(samples)):

            def run_both(callback_fn: callable = None):
                add_annotations(
                    src_dir=src_dirs[i],
                    dst_dir=dst_dirs[i],
                    annot_dir=annot_dirs[i],
                    csv_file=csv_files[i],
                    features=self.features,
                    callback_fn=callback_fn,
                )

                add_splits(
                    dst_dir=csv_dir,
                    csv_file=csv_files[i],
                    sample=samples[i],
                    num_splits=num_splits,
                    seed=seed,
                    callback_fn=callback_fn,
                )

            worker = Worker(run_both, has_progress=True if len(samples) == 1 else False)
            if len(samples) == 1:
                worker.signals.progress.connect(self._update_progress_bar)
                worker.signals.finish.connect(on_finish)
            else:
                worker.signals.finish.connect(update_samples_completed)
            worker.signals.error.connect(self._handle_thread_exception)
            self.threadpool.start(worker)

    @_catch_exceptions("generate new training splits", concurrent=True)
    def run_new_training_splits(self, *args):
        self.running = True

        def on_finish():
            self.running = False
            self.log("success", "Generate new training splits complete.")

        # Get directory from settings
        csv_dir = Path(self.csvDirectory.text())
        num_splits = self.settings.get_setting("training/splits")
        seed = self.settings.get_setting("training/split_seed")

        splits_file = self.settings.get_setting("training/splits_file")
        if not splits_file:
            splits_file = csv_dir / "splits.csv"
            self.settings.set_settings("training/splits_file", str(splits_file))
        else:
            splits_file = Path(splits_file)

        dst_name = QInputDialog.getText(self, "New Split Name", "Enter new split name:")
        if dst_name in csv_dir.glob("*.csv"):
            result = QMessageBox.warning(
                self,
                "Warning!",
                f"This will overwriting existing splits {dst_name}. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.No:
                self.running = False
                return

        worker = Worker(
            generate_new_splits,
            splits_file,
            dst_file=csv_dir / f"{dst_name}.csv",
            num_splits=num_splits,
            seed=seed,
            has_progress=True,
        )
        worker.signals.progress.connect(self._update_progress_bar)
        worker.signals.finish.connect(on_finish)
        worker.signals.error.connect(self._handle_thread_exception)
        self.threadpool.start(worker)

    @_catch_exceptions("segmentation", concurrent=True)
    def run_segmentation(self, *args):
        self.running = True

        def on_finish():
            self.running = False
            self.log("success", "Segmentation complete.")

        # Check for required settings
        model_dir = self.settings.get_setting("model/model_dir")
        if not model_dir:
            self.log(
                "error",
                f"Invalid model directory: {model_dir}. Please set it in the settings.",
            )
            self.running = False
            return
        dino_dir = self.settings.get_setting("general/dino_dir")
        if not dino_dir:
            self.log(
                "error",
                f"Invalid DINO directory: {dino_dir}. This is where the DINOv2 model will be saved. Please set it in the settings.",
            )
            self.running = False
            return
        features_dir = self.settings.get_setting("general/features_dir")
        if not features_dir:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the DINO directory in the settings.",
            )
            features_dir = None
        kwargs = {}
        if self.settings.get_setting("segmentation/batch_size"):
            kwargs["batch_size"] = self.settings.get_setting("segmentation/batch_size")
        if self.settings.get_setting("segmentation/csv_file"):
            kwargs["csv_file"] = self.settings.get_setting("segmentation/csv_file")
        # Get direcctories
        raw_dir = self.rawDirectory.text()
        replace_proc = self.replaceCheckboxProc.isChecked()
        replace_proc_dir = self.replaceDirectoryProc
        src_dir = Path(replace_proc_dir.text() if replace_proc else raw_dir)
        replace_seg = self.replaceCheckboxSeg.isChecked()
        replace_seg_dir = self.replaceDirectorySeg
        dst_dir = Path(replace_seg_dir.text() if replace_seg else src_dir)

        batch_size = self.settings.get_setting("segmentation/batch_size")
        csv_file = (
            self.settings.get_setting("segmentation/csv_file")
            if self.settings.get_setting("segmentation/csv_file")
            else None
        )

        # Get model list
        model_names = [
            page
            for page in self._models
            if self.modelTabs.indexOf(self._models[page]["widget"]) != -1
        ]

        models, configs = zip(*[load_model(model_dir, name) for name in model_names])

        self.segments_completed = 0

        def update_segments_completed():
            self.segments_completed += 1
            self._update_progress_bar(self.segments_completed - 1, len(models))
            if self.samples_completed == len(models):
                on_finish()

        for i in range(len(models)):

            def run_both(callback_fn: callable = None):
                get_dino_features(dino_dir, src_dir, dst_dir=features_dir, **kwargs)
                run_inference(
                    models[i],
                    configs[i],
                    src_dir if features_dir is None else features_dir,
                    batch_size=batch_size,
                    dst_dir=dst_dir,
                    csv_file=csv_file,
                    callback_fn=callback_fn,
                )

            worker = Worker(run_both, has_progress=True if len(models) == 1 else False)
            if len(models) == 1:
                worker.signals.progress.connect(self._update_progress_bar)
                worker.signals.finish.connect(on_finish)
            else:
                worker.signals.finish.connect(update_segments_completed)
            worker.signals.error.connect(self._handle_thread_exception)
            self.threadpool.start(worker)

    @_catch_exceptions("training", concurrent=True)
    def run_training(self, *args):
        self.running = True

        def on_finish():
            self.running = False
            self.log("success", "Training complete.")

        # Check for required settings
        model_dir = self.settings.get_setting("model/model_dir")
        if not model_dir:
            self.log(
                "error",
                f"Invalid model directory: {model_dir}. Please set it in the settings.",
            )
            self.running = False
            return
        dino_dir = self.settings.get_setting("general/dino_dir")
        if not dino_dir:
            self.log(
                "error",
                f"Invalid DINO directory: {dino_dir}. This is where the DINOv2 model will be saved. Please set it in the settings.",
            )
            self.running = False
            return
        features_dir = self.settings.get_setting("general/features_dir")
        if not features_dir:
            self.log(
                "warning",
                f"No DINO directory specified. This will add DINO features to the input tomograms. If you want to save DINO features separately, please set the DINO directory in the settings.",
            )
            features_dir = None
        if not self.train_model or self.train_model_config:
            self.log(
                "error",
                "No model selected. Please select a model in the 'Training' section in the 'Train Model' tab.",
            )
            self.running = False
            return

        # Get directories
        raw_dir = self.rawDirectory.text()
        replace_proc = self.replaceCheckboxProc.isChecked()
        replace_proc_dir = self.replaceDirectoryProc
        data_dir = Path(replace_proc_dir.text() if replace_proc else raw_dir)
        csv_dir = Path(self.csvDirectory.text())
        split_file = self.settings.get_setting("training/splits_file")
        if not split_file:
            split_file = csv_dir / "splits.csv"
            self.settings.set_settings("training/splits_file", str(split_file))
        batch_size = self.settings.get_setting("training/batch_size")
        split_id = self.settings.get_setting("training/split_id")
        seed = self.settings.get_setting("training/split_seed")

        # Run training
        def run_all(callback_fn: callable = None):
            get_dino_features(
                dino_dir,
                data_dir,
                batch_size,
                dst_dir=features_dir,
                callback_fn=callback_fn,
            )
            train_model(
                self.train_model,
                self.train_model_config,
                self.trainer_config,
                data_dir if features_dir is None else features_dir,
                split_file,
                batch_size,
                split_id,
                seed,
            )
            save_model(self.train_model_config, model_dir, model=self.train_model)

        worker = Worker(run_all, has_progress=True)
        worker.signals.progress.connect(self._update_progress_bar)
        worker.signals.finish.connect(on_finish)
        worker.signals.error.connect(self._handle_thread_exception)
        self.threadpool.start(worker)

    def setup_console(self):
        sys.stdout = EmittingStream(textWritten=partial(self.log, "info", end=""))
        sys.stderr = EmittingStream(textWritten=partial(self.log, "error", end=""))

    def setup_checkboxes(self):
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
        old_combo = self.sampleSelectCombo
        self.sampleSelectCombo = MultiSelectComboBox(parent=self.sampleSelectFrame)
        self.sampleSelectCombo.setSizePolicy(old_combo.sizePolicy())
        self.sampleSelectCombo.setObjectName(old_combo.objectName())
        self.sampleSelectCombo.setToolTip(old_combo.toolTip())
        self.sampleSelectCombo.setPlaceholderText(old_combo.placeholderText())
        self.sampleSelectLayout.replaceWidget(old_combo, self.sampleSelectCombo)
        old_combo.deleteLater()

        self.sampleAdd.clicked.connect(self._add_sample)

    @_catch_exceptions("update sample select")
    def _add_sample(self, *args):
        """Add a new sample to the sample list."""
        self.sampleSelectCombo.addNewItem()

    def setup_model_select(self):
        # Setup model cache
        self._models = {}
        # Get available models from settings
        model_dir = self.settings.get_setting("model/model_dir")
        if not model_dir:
            if not os.path.isdir(model_dir):
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
        model_dir = self.settings.get_setting("model/model_dir")
        if not model_dir:
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
            QLabel(parent=contents, text=model_config.architecture.value),
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
        form_layout.setWidget(
            4,
            QFormLayout.ItemRole.LabelRole,
            QLabel(parent=contents, text="Training Metrics:"),
        )
        form_layout.setWidget(
            4,
            QFormLayout.ItemRole.FieldRole,
            QLabel(parent=contents, text=model_config.metrics),
        )
        scroll.setWidget(contents)
        # Update cache
        self._models[model_name] = {
            "widget": scroll,
            "config": model_config,
        }
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
        model_dir = self.settings.get_setting("model/model_dir")
        if not model_dir:
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
            self.log("error", "No model selected to remove.")

    @_catch_exceptions("import segmentation model")
    def _import_model(self, *args):
        """Import a new model(s) from a file."""
        model_dir = self.settings.get_setting("model/model_dir")
        if not model_dir:
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
            file_types=".pt",
        )
        for model_path in model_paths:
            # Ask user for config
            model_name = os.path.basename(model_path)
            config_dialog = ModelDialog(self)
            result = config_dialog.exec()
            if result == config_dialog.DialogCode.Rejected:
                self.log("info", f"Skipping import for model {model_name}.")
                continue
            model_config = config_dialog.config
            # Save config to disk
            save_model(model_config, model_dir, model_name)
            # Add model to list
            self.modelCombo.addItem(model_name)
            self.modelCombo.setCurrentText(model_name)
            index = self.modelTabs.addTab(
                self._create_model_scroll(model_name, model_config), model_name
            )
            self.modelTabs.setCurrentIndex(index)
            self.log("success", f"Imported model: {model_name}")
        else:
            self.log("error", "No model file selected.")

    def setup_folder_selects(self):
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
                start_dir=(self.settings.get_setting("general/data_dir")),
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
            self.log("warning", f"Invalid folder path: {current_text}")

    def setup_feature_select(self):
        self.features = []
        self.featuresAdd.clicked.connect(self._add_feature)
        self.featuresDisplay.editingFinished.connect(lambda: self._update_features)

    @_catch_exceptions("add training feature")
    def _add_feature(self, *args):
        feature_name, _ = QInputDialog.getText(
            self, "Add Feature", "Enter feature name:"
        )
        if feature_name:
            self.features.append(feature_name)
            self.featuresDisplay.setText(", ".join(self.features))
            self._update_features()

    @_catch_exceptions("update training features")
    def _update_features(self, *args):
        current_text = self.featuresDisplay.text()
        if current_text:
            self.features = sorted(
                [feature.strip() for feature in current_text.split(",")]
            )
            self.featuresDisplay.setText(", ".join(self.features))
        else:
            self.features = []

        self.labelCombo.clear()
        self.labelCombo.addItems(self.features)

    def setup_training_config(self):
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
                "error",
                "No features specified. Please add features in the 'Annotations' section above.",
            )
        self.labelCombo.currentTextChanged.connect(self._update_train_model_label)
        self.labelCombo.setCurrentIndex(0)

    @_catch_exceptions("update training model architecture")
    def _update_train_model_arch(self, text: str, *args):
        if not self.train_model_config:
            samples = self.sampleSelectCombo.getCurrentData()
            self.train_model_config = InterfaceModelConfig(
                name=text.lower(),
                label_key="",
                model_type=ModelArch[text],
                model_params={},
                samples=samples,
                metrics={},
            )
        self.train_model_config.model_type = ModelArch[text]
        self.train_model = load_base_model(self.train_model_config)

    @_catch_exceptions("update training model label")
    def _update_train_model_label(self, text: str, *args):
        if not self.train_model_config:
            self.log(
                "error",
                "No model selected. Please select a model in the 'Training' section.",
            )
            return
        self.train_model_config.label_key = text

    @_catch_exceptions("open training config")
    def _open_train_config(self, *args):
        if not self.train_model_config:
            self.log(
                "error",
                "No model selected. Please select a model in the 'Training' section.",
            )
            return
        config_dialog = ModelDialog(self, trainer_config=self.trainer_config)
        config_dialog.exec()
        if config_dialog.result() == config_dialog.DialogCode.Accepted:
            self.train_model_config = config_dialog.config
            self.trainer_config = config_dialog.trainer_config
            self.sampleSelectCombo.setCurrentData(self.train_model_config.samples)
            self.log("success", "Training configuration updated.")

    def setup_run_buttons(self):
        self.processButtonSeg.clicked.connect(partial(self.run_preprocessing, False))
        self.processButtonTrain.clicked.connect(partial(self.run_preprocessing, True))
        self.chimeraButton.clicked.connect(self.run_chimerax)
        self.splitsButton.clicked.connect(self.run_generate_training_splits)
        self.splitsButtonNew.clicked.connect(self.run_new_training_splits)
        self.trainButton.clicked.connect(self.run_training)

    def setup_menu(self):
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
        self.progressBar.setValue((index + 1 // total) * self.progressBar.maximum())

    def log(self, mode: str, text: str, end="\n"):
        """Write text to the console with a timestamp and different colors for normal output and errors."""
        import time

        # Check for whitespaces (no timestamp)
        if not text.strip():
            self.consoleText.insertPlainText(text)
            return
        # Get timestamp
        timestamp_str = "{}> ".format(time.strftime("%X"))
        # Format text with colors and timestamp
        full_text = timestamp_str + text + end
        match mode:
            case "error":
                full_text = '<font color="#FF4500">{}</font>'.format(full_text)
            case "warning":
                full_text = '<font color="#FFA500">{}</font>'.format(full_text)
            case "success":
                full_text = '<font color="#32CD32">{}</font>'.format(full_text)
            case _:
                full_text = '<font color="white">{}</font>'.format(full_text)
        # Add line breaks in HTML
        full_text = full_text.replace("\n", "<br>")
        # Write to console ouptut
        self.consoleText.insertHtml(full_text)

    def closeEvent(self, event):
        """Override close event to save settings."""
        self.settings.save_settings()
        event.accept()


app = QApplication(sys.argv)
app.setStyle("Fusion")

window = MainWindow()
window.show()

app.exec()
