"""Define a base item model for tracking completed annotations for the CryoViT GUI application."""

import pandas as pd

from PyQt6.QtCore import (
    QAbstractTableModel,
    QSortFilterProxyModel,
    QModelIndex,
    QVariant,
    Qt,
)
from PyQt6.QtGui import QBrush, QColor

from cryovit_gui.config import *

#### Logging Setup ####

import logging

logger = logging.getLogger("cryovit.models.file")


class FileModel(QAbstractTableModel):
    """
    Base class for storing file/completion data in the CryoViT GUI application.

    This class provides a common interface for item models initialized tracking annotation completion, including methods
    for getting data, setting data, and handling model-specific logic.
    """

    def __init__(self):
        super().__init__()
        self._root_dir: Path = None
        self._colors = [
            QColor(*Colors.RED.value),
            QColor(*Colors.DARK_ORANGE.value),
            QColor(*Colors.ORANGE.value),
            QColor(*Colors.YELLOW.value),
        ]

        self.file_data: FileData = {}

    def __init__(
        self,
        root_dir: Path = None,
        prev_data: FileData = None,
        update_selection: bool = True,
    ):
        """Create new FileModel instance with the given root directory, optionally loading selection from previous data."""
        super().__init__()
        self._root_dir = root_dir
        self._colors = [
            QColor(*Colors.RED.value),
            QColor(*Colors.DARK_ORANGE.value),
            QColor(*Colors.ORANGE.value),
            QColor(*Colors.YELLOW.value),
        ]

        if self._root_dir:
            for subdir in required_directories:
                setattr(self, f"_{subdir}_dir", self._root_dir / subdir)
            self.file_data = self._read_data()
            if update_selection and prev_data is not None:
                for sample in self.file_data:
                    if sample in prev_data:
                        matching_idx = [
                            (
                                self.file_data[sample]["tomogram_files"].index(
                                    tomo_file
                                ),
                                prev_data[sample]["tomogram_files"].index(tomo_file),
                            )
                            for tomo_file in self.file_data[sample]["tomogram_files"]
                            if tomo_file in prev_data[sample]["tomogram_files"]
                        ]
                        if matching_idx:
                            for c_i, p_i in matching_idx:
                                self.file_data[sample]["selected"][c_i] = prev_data[
                                    sample
                                ]["selected"][p_i]
        else:  # Empty initialization
            self.file_data = {}

    @staticmethod
    def validate_root(root_dir: Path) -> bool:
        """Validate the root directory, returning True if the root directory has the correct structure (i.e., contains tomograms, csv, and slices subfolders), and False if not."""
        if root_dir is None or not root_dir.exists() or not root_dir.is_dir():
            logger.warning(
                f"Root directory {root_dir} does not exist or is not a directory."
            )
            return False
        for subdir in required_directories:
            if not (root_dir / subdir).exists():
                logger.warning(
                    f"Required subdirectory {subdir} does not exist in root directory {root_dir}."
                )
                return False
        return True

    def _read_data(self) -> FileData:
        """Get model data based on the current root directory."""
        # Get available tomogram files
        samples = [
            sample_dir.name
            for sample_dir in self._tomograms_dir.iterdir()
            if sample_dir.is_dir() and len(list(sample_dir.iterdir())) > 0
        ]
        data: FileData = {}
        for sample in samples:
            # Get tomogram files
            sample_dir = self._tomograms_dir / sample
            tomogram_files = [
                file
                for file in sample_dir.iterdir()
                if file.is_file() and file.suffix in tomogram_exts
            ]
            # Check slice selection state
            sample_csv = self._csv_dir / f"{sample}.csv"
            if not sample_csv.exists():
                annotated = [False] * len(tomogram_files)
            else:
                sample_df = pd.read_csv(sample_csv)
                annotated = [
                    not sample_df[sample_df["tomo_name"] == file.name].empty
                    for file in tomogram_files
                ]
            selected = [not annot for annot in annotated]
            # Check export state
            export_dir = self._slices_dir / sample
            exported = [
                bool(list(export_dir.glob(f"*{file.stem}*")) for file in tomogram_files)
            ]
            # Update the model data
            data[sample] = SampleData(
                tomogram_files=tomogram_files,
                annotated=annotated,
                selected=selected,
                exported=exported,
            )
        return data

    def get_directory(self, subdir: str) -> Path:
        """Get the directory for the given subdirectory."""
        if subdir not in required_directories:
            logger.warning(f"Subdirectory {subdir} is not a required subdirectory.")
            return None
        result = getattr(self, f"_{subdir}_dir", None)
        if not result:
            logger.warning("Subdirectory not set. Returning None.")
        return result

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return the item flags for the given index."""
        flag = Qt.ItemFlag.ItemNeverHasChildren
        match index.column():
            case 0:  # Sample Name
                return flag | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            case 1:  # Percentage
                return flag
            case 2:  # Tomogram File
                return (
                    flag | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                )
            case 3:  # Exported
                return flag
            case _:
                return Qt.ItemFlag.NoItemFlags

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows in the table."""
        rows = [
            len(self.file_data[sample]["tomogram_files"]) for sample in self.file_data
        ]
        return sum(rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns for the given parent index."""
        return 4  # Columns: Sample Name, Percentage, File Name, Exported

    def data(self, index: QModelIndex, role: int) -> QVariant:
        """Return the data for the given index and role."""
        sample, tomogram_index = self._get_data(index)
        if sample == "" or tomogram_index == -1:  # No data found
            return QVariant()
        samples_annotated = sum(self.file_data[sample]["annotated"])
        total_samples = len(self.file_data[sample]["tomogram_files"])
        percentage = round((samples_annotated / total_samples) * 100, 2)
        background_color = (
            self._colors[int(percentage // (100 // len(self._colors)))]
            if percentage < 100
            else QColor(*Colors.GREEN.value)
        )
        match index.column():
            case 0:  # Sample Name
                match role:
                    case Qt.ItemDataRole.DisplayRole:
                        return QVariant(
                            f"{sample} ({samples_annotated}/{total_samples})"
                        )
                    case Qt.ItemDataRole.BackgroundRole:
                        return QBrush(background_color)
                    case Qt.ItemDataRole.UserRole:
                        return QVariant(sample)
                    case _:
                        return QVariant()
            case 1:  # Percentage
                match role:
                    case Qt.ItemDataRole.DisplayRole:
                        return QVariant(percentage)
                    case Qt.ItemDataRole.BackgroundRole:
                        return QBrush(background_color)
                    case Qt.ItemDataRole.UserRole:
                        return QVariant((samples_annotated, total_samples))
                    case _:
                        return QVariant()
            case 2:  # Tomogram File
                match role:
                    case Qt.ItemDataRole.DisplayRole:
                        return QVariant(
                            self.file_data[sample]["tomogram_files"][
                                tomogram_index
                            ].name
                        )
                    case Qt.ItemDataRole.BackgroundRole:
                        if self.file_data[sample]["annotated"][tomogram_index]:
                            return QBrush(QColor(*Colors.GREEN.value))
                    case Qt.ItemDataRole.CheckStateRole:
                        return (
                            Qt.CheckState.Checked
                            if self.file_data[sample]["selected"][tomogram_index]
                            else Qt.CheckState.Unchecked
                        )
                    case Qt.ItemDataRole.UserRole:
                        return QVariant(
                            self.file_data[sample]["annotated"][tomogram_index]
                        )
                    case _:
                        return QVariant()
            case 3:  # Exported
                match role:
                    case Qt.ItemDataRole.BackgroundRole:
                        return (
                            QBrush(QColor(*Colors.GREEN.value))
                            if all(self.file_data[sample]["exported"])
                            else QBrush(QColor(*Colors.RED.value))
                        )
                    case Qt.ItemDataRole.UserRole:
                        return QVariant(
                            self.file_data[sample]["exported"][tomogram_index]
                        )
                    case _:
                        return QVariant()
            case _:
                return QVariant()

    def setData(
        self,
        index: QModelIndex,
        value: QVariant,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        """Set the data for the given index and role."""
        if not index.isValid() or role != Qt.ItemDataRole.CheckStateRole:
            return False
        sample, tomogram_index = self._get_data(index)
        if sample == "" or tomogram_index == -1:
            return False

        # Update selection state
        if isinstance(value, int):
            value = Qt.CheckState(value)
        if value == Qt.CheckState.Checked:
            self.file_data[sample]["selected"][tomogram_index] = True
        else:
            self.file_data[sample]["selected"][tomogram_index] = False

        self.dataChanged.emit(index, index)
        return True

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> QVariant:
        """Return the header data for the given section and orientation."""
        # Column headers
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            match section:
                case 0:
                    return QVariant("Sample Name")
                case 1:
                    return QVariant("Percentage Done")
                case 2:
                    return QVariant("Tomogram File")
                case 3:
                    return QVariant("Exported")
                case _:
                    return QVariant()

    def _get_data(self, index: QModelIndex) -> tuple[str, int]:
        """Get the sample name and tomogram index for the given index."""
        if not index.isValid():
            return "", -1
        cur_rows = index.row()
        for sample in self.file_data:
            if cur_rows < len(self.file_data[sample]["tomogram_files"]):
                return sample, cur_rows
            cur_rows -= len(self.file_data[sample]["tomogram_files"])
        logger.warning(f"Index {index} is out of bounds for the current data model.")
        return "", -1


## Proxy Models ##


class SampleModel(QSortFilterProxyModel):
    """
    Proxy model for displaying sample data in the CryoViT GUI application.

    This model provides a filtered view of the FileModel, allowing for sample-specific operations.
    """

    def __init__(self, source_model: QAbstractTableModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSourceModel(source_model)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Compare two indices for sorting."""
        if not left.isValid() or not right.isValid():
            return False
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.DisplayRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.DisplayRole)

        return (
            left_data.value() < right_data.value()
            if isinstance(left_data, QVariant) and isinstance(right_data, QVariant)
            else left_data < right_data
        )

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """Filter rows based on the sample name."""
        index = self.sourceModel().index(source_row, 0, source_parent)
        # Only accept the first row of each sample
        _, tomogram_index = self.sourceModel()._get_data(index)
        if tomogram_index == 0:
            return True
        else:
            return False

    def filterAcceptsColumn(
        self, source_column: int, source_parent: QModelIndex
    ) -> bool:
        """Filter columns based on the sample name."""
        if source_column in [0, 1, 3]:
            return True
        else:
            return False


class TomogramModel(QSortFilterProxyModel):
    """
    Proxy model for displaying tomogram data in the CryoViT GUI application.

    This model provides a filtered view of the FileModel, allowing for tomogram-specific operations.
    """

    def __init__(self, source_model: QAbstractTableModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSourceModel(source_model)
        self.sample = None

    def setSample(self, sample: str) -> None:
        """Set the sample for the model."""
        self.beginResetModel()
        self.sample = sample
        self.endResetModel()

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Compare two indices for sorting."""
        if not left.isValid() or not right.isValid():
            return False
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.DisplayRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.DisplayRole)

        return (
            left_data.value() < right_data.value()
            if isinstance(left_data, QVariant) and isinstance(right_data, QVariant)
            else left_data < right_data
        )

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """Filter rows based on the sample name."""
        # Accept nothing if no sample is set
        if self.sample is None:
            return False
        # Only accept current sample
        index = self.sourceModel().index(source_row, 0, source_parent)
        sample, _ = self.sourceModel()._get_data(index)
        if sample == self.sample:
            return True
        else:
            return False

    def filterAcceptsColumn(
        self, source_column: int, source_parent: QModelIndex
    ) -> bool:
        """Filter columns based on the sample name."""
        if source_column == 2:  # Only Tomogram File column
            return True
        else:
            return False
