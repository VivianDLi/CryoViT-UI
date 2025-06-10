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

    def __init__(self, root_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._root_dir = root_dir
        self._required_subdirs = ["tomograms", "csv", "slices"]
        self._data: FileData = {}
        self._colors = [
            QColor(*Colors.RED),
            QColor(*Colors.DARK_ORANGE),
            QColor(*Colors.ORANGE),
            QColor(*Colors.YELLOW),
        ]
        if self.validate_root():
            data = self.read_data()
            self.update_data(data, update_selection=True)

    def validate_root(self) -> bool:
        """Validate the root directory, returning True if the root directory has the correct structure (i.e., contains tomograms, csv, and slices subfolders), and False if not."""
        if not self._root_dir.exists() or not self._root_dir.is_dir():
            logger.warning(
                f"Root directory {self._root_dir} does not exist or is not a directory."
            )
            return False
        for subdir in self._required_subdirs:
            if not (self._root_dir / subdir).exists():
                logger.warning(
                    f"Required subdirectory {subdir} does not exist in root directory {self._root_dir}."
                )
                return False
            setattr(self, f"_{subdir}_dir", self._root_dir / subdir)
        return True

    def read_data(self) -> FileData:
        """Get model data based on the current root directory."""
        if not self.validate_data():
            return {}
        # Get available tomogram files
        samples = [
            sample_dir.name
            for sample_dir in self._root_dir.iterdir()
            if sample.is_dir()
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
            if update_selection:
                selected = annotated.copy()
            else:
                selected = (
                    self._data[sample].selected
                    if sample in self._data
                    else annotated.copy()
                )
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

    def update_data(data: FileData, update_selection: bool = False) -> None:
        """Update the model data with the given data."""
        for sample in data:
            if update_selection:
                data[sample].selected = data[sample].annotated.copy()
            else:
                data[sample].selected = (
                    self._data[sample].selected.copy()
                    if sample in self._data
                    else data[sample].annotated.copy()
                )
        self._data = data
        self.dataChanged.emit(
            self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1)
        )

    def set_root(self, root_dir: Path) -> None:
        """Set the root directory and update the model data."""
        self._root_dir = root_dir
        if self.validate_root():
            logger.info(f"Setting root directory to {self._root_dir}")
            data = self.read_data()
            self.update_data(data, update_selection=True)

    def get_directory(self, subdir: str) -> Path:
        """Get the directory for the given subdirectory."""
        if not self.validate_root():
            logger.warning(
                f"Root directory {self._root_dir} is not valid. Cannot get subdirectory {subdir}."
            )
            return None
        if subdir not in self._required_subdirs:
            logger.warning(f"Subdirectory {subdir} is not a required subdirectory.")
            return None
        return getattr(self, f"_{subdir}_dir", None)

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
        rows = [len(self._data[sample].tomogram_files) for sample in self._data]
        return sum(rows) if rows else 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns for the given parent index."""
        return 4  # Columns: Sample Name, Percentage, File Name, Exported

    def data(self, index: QModelIndex, role: int) -> QVariant | QBrush | Qt.CheckState:
        """Return the data for the given index and role."""
        sample, tomogram_index = self._get_data(index)
        samples_annotated = sum(self._data[sample].annotated)
        total_samples = len(self._data[sample].tomogram_files)
        percentage = round((samples_annotated / total_samples) * 100, 2)
        background_color = (
            self._colors[percentage // (100 // len(self._colors))]
            if percentage < 100
            else QColor(*Colors.GREEN)
        )
        match index.column():
            case 0:  # Sample Name
                match role:
                    case Qt.ItemDataRole.DisplayRole:
                        return QVariant(
                            f"{sample} ({samples_annotated}/{total_samples})"
                        )
                    case Qt.ItemDataRole.BackgroundRole:
                        return QBrush(
                            background_color, style=Qt.BrushStyle.Dense6Pattern
                        )
                    case _:
                        return QVariant()
            case 1:  # Percentage
                match role:
                    case Qt.ItemDataRole.DisplayRole:
                        return QVariant(percentage)
                    case Qt.ItemDataRole.BackgroundRole:
                        return QBrush(
                            background_color, style=Qt.BrushStyle.Dense6Pattern
                        )
                    case _:
                        return QVariant()
            case 2:  # Tomogram File
                match role:
                    case Qt.ItemDataRole.DisplayRole:
                        return QVariant(
                            self._data[sample].tomogram_files[tomogram_index].name
                        )
                    case Qt.ItemDataRole.BackgroundRole:
                        if self._data[sample].annotated[tomogram_index]:
                            return QBrush(
                                QColor(*Colors.GREEN), style=Qt.BrushStyle.Dense6Pattern
                            )
                    case Qt.ItemDataRole.CheckStateRole:
                        return (
                            Qt.CheckState.Checked
                            if self._data[sample].selected[tomogram_index]
                            else Qt.CheckState.Unchecked
                        )
                    case _:
                        return QVariant()
            case 3:  # Exported
                match role:
                    case Qt.ItemDataRole.DecorationRole:
                        return (
                            QBrush(QColor(*Colors.GREEN))
                            if self._data[sample].exported[tomogram_index]
                            else QBrush(QColor(*Colors.RED))
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
        if value == Qt.CheckState.Checked:
            self._data[sample].selected[tomogram_index] = True
        else:
            self._data[sample].selected[tomogram_index] = False

        self.dataChanged.emit(index, index)
        return True

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
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
        for sample in self._data:
            if cur_rows < len(self._data[sample].tomogram_files):
                return sample, cur_rows
            cur_rows -= len(self._data[sample].tomogram_files)
        logger.warning(f"Index {index} is out of bounds for the current data model.")
        return "", -1


## Proxy Models ##


class SampleModel(QSortFilterProxyModel):
    """
    Proxy model for displaying sample data in the CryoViT GUI application.

    This model provides a filtered view of the FileModel, allowing for sample-specific operations.
    """

    def __init__(self, source_model: FileModel, *args, **kwargs):
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

    def __init__(self, source_model: FileModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSourceModel(source_model)
        self.sample = None

    def setSample(self, sample: str) -> None:
        """Set the sample for the model."""
        self.sample = sample

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
