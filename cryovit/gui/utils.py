"""Functions and classes for miscellaneous GUI utilities."""

import sys
from typing import List, Union

from PyQt6.QtCore import Qt, QObject, QEvent, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QStandardItem
from PyQt6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QStyledItemDelegate,
)


class EmittingStream(QObject):
    """Class to pipe stream output to PyQt widgets."""

    textWritten = pyqtSignal(str)

    def write(self, text):
        """Write text to the stream."""
        self.textWritten.emit(str(text))


class MultiSelectComboBox(QComboBox):
    """Custom QComboBox to allow multiple selections."""

    class CustomDelegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            """Override sizeHint to set the height of the items."""
            size = super().sizeHint(option, index)
            size.setHeight(24)
            return size

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)

        # Setup item views
        self.setItemDelegate(MultiSelectComboBox.CustomDelegate())

        self.popupOpened = False
        # Setup event handlers
        self.lineEdit().installEventFilter(self)
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event) -> None:
        """Override resizeEvent to also adjust display text."""
        self.updateText()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:
        """Override showEvent to also adjust display text."""
        self.updateText()
        super().showEvent(event)

    def eventFilter(self, obj, event) -> bool:
        """Override events to handle mouse button release events and disable closing on click."""
        # Click on the display line
        if obj == self.lineEdit() and event.type() == QEvent.Type.MouseButtonPress:
            if not self.popupOpened:
                self.showPopup()
            else:
                self.hidePopup()
            return True
        # Select items on the popup menu
        if (
            obj == self.view().viewport()
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            index = self.view().indexAt(event.position().toPoint())
            item = self.model().itemFromIndex(index)
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckStaete(Qt.CheckState.Unchecked)
            else:
                item.setCheckState(Qt.CheckState.Checked)
            return True
        # Remove items from the popup menu
        if (
            obj == self.view().viewport()
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.RightButton
        ):
            index = self.view().indexAt(event.position().toPoint())
            self.removeItem(index)
            return True
        return False

    def updateText(self) -> None:
        texts = self.getCurrentData()
        if texts:
            text = ", ".join(texts)
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(
            text, Qt.TextElideMode.ElideRight, self.lineEdit().width()
        )
        self.lineEdit().setText(elidedText)

    def addItem(self, text: str) -> None:
        """Add an item to the combo box."""
        item = QStandardItem(text)
        item.setText(text)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts: List[str]) -> None:
        """Add multiple items to the combo box."""
        for text in texts:
            self.addItem(text)

    def addNewItem(self) -> None:
        """Ask the user for a new item and add it to the combo box."""
        try:
            new_item, ok = QInputDialog.getText(
                self,
                "Add New Sample",
                "Enter new sample name:",
                QLineEdit.EchoMode.Normal,
            )
            if ok and new_item:
                self.addItem(new_item)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error adding new item: {e}\n")

    def getCurrentIndices(self) -> List[int]:
        """Get the currently selected indices."""
        return [
            i
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.CheckState.Checked
        ]

    def setCurrentIndexes(self, indexes: list) -> None:
        """Set the selected items based on the provided indexes.

        Args:
            indexes (list): A list of indexes to select.
        """
        for i in range(self.model().rowCount()):
            self.model().item(i).setCheckState(
                Qt.CheckState.Checked if i in indexes else Qt.CheckState.Unchecked
            )
        self.updateText()

    def getCurrentData(self) -> List[str]:
        """Get the currently selected options as strings."""
        return [
            self.model().item(i).text()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.CheckState.Checked
        ]

    def showPopup(self) -> None:
        """Show the popup menu."""
        super().showPopup()
        self.popupOpened = True

    def hidePopup(self) -> None:
        """Hide the popup menu."""
        super().hidePopup()
        self.popupOpened = False


def select_file_folder_dialog(
    parent,
    title: str,
    is_folder: bool,
    is_multiple: bool = False,
    file_types: Union[str, List[str]] = [],
    start_dir: str = "",
) -> Union[str, List[str]]:
    dialog = QFileDialog(parent, caption=title)
    dialog.setFileMode(
        QFileDialog.FileMode.Directory
        if is_folder
        else (
            QFileDialog.FileMode.ExistingFile
            if not is_multiple
            else QFileDialog.FileMode.ExistingFiles
        )
    )
    if file_types:
        if isinstance(file_types, str):
            file_types = [file_types]
        dialog.setNameFilters(file_types)
    if start_dir:
        dialog.setDirectory(start_dir)

    if dialog.exec():
        if is_folder:
            return dialog.selectedFiles()[0]
        else:
            return dialog.selectedFiles() if dialog.selectedFiles() else None
