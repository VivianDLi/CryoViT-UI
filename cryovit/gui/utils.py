"""Functions and classes for miscellaneous GUI utilities."""

from typing import List, Union

from PyQt6.QtCore import Qt, QObject, QEvent, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QStandardItem, QMouseEvent
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

        # event handlers
        self.view().viewport().installEventFilter(self)
        self.lineEdit().installEventFilter(self)
        self.model().dataChanged.connect(self.updateText)

    def resizeEvent(self, event) -> None:
        """Override resizeEvent to also adjust display text."""
        super().resizeEvent(event)
        self.updateText()

    def showEvent(self, event) -> None:
        """Override showEvent to also adjust display text."""
        super().showEvent(event)
        self.updateText()

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """Override eventFilter to handle mouse right-click events (ignored by QComboBox), selection, and open dropdown."""
        if (
            event.type() == QEvent.Type.MouseButtonRelease
            and source == self.view().viewport()
        ):
            mouse_event: QMouseEvent = event
            # Select items from the popup menu
            if mouse_event.button() == Qt.MouseButton.LeftButton:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())
                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
                return True
            # Remove items from the popup menu
            if mouse_event.button() == Qt.MouseButton.RightButton:
                index = self.view().indexAt(event.pos())
                self.model().removeRows(index.row(), 1)
                return True
            else:
                return False
        if event.type() == QEvent.Type.MouseButtonRelease and source == self.lineEdit():
            mouse_event: QMouseEvent = event
            if self.popupOpened:
                self.hidePopup()
            else:
                self.showPopup()
            return True
        return False

    def updateText(self) -> None:
        """Update the displayed text in the combo box to display the selected items separated by commas."""
        texts = self.getCurrentData()
        if texts:
            text = ", ".join(texts)
            metrics = QFontMetrics(self.lineEdit().font())
            elidedText = metrics.elidedText(
                text, Qt.TextElideMode.ElideRight, self.lineEdit().width()
            )
            self.lineEdit().setText(elidedText)
        else:
            self.lineEdit().setText("")

    def addItem(self, text: str) -> None:
        """Add an item to the combo box.

        Args:
            text (str): The string to add."""
        item = QStandardItem(text)
        item.setText(text)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts: List[str]) -> None:
        """Add multiple items to the combo box.

        Args:
            texts (list): A list of strings to add."""
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

    def setCurrentData(self, data: List[str]) -> None:
        """Set the selected items based on the provided data.

        Args:
            data (list): A list of strings to select.
        """
        available_data = [
            self.model().item(i).text() for i in range(self.model().rowCount())
        ]
        indices = []
        for d in data:
            if d not in available_data:
                self.addItem(d)
                indices.append(self.model().rowCount() - 1)
            else:
                indices.append(available_data.index(d))
        self.setCurrentIndexes(indices)

    def showPopup(self) -> None:
        """Show the popup menu."""
        super().showPopup()
        self.popupOpened = True

    def hidePopup(self) -> None:
        """Hide the popup menu."""
        super().hidePopup()
        self.startTimer(100)  # add cooldown to prevent double open and spam

    def timerEvent(self, event) -> None:
        """Timer event handler to close the popup menu without re-opening it."""
        self.killTimer(event.timerId())
        self.popupOpened = False


def select_file_folder_dialog(
    parent,
    title: str,
    is_folder: bool,
    is_multiple: bool = False,
    file_types: Union[str, List[str]] = [],
    start_dir: str = "",
) -> Union[str, List[str]]:
    """Opens a file/folder selection dialog and returns the selected path(s).

    Args:
        parent (_type_): Parent Qt widget for the dialog.
        title (str): Title for the dialog window.
        is_folder (bool): Whether to select folders (True) or files (False).
        is_multiple (bool, optional): Enables multiple files/folders to be selected. Defaults to False.
        file_types (Union[str, List[str]], optional): Optionally filters available files to those with specified extensions. Defaults to [].
        start_dir (str, optional): Directory to start the selection from. Defaults to "".

    Returns:
        Union[str, List[str]]: Paths to the selected files/folders.
    """
    parent.log("info", f"{title}")
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
