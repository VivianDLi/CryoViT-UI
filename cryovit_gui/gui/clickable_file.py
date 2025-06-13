"""UI widget for a clickable text line."""

from PyQt6.QtCore import Qt, QTimer, QObject, QEvent
from PyQt6.QtWidgets import QApplication, QLineEdit, QFileDialog


class ClickableFileSelect(QLineEdit):
    """Custom QLineEdit that opens a file dialog when clicked and can be edited when double-clicked."""

    def __init__(self, name: str, is_folder: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.is_folder = is_folder
        self.dialog = QFileDialog(self.parent(), caption=name)
        self.dialog.setFileMode(
            QFileDialog.FileMode.Directory
            if is_folder
            else QFileDialog.FileMode.ExistingFile
        )
        self.setReadOnly(True)

        self._should_close = True
        self.installEventFilter(self)

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """Override eventFilter to prevent editor deletion on dialog open."""
        if event.type() == QEvent.Type.Hide and source == self:
            # Prevent the editor from being deleted when focus is lost
            if not self._should_close:
                return True
        return False

    def mousePressEvent(self, event):
        """Override mousePressEvent to emit the clicked signal."""
        self.last = "Click"

    def mouseDoubleClickEvent(self, a0):
        self.last = "Double Click"

    def mouseReleaseEvent(self, event):
        """Override mouseReleaseEvent to emit the clicked signal after a timer on single click, and edit text on double-click."""
        if self.last == "Click":
            self._should_close = False
            QTimer.singleShot(
                QApplication.instance().doubleClickInterval(), self.singleClickAction
            )
        else:
            self.setReadOnly(False)
            self.setCursorPosition(len(self.text()))
            self._should_close = True

    def singleClickAction(self):
        """Action to perform on single click."""
        if self.last == "Click" and self.isReadOnly():
            if self.dialog.exec():
                self.setText(self.dialog.selectedFiles()[0])
            self._should_close = True

    ## Set back to read-only after finishing editing (i.e., press Enter or lose focus)
    def focusOutEvent(self, event):
        self.setReadOnly(True)
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.setReadOnly(True)
