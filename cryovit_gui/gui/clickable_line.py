"""UI widget for a clickable text line."""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLineEdit


class ClickableLineEdit(QLineEdit):
    """Custom QLineEdit that emits a signal when clicked and can be edited when double-clicked."""

    clicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)

    def mousePressEvent(self, event):
        """Override mousePressEvent to emit the clicked signal."""
        self.last = "Click"

    def mouseDoubleClickEvent(self, a0):
        self.last = "Double Click"

    def mouseReleaseEvent(self, event):
        """Override mouseReleaseEvent to emit the clicked signal after a timer on single click, and edit text on double-click."""
        if self.last == "Click":
            QTimer.singleShot(
                QApplication.instance().doubleClickInterval(), self.singleClickAction
            )
        else:
            self.setReadOnly(False)
            self.setCursorPosition(len(self.text()))

    def singleClickAction(self):
        """Action to perform on single click."""
        if self.last == "Click" and self.isReadOnly():
            self.clicked.emit()

    ## Set back to read-only after finishing editing (i.e., press Enter or lose focus)
    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.setReadOnly(True)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.setReadOnly(True)
