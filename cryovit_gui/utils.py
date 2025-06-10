"""Functions and classes for miscellaneous GUI utilities."""

from typing import List, Union

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog

from cryovit_gui.config import Colors

#### Logging Setup ####
import logging

logger = logging.getLogger("cryovit.utils")


class TextEditLogger(logging.Handler, QObject):
    """Custom logging handler to emit log messages to a QTextEdit."""

    appendLog = pyqtSignal(str)

    def __init__(self, console):
        super().__init__()
        QObject.__init__(self)
        self.console = console
        self.appendLog.connect(self.log_to_console)

    def log_to_console(self, msg: str, level: str):
        """Append a log message to the QTextEdit console."""
        # Ignore newlines but print whitespaces (no timestamp)
        if not msg.strip():
            self.console.insertPlainText(msg)
            return
        # Format text with colors
        match level:
            case "error":
                full_text = '<font color="rgb{}">{}</font>'.format(
                    Colors.RED.value, full_text
                )
            case "warning":
                full_text = '<font color="rgb{}">{}</font>'.format(
                    Colors.ORANGE.value, full_text
                )
            case "success":
                full_text = '<font color="rgb{}">{}</font>'.format(
                    Colors.GREEN.value, full_text
                )
            case "debug":
                full_text = '<font color="rgb{}">{}</font>'.format(
                    Colors.BLUE.value, full_text
                )
            case _:
                full_text = '<font color="rgb{}">{}</font>'.format(
                    Colors.WHITE.value, full_text
                )
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

    def emit(self, record):
        try:
            msg = self.format(record)
            self.appendLog.emit(msg, record.levelname.lower())
        except Exception:
            self.handleError(record)


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
    logger.info(f"{title}")
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
        if not is_multiple:
            return dialog.selectedFiles()[0]
        else:
            return dialog.selectedFiles() or None
