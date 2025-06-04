"""Functions and classes for miscellaneous GUI utilities."""

from typing import List, Union
import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog

logger = logging.getLogger(__name__)


class EmittingStream(QObject):
    """Class to pipe stream output to PyQt widgets."""

    textWritten = pyqtSignal(str)

    def write(self, text):
        """Write text to the stream."""
        self.textWritten.emit(str(text))


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
        if is_folder:
            return dialog.selectedFiles()[0]
        else:
            return dialog.selectedFiles() if dialog.selectedFiles() else None
