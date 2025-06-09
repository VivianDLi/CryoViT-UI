"""Delegate to handle custom editing widgets for configuration fields in the CryoViT GUI application."""

from functools import partial

from PyQt6.QtWidgets import (
    QStyledItemDelegate,
    QWidget,
    QStyleOptionViewItem,
    QModelIndex,
    QLineEdit,
    QSpinBox,
    QCheckBox,
    Qt,
)
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

from cryovit_gui.config import ConfigInputType, ConfigField
from cryovit_gui.models import ConfigModel
from cryovit_gui.gui import *

#### Logging Setup ####

import logging

from cryovit_gui.utils import select_file_folder_dialog

logger = logging.getLogger("cryovit.delegates.config")


class ConfigDelegate(QStyledItemDelegate):
    """
    Delegate for handling custom editing widgets for configuration fields in the CryoViT GUI application.

    This delegate provides a custom editor for configuration fields based on their input type.
    """

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ):
        """Create and return the appropriate editor widget for the given index."""
        config_field: ConfigField = index.data(Qt.ItemDataRole.UserRole).value()

        match config_field.input_type:
            case ConfigInputType.FILE:
                editor = ClickableLineEdit(parent=parent)
                editor.clicked.connect(
                    partial(
                        self._file_directory_prompt,
                        editor,
                        config_field.name,
                        False,
                    )
                )
            case ConfigInputType.DIRECTORY:
                editor = ClickableLineEdit(parent=parent)
                editor.clicked.connect(
                    partial(
                        self._file_directory_prompt,
                        editor,
                        config_field.name,
                        True,
                    )
                )
            case ConfigInputType.TEXT:
                editor = QLineEdit(parent=parent)
            case ConfigInputType.NUMBER:
                editor = QSpinBox(parent=parent)
                editor.setRange(0, 10000)
                editor.setSingleStep(1)
                editor.setKeyboardTracking(False)
            case ConfigInputType.BOOL:
                editor = QCheckBox(parent=parent)
            case ConfigInputType.STR_LIST | ConfigInputType.INT_LIST:
                editor = QLineEdit(parent=parent)
                if config_field.input_type == ConfigInputType.INT_LIST:
                    # Add a validator to ensure only integers are entered
                    reg = QRegularExpression(r"^(\d+,?)+$")
                    editor.setValidator(QRegularExpressionValidator(reg, parent))
            case _:
                logger.warning(
                    f"Unsupported input type: {config_field.input_type} for {config_field.name}. Ignoring this field."
                )

        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        """Set the data for the editor widget based on the model index."""
        config_field: ConfigField = index.data(Qt.ItemDataRole.UserRole).value()

        if isinstance(editor, ClickableLineEdit) or isinstance(editor, QLineEdit):
            editor.setText(config_field.get_value_as_str())
        elif isinstance(editor, QSpinBox):
            editor.setValue(config_field.get_value())
        elif isinstance(editor, QCheckBox):
            editor.setChecked(config_field.get_value())
        else:
            logger.warning(
                f"Unsupported editor type: {type(editor)} for {config_field.name}. Ignoring this field."
            )
            return

    def setModelData(self, editor: QWidget, model: ConfigModel, index: QModelIndex):
        """Update the model with the data from the editor widget."""
        config_field: ConfigField = index.data(Qt.ItemDataRole.UserRole).value()

        if isinstance(editor, ClickableLineEdit) or isinstance(editor, QLineEdit):
            config_field.set_value(editor.text())
        elif isinstance(editor, QSpinBox):
            config_field.set_value(editor.value())
        elif isinstance(editor, QCheckBox):
            config_field.set_value(editor.isChecked())
        else:
            logger.warning(
                f"Unsupported editor type: {type(editor)} for {config_field.name}. Ignoring this field."
            )
            return

        # Notify the model that the data has changed
        model.dataChanged.emit(index, index)

    def _file_directory_prompt(
        self,
        data: ClickableLineEdit | QLineEdit,
        name: str,
        is_folder: bool,
    ):
        """Open a file or directory selection dialog and set the selected path to the data widget."""
        start_dir = data.text() if data.text() else ""
        selected_path = select_file_folder_dialog(
            self, f"Select {name}", is_folder, False, start_dir=start_dir
        )
        data.setText(selected_path)
