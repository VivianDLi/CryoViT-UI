"""Define a base item model supporting ConfigGroups for the CryoViT GUI application."""

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, QVariant, Qt
from PyQt6.QtGui import QFont

from cryovit_gui.config import ConfigGroup, ConfigField, ConfigKey

#### Logging Setup ####

import logging

logger = logging.getLogger("cryovit.models.config")


class ConfigModel(QAbstractItemModel):
    """
    Base class for storing configurations in the CryoViT GUI application.

    This class provides a common interface for item models initialized from a ConfigGroup dataclass, including methods
    for getting data, setting data, and handling model-specific logic.
    """

    def __init__(self, config: ConfigGroup, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return the item flags for the given index."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        target_config = index.internalPointer()
        if isinstance(target_config, ConfigGroup):
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsDragEnabled
        elif isinstance(target_config, ConfigField):
            if index.column() == 0:
                return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemNeverHasChildren
            return (
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemNeverHasChildren
            )
        # Default case for other types of items
        else:
            return Qt.ItemFlag.NoItemFlags

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows for the given parent index."""
        if not parent.isValid():
            # If no parent, return the number of top-level items
            return len(self._config.get_fields(recursive=False))
        parent_config = parent.internalPointer()
        return (
            len(parent_config.get_fields(recursive=False))
            if isinstance(parent_config, ConfigGroup)
            else 0
        )

    def hasChildren(self, parent: QModelIndex = QModelIndex()) -> bool:
        """Return whether the given parent index has children."""
        if not parent.isValid():
            # If no parent, _check if the root config has fields
            return len(self._config.get_fields(recursive=False)) > 0
        parent_config = parent.internalPointer()
        if isinstance(parent_config, ConfigGroup):
            return len(parent_config.get_fields(recursive=False)) > 0
        return False

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns for the given parent index."""
        return 2

    def index(
        self, row: int, column: int, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Return the index for the given row, column, and parent index."""
        if not parent.isValid():
            # If no parent, return the index for the root item
            config_key = self._config.get_fields(recursive=False)[row]
            config = self._config.get_field(config_key)
            return self.createIndex(row, column, config)
        parent_config = parent.internalPointer()
        if not isinstance(parent_config, ConfigGroup):
            logger.warning(
                f"Cannot create index for non-group parent at {parent}: {type(parent_config)}"
            )
            return QModelIndex()
        if (
            row < 0
            or column < 0
            or row >= self.rowCount(parent)
            or column >= self.columnCount(parent)
        ):
            logger.warning(
                f"Invalid row {row} or column {column} for parent {parent_config.name}."
            )
            return QModelIndex()
        config_key = parent_config.get_fields(recursive=False)[row]
        config = parent_config.get_field(config_key)
        if config is not None:
            return self.createIndex(row, column, config)
        else:
            logger.warning(
                f"Config at row {row}, column {column} for parent {parent_config.name} not found."
            )
            return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Return the parent index for the given index."""
        if not index.isValid():
            return QModelIndex()
        target_config = index.internalPointer()
        if not isinstance(target_config, (ConfigField, ConfigGroup)):
            logger.warning(
                f"Parent requested for non-config item at index {index}: {type(target_config)}"
            )
            return QModelIndex()
        parent_config = target_config.get_parent()
        if parent_config is None:
            # If the target config has no parent, it is the root item
            return QModelIndex()
        parent_keys = parent_config.get_fields(recursive=False)
        parent_fields = [parent_config.get_field(key) for key in parent_keys]
        if target_config not in parent_fields:
            logger.warning(
                f"Target config {target_config.name} not found in parent group {parent_config.name}."
            )
            return QModelIndex()
        row = parent_fields.index(target_config)
        return self.createIndex(row, 0, parent_config)

    def data(self, index: QModelIndex, role: int) -> QVariant:
        """Return the data for the given index and role."""
        if not index.isValid():
            return QVariant()
        target_config = index.internalPointer()
        if isinstance(target_config, ConfigGroup) and index.column() == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                # Return the name of the configuration group
                return QVariant(target_config.name)
            elif role == Qt.ItemDataRole.FontRole:
                return QFont("Segoe UI", 12, QFont.Weight.Bold)
        elif isinstance(target_config, ConfigField):
            match role:
                case Qt.ItemDataRole.DisplayRole:
                    if index.column() == 0:
                        # Return the name of the configuration field
                        return QVariant(target_config.name)
                    # Return display data as either a string
                    return QVariant(target_config.get_value_as_str())
                case Qt.ItemDataRole.EditRole:
                    # Return editable data for delegate
                    return QVariant(target_config.get_value())
                case Qt.ItemDataRole.ToolTipRole:
                    if index.column() == 0:
                        # Return tooltip for the configuration field name
                        return QVariant(target_config.description)
                case Qt.ItemDataRole.UserRole:
                    # Return the ConfigField itself for editing
                    return QVariant(target_config)
                case _:
                    return QVariant()
        else:
            return QVariant()

    def setData(
        self, index: QModelIndex, value: QVariant, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        """Set the data for the given index and role."""
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        target_config = index.internalPointer()
        # Set data for the index
        if not isinstance(target_config, ConfigField):
            logger.warning(
                f"Cannot set data for non-field config at index {index}: {type(target_config)}"
            )
            return False
        target_config.set_value(value)
        self.dataChanged.emit(index, index)  # Notify that data has changed
        return True

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> QVariant:
        """Return the header data for the given section and orientation."""
        # No headers
        return QVariant()

    def setHeaderData(
        self,
        data: QVariant,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        """Set the header data for the given section and orientation."""
        return False

    def get_config(self, key: ConfigKey) -> ConfigField:
        """
        Retrieve a specific setting by its key.

        Args:
            key (ConfigKey): The key of the setting to retrieve.

        Returns:
            ConfigField: The configuration field associated with the key.
        """
        return self._config.get_field(key)

    def _set_config(
        self, key: ConfigKey, value: str, parent: QModelIndex = None
    ) -> None:
        if parent is None or not parent.isValid():
            parent_config = self._config
            parent = QModelIndex()
        else:
            parent_config = parent.internalPointer()
        cur_key = key.pop_first()
        fields = parent_config.get_fields(recursive=False)
        if (not cur_key) or (not cur_key in fields):
            logger.warning(
                f"Setting {cur_key} not found in parent {parent_config.name}."
            )
            return
        config_field = parent_config.get_field(cur_key)
        if isinstance(config_field, ConfigField):
            config_field.set_value(value)
            index = self.index(fields.index(cur_key), 1, parent)
            self.dataChanged.emit(index, index)
        elif isinstance(config_field, ConfigGroup):
            # If the config field is a group, recursively set the setting
            self._set_config(key, value, self.index(fields.index(cur_key), 0, parent))
        else:
            logger.warning(
                f"Setting {key} is not a ConfigField or ConfigGroup in parent {parent_config.name}."
            )

    def set_config(self, key: ConfigKey, value: str) -> None:
        """
        Set a specific setting by its key.

        Args:
            key (ConfigKey): The key of the setting to set.
            value (str): The value to set for the configuration field.
        """
        self._set_config(key, value)

    def generate_commands(self) -> list[str]:
        """
        Generate a list of commands based on the current configuration.

        Returns:
            list[str]: A list of command strings representing the current configuration.
        """
        return self._config.generate_commands()
