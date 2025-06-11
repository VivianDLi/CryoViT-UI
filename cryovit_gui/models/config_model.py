"""Define a base item model supporting ConfigGroups for the CryoViT GUI application."""

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, QVariant, Qt

from cryovit_gui.config import *

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
        self._data = config

    def get_config_by_key(self, key: ConfigKey | None) -> ConfigField | ConfigGroup:
        """Get the ConfigField or ConfigGroup by its key."""
        if key is None:
            return self._data
        target_config = self._data.get_field(key)
        if target_config is None:
            logger.warning(f"Config with key {key} not found.")
            return None
        return target_config

    def generate_commands(self) -> List[str]:
        """Generate a list of commands based on the current configuration."""
        commands = []
        for key in self._data.get_fields(recursive=True):
            field = self._data.get_field(key)
            command_name = ".".join(key)
            command_value = field.get_value_as_str()
            commands.append(f"{command_name}={command_value}")
        return commands

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
            return len(self._data.get_fields(recursive=False))
        target_config = parent.internalPointer()
        return (
            len(target_config.get_fields(recursive=False))
            if isinstance(target_config, ConfigGroup)
            else 0
        )

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns for the given parent index."""
        if not parent.isValid():
            parent_config = self._data
        else:
            parent_config = parent.internalPointer()
        fields = parent_config.get_fields(recursive=False)
        has_field = any(
            [isinstance(parent_config.get_field(f), ConfigField) for f in fields]
        )
        if has_field:
            return 2
        else:
            return 1

    def index(
        self, row: int, column: int, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Return the index for the given row, column, and parent index."""
        if not parent.isValid():
            # If no parent, return the index for the root item
            if (
                row < 0
                or column < 0
                or row >= len(self._data.get_fields(recursive=False))
            ):
                logger.warning(f"Invalid row {row} or column {column} for root config.")
                return QModelIndex()
            config = self._data.get_fields(recursive=False)[row]
            return self.createIndex(row, column, self._data.get_field(config))
        parent_config = parent.internalPointer()
        if not isinstance(parent_config, ConfigGroup):
            logger.warning(
                f"Cannot create index for non-group parent at {parent}: {type(parent_config)}"
            )
            return QModelIndex()
        if (
            row < 0
            or column < 0
            or row >= len(parent_config.get_fields(recursive=False))
        ):
            logger.warning(
                f"Invalid row {row} or column {column} for parent {parent_config.name}."
            )
            return QModelIndex()
        config_key = parent_config.get_fields(recursive=False)[row]
        if config_key:
            return self.createIndex(row, column, parent_config.get_field(config_key))
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
        parent_config = target_config.parent
        if parent_config is None:
            # If the target config has no parent, it is the root item
            return QModelIndex()
        parent_fields = [
            parent_config.get_field(f)
            for f in parent_config.get_fields(recursive=False)
        ]
        parent_row = (
            parent_fields.index(target_config) if target_config in parent_fields else -1
        )
        if parent_row == -1:
            logger.warning(
                f"Parent row for {target_config.name} not found in its parent group."
            )
            return QModelIndex()
        return self.createIndex(parent_row, 0, parent_config)

    def data(self, index: QModelIndex, role: int) -> QVariant:
        """Return the data for the given index and role."""
        if not index.isValid():
            return QVariant()
        target_config = index.internalPointer()
        if (
            isinstance(target_config, ConfigGroup)
            and role == Qt.ItemDataRole.DisplayRole
        ):
            # Return the name of the configuration group
            return QVariant(target_config.name)
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
