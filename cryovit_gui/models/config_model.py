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

    def get_config(self, index: QModelIndex) -> ConfigField | ConfigGroup:
        """Get the ConfigField or ConfigGroup at the given index."""
        if not index.isValid():
            return self._data
        parent_config = self.get_config(index.parent())
        target_config_key = parent_config.get_fields(recursive=False)[index.row()]
        target_config = parent_config.get_field(target_config_key)
        if target_config is None:
            logger.warning(f"Config at index {index} not found.")
            return None
        return target_config

    def get_config_by_key(self, key: ConfigKey) -> ConfigField | ConfigGroup:
        """Get the ConfigField or ConfigGroup by its key."""
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
        target_config = self.get_config(index)
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
        target_config = self.get_config(parent)
        return (
            len(target_config.get_fields(recursive=False))
            if isinstance(target_config, ConfigGroup)
            else 0
        )

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns for the given parent index."""
        target_config = self.get_config(parent)
        if isinstance(target_config, ConfigGroup):
            return 1
        elif isinstance(target_config, ConfigField):
            return 2
        else:
            return 0

    def data(self, index: QModelIndex, role: int) -> QVariant:
        """Return the data for the given index and role."""
        target_config = self.get_config(index)
        if not index.isValid() or target_config is None:
            return QVariant()
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
            logger.warning(
                f"Data requested for unknown config type at index {index}: {type(target_config)}"
            )
            return QVariant()

    def setData(
        self, index: QModelIndex, value: QVariant, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        """Set the data for the given index and role."""
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        target_config = self.get_config(index)
        # Set data for the index
        if not isinstance(target_config, ConfigField):
            logger.warning(
                f"Cannot set data for non-field config at index {index}: {type(target_config)}"
            )
            return False
        target_config.set_value(value.value())
        self.dataChanged.emit(index, index)  # Notify that data has changed
        return True

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> QVariant:
        """Return the header data for the given section and orientation."""
        # No headers
        return QVariant()

    def setHeaderData(
        self, data: QVariant, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> bool:
        """Set the header data for the given section and orientation."""
        return False
