import configparser
import os
from pathlib import Path
from typing import Any
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings
from pydantic_settings.sources import PydanticBaseSettingsSource
import logging

logger = logging.getLogger(__name__)


def wandb_dir() -> str:
    """Return the directory wandb uses for local run data and its settings file.

    Reimplements wandb's own resolution without importing the private
    ``wandb.old.core`` module, which was removed in wandb 0.27.1. The root is
    ``$WANDB_DIR`` if set, otherwise the current working directory, and the
    hidden ``.wandb`` subdirectory is preferred over ``wandb`` when it already
    exists (mirroring ``wandb.sdk.wandb_settings``). Only the settings-file
    lookup is needed here, so the original's writability fallback to a temp
    directory is intentionally omitted.
    """
    root = os.environ.get("WANDB_DIR") or os.getcwd()
    dirname = ".wandb" if os.path.isdir(os.path.join(root, ".wandb")) else "wandb"
    return os.path.join(root, dirname)


class WandBSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._wandb_settings: dict[str, str] | None = None

    def _load_wandb_settings(self) -> dict[str, str]:
        if self._wandb_settings is not None:
            return self._wandb_settings

        settings_path = Path(wandb_dir()) / "settings"

        if not settings_path.exists():
            logger.debug("Wandb settings file not found, skipping WandBSettingsSource")
            self._wandb_settings = {}
            return self._wandb_settings

        try:
            with open(settings_path, "r") as f:
                parser = configparser.ConfigParser()
                parser.read_file(f)

            if "default" not in parser:
                logger.warning("No 'default' section found in wandb settings file")
                self._wandb_settings = {}
                return self._wandb_settings

            default_section = parser["default"]
            self._wandb_settings = {
                "entity": default_section.get("entity", ""),
                "project": default_section.get("project", ""),
            }

            logger.debug(
                f"Loaded wandb settings: entity={self._wandb_settings.get('entity')}, project={self._wandb_settings.get('project')}"
            )

        except Exception as e:
            logger.warning(f"Failed to read wandb settings file: {e}")
            self._wandb_settings = {}

        return self._wandb_settings

    def get_field_value(
        self, field_info: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        if field_info.alias not in ("WANDB_PROJECT", "WANDB_ENTITY"):
            return None, "", False

        wandb_settings = self._load_wandb_settings()

        if field_info.alias == "WANDB_PROJECT":
            value = wandb_settings.get("project")
        elif field_info.alias == "WANDB_ENTITY":
            value = wandb_settings.get("entity")
        else:
            return None, "", False

        if value:
            return value, f"wandb settings file ({field_info.alias})", False

        return None, "", False

    def __call__(self) -> dict[str, Any]:
        d: dict[str, Any] = {}

        wandb_settings = self._load_wandb_settings()
        if not wandb_settings:
            return d

        if wandb_settings.get("project"):
            d["project"] = wandb_settings["project"]
        if wandb_settings.get("entity"):
            d["entity"] = wandb_settings["entity"]

        return d
