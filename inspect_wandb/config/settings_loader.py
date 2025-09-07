from inspect_wandb.config.settings import WeaveSettings, ModelsSettings, InspectWandBSettings
from logging import getLogger
from typing import Any
from inspect_wandb.exceptions import InvalidSettingsError

logger = getLogger(__name__)

class SettingsLoader:

    @classmethod
    def load_inspect_wandb_settings(cls, settings: dict[str, Any] | None = None) -> InspectWandBSettings:
        """
        Load settings with this priority:
        1. Initial settings (programmatic overrides provided to the settings argument)
        2. Environment variables (both WANDB_* vars defined by W&B, and INSPECT_WANDB_* vars defined by this package)
        3. Wandb settings file (for entity/project - handled by WandBSettingsSource)
        4. Pyproject.toml customizations
        5. Defaults if no other source provides values
        
        Note: The WandBSettingsSource will automatically read the wandb settings file.
        If no wandb settings are found, the settings creation will fail with validation errors
        for missing entity/project unless they are provided via environment variables.
        """
        if settings is None:
            settings = {"weave": {}, "models": {}}
        else:
            if "weave" not in settings or "models" not in settings:
                raise InvalidSettingsError()
        # Simply create the settings - the sources are defined as part of the pydantic settings model
        return InspectWandBSettings(
            weave=WeaveSettings.model_validate(settings["weave"]),
            models=ModelsSettings.model_validate(settings["models"])
        )