from pydantic import Field, model_validator
from typing import Self
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource, PyprojectTomlConfigSettingsSource
from inspect_wandb.config.wandb_settings_source import WandBSettingsSource

class InspectWandBBaseSettings(BaseSettings):
    """
    Shared settings across Models and Weave integrations for the InspectWandB integration.
    """

    model_config = SettingsConfigDict(
        populate_by_name=True,
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow"
    )

    enabled: bool = Field(default=True, description="Whether to enable the InspectWandB integration")
    project: str | None = Field(default=None, alias="WANDB_PROJECT", description="Project to write to for the Weave integration")
    entity: str | None = Field(default=None, alias="WANDB_ENTITY", description="Entity to write to for the Weave integration")

    @model_validator(mode="after")
    def validate_project_and_entity(self) -> Self:
        if self.enabled and (not self.project or not self.entity):
            raise ValueError("Project and entity must be set if the Models integration is enabled")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,    
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customise the priority of settings sources to prioritise as follows:
        1. Initial settings (can be set via eval metadata fields)
        2. Environment variables (highest priority)
        3. Wandb settings file (for entity/project)
        4. Pyproject.toml (lowest priority)
        """
        return (
            init_settings,
            env_settings, 
            WandBSettingsSource(settings_cls),
            PyprojectTomlConfigSettingsSource(settings_cls)
        )