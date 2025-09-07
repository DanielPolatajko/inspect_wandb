from pydantic import BaseModel, Field
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource, PyprojectTomlConfigSettingsSource
from inspect_wandb.config.wandb_settings_source import WandBSettingsSource

class ModelsSettings(BaseSettings):
    """
    Settings model for the Models integration.
    """

    model_config = SettingsConfigDict(
        env_prefix="INSPECT_WANDB_MODELS_", 
        pyproject_toml_table_header=("tool", "inspect-wandb", "models"),
        populate_by_name=True,
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow"
    )

    enabled: bool = Field(default=True, description="Whether to enable the Models integration")
    project: str = Field(alias="WANDB_PROJECT", description="Project to write to for the Models integration")
    entity: str = Field(alias="WANDB_ENTITY", description="Entity to write to for the Models integration")
    config: dict[str, Any] | None = Field(default=None, description="Configuration to pass directly to wandb.config for the Models integration")
    files: list[str] | None = Field(default=None, description="Files to upload to the models run. Paths should be relative to the wandb directory.")
    viz: bool = Field(default=False, description="Whether to enable the inspect_viz extra")
    add_metadata_to_config: bool = Field(default=True, description="Whether to add eval metadata to wandb.config")

    tags: list[str] | None = Field(default=None, description="Tags to add to the models run")

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

class WeaveSettings(BaseSettings):
    """
    Settings model for the Weave integration.
    """

    model_config = SettingsConfigDict(
        env_prefix="INSPECT_WANDB_WEAVE_", 
        pyproject_toml_table_header=("tool", "inspect-wandb", "weave"),
        populate_by_name=True,
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow"
    )
    
    enabled: bool = Field(default=True, description="Whether to enable the Weave integration")
    project: str = Field(alias="WANDB_PROJECT", description="Project to write to for the Weave integration")
    entity: str = Field(alias="WANDB_ENTITY", description="Entity to write to for the Weave integration")

    autopatch: bool = Field(default=True, description="Whether to automatically patch Inspect with Weave calls for tracing")
    sample_name_template: str = Field(default="{task_name}-sample-{sample_id}-epoch-{epoch}", description="Template for sample display names. Available variables: {task_name}, {sample_id}, {epoch}")

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

class InspectWandBSettings(BaseModel):
    weave: WeaveSettings = Field(description="Settings for the Weave integration")
    models: ModelsSettings = Field(description="Settings for the Models integration")