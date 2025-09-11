from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Self
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource, PyprojectTomlConfigSettingsSource
from inspect_wandb.config.wandb_settings_source import WandBSettingsSource
import os

class EnvironmentValidations(BaseModel):
    """
    A set of environment variables which should be validated before enabling the integration.
    """
    wandb_base_url: str | None = Field(default=None, description="The base URL of the wandb instance")
    wandb_api_key: str | None = Field(default=None, description="The API key for the wandb instance")

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
    project: str | None = Field(default=None, alias="WANDB_PROJECT", description="Project to write to for the Models integration")
    entity: str | None = Field(default=None, alias="WANDB_ENTITY", description="Entity to write to for the Models integration")
    config: dict[str, Any] | None = Field(default=None, description="Configuration to pass directly to wandb.config for the Models integration")
    files: list[str] | None = Field(default=None, description="Files to upload to the models run. Paths should be relative to the wandb directory.")
    viz: bool = Field(default=False, description="Whether to enable the inspect_viz extra")
    add_metadata_to_config: bool = Field(default=True, description="Whether to add eval metadata to wandb.config")

    tags: list[str] | None = Field(default=None, description="Tags to add to the models run")
    environment_validations: EnvironmentValidations | None = Field(default=None, description="Environment variables to validate before enabling")

    @field_validator("environment_validations", mode="after")
    @classmethod
    def validate_environment_variables(cls, v: EnvironmentValidations | None) -> EnvironmentValidations | None:
        if v is not None:
            if v.wandb_base_url is not None and (env_wandb_base_url := os.getenv("WANDB_BASE_URL")) != v.wandb_base_url:
                cls.enabled = False
                raise ValueError(f"WANDB_BASE_URL does not match the value in the environment. Validation URL: {v.wandb_base_url}, Environment URL: {env_wandb_base_url}")
            if v.wandb_api_key is not None and (env_wandb_api_key := os.getenv("WANDB_API_KEY")) != v.wandb_api_key:
                cls.enabled = False
                raise ValueError(f"WANDB_API_KEY does not match the value in the environment. Validation Key: {v.wandb_api_key}, Environment Key: {env_wandb_api_key}")
        return v
    
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
    project: str | None = Field(default=None, alias="WANDB_PROJECT", description="Project to write to for the Weave integration")
    entity: str | None = Field(default=None, alias="WANDB_ENTITY", description="Entity to write to for the Weave integration")

    autopatch: bool = Field(default=True, description="Whether to automatically patch Inspect with Weave calls for tracing")
    sample_name_template: str = Field(default="{task_name}-sample-{sample_id}-epoch-{epoch}", description="Template for sample display names. Available variables: {task_name}, {sample_id}, {epoch}")

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

class InspectWandBSettings(BaseModel):
    weave: WeaveSettings = Field(description="Settings for the Weave integration")
    models: ModelsSettings = Field(description="Settings for the Models integration")