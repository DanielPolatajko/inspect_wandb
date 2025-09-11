from pydantic import BaseModel, Field, field_validator
from typing import Any
from pydantic_settings import SettingsConfigDict
from inspect_wandb.config.settings.base import InspectWandBBaseSettings
import os


class EnvironmentValidations(BaseModel):
    """
    A set of environment variables which should be validated before enabling the integration.
    """
    wandb_base_url: str | None = Field(default=None, description="The base URL of the wandb instance")
    wandb_api_key: str | None = Field(default=None, description="The API key for the wandb instance")

class ModelsSettings(InspectWandBBaseSettings):
    """
    Settings model for the Models integration.
    """

    model_config = SettingsConfigDict(
        env_prefix="INSPECT_WANDB_MODELS_", 
        pyproject_toml_table_header=("tool", "inspect-wandb", "models"),
    )

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