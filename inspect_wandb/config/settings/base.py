import os
from logging import getLogger
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
)
from wandb.env import API_KEY as WANDB_API_KEY_ENV, BASE_URL as WANDB_BASE_URL_ENV
from wandb.sdk.lib.wbauth import read_netrc_auth

from inspect_wandb.config.wandb_settings_source import WandBSettingsSource

logger = getLogger(__name__)


class InspectWandBBaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        populate_by_name=True,
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow",
    )

    enabled: bool = Field(
        default=True, description="Whether to enable the InspectWandB integration"
    )
    project: str | None = Field(
        default=None,
        alias="WANDB_PROJECT",
        description="Project to write to for the wandb integrations",
    )
    entity: str | None = Field(
        default=None,
        alias="WANDB_ENTITY",
        description="Entity to write to for the wandb integrations",
    )

    @model_validator(mode="after")
    def validate_api_key(self) -> Self:
        DEFAULT_WANDB_BASE_URL = "https://api.wandb.ai"
        base_url = os.getenv(WANDB_BASE_URL_ENV, DEFAULT_WANDB_BASE_URL)
        if self.enabled and not (
            os.getenv(WANDB_API_KEY_ENV) or read_netrc_auth(host=base_url)
        ):
            logger.warning(
                "WandB integration disabled: no API key found. Log in with `wandb login` or set the WANDB_API_KEY environment variable."
            )
            self.enabled = False
        return self

    @model_validator(mode="after")
    def validate_project_and_entity(self) -> Self:
        if self.enabled and (not self.project or not self.entity):
            missing = []
            if not self.project:
                missing.append("project")
            if not self.entity:
                missing.append("entity")
            logger.warning(
                f"WandB integration disabled: missing required field(s): {', '.join(missing)}. Set via environment variables (WANDB_PROJECT, WANDB_ENTITY), wandb settings file, or pyproject.toml."
            )
            self.enabled = False
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
            PyprojectTomlConfigSettingsSource(settings_cls),
        )
