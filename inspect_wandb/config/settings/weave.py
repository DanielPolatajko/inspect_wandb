

from inspect_wandb.config.settings.base import InspectWandBBaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

class WeaveSettings(InspectWandBBaseSettings):
    """
    Settings model for the Weave integration.
    """

    model_config = SettingsConfigDict(
        env_prefix="INSPECT_WANDB_WEAVE_", 
        pyproject_toml_table_header=("tool", "inspect-wandb", "weave"),
    )

    eval_traces_only: bool = Field(default=False, description="When True, only eval-level summary logging is performed and sample-level Weave traces are disabled.")
    sample_name_template: str = Field(default="{task_name}-sample-{sample_id}-epoch-{epoch}", description="Template for sample display names. Available variables: {task_name}, {sample_id}, {epoch}")