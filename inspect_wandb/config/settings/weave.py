from inspect_wandb.config.settings.base import InspectWandBBaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class WeaveSettings(InspectWandBBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INSPECT_WANDB_WEAVE_",
        pyproject_toml_table_header=("tool", "inspect-wandb", "weave"),
    )

    eval_traces_only: bool = Field(
        default=False,
        description="When True, only eval-level summary logging is performed and sample-level Weave traces are disabled.",
    )
    agent_sessions: bool = Field(
        default=False,
        description="When True, stream each Inspect sample's agent trajectory to Weave's agent Session SDK (the Agents view) as turns complete, in addition to eval logging.",
    )
    agent_sessions_include_content: bool = Field(
        default=True,
        description="When agent_sessions is enabled, include full message and tool content on the streamed spans. Set False to log only structure (turns, tool names, tokens, timing, scores) for very long-horizon evals where per-turn transcript volume is prohibitive.",
    )
    sample_name_template: str = Field(
        default="{task_name}-sample-{sample_id}-epoch-{epoch}",
        description="Template for sample display names. Available variables: {task_name}, {sample_id}, {epoch}",
    )
