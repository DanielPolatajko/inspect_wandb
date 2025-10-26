import importlib
from weave.integrations.patcher import SymbolPatcher, MultiPatcher
from weave.trace.autopatch import AutopatchSettings, IntegrationSettings
from pydantic import Field
from inspect_wandb.weave.autopatcher.inspect import patched_task_run_sample

inspect_patcher = MultiPatcher(
    [
        SymbolPatcher(
            lambda: importlib.import_module("inspect_ai._eval.task.run"),
            "task_run_sample",
            lambda *_, **__: patched_task_run_sample,
        ),
    ]
)

def get_inspect_patcher(settings: IntegrationSettings | None = None) -> MultiPatcher:
    return inspect_patcher

class CustomAutopatchSettings(AutopatchSettings):
    inspect: IntegrationSettings = Field(default_factory=IntegrationSettings)

def autopatch_inspect(settings: CustomAutopatchSettings) -> None:
    get_inspect_patcher(settings.inspect).attempt_patch()

def reset_autopatch_inspect() -> None:
    get_inspect_patcher().undo_patch()