from typing import Any, ClassVar
from inspect_ai.hooks import Hooks, TaskStart
from inspect_wandb.config.settings.base import InspectWandBBaseSettings
from typing_extensions import override


class InspectWandBHooks(Hooks):
    _settings_prefix: ClassVar[str]
    _settings_cls: ClassVar[type[InspectWandBBaseSettings]]

    settings: InspectWandBBaseSettings | None = None
    _hooks_enabled: bool | None = None
    _metadata_overrides: dict[str, Any] | None = None

    @override
    def enabled(self) -> bool:
        self.settings = self._settings_cls.model_validate(
            self._metadata_overrides or {}
        )
        return self.settings.enabled

    def _extract_settings_overrides_from_eval_metadata(
        self, data: TaskStart
    ) -> dict[str, Any] | None:
        if data.spec.metadata is None:
            return None
        return {
            k[len(self._settings_prefix) :]: v
            for k, v in data.spec.metadata.items()
            if k.lower().startswith(self._settings_prefix)
        } or None
