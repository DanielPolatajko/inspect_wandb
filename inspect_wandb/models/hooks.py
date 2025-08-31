import logging
from pathlib import Path
from typing_extensions import override

import wandb
from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, TaskStart, TaskEnd
from inspect_ai.log import EvalSample
from inspect_ai.scorer import CORRECT
from inspect_wandb.config.settings_loader import SettingsLoader
from inspect_wandb.config.settings import ModelsSettings
from inspect_wandb.config.extras_manager import INSTALLED_EXTRAS
if INSTALLED_EXTRAS["viz"]:
    from inspect_wandb.viz.inspect_viz_writer import InspectVizWriter

logger = logging.getLogger(__name__)

class Metric:
    ACCURACY: str = "accuracy"
    SAMPLES: str = "samples"

class WandBModelHooks(Hooks):

    settings: ModelsSettings | None = None

    _correct_samples: int = 0
    _total_samples: int = 0
    _wandb_initialized: bool = False
    _hooks_enabled: bool | None = None

    def __init__(self):
        if INSTALLED_EXTRAS["viz"]:
            self.viz_writer = InspectVizWriter()
        else:
            self.viz_writer = None

    @override
    def enabled(self) -> bool:
        self._load_settings()
        assert self.settings is not None
        return self.settings.enabled

    @override
    async def on_run_start(self, data: RunStart) -> None:
        self._load_settings()
        # Note: wandb.init() moved to lazy initialization in on_task_start
    
    @override
    async def on_run_end(self, data: RunEnd) -> None:
        # Only proceed with cleanup if WandB was actually initialized
        if not self._wandb_initialized:
            return

        self._log_summary(data)

        if self.settings is not None and self.settings.viz and self.viz_writer is not None:
            await self.viz_writer.log_scores_heatmap(data, self.run)

        if self.settings is not None and self.settings.files:
            for file in self.settings.files:
                file_path = Path(file)
                if file_path.exists():
                    try:
                        self.run.save(str(file), policy="now")  # TODO: fix wandb Symlinked warning for folder upload
                        logger.info(f"Successfully saved {file} to wandb")
                    except Exception as e:
                        logger.warning(f"Failed to save {file} to wandb: {e}")
                else:
                    logger.warning(f"File or folder '{file}' does not exist. Skipping wandb upload.")

        self.run.finish()

    @override
    async def on_task_start(self, data: TaskStart) -> None:
        # Ensure settings are loaded
        self._load_settings()
        assert self.settings is not None
        
        # Check enablement only on first task (all tasks share same metadata)
        if self._hooks_enabled is None:
            script_override = self._check_enable_override(data)
            # Use task-specific override if present, otherwise fall back to settings
            self._hooks_enabled = script_override if script_override is not None else self.settings.enabled
        
        if not self._hooks_enabled:
            logger.info(f"WandB model hooks disabled for run (task: {data.spec.task})")
            return
        
        # Lazy initialization: only init WandB when first task starts
        if not self._wandb_initialized:
            self.run = wandb.init(id=data.run_id, entity=self.settings.entity, project=self.settings.project) 

            if self.settings.config:
                self.run.config.update(self.settings.config)

            _ = self.run.define_metric(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)
            self._wandb_initialized = True
            logger.info(f"WandB initialized for task {data.spec.task}")
        
        inspect_tags = (
            f"inspect_task:{data.spec.task}",
            f"inspect_model:{data.spec.model}",
            f"inspect_dataset:{data.spec.dataset.name}",
        )
        if self.run.tags:
            self.run.tags = self.run.tags + inspect_tags
        else:
            self.run.tags = inspect_tags

        if self.settings.tags is not None and self.run.tags is not None:
            self.run.tags = self.run.tags + tuple(self.settings.tags)

    
    @override
    async def on_task_end(self, data: TaskEnd) -> None:
        if data.log.eval.metadata is None:
            data.log.eval.metadata = {"wandb_run_url": self.run.url}
        else:
            data.log.eval.metadata["wandb_run_url"] = self.run.url

    @override
    async def on_sample_end(self, data: SampleEnd) -> None:
        # Skip if hooks are disabled for this run
        if not self._hooks_enabled:
            return
            
        self._total_samples += 1
        if data.sample.scores:
            self._correct_samples += int(self._is_correct(data.sample))
            self.run.log(
                {Metric.SAMPLES: self._total_samples, Metric.ACCURACY: self._accuracy()}
            )

    def _log_summary(self, data: RunEnd) -> None:
        summary = {
            "samples_total": self._total_samples,
            "samples_correct": self._correct_samples,
            "accuracy": self._accuracy(),
            "logs": [log.location for log in data.logs],
        }
        self.run.summary.update(summary)
        logger.info(f"WandB Summary: {summary}")

    def _is_correct(self, sample: EvalSample) -> bool:
        if not sample.scores:
            return False

        values = [score.value for score in sample.scores.values()]
        return CORRECT in values or 1 in values or 1.0 in values or True in values

    def _accuracy(self) -> float:
        if self._total_samples == 0:
            return 0.0

        return self._correct_samples * 1.0 / self._total_samples

    def _check_enable_override(self, data: TaskStart) -> bool|None:
        """
        Check TaskStart metadata to determine if hooks should be enabled
        """
        if data.spec.metadata is None:
            return None
        return data.spec.metadata.get("models_enabled")

    def _load_settings(self) -> None:
        if self.settings is None:
            self.settings = SettingsLoader.load_inspect_wandb_settings(
                {"weave": {}, "models": {"viz": self.viz_writer is not None}}
            ).models