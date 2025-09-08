import logging
from pathlib import Path
from typing import Any
from typing_extensions import override

import wandb
from inspect_ai.hooks import Hooks, RunEnd, SampleEnd, TaskStart, TaskEnd, EvalSetStart
from inspect_ai.log import EvalSample
from inspect_ai.scorer import CORRECT
from inspect_wandb.config.settings_loader import SettingsLoader
from inspect_wandb.config.settings import ModelsSettings
from inspect_wandb.shared.utils import format_wandb_id_string
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
    _is_eval_set: bool = False
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
    async def on_eval_set_start(self, data: EvalSetStart) -> None:
        """
        Hook to run at the start of an eval set.
        Sets a flag to indicate that this is an eval set run, and adds log_dir to state
        """
        self._is_eval_set = True
        self.eval_set_log_dir = data.log_dir
    
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

        self._wandb_initialized = False

        match data.exception:
            case KeyboardInterrupt():
                logger.error("Inspect exited due to KeyboardInterrupt")
                self.run.finish(exit_code=1)
                return
            case Exception():
                logger.exception(data.exception)
                self.run.finish(exit_code=2)
                return
            case SystemExit() as sysexit:
                logger.error(f"SystemExit running eval set: {sysexit}")
                self.run.finish(exit_code=3)
                return
            
        if not(all(log.status == "success" for log in data.logs)):
            logger.warning("One or more tasks failed, may retry if eval-set")
            self.run.finish(exit_code=4)
        else:
            self.run.finish()

    @override
    async def on_task_start(self, data: TaskStart) -> None:
        """
        Hook to run at the start of each inspect task.
        Initializes WandB run if not already initialized.
        Updates tags, config, and other metadata based on user-provided settings.
        """

        self._load_settings()
        assert self.settings is not None
        
        # Override settings from eval metadata on first task only
        if self._hooks_enabled is None:
            metadata_overrides = self._extract_settings_overrides_from_eval_metadata(data)
            self._load_settings(overrides=metadata_overrides)
            self._hooks_enabled = self.settings.enabled
        
        if not self._hooks_enabled:
            logger.info(f"WandB model hooks disabled for run (task: {data.spec.task})")
            return
        
        if self._is_eval_set:
            wandb_run_id = format_wandb_id_string(self.eval_set_log_dir)
        else:
            wandb_run_id = data.eval_id
        
        # Lazy initialization: only init WandB when first task starts
        print(self._wandb_initialized)
        if not self._wandb_initialized:
            print("Initializing WandB...")
            self.run = wandb.init(
                id=wandb_run_id, 
                name=f"Inspect eval-set: {self.eval_set_log_dir}" if self._is_eval_set else None,
                entity=self.settings.entity, 
                project=self.settings.project,
                resume="allow",
            ) 

            if self.run.summary:
                self._total_samples = int(self.run.summary.get("samples_total", 0))
                self._correct_samples = int(self.run.summary.get("samples_correct", 0))

            if self.settings.add_metadata_to_config and data.spec.metadata is not None:
                self.run.config.update({k: v for k,v in data.spec.metadata.items() if k != "inspect_wandb_models_config"})
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

    def _extract_settings_overrides_from_eval_metadata(self, data: TaskStart) -> dict[str, Any] | None:
        """
        Check TaskStart metadata to determine if hooks should be enabled
        """
        if data.spec.metadata is None:
            return None
        return { k[len("inspect_wandb_models_"):]: v for k,v in data.spec.metadata.items() if k.startswith("inspect_wandb_models_")}

    def _load_settings(self, overrides: dict[str, Any] | None = None) -> None:
        if self.settings is None or overrides is not None:
            self.settings = SettingsLoader.load_inspect_wandb_settings(
                {"weave": {}, "models": overrides or {}}
            ).models