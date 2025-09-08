from typing import Any
from inspect_ai.hooks import Hooks, RunEnd, SampleEnd, SampleStart, TaskStart, TaskEnd
import weave
from weave.trace.settings import UserSettings
from inspect_wandb.weave.utils import format_score_types, format_sample_display_name
from inspect_wandb.shared.utils import format_wandb_id_string as format_model_name
from inspect_wandb.config.settings_loader import SettingsLoader
from inspect_wandb.config.settings import WeaveSettings
from logging import getLogger
from inspect_wandb.weave.autopatcher import get_inspect_patcher, CustomAutopatchSettings
from inspect_wandb.weave.custom_evaluation_logger import CustomEvaluationLogger
from inspect_wandb.exceptions import WeaveEvaluationException
from weave.trace.weave_client import Call
from weave.trace.context import call_context
from typing_extensions import override

logger = getLogger(__name__)

class WeaveEvaluationHooks(Hooks):
    """
    Provides Inspect hooks for writing eval scores to the Weave Evaluations API.
    """

    weave_eval_loggers: dict[str, CustomEvaluationLogger] = {}
    settings: WeaveSettings | None = None
    sample_calls: dict[str, Call] = {}
    task_mapping: dict[str, str] = {}
    _weave_initialized: bool = False
    _hooks_enabled: bool | None = None

    @override
    def enabled(self) -> bool:
        self._load_settings()
        assert self.settings is not None
        return self.settings.enabled

    @override
    async def on_run_end(self, data: RunEnd) -> None:
        # Only proceed with cleanup if Weave was actually initialized
        if not self._weave_initialized:
            return
            
        # Finalize all active loggers
        for weave_eval_logger in self.weave_eval_loggers.values():
            if not weave_eval_logger._is_finalized:
                if data.exception is not None:
                    weave_eval_logger.finish(exception=data.exception)
                elif errors := [eval.error for eval in data.logs]:
                    weave_eval_logger.finish(
                        exception=WeaveEvaluationException(
                            message="Inspect run failed", 
                            error="\n".join([error.message for error in errors if error is not None])
                        )
                    )
                else:
                    weave_eval_logger.finish()
        
        # Clear the loggers dict and task mapping
        self.weave_eval_loggers.clear()
        self.task_mapping.clear()
        self.weave_client.finish(use_progress_bar=False)
        if self.settings is not None and self.settings.autopatch:
            get_inspect_patcher().undo_patch()

    @override
    async def on_task_start(self, data: TaskStart) -> None:
        
        # Check enablement only on first task (all tasks share same metadata)
        if self._hooks_enabled is None:
            metadata_overrides = self._extract_settings_overrides_from_eval_metadata(data)
            self._load_settings(overrides=metadata_overrides)
            assert self.settings is not None
            self._hooks_enabled = self.settings.enabled
        
        if not self._hooks_enabled:
            logger.info(f"Weave hooks disabled for run (task: {data.spec.task})")
            return

        assert self.settings is not None
        
        # Lazy initialization: only init Weave when first task starts
        if not self._weave_initialized:
            self.weave_client = weave.init(
                project_name=f"{self.settings.entity}/{self.settings.project}",
                settings=UserSettings(
                    print_call_link=False
                )
            )
            if self.settings.autopatch:
                get_inspect_patcher(CustomAutopatchSettings().inspect).attempt_patch()
            self._weave_initialized = True
            logger.info(f"Weave initialized for task {data.spec.task}")
        
        model_name = format_model_name(data.spec.model) 
        weave_eval_logger = CustomEvaluationLogger(
            name=data.spec.task,
            dataset=data.spec.dataset.name or "test_dataset", # TODO: set a default dataset name
            model=model_name,
            eval_attributes=self._get_eval_metadata(data)
        )
        
        self.weave_eval_loggers[data.eval_id] = weave_eval_logger
        
        # Store task name mapping for use in sample hooks
        self.task_mapping[data.eval_id] = data.spec.task
        
        assert weave_eval_logger._evaluate_call is not None
        call_context.push_call(weave_eval_logger._evaluate_call)

    @override
    async def on_task_end(self, data: TaskEnd) -> None:
        if not self._hooks_enabled:
            return
            
        weave_eval_logger = self.weave_eval_loggers.get(data.eval_id)
        assert weave_eval_logger is not None
        
        summary: dict[str, dict[str, int | float]] = {}
        if data.log and data.log.results:
            for score in data.log.results.scores:
                scorer_name = score.name
                if score.metrics:
                    summary[scorer_name] = {}
                    for metric_name, metric in score.metrics.items():
                        summary[scorer_name][metric_name] = metric.value
        weave_eval_logger.log_summary(summary)

        if data.log.eval.metadata is None and weave_eval_logger._evaluate_call is not None:
            data.log.eval.metadata = {"weave_run_url": weave_eval_logger._evaluate_call.ui_url}
        elif data.log.eval.metadata is not None and weave_eval_logger._evaluate_call is not None:
            data.log.eval.metadata["weave_run_url"] = weave_eval_logger._evaluate_call.ui_url

    @override
    async def on_sample_start(self, data: SampleStart) -> None:
        if not self._hooks_enabled:
            return
        
        if self.settings is not None and self.settings.autopatch:
            task_name = self.task_mapping.get(data.eval_id, "unknown_task")
            self.sample_calls[data.sample_id] = self.weave_client.create_call(
                op="inspect-sample",
                inputs={"input": data.summary.input},
                attributes={
                    "sample_id": data.summary.id, 
                    "sample_uuid": data.sample_id, 
                    "epoch": data.summary.epoch,
                    "task_name": task_name,
                    "task_id": data.eval_id,
                    "metadata": data.summary.metadata,
                },
                display_name=format_sample_display_name(self.settings.sample_name_template, task_name, data.summary.id, data.summary.epoch)
            )

    @override
    async def on_sample_end(self, data: SampleEnd) -> None:
        # Skip if hooks are disabled for this run
        if not self._hooks_enabled:
            return
            
        weave_eval_logger = self.weave_eval_loggers.get(data.eval_id)
        assert weave_eval_logger is not None
        
        sample_id = data.sample.id
        epoch = data.sample.epoch
        input_value = data.sample.input
        with weave.attributes({"sample_id": sample_id, "epoch": epoch}):
            sample_score_logger = weave_eval_logger.log_prediction(
                inputs={"input": input_value},
                output=data.sample.output.completion,
                parent_call=self.sample_calls[data.sample_id] if self.settings is not None and self.settings.autopatch else None
            )
        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                score_metadata = (v.metadata or {}) | ({"explanation": v.explanation} if v.explanation is not None else {})
                with weave.attributes(score_metadata):
                    sample_score_logger.log_score(
                        scorer=k,
                        score=format_score_types(v.value)
                    )

                    # Log various metrics to Weave
        try:
            # Total time
            if (
                hasattr(data.sample, "total_time")
                and data.sample.total_time is not None
            ):
                sample_score_logger.log_score(
                    scorer="total_time", score=data.sample.total_time
                )

            # Total tokens - model_usage is a dict of model_name -> ModelUsage
            if hasattr(data.sample, "model_usage") and data.sample.model_usage:
                # Get the first (and usually only) model's token usage
                for model_name, usage in data.sample.model_usage.items():
                    if usage.total_tokens is not None:
                        sample_score_logger.log_score(
                            scorer="total_tokens", score=usage.total_tokens
                        )
                        break  # Only log the first model's tokens

            # Number of tools from metadata - metadata is a dict
            if (
                hasattr(data.sample, "metadata")
                and data.sample.metadata
                and "Annotator Metadata" in data.sample.metadata
                and "Number of tools" in data.sample.metadata["Annotator Metadata"]
            ):
                sample_score_logger.log_score(
                    scorer="num_tool_calls",
                    score=int(
                        data.sample.metadata["Annotator Metadata"]["Number of tools"]
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to log metrics to Weave: {e}")
            raise e

        sample_score_logger.finish()
        if self.settings is not None and self.settings.autopatch:
            # Extract model tokens as {model_name: total_tokens} dict
            model_tokens = {
                format_model_name(model_name): usage.total_tokens 
                for model_name, usage in data.sample.model_usage.items()
            }
            
            self.weave_client.finish_call(
                self.sample_calls[data.sample_id], 
                output={
                    "output": data.sample.output.completion, 
                    "scores": data.sample.scores, 
                    "total_time": data.sample.total_time, 
                    "token_usage": model_tokens
                }
            )
            self.sample_calls.pop(data.sample_id)

    def _extract_settings_overrides_from_eval_metadata(self, data: TaskStart) -> dict[str, Any] | None:
        """
        Check TaskStart metadata to determine if hooks should be enabled
        """
        if data.spec.metadata is None:
            return None
        return { k[len("inspect_wandb_weave_"):]: v for k,v in data.spec.metadata.items() if k.startswith("inspect_wandb_weave_")} or None

    def _load_settings(self, overrides: dict[str, Any] | None = None) -> None:
        if self.settings is None or overrides is not None:
            self.settings = SettingsLoader.load_inspect_wandb_settings(
                settings={"weave": overrides or {}, "models": {}}
            ).weave

    def _get_eval_metadata(self, data: TaskStart) -> dict[str, str | dict[str, Any]]:

        eval_metadata = data.spec.metadata or {}
        
        inspect_data = {
            "run_id": data.run_id,
            "task_id": data.spec.task_id,
            "eval_id": data.eval_id,
            "sample_count": data.spec.config.limit if data.spec.config.limit is not None else data.spec.dataset.samples
        }
        
        # Add task_args key-value pairs
        if data.spec.task_args:
            for key, value in data.spec.task_args.items():
                inspect_data[key] = value
        
        # Add config key-value pairs if config is not None
        if data.spec.config is not None:
            config_dict = data.spec.config.__dict__
            for key, value in config_dict.items():
                if value is not None:
                    inspect_data[key] = value
        
        eval_metadata["inspect"] = inspect_data
        
        return eval_metadata
