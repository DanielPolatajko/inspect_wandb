from typing import Any
from inspect_ai.hooks import Hooks, RunEnd, SampleEnd, SampleStart, TaskStart, TaskEnd, EvalSetStart, EvalSetEnd
import weave
from weave.trace.settings import UserSettings
from inspect_wandb.weave.utils import format_score_types, format_sample_display_name
from inspect_wandb.shared.utils import format_wandb_id_string as format_model_name
from inspect_wandb.config.settings import WeaveSettings
from logging import getLogger
from inspect_wandb.weave.autopatcher import get_inspect_patcher, CustomAutopatchSettings
from inspect_wandb.weave.custom_evaluation_logger import CustomEvaluationLogger
from inspect_wandb.exceptions import WeaveEvaluationException
from weave.trace.weave_client import Call
from weave.trace.context import call_context
from typing_extensions import override
import asyncio
from weave.trace.autopatch import IntegrationSettings, OpSettings
from weave import integrations

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
    _eval_set: bool = False
    _eval_set_log_dir: str | None = None

    @override
    def enabled(self) -> bool:
        self._load_settings()
        assert self.settings is not None
        return self.settings.enabled

    @override
    async def on_eval_set_start(self, data: EvalSetStart) -> None:
        self._eval_set = True
        self._eval_set_log_dir = data.log_dir

    @override
    async def on_eval_set_end(self, data: EvalSetEnd) -> None:
        self.weave_client.finish(use_progress_bar=False)
        if self.settings is not None and self.settings.autopatch:
            get_inspect_patcher().undo_patch()


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

        if not self._eval_set:
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
                    print_call_link=False,
                    display_viewer="print",
                    implicitly_patch_integrations=False
                ),
            )
            self._autopatch(model=data.spec.model)
            self._weave_initialized = True
            logger.info(f"Weave initialized for task {data.spec.task}")
        
        model_name = format_model_name(data.spec.model) 
        weave_eval_logger = CustomEvaluationLogger(
            name=data.spec.task,
            dataset=data.spec.dataset.name or "test_dataset", # TODO: set a default dataset name
            model=model_name,
            eval_attributes=self._get_eval_metadata(data, self._eval_set_log_dir)
        )
        
        self.weave_eval_loggers[data.eval_id] = weave_eval_logger
        
        # Store task name mapping for use in sample hooks
        self.task_mapping[data.eval_id] = data.spec.task
        
        assert weave_eval_logger._evaluate_call is not None
        call_context.push_call(weave_eval_logger._evaluate_call)

        if weave_eval_logger._evaluate_call is not None:
            weave_url = weave_eval_logger._evaluate_call.ui_url
        else:
            weave_url = None

        data.spec.metadata = (data.spec.metadata or {}) | {"weave_run_url": weave_url}

    @override
    async def on_task_end(self, data: TaskEnd) -> None:
        if not self._hooks_enabled:
            return
            
        weave_eval_logger = self.weave_eval_loggers.get(data.eval_id)
        assert weave_eval_logger is not None
        
        summary: dict = {}
        if data.log and data.log.results:
            for score in data.log.results.scores:
                scorer_name = score.name
                if score.metrics:
                    summary[scorer_name] = {}
                    for metric_name, metric in score.metrics.items():
                        summary[scorer_name][metric_name] = metric.value
            summary["sample_count"] = data.log.results.total_samples
        weave_eval_logger.log_summary(summary)

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
        if not self._hooks_enabled:
            return

        task = asyncio.create_task(self._log_sample_to_weave_async(data))
        task.add_done_callback(self._handle_weave_task_result)

    def _handle_weave_task_result(self, task: asyncio.Task) -> None:
        """Handle results/exceptions from Weave logging tasks"""
        if (e:= task.exception()):
            raise e

    async def _log_sample_to_weave_async(self, data: SampleEnd) -> None:
        """
        Log sample data to Weave asynchronously but outside Inspect's semaphore context.
        This prevents Weave operations from interfering with Inspect's semaphore permits.
        """

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
                    await sample_score_logger.alog_score(
                        scorer=k,
                        score=format_score_types(v.value)
                    )

        # Total time
        if (
            hasattr(data.sample, "total_time")
            and data.sample.total_time is not None
        ):
            await sample_score_logger.alog_score(
                scorer="total_time", score=data.sample.total_time
            )

        # Total tokens
        if hasattr(data.sample, "model_usage") and data.sample.model_usage:
            for model_name, usage in data.sample.model_usage.items():
                if usage.total_tokens is not None:
                    await sample_score_logger.alog_score(
                        scorer="total_tokens", score=usage.total_tokens
                    )
                    break

        # Number of tools
        if (
            hasattr(data.sample, "metadata")
            and data.sample.metadata
            and "Annotator Metadata" in data.sample.metadata
            and "Number of tools" in data.sample.metadata["Annotator Metadata"]
        ):
            await sample_score_logger.alog_score(
                scorer="num_tool_calls",
                score=int(
                    data.sample.metadata["Annotator Metadata"]["Number of tools"]
                ),
            )

        if not getattr(sample_score_logger, '_has_finished', False):
            sample_score_logger.finish()

        if self.settings is not None and self.settings.autopatch:
            if data.sample_id in self.sample_calls:
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
        return { k[len("inspect_wandb_weave_"):]: v for k,v in data.spec.metadata.items() if k.lower().startswith("inspect_wandb_weave_")} or None

    def _load_settings(self, overrides: dict[str, Any] | None = None) -> None:
        if self.settings is None or overrides is not None:
            self.settings = WeaveSettings.model_validate(overrides or {})

    def _get_eval_metadata(self, data: TaskStart, log_dir: str | None = None) -> dict[str, str | dict[str, Any]]:

        eval_metadata = data.spec.metadata or {}
        
        inspect_data = {
            "run_id": data.run_id,
            "task_id": data.spec.task_id,
            "eval_id": data.eval_id,
        }

        if log_dir is not None:
            eval_metadata["eval_set_log_dir"] = log_dir
        
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

    def _autopatch(self, model: str) -> None:
        assert self.settings is not None
        if model.startswith("openrouter"):
            openai_settings=IntegrationSettings(
                op_settings=OpSettings(
                    name="openrouter.api.call"
                )
            )
        else:
            openai_settings = None
        autopatch_settings = CustomAutopatchSettings(
            openai=openai_settings
        )
        integrations.patch_openai(autopatch_settings.openai)
        integrations.patch_anthropic(autopatch_settings.anthropic)
        integrations.patch_google_genai(autopatch_settings.google_genai)
        integrations.patch_groq(autopatch_settings.groq)
        integrations.patch_huggingface(autopatch_settings.huggingface)
        integrations.patch_mistral(autopatch_settings.mistral)
        integrations.patch_vertexai(autopatch_settings.vertexai)
        integrations.patch_cohere(autopatch_settings.cohere)
        integrations.patch_llamaindex()
        if self.settings.autopatch:
            get_inspect_patcher(autopatch_settings.inspect).attempt_patch()
