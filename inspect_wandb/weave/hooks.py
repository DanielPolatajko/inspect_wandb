from typing import Any
from inspect_ai.hooks import RunEnd, SampleEnd, SampleStart, TaskStart, TaskEnd, EvalSetStart, EvalSetEnd
from os import environ
environ["WANDB_DISABLE_WEAVE"] = "1"
from weave import init as weave_init, attributes as weave_attributes
from weave.evaluation.eval_imperative import ScoreLogger, EvaluationLogger
from weave.trace.weave_client import WeaveClient
from weave.trace.settings import UserSettings
from inspect_wandb.weave.utils import format_score_types, format_sample_display_name
from inspect_wandb.shared.utils import format_wandb_id_string as format_model_name
from inspect_wandb.config.settings import WeaveSettings
from logging import getLogger
from inspect_wandb.weave.autopatcher import get_inspect_patcher, CustomAutopatchSettings
from inspect_wandb.exceptions import WeaveEvaluationException
from weave.trace.context import call_context
from typing_extensions import override
import asyncio
from weave.trace.autopatch import IntegrationSettings, OpSettings
from weave import integrations
from importlib.util import find_spec
from gql.transport.exceptions import TransportQueryError
from inspect_wandb.shared.base_hooks import InspectWandBHooks

logger = getLogger(__name__)

class WeaveEvaluationHooks(InspectWandBHooks):

    _settings_prefix = "inspect_wandb_weave_"
    _settings_cls = WeaveSettings

    weave_client: WeaveClient
    weave_eval_loggers: dict[str, EvaluationLogger] = {}
    settings: WeaveSettings | None = None
    sample_calls: dict[str, ScoreLogger] = {}
    task_mapping: dict[str, str] = {}
    _weave_initialized: bool = False
    _eval_set: bool = False
    _eval_set_log_dir: str | None = None

    @override
    async def on_eval_set_start(self, data: EvalSetStart) -> None:
        self._eval_set = True
        self._eval_set_log_dir = data.log_dir

    @override
    async def on_eval_set_end(self, data: EvalSetEnd) -> None:
        self.weave_client.finish(use_progress_bar=False)
        if self.settings is not None and not self.settings.eval_traces_only:
            get_inspect_patcher().undo_patch()


    @override
    async def on_run_end(self, data: RunEnd) -> None:
        if not self._weave_initialized:
            return

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

        self.weave_eval_loggers.clear()
        self.task_mapping.clear()

        if not self._eval_set:
            self.weave_client.finish(use_progress_bar=False)
            if self.settings is not None and not self.settings.eval_traces_only:
                get_inspect_patcher().undo_patch()


    @override
    async def on_task_start(self, data: TaskStart) -> None:

        if self._hooks_enabled is None:
            self._metadata_overrides = self._extract_settings_overrides_from_eval_metadata(data)
            self.settings = self._settings_cls.model_validate(self._metadata_overrides or {})
            assert self.settings is not None
            self._hooks_enabled = self.settings.enabled

        if not self._hooks_enabled:
            logger.info(f"Weave hooks disabled for run (task: {data.spec.task})")
            return

        assert self.settings is not None

        if not self._weave_initialized:
            try:
                self.weave_client = weave_init(
                    project_name=f"{self.settings.entity}/{self.settings.project}",
                    settings=UserSettings(
                        print_call_link=False,
                        display_viewer="print",
                        implicitly_patch_integrations=False
                    ),
                )
            except TransportQueryError as e:
                if f"Entity {self.settings.entity} not found" in str(e):
                    logger.warning(f"Weave integration disabled: invalid entity: {self.settings.entity}. {e}")
                elif f"Project {self.settings.project} not found" in str(e):
                    logger.warning(f"Weave integration disabled: invalid project: {self.settings.project}. {e}")
                else:
                    logger.warning(f"Weave integration disabled: {e}")
                self.settings.enabled = False
                self._hooks_enabled = False
                return

            self._autopatch(model=data.spec.model)
            self._weave_initialized = True
            logger.info(f"Weave initialized for task {data.spec.task}")

        model_name = format_model_name(data.spec.model)
        weave_eval_logger = EvaluationLogger(
            name=data.spec.task,
            dataset=data.spec.dataset.name or "test_dataset",
            model=model_name,
            eval_attributes=self._get_eval_metadata(data, self._eval_set_log_dir),
            scorers=None
        )

        self.weave_eval_loggers[data.eval_id] = weave_eval_logger
        self.task_mapping[data.eval_id] = data.spec.task

        assert weave_eval_logger._evaluate_call is not None
        call_context.push_call(weave_eval_logger._evaluate_call)

        weave_url = weave_eval_logger._evaluate_call.ui_url

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
        weave_eval_logger.log_summary({"summary": summary}, auto_summarize=False)

    @override
    async def on_sample_start(self, data: SampleStart) -> None:
        if (not self._hooks_enabled) or (
            self.settings is not None and self.settings.eval_traces_only
        ):
            return

        weave_eval_logger = self.weave_eval_loggers.get(data.eval_id)
        assert weave_eval_logger is not None

        task_name = self.task_mapping.get(data.eval_id, "unknown_task")

        if self.settings is not None:
            with weave_attributes(
                {
                    "sample_id": data.summary.id,
                    "sample_uuid": data.sample_id,
                    "epoch": data.summary.epoch,
                    "task_name": task_name,
                    "task_id": data.eval_id,
                    "metadata": data.summary.metadata,
                }
            ):
                sample_logger = weave_eval_logger.log_prediction(
                    inputs={"input": data.summary.input},
                )

            sample_logger.predict_call.display_name = format_sample_display_name(
                self.settings.sample_name_template, task_name=task_name, sample_id=data.summary.id, epoch=data.summary.epoch
            )

            call_context.push_call(sample_logger.predict_call)

            self.sample_calls[data.sample_id] = sample_logger

    @override
    async def on_sample_end(self, data: SampleEnd) -> None:
        if (not self._hooks_enabled) or (
            self.settings is not None and self.settings.eval_traces_only
        ):
            return

        task = asyncio.create_task(self._log_sample_to_weave_async(data))
        task.add_done_callback(self._handle_weave_task_result)

    def _handle_weave_task_result(self, task: asyncio.Task) -> None:
        if (e:= task.exception()):
            raise e

    async def _log_sample_to_weave_async(self, data: SampleEnd) -> None:
        weave_eval_logger = self.weave_eval_loggers.get(data.eval_id)
        assert weave_eval_logger is not None

        sample_score_logger = self.sample_calls.get(data.sample_id)
        if sample_score_logger is None or sample_score_logger._has_finished:
            logger.info(f"Sample {data.sample_id} already logged, skipping")
            return
        sample_score_logger.output = data.sample.output.completion

        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                score_metadata = (v.metadata or {}) | ({"explanation": v.explanation} if v.explanation is not None else {}) | ({"answer": v.answer} if v.answer is not None else {})
                with weave_attributes(score_metadata):
                    await sample_score_logger.alog_score(
                        scorer=k,
                        score=format_score_types(v.value, scorer_name=k)
                    )

        if (
            hasattr(data.sample, "total_time")
            and data.sample.total_time is not None
        ):
            await sample_score_logger.alog_score(
                scorer="total_time", score=data.sample.total_time
            )

        if hasattr(data.sample, "model_usage") and data.sample.model_usage:
            for model_name, usage in data.sample.model_usage.items():
                if usage.total_tokens is not None:
                    await sample_score_logger.alog_score(
                        scorer="total_tokens", score=usage.total_tokens
                    )
                    break

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
        self.sample_calls.pop(data.sample_id)

    def _get_eval_metadata(self, data: TaskStart, log_dir: str | None = None) -> dict[str, str | dict[str, Any]]:

        eval_metadata = data.spec.metadata or {}

        inspect_data = {
            "run_id": data.run_id,
            "task_id": data.spec.task_id,
            "eval_id": data.eval_id,
        }

        if log_dir is not None:
            eval_metadata["eval_set_log_dir"] = log_dir

        if data.spec.task_args:
            for key, value in data.spec.task_args.items():
                inspect_data[key] = value

        if data.spec.config is not None:
            config_dict = data.spec.config.__dict__
            for key, value in config_dict.items():
                if value is not None:
                    inspect_data[key] = value

        eval_metadata["inspect"] = inspect_data

        return eval_metadata

    def _autopatch(self, model: str) -> None:
        assert self.settings is not None
        if self.settings.eval_traces_only:
            return
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
        if find_spec("openai"):
            integrations.patch_openai(autopatch_settings.openai)
        if find_spec("anthropic"):
            integrations.patch_anthropic(autopatch_settings.anthropic)
        if find_spec("google.genai"):
            integrations.patch_google_genai(autopatch_settings.google_genai)
        if find_spec("groq"):
            integrations.patch_groq(autopatch_settings.groq)
        if find_spec("huggingface_hub"):
            integrations.patch_huggingface(autopatch_settings.huggingface)
        if find_spec("mistralai"):
            integrations.patch_mistral(autopatch_settings.mistral)
        if find_spec("vertexai"):
            integrations.patch_vertexai(autopatch_settings.vertexai)
        if find_spec("cohere"):
            integrations.patch_cohere(autopatch_settings.cohere)
        if find_spec("llama_index"):
            integrations.patch_llamaindex()
        get_inspect_patcher(autopatch_settings.inspect).attempt_patch()
