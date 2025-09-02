from inspect_ai.log import EvalLog
from unittest.mock import MagicMock
from inspect_ai.hooks import SampleEnd, TaskEnd, RunEnd, TaskStart, SampleStart
from inspect_ai.model import ChatCompletionChoice, ModelOutput, ChatMessageAssistant
from inspect_ai.log import EvalSample,EvalSampleSummary
from inspect_ai._eval.eval import EvalLogs
from inspect_wandb.weave.hooks import WeaveEvaluationHooks
from inspect_ai.scorer import Score
import pytest
from weave.evaluation.eval_imperative import ScoreLogger, EvaluationLogger
from inspect_wandb.config.settings import WeaveSettings
from weave.trace.weave_client import WeaveClient, Call
from typing import Callable

@pytest.fixture(scope="function")
def test_settings() -> WeaveSettings:
    return WeaveSettings(
        enabled=True,
        entity="test-entity",
        project="test-project",
        autopatch=False
    )
    

class TestWeaveEvaluationHooks:
    """
    Tests for individual hook functionalities
    """

    @pytest.mark.asyncio
    async def test_writes_eval_score_to_weave_on_sample_end(self, test_settings: WeaveSettings) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True  # Enable hooks for this test
        sample = SampleEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            sample_id="test_sample_id",
            sample=EvalSample(
                id=1,
                epoch=1,
                input="test_input",
                target="test_output",
                scores={"test_score": Score(value=1.0)},
                output=ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=ChatMessageAssistant(content="test_output"))])
            )
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_score_logger = MagicMock(spec=ScoreLogger)
        mock_weave_eval_logger.log_prediction.return_value = mock_score_logger
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_sample_end(sample)

        # Then
        mock_weave_eval_logger.log_prediction.assert_called_once_with(
            inputs={"input": "test_input"},
            output="test_output",
            parent_call=None
        )
        mock_score_logger.log_score.assert_called_once_with(
            scorer="test_score",
            score=1.0,
        )
        mock_score_logger.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_eval_score_to_weave_on_sample_end_with_metadata(self, test_settings: WeaveSettings) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True  # Enable hooks for this test
        sample = SampleEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            sample_id="test_sample_id",
            sample=EvalSample(
                id=1,
                epoch=1,
                input="test_input",
                target="test_output",
                scores={"test_score": Score(value=1.0, metadata={"test": "test"})},
                output=ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=ChatMessageAssistant(content="test_output"))])
            )
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_score_logger = MagicMock(spec=ScoreLogger)
        mock_weave_eval_logger.log_prediction.return_value = mock_score_logger
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_sample_end(sample)

        # Then
        mock_weave_eval_logger.log_prediction.assert_called_once_with(
            inputs={"input": "test_input"},
            output="test_output",
            parent_call=None
        )
        mock_score_logger.log_score.assert_called_once_with(
            scorer="test_score",
            score=1.0
        )
        mock_score_logger.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_inspect_eval_summary_metrics_to_weave_on_task_end(self, task_end_eval_log: EvalLog, test_settings: WeaveSettings) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True  # Enable hooks for this test
        task_end = TaskEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            log=task_end_eval_log
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_weave_eval_logger._evaluate_call = MagicMock(spec=Call)
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_task_end(task_end)

        # Then
        expected_summary = {
            "test_score": {
                "test_metric": 1.0
            }
        }
        mock_weave_eval_logger.log_summary.assert_called_once_with(
            expected_summary
        )

    @pytest.mark.asyncio
    async def test_passes_exception_to_weave_on_error_run_end(self, test_settings: WeaveSettings) -> None:
        # Given
        e = Exception("test_exception")
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True  # Enable hooks for this test
        hooks._weave_initialized = True  # Mark as initialized for cleanup
        hooks.weave_client = MagicMock(spec=WeaveClient)
        task_end = RunEnd(
            run_id="test_run_id",
            logs=EvalLogs([]),       
            exception=e
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_weave_eval_logger.finish = MagicMock()
        mock_weave_eval_logger._is_finalized = False
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        # When
        await hooks.on_run_end(task_end)

        # Then
        mock_weave_eval_logger.finish.assert_called_once_with(
            exception=e
        )

    @pytest.mark.asyncio
    async def test_adds_sample_call_with_metadata_on_sample_start(self, test_settings: WeaveSettings) -> None:
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks.settings.autopatch = True
        hooks._hooks_enabled = True  # Enable hooks for this test
        hooks._weave_initialized = True  # Mark as initialized for cleanup
        hooks.weave_client = MagicMock(spec=WeaveClient)
        
        # Set up task mapping (simulating task start)
        hooks.task_mapping["test_eval_id"] = "test_task"
        
        sample = SampleStart(
            run_id="test_run_id",
            eval_id="test_eval_id",
            sample_id="test_sample_id",
            summary=EvalSampleSummary(
                id=1,
                epoch=1,
                input="test_input",
                target="test_output",
                uuid="test_sample_id"
            )
        )

        # When  
        await hooks.on_sample_start(sample)

        # Then
        hooks.weave_client.create_call.assert_called_once_with(
            op="inspect-sample",
            inputs={"input": "test_input"},
            attributes={
                "sample_id": 1, 
                "sample_uuid": "test_sample_id", 
                "epoch": 1,
                "task_name": "test_task",
                "task_id": "test_eval_id",
                "metadata": {}
            },
            display_name="test_task-sample-1-epoch-1"
        )


class TestWeaveEnablementPriority:
    """
    Tests for the new enablement priority logic: script metadata > project config
    """

    def test_check_enable_override_with_true_metadata(self, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test _check_enable_override returns True when metadata has weave_enabled: true"""
        # Given
        hooks = WeaveEvaluationHooks()
        task_start = create_task_start({"inspect_wandb_weave_enabled": True})
        
        # When
        result = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        
        # Then
        assert result == {"inspect_wandb_weave_enabled": True}

    def test_check_enable_override_with_false_metadata(self, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test _check_enable_override returns False when metadata has weave_enabled: false"""
        # Given
        hooks = WeaveEvaluationHooks()
        task_start = create_task_start({"inspect_wandb_weave_enabled": False})
        
        # When
        result = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        
        # Then
        assert result == {"inspect_wandb_weave_enabled": False}

    def test_check_enable_override_with_no_weave_enabled_key(self, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test _check_enable_override returns None when metadata exists but no weave_enabled key"""
        # Given 
        hooks = WeaveEvaluationHooks()
        task_start = create_task_start({"other_key": "value"})
        
        # When
        result = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        
        # Then
        assert result is None

    def test_check_enable_override_with_none_metadata(self, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test _check_enable_override returns None when metadata is None"""
        # Given
        hooks = WeaveEvaluationHooks()
        task_start = create_task_start(None)
        
        # When
        result = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        
        # Then
        assert result is None

    @pytest.mark.asyncio
    async def test_script_metadata_overrides_settings_enabled_true(self, test_settings: WeaveSettings, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test script metadata weave_enabled: true overrides settings.enabled: false"""
        # Given
        test_settings.enabled = False
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        task_start = create_task_start({"inspect_wandb_weave_enabled": True})
        
        # When
        metadata_overrides = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        hooks._load_settings(overrides=metadata_overrides)
        
        # Then
        assert hooks.settings.enabled is True

    @pytest.mark.asyncio
    async def test_fallback_to_settings_when_no_metadata_override(self, test_settings: WeaveSettings, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test falls back to settings.enabled when no script metadata override"""
        # Given
        test_settings.enabled = True
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        task_start = create_task_start(None)
        
        # When
        metadata_overrides = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        hooks._load_settings(overrides=metadata_overrides)
        
        # Then
        assert hooks.settings.enabled

    @pytest.mark.asyncio
    async def test_fallback_to_settings_when_metadata_has_no_weave_enabled_key(self, test_settings: WeaveSettings, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test falls back to settings.enabled when metadata exists but has no weave_enabled key"""
        # Given
        test_settings.enabled = False  # Project config says disabled
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        task_start = create_task_start(metadata={"other_key": "value"})  # No weave_enabled key
        
        # When
        metadata_overrides = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        hooks._load_settings(overrides=metadata_overrides)
        
        # Then
        assert not hooks.settings.enabled

    @pytest.mark.asyncio
    async def test_weave_run_url_added_to_eval_metadata(self, test_settings: WeaveSettings, task_end_eval_log: EvalLog) -> None:
        """Test weave_run_url is added to eval metadata"""
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True  # Enable hooks for this test
        hooks._weave_initialized = True  # Mark as initialized for cleanup
        hooks.weave_eval_loggers["test_eval_id"] = MagicMock(spec=EvaluationLogger)
        hooks.weave_eval_loggers["test_eval_id"]._evaluate_call = MagicMock(spec=Call)
        hooks.weave_eval_loggers["test_eval_id"]._evaluate_call.ui_url = "test_url"
        
        # When
        await hooks.on_task_end(TaskEnd(
            run_id="test_run_id",
            eval_id="test_eval_id",
            log=task_end_eval_log
        ))

        # Then
        assert task_end_eval_log.eval.metadata["weave_run_url"] == "test_url"
