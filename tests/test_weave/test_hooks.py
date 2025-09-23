from inspect_ai.log import EvalLog
from unittest.mock import MagicMock, PropertyMock, patch
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
from .conftest import WeaveTestClient

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
            eval_set_id=None,
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
        await hooks._log_sample_to_weave_async(sample)

        # Then
        mock_weave_eval_logger.log_prediction.assert_called_once_with(
            inputs={"input": "test_input"},
            output="test_output",
            parent_call=None
        )
        mock_score_logger.alog_score.assert_called_once_with(
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
            eval_set_id=None,
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
        await hooks._log_sample_to_weave_async(sample)

        # Then
        mock_weave_eval_logger.log_prediction.assert_called_once_with(
            inputs={"input": "test_input"},
            output="test_output",
            parent_call=None
        )
        mock_score_logger.alog_score.assert_called_once_with(
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
            eval_set_id=None,
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
            },
            "sample_count": 1
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
            eval_set_id=None,
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
            eval_set_id=None,
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

    @pytest.mark.parametrize("metadata_key", [
        "INSPECT_WANDB_WEAVE_ENABLED",
        "inspect_wandb_weave_enabled",
        "iNsPecT_wAnDb_WeAvE_EnAbLeD",
    ])
    def parse_settings_from_metadata_is_case_insensitive(self, create_task_start: Callable[dict | None, TaskStart], metadata_key: str) -> None:
        """Test that parse_settings_from_metadata is case insensitive"""
        # Given
        hooks = WeaveEvaluationHooks()
        metadata = create_task_start({
            metadata_key: True,
        })
        
        
        # When
        settings = hooks._extract_settings_overrides_from_eval_metadata(metadata)

        # Then
        assert settings is not None
        assert settings["enabled"] is True


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
        assert result == {"enabled": True}

    def test_check_enable_override_with_false_metadata(self, create_task_start: Callable[[dict | None], TaskStart]) -> None:
        """Test _check_enable_override returns False when metadata has weave_enabled: false"""
        # Given
        hooks = WeaveEvaluationHooks()
        task_start = create_task_start({"inspect_wandb_weave_enabled": False})
        
        # When
        result = hooks._extract_settings_overrides_from_eval_metadata(task_start)
        
        # Then
        assert result == {"enabled": False}

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
    async def test_weave_run_url_added_to_eval_metadata(self, test_settings: WeaveSettings, create_task_start: Callable[[dict | None], TaskStart], weave_test_client: WeaveTestClient) -> None:
        """Test weave_run_url is added to eval metadata"""
        # Given
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True  # Enable hooks for this test
        hooks._weave_initialized = True  # Mark as initialized for cleanup
        
        # Mock CustomEvaluationLogger to return our mock with proper ui_url
        mock_evaluation_logger = MagicMock(spec=EvaluationLogger)
        mock_call = MagicMock()
        type(mock_call).ui_url = PropertyMock(return_value="test_url")
        mock_evaluation_logger._evaluate_call = mock_call
        
        with patch('inspect_wandb.weave.hooks.CustomEvaluationLogger', return_value=mock_evaluation_logger):
            task_start = create_task_start()
            
            # When
            await hooks.on_task_start(
                task_start      
                )

        # Then
        assert task_start.spec.metadata["weave_run_url"] == "test_url"


class TestConcurrencyOnSampleEnd:
    """
    Tests for asynchronous Weave logging on sample end
    """

    @pytest.mark.asyncio
    async def test_on_sample_end_returns_immediately_without_blocking(self, test_settings: WeaveSettings) -> None:
        """Test that on_sample_end returns immediately without waiting for Weave operations"""
        import time

        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True

        sample = SampleEnd(
            eval_set_id=None,
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
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        start_time = time.time()
        await hooks.on_sample_end(sample)
        duration = time.time() - start_time

        assert duration < 0.01

    @pytest.mark.asyncio
    async def test_on_sample_end_creates_background_task(self, test_settings: WeaveSettings) -> None:
        """Test that on_sample_end creates a background task for Weave operations"""
        from unittest.mock import patch

        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True

        sample = SampleEnd(
            eval_set_id=None,
            run_id="test_run_id",
            eval_id="test_eval_id",
            sample_id="test_sample_id",
            sample=EvalSample(
                id=1,
                epoch=1,
                input="test_input",
                target="test_output",
                scores={},
                output=ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=ChatMessageAssistant(content="test_output"))])
            )
        )

        with patch('asyncio.create_task') as mock_create_task:
            await hooks.on_sample_end(sample)
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_prevents_double_finish_on_concurrent_samples(self, test_settings: WeaveSettings) -> None:
        """Test that finish() is only called once even with concurrent access"""
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True

        sample = SampleEnd(
            eval_set_id=None,
            run_id="test_run_id",
            eval_id="test_eval_id",
            sample_id="test_sample_id",
            sample=EvalSample(
                id=1,
                epoch=1,
                input="test_input",
                target="test_output",
                scores={},
                output=ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=ChatMessageAssistant(content="test_output"))])
            )
        )

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_score_logger = MagicMock(spec=ScoreLogger)
        mock_score_logger._has_finished = False
        mock_weave_eval_logger.log_prediction.return_value = mock_score_logger
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger

        await hooks._log_sample_to_weave_async(sample)

        mock_score_logger._has_finished = True
        await hooks._log_sample_to_weave_async(sample)

        mock_score_logger.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_alog_score_instead_of_log_score(self, test_settings: WeaveSettings) -> None:
        """Test that alog_score is used instead of log_score to avoid event loop conflicts"""
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True

        sample = SampleEnd(
            eval_set_id=None,
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

        await hooks._log_sample_to_weave_async(sample)

        mock_score_logger.alog_score.assert_called_once_with(
            scorer="test_score",
            score=1.0
        )
        mock_score_logger.log_score.assert_not_called()

    @pytest.mark.asyncio
    async def test_background_task_exceptions_handled_properly(self, test_settings: WeaveSettings) -> None:
        """Test that exceptions in background tasks are handled without crashing evaluation"""
        hooks = WeaveEvaluationHooks()
        hooks.settings = test_settings
        hooks._hooks_enabled = True

        mock_weave_eval_logger = MagicMock(spec=EvaluationLogger)
        mock_weave_eval_logger.log_prediction.side_effect = Exception("Weave error")
        hooks.weave_eval_loggers["test_eval_id"] = mock_weave_eval_logger
        with pytest.raises(Exception, match="Weave error"):
            hooks._handle_weave_task_result(MagicMock(exception=lambda: Exception("Weave error")))
