from typing import Callable
from unittest.mock import MagicMock
from pytest import MonkeyPatch
from inspect_ai import Task, eval as inspect_eval

class TestEndToEndInspectRuns:
    """
    A test class for tests which simulate an entire Inspect eval run
    """
    def test_weave_init_not_called_on_run_start_when_disabled(self, patched_weave_evaluation_hooks: dict[str, MagicMock], hello_world_eval: Callable[[], Task], monkeypatch: MonkeyPatch) -> None:
        # Given - Mock settings loader to return disabled weave settings
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "false")
        
        weave_init = patched_weave_evaluation_hooks["weave_init"]
        
        # When
        inspect_eval(hello_world_eval, model="mockllm/model")

        # Then
        assert isinstance(weave_init, MagicMock)
        weave_init.assert_not_called()

        # Cleanup
        monkeypatch.delenv("INSPECT_WANDB_WEAVE_ENABLED")

    def test_weave_init_called_on_run_start(self, patched_weave_evaluation_hooks: dict[str, MagicMock], hello_world_eval: Callable[[], Task]) -> None:
        weave_init = patched_weave_evaluation_hooks["weave_init"]

        # When
        inspect_eval(hello_world_eval, model="mockllm/model")

        # Then
        assert isinstance(weave_init, MagicMock)
        weave_init.assert_called_once()

    def test_weave_evaluation_finalised_with_exception_on_error(self, patched_weave_evaluation_hooks: dict[str, MagicMock], error_eval: Callable[[], Task]) -> None:
        # Given
        weave_evaluation_logger = patched_weave_evaluation_hooks["weave_evaluation_logger"]
        weave_evaluation_logger.finish = MagicMock()
        weave_evaluation_logger._is_finalized = False

        # When
        inspect_eval(error_eval, model="mockllm/model")

        # Then
        assert weave_evaluation_logger.finish.call_args_list[0][1]["exception"].error == "RuntimeError('Simulated failure')"

    def test_weave_evaluation_logger_created_on_task_start(self, patched_weave_evaluation_hooks: dict[str, MagicMock], hello_world_eval: Callable[[], Task]) -> None:
        # Given
        weave_evaluation_logger = patched_weave_evaluation_hooks["weave_evaluation_logger"]

        # When
        eval_logs = inspect_eval(hello_world_eval, model="mockllm/model")

        # Then
        assert isinstance(weave_evaluation_logger, MagicMock)
        assert len(eval_logs) == 1
        run_id = eval_logs[0].eval.run_id
        task_id = eval_logs[0].eval.task_id
        eval_id = eval_logs[0].eval.eval_id
        epochs = eval_logs[0].eval.config.epochs
        epochs_reducer = eval_logs[0].eval.config.epochs_reducer
        fail_on_error = eval_logs[0].eval.config.fail_on_error
        continue_on_fail = eval_logs[0].eval.config.continue_on_fail
        sandbox_cleanup = eval_logs[0].eval.config.sandbox_cleanup
        log_samples = eval_logs[0].eval.config.log_samples
        log_realtime = eval_logs[0].eval.config.log_realtime
        log_images = eval_logs[0].eval.config.log_images
        score_display = eval_logs[0].eval.config.score_display

        weave_evaluation_logger.assert_called_once_with(
            name="hello_world_eval",
            dataset="test_dataset",
            model="mockllm__model",
            eval_attributes={
                "test": "test",
                "inspect": {
                    "run_id": run_id,
                    "task_id": task_id,
                    "eval_id": eval_id,
                    'epochs': epochs, 
                    'epochs_reducer': epochs_reducer, 
                    'fail_on_error': fail_on_error, 
                    'continue_on_fail': continue_on_fail,
                    'sandbox_cleanup': sandbox_cleanup, 
                    'log_samples': log_samples, 
                    'log_realtime': log_realtime, 
                    'log_images': log_images, 
                    'score_display': score_display
                }
            }
        )