from inspect_wandb.models.hooks import WandBModelHooks
from inspect_wandb.config.settings import ModelsSettings
from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from wandb.sdk.wandb_run import Run
from wandb.sdk.wandb_config import Config
from wandb.sdk.wandb_summary import Summary
from typing import Callable
from inspect_ai.hooks import TaskStart, SampleEnd, RunEnd
from inspect_ai.log import EvalSample
from inspect_ai.scorer import Score 
from inspect_wandb.models.hooks import Metric

@pytest.fixture(scope="function")
def mock_wandb_run() -> Run:
    mock_run = MagicMock()  # Remove spec=Run to allow property mocking
    mock_run.config = MagicMock(spec=Config)
    mock_run.config.update = MagicMock()
    mock_run.define_metric = MagicMock()
    mock_run.tags = []
    mock_run.summary = MagicMock(spec=Summary)
    mock_run.summary.update = MagicMock()
    mock_run.save = MagicMock()
    mock_run.finish = MagicMock()
    
    # Mock the url property using PropertyMock
    type(mock_run).url = PropertyMock(return_value="mock_wandb_url")
    
    return mock_run

class TestWandBModelHooks:
    """
    Tests for the WandBModelHooks class.
    """

    def test_enabled(self, initialise_wandb: None) -> None:
        """
        Test that the enabled method returns True when the settings are set to True.
        """
        hooks = WandBModelHooks()
        assert hooks.enabled()

    def test_enabled_returns_false_when_settings_are_set_to_false(self) -> None:
        """
        Test that the enabled method returns False when the settings are set to False.
        """
        # Mock the settings loader to return disabled models settings
        disabled_settings = ModelsSettings(
            enabled=False, 
            entity="test-entity", 
            project="test-project",
        )
        with patch('inspect_wandb.models.hooks.ModelsSettings.model_validate') as mock_models_settings:
            mock_models_settings.return_value = disabled_settings
            hooks = WandBModelHooks()
            assert not hooks.enabled()

    @pytest.mark.asyncio
    async def test_wandb_initialised_on_task_start(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart], initialise_wandb: None) -> None:
        """
        Test that the on_task_start method initializes the WandB run.
        """
        hooks = WandBModelHooks()
        mock_init = MagicMock(return_value=mock_wandb_run)
        task_start = create_task_start()
        with patch('inspect_wandb.models.hooks.wandb.init', mock_init):
            await hooks.on_task_start(task_start)

            mock_init.assert_called_once_with(id="test_run_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_not_called()
            hooks.run.define_metric.assert_called_once_with(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)
            assert hooks.run.tags == ("inspect_task:test_task", "inspect_model:mockllm/model", "inspect_dataset:test-dataset")

    @pytest.mark.asyncio
    async def test_wandb_config_updated_on_task_start_if_settings_config_is_set(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart], initialise_wandb: None) -> None:
        """
        Test that the on_task_start method initializes the WandB run with config.
        """
        hooks = WandBModelHooks()
        mock_init = MagicMock(return_value=mock_wandb_run)
        task_start = create_task_start()
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
            config={"test": "test"}
        )
        with patch('inspect_wandb.models.hooks.wandb.init', mock_init):
            await hooks.on_task_start(task_start)

            mock_init.assert_called_once_with(id="test_run_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_called_once_with({"test": "test"}, allow_val_change=True)
            hooks.run.define_metric.assert_called_once_with(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)
            assert hooks.run.tags == ("inspect_task:test_task", "inspect_model:mockllm/model", "inspect_dataset:test-dataset")

    @pytest.mark.asyncio
    async def test_wandb_init_called_with_eval_set_log_dir_if_eval_set(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart], initialise_wandb: None) -> None:
        """
        Test that the on_task_start method initializes the WandB run with eval-set log dir.
        """
        hooks = WandBModelHooks()
        mock_init = MagicMock(return_value=mock_wandb_run)
        task_start = create_task_start()
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
        )
        hooks._is_eval_set = True
        hooks.eval_set_log_dir = "test_eval_set_log_dir"
        with patch('inspect_wandb.models.hooks.wandb.init', mock_init):
            await hooks.on_task_start(task_start)

            mock_init.assert_called_once_with(id="test_eval_set_id", name="Inspect eval-set: test_eval_set_log_dir", entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True

    @pytest.mark.asyncio
    async def test_wandb_config_updated_with_eval_metadata(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart], initialise_wandb: None) -> None:
        """
        Test that the on_task_start method initializes the WandB run with config.
        """
        hooks = WandBModelHooks()
        mock_init = MagicMock(return_value=mock_wandb_run)
        task_start = create_task_start()
        task_start.spec.metadata = {"test": "test"}
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        with patch('inspect_wandb.models.hooks.wandb.init', mock_init):
            await hooks.on_task_start(task_start)
            mock_init.assert_called_once_with(id="test_run_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_called_once_with({"test": "test"}, allow_val_change=True)

    @pytest.mark.asyncio
    async def test_wandb_config_not_updated_with_eval_metadata_if_add_metadata_to_config_is_false(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart], initialise_wandb: None) -> None:
        """
        Test that the on_task_start method initializes the WandB run with config.
        """
        hooks = WandBModelHooks()
        mock_init = MagicMock(return_value=mock_wandb_run)
        task_start = create_task_start()
        task_start.spec.metadata = {"test": "test"}
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
            add_metadata_to_config=False,
            config=None
        )
        hooks._hooks_enabled = True
        with patch('inspect_wandb.models.hooks.wandb.init', mock_init):
            await hooks.on_task_start(task_start)
            mock_init.assert_called_once_with(id="test_run_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_wandb_tags_updated_on_task_start_if_settings_tags_are_set(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart], initialise_wandb: None) -> None:
        """
        Test that the on_task_start method adds settings tags to the run tags.
        """
        hooks = WandBModelHooks()
        mock_init = MagicMock(return_value=mock_wandb_run)
        task_start = create_task_start()
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
            tags=["custom-tag1", "custom-tag2"]
        )
        with patch('inspect_wandb.models.hooks.wandb.init', mock_init):
            await hooks.on_task_start(task_start)

            mock_init.assert_called_once_with(id="test_run_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_not_called()
            hooks.run.define_metric.assert_called_once_with(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)
            expected_tags = ("inspect_task:test_task", "inspect_model:mockllm/model", "inspect_dataset:test-dataset", "custom-tag1", "custom-tag2")
            assert hooks.run.tags == expected_tags

    @pytest.mark.asyncio
    async def test_accuracy_and_samples_logged_on_sample_end(self, mock_wandb_run: Run) -> None:
        """
        Test that the on_sample_end method logs the accuracy and samples.
        """
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._total_samples = 9
        hooks._correct_samples = 4
        hooks._hooks_enabled = True

        # When
        await hooks.on_sample_end(
            SampleEnd(
                eval_set_id=None,
                run_id="test-run-id",
                eval_id="test-eval-id",
                sample_id="test-sample-id",
                sample=EvalSample(
                    id="test-sample-id",
                    epoch=1,
                    scores={"score": Score(value=True)},
                    input="test-input",
                    target="test-target"
                )
            )
        )

        # Then
        hooks.run.log.assert_called_once_with({Metric.SAMPLES: 10, Metric.ACCURACY: 0.5})
        assert hooks._total_samples == 10
        assert hooks._correct_samples == 5

    @pytest.mark.asyncio
    async def test_summary_logged_on_run_end(self, mock_wandb_run: Run) -> None:
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._total_samples = 10
        hooks._correct_samples = 5
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": True, "exception": None}}

        # When
        await hooks.on_run_end(
            RunEnd(
                eval_set_id=None,
                run_id="test-run",
                exception=None,
                logs=[]
            )
        )

        # Then
        hooks.run.summary.update.assert_called_once_with({
            "samples_total": 10,
            "samples_correct": 5,
            "accuracy": 0.5,
            "logs": []
        })

    @pytest.mark.asyncio
    async def test_files_saved_on_run_end_when_file_exists(self, mock_wandb_run: Run) -> None:
        """Test that existing files are saved to wandb"""
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
            files=["test-file.txt"]
        )
        hooks._total_samples = 10
        hooks._correct_samples = 5
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": True, "exception": None}}

        # When - mock Path.exists() to return True
        with patch('inspect_wandb.models.hooks.Path.exists', return_value=True):
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=None,
                    logs=[]
                )
            )

        # Then
        hooks.run.save.assert_called_once_with("test-file.txt", policy="now")

    @pytest.mark.asyncio
    async def test_files_not_saved_when_file_missing(self, mock_wandb_run: Run) -> None:
        """Test that missing files are skipped with warning"""
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
            files=["missing-file.txt"]
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": True, "exception": None}}

        # When - mock Path.exists() to return False and capture logger
        with patch('inspect_wandb.models.hooks.Path.exists', return_value=False), \
             patch('inspect_wandb.models.hooks.logger') as mock_logger:
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=None,
                    logs=[]
                )
            )

        # Then
        hooks.run.save.assert_not_called()
        mock_logger.warning.assert_called_with("File or folder 'missing-file.txt' does not exist. Skipping wandb upload.")

    @pytest.mark.asyncio
    async def test_files_save_handles_exceptions(self, mock_wandb_run: Run) -> None:
        """Test that exceptions during file save are handled gracefully"""
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.run.save.side_effect = Exception("Upload failed")
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
            files=["test-file.txt"]
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": True, "exception": None}}

        # When - mock Path.exists() to return True and capture logger
        with patch('inspect_wandb.models.hooks.Path.exists', return_value=True), \
             patch('inspect_wandb.models.hooks.logger') as mock_logger:
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=None,
                    logs=[]
                )
            )

        # Then
        hooks.run.save.assert_called_once_with("test-file.txt", policy="now")
        mock_logger.warning.assert_called_with("Failed to save test-file.txt to wandb: Upload failed")

    @pytest.mark.asyncio
    async def test_multiple_files_mixed_existence(self, mock_wandb_run: Run) -> None:
        """Test handling multiple files with mixed existence"""
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project",
            files=["existing-file.txt", "missing-file.txt", "another-existing.txt"]
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": True, "exception": None}}

        # When - mock Path constructor to control exists() method
        def mock_path_constructor(file_str):
            mock_path = MagicMock()
            mock_path.exists.return_value = file_str in ["existing-file.txt", "another-existing.txt"]
            return mock_path

        with patch('inspect_wandb.models.hooks.Path', side_effect=mock_path_constructor), \
             patch('inspect_wandb.models.hooks.logger') as mock_logger:
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=None,
                    logs=[]
                )
            )

        # Then
        assert hooks.run.save.call_count == 2
        hooks.run.save.assert_any_call("existing-file.txt", policy="now")
        hooks.run.save.assert_any_call("another-existing.txt", policy="now")
        mock_logger.warning.assert_called_with("File or folder 'missing-file.txt' does not exist. Skipping wandb upload.")

    @pytest.mark.asyncio
    async def test_wandb_run_url_added_to_eval_metadata(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart]) -> None:
        """Test wandb_run_url is added to eval metadata"""
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = False
        hooks._active_runs = {"test-run": {"running": True, "exception": None}}

        task_start = create_task_start()

        # When
        await hooks.on_task_start(
            task_start
        )

        # Then
        assert task_start.spec.metadata["wandb_run_url"] == "mock_wandb_url"

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_exception_finishes_with_exit_code_1(self, mock_wandb_run: Run) -> None:
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": False, "exception": None}}

        # When
        with patch('inspect_wandb.models.hooks.logger') as mock_logger:
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=KeyboardInterrupt(),
                    logs=[]
                )
            )

        # Then
        hooks.run.finish.assert_called_once_with(exit_code=1)
        mock_logger.error.assert_called_with("Inspect exited due to KeyboardInterrupt")

    @pytest.mark.asyncio
    async def test_system_exit_exception_finishes_with_exit_code_3(self, mock_wandb_run: Run) -> None:
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": False, "exception": None}}

        # When
        with patch('inspect_wandb.models.hooks.logger') as mock_logger:
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=SystemExit(5),
                    logs=[]
                )
            )

        # Then
        hooks.run.finish.assert_called_once_with(exit_code=3)
        mock_logger.error.assert_called_with("SystemExit running eval set: 5")

    @pytest.mark.asyncio
    async def test_general_exception_when_last_run_finishes_with_exit_code_2(self, mock_wandb_run: Run) -> None:
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": False, "exception": None}}

        # When
        with patch('inspect_wandb.models.hooks.logger') as mock_logger:
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=ValueError("Test error"),
                    logs=[]
                )
            )

        # Then
        hooks.run.finish.assert_called_once_with(exit_code=2)
        mock_logger.error.assert_called_with("Inspect exited due to exception")

    @pytest.mark.asyncio
    async def test_failed_tasks_when_last_run_finishes_with_exit_code_4(self, mock_wandb_run: Run) -> None:
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": False, "exception": None}}

        mock_failed_log = MagicMock()
        mock_failed_log.status = "failed"
        
        # When
        with patch('inspect_wandb.models.hooks.logger') as mock_logger:
            await hooks.on_run_end(
                RunEnd(
                    eval_set_id=None,
                    run_id="test-run",
                    exception=None,
                    logs=[mock_failed_log]
                )
            )

        # Then
        hooks.run.finish.assert_called_once_with(exit_code=4)
        mock_logger.warning.assert_called_with("One or more tasks failed, may retry if eval-set")

    @pytest.mark.asyncio
    async def test_successful_completion_when_last_run_finishes_with_exit_code_0(self, mock_wandb_run: Run) -> None:
        # Given
        hooks = WandBModelHooks()
        hooks.run = mock_wandb_run
        hooks.settings = ModelsSettings(
            enabled=True, 
            entity="test-entity", 
            project="test-project"
        )
        hooks._hooks_enabled = True
        hooks._wandb_initialized = True
        hooks._active_runs = {"test-run": {"running": False, "exception": None}}

        mock_success_log = MagicMock()
        mock_success_log.status = "success"
        
        # When
        await hooks.on_run_end(
            RunEnd(
                eval_set_id=None,
                run_id="test-run",
                exception=None,
                logs=[mock_success_log]
            )
        )

        # Then
        hooks.run.finish.assert_called_once_with(exit_code=0)

    @pytest.mark.parametrize("metadata_key", [
        "INSPECT_WANDB_MODELS_ENABLED",
        "inspect_wandb_models_enabled",
        "iNsPecT_wAnDb_MoDeLs_EnAbLeD",
    ])
    def parse_settings_from_metadata_is_case_insensitive(self, create_task_start: Callable[dict | None, TaskStart], metadata_key: str) -> None:
        """Test that parse_settings_from_metadata is case insensitive"""
        # Given
        hooks = WandBModelHooks()
        metadata = create_task_start({
            metadata_key: True,
        })
        
        
        # When
        settings = hooks._extract_settings_overrides_from_eval_metadata(metadata)

        # Then
        assert settings is not None
        assert settings["enabled"] is True
