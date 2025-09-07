from inspect_wandb.models.hooks import WandBModelHooks
from inspect_wandb.config.settings import ModelsSettings
from unittest.mock import patch, MagicMock
import pytest
from wandb.sdk.wandb_run import Run
from wandb.sdk.wandb_config import Config
from wandb.sdk.wandb_summary import Summary
from typing import Callable
from inspect_ai.hooks import TaskStart, SampleEnd, RunEnd, TaskEnd
from inspect_ai.log import EvalSample, EvalLog
from inspect_ai.scorer import Score 
from inspect_wandb.models.hooks import Metric

@pytest.fixture(scope="function")
def mock_wandb_run() -> Run:
    mock_run = MagicMock(spec=Run)
    mock_run.config = MagicMock(spec=Config)
    mock_run.config.update = MagicMock()
    mock_run.define_metric = MagicMock()
    mock_run.tags = []
    mock_run.summary = MagicMock(spec=Summary)
    mock_run.summary.update = MagicMock()
    mock_run.save = MagicMock()
    mock_run.finish = MagicMock()
    return mock_run

class TestWandBModelHooks:
    """
    Tests for the WandBModelHooks class.
    """

    def test_enabled(self) -> None:
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
        
        with patch('inspect_wandb.models.hooks.SettingsLoader.load_inspect_wandb_settings') as mock_loader:
            mock_loader.return_value.models = disabled_settings
            hooks = WandBModelHooks()
            assert not hooks.enabled()

    @pytest.mark.asyncio
    async def test_wandb_initialised_on_task_start(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart]) -> None:
        """
        Test that the on_task_start method initializes the WandB run.
        """
        hooks = WandBModelHooks()
        mock_init = MagicMock(return_value=mock_wandb_run)
        task_start = create_task_start()
        with patch('inspect_wandb.models.hooks.wandb.init', mock_init):
            await hooks.on_task_start(task_start)

            mock_init.assert_called_once_with(id="test_eval_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_not_called()
            hooks.run.define_metric.assert_called_once_with(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)
            assert hooks.run.tags == ("inspect_task:test_task", "inspect_model:mockllm/model", "inspect_dataset:test-dataset")

    @pytest.mark.asyncio
    async def test_wandb_config_updated_on_task_start_if_settings_config_is_set(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart]) -> None:
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

            mock_init.assert_called_once_with(id="test_eval_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_called_once_with({"test": "test"})
            hooks.run.define_metric.assert_called_once_with(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)
            assert hooks.run.tags == ("inspect_task:test_task", "inspect_model:mockllm/model", "inspect_dataset:test-dataset")

    @pytest.mark.asyncio
    async def test_wandb_init_called_with_eval_set_log_dir_if_eval_set(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart]) -> None:
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

            mock_init.assert_called_once_with(id="test_eval_set_log_dir", name="Inspect eval-set: test_eval_set_log_dir", entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True

    @pytest.mark.asyncio
    async def test_wandb_config_updated_with_eval_metadata(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart]) -> None:
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
            mock_init.assert_called_once_with(id="test_eval_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_called_once_with({"test": "test"})

    @pytest.mark.asyncio
    async def test_wandb_config_not_updated_with_eval_metadata_if_add_metadata_to_config_is_false(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart]) -> None:
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
            mock_init.assert_called_once_with(id="test_eval_id", name=None, entity="test-entity", project="test-project", resume="allow")
            assert hooks._wandb_initialized is True
            assert hooks.run is mock_wandb_run
            hooks.run.config.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_wandb_tags_updated_on_task_start_if_settings_tags_are_set(self, mock_wandb_run: Run, create_task_start: Callable[dict | None, TaskStart]) -> None:
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

            mock_init.assert_called_once_with(id="test_eval_id", name=None, entity="test-entity", project="test-project", resume="allow")
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
    async def test_wandb_run_url_added_to_eval_metadata(self, mock_wandb_run: Run, task_end_eval_log: EvalLog) -> None:
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
        hooks._wandb_initialized = True
        hooks.run.url = "test_url"

        # When
        await hooks.on_task_end(
            TaskEnd(
                eval_set_id=None,
                run_id="test_run_id",
                eval_id="test_eval_id",
                log=task_end_eval_log
            )
        )

        # Then
        assert task_end_eval_log.eval.metadata["wandb_run_url"] == "test_url"