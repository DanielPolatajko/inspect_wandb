import pytest
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch
from inspect_ai import Task, eval_set, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import Solver, TaskState, Generate, solver, generate
from inspect_wandb.config.settings import ModelsSettings, WeaveSettings
from inspect_ai._util.registry import registry_find
import inspect_ai.hooks._startup as hooks_startup_module
from inspect_wandb.shared.utils import format_wandb_id_string
from uuid import uuid4
from typing import Sequence

@solver
def failing_solver_that_retries(fail: Sequence[bool]) -> Solver:
    """Solver that fails on first few attempts but succeeds eventually."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        should_fail = iter(fail)
        if should_fail:
            raise ValueError("Simulated failure attempt")
        return state
    
    return solve


@pytest.fixture(scope="function")
def failing_retry_eval() -> Callable[[], Task]:
    """Returns an eval that fails initially but succeeds on retry."""
    @task
    def retry_task():
        failures = [True, False]
        return Task(
            dataset=[
                Sample(
                    input="Just reply with Hello World",
                    target="Hello World",
                )
            ],
            solver=[failing_solver_that_retries(failures), generate()],
            scorer=exact(),
            metadata={"test": "retry_scenario"},
            name="hello_world_eval"
        )
    
    return retry_task

class TestWandBModelHooksE2ERetryScenarios:
    """E2E tests for wandb model hooks with actual eval_set calls covering retry and rerun scenarios."""

    def test_wandb_uses_same_run_id_across_retries_with_eval_set(
        self,
        failing_retry_eval: Callable[[], Task],
        patch_wandb_client: tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock],
        tmp_path: Path,
        reset_inspect_ai_hooks: None,
    ) -> None:
        """
        Test that when hooks are enabled and eval_set is run with retry_attempts,
        if there are task failures that get retried, all data gets logged to the same wandb run.
        """

        # Given
        mock_wandb_init, _, _, _, _ = patch_wandb_client
        mock_run = MagicMock()
        mock_run.config = MagicMock()
        mock_run.config.update = MagicMock()
        mock_run.define_metric = MagicMock()
        mock_run.tags = []
        mock_run.summary = MagicMock()
        mock_run.save = MagicMock() 
        mock_run.finish = MagicMock()
        mock_run.log = MagicMock()
        mock_run.url = "https://wandb.ai/test/test-project/runs/test_run"
        mock_wandb_init.return_value = mock_run
        
        with patch('inspect_wandb.models.hooks.SettingsLoader.load_inspect_wandb_settings') as mock_loader:
            enabled_settings = ModelsSettings(
                enabled=True,
                entity="test-entity",
                project="test-project"
            )
            mock_loader.return_value.models = enabled_settings
            mock_loader.return_value.weave = WeaveSettings(
                enabled=False,
                entity="test-entity",
                project="test-project"
            )

            # When
            uid = str(uuid4())
            logs = eval_set(
                tasks=[failing_retry_eval()],
                log_dir=str(tmp_path / uid),
                retry_attempts=2,
                retry_wait=1.0,
                model="mockllm/model",
                display="plain"
            )
            
            # Then
            assert mock_wandb_init.call_count == 2
            
            eval_set_ids = [call[1]['id'] for call in mock_wandb_init.call_args_list]

            assert len(set(eval_set_ids)) == 1
            
            assert eval_set_ids[0] == format_wandb_id_string(str(tmp_path / uid))
            
            assert len(logs) == 2

    def test_wandb_uses_same_run_id_when_rerunning_same_log_dir(
        self,
        error_eval: Callable[[], Task],
        hello_world_eval: Callable[[], Task],
        patch_wandb_client: tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock],
        tmp_path: Path,
        reset_inspect_ai_hooks: None,
    ) -> None:
        """
        Test that when user reruns eval_set with the same log_dir,
        the wandb run_id should be the same (simulating accidental cancel + rerun).
        """

        # Given
        mock_wandb_init, _, _, _, _ = patch_wandb_client
        mock_run = MagicMock()
        mock_run.config = MagicMock()
        mock_run.config.update = MagicMock()
        mock_run.define_metric = MagicMock()
        mock_run.tags = []
        mock_run.summary = MagicMock()
        mock_run.save = MagicMock()
        mock_run.finish = MagicMock()
        mock_run.log = MagicMock()
        mock_run.url = "https://wandb.ai/test/test-project/runs/test_run"
        mock_wandb_init.return_value = mock_run
        
        with patch('inspect_wandb.models.hooks.SettingsLoader.load_inspect_wandb_settings') as mock_loader:
            enabled_settings = ModelsSettings(
                enabled=True,
                entity="test-entity",
                project="test-project"
            )
            mock_loader.return_value.models = enabled_settings
            mock_loader.return_value.weave = WeaveSettings(
                enabled=False,
                entity="test-entity",
                project="test-project"
            )

            # When            
            logs1 = eval_set(
                tasks=[error_eval()],
                log_dir=str(tmp_path),
                model="mockllm/model",
                retry_attempts=0,
            )
            
            first_run_id = mock_wandb_init.call_args_list[0][1]['id']
            
            # Reset hooks state to allow second initialization                
            hooks_startup_module._registry_hooks_loaded = False
            hooks = registry_find(lambda x: x.type == "hooks")
            if hooks:
                for hook in hooks:
                    hook.settings = None
                    if hasattr(hook, '_hooks_enabled'):
                        hook._hooks_enabled = None
                    if hasattr(hook, '_wandb_initialized'):
                        hook._wandb_initialized = False
            
            mock_run.reset_mock()
            
            logs2 = eval_set(
                tasks=[hello_world_eval()],
                log_dir=str(tmp_path),
                model="mockllm/model",
                retry_attempts=0,
            )
            
            # Then
            second_run_id = mock_wandb_init.call_args_list[0][1]['id']
            assert first_run_id == second_run_id, f"Expected same run_id for same log_dir, got {first_run_id} != {second_run_id}"
            
            assert len(logs1) == 2
            assert len(logs2) == 2

    def test_wandb_uses_different_run_id_for_different_log_dirs(
        self,
        hello_world_eval: Callable[[], Task],
        patch_wandb_client: tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock],
        tmp_path: Path,
        reset_inspect_ai_hooks: None,
    ) -> None:
        """
        Test that different log_dirs result in different wandb run_ids.
        This validates that our same-log_dir test is meaningful.
        """

        # Given
        mock_wandb_init, _, _, _, _ = patch_wandb_client
        mock_run = MagicMock()
        mock_run.config = MagicMock()
        mock_run.config.update = MagicMock()
        mock_run.define_metric = MagicMock()
        mock_run.tags = []
        mock_run.summary = MagicMock()
        mock_run.save = MagicMock()
        mock_run.finish = MagicMock()
        mock_run.log = MagicMock()
        mock_run.url = "https://wandb.ai/test/test-project/runs/test_run"
        mock_wandb_init.return_value = mock_run
        
        temp_dir1 = tmp_path / "dir1"
        temp_dir2 = tmp_path / "dir2"
        temp_dir1.mkdir()
        temp_dir2.mkdir()
        
        with patch('inspect_wandb.models.hooks.SettingsLoader.load_inspect_wandb_settings') as mock_loader:
            enabled_settings = ModelsSettings(
                enabled=True,
                entity="test-entity", 
                project="test-project"
            )
            mock_loader.return_value.models = enabled_settings
            mock_loader.return_value.weave = WeaveSettings(
                enabled=False,
                entity="test-entity",
                project="test-project"
            )

            # When
            eval_set(
                tasks=[hello_world_eval()],
                log_dir=str(temp_dir1),
                model="mockllm/model",
            )
            
            first_run_id = mock_wandb_init.call_args_list[0][1]['id']
            
            hooks_startup_module._registry_hooks_loaded = False
            hooks = registry_find(lambda x: x.type == "hooks")
            if hooks:
                for hook in hooks:
                    hook.settings = None
                    if hasattr(hook, '_hooks_enabled'):
                        hook._hooks_enabled = None
                    if hasattr(hook, '_wandb_initialized'):
                        hook._wandb_initialized = False
            
            mock_wandb_init.reset_mock()
            
            eval_set(
                tasks=[hello_world_eval()],
                log_dir=str(temp_dir2),
                model="mockllm/model"
            )
            
            # Then
            second_run_id = mock_wandb_init.call_args_list[0][1]['id']
            assert first_run_id != second_run_id, f"Expected different run_ids for different log_dirs, got {first_run_id} == {second_run_id}"

    def test_hooks_disabled_no_wandb_init_during_retries(
        self,
        failing_retry_eval: Callable[[], Task],
        patch_wandb_client: tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock],
        tmp_path: Path,
        reset_inspect_ai_hooks: None,
    ) -> None:
        """
        Test that when wandb model hooks are disabled, 
        no wandb initialization happens even during retries.
        """

        # Given
        mock_wandb_init, _, _, _, _ = patch_wandb_client
        
        with patch('inspect_wandb.models.hooks.SettingsLoader.load_inspect_wandb_settings') as mock_loader:
            disabled_settings = ModelsSettings(
                enabled=False,
                entity="test-entity",
                project="test-project"
            )
            mock_loader.return_value.models = disabled_settings
            mock_loader.return_value.weave = WeaveSettings(
                enabled=False,
                entity="test-entity",
                project="test-project"
            )

            # When
            logs = eval_set(
                tasks=[failing_retry_eval()],
                log_dir=str(tmp_path),
                retry_attempts=3,
                model="mockllm/model",
                fail_on_error=False
            )
            
            # Then
            mock_wandb_init.assert_not_called()
            assert len(logs) >= 1