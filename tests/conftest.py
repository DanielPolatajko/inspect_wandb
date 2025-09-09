import pytest
import configparser
import os
from typing import Callable, Generator
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, Solver, TaskState, Generate, solver
from unittest.mock import MagicMock, PropertyMock
import inspect_ai.hooks._startup as hooks_startup_module
from unittest.mock import patch
from inspect_wandb.providers import weave_evaluation_hooks
from pytest import TempPathFactory
from inspect_ai._util.registry import registry_find
from weave.evaluation.eval_imperative import EvaluationLogger
from inspect_ai.hooks import TaskStart
from inspect_ai.log import EvalSpec, EvalConfig, EvalDataset, EvalLog, EvalResults, EvalScore, EvalMetric
from datetime import datetime

## Setup wandb directory and settings

@pytest.fixture(scope="session")
def wandb_path(tmp_path_factory: TempPathFactory) -> Path:
    """
    Returns the path to the wandb directory.
    """
    path = tmp_path_factory.mktemp("test")
    os.makedirs(path / "wandb")
    os.chdir(path)
    return path / "wandb"

@pytest.fixture(scope="session", autouse=True)
def initialise_wandb(wandb_path: Path) -> None:
    """
    Writes a wandb settings file to the tmp_path directory, and changes the current working directory to the tmp_path.
    """
    config = configparser.ConfigParser()
    config["default"] = {
        "entity": "test-entity",
        "project": "test-project"
    }
    with open(wandb_path / "settings", "w") as f:
        config.write(f)


## Mock wandb/weave client calls

@pytest.fixture(scope="function", autouse=True)
def patch_wandb_client() -> Generator[tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock], None, None]:
    mock_config = MagicMock()
    mock_config.update = MagicMock()
    mock_summary = MagicMock()
    mock_summary.update = MagicMock()
    mock_log = MagicMock()
    mock_save = MagicMock()
    
    # Create a mock run that wandb.init() will return
    mock_run = MagicMock()
    mock_run.config = mock_config
    mock_run.summary = mock_summary
    mock_run.define_metric = MagicMock()
    mock_run.tags = []
    mock_run.save = MagicMock()
    mock_run.finish = MagicMock()
    
    # Mock the url property using PropertyMock
    type(mock_run).url = PropertyMock(return_value="mock_wandb_url")
    
    mock_wandb_init = MagicMock(return_value=mock_run)
    
    with (
        patch("inspect_wandb.models.hooks.wandb.init", mock_wandb_init),
        patch("inspect_wandb.models.hooks.wandb.save", mock_save),
        patch("inspect_wandb.models.hooks.wandb.config", mock_config),
        patch("inspect_wandb.models.hooks.wandb.summary", mock_summary),
        patch("inspect_wandb.models.hooks.wandb.log", mock_log)
    ):
        yield mock_wandb_init, mock_save, mock_config, mock_summary, mock_log

@pytest.fixture(scope="function")
def reset_inspect_ai_hooks() -> Generator[None, None, None]:
    hooks_startup_module._registry_hooks_loaded = False
    yield
    # reload settings for every test
    hooks = registry_find(lambda x: x.type == "hooks")
    if hooks:
        for hook in hooks:
            hook.settings = None # type: ignore
            # Reset our new state variables to ensure clean test isolation
            if hasattr(hook, '_hooks_enabled'):
                hook._hooks_enabled = None
            if hasattr(hook, '_weave_initialized'):
                hook._weave_initialized = False
            if hasattr(hook, '_wandb_initialized'):
                hook._wandb_initialized = False

@pytest.fixture(scope="function")
def patched_weave_evaluation_hooks(reset_inspect_ai_hooks: None):
    patched_evaluation_logger_class = MagicMock(spec=EvaluationLogger)
    patched_evaluation_logger_class.return_value = patched_evaluation_logger_class
    patched_evaluation_logger_class._is_finalized = False
    patched_evaluation_logger_class.finish = MagicMock(side_effect=lambda *args, **kwargs: setattr(patched_evaluation_logger_class, "_is_finalized", True))
    patched_evaluation_logger_class.log_summary = MagicMock()
    patched_evaluation_logger_class.log_prediction = MagicMock()
    patched_evaluation_logger_class._evaluate_call = MagicMock()

    with (
        patch("inspect_wandb.weave.hooks.weave.init", MagicMock()) as weave_init,
        patch("inspect_wandb.weave.hooks.weave.finish", MagicMock()) as weave_finish,
        patch("inspect_wandb.weave.hooks.CustomEvaluationLogger", patched_evaluation_logger_class)
    ):
        weave_evaluation_hooks_instance = weave_evaluation_hooks() # type: ignore
        with patch("inspect_wandb._registry.weave_evaluation_hooks", lambda: weave_evaluation_hooks_instance):
            yield {
                "weave_evaluation_hooks": weave_evaluation_hooks_instance,
                "weave_init": weave_init,
                "weave_finish": weave_finish,
                "weave_evaluation_logger": patched_evaluation_logger_class
            }

@pytest.fixture(scope="function")
def hello_world_eval() -> Callable[[], Task]:
    """
    Returns a mock Inspect eval plus a set of mocks that can be used to check that Weave was called correctly.
    """
    @task
    def hello_world():
        return Task(
            dataset=[
                Sample(
                    input="Just reply with Hello World",
                    target="Hello World",
                )
            ],
            solver=[generate()],
            scorer=exact(),
            metadata={"test": "test"},
            name="hello_world_eval"
        )

    return hello_world


@solver
def raise_error() -> Solver:
    """Raises an error."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raise RuntimeError("Simulated failure")

    return solve

@pytest.fixture(scope="function")
def error_eval() -> Callable[[], Task]:
    """
    Returns a mock Inspect eval plus a set of mocks that can be used to check that Weave was called correctly.
    """
    @task
    def hello_world():
        return Task(
            dataset=[
                Sample(
                    input="Just reply with Hello World",
                    target="Hello World",
                )
            ],
            solver=[raise_error()],
            scorer=exact(),
            metadata={"test": "test"},
            name="hello_world_eval"
        )

    return hello_world


### Inspect Hooks DTOs

@pytest.fixture(scope="function")
def create_task_start() -> Callable[[dict | None], TaskStart]:
    """Helper to create TaskStart with optional metadata"""
    def _create_task_start(metadata: dict | None = None) -> TaskStart:
        return TaskStart(
            eval_set_id="test_eval_set_id",
            run_id="test_run_id",
            eval_id="test_eval_id",
            spec=EvalSpec(
                run_id="test_run_id",
                task_id="test_task_id", 
                created=datetime.now().isoformat(),
                task="test_task",
                dataset=EvalDataset(name="test-dataset"),
                model="mockllm/model",
                config=EvalConfig(),
                metadata=metadata
            )
        )
    return _create_task_start

@pytest.fixture(scope="function")
def task_end_eval_log() -> EvalLog:
    return EvalLog(
        eval=EvalSpec(
            eval_set_id="test_eval_set_id",
            run_id="test_run_id",
            task_id="test_task_id",
            created=datetime.now().isoformat(),
            task="test_task",
            dataset=EvalDataset(),
            model="mockllm/model",
            config=EvalConfig()
        ),
        results=EvalResults(
            total_samples=1,
            scores=[
                EvalScore(
                    name="test_score",
                    scorer="test_scorer",
                    metrics={"test_metric": EvalMetric(name="test_metric", value=1.0)}
                )
            ]
        )
    )


