from inspect_ai import task, Task, eval
from inspect_ai.solver import generate
from inspect_ai.scorer import exact
from inspect_ai.dataset import Sample
from typing import Generator
from pytest import MonkeyPatch
import pytest
from unittest.mock import MagicMock, patch
from .conftest import WeaveTestClient

@pytest.fixture(scope="function")
def patch_weave_client_in_hooks(weave_test_client: WeaveTestClient) -> Generator[WeaveTestClient, None, None]:
    with patch("inspect_wandb.weave.hooks.weave.init", MagicMock(return_value=weave_test_client)):
        yield weave_test_client


def test_inspect_quickstart(
    patch_weave_client_in_hooks: WeaveTestClient,
    monkeypatch: MonkeyPatch,
    reset_inspect_ai_hooks: None
) -> None:
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
            display_name="test task"
        )
    
    # configure settings via env variables
    monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "false")
    monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "true")
    monkeypatch.setenv("INSPECT_WANDB_WEAVE_AUTOPATCH", "true")

    eval(hello_world, model="mockllm/model")

    calls = list(patch_weave_client_in_hooks.get_test_calls())
    assert len([call.name for call in calls]) == 8

    # check for inspect AI patched calls
    assert "sample" in calls[1].name
    assert "inspect_ai/generate" in calls[2].name

    # reset the env variables
    monkeypatch.delenv("INSPECT_WANDB_MODELS_ENABLED")
    monkeypatch.delenv("INSPECT_WANDB_WEAVE_ENABLED")
    monkeypatch.delenv("INSPECT_WANDB_WEAVE_AUTOPATCH")