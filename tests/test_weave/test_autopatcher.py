from inspect_ai import task, Task, eval
from inspect_ai.solver import generate
from inspect_ai.scorer import exact
from inspect_ai.dataset import Sample
from weave.trace.weave_client import WeaveClient
from typing import Generator
from pytest import MonkeyPatch
import pytest
from unittest.mock import MagicMock, patch
from ..conftest_weave_client import TEST_ENTITY

@pytest.fixture(scope="function")
def patch_weave_client_in_hooks(client: WeaveClient) -> Generator[WeaveClient, None, None]:
    with patch("inspect_wandb.weave.hooks.weave.init", MagicMock(return_value=client)):
        yield client


def test_inspect_quickstart(
    patch_weave_client_in_hooks: WeaveClient,
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

    calls = list(patch_weave_client_in_hooks.calls())
    
    # Debug: Print all calls to understand what the 9 calls are
    print(f"\n=== Found {len(calls)} calls ===")
    for i, call in enumerate(calls):
        print(f"{i+1}. {call._op_name}")
        if hasattr(call, 'attributes') and call.attributes:
            print(f"   Attributes: {call.attributes}")
        print()
    
    assert len(calls) == 9  # Updated to account for scorer tracing
    for call in calls:
        # this checks all calls were made to mock client
        assert TEST_ENTITY in call._op_name

    # check for inspect AI patched calls
    assert "sample" in calls[1]._op_name
    assert "inspect_ai-generate" in calls[2]._op_name

    # reset the env variables
    monkeypatch.delenv("INSPECT_WANDB_MODELS_ENABLED")
    monkeypatch.delenv("INSPECT_WANDB_WEAVE_ENABLED")
    monkeypatch.delenv("INSPECT_WANDB_WEAVE_AUTOPATCH")