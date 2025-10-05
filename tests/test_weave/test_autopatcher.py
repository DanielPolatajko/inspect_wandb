from inspect_ai import task, Task, eval
from inspect_ai.solver import generate
from inspect_ai.scorer import exact, match, Target
from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput
from typing import Generator
import pytest
from unittest.mock import MagicMock, patch
from .conftest import WeaveTestClient
from inspect_wandb.weave.autopatcher import PatchedScorer
from inspect_ai._util.registry import registry_info, is_registry_object, set_registry_info
from inspect_ai.scorer._metric import Score

@pytest.fixture(scope="function")
def patch_weave_client_in_hooks(weave_test_client: WeaveTestClient) -> Generator[WeaveTestClient, None, None]:
    with patch("inspect_wandb.weave.hooks.weave.init", MagicMock(return_value=weave_test_client)):
        yield weave_test_client


def test_inspect_quickstart(
    patch_weave_client_in_hooks: WeaveTestClient,
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
            metadata={"test": "test", "inspect_wandb_weave_enabled": "true", "inspect_wandb_models_enabled": "false"},
            display_name="test task",
            name="hello_world_autopatcher"
        )

    eval(hello_world, model="mockllm/model")

    calls = list(patch_weave_client_in_hooks.get_test_calls())

    assert len([call.name for call in calls]) == 9

    # check for inspect AI patched calls
    assert "sample" in calls[1].name
    assert "inspect_ai/generate" in calls[2].name
    assert "scorer_inspect_ai/exact" in calls[3].name


class TestPatchedScorerRegistryManagement:
    """Test suite for PatchedScorer's registry management functionality."""

    def test_patched_scorer_copies_registry_info_from_registered_scorer(self):
        """Test that PatchedScorer correctly copies registry information from registered scorers."""
        # Test with exact scorer
        exact_scorer = exact()
        assert is_registry_object(exact_scorer)
        
        patched_exact = PatchedScorer(exact_scorer)
        assert is_registry_object(patched_exact)
        assert registry_info(patched_exact).name == "inspect_ai/exact"
        assert patched_exact.scorer_name == "inspect_ai/exact"
        
        # Test with match scorer to verify separation
        match_scorer = match()
        patched_match = PatchedScorer(match_scorer)
        assert registry_info(patched_match).name == "inspect_ai/match"
        assert patched_match.scorer_name == "inspect_ai/match"
        
        # Verify scorers maintain separate registry info
        assert registry_info(patched_exact).name != registry_info(patched_match).name

    def test_patched_scorer_handles_non_registered_scorer(self):
        """Test that PatchedScorer handles non-registered scorers gracefully."""
        # Create a mock scorer that's not in the registry
        class MockScorer:
            async def __call__(self, state: TaskState, target: Target) -> Score:
                return Score(value=1.0, explanation="Mock score")
        
        mock_scorer = MockScorer()
        
        # Manually set registry info to simulate a scorer with registry info
        from inspect_ai._util.registry import RegistryInfo
        mock_info = RegistryInfo(name="test/mock_scorer", type="scorer", doc="Test scorer")
        set_registry_info(mock_scorer, mock_info)
        
        # Create patched scorer
        patched = PatchedScorer(mock_scorer)
        
        # Verify registry info was copied
        assert is_registry_object(patched)
        patched_info = registry_info(patched)
        assert patched_info.name == "test/mock_scorer"
        assert patched.scorer_name == "test/mock_scorer"

    def test_patched_scorer_preserves_original_scorer_reference(self):
        """Test that PatchedScorer maintains a reference to the original scorer."""
        original_scorer = exact()
        patched = PatchedScorer(original_scorer)

        assert patched.original_scorer is original_scorer


class TestPatchedScorerCall:
    """Test suite for PatchedScorer.__call__ functionality."""

    @pytest.mark.asyncio
    async def test_patched_scorer_call_with_sample_context(self):
        # Given
        scorer = exact()
        patched_scorer = PatchedScorer(scorer)

        state = TaskState(
            model="test_model",
            sample_id=1,
            epoch=1,
            input="test input",
            messages=[],
            output=ModelOutput.from_content(model="test_model", content="Hello World"),
            completed=False
        )
        target = Target("Hello World")

        mock_sample_call = MagicMock()
        mock_sample_call.id = "sample_call_123"
        mock_sample_call.attributes = {"sample_id": 1, "epoch": 1}

        mock_parent_call = MagicMock()
        mock_parent_call._children = [mock_sample_call]

        with patch("inspect_wandb.weave.autopatcher.call_context") as mock_call_context:
            mock_call_context.get_current_call.return_value = mock_parent_call

            # When
            result = await patched_scorer(state, target)

            # Then
            assert result is not None
            mock_call_context.push_call.assert_called_once_with(mock_sample_call)
            mock_call_context.pop_call.assert_called_once_with(mock_sample_call.id)

    @pytest.mark.asyncio
    async def test_patched_scorer_call_without_sample_context(self):
        # Given
        scorer = exact()
        patched_scorer = PatchedScorer(scorer)

        state = TaskState(
            model="test_model",
            sample_id=1,
            epoch=1,
            input="test input",
            messages=[],
            output=ModelOutput.from_content(model="test_model", content="Hello World"),
            completed=False
        )
        target = Target("Hello World")

        with patch("inspect_wandb.weave.autopatcher.call_context") as mock_call_context:
            mock_call_context.get_current_call.return_value = None

            # When
            result = await patched_scorer(state, target)

            # Then
            assert result is not None
            mock_call_context.push_call.assert_not_called()
            mock_call_context.pop_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_patched_scorer_call_pops_context_on_exception(self):
        # Given
        class FailingScorer:
            __name__ = "FailingScorer"

            async def __call__(self, state: TaskState, target: Target) -> Score:
                raise ValueError("Test error")

        failing_scorer = FailingScorer()

        from inspect_ai._util.registry import RegistryInfo
        mock_info = RegistryInfo(name="test/failing_scorer", type="scorer", doc="Test scorer")
        set_registry_info(failing_scorer, mock_info)

        patched_scorer = PatchedScorer(failing_scorer)

        state = TaskState(
            model="test_model",
            sample_id=1,
            epoch=1,
            input="test input",
            messages=[],
            output=ModelOutput.from_content(model="test_model", content="test"),
            completed=False
        )
        target = Target("test")

        mock_sample_call = MagicMock()
        mock_sample_call.id = "sample_call_123"
        mock_sample_call.attributes = {"sample_id": 1, "epoch": 1}

        mock_parent_call = MagicMock()
        mock_parent_call._children = [mock_sample_call]

        with patch("inspect_wandb.weave.autopatcher.call_context") as mock_call_context:
            mock_call_context.get_current_call.return_value = mock_parent_call

            # When/Then
            with pytest.raises(ValueError, match="Test error"):
                await patched_scorer(state, target)

            mock_call_context.push_call.assert_called_once_with(mock_sample_call)
            mock_call_context.pop_call.assert_called_once_with(mock_sample_call.id)

    @pytest.mark.asyncio
    async def test_patched_scorer_call_with_no_matching_sample(self):
        # Given
        scorer = exact()
        patched_scorer = PatchedScorer(scorer)

        state = TaskState(
            model="test_model",
            sample_id=99,
            epoch=5,
            input="test input",
            messages=[],
            output=ModelOutput.from_content(model="test_model", content="Hello World"),
            completed=False
        )
        target = Target("Hello World")

        mock_sample_call = MagicMock()
        mock_sample_call.id = "sample_call_123"
        mock_sample_call.attributes = {"sample_id": 1, "epoch": 1}

        mock_parent_call = MagicMock()
        mock_parent_call._children = [mock_sample_call]

        with patch("inspect_wandb.weave.autopatcher.call_context") as mock_call_context:
            mock_call_context.get_current_call.return_value = mock_parent_call

            # When
            result = await patched_scorer(state, target)

            # Then
            assert result is not None
            mock_call_context.push_call.assert_not_called()
            mock_call_context.pop_call.assert_not_called()

