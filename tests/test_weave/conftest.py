import pytest
from unittest.mock import MagicMock, patch
from weave.trace.weave_client import WeaveClient
from typing import Generator
from weave.trace.weave_client import Op, Call, ObjectRef
from concurrent.futures import Future
from typing import Any, Callable
from uuid import uuid4

class TestCall:

    name: str
    inputs: dict[str, Any]
    attributes: dict[str, Any] | None

    def __init__(
        self,
        name: str,
        inputs: dict[str, Any],
        attributes: dict[str, Any] | None = None
    ):
        self.name = name
        self.inputs = inputs
        self.attributes = attributes

    def __repr__(self):
        return f"TestCall(name={self.name}, inputs={self.inputs}, attributes={self.attributes})"

class WeaveTestClient(MagicMock):
    """
    A mocked-out weave client for testing.
    Some methods for creating calls are overriden from the base mock, to ensure that we capture autopatched calls correctly
    """
    def __init__(self, *args, **kwargs):
        super().__init__(spec=WeaveClient, *args, **kwargs)
        self.calls: dict[str, TestCall] = {}

    def create_call(
        self, 
        op: str | Op,
        inputs: dict[str, Any],
        parent: Call | None = None,
        attributes: dict[str, Any] | None = None,
        display_name: str | Callable[[Call], str] | None = None,
        *args,
        **kwargs
    ) -> Call:
        if isinstance(op, str):
            op_name = op
        else:
            op_name = op.name
        self.calls[op_name] = TestCall(name=op_name, inputs=inputs, attributes=attributes)

        # returning an actual call rather than a mock here is required to ensure downstream calls are passed correctly to the test client
        mock_call = Call(
            _op_name=op_name,
            project_id="test_project",
            trace_id=uuid4(),
            parent_id=parent.trace_id if parent else None,
            inputs=inputs,
            attributes=attributes,
        )

        return mock_call
    
    def _send_score_call(
        self,
        predict_call: Call,
        score_call: Call,
        scorer_object_ref: ObjectRef | None = None,
    ) -> Future[str]:
        self.calls[score_call._op_name] = TestCall(
            name=score_call._op_name,
            inputs=score_call.inputs,
            attributes=score_call.attributes
        )
        return MagicMock(spec=Future[str])

    def get_test_calls(self) -> list[TestCall]:
        return self.calls.values()
        


@pytest.fixture(scope="function")
def weave_test_client() -> Generator[WeaveTestClient, None, None]:
    """
    A mocked-out weave client for testing.
    We patch the context method from weave so that autopatching will use this client.
    """
    client = WeaveTestClient()
    with patch("weave.trace.context.weave_client_context.get_weave_client", return_value=client):
        yield client
