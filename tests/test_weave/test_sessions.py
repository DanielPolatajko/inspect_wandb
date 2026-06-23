from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.scorer import Score

from inspect_wandb.weave.sessions import (
    AgentSessionEmitter,
    build_outcome,
    flatten_metadata,
    llm_span_attrs,
    to_messages,
    tool_span_attrs,
    usage_attrs,
    usage_from_event,
)

T0 = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 6, 21, 12, 0, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 6, 21, 12, 0, 2, tzinfo=timezone.utc)


def make_model_event(input_tokens: int = 100, output_tokens: int = 20) -> ModelEvent:
    return ModelEvent(
        model="anthropic/claude-haiku-4-5",
        input=[ChatMessageUser(content="solve the task")],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(temperature=0.5, max_tokens=1024, top_k=40),
        output=ModelOutput(
            model="anthropic/claude-haiku-4-5",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content="thinking"),
                    stop_reason="tool_calls",
                )
            ],
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        ),
        timestamp=T0,
        completed=T1,
    )


def make_tool_event(tool_id: str = "call_1", function: str = "bash") -> ToolEvent:
    return ToolEvent(
        id=tool_id,
        function=function,
        arguments={"cmd": "ls -la"},
        result="command output",
        timestamp=T1,
        completed=T2,
    )


class TestPureBuilders:
    def test_to_messages_maps_roles(self) -> None:
        # Given
        messages = [ChatMessageUser(content="hi"), ChatMessageAssistant(content="yo")]

        # When
        result = to_messages(messages)

        # Then
        assert [(m.role, m.content) for m in result] == [
            ("user", "hi"),
            ("assistant", "yo"),
        ]

    def test_llm_span_attrs_has_usage_provider_and_inspect_extras(self) -> None:
        # Given
        event = make_model_event(input_tokens=321, output_tokens=99)

        # When
        attrs = llm_span_attrs(event, conversation_id="sess-1")

        # Then
        assert attrs["gen_ai.request.model"] == "anthropic/claude-haiku-4-5"
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.usage.input_tokens"] == 321
        assert attrs["gen_ai.usage.output_tokens"] == 99
        assert attrs["gen_ai.request.temperature"] == 0.5
        assert attrs["inspect.generate.top_k"] == 40

    def test_tool_span_attrs_truncates_and_adds_inspect_extras(self) -> None:
        # Given
        event = make_tool_event()
        event.result = "x" * 10000
        event.working_time = 1.5

        # When
        attrs = tool_span_attrs(event, conversation_id="sess-1")

        # Then
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["inspect.tool.working_time"] == 1.5
        assert any("…[truncated]" in str(v) for v in attrs.values())

    def test_usage_rollup_sums_and_builds_gen_ai_keys(self) -> None:
        # Given
        e1 = make_model_event(input_tokens=100, output_tokens=10)
        e2 = make_model_event(input_tokens=50, output_tokens=5)

        # When
        from inspect_wandb.weave.sessions import _add_usage

        total = _add_usage(usage_from_event(e1), usage_from_event(e2))
        attrs = usage_attrs(total)

        # Then
        assert attrs["gen_ai.usage.input_tokens"] == 150
        assert attrs["gen_ai.usage.output_tokens"] == 15

    def test_flatten_metadata(self) -> None:
        # Given
        metadata = {"difficulty": "hard", "category": "crypto"}

        # When
        result = flatten_metadata(metadata)

        # Then
        assert result == {"metadata.difficulty": "hard", "metadata.category": "crypto"}

    def test_build_outcome_includes_scores_timing_and_tokens(self) -> None:
        # Given
        sample = SimpleNamespace(
            total_time=12.3,
            working_time=10.1,
            error=None,
            limit=None,
            scores={"includes": Score(value=1.0, answer="73")},
            model_usage={"anthropic/claude-haiku-4-5": ModelUsage(total_tokens=7815)},
        )

        # When
        outcome = build_outcome(sample)

        # Then
        assert outcome["total_time"] == 12.3
        assert outcome["score.includes"] == 1.0
        assert outcome["score.includes.answer"] == "73"
        assert outcome["total_tokens"] == 7815


class TestAgentSessionEmitter:
    def _run(self, events: list, outcome: dict | None = None) -> list:
        recorded: list = []

        def fake_emit(tracer, name, parent_ctx, start_ns, end_ns, attrs):  # noqa: ANN001
            recorded.append((name, dict(attrs)))
            return MagicMock()

        emitter = AgentSessionEmitter(
            session_id="sess-uuid",
            session_name="task-sample-1",
            agent_name="my_task",
            model="anthropic/claude-haiku-4-5",
            identity={"task": "my_task", "sample_id": 1},
        )
        with (
            patch("inspect_wandb.weave.sessions._emit_span", side_effect=fake_emit),
            patch(
                "inspect_wandb.weave.sessions.set_span_in_context", return_value=None
            ),
            patch(
                "inspect_wandb.weave.sessions.otel_trace.get_tracer",
                return_value=MagicMock(),
            ),
        ):
            for event in events:
                emitter.handle_event(event)
            emitter.finish(outcome)
        return recorded

    def test_segments_turns_and_rolls_up_usage(self) -> None:
        # Given
        events = [
            make_model_event(input_tokens=100, output_tokens=10),
            make_tool_event(tool_id="a"),
            make_model_event(input_tokens=50, output_tokens=5),
            make_tool_event(tool_id="b"),
        ]

        # When
        recorded = self._run(events)

        # Then
        turns = [(n, a) for n, a in recorded if n.startswith("invoke_agent")]
        assert len(turns) == 2
        assert turns[0][1]["gen_ai.usage.input_tokens"] == 100
        assert turns[0][1]["inspect.turn_index"] == 0
        assert turns[1][1]["inspect.turn_index"] == 1
        assert turns[0][1]["inspect.task"] == "my_task"

    def test_final_turn_carries_outcome(self) -> None:
        # Given
        events = [make_model_event(), make_tool_event()]

        # When
        recorded = self._run(events, outcome={"score.includes": 1.0, "total_time": 5.0})

        # Then
        turns = [(n, a) for n, a in recorded if n.startswith("invoke_agent")]
        assert turns[-1][1]["inspect.score.includes"] == 1.0
        assert turns[-1][1]["inspect.total_time"] == 5.0

    def test_emit_failure_is_swallowed(self) -> None:
        # Given
        emitter = AgentSessionEmitter(
            session_id="s",
            session_name="n",
            agent_name="a",
            model="m",
            identity={},
        )

        # When / Then
        with patch(
            "inspect_wandb.weave.sessions._emit_span",
            side_effect=RuntimeError("otel down"),
        ):
            emitter.handle_event(make_model_event())
            emitter.finish()  # must not raise

    def test_tool_before_model_is_ignored(self) -> None:
        # Given / When
        recorded = self._run([make_tool_event()])

        # Then
        assert recorded == []
