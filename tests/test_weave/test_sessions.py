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
    _coerce,
    _emit_span,
    build_outcome,
    flatten_metadata,
    llm_span_attrs,
    to_messages,
    tool_span_attrs,
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

    def test_include_content_false_drops_messages_keeps_usage(self) -> None:
        # Given
        event = make_model_event(input_tokens=100, output_tokens=20)

        # When
        attrs = llm_span_attrs(event, conversation_id="sess-1", include_content=False)

        # Then
        assert "gen_ai.input.messages" not in attrs
        assert "gen_ai.output.messages" not in attrs
        assert attrs["gen_ai.usage.input_tokens"] == 100

    def test_tool_include_content_false_drops_args_and_result(self) -> None:
        # Given
        event = make_tool_event()

        # When
        attrs = tool_span_attrs(event, conversation_id="sess-1", include_content=False)

        # Then
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert all("ls -la" not in str(v) for v in attrs.values())

    def test_flatten_metadata(self) -> None:
        # Given
        metadata = {"difficulty": "hard", "category": "crypto"}

        # When
        result = flatten_metadata(metadata)

        # Then
        assert result == {"metadata.difficulty": "hard", "metadata.category": "crypto"}

    def test_flatten_metadata_ignores_non_dict(self) -> None:
        # Given / When / Then
        assert flatten_metadata("not a dict") == {}

    def test_coerce_preserves_scalars_and_json_encodes_collections(self) -> None:
        # Given / When / Then
        assert _coerce(True) is True  # bool checked before int
        assert _coerce(5) == 5
        assert _coerce("x") == "x"
        assert _coerce(["a", "b"]) == '["a", "b"]'
        assert _coerce(None) is None

    def test_usage_from_event_handles_missing_usage(self) -> None:
        # Given
        event = make_model_event()
        event.output.usage = None

        # When
        usage = usage_from_event(event)

        # Then
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_emit_span_sets_attrs_skips_empty_and_ends(self) -> None:
        # Given
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span

        # When
        _emit_span(tracer, "chat x", None, 100, 200, {"a": 1, "b": None, "c": ""})

        # Then
        tracer.start_span.assert_called_once_with(
            "chat x", context=None, start_time=100
        )
        span.set_attribute.assert_called_once_with("a", 1)
        span.end.assert_called_once_with(end_time=200)

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
    def _run(
        self,
        events: list,
        outcome: dict | None = None,
        finish_run: bool = True,
    ) -> list:
        """Drive the emitter, recording (kind, name, attrs) in emission order.

        Kinds are "turn_open" (turn span started and left open), "child" (a
        complete chat/execute_tool span) and "turn_close" (turn span ended).
        """
        recorded: list = []

        def fake_start(tracer, name, parent_ctx, start_ns, attrs):  # noqa: ANN001
            recorded.append(("turn_open", name, dict(attrs)))
            return MagicMock()

        def fake_emit(tracer, name, parent_ctx, start_ns, end_ns, attrs):  # noqa: ANN001
            recorded.append(("child", name, dict(attrs)))
            return MagicMock()

        def fake_end(span, end_ns, attrs=None):  # noqa: ANN001
            recorded.append(("turn_close", None, dict(attrs or {})))

        emitter = AgentSessionEmitter(
            session_id="sess-uuid",
            session_name="task-sample-1",
            agent_name="my_task",
            model="anthropic/claude-haiku-4-5",
            identity={"task": "my_task", "sample_id": 1},
        )
        with (
            patch("inspect_wandb.weave.sessions._start_span", side_effect=fake_start),
            patch("inspect_wandb.weave.sessions._emit_span", side_effect=fake_emit),
            patch("inspect_wandb.weave.sessions._end_span", side_effect=fake_end),
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
            if finish_run:
                emitter.finish(outcome)
        return recorded

    def test_in_flight_turn_emits_children_before_turn_closes(self) -> None:
        # Given
        events = [make_model_event(), make_tool_event()]

        # When: the turn has started but the sample has not ended
        recorded = self._run(events, finish_run=False)

        # Then: completed steps are already emitted while the turn is still open,
        # which is what makes an in-progress turn observable in the Agents view
        assert [kind for kind, _, _ in recorded] == ["turn_open", "child", "child"]
        assert recorded[1][1].startswith("chat")
        assert recorded[2][1].startswith("execute_tool")

    def test_hung_tool_leaves_chat_child_emitted_with_no_tool_child(self) -> None:
        # Given: the model requested a tool that never completed, so Inspect never
        # delivers a ToolEvent
        events = [make_model_event()]

        # When
        recorded = self._run(events, finish_run=False)

        # Then: the chat span is still visible with no execute_tool following it —
        # the signal a Monitor keys on to detect a hung tool call
        assert [kind for kind, _, _ in recorded] == ["turn_open", "child"]
        assert recorded[1][1].startswith("chat")

    def test_segments_turns_with_usage_on_llm_children_not_turn(self) -> None:
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
        turns = [(n, a) for kind, n, a in recorded if kind == "turn_open"]
        chats = [(n, a) for kind, n, a in recorded if n and n.startswith("chat")]
        assert len(turns) == 2
        assert turns[0][1]["inspect.turn_index"] == 0
        assert turns[1][1]["inspect.turn_index"] == 1
        assert turns[0][1]["inspect.task"] == "my_task"
        # Usage lives on the child chat spans, not the turn span; weave rolls it
        # up, so setting it on the turn too would double-count in the Agents view
        assert "gen_ai.usage.input_tokens" not in turns[0][1]
        assert chats[0][1]["gen_ai.usage.input_tokens"] == 100
        # Each turn is closed before the next one opens
        assert [kind for kind, _, _ in recorded] == [
            "turn_open",
            "child",
            "child",
            "turn_close",
            "turn_open",
            "child",
            "child",
            "turn_close",
        ]

    def test_final_turn_carries_outcome(self) -> None:
        # Given
        events = [make_model_event(), make_tool_event()]

        # When
        recorded = self._run(events, outcome={"score.includes": 1.0, "total_time": 5.0})

        # Then: outcome is attached when the last turn is closed
        closes = [a for kind, _, a in recorded if kind == "turn_close"]
        assert closes[-1]["inspect.score.includes"] == 1.0
        assert closes[-1]["inspect.total_time"] == 5.0

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
            "inspect_wandb.weave.sessions._start_span",
            side_effect=RuntimeError("otel down"),
        ):
            emitter.handle_event(make_model_event())
            emitter.finish()  # must not raise

    def test_tool_before_model_is_ignored(self) -> None:
        # Given / When
        recorded = self._run([make_tool_event()])

        # Then
        assert recorded == []
