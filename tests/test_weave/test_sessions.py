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
        messages = [ChatMessageUser(content="hi"), ChatMessageAssistant(content="yo")]
        result = to_messages(messages)
        assert [(m.role, m.content) for m in result] == [
            ("user", "hi"),
            ("assistant", "yo"),
        ]

    def test_llm_span_attrs_has_usage_provider_and_inspect_extras(self) -> None:
        event = make_model_event(input_tokens=321, output_tokens=99)
        attrs = llm_span_attrs(event, conversation_id="sess-1")
        assert attrs["gen_ai.request.model"] == "anthropic/claude-haiku-4-5"
        assert attrs["gen_ai.provider.name"] == "anthropic"
        assert attrs["gen_ai.usage.input_tokens"] == 321
        assert attrs["gen_ai.usage.output_tokens"] == 99
        assert attrs["gen_ai.request.temperature"] == 0.5
        assert attrs["inspect.generate.top_k"] == 40

    def test_tool_span_attrs_truncates_and_adds_inspect_extras(self) -> None:
        event = make_tool_event()
        event.result = "x" * 10000
        event.working_time = 1.5
        attrs = tool_span_attrs(event, conversation_id="sess-1")
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["inspect.tool.working_time"] == 1.5
        assert any("…[truncated]" in str(v) for v in attrs.values())

    def test_include_content_false_drops_messages_keeps_usage(self) -> None:
        event = make_model_event(input_tokens=100, output_tokens=20)
        attrs = llm_span_attrs(event, conversation_id="sess-1", include_content=False)
        assert "gen_ai.input.messages" not in attrs
        assert "gen_ai.output.messages" not in attrs
        assert attrs["gen_ai.usage.input_tokens"] == 100

    def test_tool_include_content_false_drops_args_and_result(self) -> None:
        event = make_tool_event()
        attrs = tool_span_attrs(event, conversation_id="sess-1", include_content=False)
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert all("ls -la" not in str(v) for v in attrs.values())

    def test_flatten_metadata(self) -> None:
        metadata = {"difficulty": "hard", "category": "crypto"}
        result = flatten_metadata(metadata)
        assert result == {"metadata.difficulty": "hard", "metadata.category": "crypto"}

    def test_flatten_metadata_ignores_non_dict(self) -> None:
        assert flatten_metadata("not a dict") == {}

    def test_coerce_preserves_scalars_and_json_encodes_collections(self) -> None:
        assert _coerce(True) is True
        assert _coerce(5) == 5
        assert _coerce("x") == "x"
        assert _coerce(["a", "b"]) == '["a", "b"]'
        assert _coerce(None) is None

    def test_usage_from_event_handles_missing_usage(self) -> None:
        event = make_model_event()
        event.output.usage = None
        usage = usage_from_event(event)
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_emit_span_sets_attrs_skips_empty_and_ends(self) -> None:
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        _emit_span(tracer, "chat x", None, 100, 200, {"a": 1, "b": None, "c": ""})
        tracer.start_span.assert_called_once_with("chat x", context=None, start_time=100)
        span.set_attribute.assert_called_once_with("a", 1)
        span.end.assert_called_once_with(end_time=200)
        span.set_status.assert_not_called()

    def test_emit_span_sets_error_status_on_failure(self) -> None:
        from opentelemetry.trace import Status, StatusCode
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        exc = ValueError("something went wrong")
        _emit_span(tracer, "chat x", None, 100, 200, {"a": 1}, failed=True, exception=exc)
        span.set_status.assert_called_once_with(Status(StatusCode.ERROR, "something went wrong"))
        span.record_exception.assert_called_once_with(exc)
        span.end.assert_called_once_with(end_time=200)

    def test_emit_span_sets_error_status_without_exception(self) -> None:
        from opentelemetry.trace import Status, StatusCode
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        _emit_span(tracer, "chat x", None, 100, 200, {"a": 1}, failed=True)
        span.set_status.assert_called_once_with(Status(StatusCode.ERROR, None))
        span.record_exception.assert_not_called()
        span.end.assert_called_once_with(end_time=200)

    def test_build_outcome_includes_scores_timing_and_tokens(self) -> None:
        sample = SimpleNamespace(
            total_time=12.3,
            working_time=10.1,
            error=None,
            limit=None,
            scores={"includes": Score(value=1.0, answer="73")},
            model_usage={"anthropic/claude-haiku-4-5": ModelUsage(total_tokens=7815)},
        )
        outcome = build_outcome(sample)
        assert outcome["total_time"] == 12.3
        assert outcome["score.includes"] == 1.0
        assert outcome["score.includes.answer"] == "73"
        assert outcome["total_tokens"] == 7815


class TestAgentSessionEmitter:
    def _run(self, events: list, outcome: dict | None = None) -> list:
        recorded: list = []

        def fake_emit(tracer, name, parent_ctx, start_ns, end_ns, attrs, *, failed=False, exception=None):
            recorded.append((name, dict(attrs), failed, exception))
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
            patch("inspect_wandb.weave.sessions.set_span_in_context", return_value=None),
            patch("inspect_wandb.weave.sessions.otel_trace.get_tracer", return_value=MagicMock()),
        ):
            for event in events:
                emitter.handle_event(event)
            emitter.finish(outcome)
        return recorded

    def test_segments_turns_with_usage_on_llm_children_not_turn(self) -> None:
        events = [
            make_model_event(input_tokens=100, output_tokens=10),
            make_tool_event(tool_id="a"),
            make_model_event(input_tokens=50, output_tokens=5),
            make_tool_event(tool_id="b"),
        ]
        recorded = self._run(events)
        turns = [(n, a) for n, a in recorded if n.startswith("invoke_agent")]
        chats = [(n, a) for n, a in recorded if n.startswith("chat")]
        assert len(turns) == 2
        assert turns[0][1]["inspect.turn_index"] == 0
        assert turns[1][1]["inspect.turn_index"] == 1
        assert turns[0][1]["inspect.task"] == "my_task"
        assert "gen_ai.usage.input_tokens" not in turns[0][1]
        assert chats[0][1]["gen_ai.usage.input_tokens"] == 100

    def test_final_turn_carries_outcome(self) -> None:
        events = [make_model_event(), make_tool_event()]
        recorded = self._run(events, outcome={"score.includes": 1.0, "total_time": 5.0})
        turns = [(n, a) for n, a in recorded if n.startswith("invoke_agent")]
        assert turns[-1][1]["inspect.score.includes"] == 1.0
        assert turns[-1][1]["inspect.total_time"] == 5.0

    def test_emit_failure_is_swallowed(self) -> None:
        emitter = AgentSessionEmitter(
            session_id="s",
            session_name="n",
            agent_name="a",
            model="m",
            identity={},
        )
        with patch("inspect_wandb.weave.sessions._emit_span", side_effect=RuntimeError("otel down")):
            emitter.handle_event(make_model_event())
            emitter.finish()

    def test_tool_before_model_is_ignored(self) -> None:
        recorded = self._run([make_tool_event()])
        assert recorded == []

    def test_failed_tool_emits_span_with_error(self) -> None:
        from inspect_ai.tool._tool_call import ToolCallError
        tool = make_tool_event(tool_id="call_1", function="bash")
        tool.error = ToolCallError(type="timeout", message="Tool call timed out")
        tool.failed = True
        recorded = self._run([make_model_event(), tool])
        tool_spans = [(n, a, f) for n, a, f, e in recorded if n.startswith("execute_tool")]
        assert len(tool_spans) == 1
        _, _, failed = tool_spans[0]
        assert failed is True

    def test_failed_model_emits_span_with_error(self) -> None:
        model = make_model_event()
        model.error = "Model API returned rate limit error"
        recorded = self._run([model])
        chat_spans = [(n, a, f, e) for n, a, f, e in recorded if n.startswith("chat")]
        assert len(chat_spans) == 1
        _, _, failed, exception = chat_spans[0]
        assert failed is True
        assert str(exception) == "Model API returned rate limit error"

    def test_successful_tool_no_error_flag(self) -> None:
        tool = make_tool_event(tool_id="call_1", function="bash")
        recorded = self._run([make_model_event(), tool])
        tool_spans = [(n, a, f) for n, a, f, e in recorded if n.startswith("execute_tool")]
        assert len(tool_spans) == 1
        _, _, failed = tool_spans[0]
        assert failed is False