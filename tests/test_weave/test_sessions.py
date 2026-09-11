from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

from inspect_ai.event import CompactionEvent, ModelEvent, ToolEvent
from inspect_ai.log import EvalError, EvalSample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.scorer import Score
from opentelemetry.context import Context
from opentelemetry.trace import Span, StatusCode

from inspect_wandb.weave.sessions import (
    AgentSessionEmitter,
    ChatSpan,
    SampleOutcome,
    ScoreOutcome,
    SessionSpan,
    SessionSpanWriter,
    ToolSpan,
    _coerce_to_otel_scalar,
)

T0 = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 6, 21, 12, 0, 1, tzinfo=UTC)
T2 = datetime(2026, 6, 21, 12, 0, 2, tzinfo=UTC)


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


def make_model_event_with_history(contents: list[str]) -> ModelEvent:
    event = make_model_event()
    event.input = [ChatMessageUser(content=text) for text in contents]
    return event


def make_sample(**overrides: Any) -> EvalSample:
    sample = EvalSample(id=1, epoch=1, input="solve the task", target="73")
    for name, value in overrides.items():
        setattr(sample, name, value)
    return sample


def chat_attributes(
    event: ModelEvent,
    *,
    conversation_id: str = "sess-1",
    include_content: bool = True,
    input_from_index: int = 0,
) -> dict[str, Any]:
    return ChatSpan.from_event(
        event,
        conversation_id=conversation_id,
        include_content=include_content,
        input_from_index=input_from_index,
    ).attributes


def tool_attributes(
    event: ToolEvent,
    *,
    conversation_id: str = "sess-1",
    include_content: bool = True,
) -> dict[str, Any]:
    return ToolSpan.from_event(
        event, conversation_id=conversation_id, include_content=include_content
    ).attributes


class TestSpanBuilders:
    def test_chat_span_maps_message_roles(self) -> None:
        # Given
        messages_in = [
            ChatMessageUser(content="hi"),
            ChatMessageAssistant(content="yo"),
        ]

        # When
        messages = ChatSpan._to_messages(messages_in)

        # Then
        assert [(message.role, message.content) for message in messages] == [
            ("user", "hi"),
            ("assistant", "yo"),
        ]

    def test_chat_span_has_usage_provider_and_inspect_extras(self) -> None:
        # Given
        event = make_model_event(input_tokens=321, output_tokens=99)

        # When
        attributes = chat_attributes(event)

        # Then
        assert attributes["gen_ai.request.model"] == "anthropic/claude-haiku-4-5"
        assert attributes["gen_ai.provider.name"] == "anthropic"
        assert attributes["gen_ai.usage.input_tokens"] == 321
        assert attributes["gen_ai.usage.output_tokens"] == 99
        assert attributes["gen_ai.request.temperature"] == 0.5
        assert attributes["inspect.generate.top_k"] == 40

    def test_chat_span_names_and_times_from_event(self) -> None:
        # Given
        event = make_model_event()

        # When
        span = ChatSpan.from_event(
            event, conversation_id="s", include_content=True, input_from_index=0
        )

        # Then
        assert span.name == "chat anthropic/claude-haiku-4-5"
        assert span.start_nanoseconds == int(T0.timestamp() * 1_000_000_000)
        assert span.end_nanoseconds == int(T1.timestamp() * 1_000_000_000)

    def test_tool_span_truncates_and_adds_inspect_extras(self) -> None:
        # Given
        event = make_tool_event()
        event.result = "x" * 10000
        event.working_time = 1.5

        # When
        attributes = tool_attributes(event)

        # Then
        assert attributes["gen_ai.operation.name"] == "execute_tool"
        assert attributes["inspect.tool.working_time"] == 1.5
        assert any("…[truncated]" in str(value) for value in attributes.values())

    def test_include_content_false_drops_messages_keeps_usage(self) -> None:
        # Given
        event = make_model_event(input_tokens=100, output_tokens=20)

        # When
        attributes = chat_attributes(event, include_content=False)

        # Then
        assert "gen_ai.input.messages" not in attributes
        assert "gen_ai.output.messages" not in attributes
        assert attributes["gen_ai.usage.input_tokens"] == 100

    def test_chat_span_trims_input_to_index(self) -> None:
        # Given
        event = make_model_event_with_history(["MSG_A", "MSG_B", "MSG_C"])

        # When
        attributes = chat_attributes(event, input_from_index=1)

        # Then
        serialized_input = attributes["gen_ai.input.messages"]
        assert "MSG_B" in serialized_input
        assert "MSG_C" in serialized_input
        assert "MSG_A" not in serialized_input

    def test_chat_span_include_content_false_ignores_index(self) -> None:
        # Given
        event = make_model_event_with_history(["MSG_A", "MSG_B"])

        # When
        attributes = chat_attributes(event, include_content=False, input_from_index=1)

        # Then
        assert "gen_ai.input.messages" not in attributes

    def test_tool_include_content_false_drops_args_and_result(self) -> None:
        # Given
        event = make_tool_event()

        # When
        attributes = tool_attributes(event, include_content=False)

        # Then
        assert attributes["gen_ai.operation.name"] == "execute_tool"
        assert all("ls -la" not in str(value) for value in attributes.values())

    def test_tool_span_marks_failed_only_on_failure(self) -> None:
        # Given
        failed_event = make_tool_event()
        failed_event.failed = True
        ok_event = make_tool_event()

        # When
        failed_span = ToolSpan.from_event(
            failed_event, conversation_id="s", include_content=True
        )
        ok_span = ToolSpan.from_event(
            ok_event, conversation_id="s", include_content=True
        )

        # Then
        assert failed_span.attributes["inspect.tool.failed"] is True
        assert "inspect.tool.failed" not in ok_span.attributes
        assert failed_span.failed is True
        assert ok_span.failed is False

    def test_chat_span_surfaces_model_error(self) -> None:
        # Given
        errored_event = make_model_event()
        errored_event.error = "rate limit exceeded"
        ok_event = make_model_event()

        # When
        errored_span = ChatSpan.from_event(
            errored_event, conversation_id="s", include_content=True, input_from_index=0
        )
        ok_span = ChatSpan.from_event(
            ok_event, conversation_id="s", include_content=True, input_from_index=0
        )

        # Then
        assert errored_span.attributes["inspect.model.error"] == "rate limit exceeded"
        assert errored_span.failed is True
        assert errored_span.error == "rate limit exceeded"
        assert "inspect.model.error" not in ok_span.attributes

    def test_coerce_preserves_scalars_and_json_encodes_collections(self) -> None:
        # Given / When / Then
        assert _coerce_to_otel_scalar(True) is True  # bool checked before int
        assert _coerce_to_otel_scalar(5) == 5
        assert _coerce_to_otel_scalar("x") == "x"
        assert _coerce_to_otel_scalar(["a", "b"]) == '["a", "b"]'
        assert _coerce_to_otel_scalar(None) is None

    def test_usage_from_event_handles_missing_usage(self) -> None:
        # Given
        event = make_model_event()
        event.output.usage = None

        # When
        usage = ChatSpan._usage_from_event(event)

        # Then
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestSessionSpan:
    def test_defaults_are_supplied_without_explicit_init(self) -> None:
        # Given / When
        span = SessionSpan(name="chat x")

        # Then
        assert span.attributes == {}
        assert span.start_nanoseconds is None
        assert span.end_nanoseconds is None
        assert span.failed is False
        assert span.error is None

    def test_each_span_gets_its_own_attributes_dict(self) -> None:
        # Given
        first = SessionSpan(name="a")
        second = SessionSpan(name="b")

        # When
        first.attributes["only_on_first"] = 1

        # Then
        assert second.attributes == {}


class TestSessionSpanWriter:
    def _writer(self) -> tuple[SessionSpanWriter, MagicMock, MagicMock]:
        tracer = MagicMock()
        open_span = MagicMock()
        tracer.start_span.return_value = open_span
        with patch(
            "inspect_wandb.weave.sessions.otel_trace.get_tracer", return_value=tracer
        ):
            return SessionSpanWriter(), tracer, open_span

    def test_emit_sets_attributes_skips_empty_and_ends(self) -> None:
        # Given
        writer, tracer, open_span = self._writer()
        span = SessionSpan(
            name="chat x",
            attributes={"a": 1, "b": None, "c": ""},
            start_nanoseconds=100,
            end_nanoseconds=200,
        )

        # When
        writer.emit(span, None)

        # Then
        tracer.start_span.assert_called_once_with(
            "chat x", context=None, start_time=100
        )
        open_span.set_attribute.assert_called_once_with("a", 1)
        open_span.end.assert_called_once_with(end_time=200)

    def test_emit_without_timestamps_omits_them(self) -> None:
        # Given
        writer, tracer, open_span = self._writer()

        # When
        writer.emit(SessionSpan(name="chat x"), None)

        # Then
        tracer.start_span.assert_called_once_with("chat x", context=None)
        open_span.end.assert_called_once_with()

    def test_status_marks_error_with_message_and_ok_otherwise(self) -> None:
        # Given
        writer, _tracer, errored_span = self._writer()

        # When
        writer.emit(SessionSpan(name="x", failed=True, error="boom"), None)

        # Then
        error_status = errored_span.set_status.call_args.args[0]
        assert error_status.status_code == StatusCode.ERROR
        assert error_status.description == "boom"

        # Given / When
        writer, _tracer, ok_span = self._writer()
        writer.emit(SessionSpan(name="x"), None)

        # Then
        assert ok_span.set_status.call_args.args[0].status_code == StatusCode.OK


class TestSampleOutcome:
    def test_from_sample_captures_scores_timing_and_tokens(self) -> None:
        # Given
        sample = make_sample(
            total_time=12.3,
            working_time=10.1,
            scores={"includes": Score(value=1.0, answer="73")},
            model_usage={"anthropic/claude-haiku-4-5": ModelUsage(total_tokens=7815)},
        )

        # When
        outcome = SampleOutcome.from_sample(sample)

        # Then
        assert outcome.total_time == 12.3
        assert outcome.working_time == 10.1
        assert outcome.total_tokens == 7815
        assert outcome.scores == {"includes": ScoreOutcome(value=1.0, answer="73")}

    def test_to_attributes_namespaces_and_flattens_scores(self) -> None:
        # Given
        sample = make_sample(
            total_time=12.3,
            scores={"includes": Score(value=1.0, answer="73")},
            model_usage={"anthropic/claude-haiku-4-5": ModelUsage(total_tokens=7815)},
        )

        # When
        attributes = SampleOutcome.from_sample(sample).to_attributes()

        # Then
        assert attributes["inspect.total_time"] == 12.3
        assert attributes["inspect.score.includes"] == 1.0
        assert attributes["inspect.score.includes.answer"] == "73"
        assert attributes["inspect.total_tokens"] == 7815

    def test_score_without_answer_omits_answer_attribute(self) -> None:
        # Given
        sample = make_sample(scores={"accuracy": Score(value=0.0)})

        # When
        attributes = SampleOutcome.from_sample(sample).to_attributes()

        # Then
        assert "inspect.score.accuracy.answer" not in attributes

    def test_absent_fields_are_dropped_from_attributes(self) -> None:
        # Given
        sample = make_sample()

        # When
        outcome = SampleOutcome.from_sample(sample)

        # Then
        assert outcome.total_tokens is None
        assert outcome.scores == {}
        assert outcome.to_attributes() == {}

    def test_error_message_reads_through_to_eval_error(self) -> None:
        # Given
        sample = make_sample(
            error=EvalError(
                message="sample crashed", traceback="tb", traceback_ansi="tb"
            )
        )

        # When
        outcome = SampleOutcome.from_sample(sample)

        # Then
        assert outcome.error_message == "sample crashed"

    def test_error_message_is_none_when_sample_succeeded(self) -> None:
        # Given / When
        outcome = SampleOutcome.from_sample(make_sample())

        # Then
        assert outcome.error_message is None


@dataclass
class Recorded:
    kind: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    failed: bool = False
    error: str | None = None


class RecordingWriter(SessionSpanWriter):
    """Captures spans instead of writing them, snapshotting mutable attributes.

    The turn span is mutated after it is opened, so each record keeps a copy.
    """

    def __init__(self) -> None:
        self.records: list[Recorded] = []

    def _record(self, kind: str, span: SessionSpan) -> None:
        self.records.append(
            Recorded(kind, span.name, dict(span.attributes), span.failed, span.error)
        )

    def emit(self, span: SessionSpan, parent_context: Context | None) -> None:
        self._record("child", span)

    def open(self, span: SessionSpan, parent_context: Context | None) -> Span:
        self._record("turn_open", span)
        return MagicMock()

    def close(self, open_span: Span, span: SessionSpan) -> None:
        self._record("turn_close", span)


class TestAgentSessionEmitter:
    def _run(
        self,
        events: list,
        outcome: SampleOutcome | None = None,
        finish_run: bool = True,
    ) -> list[Recorded]:
        writer = RecordingWriter()
        emitter = AgentSessionEmitter(
            session_id="sess-uuid",
            session_name="task-sample-1",
            agent_name="my_task",
            model="anthropic/claude-haiku-4-5",
            identity={"task": "my_task", "sample_id": 1},
            writer=writer,
        )
        for event in events:
            emitter.handle_event(event)
        if finish_run:
            emitter.finish(outcome)
        return writer.records

    def test_in_flight_turn_emits_children_before_turn_closes(self) -> None:
        # Given
        events = [make_model_event(), make_tool_event()]

        # When: the turn has started but the sample has not ended
        recorded = self._run(events, finish_run=False)

        # Then: completed steps are already emitted while the turn is still open,
        # which is what makes an in-progress turn observable in the Agents view
        assert [record.kind for record in recorded] == ["turn_open", "child", "child"]
        assert recorded[1].name.startswith("chat")
        assert recorded[2].name.startswith("execute_tool")

    def test_hung_tool_leaves_chat_child_emitted_with_no_tool_child(self) -> None:
        # Given: the model requested a tool that never completed, so Inspect never
        # delivers a ToolEvent
        events = [make_model_event()]

        # When
        recorded = self._run(events, finish_run=False)

        # Then: the chat span is still visible with no execute_tool following it —
        # the signal a Monitor keys on to detect a hung tool call
        assert [record.kind for record in recorded] == ["turn_open", "child"]
        assert recorded[1].name.startswith("chat")

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
        turns = [record for record in recorded if record.kind == "turn_open"]
        chats = [record for record in recorded if record.name.startswith("chat")]
        assert len(turns) == 2
        assert turns[0].attributes["inspect.turn_index"] == 0
        assert turns[1].attributes["inspect.turn_index"] == 1
        assert turns[0].attributes["inspect.task"] == "my_task"
        # Usage lives on the child chat spans, not the turn span; weave rolls it
        # up, so setting it on the turn too would double-count in the Agents view
        assert "gen_ai.usage.input_tokens" not in turns[0].attributes
        assert chats[0].attributes["gen_ai.usage.input_tokens"] == 100
        # Each turn is closed before the next one opens
        assert [record.kind for record in recorded] == [
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
        sample = make_sample(
            total_time=5.0, scores={"includes": Score(value=1.0, answer="73")}
        )

        # When
        recorded = self._run(
            [make_model_event(), make_tool_event()],
            outcome=SampleOutcome.from_sample(sample),
        )

        # Then: outcome is attached when the last turn is closed
        closes = [record for record in recorded if record.kind == "turn_close"]
        assert closes[-1].attributes["inspect.score.includes"] == 1.0
        assert closes[-1].attributes["inspect.total_time"] == 5.0

    def test_outcome_is_absent_from_the_open_turn(self) -> None:
        # Given
        sample = make_sample(total_time=5.0)

        # When
        recorded = self._run(
            [make_model_event()], outcome=SampleOutcome.from_sample(sample)
        )

        # Then: outcome lands only at close, not on the turn as it was opened
        opens = [record for record in recorded if record.kind == "turn_open"]
        assert "inspect.total_time" not in opens[0].attributes

    def test_emit_failure_is_swallowed(self) -> None:
        # Given
        writer = MagicMock()
        writer.open.side_effect = RuntimeError("otel down")
        emitter = AgentSessionEmitter(
            session_id="s",
            session_name="n",
            agent_name="a",
            model="m",
            identity={},
            writer=writer,
        )

        # When / Then
        emitter.handle_event(make_model_event())
        emitter.finish()  # must not raise

    def test_tool_before_model_is_ignored(self) -> None:
        # Given / When
        recorded = self._run([make_tool_event()])

        # Then
        assert recorded == []

    def _chat_inputs(self, recorded: list[Recorded]) -> list[str]:
        return [
            record.attributes.get("gen_ai.input.messages", "")
            for record in recorded
            if record.name.startswith("chat")
        ]

    def test_input_trimmed_to_delta_across_turns(self) -> None:
        # Given: turn 1 sends one message, turn 2 re-ships it plus two new ones
        events = [
            make_model_event_with_history(["MSG_A"]),
            make_model_event_with_history(["MSG_A", "MSG_B", "MSG_C"]),
        ]

        # When
        first_input, second_input = self._chat_inputs(self._run(events))

        # Then: turn 1 is full (prev=0); turn 2 carries only the new delta
        assert "MSG_A" in first_input
        assert "MSG_B" in second_input
        assert "MSG_C" in second_input
        assert "MSG_A" not in second_input

    def test_compaction_event_resets_to_full_input(self) -> None:
        # Given: history is rewritten to a summary between the two turns
        events = [
            make_model_event_with_history(["MSG_A", "MSG_B"]),
            CompactionEvent(type="summary"),
            make_model_event_with_history(["SUMMARY"]),
        ]

        # When
        first_input, second_input = self._chat_inputs(self._run(events))

        # Then: without the reset, prev_len=2 would slice the 1-message input to
        # empty; the CompactionEvent forces the next turn to re-ship in full
        assert "MSG_A" in first_input
        assert "SUMMARY" in second_input

    def test_aggregate_input_volume_is_linear_not_quadratic(self) -> None:
        # Given: an agent whose history grows by one message each turn, so the
        # previous input length equals the turn index
        turns = 20
        events = [
            make_model_event_with_history([f"m{i}" for i in range(turn + 1)])
            for turn in range(turns)
        ]

        def serialized_input_length(event: ModelEvent, input_from_index: int) -> int:
            attributes = chat_attributes(event, input_from_index=input_from_index)
            return len(attributes["gen_ai.input.messages"])

        # When: comparing the delta (from the previous turn's length) against
        # re-shipping the full history every turn, using the same serialization
        delta_volume = sum(
            serialized_input_length(event, turn) for turn, event in enumerate(events)
        )
        full_volume = sum(serialized_input_length(event, 0) for event in events)

        # Then: delta emits ~one message per turn (linear) rather than the whole
        # history each turn (quadratic)
        assert delta_volume < full_volume / 5

    def _status(self, recorded: list[Recorded], name_prefix: str) -> tuple:
        for record in recorded:
            if record.name.startswith(name_prefix):
                return (record.failed, record.error)
        raise AssertionError(f"no span starting with {name_prefix!r}")

    def test_failed_tool_marks_span_status_error(self) -> None:
        # Given
        tool = make_tool_event()
        tool.failed = True
        events = [make_model_event(), tool]

        # When
        recorded = self._run(events, finish_run=False)

        # Then: the execute_tool span carries ERROR status so it reads as failed
        assert self._status(recorded, "execute_tool") == (True, None)

    def test_model_error_marks_chat_span_status_error(self) -> None:
        # Given
        model = make_model_event()
        model.error = "rate limit exceeded"

        # When
        recorded = self._run([model], finish_run=False)

        # Then
        assert self._status(recorded, "chat") == (True, "rate limit exceeded")

    def test_successful_steps_marked_ok_not_unset(self) -> None:
        # Given
        events = [make_model_event(), make_tool_event()]

        # When
        recorded = self._run(events, finish_run=False)

        # Then: non-failing spans are explicitly OK rather than left UNSET
        assert self._status(recorded, "chat")[0] is False
        assert self._status(recorded, "execute_tool")[0] is False

    def test_turn_marked_error_when_sample_errored(self) -> None:
        # Given
        sample = make_sample(
            error=EvalError(
                message="sample crashed", traceback="tb", traceback_ansi="tb"
            )
        )

        # When
        recorded = self._run(
            [make_model_event()], outcome=SampleOutcome.from_sample(sample)
        )

        # Then: the closed turn reflects the sample-level failure
        closes = [record for record in recorded if record.kind == "turn_close"]
        assert (closes[-1].failed, closes[-1].error) == (True, "sample crashed")
