import json
from datetime import datetime
from logging import getLogger
from typing import Any, get_args

from inspect_ai.event import CompactionEvent, Event, ModelEvent, ToolEvent
from inspect_ai.log import EvalError, EvalSample, EvalSampleLimit
from inspect_ai.model import ChatMessage
from inspect_ai.scorer import Value
from opentelemetry import trace as otel_trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context
from pydantic import BaseModel, Field
from weave.conversation.conversation_otel import (
    execute_tool_attributes,
    invoke_agent_attributes,
    llm_attributes,
)
from weave.conversation.types import Message, Usage

logger = getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 4000
MAX_ATTRIBUTE_VALUE_CHARS = 16000
_WEAVE_ROLES = frozenset(get_args(Message.model_fields["role"].annotation))
_TRACER_NAME = "weave.conversation"


def _to_nanoseconds(event_time: datetime | None) -> int | None:
    return (
        int(event_time.timestamp() * 1_000_000_000) if event_time is not None else None
    )


def _coerce_to_otel_scalar(value: Any) -> str | int | float | bool | None:
    """Coerce a value to a valid OTel attribute scalar, or None to skip."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_ATTRIBUTE_VALUE_CHARS]
    return json.dumps(value, default=str)[:MAX_ATTRIBUTE_VALUE_CHARS]


def _inspect_attributes(values: dict[str, Any]) -> dict[str, Any]:
    """Build namespaced ``inspect.*`` attributes, coercing and dropping empties."""
    out: dict[str, Any] = {}
    for key, raw in values.items():
        coerced = _coerce_to_otel_scalar(raw)
        if coerced is not None and coerced != "":
            out[f"inspect.{key}"] = coerced
    return out


class SessionSpan(BaseModel):
    """A gen_ai OTel span to emit: its name, attributes, timing and status.

    Subclasses build the attributes for each span kind (``chat``, ``execute_tool``,
    ``invoke_agent``) from the corresponding Inspect event; the ``SessionSpanWriter``
    turns them into real OTel spans.
    """

    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    start_nanoseconds: int | None = None
    end_nanoseconds: int | None = None
    failed: bool = False
    error: str | None = None


class ChatSpan(SessionSpan):
    @staticmethod
    def _to_messages(messages: list[ChatMessage]) -> list[Message]:
        return [
            Message(
                role=message.role if message.role in _WEAVE_ROLES else "user",
                content=message.text or "",
            )
            for message in messages
        ]

    @staticmethod
    def _usage_from_event(event: ModelEvent) -> Usage:
        usage = event.output.usage
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            reasoning_tokens=usage.reasoning_tokens or 0,
            cache_creation_input_tokens=usage.input_tokens_cache_write or 0,
            cache_read_input_tokens=usage.input_tokens_cache_read or 0,
        )

    @classmethod
    def from_event(
        cls,
        event: ModelEvent,
        *,
        conversation_id: str,
        include_content: bool,
        input_from_index: int,
    ) -> "ChatSpan":
        config = event.config
        output = event.output
        output_messages = (
            cls._to_messages([output.choices[0].message])
            if output.choices
            else [Message.assistant(output.completion)]
        )
        provider_name = event.model.split("/", 1)[0] if "/" in event.model else ""
        attributes = {
            **llm_attributes(
                model=event.model,
                provider_name=provider_name,
                conversation_id=conversation_id,
                input_messages=cls._to_messages(event.input[input_from_index:])
                if include_content
                else None,
                output_messages=output_messages if include_content else None,
                usage=cls._usage_from_event(event),
                finish_reasons=[
                    choice.stop_reason
                    for choice in output.choices
                    if choice.stop_reason
                ],
                response_model=output.model or "",
                request_temperature=config.temperature,
                request_max_tokens=config.max_tokens,
                request_top_p=config.top_p,
                request_frequency_penalty=config.frequency_penalty,
                request_presence_penalty=config.presence_penalty,
                request_seed=config.seed,
                request_stop_sequences=config.stop_seqs,
            ),
            **_inspect_attributes(
                {
                    # generation/sampling knobs not covered by gen_ai request.*
                    "generate.top_k": config.top_k,
                    "generate.best_of": config.best_of,
                    "generate.num_choices": config.num_choices,
                    "generate.logprobs": config.logprobs,
                    "generate.top_logprobs": config.top_logprobs,
                    # reasoning knobs
                    "generate.reasoning_effort": config.reasoning_effort,
                    "generate.effort": config.effort,
                    "generate.reasoning_tokens": config.reasoning_tokens,
                    "generate.verbosity": config.verbosity,
                    # this model call
                    "model.retries": event.retries,
                    "model.cache": event.cache,
                    "model.error": event.error,
                }
            ),
        }
        return cls(
            name=f"chat {event.model}",
            attributes=attributes,
            start_nanoseconds=_to_nanoseconds(event.timestamp),
            end_nanoseconds=_to_nanoseconds(event.completed),
            failed=event.error is not None,
            error=event.error,
        )


class ToolSpan(SessionSpan):
    @classmethod
    def from_event(
        cls, event: ToolEvent, *, conversation_id: str, include_content: bool
    ) -> "ToolSpan":
        result = str(event.result)
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = result[:MAX_TOOL_RESULT_CHARS] + "…[truncated]"
        attributes = {
            **execute_tool_attributes(
                tool_name=event.function,
                conversation_id=conversation_id,
                tool_call_arguments=json.dumps(event.arguments, default=str)
                if include_content
                else "",
                tool_call_result=result if include_content else "",
                tool_call_id=event.id,
            ),
            **_inspect_attributes(
                {
                    "tool.error": getattr(event.error, "message", None)
                    if event.error
                    else None,
                    "tool.failed": event.failed or None,
                    "tool.truncated": event.truncated is not None,
                    "tool.working_time": event.working_time,
                }
            ),
        }
        return cls(
            name=f"execute_tool {event.function}",
            attributes=attributes,
            start_nanoseconds=_to_nanoseconds(event.timestamp),
            end_nanoseconds=_to_nanoseconds(event.completed),
            failed=bool(event.failed) or event.error is not None,
            error=getattr(event.error, "message", None),
        )


class TurnSpan(SessionSpan):
    @classmethod
    def opened(
        cls,
        *,
        agent_name: str,
        conversation_id: str,
        conversation_name: str,
        model: str,
        identity_attributes: dict[str, Any],
        turn_index: int,
        start: datetime | None,
    ) -> "TurnSpan":
        attributes = {
            **invoke_agent_attributes(
                agent_name=agent_name,
                conversation_id=conversation_id,
                conversation_name=conversation_name,
                model=model,
                agent_version=model,
            ),
            **identity_attributes,
            "inspect.turn_index": turn_index,
        }
        return cls(
            name=f"invoke_agent {agent_name}",
            attributes=attributes,
            start_nanoseconds=_to_nanoseconds(start),
        )


class SessionSpanWriter:
    """Writes ``SessionSpan``s to the weave-configured global OTel tracer.

    ``emit`` is for complete spans (a ``chat``/``execute_tool`` child, started and
    ended in one go); ``open``/``close`` bracket the ``invoke_agent`` turn span,
    which stays open while its children are emitted.
    """

    def __init__(self) -> None:
        self._tracer = otel_trace.get_tracer(_TRACER_NAME)

    def emit(self, span: SessionSpan, parent_context: Context | None) -> None:
        self._finalise(self._start(span, parent_context), span)

    def open(self, span: SessionSpan, parent_context: Context | None) -> Span:
        return self._start(span, parent_context)

    def close(self, open_span: Span, span: SessionSpan) -> None:
        self._set_attributes(open_span, span.attributes)
        self._finalise(open_span, span)

    def _start(self, span: SessionSpan, parent_context: Context | None) -> Span:
        open_span = (
            self._tracer.start_span(
                span.name, context=parent_context, start_time=span.start_nanoseconds
            )
            if span.start_nanoseconds is not None
            else self._tracer.start_span(span.name, context=parent_context)
        )
        self._set_attributes(open_span, span.attributes)
        return open_span

    def _finalise(self, open_span: Span, span: SessionSpan) -> None:
        # ERROR (with the message as the status description) makes a failed step
        # show up in Weave's native status column and error-rate; OK marks the rest
        # so a successful span reads as OK rather than the default UNSET.
        if span.failed:
            open_span.set_status(Status(StatusCode.ERROR, span.error or None))
        else:
            open_span.set_status(Status(StatusCode.OK))
        if span.end_nanoseconds is not None:
            open_span.end(end_time=span.end_nanoseconds)
        else:
            open_span.end()

    @staticmethod
    def _set_attributes(open_span: Span, attributes: dict[str, Any]) -> None:
        for key, value in attributes.items():
            if value is not None and value != "":
                open_span.set_attribute(key, value)


class ScoreOutcome(BaseModel):
    value: Value
    answer: str | None = None


class SampleOutcome(BaseModel):
    """Sample-outcome metadata, known only at sample end, for the final turn."""

    total_time: float | None = None
    working_time: float | None = None
    error: EvalError | None = None
    limit: EvalSampleLimit | None = None
    total_tokens: int | None = None
    scores: dict[str, ScoreOutcome] = Field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: EvalSample) -> "SampleOutcome":
        total_tokens = sum(
            (usage.total_tokens or 0)
            for usage in sample.model_usage.values()
            if usage.total_tokens is not None
        )
        return cls(
            total_time=sample.total_time,
            working_time=sample.working_time,
            error=sample.error,
            limit=sample.limit,
            total_tokens=total_tokens or None,
            scores={
                name: ScoreOutcome(value=score.value, answer=score.answer)
                for name, score in (sample.scores or {}).items()
            },
        )

    @property
    def error_message(self) -> str | None:
        return self.error.message if self.error is not None else None

    def to_attributes(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "total_time": self.total_time,
            "working_time": self.working_time,
            "error": self.error,
            "limit": self.limit,
            "total_tokens": self.total_tokens,
        }
        for name, score in self.scores.items():
            values[f"score.{name}"] = score.value
            if score.answer:
                values[f"score.{name}.answer"] = score.answer
        return _inspect_attributes(values)


class AgentSessionEmitter:
    """Reconstructs an Inspect sample's agent trajectory and streams it to
    Weave's agent Session SDK as gen_ai OpenTelemetry spans, one turn at a time.

    Emits the spans directly via the weave-configured global tracer (rather than
    weave's imperative ``log_turn``) so we can attach rich ``inspect.*`` metadata
    and preserve the original Inspect event timestamps.

    Emission is *incremental*: the ``invoke_agent`` turn span is opened when the
    turn starts and each child (``chat``/``execute_tool``) span is emitted as its
    event arrives, rather than buffering the whole turn and emitting at turn end.
    This enables live monitoring of agent rollouts.
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_name: str,
        agent_name: str,
        model: str,
        identity: dict[str, Any],
        include_content: bool = True,
        writer: SessionSpanWriter | None = None,
    ) -> None:
        self._session_id = session_id
        self._session_name = session_name
        self._agent_name = agent_name
        self._model = model
        self._identity_attributes = _inspect_attributes(identity)
        self._include_content = include_content
        self._writer = writer or SessionSpanWriter()
        self._turn_index = 0
        self._prev_input_length = 0
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._turn: TurnSpan | None = None
        self._open_turn_span: Span | None = None
        self._child_context: Context | None = None
        self._turn_end: datetime | None = None

    def handle_event(self, event: Event) -> None:
        try:
            if isinstance(event, ModelEvent):
                self._close_turn()
                self._open_turn(event.timestamp)
                self._turn_end = event.completed or event.timestamp
                self._writer.emit(
                    ChatSpan.from_event(
                        event,
                        conversation_id=self._session_id,
                        include_content=self._include_content,
                        input_from_index=self._prev_input_length,
                    ),
                    self._child_context,
                )
                self._prev_input_length = len(event.input)
            elif isinstance(event, CompactionEvent):
                # History was rewritten; the next model input is a new stream, so
                # re-ship it in full rather than delta against the old one.
                self._prev_input_length = 0
            elif isinstance(event, ToolEvent) and self._turn is not None:
                self._writer.emit(
                    ToolSpan.from_event(
                        event,
                        conversation_id=self._session_id,
                        include_content=self._include_content,
                    ),
                    self._child_context,
                )
                if event.completed is not None:
                    self._turn_end = event.completed
        except Exception:
            logger.warning(
                "Failed to handle event for Weave agent session", exc_info=True
            )

    def finish(self, outcome: SampleOutcome | None = None) -> None:
        try:
            self._close_turn(
                outcome_attributes=outcome.to_attributes() if outcome else {},
                failed=outcome is not None and outcome.error is not None,
                error=outcome.error_message if outcome else None,
            )
        except Exception:
            logger.warning("Failed to finish Weave agent session", exc_info=True)

    def _open_turn(self, start: datetime | None) -> None:
        self._turn = TurnSpan.opened(
            agent_name=self._agent_name,
            conversation_id=self._session_id,
            conversation_name=self._session_name,
            model=self._model,
            identity_attributes=self._identity_attributes,
            turn_index=self._turn_index,
            start=start,
        )
        self._open_turn_span = self._writer.open(self._turn, Context())
        self._child_context = set_span_in_context(self._open_turn_span)
        self._turn_index += 1

    def _close_turn(
        self,
        *,
        outcome_attributes: dict[str, Any] | None = None,
        failed: bool = False,
        error: str | None = None,
    ) -> None:
        turn, open_span = self._turn, self._open_turn_span
        if turn is None or open_span is None:
            return
        turn.end_nanoseconds = _to_nanoseconds(self._turn_end)
        turn.attributes.update(outcome_attributes or {})
        turn.failed = failed
        turn.error = error
        self._reset_turn()
        self._writer.close(open_span, turn)
