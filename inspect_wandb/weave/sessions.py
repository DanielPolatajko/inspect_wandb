import json
from datetime import datetime
from logging import getLogger
from typing import Any

from inspect_ai.event import CompactionEvent, Event, ModelEvent, ToolEvent
from inspect_ai.model import ChatMessage
from inspect_ai.scorer import Score

logger = getLogger(__name__)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.context import Context
    from opentelemetry.trace import set_span_in_context
    from weave.session.session_otel import (
        execute_tool_attributes,
        invoke_agent_attributes,
        llm_attributes,
    )
    from weave.session.types import Message, Usage

    SESSIONS_AVAILABLE = True
except Exception:  # pragma: no cover - guards against weave internal changes
    SESSIONS_AVAILABLE = False
    logger.warning(
        "Weave agent sessions unavailable: incompatible weave version", exc_info=True
    )

MAX_TOOL_RESULT_CHARS = 4000
MAX_ATTRIBUTE_VALUE_CHARS = 16000
_WEAVE_ROLES = {"user", "assistant", "system", "tool"}
_TRACER_NAME = "weave.session"


def _to_nanoseconds(event_time: datetime | None) -> int | None:
    return (
        int(event_time.timestamp() * 1_000_000_000) if event_time is not None else None
    )


def _provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def _coerce(value: Any) -> str | int | float | bool | None:
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
        coerced = _coerce(raw)
        if coerced is not None and coerced != "":
            out[f"inspect.{key}"] = coerced
    return out


def flatten_metadata(metadata: Any, prefix: str = "metadata") -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {f"{prefix}.{key}": value for key, value in metadata.items()}


def to_messages(messages: list[ChatMessage]) -> list[Message]:
    return [
        Message(
            role=message.role if message.role in _WEAVE_ROLES else "user",
            content=message.text or "",
        )
        for message in messages
    ]


def usage_from_event(event: ModelEvent) -> Usage:
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


def llm_span_attributes(
    event: ModelEvent,
    *,
    conversation_id: str,
    include_content: bool = True,
    input_from_index: int = 0,
) -> dict[str, Any]:
    config = event.config
    output = event.output
    output_messages = (
        to_messages([output.choices[0].message])
        if output.choices
        else [Message.assistant(output.completion)]
    )
    base = llm_attributes(
        model=event.model,
        provider_name=_provider(event.model),
        conversation_id=conversation_id,
        input_messages=to_messages(event.input[input_from_index:])
        if include_content
        else None,
        output_messages=output_messages if include_content else None,
        usage=usage_from_event(event),
        finish_reasons=[
            choice.stop_reason for choice in output.choices if choice.stop_reason
        ],
        response_model=output.model or "",
        request_temperature=config.temperature,
        request_max_tokens=config.max_tokens,
        request_top_p=config.top_p,
        request_frequency_penalty=config.frequency_penalty,
        request_presence_penalty=config.presence_penalty,
        request_seed=config.seed,
        request_stop_sequences=config.stop_seqs,
    )
    extra = _inspect_attributes(
        {
            "generate.top_k": config.top_k,
            "generate.reasoning_effort": config.reasoning_effort,
            "model.retries": event.retries,
            "model.cache": event.cache,
            "model.error": event.error,
        }
    )
    return {**base, **extra}


def tool_span_attributes(
    event: ToolEvent, *, conversation_id: str, include_content: bool = True
) -> dict[str, Any]:
    result = str(event.result)
    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS] + "…[truncated]"
    base = execute_tool_attributes(
        tool_name=event.function,
        conversation_id=conversation_id,
        tool_call_arguments=json.dumps(event.arguments, default=str)
        if include_content
        else "",
        tool_call_result=result if include_content else "",
        tool_call_id=event.id,
    )
    extra = _inspect_attributes(
        {
            "tool.error": getattr(event.error, "message", None)
            if event.error
            else None,
            "tool.failed": event.failed or None,
            "tool.truncated": event.truncated is not None,
            "tool.working_time": event.working_time,
        }
    )
    return {**base, **extra}


def _set_attributes(span: Any, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None and value != "":
            span.set_attribute(key, value)


def _start_span(
    tracer: Any,
    name: str,
    parent_context: Any,
    start_nanoseconds: int | None,
    attributes: dict[str, Any],
) -> Any:
    """Start a span and leave it open. The caller must call ``_end_span``."""
    span = (
        tracer.start_span(name, context=parent_context, start_time=start_nanoseconds)
        if start_nanoseconds is not None
        else tracer.start_span(name, context=parent_context)
    )
    _set_attributes(span, attributes)
    return span


def _end_span(
    span: Any, end_nanoseconds: int | None, attributes: dict[str, Any] | None = None
) -> None:
    if attributes:
        _set_attributes(span, attributes)
    span.end(end_time=end_nanoseconds) if end_nanoseconds is not None else span.end()


def _emit_span(
    tracer: Any,
    name: str,
    parent_context: Any,
    start_nanoseconds: int | None,
    end_nanoseconds: int | None,
    attributes: dict[str, Any],
) -> Any:
    """Emit a complete (already finished) span: start and end it immediately."""
    span = _start_span(tracer, name, parent_context, start_nanoseconds, attributes)
    _end_span(span, end_nanoseconds)
    return span


class AgentSessionEmitter:
    """Reconstructs an Inspect sample's agent trajectory and streams it to
    Weave's agent Session SDK as gen_ai OpenTelemetry spans, one turn at a time.

    Emits the spans directly via the weave-configured global tracer (rather than
    weave's imperative ``log_turn``) so we can attach rich ``inspect.*`` metadata
    and preserve the original Inspect event timestamps.

    Emission is *incremental*: the ``invoke_agent`` turn span is opened when the
    turn starts and each child (``chat``/``execute_tool``) span is emitted as its
    event arrives, rather than buffering the whole turn and emitting at turn end.
    Because the OTel batch processor exports a span only once it ends, emitting
    children eagerly is what makes an in-progress turn observable in the Agents
    view — a turn stuck on a slow or hung tool shows its completed steps instead
    of nothing at all. Weave groups those children into the conversation by
    ``gen_ai.conversation.id`` and nests them under the turn once it closes.

    Token usage is carried on the child ``chat`` spans only, matching weave's own
    contract (its ``invoke_agent`` spans never carry ``gen_ai.usage.*``); weave
    rolls usage up onto the turn, so also setting it on the turn span would
    double-count in the Agents view. A turn is one model generation plus the tool
    calls it triggered; a new ``ModelEvent`` closes the open turn and starts the
    next, and ``finish`` closes the last turn with sample outcome metadata
    attached.

    Each ``chat`` span's ``input.messages`` carries only the messages *new that
    turn* (``event.input`` sliced from the previous turn's length), not the whole
    re-shipped history — keeping aggregate transcript volume ~O(n) instead of
    O(n²) on long-horizon runs. Token counts are unchanged (they reflect the real
    API call). A ``CompactionEvent`` rewrites the message stream, so it resets the
    slice offset and the next turn re-ships its full (compacted) input. All
    emission is best-effort: failures are logged and never propagate into the eval
    run.
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
    ) -> None:
        self._session_id = session_id
        self._session_name = session_name
        self._agent_name = agent_name
        self._model = model
        self._identity_attributes = _inspect_attributes(identity)
        self._include_content = include_content
        self._turn_index = 0
        self._prev_input_length = 0
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._turn_span: Any = None
        self._child_context: Any = None
        self._turn_end: datetime | None = None

    def handle_event(self, event: Event) -> None:
        if not SESSIONS_AVAILABLE:
            return
        try:
            if isinstance(event, ModelEvent):
                self._close_turn()
                self._open_turn(event.timestamp)
                self._turn_end = event.completed or event.timestamp
                self._emit_child(
                    f"chat {event.model}",
                    llm_span_attributes(
                        event,
                        conversation_id=self._session_id,
                        include_content=self._include_content,
                        input_from_index=self._prev_input_length,
                    ),
                    _to_nanoseconds(event.timestamp),
                    _to_nanoseconds(event.completed),
                )
                self._prev_input_length = len(event.input)
            elif isinstance(event, CompactionEvent):
                # History was rewritten; the next model input is a new stream, so
                # re-ship it in full rather than delta against the old one.
                self._prev_input_length = 0
            elif isinstance(event, ToolEvent) and self._turn_span is not None:
                self._emit_child(
                    f"execute_tool {event.function}",
                    tool_span_attributes(
                        event,
                        conversation_id=self._session_id,
                        include_content=self._include_content,
                    ),
                    _to_nanoseconds(event.timestamp),
                    _to_nanoseconds(event.completed),
                )
                if event.completed is not None:
                    self._turn_end = event.completed
        except Exception:
            logger.warning(
                "Failed to handle event for Weave agent session", exc_info=True
            )

    def finish(self, outcome: dict[str, Any] | None = None) -> None:
        if not SESSIONS_AVAILABLE:
            return
        try:
            self._close_turn(outcome=_inspect_attributes(outcome) if outcome else {})
        except Exception:
            logger.warning("Failed to finish Weave agent session", exc_info=True)

    def _open_turn(self, start: datetime | None) -> None:
        turn_attributes = {
            **invoke_agent_attributes(
                agent_name=self._agent_name,
                conversation_id=self._session_id,
                conversation_name=self._session_name,
                model=self._model,
                agent_version=self._model,
            ),
            **self._identity_attributes,
            "inspect.turn_index": self._turn_index,
        }
        self._turn_span = _start_span(
            otel_trace.get_tracer(_TRACER_NAME),
            f"invoke_agent {self._agent_name}",
            Context(),
            _to_nanoseconds(start),
            turn_attributes,
        )
        self._child_context = set_span_in_context(self._turn_span)
        self._turn_index += 1

    def _emit_child(
        self,
        name: str,
        attributes: dict[str, Any],
        start_nanoseconds: int | None,
        end_nanoseconds: int | None,
    ) -> None:
        _emit_span(
            otel_trace.get_tracer(_TRACER_NAME),
            name,
            self._child_context,
            start_nanoseconds,
            end_nanoseconds,
            attributes,
        )

    def _close_turn(self, outcome: dict[str, Any] | None = None) -> None:
        if self._turn_span is None:
            return
        turn_span, turn_end = self._turn_span, self._turn_end
        self._reset_turn()
        _end_span(turn_span, _to_nanoseconds(turn_end), outcome or None)


def build_outcome(sample: Any) -> dict[str, Any]:
    """Build sample-outcome metadata (known only at sample end) for the final turn."""
    outcome: dict[str, Any] = {
        "total_time": sample.total_time,
        "working_time": getattr(sample, "working_time", None),
        "error": sample.error,
        "limit": getattr(sample, "limit", None),
    }
    scores: dict[str, Score] | None = sample.scores
    if scores:
        for name, score in scores.items():
            outcome[f"score.{name}"] = score.value
            if score.answer:
                outcome[f"score.{name}.answer"] = score.answer
    usages = getattr(sample, "model_usage", None) or {}
    total_tokens = sum(
        (usage.total_tokens or 0)
        for usage in usages.values()
        if usage.total_tokens is not None
    )
    if total_tokens:
        outcome["total_tokens"] = total_tokens
    return outcome
