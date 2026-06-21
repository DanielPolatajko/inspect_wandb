import json
from logging import getLogger

from inspect_ai.event import Event, ModelEvent, ToolEvent
from inspect_ai.model import ChatMessage
from weave.session.session import LLM, SubAgent, Tool, log_turn
from weave.session.types import Message, Usage

logger = getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 4000

_WEAVE_ROLES = {"user", "assistant", "system", "tool"}


def _provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def to_messages(messages: list[ChatMessage]) -> list[Message]:
    return [
        Message(
            role=m.role if m.role in _WEAVE_ROLES else "user",
            content=m.text or "",
        )
        for m in messages
    ]


def _usage(event: ModelEvent) -> Usage:
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


def model_event_to_llm(event: ModelEvent) -> LLM:
    config = event.config
    return LLM(
        model=event.model,
        provider_name=_provider(event.model),
        usage=_usage(event),
        input_messages=to_messages(event.input),
        output_messages=to_messages([event.output.choices[0].message])
        if event.output.choices
        else [Message.assistant(event.output.completion)],
        finish_reasons=[
            choice.stop_reason for choice in event.output.choices if choice.stop_reason
        ],
        request_temperature=config.temperature,
        request_max_tokens=config.max_tokens,
        request_top_p=config.top_p,
        started_at=event.timestamp,
        ended_at=event.completed,
    )


def tool_event_to_tool(event: ToolEvent) -> Tool:
    result = str(event.result)
    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS] + "…[truncated]"
    return Tool(
        name=event.function,
        arguments=json.dumps(event.arguments, default=str),
        result=result,
        tool_call_id=event.id,
        started_at=event.timestamp,
        ended_at=event.completed,
    )


class AgentSessionEmitter:
    """Reconstructs an Inspect sample's agent trajectory into a Weave agent
    session, streaming each turn to the Agents view as it completes.

    A turn is one model generation plus the tool calls it triggered. Fed
    Inspect events in order via `handle_event`; a new `ModelEvent` closes the
    open turn (emitted via `log_turn`) and starts the next. `finish` flushes
    the final turn. Emission failures are logged and never propagate, so they
    cannot interfere with the eval run.
    """

    def __init__(
        self, *, session_id: str, session_name: str, agent_name: str, model: str
    ) -> None:
        self._session_id = session_id
        self._session_name = session_name
        self._agent_name = agent_name
        self._model = model
        self._open_spans: list[LLM | Tool | SubAgent] = []

    def handle_event(self, event: Event) -> None:
        try:
            if isinstance(event, ModelEvent):
                self._flush_turn()
                self._open_spans = [model_event_to_llm(event)]
            elif isinstance(event, ToolEvent) and self._open_spans:
                self._open_spans.append(tool_event_to_tool(event))
        except Exception:
            logger.warning(
                "Failed to handle event for Weave agent session", exc_info=True
            )

    def finish(self) -> None:
        self._flush_turn()

    def _flush_turn(self) -> None:
        if not self._open_spans:
            return
        spans = self._open_spans
        self._open_spans = []
        try:
            log_turn(
                session_id=self._session_id,
                session_name=self._session_name,
                agent_name=self._agent_name,
                model=self._model,
                spans=spans,
                started_at=spans[0].started_at,
                ended_at=spans[-1].ended_at,
            )
        except Exception:
            logger.warning("Failed to emit Weave agent turn", exc_info=True)
