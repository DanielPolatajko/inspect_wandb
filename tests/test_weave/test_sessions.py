from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)

from inspect_wandb.weave.sessions import (
    AgentSessionEmitter,
    model_event_to_llm,
    to_messages,
    tool_event_to_tool,
)

T0 = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 6, 21, 12, 0, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 6, 21, 12, 0, 2, tzinfo=timezone.utc)


def make_model_event(
    completion: str = "thinking",
    input_tokens: int = 100,
    output_tokens: int = 20,
    timestamp: datetime = T0,
    completed: datetime = T1,
) -> ModelEvent:
    return ModelEvent(
        model="anthropic/claude-haiku-4-5",
        input=[ChatMessageUser(content="solve the task")],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(temperature=0.5, max_tokens=1024),
        output=ModelOutput(
            model="anthropic/claude-haiku-4-5",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=completion),
                    stop_reason="tool_calls",
                )
            ],
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        ),
        timestamp=timestamp,
        completed=completed,
    )


def make_tool_event(
    tool_id: str = "call_1",
    function: str = "bash",
    result: str = "command output",
    timestamp: datetime = T1,
    completed: datetime = T2,
) -> ToolEvent:
    return ToolEvent(
        id=tool_id,
        function=function,
        arguments={"cmd": "ls -la"},
        result=result,
        timestamp=timestamp,
        completed=completed,
    )


class TestPureMappers:
    def test_to_messages_maps_roles(self) -> None:
        # Given
        messages = [
            ChatMessageUser(content="hello"),
            ChatMessageAssistant(content="hi there"),
        ]

        # When
        result = to_messages(messages)

        # Then
        assert [(m.role, m.content) for m in result] == [
            ("user", "hello"),
            ("assistant", "hi there"),
        ]

    def test_model_event_maps_usage_timing_and_provider(self) -> None:
        # Given
        event = make_model_event(input_tokens=321, output_tokens=99)

        # When
        llm = model_event_to_llm(event)

        # Then
        assert llm.model == "anthropic/claude-haiku-4-5"
        assert llm.provider_name == "anthropic"
        assert llm.usage.input_tokens == 321
        assert llm.usage.output_tokens == 99
        assert llm.finish_reasons == ["tool_calls"]
        assert llm.request_temperature == 0.5
        assert llm.request_max_tokens == 1024
        assert llm.started_at == T0
        assert llm.ended_at == T1

    def test_tool_event_maps_fields(self) -> None:
        # Given
        event = make_tool_event(tool_id="call_42", function="python")

        # When
        tool = tool_event_to_tool(event)

        # Then
        assert tool.name == "python"
        assert tool.tool_call_id == "call_42"
        assert '"cmd"' in tool.arguments
        assert tool.result == "command output"
        assert tool.started_at == T1
        assert tool.ended_at == T2

    def test_tool_result_is_truncated(self) -> None:
        # Given
        event = make_tool_event(result="x" * 10000)

        # When
        tool = tool_event_to_tool(event)

        # Then
        assert tool.result.endswith("…[truncated]")
        assert len(tool.result) < 10000


class TestAgentSessionEmitter:
    def _emitter(self) -> AgentSessionEmitter:
        return AgentSessionEmitter(
            session_id="sample-uuid",
            session_name="task-sample-1",
            agent_name="my_agent",
            model="anthropic/claude-haiku-4-5",
        )

    def test_segments_turns_on_model_event_boundary(self) -> None:
        # Given
        emitter = self._emitter()
        events = [
            make_model_event(),
            make_tool_event(tool_id="a"),
            make_tool_event(tool_id="b"),
            make_model_event(),
            make_tool_event(tool_id="c"),
        ]

        # When
        with patch("inspect_wandb.weave.sessions.log_turn") as mock_log_turn:
            for event in events:
                emitter.handle_event(event)
            emitter.finish()

        # Then
        assert mock_log_turn.call_count == 2
        first_spans = mock_log_turn.call_args_list[0].kwargs["spans"]
        second_spans = mock_log_turn.call_args_list[1].kwargs["spans"]
        assert len(first_spans) == 3
        assert len(second_spans) == 2
        assert all(
            call.kwargs["session_id"] == "sample-uuid"
            for call in mock_log_turn.call_args_list
        )

    def test_final_turn_flushed_on_finish(self) -> None:
        # Given
        emitter = self._emitter()

        # When
        with patch("inspect_wandb.weave.sessions.log_turn") as mock_log_turn:
            emitter.handle_event(make_model_event())
            assert mock_log_turn.call_count == 0
            emitter.finish()

        # Then
        assert mock_log_turn.call_count == 1

    def test_tool_event_before_model_event_is_ignored(self) -> None:
        # Given
        emitter = self._emitter()

        # When
        with patch("inspect_wandb.weave.sessions.log_turn") as mock_log_turn:
            emitter.handle_event(make_tool_event())
            emitter.finish()

        # Then
        assert mock_log_turn.call_count == 0

    def test_emit_failure_is_swallowed(self) -> None:
        # Given
        emitter = self._emitter()

        # When / Then
        with patch(
            "inspect_wandb.weave.sessions.log_turn",
            side_effect=RuntimeError("weave down"),
        ):
            emitter.handle_event(make_model_event())
            emitter.finish()  # must not raise


class TestHookWiring:
    @pytest.mark.asyncio
    async def test_agent_sessions_disabled_creates_no_emitter(self) -> None:
        # Given
        from inspect_wandb.config.settings import WeaveSettings
        from inspect_wandb.weave.hooks import WeaveEvaluationHooks

        hooks = WeaveEvaluationHooks()
        hooks._hooks_enabled = True
        hooks.settings = WeaveSettings(
            enabled=True,
            entity="e",
            project="p",
            agent_sessions=False,
        )

        # When / Then
        assert hooks._agent_sessions_active() is False
