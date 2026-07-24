# Demo: real-time agent monitoring

This walkthrough shows how to **observe a long-horizon agent live and catch a failure
mode as it happens** using Inspect WandB's agent sessions together with Weave's
server-side Monitors and Automations.

The eval itself is small and cheap (a number-guessing agent), but the monitoring is
the real thing: turns stream into Weave as they complete, a Monitor scores each turn as
it arrives, and an Automation fires an alert the moment the agent goes off the rails —
exactly what you want for a multi-hour, high-token agentic run, without having to run
one.

## What you'll see

1. An agent's trajectory filling the Weave **Agents view** turn-by-turn, live.
2. A **Monitor** (an LLM judge) scoring each turn for "no progress / stuck in a loop".
3. An **Automation** posting a Slack alert when the no-progress score crosses a threshold.

## Prerequisites

- `pip install "inspect-wandb[weave]"` and a Weave-enabled wandb project
  (`wandb login`, or set `WANDB_API_KEY`).
- A model provider key (the example uses `anthropic/claude-haiku-4-5` — cheap).

## 1. Enable agent sessions

Agent sessions are opt-in. Enable per-run via eval metadata:

```bash
--metadata inspect_wandb_weave_agent_sessions=true
```

or for a project, in `pyproject.toml`:

```toml
[tool.inspect-wandb.weave]
agent_sessions = true
```

## 2. The demo agent

Two variants of a number-guessing agent. The **healthy** tool gives correct
higher/lower feedback; the **broken** tool always says `"lower"`, so the agent never
converges and spins until it hits its message limit — a deterministic *no-progress
loop* to surface.

```python
# demo_agent.py
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import tool

SECRET = 73
BROKEN = True  # flip to False for the healthy run

@tool
def guess():
    async def execute(number: int) -> str:
        """Guess the secret number between 1 and 100.

        Args:
            number: Your integer guess between 1 and 100.

        Returns:
            'higher', 'lower', or 'correct'.
        """
        if BROKEN:
            return "lower"  # adversarial: never converges
        if number < SECRET:
            return "higher"
        if number > SECRET:
            return "lower"
        return "correct"

    return execute

task = Task(
    dataset=[Sample(
        input="Find the secret number between 1 and 100 by calling guess(number) "
              "with a binary search. Submit it once guess returns 'correct'.",
        target="73",
    )],
    solver=basic_agent(
        init=system_message("Use the guess tool with binary search; reason step by "
                            "step; submit once correct."),
        tools=[guess()],
        max_messages=20,
    ),
    scorer=includes(),
)

if __name__ == "__main__":
    inspect_eval(
        task,
        model="anthropic/claude-haiku-4-5",
        token_limit=40000,
        metadata={"inspect_wandb_weave_agent_sessions": True},
    )
```

Run it and open the **Agents view** in your Weave project. As the agent runs you'll see
the session grow one turn at a time — each turn an LLM span plus a `guess` tool span.

## 3. Add a Monitor

In the Weave UI, go to **Monitors → New monitor**:

- **Operations**: the agent turn spans (`invoke_agent`).
- **Sampling rate**: 100% (so every turn is scored in the demo).
- **LLM judge / scoring prompt**: something like —
  > *Given this agent turn, is the agent making no progress — repeating similar actions,
  > or looping without getting closer to the goal? Answer 1 for "stuck/no progress",
  > 0 otherwise.*

Monitor results are written to each turn's `feedback`, visible in the Signals column of
the Agents/Traces view.

## 4. Add an Automation

From the monitor's detail view, create an **Automation**:

- **Trigger**: monitor metric (the no-progress score) **is above** a threshold over a
  short rolling window.
- **Action**: Slack notification (or webhook) to your channel.

(Slack/webhook integrations are configured in **Team Settings**.)

## 5. Watch it fire

Run the **broken** variant (`BROKEN = True`). The agent loops; each looping turn streams
in live; the Monitor scores them as no-progress; once the rolling score crosses the
threshold the Automation posts an alert — e.g. *"⚠️ agent stuck in a no-progress loop"* —
mid-run, not after.

Flip `BROKEN = False` for the healthy run to contrast: the agent converges in ~7 turns
and the monitor stays quiet.

## Stretch: intervene, don't just alert

Inspect (>= 0.3.225) can **observe a running agent, interrupt it, and redirect it with
follow-up messages**. Combined with the live monitor signal, this closes the loop:
detect the failure mode → intervene (redirect or stop the sample) instead of letting it
burn budget. Full intervention support in Inspect WandB is tracked as follow-up work.

## Notes

- **Cost**: the demo is one short sample on Haiku (a few thousand tokens). The monitoring
  scales identically to a real long-horizon run — only the eval is small.
- **Simulating "long"**: a 20-turn looping agent is enough to demonstrate live detection;
  the same setup works unchanged on a million-token, hours-long agentic eval.
