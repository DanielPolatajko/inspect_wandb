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

```{important}
Eval metadata can refine the integration's settings, but it cannot *bootstrap* it.
The entity and project must be resolvable before the first hook fires — from the wandb
settings file (`wandb init`), `pyproject.toml`, or the `WANDB_ENTITY` / `WANDB_PROJECT`
environment variables.

Inspect asks each hook `enabled()` before dispatching `on_task_start`, and
`on_task_start` is where eval metadata is read. If entity/project are only supplied via
metadata, the hook reports itself disabled on that first check and never receives the
metadata that would have enabled it — the run completes with

> `WandB integration disabled: missing required field(s): project, entity.`

and nothing reaches Weave.
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
        message_limit=20,
        max_attempts=1,
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

```{note}
The cap is `message_limit`, not `max_messages`. `basic_agent` forwards unrecognised
keywords into `**kwargs`, so a misspelled cap is silently ignored and the broken agent
loops until it exhausts `token_limit` instead of stopping at 20 messages.
```

Run it with the entity and project in the environment, so the integration is enabled
before the first hook fires:

```bash
WANDB_ENTITY=<your-entity> WANDB_PROJECT=<your-project> python demo_agent.py
```

Then open the **Agents view** in your Weave project. As the agent runs you'll see the
session grow one turn at a time — each turn an LLM span plus a `guess` tool span.

A run of the two variants looks like this:

| Variant | Turns | Guesses | Outcome |
| --- | --- | --- | --- |
| healthy (`BROKEN = False`) | 7 | 50, 75, 62, 68, 71, 73 | converges, scores `C` |
| broken (`BROKEN = True`) | 9 | 50, 25, 12, 6, 3, 1, 2, 75, 100 | hits the 20-message limit, scores `I` |

The broken trajectory is the interesting one: the agent binary-searches downward, bottoms
out at 1, then flails (2, 75, 100) because every response says `lower`. That is the
no-progress signature the Monitor below is looking for.

## 3. Add a Monitor

In the Weave UI, go to **Monitors → New monitor**:

- **Operations**: the agent turn spans — select the agent-turn operation
  (`weave.genai.turn_ended`).
- **Sampling rate**: 100% (so every turn is scored in the demo).
- **LLM judge / scoring prompt**: something like —
  > *Given the agent's recent turns, is the agent making no progress — repeating similar
  > actions, or looping without getting closer to the goal? Answer 1 for "stuck/no
  > progress", 0 otherwise.*

Monitor results are written to each turn's `feedback`, visible in the Signals column of
the Agents/Traces view.

### Or define it in code

Monitors are also a first-class SDK object, which makes them reviewable and reproducible
alongside the eval:

```python
import weave
from weave.flow.monitor import Monitor
from weave.scorers import LLMAsAJudgeScorer
from weave.trace_server.interface.builtin_object_classes.llm_structured_model import (
    LLMStructuredCompletionModel,
)

weave.init("<entity>/<project>")

monitor = Monitor(
    name="agent-no-progress",
    description="Scores Inspect agent turns for a stuck/no-progress loop.",
    sampling_rate=1.0,
    op_names=["weave.genai.turn_ended"],
    scorers=[
        LLMAsAJudgeScorer(
            name="no-progress-judge",
            model=LLMStructuredCompletionModel(
                llm_model_id="anthropic/claude-haiku-4-5"
            ),
            scoring_prompt=(
                "You are monitoring a long-horizon agent for a stuck / no-progress "
                "failure mode. Answer 1 if the agent is repeating the same or "
                "near-identical tool calls, receiving the same tool response without "
                "adapting, or cycling without getting closer to the goal. Else 0. "
                "Return only the integer."
            ),
        )
    ],
    # A single turn cannot reveal a loop — window the recent turns of one
    # conversation together so the judge can see the repetition.
    scorer_debounce_config={
        "aggregation_field": "thread_id",
        "aggregation_method": "all_messages",
        "timeout_seconds": 30,
    },
)

monitor.activate()
```

`op_names=["weave.genai.turn_ended"]` is the agent-span literal — it is what an
`invoke_agent` turn span becomes once ingested through the agents OTLP endpoint, and it
is passed through unexpanded rather than resolved to a `weave:///` op ref.

```{note}
`scorer_debounce_config` is doing real work here, not tuning. A no-progress judge
scoring one turn in isolation has nothing to compare against — looping is only visible
*across* turns. Windowing by `thread_id` with `all_messages` gives the judge the recent
trajectory instead of a single step.
```

`monitor.activate()` publishes it live; until then it is stored but inert. Note that an
active monitor runs its judge against every matching turn, so it draws LLM spend on an
ongoing basis.

## 4. Add an Automation

From the monitor's detail view, create an **Automation**:

- **Trigger**: monitor metric (the no-progress score) **is above** a threshold over a
  short rolling window.
- **Action**: Slack notification (or webhook) to your channel.

(Slack/webhook integrations are configured in **Team Settings**.)

```{note}
Unlike Monitors, Automations have no SDK equivalent — there is no `weave.Automation`
or equivalent client method. This step is UI-only, so it cannot be scripted or checked
into the repo alongside the monitor definition above.
```

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

- **Cost**: the demo is one short sample on Haiku — measured at ~10k tokens for the
  broken variant and ~7k for the healthy one. The monitoring scales identically to a real
  long-horizon run — only the eval is small. An active Monitor adds its own judge cost per
  turn, which is the part that scales with the run.
- **Simulating "long"**: a 20-turn looping agent is enough to demonstrate live detection;
  the same setup works unchanged on a million-token, hours-long agentic eval.
- **Token counts vs. Inspect**: the Agents view's headline **Tokens** tile counts prompt
  (input) + completion (output) tokens only. Inspect's per-sample token total additionally
  folds in prompt-cache reads and writes, so when caching is active the two totals differ
  by the cached amount — Weave will read *lower*. This is a difference in what each headline
  rolls up, not lost data: cache-read and cache-write tokens are streamed on every LLM span
  and shown separately in a conversation's **Token breakdown** panel, and the per-span
  input/output/cache figures match Inspect exactly.
- **Per-turn input is a delta**: each turn's `chat` span carries only the messages *new that
  turn*, not the whole re-shipped history — this keeps transcript volume linear rather than
  quadratic on long runs. The full conversation is still reconstructable turn-by-turn, and a
  context compaction re-ships the (compacted) history in full on the next turn. A consequence:
  a turn may show only a couple of input messages while its input-token count is large (the
  model still processed the full, possibly-cached context) — that is expected, not a bug.
