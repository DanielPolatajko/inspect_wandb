# Concepts

This page explains some important concepts from Inspect and Weights & Biases, and how this integration maps one set of concepts onto the other. For most users, this level of detail will not be necessary to get value out of `inspect-wandb`, but please read on if you're interested.

## W&B Models

### What is a Run?

A `Run` in the W&B sense maps to a single `inspect` log file or log dir. That is:
- `inspect eval ...` will have a single corresponding `Run` in the W&B Models console. This is identified by the `run_id` from Inspect.
- `inspect eval-set ...` will also have a single corresponding `Run` in the W&B Models console. This is identified by the `eval_set_id` from Inspect.
    - Because the `Run` corresponds to the `eval_set_id` which persists across multiple invocations of `inspect eval-set`, as long as the log dir doesn't change, the `Run` will be updated across multiple invocations.

## W&B Weave

### What is an Evaluation?

An `Evaluation` in the Weave sense maps to a single model run on a given dataset within a given task. That is, one `Evaluation` maps to one `task_id` in Inspect.

### What is a Trace?

Traces in Weave have multiple different granularities. There are traces for individual model API calls, and there are traces capturing entire Inspect tasks. This integration adds traces for each Inspect Task, Sample, Solver and Scorer. Weave then by default adds some additional traces which go into more detail on individual model calls, and capture some evaluation statistics. Most of the traces added by our extension start with Inspect by default (except the task which has the task name), although you can customise this in the configuration.

### What is an Agent Session?

Where a `Trace` is a flat tree of calls, an **agent session** is Weave's purpose-built view of an *agent trajectory*: a conversation made up of **turns**, where each turn is one model generation plus the tool calls it triggered. Sessions populate Weave's [Agents view](https://docs.wandb.ai/weave/guides/tracking/view-agent-activity) and can be scored per-turn by server-side Monitors and Signals.

When the `agent_sessions` setting is enabled, this integration reconstructs each Inspect sample as one agent session, mapping sample → session, agent loop step → turn, model call → LLM span, and tool call → tool span. The agent is named after the task and versioned by the model (so the same task on different models compares as agent versions), and each sample becomes a named session. Turns carry rolled-up token usage and rich `inspect.*` metadata — task/eval/sample identity, sample metadata, generation config, per-tool errors and timing, and final scores — so you can filter and analyse trajectories by any of these in the Agents view. Crucially, turns are streamed to Weave **as they complete** (not in a batch at the end of the sample), so you can observe a long-horizon agent live and, paired with Weave Monitors and Automations, surface failure modes in real time. See {doc}`demo-agent-monitoring` for a walkthrough. This is complementary to, and independent of, the trace tree described above.