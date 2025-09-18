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