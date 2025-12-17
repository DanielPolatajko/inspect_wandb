from inspect_ai.scorer._metric import SampleScore
from inspect_ai._eval.task.run import task_run_sample
from inspect_wandb.weave.autopatcher.plan import PatchedPlan
from inspect_wandb.weave.autopatcher.scorer import PatchedScorer
from inspect_ai.util._early_stopping import (
    EarlyStop
)
from inspect_ai.solver import Plan
from inspect_ai.scorer import Scorer

async def patched_task_run_sample(*args, **kwargs) -> dict[str, SampleScore] | EarlyStop | None:
    plan: Plan = kwargs.pop("plan")
    scorers: list[Scorer] | None = kwargs.pop("scorers")

    patched_plan = PatchedPlan(plan.steps, plan.finish, plan.cleanup, plan.name, internal=True)
    patched_scorers: list[Scorer] | None = [PatchedScorer(scorer) for scorer in scorers] if scorers else None

    return await task_run_sample(*args, plan=patched_plan, scorers=patched_scorers, **kwargs)