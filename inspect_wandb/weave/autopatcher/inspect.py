from typing import Callable

import anyio
from inspect_ai.dataset import Sample
from inspect_ai.log import (
    EvalError,
)
from inspect_ai.scorer._metric import SampleScore
from inspect_ai.solver import Generate, Plan, TaskState
from inspect_ai.util._sandbox.environment import SandboxEnvironmentSpec
from inspect_ai._eval.task.run import task_run_sample
from inspect_ai._eval.task.log import TaskLogger
from inspect_ai._eval.task.run import EvalSampleSource, SampleErrorHandler
from inspect_ai.scorer import Scorer
from inspect_wandb.weave.autopatcher.plan import PatchedPlan
from inspect_wandb.weave.autopatcher.scorer import PatchedScorer

async def patched_task_run_sample(
    *,
    task_name: str,
    log_location: str,
    sample: Sample,
    state: TaskState,
    sandbox: SandboxEnvironmentSpec | None,
    max_sandboxes: int | None,
    sandbox_cleanup: bool,
    plan: Plan,
    scorers: list[Scorer] | None,
    generate: Generate,
    progress: Callable[[int], None],
    logger: TaskLogger | None,
    log_images: bool,
    sample_source: EvalSampleSource | None,
    sample_error: SampleErrorHandler,
    sample_complete: Callable[[dict[str, SampleScore]], None],
    fails_on_error: bool,
    retry_on_error: int,
    error_retries: list[EvalError],
    time_limit: int | None,
    working_limit: int | None,
    semaphore: anyio.Semaphore | None,
    eval_set_id: str | None,
    run_id: str,
    task_id: str,
) -> dict[str, SampleScore] | None:
        patched_plan = PatchedPlan(plan.steps, plan.finish, plan.cleanup, plan.name, internal=True)
        
        # Create patched scorers using PatchedScorer class
        if scorers:
            patched_scorers: list[Scorer] | None = [
                PatchedScorer(scorer)
                for scorer in scorers
            ]
        else:
            patched_scorers = None

        return await task_run_sample(
            task_name=task_name,
            log_location=log_location,
            sample=sample,
            state=state,
            sandbox=sandbox,
            max_sandboxes=max_sandboxes,
            sandbox_cleanup=sandbox_cleanup,
            plan=patched_plan,
            scorers=patched_scorers,
            generate=generate,
            progress=progress,
            logger=logger,
            log_images=log_images,
            sample_source=sample_source,
            sample_error=sample_error,
            sample_complete=sample_complete,
            fails_on_error=fails_on_error,
            retry_on_error=retry_on_error,
            error_retries=error_retries,
            time_limit=time_limit,
            working_limit=working_limit,
            semaphore=semaphore,
            eval_set_id=eval_set_id,
            run_id=run_id,
            task_id=task_id,
        )