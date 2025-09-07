import importlib

import weave
from weave.integrations.patcher import SymbolPatcher, MultiPatcher
from weave.trace.autopatch import AutopatchSettings, IntegrationSettings
from pydantic import Field
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
from inspect_ai.scorer import Scorer, Target
from inspect_ai.scorer._metric import Score
from inspect_ai._eval.task.run import EvalSampleSource, SampleErrorHandler
from inspect_ai.solver._transcript import solver_transcript
from inspect_ai.solver._plan import logger
from inspect_ai._util.registry import registry_info, is_registry_object, set_registry_info
from weave.trace.context import call_context

class PatchedPlan(Plan):
    async def __call__(self, state: TaskState, generate: Generate) -> TaskState:
        try:
            # execute steps
            for _, solver in enumerate(self.steps):

                # run solver
                async with solver_transcript(solver, state) as st:
                    solver_name = registry_info(solver).name
                    state = await weave.op(name=solver_name)(solver)(state, generate)
                    st.complete(state)

                # check for completed
                if state.completed:
                    # exit loop
                    break

            # execute finish
            if self.finish:
                async with solver_transcript(self.finish, state) as st:
                    finish_name = registry_info(self.finish).name
                    state = await weave.op(name=finish_name)(self.finish)(state, generate)
                    st.complete(state)

        finally:
            # always do cleanup if we have one
            if self.cleanup:
                try:
                    await weave.op(name="inspect_sample_cleanup")(self.cleanup)(state)
                except Exception as ex:
                    logger.warning(
                        f"Exception occurred during plan cleanup: {ex}", exc_info=ex
                    )

        return state

class PatchedScorer:
    """A scorer wrapper that creates individual Weave traces for each scoring operation."""
    
    def __init__(self, original_scorer: Scorer):
        self.original_scorer = original_scorer
        
        self.scorer_name = registry_info(original_scorer).name
        
        # Copy registry information from original scorer to this instance
        if is_registry_object(original_scorer):
            set_registry_info(self, registry_info(original_scorer))
    
    async def __call__(self, state: TaskState, target: Target) -> Score:
        """Execute the scorer with Weave tracing under the current sample context."""
        current_call = call_context.get_current_call()
        
        # Try to find the sample call for this specific sample
        if current_call and hasattr(current_call, '_children'):
            sample_calls = [child for child in current_call._children 
                           if hasattr(child, 'attributes') and 
                           child.attributes is not None and
                           child.attributes.get('sample_id') == state.sample_id]
            
            if sample_calls:
                sample_call = sample_calls[0]
                # Manually activate this sample call as the context
                call_context.push_call(sample_call)
                try:
                    result = await weave.op(name=f"scorer_{self.scorer_name}")(self.original_scorer)(state, target)
                    return result
                finally:
                    call_context.pop_call(sample_call.id)
        
        # Fallback to original behavior
        return await weave.op(name=f"scorer_{self.scorer_name}")(self.original_scorer)(state, target)

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


inspect_patcher = MultiPatcher(
    [
        SymbolPatcher(
            lambda: importlib.import_module("inspect_ai._eval.task.run"),
            "task_run_sample",
            lambda *_, **__: patched_task_run_sample,
        ),
    ]
)

def get_inspect_patcher(settings: IntegrationSettings | None = None) -> MultiPatcher:
    return inspect_patcher

class CustomAutopatchSettings(AutopatchSettings):
    inspect: IntegrationSettings = Field(default_factory=IntegrationSettings)

def autopatch_inspect(settings: CustomAutopatchSettings) -> None:
    get_inspect_patcher(settings.inspect).attempt_patch()

def reset_autopatch_inspect() -> None:
    get_inspect_patcher().undo_patch()