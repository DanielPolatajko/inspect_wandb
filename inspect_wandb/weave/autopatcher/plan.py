import weave
from inspect_ai.solver import Generate, Plan, TaskState
from inspect_ai.solver._transcript import solver_transcript
from inspect_ai.solver._plan import logger
from inspect_ai._util.registry import registry_info

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
