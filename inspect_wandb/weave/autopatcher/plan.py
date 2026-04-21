from typing import Any

from weave import op as weave_op
from inspect_ai.solver import Generate, Plan, TaskState
from inspect_ai.solver._task_state import state_jsonable
from inspect_ai.solver._transcript import solver_transcript
from inspect_ai.solver._plan import logger
from inspect_ai._util.registry import registry_info


def _postprocess_solver_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        result = {}
        if isinstance(inputs.get("state"), TaskState):
            result["state"] = state_jsonable(inputs["state"])
        return result
    except Exception:
        return inputs


def _postprocess_solver_output(output: Any) -> Any:
    try:
        if isinstance(output, TaskState):
            return state_jsonable(output)
        return output
    except Exception:
        return output


class PatchedPlan(Plan):
    async def __call__(self, state: TaskState, generate: Generate) -> TaskState:
        try:
            for _, solver in enumerate(self.steps):

                async with solver_transcript(solver, state) as st:
                    solver_name = registry_info(solver).name
                    state = await weave_op(
                        name=solver_name,
                        postprocess_inputs=_postprocess_solver_inputs,
                        postprocess_output=_postprocess_solver_output,
                    )(solver)(state, generate)
                    st.complete(state)

                if state.completed:
                    break

            if self.finish:
                async with solver_transcript(self.finish, state) as st:
                    finish_name = registry_info(self.finish).name
                    state = await weave_op(
                        name=finish_name,
                        postprocess_inputs=_postprocess_solver_inputs,
                        postprocess_output=_postprocess_solver_output,
                    )(self.finish)(state, generate)
                    st.complete(state)

        finally:
            if self.cleanup:
                try:
                    await weave_op(
                        name="inspect_sample_cleanup",
                        postprocess_inputs=_postprocess_solver_inputs,
                        postprocess_output=_postprocess_solver_output,
                    )(self.cleanup)(state)
                except Exception as ex:
                    logger.warning(
                        f"Exception occurred during plan cleanup: {ex}", exc_info=ex
                    )

        return state
