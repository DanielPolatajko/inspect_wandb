from __future__ import annotations
from typing import Any, TypeVar, Union
import logging

import weave
from weave.trace.context import call_context
from weave.trace.weave_client import Call
from weave.evaluation.eval_imperative import  EvaluationLogger, current_predict_call, IMPERATIVE_EVAL_MARKER
from weave.evaluation.eval_imperative import ScoreLogger, _set_current_output, _set_current_summary
from weave.trace.api import attributes





T = TypeVar("T")
ID = str
ScoreType = Union[float, bool, dict]

logger = logging.getLogger(__name__)

class CustomEvaluationLogger(EvaluationLogger):
    """
    This class is a modified version of the EvaluationLogger class which allows for the parent call to be specified.
    This allows us to specify an Inspect specific call as the parent when autopatching Inspect.
    """

    def log_prediction(self, inputs: dict, output: Any, parent_call: Call | None = None) -> ScoreLogger:
        """Log a prediction to the Evaluation, and return a reference.

        The reference can be used to log scores which are attached to the specific
        prediction instance."""
        # Use set_call_stack to temporarily set the evaluation as the parent
        assert self._evaluate_call is not None
        call_stack = [self._evaluate_call] if parent_call is None else [parent_call]

        with call_context.set_call_stack(call_stack):
            # Make the prediction call
            with _set_current_output(output):
                with weave.attributes(IMPERATIVE_EVAL_MARKER):
                    _, predict_and_score_call = (
                        self._pseudo_evaluation.predict_and_score.call(
                            self._pseudo_evaluation,
                            self.model,
                            inputs,
                            __require_explicit_finish=True,
                        )
                    )

            # Get the predict_call from the context variable
            predict_call = current_predict_call.get()
            if predict_call is None:
                raise ValueError("predict_call should not be None")

            pred = ScoreLogger(
                predict_and_score_call=predict_and_score_call,
                evaluate_call=parent_call if parent_call is not None else self._evaluate_call,
                predict_call=predict_call,
            )
            self._accumulated_predictions.append(pred)
            return pred
        
    def log_summary(
        self,
        summary: dict | None = None,
        auto_summarize: bool = True,
    ) -> None:
        """Log a summary dict to the Evaluation.

        This will calculate the summary, call the summarize op, and then finalize
        the evaluation, meaning no more predictions or scores can be logged.
        """
        # We patch this function because the original implamentation contains redundancy logic when calculating the summary
        # Note: auto_summarize parameter is ignored in this implementation
        # We always use the user-provided summary directly
        if self._is_finalized:
            logger.warning("(NO-OP): Evaluation already finalized, cannot log summary.")
            return

        final_summary = {"scoring": summary or {}}

        # Call the summarize op
        assert self._evaluate_call is not None, (
            "Evaluation call should exist for summary"
        )

        # Use set_call_stack to temporarily set the evaluation as the parent
        with call_context.set_call_stack([self._evaluate_call]):
            try:
                with _set_current_summary(final_summary):
                    with attributes(IMPERATIVE_EVAL_MARKER):
                        self._pseudo_evaluation.summarize()
            except Exception:
                logger.error("Error during execution of summarize op.", exc_info=True)
                # Even if summarize fails, try to finalize with the calculated summary

        self._finalize_evaluation(output=final_summary)
