from __future__ import annotations
from typing import TypeVar, Union
import logging

from weave.trace.context import call_context
from weave.evaluation.eval_imperative import  EvaluationLogger, IMPERATIVE_EVAL_MARKER
from weave.evaluation.eval_imperative import _set_current_summary
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

        final_summary = {"summary": summary or {}}

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
