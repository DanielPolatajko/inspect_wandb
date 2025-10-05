import weave

from inspect_ai.scorer import Scorer, Target
from inspect_ai.scorer._metric import Score
from inspect_ai.solver._plan import logger
from inspect_ai._util.registry import registry_info, is_registry_object, set_registry_info
from weave.trace.context import call_context
from inspect_ai.solver import TaskState

class PatchedScorer:
    """A scorer wrapper that creates individual Weave traces for each scoring operation."""
    
    def __init__(self, original_scorer: Scorer):
        self.original_scorer = original_scorer
        
        self.scorer_name = registry_info(original_scorer).name
        
        # Copy registry information from original scorer to this instance
        if is_registry_object(original_scorer):
            set_registry_info(self, registry_info(original_scorer))
    
    async def __call__(self, state: TaskState, target: Target) -> Score | None:
        """Execute the scorer with Weave tracing under the current sample context."""
        current_call = call_context.get_current_call()
        
        # Try to find the sample call for this specific sample
        if current_call and hasattr(current_call, '_children'):
            sample_calls = [child for child in current_call._children 
                           if hasattr(child, 'attributes') and 
                           child.attributes is not None and
                           child.attributes.get('sample_id') == state.sample_id and child.attributes.get("epoch") == state.epoch]
            
            if sample_calls:
                if len(sample_calls) > 1:
                    logger.warning(f"Found multiple sample calls for sample {state.sample_id} and epoch {state.epoch}. This could result in an incorrect Weave trace tree.")
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