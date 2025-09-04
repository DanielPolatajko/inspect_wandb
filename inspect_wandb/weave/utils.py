from weave.evaluation.eval_imperative import ScoreType
from inspect_ai.scorer import Value
from typing import Sequence, Mapping
from logging import getLogger

utils_logger = getLogger(__name__)

def format_score_types(score_value: Value) -> ScoreType:
    if isinstance(score_value, str):
        return {"score": score_value}
    elif isinstance(score_value, int):
        return float(score_value)
    elif isinstance(score_value, Sequence):
        if len(score_value) != 1:
            raise ValueError("Sequence score cannot be passed to Weave")
        return {"score": score_value[0]}
    elif isinstance(score_value, Mapping):
        return dict(score_value)
    else:
        return score_value
    
def format_sample_display_name(template: str, task_name: str, sample_id: int | str, epoch: int) -> str:
    """
    Format sample display name using template string with safe variable substitution.
    
    Args:
        template: Template string with placeholders like {task_name}, {sample_id}, etc.
        task_name: Name of the task being evaluated
        sample_id: ID of the sample (numeric or string)
        epoch: Epoch number
        
    Returns:
        Formatted display name string
    """
    try:
        if not template.strip():
            raise ValueError("Empty template")
        return template.format(task_name=task_name, sample_id=sample_id, epoch=epoch)
    except (KeyError, ValueError):
        return f"{task_name}-sample-{sample_id}-epoch-{epoch}"
