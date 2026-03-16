from inspect_wandb.config.extras_manager import INSTALLED_EXTRAS
from inspect_wandb.providers import wandb_models_hooks
if INSTALLED_EXTRAS["weave"]:
    from inspect_wandb.providers import weave_evaluation_hooks
    __all__ = ["weave_evaluation_hooks", "wandb_models_hooks"]
else:
    __all__ = ["wandb_models_hooks"]

__version__ = "0.2.3"