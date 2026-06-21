from inspect_ai.hooks import hooks
from inspect_wandb.config.extras_manager import INSTALLED_EXTRAS
from inspect_wandb.models import WandBModelHooks

if INSTALLED_EXTRAS["weave"]:
    from inspect_wandb.weave import WeaveEvaluationHooks

    @hooks(name="weave_evaluation_hooks", description="Weave evaluation integration")
    def weave_evaluation_hooks():
        return WeaveEvaluationHooks


@hooks(name="wandb_models_hooks", description="Weights & Biases model integration")
def wandb_models_hooks():
    return WandBModelHooks
