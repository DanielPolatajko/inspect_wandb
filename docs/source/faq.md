# Frequently asked questions

Here are a few common issues that people have reported using Inspect WandB

:::{dropdown} `ModuleNotFoundError: inspect_evals` in `inspect_evals`
:open:
We found that sometimes the environment breaks and `ModuleNotFoundError: No module named 'inspect_evals'` appears. It seems that `uv sync --reinstall` fixes the issue.
:::

:::{dropdown} Dataset comparison in Weave is not working
:open:
We've noticed "dataset comparison" in the top left of the Evaluation view in Weave is not working as expected. This bug is being tracked [here](https://github.com/DanielPolatajko/inspect_wandb/issues/122)

:::{dropdown} Models Run Log view is illegible
:open:
Currently, `output.log` and the `Logs` tab contain illegible rich text instead of the executing command's stdout. This is a known bug being tracked [here](https://github.com/DanielPolatajko/inspect_wandb/issues/60) 
:::