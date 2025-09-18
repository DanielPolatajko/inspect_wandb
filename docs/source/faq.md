# Frequently asked questions

Here are a few common issues that people have reported using Inspect WandB

:::{note} `ModuleNotFoundError: inspect_evals` in `inspect_evals`
:class: dropdown
We found that sometimes the environment breaks and `ModuleNotFoundError: No module named 'inspect_evals'` appears. It seems that `uv sync --reinstall` fixes the issue.
:::

:::{note} Dataset comparison in Weave is not working
:class: dropdown
We've noticed "dataset comparison" in the top left of the Evaluation view in Weave is not working as expected. This bug is being tracked [here](https://github.com/DanielPolatajko/inspect_wandb/issues/122)

:::{note} Models Run Log view is illegible
:class: dropdown
Currently, `output.log` and the `Logs` tab contain illegible rich text instead of the executing command's stdout. This is a known bug being tracked [here](https://github.com/DanielPolatajko/inspect_wandb/issues/60) 
:::