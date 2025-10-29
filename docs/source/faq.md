# Frequently asked questions

Here are a few common issues that people have reported using Inspect WandB

---

__`ModuleNotFoundError: inspect_evals` in `inspect_evals`__

We found that sometimes the environment breaks and `ModuleNotFoundError: No module named 'inspect_evals'` appears. It seems that `uv sync --reinstall` fixes the issue.

---

__Models Run Log view is illegible__

Currently, `output.log` and the `Logs` tab contain illegible rich text instead of the executing command's stdout. This is a known bug being tracked [here](https://github.com/DanielPolatajko/inspect_wandb/issues/60)

---

__Using `wandb.init` as a context manager does not set the wandb project and entity__

There are currently 4 ways to configure the WandB project and entity, and using:

```python
with wandb.init(project=..., entity=...):
    inspect_ai.eval(...)
```

is not one of them - the integration will not detect project and entity set in this way, and will cause an error. Check out {doc}`installation` for currently supported methods.