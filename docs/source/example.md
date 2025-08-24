# Tutorial
In this page we'll teach the basics of using Inspect WandB through an informative example.
For standalone installation instructions, please see {doc}`installation`.
The example is divided as follows 
1. Quick install
2. Obtaining and running an example eval with Inspect WandB 
3. WandB UI basics
4. Where is my data in the WandB UI and how can I process it?

## Example setup
Inspect Wandb is compatible with any Inspect eval and you can follow along with this tutorial on eval of your choosing. 
If you don't have such an eval, feel free to clone `inspect_evals` a collection of evals for Inspect AI:
We recommend using `uv` for this tutorial which can be installed fron https://docs.astral.sh/uv/#projects

```bash 
git clone https://github.com/UKGovernmentBEIS/inspect_evals.git 
#uv synv installs inspect_ai and other dependencies required for inspect_evals
uv sync   
# set your API key for whatever model you want to run
export ANTHROPIC_API_KEY=...
```
> note we found that sometimes the environement breaks and `ModuleNotFoundError: No module named 'inspect_evals'` apears. It seems that ˝

Next, to install the latest version of the extension with support for WandB Models (by default) and WandB Weave (`[weave]` option), please

```bash
uv pip install "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave]"```

and tell WandB which account and project to log to, please

```bash
wandb login
wandb init
```

or if running an interactive shell session is not feesible, configure env variables as specified in {doc}`configuration`. 
We're ready to run! Let's try running an eval  
```bash
uv run inspect eval inspect_evals/gpqa_diamond --model anthropic/claude-3-5-haiku-latest --limit 10
```
which will run the first 10 questions of GPQA_DIAMOND eval on `claude-3-5-haiku-latest` (for `4` epochs, which is the default for this eval). 
Once the eval completes, you should see a line like this  
```bash 
wandb:  View project at: https://wandb.ai/danielpolatajko-mars/docs-tutorial
``` 
and your view should look something like 
![inital view](img/initial.png)

## What's to expect in every tab? 
I ran a couple more evals using `uv run inspect eval inspect_evals/...` so we'll have a bit more to look at (note some require docker / specific LLM API keys to standardise judge) and now we can review the important tabs of the UI.

###  Workspace

