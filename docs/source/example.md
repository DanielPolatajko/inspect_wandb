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
> note we found that sometimes the environement breaks and `ModuleNotFoundError: No module named 'inspect_evals'` apears. It seems that `uv sync --reinstall` fixes the issue

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

###  Models: Workspace
The current primary purpose of the WandB Models integration is to auto-log information about a run so it can be reproduced and further investigated if needed. 
The rule of thumb is that one `inspect eval ...` = 1 run in Inspect Models. 

Your workspace tab might look something like 
![](img/workspace.png)

On the left we can see all the runs that we have executed and on the right 2 tabs: 
1. Charts: at present only logs the number of samples so far (a diagonal line always) and current accuracy metric if runs have it. We hope to make this more useful in the future
2. System: auto-logged wandb metrics -- probably not relavent if you're running API models but perhaps useful if you are self-serving. 

### Models: specific run

Clicking on a run on the left, we can see the run overview 
![](run-models-overview.png)
which contains information about the system, git state, and command used to trigger the evaluation. 
Clicking on files, we see 
![](run-models-files.png)
which contains files which can be helpful for reproduction, such as `requirements.txt` which contains versioning info. 
You can select additional files to upload by setting  

```bash
INSPECT_WANDB_MODELS_FILES='["README.md", "Makefile"]'
```

Currently `output.log` and the `Logs` tab contain illegible rich text instead of the executing command's stdout, but this will be resolved in https://github.com/DanielPolatajko/inspect_wandb/issues/60. 

### Weave: Evals
Weave Evals might look something like this 
![](img/weave-evals.png)
this tab contains every eval you ran, with the every Inspect scorer logged + additional metadata.  
The rule of thumb is that one model + Inspect task = 1 weave eval.  
The first field is status which shows if the eval is in progress, succeeded, or failed. This is particularly nice on long-running evals as one can connect to WandB on mobile to check status.  

This view can get overwheling as the number of metrics grows large, and not every metric is applicable to every eval. 
Clicking on "Filter" at the top left, its possible to filter by certain attributes, and once done, by clicking on "Save View" in the top left, save the view. 
Now we have only agentrun evals 
![](img/filtered-view.png)

### Weave: exploring a particular eval
Clicking on an eval and then clicking on trace tree (the stack of cards at the top right) you will see all the API calls made during the eval. 
![](img/trace.png)
Clicking on Playground at the top right takes one to an interactive chat view where the chat history is editable and its possible to query various model and perform quick experiments.


### Comparing evals
To run multiple evals on the same dataset you can ` uv run inspect eval inspect_evals/agentharm --model openai/gpt-4o,anthropic/claude-3.7-sonnet-latest`
Marking two evals on the left and clicking compare 
![](img/compare-enter.png)
we see 
![](img/compare.png)
which shows various comparison metrics between gpt-4o and claude-3.7-sonnet on agentharm. 
**I am not getting this to work -- see https://github.com/DanielPolatajko/inspect_wandb/issues/92#issuecomment-3218202638**

### Inspect Weave: obtainign reproducability info from an eval interest
Once having filtered and found an eval of interest in weave UI, click on the eval > Summary > Scroll down and click on to Inspect > `run_id`. This is the same `run_id` that is used to index Inspect Models runs, from which we have already shown how to retrieve reproducability information .  