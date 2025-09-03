# Tutorial
In this page we'll teach the basics of using Inspect WandB through an informative example.
Through this tutorial you will learn how to install Inspect WandB, run an example eval, and navigate the information logged to WandB Models and WandB Weave.

This tutorial is divided as follows:
1. [**Example setup**](#example-setup) - Setting up the environment and installing dependencies
2. [**Understanding WandB UI tabs**](#understanding-wandb-ui-tabs) - Overview of the different interface sections
3. [**WandB Models integration**](#wandb-models-workspace) - Using WandB Models for run tracking and reproducibility
4. [**WandB Weave integration**](#wandb-weave-evals) - Using WandB Weave for eval analysis and comparison
5. [**Comparing evaluations**](#comparing-evals) - How to compare results across different models
6. [**Accessing reproducibility information**](#inspect-wandb-weave-obtaining-reproducibility-info-from-an-eval-of-interest) - Retrieving data for reproducing results

## Example setup
Inspect WandB is compatible with any Inspect eval and you can follow along with this tutorial on eval of your choosing.
If you don't have such an eval, feel free to clone `inspect_evals`, a collection of evals for Inspect AI.
We recommend using `uv` for this tutorial which can be installed from https://docs.astral.sh/uv/#projects.
For standalone installation instructions, please see {doc}`installation`.


```bash
git clone https://github.com/UKGovernmentBEIS/inspect_evals.git
# uv sync installs inspect_ai and other dependencies required for inspect_evals
uv sync
# set your API key for whatever model you want to run
export ANTHROPIC_API_KEY=...
```
> **Note:** We found that sometimes the environment breaks and `ModuleNotFoundError: No module named 'inspect_evals'` appears. It seems that `uv sync --reinstall` fixes the issue.

Next, to install the latest version of the extension with support for WandB Models (by default) and WandB Weave (`[weave]` option), please:

```bash
uv pip install "inspect_wandb @ git+https://github.com/DanielPolatajko/inspect_wandb.git#[weave]"
```

and to tell WandB which account and project to log to, please:

```bash
wandb login
wandb init
```

or if running an interactive shell session is not feasible, configure env variables as specified in {doc}`configuration`.
We're ready to run! Let's try running an eval:
```bash
uv run inspect eval inspect_evals/gpqa_diamond --model anthropic/claude-3-5-haiku-latest --limit 10
```
which will run `claude-3-5-haiku-latest` on the first 10 questions of `GPQA_DIAMOND` (for 4 epochs, which is the default for this eval).
Once the eval completes, you should see in your console output a line like:
```bash
wandb:  View project at: https://wandb.ai/your-team-name/your-proj-name
```
Going to that link, your view should look something like:
![initial view](img/initial.png)

## Understanding WandB UI tabs
I ran a couple more evals using `uv run inspect eval inspect_evals/...` so we'll have a bit more to look at (note some require docker / specific LLM API keys to standardize judge) and now we can review the important tabs of the UI.

### WandB Models: Workspace
The current primary aim of the WandB Models integration is to auto-log information about a run so it can be reproduced and further investigated if needed.
The rule of thumb is that one `inspect eval ...` or `inspect eval-set ...` = 1 run in WandB Models. So even if you execute multiple models or dataset, with one `inspect eval ...` command, all the data will belong to a single WandB Models run. 

Your workspace tab might look something like:
![](img/workspace.png)

On the left we can see all the runs that we have executed and on the right we see:
1. **Charts**: at present only logs the number of samples so far (usually `y=x` line) and the current `accuracy` metric if the eval has a scorer called `accuracy`. We hope to make this more useful in the future.
2. **System**: auto-logged wandb metrics -- probably not very interesting if you're running API models but perhaps useful if you are self-serving. 

### WandB Models: specific run

Clicking on a run on the left, we can see the run overview:
![](img/run-models-overview.png)
which contains information about the system, git state, and the `inspect eval ...` command used to trigger the evaluation. 
Clicking on files, we see:
![](img/run-models-files.png)
which contains files auto-logged by WandB Models such as `requirements.txt` which contains versioning info.
You can choose to have additional files auto-uploaded by by setting:

```bash
INSPECT_WANDB_MODELS_FILES='["README.md", "Makefile"]'
```
The files and state information can be useful for reproducing and further investigating the run. 

Currently, `output.log` and the `Logs` tab contain illegible rich text instead of the executing command's stdout, but this will be resolved in https://github.com/DanielPolatajko/inspect_wandb/issues/60. 

### WandB Weave: Evals
The Evals tab under WandB Weave might look something like:
![](img/weave-evals.png)
This tab contains evals which previously ran, alongs with attributes which primarily consist of an aggregation across samples of any Inspect Scorer in the eval  logged + additional metadata.
The rule of thumb is that 1 model + Inspect task = 1 eval in the Evals dashboard.

The first field is status which shows if the eval is in progress, succeeded, or failed. This is particularly nice on long-running evals as one can connect to WandB on mobile to check status.  

This view can get overwhelming as the number of metrics grows large, and not every metric is applicable to every eval.
Clicking on "Filter" at the top left, it's possible to filter by certain attributes, and once done, by clicking on "Save View" in the top left, save the view.
Saved viewed can be edited and returned to at a later time. 

The current view shows only `agentharm` runs:
![](img/filtered-view.png)

### WandB Weave: exploring a particular eval
Clicking on an eval and then clicking on trace tree (the stack of cards at the top right) you will see all the traced function calls made during the eval run.
![](img/trace.png)
By default, this will primarily contain LLM API calls from various providers, but may contain some other info as well. Individual traces can also be explored under the "Traces" tab.

Clicking on "Playground" at the top right takes one to an interactive chat view where the chat history is editable and it's possible to query various models and perform quick experiments.


### Comparing evals
To run multiple evals on the same dataset you can:
```bash
uv run inspect eval inspect_evals/agentharm --model openai/gpt-4o,anthropic/claude-3.7-sonnet-latest
```
Marking two evals on the left and clicking compare:
![](img/compare-enter.png)
we see:
![](img/compare.png)
which shows various comparison metrics between gpt-4o and claude-3.7-sonnet on `agentharm`.
It is also possible to compare multiple models on the same and different evals.


### Inspect WandB Weave: obtaining reproducibility info from an eval of interest
Once having filtered and found an eval of interest in WandB Weave UI, click on the eval from the list > `Summary` > Scroll down and click on to `Inspect` > `run_id`. This is the same `run_id` that is used to index WandB Models runs, from which we have already shown how to retrieve reproducibility information.  