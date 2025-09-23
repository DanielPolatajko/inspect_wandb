# Tutorial
In this page we'll teach the basics of using Inspect WandB through an informative example.
Through this tutorial you will learn how to install Inspect WandB, run an example eval, and navigate the information logged to WandB Models and WandB Weave.

This tutorial is divided as follows:
1. [**Example setup**](example_setup) - Setting up the environment and installing dependencies
2. [**Understanding WandB UI tabs**](understanding-wandb-ui-tabs) - Overview of the different interface sections
3. [**WandB Models integration**](wandb-models-workspace) - Using WandB Models for run tracking and reproducibility
4. [**WandB Weave integration**](wandb-weave-evals) - Using WandB Weave for eval analysis and comparison
5. [**Comparing evaluations**](comparing-evals) - How to compare results across different models
6. [**Accessing reproducibility information**](inspect-wandb-weave-obtaining-reproducibility-info-from-an-eval-of-interest) - Retrieving data for reproducing results

(example_setup)=
## Example setup
Inspect WandB is compatible with any Inspect eval and you can follow along with this tutorial with an eval of your choice.
If you don't have such an eval, feel free to clone [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals), a collection of evals for Inspect AI.
We recommend using `uv` for this tutorial which can be installed from [here](https://docs.astral.sh/uv/#installation).
For standalone installation instructions, please see {doc}`installation`.


```bash
git clone https://github.com/UKGovernmentBEIS/inspect_evals.git
# uv sync installs inspect_ai and other dependencies required for inspect_evals
uv sync
# set your API key for whatever model you want to run
export ANTHROPIC_API_KEY=...
```

Next, to install the latest version of the extension with support for WandB Models (by default) and WandB Weave (`[weave]` option), run:

```bash
uv pip install inspect-wandb[weave]
```

and to tell WandB which account and project to log to, run:

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
Once the eval completes, you should see the following line in the console logs:
```bash
wandb:  View project at: https://wandb.ai/YOUR-TEAM-NAME/YOUR-PROJECT-NAME
```
Going to that link, your view should look something like:
![initial view](_static/initial.png)

(understanding-wandb-ui-tabs)=
## Understanding WandB UI tabs
I ran a couple more evals using `uv run inspect eval inspect_evals/...` so we'll have a bit more to look at; now we can review the important tabs of the UI.

(wandb-models-workspace)=
### WandB Models: Workspace
The primary function of the WandB Models integration is to auto-log information about a run so it can be reproduced and further investigated if needed.
The rule of thumb is that one `inspect eval ...` or `inspect eval-set ...` produces one Run in WandB Models. So even if you execute multiple models or tasks, with one `inspect eval ...` command, all the data will belong to a single WandB Models Run. 

Your workspace tab might look something like:
![](_static/workspace.png)

On the left we can see all the runs that we have executed and on the right we see:
1. **Charts**: at present only logs the number of samples so far (usually `y=x` line) and the current `accuracy` metric if the eval has a scorer called `accuracy`. We plan to make this more useful in the future.
2. **System**: auto-logged wandb metrics -- probably not very interesting if you're running API models but perhaps useful if you are self-serving. 

### WandB Models: specific run

Clicking on a run on the left, we can see the Overview:
![](_static/run-models-overview.png)
which contains information about the system, git state, and the `inspect eval ...` command used to trigger the evaluation.  

Clicking on Files tab, we see:
![](_static/run-models-files.png)
which contains files auto-logged by WandB Models such as `requirements.txt` which contains versioning info.
You can choose to have additional files auto-uploaded by setting:

```bash
INSPECT_WANDB_MODELS_FILES='["README.md", "Makefile"]'
```
The files and state information can be useful for reproducing and further investigating the run. 

(wandb-weave-evals)=
### WandB Weave: Evals
The Evals tab under WandB Weave might look something like:
![](_static/weave-evals.png)
This tab contains evals which previously ran, alongs with attributes which primarily consist of an aggregation across samples of any Inspect Scorer in the eval, and some additional metadata.
The rule of thumb is that one model on one Inspect task corresponds to one eval in the Evals dashboard.

The first field is status which shows if the eval is in progress, succeeded, or failed. This can be useful for long-running evals, allowing you to connect to WandB on mobile to check status.  

This view can get overwhelming as the number of metrics grows large, and not every metric is applicable to every eval.
Clicking on "Filter" at the top left, it's possible to filter by certain attributes, and once done, by clicking on "Save View" at the top right, save the view.
Saved views can be edited and returned to at a later time. 

The current view shows only `agentharm` runs:
![](_static/filtered-view.png)

### WandB Weave: exploring a particular eval
Clicking on an eval and then clicking on trace tree (the stack of cards at the top right) you will see all the traced function calls made during the eval run.

The Trace tree will contain traces of every solver step, model call, and scoring operation that contributed to each sample's final result.
![](_static/sample-traces.png)

The model call traces also allow you to view transcripts for that model interaction.
![](_static/trace.png)
Clicking on "Playground" at the top right opens an interactive chat view where the chat history is editable and it's possible to query various models and perform quick experiments.

Individual traces can also be explored under the "Traces" tab.

(comparing-evals)=
### Comparing evals
To run multiple evals on the same dataset you can:
```bash
uv run inspect eval inspect_evals/agentharm --model openai/gpt-4o,anthropic/claude-3.7-sonnet-latest
```
Marking two evals on the left and clicking compare:
![](_static/compare-enter.png)
we see:
![](_static/compare.png)
which shows various comparison metrics between gpt-4o and claude-3.7-sonnet on `agentharm`.
It is also possible to compare more than two models on the same and different evals.

(inspect-wandb-weave-obtaining-reproducibility-info-from-an-eval-of-interest)=
### Referencing an eval of interest
Having filtered and found an eval of interest in WandB Weave UI, click on the eval from the list > `Summary` > Scroll down to `Attributes` and click on to `inspect` > `run_id` (if not using `eval-set`) or `inspect` > `eval_set_id`. These IDs are used to track Models runs, so you can cross reference between Weave and Models interfaces.