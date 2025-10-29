# Welcome to Inspect WandB's documentation!

```{toctree}
:titlesonly:
installation.md
tutorial.md
concepts.md
configuration.md
contributing.md
faq.md
```

```{toctree}
:hidden:
:caption: Links

GitHub <https://github.com/DanielPolatajko/inspect_wandb>
PyPI <https://pypi.org/project/inspect-wandb/>
```

**Inspect WandB** is a Python library for integrating the [Inspect AI framework](https://inspect.aisi.org.uk/) with Weights & Biases (WandB) [Models](https://wandb.ai/site/models/) API and [Weave](https://wandb.ai/site/weave/).
Inspect is a framework for developing and executing LLM evaluations developed by UK AI Security Institute.
WandB Models and WandB Weave are tools for logging, managing, and visualizing AI model runs, where WandB Models is focused on experiment tracking and training runs while WandB Weave is specifically for LLM evaluations.

## Quickstart

For detailed installation instructions, see {doc}`installation`.
```bash
pip install inspect-wandb
```

The Weave features are not installed by default and are available as an optional extra; to install:
```bash
pip install inspect-wandb[weave]
```

Next, ensure WandB is authenticated by setting the `WANDB_API_KEY` environment variable, or by running:
```bash
wandb login
```

You will also have to configure the WandB project and entity to which you want to log Inspect runs. There are 4 ways to do so depending on your use case - see {doc}`installation` for more details. The simplest way to get started is to run the following command and follow the interactive instructions in the same directory from which you will run Inspect:

```bash
wandb init
```

You can then run any Inspect eval with:
```bash
inspect eval YOUR_EVAL     
```

In the terminal you should see:
```bash
wandb: Syncing run UID
wandb: ⭐️ View project at https://wandb.ai/YOUR_TEAM_NAME/YOUR_PROJECT_NAME
wandb: 🚀 View run at https://wandb.ai/YOUR_TEAM_NAME/YOUR_PROJECT_NAME/runs/UID
```

Clicking the second link will take you to the WandB Models UI tab for the eval.

Please see {doc}`tutorial` for more details on how to navigate and use the WandB Models and Weave UIs!

(use_cases)=
## Use cases
Some common use cases for Inspect WandB include:
* **Filtering across Inspect eval runs:** A common pain point with Inspect is the lack of a visualization/UI-friendly way to search and process data across eval runs. WandB Weave's has built-in support for filtering and searching eval results across a range of axes.
* **Comparison across Inspect eval runs:** In addition to filtering, WandB Weave offers UI-interactive ways to compare results across eval runs and across different models on the same eval.
* **Structured tracing:** Inspect WandB builds a trace tree in Weave, which lays out the task, solver, scorers, and traces model invocations and tool calls, making it easy to navigate through the lifecycle of an evaluation. One common use case for this feature is finding and reading interesting transcripts.
* **Shareability & Persistence:** While evals are often developed and assessed collaboratively, by default, Inspect stores all logs locally, making it difficult for teams to share and collaborate and easy for data to be lost. WandB Models and WandB Weave natively store all the data in the cloud in a way that is easy for the entire team to access.
- **Repeatability:** WandB Models API allows you to capture the exact configuration that was used to run an eval, making it easy for teams to repeat experiments without sending complex environmental setups. Direct links between WandB Runs and Inspect log locations make it simple to keep track of evals across platforms.

Check out our [tutorial video](link) for a more in-depth walkthrough of Inspect WandB and some of the ways it can be used to augment Inspect workflows.

## Troubleshooting

If you have any issues using the extension, the first place to look for guidance is the {doc}`faq` page. 

If you don't find an answer there, you can reach out to the maintainers in the [#inspect-wandb channel in the Inspect Community Slack](https://inspectcommunity.slack.com/archives/C09B5B00459).


## Credits
Inspect WandB was originally developed by Daniel Polatajko, Qi Guo, and Matan Shtepel with Justin Olive's mentorship as part of the Mentorship for Alignment Research Students (MARS) 3.0 programme at the Cambridge AI Safety Hub ([caish.org/mars)](http://caish.org/mars)).
We are grateful for open-source contributions from the community, and for invaluable feedback from our users, particularly Alex Remedios (UK AISI) and Sami Jawhar (METR) whose early engagement helped shape this package. 