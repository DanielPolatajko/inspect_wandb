(installation)=
# Installation and Setup

To use this integration, you should install the package in the Python environment where you are running Inspect - Inspect will automatically detect the hooks and utilise them during eval runs. The `inspect_wandb` integration has 2 components:

- **WandB Models**: This integrates Inspect with the WandB Models API to store eval run statistics and configuration files for reproducibility.
- **WandB Weave**: This integrates Inspect with the WandB Weave API which can be used to track and analyse eval scores, transcripts and metadata.

For more information on which use-cases are facilitated by each of these components, see the {doc}`tutorial`.

## PyPI Installation

By default, this integration will only install and enable the WandB Models component, but WandB Weave is easy to add as an extra. 

To install just WandB Models:

```bash
pip install inspect-wandb
```
To install WandB Models and WandB Weave:

```bash
pip install "inspect-wandb[weave]"
```
> Note: On shells like `zsh` (default on macOS), quoting the extra
> (`"inspect-wandb[weave]"`) avoids globbing errors such as
> `zsh: no matches found: inspect-wandb[weave]`.

## Setup

Once you've installed the extension, you also need to ensure that Weights & Biases is properly configured for your project.

### Authentication

There are two ways to authenticate with WandB. For interactive use cases (e.g. developing a new eval in Inspect), you can run:

```bash
wandb login
```

in your terminal and follow the ensuing instructions.

If terminal access is not possible (e.g. deployed batch eval run for benchmarking), you can set the `WANDB_API_KEY` environment variable with an API key from Weights & Biases. More information about authenticating with Weights & Biases can be found [here](https://docs.wandb.ai/models/quickstart)

### Setting the W&B project and entity

WandB organises data according to "projects" and "entities". An entity is a user account or team, which may have multiple projects, which are collections of individual runs, evaluationsand experiments. 

In order to use the Inspect WandB integration, you'll have to set the WandB project and entity. There are 4 ways to do so, each with a slightly different use case:

| Method                | Instructions                                                                                                                                                                                                                                                                                     | Granularity          | Use Case                                                                                                                                                     |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `wandb init`          | Navigate to the directory where you will run the `inspect eval` command, run `wandb init` and follow the instructions. Note that this approach only applies the settings to the directory you run the command in.                                                                                                                                                                           | Working directory    | Simplest way to set project and entity for basic research workflows, where terminal access is possible                                                      |
| `pyproject.toml`      | Add the `[tool.inspect-wandb.models]` block to `pyproject.toml`, and under this block add `project = <your_project>` and `entity = <your_entity>`                                                                                                                                               | Project-level        | Set project and entity at the project level, which can be useful for ensuring consistency across collaborators on a single Git repo                         |
| Environment variables | Set `WANDB_PROJECT` and `WANDB_ENTITY` environment variables to your project and entity                                                                                                                                                                                                          | Environment-level    | Useful for use-cases where interactive terminal access is not possible e.g. automated batch jobs                                                             |
| Inspect metadata      | Add `--metadata inspect_wandb_models_project=<your_project>, --metadata inspect_wandb_models_entity=<your_entity_>, --metadata inspect_wandb_weave_project=<your_project>, --metadata inspect_wandb_weave_entity=<your_entity>` to the `inspect eval` or `inspect eval-set` invocation          | Inspect eval run     | Useful for changing the project or entity for a single Inspect run e.g. for testing purposes                                                                |

For more information on the interplay between these different options, please see {doc}`configuration`