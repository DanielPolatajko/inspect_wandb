
# `inspect_wandb`
Integration with [Inspect](https://inspect.aisi.org.uk/) and Weights & Biases. Initially, this integration was focused primarily on [Weave](https://weave-docs.wandb.ai/), but we are also expanding to include the [wandb Models API](https://docs.wandb.ai/guides/models/)

## Demo Video

<div>
    <a href="https://www.loom.com/share/1578ad78581146d08348cfe2a13270b0">
      <p>WIP: Integrating Inspect WandB with Inspect AI for LLM Evaluations 🚀 - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/1578ad78581146d08348cfe2a13270b0">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/1578ad78581146d08348cfe2a13270b0-d6183465b48a6d2b-full-play.gif">
    </a>
  </div>

The integration is implemented as an Inspect [hook](https://inspect.aisi.org.uk/extensions.html#hooks).

## Usage

### Installation

To use this integration, you should install the package in the Python environment where you are running Inspect - Inspect will automatically detect the hooks and utilise them during eval runs. The `inspect_wandb` integration has 3 components:

- __Models__: This integrates Inspect with the W&B Models API to store eval run statistics and configuration files for reproducability
- __Weave__: This integrates Inspcet with the W&B Weave API which can be used to track and analyse eval scores, transcripts and metadata
- __Viz__: An experimental integration with [inspect_viz](https://github.com/meridianlabs-ai/inspect_viz) which allows you to generate visualisations using inspect viz and save them as images to the Models API run

By default, this integration will only install and enable the Models component, but the Weave and Viz components are easy to add as extras. To install just Models:

__pip__
```bash
pip install git+https://github.com/DanielPolatajko/inspect_wandb.git
```

__uv__
```bash
uv pip install git+https://github.com/DanielPolatajko/inspect_wandb.git
```

To install Models and Weave

__pip__
```bash
pip install "inspect_wandb[weave] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

__uv__
```bash
uv pip install "inspect_wandb[weave] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

And to install Models, Weave and Viz

__pip__
```bash
pip install "inspect_wandb[weave,viz] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

__uv__
```bash
uv pip install "inspect_wandb[weave,viz] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

If you intend to use the Viz integration, you also need to subsequently install `chromium` with:

```bash
playwright install-deps chromium
```

### W&B setup

In order to utilise the W&B integration, you will also need to setup a W&B project, authenticate your environment with W&B, and initialise the wandb client.

To get set up with a new Weave project, follow the instructions [here](https://weave-docs.wandb.ai/), or to set up a W&B project, look [here](https://docs.wandb.ai/quickstart/) (they are basically the same, but it might be useful to follow the guide of the feature you're more interested in)

#### Console configuration

If you have an existing project, for example `test-project`, you can run 

```bash
wandb login
```

in the directory where you will run Inspect from and follow the outlined steps.

You should then set the project which you'd like to write eval results to. This can be done with:

```bash
wandb init
```

`inspect_wandb` will then by default use whichever project you set as default during the `wandb init` flow when writing to Weave.

#### Environment variables

If you are running Inspect in an automated environment where stepping through `wandb` CLI configurations is impractical, you can instead configure the integration with environment variables. To achieve an equivalent setup to the above, you can set the following env variables:

- `WANDB_API_KEY` - set this to your `wandb` API key for authentication
- `WANDB_ENTITY` - set this to the name of your `wandb` entity (i.e. team name)
- `WANDB_PROJECT` - set this to the `wandb` project which you would like to write data to

Environment variables take precedence over `wandb` settings set via the CLI, so if you want to override the settings, using env vars is a viable option.

There are also a handful of `wandb` environment variables which are not directly parsed by the inspect_wandb integration, but will influence the behaviour of `wandb` if passed at runtime. These can be found [here](https://docs.wandb.ai/guides/track/environment-variables/)

### Configuration

`inspect_wandb` works out-of-the-box after running `wandb init` - no additional configuration is required! By default, both Weave and Models integrations are enabled, using the project and entity from your wandb settings or set via env variables.

#### Optional Customization

For advanced users who want to customize the behavior, you can add a `[tool.inspect-wandb]` section to your project's `pyproject.toml` file:

```toml
[tool.inspect-wandb.weave]
enabled = true  # Enable/disable Weave integration (default: true)
sample_name_template = "{task_name}_s{sample_id}_e{epoch}"  # Customize sample names in Weave traces (default: "{task_name}-sample-{sample_id}-epoch-{epoch}")

[tool.inspect-wandb.models]
enabled = false  # Enable/disable Models integration (default: true)
files = ["pyproject.toml", "log/*"]  # Files/folders to upload with Models run, path relative to your current working directory (default: none)
```

You can also manually set the `wandb` entity and project in `pyproject.toml` e.g.

```toml
[tool.inspect-wandb.weave]
wandb_entity = "test-entity"
wandb_project = "test-project"

[tool.inspect-wandb.models]
enabled = false  # Enable/disable Models integration (default: true)
wandb_entity = "test-entity"
wandb_project = "test-project"
files = ["pyproject.toml", "log/*"]  # Files/folders to upload with Models run, path relative to your current working directory (default: none)
```

#### Autopatching

For the Weave integration, there is an experimental autopatching feature which is disabled by default. This patches some Inspect functions with Weave tracing calls, such that the Weave traces UI displays a call trace which more closely resembles the structure of an Inspect eval (e.g. one call per sample, with child calls for each solver and scorer).

This feature can be configured with the `autopatch` parameter e.g.

```toml
[tool.inspect-wandb.weave]
autopatch = true
```

or by setting the environment variable `INSPECT_WANDB_WEAVE_AUTOPATCH=true`.

#### Sample Display Name Customization

When using the Weave integration with autopatching enabled, you can customize how sample traces are named in the Weave dashboard. This helps organize and identify traces according to your preferences.

**Environment Variable (Recommended)**
```bash
export INSPECT_WANDB_WEAVE_SAMPLE_NAME_TEMPLATE="{task_name}_s{sample_id}_e{epoch}"
```

**Available Variables:**
- `{task_name}` - Name of the evaluation task
- `{sample_id}` - Numeric ID of the sample (1, 2, 3, ...)
- `{epoch}` - Epoch number during evaluation

**Examples:**
```bash
# Compact format
export INSPECT_WANDB_WEAVE_SAMPLE_NAME_TEMPLATE="{task_name}_s{sample_id}"
# Result: "my_task_s1", "my_task_s2", ...

# Descriptive format
export INSPECT_WANDB_WEAVE_SAMPLE_NAME_TEMPLATE="Task: {task_name} | Sample {sample_id}"
# Result: "Task: my_task | Sample 1", "Task: my_task | Sample 2", ...

# Epoch-focused format
export INSPECT_WANDB_WEAVE_SAMPLE_NAME_TEMPLATE="{task_name}-epoch{epoch}-{sample_id}"
# Result: "my_task-epoch1-1", "my_task-epoch1-2", ...
```

If no custom template is provided, sample traces will use the format: `"{task_name}-sample-{sample_id}-epoch-{epoch}"` (e.g., "my_task-sample-1-epoch-1").

This template can also be configured with the `sample_name_template` parameter in `pyproject.toml` file, e.g.

```toml
[tool.inspect-wandb.weave]
sample_name_template = "{task_name}_s{sample_id}"
```

### Running Inspect with the integration

Once you have performed the above steps, the integration will be enabled for future Inspect runs in your environment by default. The Inspect logger output will link to the Weave dashboard where you can track and visualise eval results.

### Disabling the integration

You can disable either integration by adding configuration to your `pyproject.toml`. For example:

```toml
[tool.inspect-wandb.weave]
enabled = false  # Disable Weave integration

[tool.inspect-wandb.models]  
enabled = true   # Keep Models integration enabled
```

This would disable the Weave integration while keeping the Models API integration enabled.

#### Environment Variables

You can also enable or disable the integrations by setting the following environment variables:

- `INSPECT_WANDB_MODELS_ENABLED`
- `INSPECT_WANDB_WEAVE_ENABLED`

If the former is set to anything truthy, the Models integration will be enabled, and if it is set to anything falsey, the integration will be disabled. If it is unset, the settings loader will fallback to the `wandb` settings and `pyproject.toml` to determine whether to enable the integration.

The latter env var controls the Weave integration in the same manner.

#### Script-Level Control (Highest Priority)

For fine-grained control, you can override any settings at the script level using task metadata. This takes **highest priority** over all other configuration methods.
With script:
```python
eval(my_eval, 
  model="mockllm/model", 
  metadata={
    "inspect_wandb_weave_enabled": True, 
    "inspect_wandb_models_enabled": False
    }
  )
```
or with command:
`inspect eval my_eval --metadata inspect_wandb_weave_enabled=True`



## Development

If you want to develop this project, you can fork and clone the repo and then run:

```bash
uv sync --group dev
source .venv/bin/activate
uv add pre-commit 
pre-commit install
```

to install for development locally.
For editing the docs and then displaying them with `html` you additionally need 

### Testing

We write unit tests with `pytest`. If you want to run the tests, you can simply run `pytest`. Please consider writing a test if adding a new feature, and make sure that tests are passing before submitting changes.

## Project notes

This project in a work-in-progress, being developed as a [MARS](https://www.cambridgeaisafety.org/mars) project by [DanielPolatajko](https://github.com/DanielPolatajko), [Qi Guo](https://github.com/Esther-Guo), [Matan Shtepel](https://github.com/GnarlyMshtep), and supervised by Justin Olive. We are open to feature requests and suggestions for future directions (including extensions of this integration as well as other possible Inspect integrations).

