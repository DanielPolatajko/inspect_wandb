# Configuration

Inspect WandB works out-of-the-box after running `wandb init` from the command line where you will run Inspect - no additional configuration is required! There are also programmatic configuration options that can be used for automated environments. For basic setup, see {doc}`installation`.

For advanced users who want to customize the behavior, you can configure Inspect WandB using environment variables or `pyproject.toml`. This page provides detailed configuration options.

## Available settings

The following is a list of all of the configurable settings you can tweak in Inspect WandB.

### WandB Models Configuration

`INSPECT_WANDB_MODELS_#` where `#` can be set to:

| Setting name | Purpose | Default |
|--------------|---------|---------|
| **ENABLED** | Controls whether the WandB Models integration is active. | `True` |
| **PROJECT** | Specifies the WandB project for the WandB Models integration. Can also be set using the `WANDB_PROJECT` environment variable. | __Required__ |
| **ENTITY** | Defines the WandB entity (team or username) for the WandB Models integration. Can also be set using the `WANDB_ENTITY` environment variable. | __Required__ |
| **CONFIG** | Optional dictionary containing configuration parameters that will be passed directly to `wandb.config` for the WandB Models integration. Example: `INSPECT_WANDB_MODELS_CONFIG='{"learning_rate": 1e-5}'`. See more details in https://docs.wandb.ai/guides/track/config/. | `None` |
| **FILES** | Optional list of files/folders to upload during the models run. File paths should be specified relative to the current working directory (in which Inspect is being run). Example: `INSPECT_WANDB_MODELS_FILES='["README.md", "Makefile"]'` | `None` |
| **TAGS** | Optional tags to add to the models run, e.g. `INSPECT_WANDB_MODELS_TAGS="['tag1','tag2']"` | `None` |
| **ADD_METADATA_TO_CONFIG** | Whether or not to write the entire Inspect metadata to WandB run config. | `True` |
| **ENVIRONMENT_VALIDATIONS** | An optional dict with the format `{"wandb_base_url": "<your base url>", "wandb_api_key": "<your api key>"}`. This provides a secondary validation layer for environment variables which influence WandB's behaviour, but are not directly accessed by the Inspect WandB extension. | `None` |


### WandB Weave Configuration

`INSPECT_WANDB_WEAVE_#` where `#` can be set to:

| Setting name | Purpose | Default |
|--------------|---------|---------|
| **ENABLED** | Controls whether the WandB Weave integration is active. | `True` |
| **PROJECT** | Specifies the WandB project for the WandB Weave integration. Can also be set using the `WANDB_PROJECT` environment variable. | __Required__ |
| **ENTITY** | Defines the WandB entity (team or username) for the WandB Weave integration. Can also be set using the `WANDB_ENTITY` environment variable. | __Required__ |
| **SAMPLE_NAME_TEMPLATE** | Sets a template which is used to name sample traces in the Weave UI. The three variables `task_name, sample_id, epoch` will be filled from the Inspect context, allowing you to change the static text that appears around them. | `{task_name}-sample-{sample_id}-epoch-{epoch}` |

## Configuration Priority

The priority for settings is:
1. `eval` or `eval-set` metadata configurations (highest priority)
2. Environment variables (for automated environments)
3. WandB settings file (for entity/project only)
4. `pyproject.toml` (repo-level settings, lowest priority)


### Configuring using env variables
The simplest way to configure WandB (Models and Weave) is with `wandb init` in the terminal, which must be run in the same directory you'll run Inspect in. For use cases where an interactive terminal session is not an option, or for finer config granularity, the following environment variables can be set instead. 

- `WANDB_PROJECT`: Sets the WandB project for both Models and Weave
- `WANDB_ENTITY`: Sets the Wandb entity/team for both Models and Weave

You can also configure any of the settings listed above by setting environment variables using the syntax `INSPECT_WANDB_MODELS_#` or `INSPECT_WANDB_WEAVE_#`, replacing `#` appropriately.

> Note: If you want to write data for a single eval to different WandB projects based on whether Inspect is writing to the Models API or Weave, you can set  `INSPECT_WANDB_MODELS_PROJECT` and `INSPECT_WANDB_WEAVE_PROJECT` to different values, pushing Models and Weave data to different projects. Likewise for `INSPECT_WANDB_MODELS_ENTITY` and `INSPECT_WANDB_WEAVE_ENTITY`.

### Configuring using `eval` or `eval-set` metadata
For fine-grained control, you can override any settings at the script level using `eval` or `eval-set` metadata. The syntax for overriding this way is the same as environment variables (neither are case-sensitive). This takes **highest priority** over all other configuration methods.
With script:
```python
eval(
  my_eval, 
  model="mockllm/model", 
  metadata={
    "inspect_wandb_weave_enabled": True, 
    "inspect_wandb_models_enabled": False
  }
)
```
or with command:
`inspect eval my_eval --metadata inspect_wandb_weave_enabled=True`

### Configuring using `pyproject.toml`
It is possible to configure using a `pyproject.toml` as follows.

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

As a general rule, settings in `pyproject.toml` are the same as the environment variable settings above, but rather than having prefixes like `INSPECT_WANDB_MODELS_`, they are adding to the relevant block.

This configuration level is **lowest priority** i.e. any environment variables or metadata settings will override settings in `pyproject.toml`

## Full Configuration Options

The following table shows all available settings and how to configure them using different methods:

| Setting | Purpose | `pyproject.toml` | Environment Variables | `wandb init` | Inspect metadata |
|---------|---------|------------------|----------------------|--------------|------------------|
| **Models: Enabled** | Controls whether the WandB Models integration is active | `[tool.inspect-wandb.models]`<br>`enabled = true` | `INSPECT_WANDB_MODELS_ENABLED=True` | N/A | `--metadata inspect_wandb_models_enabled=True` |
| **Models: Project** | Specifies the WandB project for Models integration | `[tool.inspect-wandb.models]`<br>`wandb_project = "my-project"` | `INSPECT_WANDB_MODELS_PROJECT="my-project"`<br>or `WANDB_PROJECT="my-project"` | Run in directory from which you'll run Inspect | `--metadata inspect_wandb_models_project="my-project"` |
| **Models: Entity** | Defines the WandB entity (team or username) for Models integration | `[tool.inspect-wandb.models]`<br>`wandb_entity = "my-entity"` | `INSPECT_WANDB_MODELS_ENTITY="my-entity"`<br>or `WANDB_ENTITY="my-entity"` | Run in directory from which you'll run Inspect | `--metadata inspect_wandb_models_entity="my-entity"` |
| **Models: Config** | Dictionary of configuration parameters passed to `wandb.config` | `[tool.inspect-wandb.models]`<br>`config = {learning_rate = 1e-5}` | `INSPECT_WANDB_MODELS_CONFIG='{"learning_rate": 1e-5}'` | N/A | `--metadata inspect_wandb_models_config={"learning_rate": 1e-5}` |
| **Models: Files** | List of files/folders to upload during the models run | `[tool.inspect-wandb.models]`<br>`files = ["README.md", "logs/*"]` | `INSPECT_WANDB_MODELS_FILES='["README.md", "logs/*"]'` | N/A | `--metadata inspect_wandb_models_files=["README.md", "logs/*"]` |
| **Models: Tags** | Tags to add to the models run | `[tool.inspect-wandb.models]`<br>`tags = ["tag1", "tag2"]` | `INSPECT_WANDB_MODELS_TAGS='["tag1", "tag2"]'` | N/A | `--metadata inspect_wandb_models_tags=["tag1", "tag2"]` |
| **Models: Add Metadata to Config** | Whether to write the entire Inspect metadata to WandB run config | `[tool.inspect-wandb.models]`<br>`add_metadata_to_config = true` | `INSPECT_WANDB_MODELS_ADD_METADATA_TO_CONFIG=True` | N/A | `--metadata inspect_wandb_models_add_metadata_to_config=True` |
| **Models: Environment Validations** | Dict for secondary validation of environment variables | `[tool.inspect-wandb.models]`<br>`environment_validations = {wandb_base_url = "...", wandb_api_key = "..."}` | `INSPECT_WANDB_MODELS_ENVIRONMENT_VALIDATIONS='{"wandb_base_url": "...", "wandb_api_key": "..."}'` | N/A | `--metadata inspect_wandb_models_environment_validations={"wandb_base_url": "...", "wandb_api_key": "..."}` |
| **Weave: Enabled** | Controls whether the WandB Weave integration is active | `[tool.inspect-wandb.weave]`<br>`enabled = true` | `INSPECT_WANDB_WEAVE_ENABLED=True` | N/A | `--metadata inspect_wandb_weave_enabled=True` |
| **Weave: Project** | Specifies the WandB project for Weave integration | `[tool.inspect-wandb.weave]`<br>`wandb_project = "my-project"` | `INSPECT_WANDB_WEAVE_PROJECT="my-project"`<br>or `WANDB_PROJECT="my-project"` | Run in directory from which you'll run Inspect | `--metadata inspect_wandb_weave_project="my-project"` |
| **Weave: Entity** | Defines the WandB entity (team or username) for Weave integration | `[tool.inspect-wandb.weave]`<br>`wandb_entity = "my-entity"` | `INSPECT_WANDB_WEAVE_ENTITY="my-entity"`<br>or `WANDB_ENTITY="my-entity"` | Run in directory from which you'll run Inspect | `--metadata inspect_wandb_weave_entity="my-entity"` |
| **Weave: Sample Name Template** | Template for naming sample traces in the Weave UI | `[tool.inspect-wandb.weave]`<br>`sample_name_template = "{task_name}_s{sample_id}"` | `INSPECT_WANDB_WEAVE_SAMPLE_NAME_TEMPLATE="{task_name}_s{sample_id}"` | N/A | `--metadata inspect_wandb_weave_sample_name_template="{task_name}_s{sample_id}"` |


