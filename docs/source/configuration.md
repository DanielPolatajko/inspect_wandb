# Configuration

Inspect WandB works out-of-the-box after running `wandb init` - no additional configuration is required! There are also programmatic configuration options that can be used for automated environments. For basic setup, see {doc}`installation`.

For advanced users who want to customize the behavior, you can configure Inspect WandB using environment variables or `pyproject.toml`. This page provides detailed configuration options.

## Configuration Priority

The priority for settings is:
1. `eval` configs 
2. Environment variables (highest priority)
3. WandB settings file (for entity/project)
4. Initial settings (programmatic overrides)
5. `pyproject.toml` (lowest priority)


## WandB using env variables
The simplest way to configure WandB (Models and Weave) is with `wandb init`. For use cases where an interactive terminal session is not an option or for finer config granularity, the following environment variables can be set. 
> Note: `INSPECT_WANDB_MODELS_PROJECT` and `WANDB_INSPECT_WANDB_WEAVE_PROJECT` can be set to different values, pushing Models and Weave data to different projects. Likewise for `INSPECT_WANDB_MODELS_ENTITY` and `WANDB_INSPECT_WANDB_WEAVE_ENTITY`.

### WandB Models Configuration

`INSPECT_WANDB_MODELS_#` where `#` can be set to:

1. **ENABLED**: Controls whether the WandB Models integration is active. Defaults to `True`.
2. **PROJECT**: Specifies the WandB project for the WandB Models integration. Can also be set using the `WANDB_PROJECT` environment variable.
3. **ENTITY**: Defines the WandB entity (team or username) for the WandB Models integration. Can also be set using the `WANDB_ENTITY` environment variable.
4. **CONFIG**: Optional dictionary containing configuration parameters that will be passed directly to `wandb.config` for the WandB Models integration. Example: 
   ```bash
   INSPECT_WANDB_MODELS_CONFIG='{"learning_rate":  1e-5}'
   ```
   See more details in https://docs.wandb.ai/guides/track/config/.
5. **FILES**: Optional list of files to upload during the models run. File paths should be specified relative to the current working directory. Example: 
   ```bash
   INSPECT_WANDB_MODELS_FILES='["README.md", "Makefile"]'
   ```
6. **VIZ**: Controls whether to enable the inspect_viz extra functionality. Defaults to `False`. We recommend against this as the feature is still experimental (note it also requires additional installation). 
7. **TAGS**: Tags to add to the models run, e.g. `INSPECT_WANDB_MODELS_TAGS="['tag1','tag2']"`


### WandB Weave Configuration

`INSPECT_WANDB_WEAVE_#` where `#` can be set to:

1. **ENABLED**: Controls whether the WandB Weave integration is active. Defaults to `True`.
2. **PROJECT**: Specifies the WandB project for the WandB Weave integration. Can also be set using the `WANDB_PROJECT` environment variable.
3. **ENTITY**: Defines the WandB entity (team or username) for the WandB Weave integration. Can also be set using the `WANDB_ENTITY` environment variable.
4. **AUTOPATCH**: Controls whether to automatically patch various Inspect calls through Inspect with WandB Weave for tracing. Defaults to `True`. We expect most users to want this as it allows for the nice traces seen in {doc}`example`. 

## Sample display name configuration

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


## Alternative configuration methods: `pyproject.toml`
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

## Alternate configuration methods: `eval`
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

 
