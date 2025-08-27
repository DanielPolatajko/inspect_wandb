# Configuration

Inspect WandB works out-of-the-box after running `wandb init` - no additional configuration is required! For basic setup, see {doc}`installation`.

For advanced users who want to customize the behavior, you can configure Inspect WandB using environment variables or `pyproject.toml`. This page provides detailed configuration options.

## WandB Models Configuration

`INSPECT_WANDB_MODELS_#` where `#` can be set to:

1. **ENABLED**: Controls whether the WandB Models integration is active. Defaults to `True`.
2. **PROJECT**: Specifies the WandB project for the WandB Models integration. Can also be set using the `WANDB_PROJECT` environment variable.
3. **ENTITY**: Defines the WandB entity (team or username) for the WandB Models integration. Can also be set using the `WANDB_ENTITY` environment variable.
4. **CONFIG**: Optional dictionary containing configuration parameters that will be passed directly to `wandb.config` for the WandB Models integration. Example: 
   ```bash
   INSPECT_WANDB_MODELS_CONFIG='{"learning_rate":  1e-5}'
   ```
   See more details in https://docs.wandb.ai/guides/track/config/.
5. **FILES**: Optional list of files to upload during the models run. File paths should be specified relative to the wandb directory. Example: 
   ```bash
   INSPECT_WANDB_MODELS_FILES='["README.md", "Makefile"]'
   ```
6. **VIZ**: Controls whether to enable the inspect_viz extra functionality. Defaults to `False`. We recommend against this as the feature is still experimental (note it also requires additional installation). 

## WandB Weave Configuration

`INSPECT_WANDB_WEAVE_#` where `#` can be set to:

1. **ENABLED**: Controls whether the WandB Weave integration is active. Defaults to `True`.
2. **PROJECT**: Specifies the WandB project for the WandB Weave integration. Can also be set using the `WANDB_PROJECT` environment variable.
3. **ENTITY**: Defines the WandB entity (team or username) for the WandB Weave integration. Can also be set using the `WANDB_ENTITY` environment variable.
4. **AUTOPATCH**: Controls whether to automatically patch Inspect with WandB Weave calls for tracing. Defaults to `False`. 
    > **NOTE** I did not understand what this does yet

## Configuration Priority

The priority for settings is:
1. Environment variables (highest priority)
2. WandB settings file (for entity/project)
3. Initial settings (programmatic overrides)
4. `pyproject.toml` (lowest priority)

## Alternative Configuration Methods

For alternative configuration methods including `pyproject.toml` configuration and script-level control, see the main README documentation.