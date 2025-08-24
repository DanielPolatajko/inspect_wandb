# Configuration 
Inspect WandB has the following configuration info: 
`INSPECT_WANDB_MODELS_#` where `#` can be set to 
1. **ENABLED**: Controls whether the Models integration is active. Defaults to `True`.
2. **PROJECT**: Specifies the Weights & Biases project for the Models integration. Can also be set using the `WANDB_PROJECT` environment variable.
3. **ENTITY**: Defines the Weights & Biases entity (team or username) for the Models integration. Can also be set using the `WANDB_ENTITY` environment variable. 
4. **CONFIG**: Optional dictionary containing configuration parameters that will be passed directly to `wandb.config` for the Models integration. `INSPECT_WANDB_MODELS_CONFIG='{"learning_rate":  1e-5}'` See more details in https://docs.wandb.ai/guides/track/config/.
5. **FILES**: Optional list of files to upload during the models run. File paths should be specified relative to the wandb directory. e.g. `INSPECT_WANDB_MODELS_FILES='["README.md", "Makefile"]'`
6. **VIZ**: Controls whether to enable the inspect_viz extra functionality. Defaults to `False`. We recommend this as the feauture is still experimental (note it also requires additional installation). 

Also `INSPECT_WANDB_WEAVE_#` where `#` can be set to 

1. **ENABLED**: Controls whether the Weave integration is active. Defaults to `True`.
2. **PROJECT**: Specifies the Weights & Biases project for the Weave integration. Can also be set using the `WANDB_PROJECT` environment variable.
3. **ENTITY**: Defines the Weights & Biases entity (team or username) for the Weave integration. Can also be set using the `WANDB_ENTITY` environment variable.
4. **AUTOPATCH**: Controls whether to automatically patch Inspect with Weave calls for tracing. Defaults to `False`.

The priority for setting settings is 
1. Environment variables (highest priority)
2. Wandb settings file (for entity/project)
3. Initial settings (programmatic overrides)
4. Pyproject.toml (lowest priority)