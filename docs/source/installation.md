(installation)=
# Installation

To use this integration, you should install the package in the Python environment where you are running Inspect - Inspect will automatically detect the hooks and utilise them during eval runs. The `inspect_wandb` integration has 3 components:

- **WandB Models**: This integrates Inspect with the WandB Models API to store eval run statistics and configuration files for reproducibility.
- **WandB Weave**: This integrates Inspect with the WandB Weave API which can be used to track and analyse eval scores, transcripts and metadata.
- **Inspect Viz**: An experimental integration with [inspect_viz](https://github.com/meridianlabs-ai/inspect_viz) which allows you to generate visualisations using inspect viz and save them as images to the WandB Models API run. We would **not** recommend trying this at the moment. 

By default, this integration will only install and enable the WandB Models component, but the WandB Weave and Viz components are easy to add as extras. To install just WandB Models:

**pip**
```bash
pip install "inspect_wandb @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

**uv**
```bash
uv pip install "inspect_wandb @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```
To install WandB Models and WandB Weave:

**pip**
```bash
pip install "inspect_wandb[weave] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

**uv**
```bash
uv pip install "inspect_wandb[weave] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

And to install WandB Models, WandB Weave and Viz **(experimental)**: 

**pip**
```bash
pip install "inspect_wandb[weave,viz] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

**uv**
```bash
uv pip install "inspect_wandb[weave,viz] @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

If you intend to use the Viz integration, you also need to subsequently install `chromium` with:

```bash
playwright install-deps chromium
```