# Installation

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