(installation)=
# Installation

To use this integration, you should install the package in the Python environment where you are running Inspect - Inspect will automatically detect the hooks and utilise them during eval runs. The `inspect_wandb` integration has 2 components:

- **WandB Models**: This integrates Inspect with the WandB Models API to store eval run statistics and configuration files for reproducibility.
- **WandB Weave**: This integrates Inspect with the WandB Weave API which can be used to track and analyse eval scores, transcripts and metadata.

By default, this integration will only install and enable the WandB Models component, but WandB Weave is easy to add as an extra. To install just WandB Models:

```bash
pip install inspect-wandb
```
To install WandB Models and WandB Weave:

```bash
pip install inspect-wandb[weave]
```