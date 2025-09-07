# Contributing to Inspect WandB

We welcome contributions from the community! This is a young project, so we are very open to feature requests and/or PRs for features or bug fixes that you'd like to see. If you want to discuss new features, bugs or the overall direction of Inspect WandB with us, the best place to do so is the [Inspect Community #inspect_wandb Slack Channel](https://inspectcommunity.slack.com/archives/C09B5B00459)

## Development

If you want to develop this project, you can fork and clone the repo and then run:

```bash
uv sync --group dev
source .venv/bin/activate
uv add pre-commit 
pre-commit install
```

to install for local development.

If you want to develop the Weave extra, the first command should instead be:

```bash
uv sync --group dev --extra weave
```

## Updating the docs

If you are making changes to or adding core features, please consider updating the documentation as necessary. We use [MyST](https://myst-parser.readthedocs.io/en/latest/index.html) to write documentation using Markdown syntax.

If editing the docs you can test your changes locally by running:

```bash
uv sync --group docs-dev
cd docs
make html
open build/html/index.html
```

or on Windows:

```bash
uv sync --group docs-dev
cd docs
.\make.bat html
start build/html/index.html
```

## Testing

We write unit tests with `pytest`. If you want to run the tests, you can simply run `pytest`. Please consider writing at least one test if adding a new feature, or covering edge cases with a test if submitting bug fixes.