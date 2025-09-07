# Contributing 
We are currently actively developing the repo and are happy to receive new issues and pull requests. We are also in contact with WandB and are able to raise requests which we are not able to directly accommodate to their team.

## Development

If you want to develop this project, you can fork and clone the repo and then run:

```bash
uv sync --group dev
pre-commit install
```

to install for development locally.

### Testing

We write unit tests with `pytest`. If you want to run the tests, you can simply run `pytest`. Please consider writing a test if adding a new feature, and make sure that tests are passing before submitting changes.
