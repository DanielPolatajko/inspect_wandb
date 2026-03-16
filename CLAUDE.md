## Project context

This project is an open-source extension for the Inspect AI LLM evals framework. Any other context required should be clear from the project @README.md.

## Working norms

### Autonomy and scope

- When uncertain between two reasonable design or implementation approaches, always ask before proceeding.
- Before making any structural changes (new modules, reorganising existing layout), ask first. Clean conceptual separation is valued — new modules are fine when warranted, but the decision should be deliberate.
- If you notice a related bug or improvement while working on a task, flag it. Do not fix it in the same branch — it should be discussed and addressed in a separate, tightly scoped commit.
- Keep commits tightly scoped to the task at hand.

### Pull request standards

- New functionality requires tests.
- Update @CHANGELOG.md for new features and significant bugfixes. Chores and trivial fixes do not need changelog entries.
- Update docs only for user-facing changes (new config, new behaviour, new interfaces).
- Run the full check suite before marking work as done: type-checking, linting, and tests.

### Running commands

- Always run commands with `uv run ...`.

## Code style

### General principles

- Avoid unnecessary comments. Code should be written clearly with explanatory variable names and obvious flow. Only use comments when absolutely necessary to explain something unintuitive.
- Never write module-level docstrings.
- Keep the codebase lean and DRY. Remove dead code rather than leaving it around. Avoid duplication — if the same logic appears more than once, extract it.
- We use ruff for linting — check with `uv run ruff check inspect_wandb tests`.
- Prefer `from lib import Class` over `import lib` followed by `lib.Class` usage.

### Typing

- Always use modern Python typing, especially for unions (e.g. `X | Y` rather than `Union[X, Y]`).
- Don't import from `typing` for types which are natively supported (e.g. use `list` rather than `typing.List`).
- Type-checking is done with MyPy — ensure this passes before submitting changes.
- Only use `from __future__ import annotations` if you need to resolve types which have not been defined yet.

### Public vs internal API

- Maintain strict separation between public and internal interfaces. Use `_` prefix for internal functions and non-public attributes/methods within classes. Do not prefix module files with `_`.
- This package is pre-1.0 (currently v0.2.x), so breaking changes in minor versions are acceptable, but should still be flagged in the changelog.

### Error handling

- The general principle is to handle errors gracefully and avoid interfering with the Inspect eval run.
- Inspect has its own error handling, so raising exceptions is generally fine — but **settings loading is an exception**: errors here can break the entire run, so they must be caught and handled gracefully.
- Always raise and handle exceptions explicitly. Do not silently swallow errors without good reason.

### Logging

- Use Python's standard `logging` module. Do not use `print` for runtime output.
- Log only where it is genuinely useful to the user. Avoid verbose or redundant log statements.

### Adding dependencies

- This is an open-source package, so new dependencies are added with caution. If you need functionality that an existing library provides and reimplementing it would take more than ~20 lines, present adding the dependency as an option rather than just doing it.

## Testing

- Run tests with pytest. Always run new tests you have written and ensure they pass.
- Prefer running subsets of tests over the whole suite, but run all tests (except skipped) when touching core functionality or adding significant features.
- When writing tests, use the Given/When/Then structure. The only inline comments allowed in tests are the `# Given`, `# When`, `# Then` markers.
- When mocking, use `unittest.mock.patch` as a context manager (`with patch(...):`) rather than as a decorator.
- Avoid mocking where possible. When mocking is necessary (e.g. external services such as wandb or weave), mock at the SDK client level — do not test against real services.
- If you touch an area that appears undertested, flag it.

### Manual testing

```
uv run inspect eval inspect_evals/hellaswag --model anthropic/claude-sonnet-4-5 --metadata inspect_wandb_models_project=test --metadata inspect_wandb_weave_project=test --limit 1
```

Other configurations will be specified by the developer where needed. Use the above as a base template.

## Documentation

- Docs are deployed on readthedocs and found at @docs. They update automatically on release and push to main.
- Document new user-facing features — new config options, interfaces, or behaviour changes.
- We use MyST Parser to write documentation in Markdown — follow its conventions and best practices.
- Suggestions for improving existing documentation gaps are welcome outside the scope of any current task.

### Changelog

- We maintain @CHANGELOG.md to track changes across releases.
- Follow the existing format closely. Add entries under `## Unreleased`.

## Navigation

- When investigating lib code or dependency behaviour, look in @.venv/Lib/site-packages.
- The most relevant packages are `weave` and `inspect_ai`.
