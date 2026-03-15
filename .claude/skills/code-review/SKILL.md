Review the current branch against a base branch, fix any issues found, and confirm nothing is broken.

The base branch is `$ARGUMENTS` if provided, otherwise `main`.

## Steps

1. Run `git diff <base>...HEAD --name-only` to list changed files.
2. Run `git diff <base>...HEAD` to get the full diff.
3. Run the quality gate suite and note any failures:
   - `uv run ruff check inspect_wandb tests`
   - `uv run mypy inspect_wandb`
   - `uv run pytest .`
4. Review the diff against each norm below and identify all violations.
5. Fix every issue you can fix confidently. For each fix, make the minimal change required — do not refactor beyond the scope of the violation.
6. Re-run the full quality gate suite to confirm nothing is broken.
7. Present a summary of what was fixed. If any issues could not be fixed automatically (e.g. missing tests for new functionality, missing changelog entry, architectural concerns), list them clearly as things the developer must address manually.

## What to check and fix

**Ruff / MyPy failures** — fix any linting or type errors surfaced by the tools.

**Typing**
- Unions should be `X | Y`, not `Union[X, Y]`
- Use built-in generics (`list[...]`, `dict[...]`), not `typing.List` etc.
- Remove `from __future__ import annotations` unless forward references strictly require it

**Code style**
- Remove unnecessary comments and module-level docstrings
- Replace any `print()` calls with appropriate `logging` calls
- Remove dead code — unused functions, variables, imports, and unreachable branches
- Flag any duplication that could be extracted into a shared helper
- Prefer `from lib import Class` over `import lib` followed by `lib.Class` usage — fix any violations

**Error handling**
- Ensure exceptions are not silently swallowed without good reason
- Settings-loading code must catch and handle errors gracefully — never let exceptions propagate from there as they will break the entire Inspect run

**Public/internal API**
- Add `_` prefix to any functions, classes, or modules that are internal but not currently marked as such

## What to flag but not fix automatically

- Missing tests for new functionality — flag which code paths lack coverage
- Missing or incomplete `CHANGELOG.md` entry under `## Unreleased` (required for new features and significant bugfixes, not for chores or trivial fixes)
- Missing docs updates for user-facing changes (new config, interfaces, or behaviour)
- New third-party imports not in `pyproject.toml`
- Anything requiring a design decision or structural change
