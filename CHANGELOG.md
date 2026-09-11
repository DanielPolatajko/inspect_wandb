## [v0.3.0](https://pypi.org/project/inspect-wandb/0.3.0/) (11 September 2026)

### Added
- Add `agent_sessions_include_content` Weave setting controlling whether full message and tool content is included on the streamed spans. Set `false` to log only structure (turns, tool names, tokens, timing, scores) for very long-horizon evals where per-turn transcript volume is prohibitive.
- Set OpenTelemetry status on the streamed agent-session spans: tool and model failures are marked `ERROR` with the failure message, successful spans `OK`, so failed turns and tool calls surface as errors in the Weave Agents view rather than looking successful.

### Changed
- **Breaking:** the `weave` extra now requires `weave >= 0.53.0` (the agent Conversation SDK), raised from `0.52.43`, and `inspect_ai` requires `>= 0.3.217` (the `on_sample_event` hook). Environments pinned to weave 0.52.x must upgrade.
- Add `agent_sessions` Weave setting that streams each Inspect sample's agent trajectory to Weave's agent Conversation SDK (the Agents view) as turns complete, enabling live, turn-level observability and server-side Monitors/Signals on long-horizon agentic evals. 
- Bump minimum `weave` to `0.53.0` (the agent Conversation SDK) and `inspect_ai` to `0.3.217` (the `on_sample_event` hook).

### Fixed
- Follow weave's `weave.session` -> `weave.conversation` rename (weave 0.53). The old module was reduced to a deprecation shim with `session_otel`/`types` removed, which made `inspect_wandb.weave.hooks` unimportable and took the whole hook entry point down on any weave >= 0.53. Thanks to [@mlsimon734](https://github.com/mlsimon734) for the diagnosis.
- Run the linting and testing checks on dependency changes (`pyproject.toml`, `uv.lock`), which previously skipped CI entirely and let incompatible dependency bumps merge green.
- Postprocess Task States using Inspect AI's state_jsonable method to show readable dicts instead of showing Python object reprs in Weave.
- Stop importing `wandb.old.core.wandb_dir`, which was removed in wandb 0.27.1 and made the package fail to import on newer wandb versions. The wandb settings file is now located via wandb's public `Settings` API.
- Drain pending per-sample Weave logging tasks before finalizing the evaluation in `on_task_end`, so scores are no longer silently dropped by a race with `log_summary` (`Cannot log score after finish has been called`).

### New Contributors

- [@mlsimon734](https://github.com/mlsimon734)
- [@abhiramvsmg](https://github.com/abhiramvsmg)

## [v0.2.3](https://pypi.org/project/inspect-wandb/0.2.3/) (16 March 2026)

### Fixed
- Pin `wandb` dependency to ensure stable version

## New Contributors

- [@ItsTania](https://github.com/ItsTania)

## [v0.2.2](https://pypi.org/project/inspect-wandb/0.2.2/) (15 March 2026)

### Added
- Add `eval_traces_only` setting to disable sample-level Weave traces and only log eval-level summaries
- Validate user is authed with wandb and disable hooks if not

### Fixed
- Disable auto-init for Weave client on import when active wandb run is present
- Properly close Anthropic streaming calls

## [v0.2.1](https://pypi.org/project/inspect-wandb/0.2.1/) (17 December 2025)

### Added
- Add task metadata to WandB Run config

### Fixed
- Handle incorrect wandb entity and project exceptions
- Handle changing args in autopatched functions

### New contributors
- [@kohankhaki](https://github.com/kohankhaki)

## [v0.2.0](https://pypi.org/project/inspect-wandb/0.2.0/) (29 October 2025)

### Added
- Convert built-in Inspect `choice` and `match` scorer values to booleans when logging scores to Weave
- Add eval-set log dir to Weave Evaluation metadata

### Fixed
- Bump minimum Weave version to fix Pydantic validation error on summary aggregation
- Autopatch based on installed libs, rather than blanket patching
- Load settings every time enabled is called
- Remove custom EvaluationLogger to enable dataset comparison
- Bug with out-of-date syntax for importlib utils

### New Contributors
- [@alex-remedios-aisi](https://github.com/alex-remedios-aisi)

## [v0.1.7](https://pypi.org/project/inspect-wandb/0.1.7/) (06 October 2025)


### Added
- Custom trace names to differentiate OpenRouter API calls from OpenAI completions

### Fixed
- Add scorer traces to correct parent sample when running with multiple epochs

## [v0.1.6](https://pypi.org/project/inspect-wandb/0.1.6/) (23 September 2025)

### Added
- Bumped Inspect to v0.3.133 in order to handle exit exceptions gracefully

### Fixed
- Concurrency issues for Weave writes on sample end

## [v0.1.5](https://pypi.org/project/inspect-wandb/0.1.5/) (19 September 2025)

### Fixed
- Broken docs build

## [v0.1.4](https://pypi.org/project/inspect-wandb/0.1.4/) (19 September 2025)

### Added
- Updated docs to include links and concepts page


## [v0.1.3](https://pypi.org/project/inspect-wandb/0.1.3/) (16 September 2025)

### Fixed
- Use `run_id` to track Models runs for `inspect eval` rather than `eval_id`


## [v0.1.2](https://pypi.org/project/inspect-wandb/0.1.2/) (12 September 2025)

### Added
- Write wandb and weave URLs to Inspect eval metadata in log files
- Environment variable validations for wandb base url and API key

### Fixed
- Case sensitivity when parsing settings from eval(-set) metadata

## [v0.1.1](https://pypi.org/project/inspect-wandb/0.1.1/) (08 September 2025)

### Added

- This CHANGELOG!
- Contributor guidelines

### Fixed
- Simplified log summary of outputs metric on Weave
- Better handling of error states for Models runs

## [v0.1.0](https://pypi.org/project/inspect-wandb/0.1.0/) (07 September 2025)

### Added

- Initial release

### New Contributors

- [@DanielPolatajko](https://github.com/DanielPolatajko)
- [@Esther-Guo](https://github.com/Esther-Guo)
- [@scottire](https://github.com/scottire)
- [@GnarlyMshtep](https://github.com/GnarlyMshtep)