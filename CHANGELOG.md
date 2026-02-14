## Unreleased

### Fixed
- Auto-disable plugin when user is not authenticated with WandB

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