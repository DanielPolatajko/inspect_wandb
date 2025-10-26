## Unreleased

### Added
- Add eval-set log dir to Weave Evaluation metadata

### Fixed
- Bump minimum Weave version to fix Pydantic validation error on summary aggregation
- Autopatch based on installed libs, rather than blanket patching

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