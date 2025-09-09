
# Inspect WandB
Integration between [Inspect](https://inspect.aisi.org.uk/) and [Weights & Biases](https://wandb.ai/site/), including support for both the Models API for experiment tracking, and Weave for evaluation analysis and transcripts.

![Docs Build](https://app.readthedocs.org/projects/inspect-wandb/badge/?version=latest)
[![Package Build](https://github.com/DanielPolatajko/inspect_wandb/actions/workflows/test-build.yml/badge.svg)](https://github.com/DanielPolatajko/inspect_wandb/actions/workflows/test-build.yml)
[![Publish to PyPI](https://github.com/DanielPolatajko/inspect_wandb/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/DanielPolatajko/inspect_wandb/actions/workflows/publish-to-pypi.yml)

## Demo Video

Check out this brief demo video for an overview of Inspect WandB

<div>
    <a href="https://www.loom.com/share/1578ad78581146d08348cfe2a13270b0">
      <p>WIP: Integrating Inspect WandB with Inspect AI for LLM Evaluations 🚀 - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/1578ad78581146d08348cfe2a13270b0">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/1578ad78581146d08348cfe2a13270b0-d6183465b48a6d2b-full-play.gif">
    </a>
  </div>

## Usage

Inspect WandB can be installed with:

```bash
pip install inspect-wandb
```

To install the optional Weave extra:
```bash
pip install inspect-wandb[weave]
```

Once Inspect WandB is installed in an environment authenticated with Weights & Biases (either by running `wandb login` or setting `WANDB_API_KEY`), the integration will be enabled for future Inspect runs by default. The Inspect logger output will link to the Models dashboard where you can track runs, and also, if you have enabled the `weave` extra, to the Weave dashboard where you can visualise eval results.

Some configuration options are available, including adjusting `wandb` config, settings tags, and adjusting Weave trace naming. To dive deeper with Inspect WandB, please see the documentation at [https://inspect-wandb.readthedocs.io/en/latest/](https://inspect-wandb.readthedocs.io/en/latest/)

## Contributing

Please see our [contributing guidelines](./CONTRIBUTING.md) if you'd like to make contributions to Inspect WandB

## Feedback

We welcome all feedback; the best way to get in touch to discuss the project is the [Inspect Community #inspect_wandb Slack Channel](https://inspectcommunity.slack.com/archives/C09B5B00459)

## Project notes

This project was primarily developed by [DanielPolatajko](https://github.com/DanielPolatajko), [Qi Guo](https://github.com/Esther-Guo), [Matan Shtepel](https://github.com/GnarlyMshtep), and supervised by Justin Olive. It was supported through the MARS (Mentorship for Alignment Research Students) program at the [Cambridge AI Safety Hub](https://www.cambridgeaisafety.org/mars).

