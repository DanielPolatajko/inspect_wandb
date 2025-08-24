* Quickstart (5min)
    * installation (i'll introduce with the example but its not specific) 
    * running stuff
    * some things to see in the UI
        * weave
            * evals table 
            * how to look at individual samples
            * filtering 
                *  saved views
            * trace tree 
                * sample --> summaary (metadata per sample)
            * eval comparison 
            * status 
        * models
            * one command --> one models run (is it the run ID?)
            * logs (need fixing)
            * saved files (point to config to understand how to choose files)

* Configuration 
    * DanielPolatajko/inspect_weave/inspect_weave/config/settings.py
    * priorty
        1. Environment variables (highest priority)
        2. Wandb settings file (for entity/project)
        3. Initial settings (programmatic overrides)
        4. Pyproject.toml (lowest priority)
    * INSPECT_WEAVE_MODELS_{name}
    * inspect_wandb

* Reproducability (weave filter --> models run --> config and command --> rerun)


check
* run stop and start 
~ Filter by user Scott
* add an issue for weave pointers