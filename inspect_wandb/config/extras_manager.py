import importlib

class ExtrasManager:
    def __init__(self):
        self.extras = {}

    def detect_extras(self) -> dict[str, bool]:
        self._check_for_weave_extra()
        self._check_for_viz_extra()
        return self.extras

    def _check_for_weave_extra(self) -> None:
        if importlib.util.find_spec("weave") is not None:
            self.extras["weave"] = True
        else:
            self.extras["weave"] = False

    def _check_for_viz_extra(self) -> None:
        if importlib.util.find_spec("inspect_viz") is not None and importlib.util.find_spec("playwright") is not None:
            self.extras["viz"] = False
        else:
            self.extras["viz"] = False

INSTALLED_EXTRAS = ExtrasManager().detect_extras()