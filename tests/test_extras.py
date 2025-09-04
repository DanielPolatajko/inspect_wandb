import pytest
from pathlib import Path
import subprocess
from typing import Generator

@pytest.fixture(scope="function")
def base_install_venv(tmp_path: Path) -> Generator[Path, None, None]:
    subprocess.call(["uv", "venv", str(tmp_path)])
    subprocess.call(
        ["uv", "pip", "install", ".", "--python", tmp_path, ],
        cwd=Path(__file__).parent.parent,
    )
    yield tmp_path

@pytest.fixture(scope="function")
def base_and_weave_install_venv(tmp_path: Path) -> Generator[Path, None, None]:
    subprocess.call(["uv", "venv", str(tmp_path)])
    subprocess.call(
        ["uv", "pip", "install", ".[weave]", "--python", tmp_path, ],
        cwd=Path(__file__).parent.parent,
    )
    yield tmp_path

@pytest.fixture(scope="function")
def base_and_vision_install_venv(tmp_path: Path) -> Generator[Path, None, None]:
    subprocess.call(["uv", "venv", str(tmp_path)])
    subprocess.call(
        ["uv", "pip", "install", ".[viz]", "--python", tmp_path, ],
        cwd=Path(__file__).parent.parent,
    )
    yield tmp_path

@pytest.fixture(scope="function")
def all_extras_install_venv(tmp_path: Path) -> Generator[Path, None, None]:
    subprocess.call(["uv", "venv", str(tmp_path)])
    subprocess.call(
        ["uv", "pip", "install", ".[weave,viz]", "--python", tmp_path, ],
        cwd=Path(__file__).parent.parent,
    )
    yield tmp_path

@pytest.mark.skip(reason="Long running, only run if testing the extras")
class TestExtraInstallations:

    def test_weave_hooks_dont_run_with_base_install(self, base_install_venv: Path) -> None:

        # Given (base_install)
        # When
        proc = subprocess.Popen(
            ["uv", "run", "--python", base_install_venv, "python", "-c", "from inspect_ai.hooks._startup import init_hooks; init_hooks()"],
            stdout=subprocess.PIPE
        )

        stdout, _ = proc.communicate()
        stdout = stdout.decode("utf-8")

        # Then
        assert "wandb_models_hooks" in stdout
        assert "weave_evaluation_hooks" not in stdout

    def test_weave_hooks_run_with_weave_install(self, base_and_weave_install_venv: Path) -> None:

        # Given (base_and_weave_install)
        # When
        proc = subprocess.Popen(
            ["uv", "run", "--python", base_and_weave_install_venv, "python", "-c", "from inspect_ai.hooks._startup import init_hooks; init_hooks()"],
            stdout=subprocess.PIPE
        )
        
        stdout, _ = proc.communicate()
        stdout = stdout.decode("utf-8")

        # Then
        assert "wandb_models_hooks" in stdout
        assert "weave_evaluation_hooks" in stdout

    def test_all_hooks_run_with_all_extras_install(self, all_extras_install_venv: Path) -> None:

        # Given (all_extras_install)
        # When
        proc = subprocess.Popen(
            ["uv", "run", "--python", all_extras_install_venv, "python", "-c", "from inspect_ai.hooks._startup import init_hooks; init_hooks()"],
            stdout=subprocess.PIPE
        )

        stdout, _ = proc.communicate()
        stdout = stdout.decode("utf-8")

        # Then
        assert "wandb_models_hooks" in stdout
        assert "weave_evaluation_hooks" in stdout

    def test_models_hooks_run_with_viz_install(self, base_and_vision_install_venv: Path) -> None:

        # Given (base_and_vision_install)
        # When
        proc = subprocess.Popen(
            ["uv", "run", "--python", base_and_vision_install_venv, "python", "-c", "from inspect_ai.hooks._startup import init_hooks; init_hooks()"],
            stdout=subprocess.PIPE
        )

        stdout, _ = proc.communicate()
        stdout = stdout.decode("utf-8")

        # Then
        assert "wandb_models_hooks" in stdout
        assert "weave_evaluation_hooks" not in stdout

