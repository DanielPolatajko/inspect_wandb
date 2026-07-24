import os
from inspect_wandb.config.wandb_settings_source import WandBSettingsSource, wandb_dir
from inspect_wandb.config.settings import ModelsSettings
from pathlib import Path
from unittest.mock import patch
import pytest


class TestWandBSettingsSource:
    def test_wandb_settings_source_with_valid_file(self, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """
        [default]
        entity = source-test-entity
        project = source-test-project
        """
        settings_file.write_text(settings_content)

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source.wandb_dir",
            return_value=str(wandb_dir),
        ):
            source = WandBSettingsSource(ModelsSettings)
            result = source()

        # Then
        assert result == {
            "entity": "source-test-entity",
            "project": "source-test-project",
        }

    def test_wandb_settings_source_with_missing_file(self, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source.wandb_dir",
            return_value=str(wandb_dir),
        ):
            source = WandBSettingsSource(ModelsSettings)
            result = source()

        # Then
        assert result == {}

    def test_wandb_settings_source_with_invalid_file(self, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_file.write_text("invalid content")

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source.wandb_dir",
            return_value=str(wandb_dir),
        ):
            source = WandBSettingsSource(ModelsSettings)
            result = source()

        # Then
        assert result == {}

    def test_wandb_settings_source_caches_results(self, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """
        [default]
        entity = cached-entity
        project = cached-project
        """
        settings_file.write_text(settings_content)

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source.wandb_dir",
            return_value=str(wandb_dir),
        ):
            source = WandBSettingsSource(ModelsSettings)
            result1 = source()

            settings_file.write_text("[default]\nentity=modified\nproject=modified")
            result2 = source()

        # Then
        assert result1 == result2
        assert result1["entity"] == "cached-entity"


class TestWandBDirResolution:
    def test_honors_wandb_dir_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))

        # When / Then
        assert wandb_dir() == str(tmp_path / "wandb")

    def test_prefers_hidden_directory_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        (tmp_path / ".wandb").mkdir()

        # When / Then
        assert wandb_dir() == str(tmp_path / ".wandb")

    def test_falls_back_to_cwd_when_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.delenv("WANDB_DIR", raising=False)
        monkeypatch.chdir(tmp_path)

        # When / Then
        assert wandb_dir() == os.path.join(os.getcwd(), "wandb")

    def test_source_reads_settings_located_via_wandb_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        wandb_subdir = tmp_path / "wandb"
        wandb_subdir.mkdir()
        (wandb_subdir / "settings").write_text(
            "[default]\nentity = env-entity\nproject = env-project\n"
        )

        # When
        source = WandBSettingsSource(ModelsSettings)
        result = source()

        # Then
        assert result == {"entity": "env-entity", "project": "env-project"}
