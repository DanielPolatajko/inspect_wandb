from inspect_wandb.config.wandb_settings_source import WandBSettingsSource
from inspect_wandb.config.settings import ModelsSettings
from pathlib import Path
from unittest.mock import patch
import pytest


class TestWandBSettingsSource:
    def test_wandb_settings_source_with_valid_file(self, tmp_path: Path) -> None:
        # Given
        settings_file = tmp_path / "settings"
        settings_file.write_text(
            """
        [default]
        entity = source-test-entity
        project = source-test-project
        """
        )

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source._wandb_settings_path",
            return_value=settings_file,
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
        settings_file = tmp_path / "settings"

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source._wandb_settings_path",
            return_value=settings_file,
        ):
            source = WandBSettingsSource(ModelsSettings)
            result = source()

        # Then
        assert result == {}

    def test_wandb_settings_source_with_invalid_file(self, tmp_path: Path) -> None:
        # Given
        settings_file = tmp_path / "settings"
        settings_file.write_text("invalid content")

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source._wandb_settings_path",
            return_value=settings_file,
        ):
            source = WandBSettingsSource(ModelsSettings)
            result = source()

        # Then
        assert result == {}

    def test_wandb_settings_source_caches_results(self, tmp_path: Path) -> None:
        # Given
        settings_file = tmp_path / "settings"
        settings_file.write_text(
            """
        [default]
        entity = cached-entity
        project = cached-project
        """
        )

        # When
        with patch(
            "inspect_wandb.config.wandb_settings_source._wandb_settings_path",
            return_value=settings_file,
        ):
            source = WandBSettingsSource(ModelsSettings)
            result1 = source()

            settings_file.write_text("[default]\nentity=modified\nproject=modified")
            result2 = source()

        # Then
        assert result1 == result2
        assert result1["entity"] == "cached-entity"


class TestWandBSettingsSourceIntegration:
    def test_source_reads_settings_resolved_via_wandb(
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
