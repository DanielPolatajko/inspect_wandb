from inspect_wandb.config.settings.base import InspectWandBBaseSettings
import pytest
from pathlib import Path
from unittest.mock import patch
import os

class TestInspectWandBBaseSettings:
    """
    Tests the shared ordering and validation logic of the base settings class.
    """

    def test_reads_settings_from_wandb_settings_file(self, initialise_wandb: None) -> None:
        # Given
        cwd = Path.cwd()
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(cwd / "wandb")):
            settings = InspectWandBBaseSettings.model_validate({})
            
        # Then
        assert settings.enabled is True
        assert settings.entity == "test-entity"
        assert settings.project == "test-project"   

    def test_environment_variables_set_to_bools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("ENABLED", False)

        # When
        settings = InspectWandBBaseSettings.model_validate({})
            
        # Then
        assert settings.enabled is False

    def test_hooks_disabled_when_project_and_entity_are_not_set_but_hooks_are_enabled(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Given
        cwd = Path.cwd()
        os.chdir(tmp_path) # prevents settings being read from non-test settings file

        monkeypatch.setenv("ENABLED", True)

        # When
        settings = InspectWandBBaseSettings.model_validate({})

        # Then
        assert settings.enabled is False

        os.chdir(cwd) # restore cwd

    def test_no_validation_errors_when_hooks_are_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("ENABLED", False)

        # When / Then
        InspectWandBBaseSettings.model_validate({})

class TestPriorityOrdering:
    """
    Tests the priority ordering of the different ways to pass settings to the base settings class.
    """

    def test_init_settings_highest_priority(self, monkeypatch: pytest.MonkeyPatch, initialise_wandb: None) -> None:
        # Given
        cwd = Path.cwd()
        
        monkeypatch.setenv("ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(cwd / "wandb")):
            settings = InspectWandBBaseSettings(
                enabled=True,
                WANDB_PROJECT="init-project",
                WANDB_ENTITY="init-entity",
            )
            
        # Then
        assert settings.enabled is True
        assert settings.project == "init-project"
        assert settings.entity == "init-entity"

    def test_environment_variables_second_priority(self, monkeypatch: pytest.MonkeyPatch, initialise_wandb: None) -> None:
        # Given
        cwd = Path.cwd()
        
        monkeypatch.setenv("ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(cwd / "wandb")):
            settings = InspectWandBBaseSettings.model_validate({})
            
        # Then
        assert settings.enabled is False
        assert settings.project == "env-project"
        assert settings.entity == "env-entity"

    def test_wandb_settings_third_priority(self, initialise_wandb: None) -> None:
        # Given
        cwd = Path.cwd()
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(cwd / "wandb")):
            settings = InspectWandBBaseSettings.model_validate({})
            
        # Then
        assert settings.project == "test-project"
        assert settings.entity == "test-entity"
        assert settings.enabled is True
        
