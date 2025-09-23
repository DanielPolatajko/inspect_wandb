import os
from pathlib import Path
from unittest.mock import patch
import pytest
from inspect_wandb.config.settings import ModelsSettings
from pydantic import ValidationError

class TestModelsSettings:

    def test_default_values(self, initialise_wandb: None) -> None:
        # Given
        cwd = Path.cwd()
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(cwd / "wandb")):
            settings = ModelsSettings.model_validate({})
            
        # Then
        assert settings.enabled is True
        assert settings.config is None
        assert settings.files is None
        assert settings.viz is False
        assert settings.add_metadata_to_config is True
        assert settings.tags is None
        assert settings.environment_validations is None
        assert settings.entity == "test-entity"
        assert settings.project == "test-project"

    def test_errors_for_invalid_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "False")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENVIRONMENT_VALIDATIONS", '{"wandb_base_url": "https://api.wandb.ai", "wandb_api_key": "1234567890"}')

        # When / Then
        with pytest.raises(ValidationError):
            ModelsSettings.model_validate({})

    def test_passes_validations_when_environment_variables_are_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "False")
        monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.ai")
        monkeypatch.setenv("WANDB_API_KEY", "1234567890")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENVIRONMENT_VALIDATIONS", '{"wandb_base_url": "https://api.wandb.ai", "wandb_api_key": "1234567890"}')

        # When
        models_settings = ModelsSettings.model_validate({})

        # Then
        assert models_settings.enabled is False
        assert models_settings.environment_validations.wandb_base_url == "https://api.wandb.ai"
        assert models_settings.environment_validations.wandb_api_key == "1234567890"

    def test_complete_priority_order(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """
        [default]
        entity = wandb-entity
        project = wandb-project
        """
        settings_file.write_text(settings_content)
        
        pyproject_content = """
        [tool.inspect-wandb.models]
        enabled = false
        files = ["toml-file.txt"]
        """
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "true")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_FILES", '["env-file.txt"]')
        
        original_cwd = os.getcwd()
        
        # When
        try:
            os.chdir(tmp_path)
            with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = ModelsSettings(
                    WANDB_PROJECT="init-project",
                    WANDB_ENTITY="init-entity"
                )
                
        # Then
            assert settings.enabled is True
            assert settings.files == ["env-file.txt"]
            assert settings.project == "init-project"
            assert settings.entity == "init-entity"
        finally:
            os.chdir(original_cwd)

    def test_config_field_serialization(self, monkeypatch: pytest.MonkeyPatch, initialise_wandb: None) -> None:
        # Given
        
        config_json = '{"learning_rate": 0.001, "batch_size": 32, "nested": {"value": true}}'
        monkeypatch.setenv("INSPECT_WANDB_MODELS_CONFIG", config_json)
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(Path.cwd() / "wandb")):
            settings = ModelsSettings.model_validate({})
            
        # Then
        expected_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "nested": {"value": True}
        }
        assert settings.config == expected_config

    def test_pyproject_toml_field_names(self, tmp_path: Path) -> None:
        # Given
        pyproject_content = """
        [tool.inspect-wandb.models]
        enabled = false
        entity = "field-entity"
        project = "field-project"
        files = ["field-file.txt"]
        """
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        original_cwd = os.getcwd()
        
        # When
        try:
            os.chdir(tmp_path)
            with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = ModelsSettings()
                
        # Then
            assert settings.enabled is False
            assert settings.entity == "field-entity"
            assert settings.project == "field-project"
            assert settings.files == ["field-file.txt"]
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_alias_names(self, tmp_path: Path) -> None:
        # Given
        pyproject_content = """
        [tool.inspect-wandb.models]
        enabled = false
        WANDB_ENTITY = "alias-entity"
        WANDB_PROJECT = "alias-project"
        files = ["alias-file.txt"]
        """
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        original_cwd = os.getcwd()
        
        # When
        try:
            os.chdir(tmp_path)
            with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = ModelsSettings.model_validate({})
                
        # Then
            assert settings.enabled is False
            assert settings.entity == "alias-entity"
            assert settings.project == "alias-project"
            assert settings.files == ["alias-file.txt"]
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_field_vs_alias_consistency(self, tmp_path: Path) -> None:
        # Given
        pyproject_content_field = """
        [tool.inspect-wandb.models]
        entity = "test-entity"
        project = "test-project"
        """
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content_field)
        
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        original_cwd = os.getcwd()
        
        # When/Then
        try:
            os.chdir(tmp_path)
            with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings_field = ModelsSettings.model_validate({})
                
            pyproject_content_alias = """
            [tool.inspect-wandb.models]
            WANDB_ENTITY = "test-entity"
            WANDB_PROJECT = "test-project"
            """
            pyproject_path.write_text(pyproject_content_alias)
            
            with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings_alias = ModelsSettings.model_validate({})
                
            assert settings_field.entity == settings_alias.entity == "test-entity"
            assert settings_field.project == settings_alias.project == "test-project"
            assert settings_field.enabled == settings_alias.enabled
        finally:
            os.chdir(original_cwd)