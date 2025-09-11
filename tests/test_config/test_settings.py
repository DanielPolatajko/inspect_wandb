import pytest
import os
from typing import Any
from pathlib import Path
from unittest.mock import patch
from inspect_wandb.config.settings import ModelsSettings, WeaveSettings, InspectWandBSettings
from inspect_wandb.config.wandb_settings_source import WandBSettingsSource
from inspect_wandb.config.settings_loader import SettingsLoader
from pydantic import ValidationError

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
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            source = WandBSettingsSource(ModelsSettings)
            result = source()
            
        # Then
        assert result == {
            'WANDB_ENTITY': 'source-test-entity',
            'WANDB_PROJECT': 'source-test-project'
        }
    
    def test_wandb_settings_source_with_missing_file(self, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
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
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
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
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            source = WandBSettingsSource(ModelsSettings)
            result1 = source()
            
            settings_file.write_text("[default]\nentity=modified\nproject=modified")
            result2 = source()
            
        # Then
        assert result1 == result2
        assert result1['WANDB_ENTITY'] == 'cached-entity'


class TestModelsSettings:
    
    def test_default_values_with_mock_wandb(self, tmp_path: Path) -> None:
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
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings.model_validate({})
            
        # Then
        assert settings.enabled is True
        assert settings.config is None
        assert settings.files is None
        assert settings.entity == "wandb-entity"
        assert settings.project == "wandb-project"

    def test_init_settings_highest_priority(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_FILES", '["env-file1.txt", "env-file2.txt"]')
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings(
                enabled=True,
                WANDB_PROJECT="init-project",
                WANDB_ENTITY="init-entity",
                files=["init-file1.txt", "init-file2.txt"]
            )
            
        # Then
        assert settings.enabled is True
        assert settings.project == "init-project"
        assert settings.entity == "init-entity"
        assert settings.files == ["init-file1.txt", "init-file2.txt"]

    def test_environment_variables_second_priority(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_FILES", '["env-file1.txt", "env-file2.txt"]')
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings.model_validate({})
            
        # Then
        assert settings.enabled is False
        assert settings.project == "env-project"
        assert settings.entity == "env-entity"
        assert settings.files == ["env-file1.txt", "env-file2.txt"]

    def test_environment_variables_set_to_bools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", False)
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", False)

        # When
        models_settings = ModelsSettings.model_validate({})
        weave_settings = WeaveSettings.model_validate({})
            
        # Then
        assert models_settings.enabled is False
        assert weave_settings.enabled is False

    def test_validation_errors_when_project_and_entity_are_not_set_but_hooks_are_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", True)
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", True)

        # When / Then
        with pytest.raises(ValidationError):
            ModelsSettings.model_validate({})
        with pytest.raises(ValidationError):
            WeaveSettings.model_validate({})

    def test_no_validation_errors_when_hooks_are_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", False)
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", False)

        # When / Then
        ModelsSettings.model_validate({})
        WeaveSettings.model_validate({})

    def test_errors_for_invalid_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", False)
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENVIRONMENT_VALIDATIONS", '{"wandb_base_url": "https://api.wandb.ai", "wandb_api_key": "1234567890"}')

        # When / Then
        with pytest.raises(ValidationError):
            ModelsSettings.model_validate({})

    def test_passes_validations_when_environment_variables_are_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given

        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", False)
        monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.ai")
        monkeypatch.setenv("WANDB_API_KEY", "1234567890")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENVIRONMENT_VALIDATIONS", '{"wandb_base_url": "https://api.wandb.ai", "wandb_api_key": "1234567890"}')

        # When
        models_settings = ModelsSettings.model_validate({})

        # Then
        assert models_settings.enabled is False
        assert models_settings.environment_validations.wandb_base_url == "https://api.wandb.ai"
        assert models_settings.environment_validations.wandb_api_key == "1234567890"

    def test_wandb_settings_third_priority(self, tmp_path: Path) -> None:
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
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings.model_validate({})
            
        # Then
        assert settings.project == "wandb-project"
        assert settings.entity == "wandb-entity"
        assert settings.enabled is True

    def test_init_settings_override_environment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_FILES", '["env-file.txt"]')
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = ModelsSettings(
                enabled=True,
                WANDB_PROJECT="init-project",
                WANDB_ENTITY="init-entity",
                files=["init-file.txt"]
            )
            
        # Then
            assert settings.enabled is True
            assert settings.files == ["init-file.txt"]
            assert settings.project == "init-project"
            assert settings.entity == "init-entity"

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

    def test_config_field_serialization(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """
        [default]
        entity = test-entity
        project = test-project
        """
        settings_file.write_text(settings_content)
        
        config_json = '{"learning_rate": 0.001, "batch_size": 32, "nested": {"value": true}}'
        monkeypatch.setenv("INSPECT_WANDB_MODELS_CONFIG", config_json)
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
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


class TestWeaveSettings:
    
    def test_default_values_with_mock_wandb(self, tmp_path: Path) -> None:
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
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = WeaveSettings.model_validate({})
            
        # Then
        assert settings.enabled is True
        assert settings.entity == "wandb-entity"
        assert settings.project == "wandb-project"

    def test_init_settings_highest_priority(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-weave-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-weave-entity")
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = WeaveSettings(
                enabled=True,
                WANDB_PROJECT="init-weave-project",
                WANDB_ENTITY="init-weave-entity"
            )
            
        # Then
        assert settings.enabled is True
        assert settings.project == "init-weave-project"
        assert settings.entity == "init-weave-entity"

    def test_environment_variables_second_priority(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-weave-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-weave-entity")
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = WeaveSettings.model_validate({})
            
        # Then
        assert settings.enabled is False
        assert settings.project == "env-weave-project"
        assert settings.entity == "env-weave-entity"

    def test_wandb_settings_third_priority(self, tmp_path: Path) -> None:
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
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = WeaveSettings.model_validate({})
            
        # Then
        assert settings.project == "wandb-project"
        assert settings.entity == "wandb-entity"
        assert settings.enabled is True
    
    def test_init_settings_override_environment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-weave-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-weave-entity")
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = WeaveSettings(
                enabled=True,
                WANDB_PROJECT="init-weave-project",
                WANDB_ENTITY="init-weave-entity"
            )
            
        # Then
            assert settings.enabled is True
            assert settings.project == "init-weave-project"
            assert settings.entity == "init-weave-entity"
    
    def test_pyproject_toml_lowest_priority(self, tmp_path: Path) -> None:
        # Given
        pyproject_content = """
        [tool.inspect-wandb.weave]
        enabled = false
        """
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """
        [default]
        entity = wandb-entity
        project = wandb-project
        """
        settings_file.write_text(settings_content)
        
        original_cwd = os.getcwd()
        
        # When
        try:
            os.chdir(tmp_path)
            with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings = WeaveSettings.model_validate({})
                
        # Then
            assert settings.enabled is False
            assert settings.project == "wandb-project"
            assert settings.entity == "wandb-entity"
        finally:
            os.chdir(original_cwd)

    def test_pyproject_toml_field_names(self, tmp_path: Path) -> None:
        # Given
        pyproject_content = """
        [tool.inspect-wandb.weave]
        enabled = false
        entity = "field-entity"
        project = "field-project"
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
                settings = WeaveSettings.model_validate({})
                
        # Then
            assert settings.enabled is False
            assert settings.entity == "field-entity"
            assert settings.project == "field-project"
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_alias_names(self, tmp_path: Path) -> None:
        # Given
        pyproject_content = """
        [tool.inspect-wandb.weave]
        enabled = false
        WANDB_ENTITY = "alias-entity"
        WANDB_PROJECT = "alias-project"
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
                settings = WeaveSettings.model_validate({})
                
        # Then
            assert settings.enabled is False
            assert settings.entity == "alias-entity"
            assert settings.project == "alias-project"
        finally:
            os.chdir(original_cwd)
    
    def test_pyproject_toml_field_vs_alias_consistency(self, tmp_path: Path) -> None:
        # Given
        pyproject_content_field = """
        [tool.inspect-wandb.weave]
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
                settings_field = WeaveSettings.model_validate({})
                
            pyproject_content_alias = """
            [tool.inspect-wandb.weave]
            WANDB_ENTITY = "test-entity"
            WANDB_PROJECT = "test-project"
            """
            pyproject_path.write_text(pyproject_content_alias)
            
            with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
                settings_alias = WeaveSettings.model_validate({})
                
            assert settings_field.entity == settings_alias.entity == "test-entity"
            assert settings_field.project == settings_alias.project == "test-project"
            assert settings_field.enabled == settings_alias.enabled
        finally:
            os.chdir(original_cwd)


class TestInspectWeaveSettings:
    
    def test_composite_model_structure(self, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """
        [default]
        entity = test-entity
        project = test-project
        """
        settings_file.write_text(settings_content)
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            settings = InspectWandBSettings(
                weave=WeaveSettings.model_validate({}),
                models=ModelsSettings.model_validate({})
            )
            
        # Then
        assert isinstance(settings.weave, WeaveSettings)
        assert isinstance(settings.models, ModelsSettings)
        assert settings.weave.project == "test-project"
        assert settings.models.project == "test-project"


class TestSettingsLoader:
    
    @patch('inspect_wandb.config.wandb_settings_source.wandb_dir')
    def test_load_inspect_wandb_settings_success(self, mock_wandb_dir: Any, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        settings_file = wandb_dir / "settings"
        settings_content = """
        [default]
        entity = test-entity
        project = test-project
        """
        settings_file.write_text(settings_content)
        mock_wandb_dir.return_value = str(wandb_dir)
        
        # When
        settings = SettingsLoader.load_inspect_wandb_settings()
        
        # Then
        assert isinstance(settings, InspectWandBSettings)
        assert settings.weave.entity == "test-entity"
        assert settings.weave.project == "test-project"
        assert settings.models.entity == "test-entity"
        assert settings.models.project == "test-project"
        assert settings.weave.enabled is True
        assert settings.models.enabled is True

    @patch('inspect_wandb.config.wandb_settings_source.wandb_dir')
    def test_load_inspect_wandb_settings_with_env_override(self, mock_wandb_dir: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        mock_wandb_dir.return_value = str(wandb_dir)
        
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "false")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        
        # When
        settings = SettingsLoader.load_inspect_wandb_settings()
        
        # Then
        assert settings.weave.entity == "env-entity"
        assert settings.weave.project == "env-project"
        assert settings.models.entity == "env-entity"
        assert settings.models.project == "env-project"
        assert settings.weave.enabled is False
        assert settings.models.enabled is False

    @patch('inspect_wandb.config.wandb_settings_source.wandb_dir')
    def test_load_inspect_wandb_settings_with_init_override(self, mock_wandb_dir: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        mock_wandb_dir.return_value = str(wandb_dir)
        
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "false")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_ENABLED", "false")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        
        # When
        settings = SettingsLoader.load_inspect_wandb_settings({
            "weave": {"enabled": True, "WANDB_PROJECT": "init-weave-project", "WANDB_ENTITY": "init-weave-entity"},
            "models": {"enabled": True, "WANDB_PROJECT": "init-models-project", "WANDB_ENTITY": "init-models-entity"}
        })
        
        # Then
        assert settings.weave.entity == "init-weave-entity"
        assert settings.weave.project == "init-weave-project"
        assert settings.models.entity == "init-models-entity"
        assert settings.models.project == "init-models-project"
        assert settings.weave.enabled is True
        assert settings.models.enabled is True

    @patch('inspect_wandb.config.wandb_settings_source.wandb_dir')
    def test_load_inspect_wandb_settings_with_pyproject_customization(self, mock_wandb_dir: Any, tmp_path: Path) -> None:
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
        mock_wandb_dir.return_value = str(wandb_dir)
        
        pyproject_content = """
        [tool.inspect-wandb.weave]
        enabled = false

        [tool.inspect-wandb.models]
        enabled = true
        files = ["model_config.yaml"]
        """
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        original_cwd = os.getcwd()
        
        # When
        try:
            os.chdir(tmp_path)
            settings = SettingsLoader.load_inspect_wandb_settings()
            
        # Then
            assert settings.weave.entity == "wandb-entity"
            assert settings.weave.project == "wandb-project"
            assert settings.models.entity == "wandb-entity"
            assert settings.models.project == "wandb-project"
            assert settings.weave.enabled is False
            assert settings.models.enabled is True
            assert settings.models.files == ["model_config.yaml"]
        finally:
            os.chdir(original_cwd)

    @patch('inspect_wandb.config.wandb_settings_source.wandb_dir')
    def test_wandb_settings_file_not_found(self, mock_wandb_dir: Any, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        mock_wandb_dir.return_value = str(wandb_dir)
        
        # When/Then
        with pytest.raises(Exception):
            SettingsLoader.load_inspect_wandb_settings()


class TestPriorityOrderIntegration:
    
    @patch('inspect_wandb.config.wandb_settings_source.wandb_dir')
    def test_complete_priority_integration(self, mock_wandb_dir: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        mock_wandb_dir.return_value = str(wandb_dir)
        
        pyproject_content = """
        [tool.inspect-wandb.weave]
        enabled = false

        [tool.inspect-wandb.models]
        enabled = false
        files = ["pyproject-file.yaml"]
        config = {from = "pyproject"}
        """
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        monkeypatch.setenv("INSPECT_WANDB_WEAVE_ENABLED", "true")
        monkeypatch.setenv("INSPECT_WANDB_MODELS_FILES", '["env-file.yaml"]')
        
        original_cwd = os.getcwd()
        
        # When
        try:
            os.chdir(tmp_path)
            settings = SettingsLoader.load_inspect_wandb_settings()
            
        # Then
            assert settings.weave.enabled is True
            assert settings.models.files == ["env-file.yaml"]
            assert settings.models.enabled is False
            assert settings.models.config == {"from": "pyproject"}
            assert settings.weave.entity == "wandb-entity"
            assert settings.weave.project == "wandb-project"
            assert settings.models.entity == "wandb-entity"
            assert settings.models.project == "wandb-project"
        finally:
            os.chdir(original_cwd)

    def test_validation_errors_without_wandb(self, tmp_path: Path) -> None:
        # Given
        wandb_dir = tmp_path / "wandb"
        wandb_dir.mkdir()
        
        # When/Then
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(wandb_dir)):
            with pytest.raises(Exception):
                ModelsSettings.model_validate({})
                
            with pytest.raises(Exception):
                WeaveSettings.model_validate({})