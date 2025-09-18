from inspect_wandb.config.settings import WeaveSettings
from unittest.mock import patch
from pathlib import Path
import os

class TestWeaveSettings:
    
    def test_default_values(self, initialise_wandb: None) -> None:
        # Given
        cwd = Path.cwd()
        
        # When
        with patch('inspect_wandb.config.wandb_settings_source.wandb_dir', return_value=str(cwd / "wandb")):
            settings = WeaveSettings.model_validate({})
            
        # Then
        assert settings.enabled is True
        assert settings.autopatch is True
        assert settings.sample_name_template == "{task_name}-sample-{sample_id}-epoch-{epoch}"
        assert settings.entity == "test-entity"
        assert settings.project == "test-project"
    
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
        autopatch = false
        sample_name_template = "field-sample-name-template"
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
            assert settings.autopatch is False
            assert settings.sample_name_template == "field-sample-name-template"
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