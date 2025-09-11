from inspect_wandb.config.wandb_settings_source import WandBSettingsSource
from inspect_wandb.config.settings import ModelsSettings
from pathlib import Path
from unittest.mock import patch


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