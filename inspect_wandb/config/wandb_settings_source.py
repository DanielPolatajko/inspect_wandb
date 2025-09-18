import configparser
from pathlib import Path
from typing import Any
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings
from pydantic_settings.sources import PydanticBaseSettingsSource
from wandb.old.core import wandb_dir
import logging

logger = logging.getLogger(__name__)


class WandBSettingsSource(PydanticBaseSettingsSource):
    """
    A pydantic settings source that reads settings from a wandb settings file,
    as set during the wandb init CLI flow. 
    
    This source provides entity and project values from the wandb settings file
    for fields that have WANDB_PROJECT or WANDB_ENTITY aliases.
    """
    
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._wandb_settings: dict[str, str] | None = None
        
    def _load_wandb_settings(self) -> dict[str, str]:
        """Load wandb settings from the wandb settings file."""
        if self._wandb_settings is not None:
            return self._wandb_settings
            
        settings_path = Path(wandb_dir()) / "settings"
        
        if not settings_path.exists():
            logger.debug("Wandb settings file not found, skipping WandBSettingsSource")
            self._wandb_settings = {}
            return self._wandb_settings
            
        try:
            with open(settings_path, "r") as f:
                parser = configparser.ConfigParser()
                parser.read_file(f)
                
            if 'default' not in parser:
                logger.warning("No 'default' section found in wandb settings file")
                self._wandb_settings = {}
                return self._wandb_settings
                
            default_section = parser['default']
            self._wandb_settings = {
                'entity': default_section.get('entity', ''),
                'project': default_section.get('project', '')
            }
            
            logger.debug(f"Loaded wandb settings: entity={self._wandb_settings.get('entity')}, project={self._wandb_settings.get('project')}")
            
        except Exception as e:
            logger.warning(f"Failed to read wandb settings file: {e}")
            self._wandb_settings = {}
            
        return self._wandb_settings
    
    def get_field_value(self, field_info: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """
        Get the value for a field from wandb settings.
        
        Args:
            field_info: Tuple of (field_name, FieldInfo)
            field_name: The name of the field
            
        Returns:
            Tuple of (value, source_description, is_complex)
        """
        
        # Only handle fields with WANDB_PROJECT or WANDB_ENTITY aliases
        if field_info.alias not in ('WANDB_PROJECT', 'WANDB_ENTITY'):
            return None, '', False
            
        wandb_settings = self._load_wandb_settings()
        
        if field_info.alias == 'WANDB_PROJECT':
            value = wandb_settings.get('project')
        elif field_info.alias == 'WANDB_ENTITY':
            value = wandb_settings.get('entity')
        else:
            return None, '', False
            
        if value:
            return value, f'wandb settings file ({field_info.alias})', False
        
        return None, '', False
    
    def __call__(self) -> dict[str, Any]:
        """
        Called by pydantic-settings to get all field values from this source.
        
        Returns:
            Dictionary of field values from wandb settings using alias names
        """
        d: dict[str, Any] = {}
        
        wandb_settings = self._load_wandb_settings()
        if not wandb_settings:
            return d
        
        # Return values using the alias names that pydantic expects
        if wandb_settings.get('project'):
            d['WANDB_PROJECT'] = wandb_settings['project']
        if wandb_settings.get('entity'):
            d['WANDB_ENTITY'] = wandb_settings['entity']
                
        return d