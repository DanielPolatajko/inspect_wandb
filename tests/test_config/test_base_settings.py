import os
import pytest
from unittest.mock import patch

from inspect_wandb.config.settings import ModelsSettings


READ_NETRC_PATH = "inspect_wandb.config.settings.base.read_netrc_auth"


@pytest.mark.no_mock_api_key
class TestBaseSettingsApiKeyValidation:
    def test_disables_when_no_api_key(self) -> None:
        # Given
        env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
        # When
        with patch.dict(os.environ, env, clear=True):
            with patch(READ_NETRC_PATH, return_value=None):
                settings = ModelsSettings(
                    enabled=True,
                    project="test-project",
                    entity="test-entity",
                )

        # Then
        assert settings.enabled is False

    def test_stays_enabled_when_api_key_present(self) -> None:
        # Given/When
        with patch.dict(os.environ, {"WANDB_API_KEY": "test-api-key"}):
            settings = ModelsSettings(
                enabled=True,
                project="test-project",
                entity="test-entity",
            )

        # Then
        assert settings.enabled is True

    def test_skips_api_key_check_when_already_disabled(self) -> None:
        # Given
        env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
        # When
        with patch.dict(os.environ, env, clear=True):
            with patch(READ_NETRC_PATH, return_value=None) as mock_read_netrc:
                settings = ModelsSettings(
                    enabled=False,
                    project="test-project",
                    entity="test-entity",
                )

        # Then
        assert settings.enabled is False
        mock_read_netrc.assert_not_called()
