from unittest.mock import patch

from inspect_wandb.config.settings import ModelsSettings


API_KEY_PATH = "inspect_wandb.config.settings.base.api_key"


class TestBaseSettingsApiKeyValidation:

    def test_disables_when_no_api_key(self) -> None:
        # Given/When
        with patch(API_KEY_PATH, return_value=None):
            settings = ModelsSettings(
                enabled=True,
                project="test-project",
                entity="test-entity",
            )

        # Then
        assert settings.enabled is False

    def test_stays_enabled_when_api_key_present(self) -> None:
        # Given/When
        with patch(API_KEY_PATH, return_value="test-api-key"):
            settings = ModelsSettings(
                enabled=True,
                project="test-project",
                entity="test-entity",
            )

        # Then
        assert settings.enabled is True

    def test_skips_api_key_check_when_already_disabled(self) -> None:
        # Given/When
        with patch(API_KEY_PATH, return_value=None) as mock_api_key:
            settings = ModelsSettings(
                enabled=False,
                project="test-project",
                entity="test-entity",
            )

        # Then
        assert settings.enabled is False
        mock_api_key.assert_not_called()
