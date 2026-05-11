import pytest
from unittest.mock import MagicMock, patch
from secret_manager import SecretManager

def test_secret_manager_env_fallback():
    cfg = {
        "secret_management": {
            "provider": "env",
            "key_name": "TEST_API_KEY"
        }
    }
    with patch("os.getenv", return_value="env_secret"):
        sm = SecretManager(cfg)
        assert sm.get_secret() == "env_secret"

def test_secret_manager_keyring():
    cfg = {
        "secret_management": {
            "provider": "keyring",
            "key_name": "TEST_API_KEY",
            "keyring_service": "test-service"
        }
    }
    with patch("keyring.get_password", return_value="keyring_secret"):
        sm = SecretManager(cfg)
        assert sm.get_secret() == "keyring_secret"

def test_secret_manager_aws():
    cfg = {
        "secret_management": {
            "provider": "aws",
            "key_name": "TEST_API_KEY",
            "aws_region": "us-east-1"
        }
    }
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.get_secret_value.return_value = {"SecretString": "aws_secret"}
    
    with patch("boto3.client", return_value=mock_client):
        sm = SecretManager(cfg)
        assert sm.get_secret() == "aws_secret"

def test_secret_manager_aws_json():
    cfg = {
        "secret_management": {
            "provider": "aws",
            "key_name": "TEST_API_KEY",
            "aws_region": "us-east-1"
        }
    }
    mock_client = MagicMock()
    # AWS often returns a JSON string
    mock_client.get_secret_value.return_value = {"SecretString": '{"TEST_API_KEY": "json_secret"}'}
    
    with patch("boto3.client", return_value=mock_client):
        sm = SecretManager(cfg)
        assert sm.get_secret() == "json_secret"
