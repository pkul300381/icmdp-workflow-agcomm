"""
secret_manager.py
-----------------
Secure secret management utility for the ICMDP application.
Supports fetching API keys from:
  - Python Keyring
  - AWS Secrets Manager
  - Azure Key Vault
  - Atlassian Vault (Placeholder)
"""
import os
import logging

logger = logging.getLogger(__name__)

class SecretManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("secret_management", {})
        self.provider = self.cfg.get("provider", "env").lower()
        self.key_name = self.cfg.get("key_name", "CLAUDE_API_KEY")

    def get_secret(self) -> str:
        """Fetch the secret from the configured provider."""
        if self.provider == "env":
            return os.getenv(self.key_name, "")
        
        if self.provider == "keyring":
            return self._get_keyring_secret()
        
        if self.provider == "aws":
            return self._get_aws_secret()
        
        if self.provider == "azure":
            return self._get_azure_secret()
        
        if self.provider == "atlassian":
            return self._get_atlassian_secret()
        
        logger.warning(f"Unknown secret provider: {self.provider}. Falling back to environment variables.")
        return os.getenv(self.key_name, "")

    def _get_keyring_secret(self) -> str:
        try:
            import keyring
            service = self.cfg.get("keyring_service", "icmdp-agent")
            secret = keyring.get_password(service, self.key_name)
            return secret if secret else ""
        except ImportError:
            logger.error("keyring library not installed. pip install keyring")
            return ""

    def _get_aws_secret(self) -> str:
        try:
            import boto3
            import json
            region = self.cfg.get("aws_region", "us-east-1")
            client = boto3.client("secretsmanager", region_name=region)
            response = client.get_secret_value(SecretId=self.key_name)
            if "SecretString" in response:
                secret_data = response["SecretString"]
                # AWS often stores secrets as JSON strings
                try:
                    return json.loads(secret_data).get(self.key_name, secret_data)
                except json.JSONDecodeError:
                    return secret_data
            return ""
        except ImportError:
            logger.error("boto3 library not installed. pip install boto3")
            return ""
        except Exception as e:
            logger.error(f"Error fetching secret from AWS: {e}")
            return ""

    def _get_azure_secret(self) -> str:
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            vault_url = self.cfg.get("azure_vault_url")
            if not vault_url:
                logger.error("Azure Vault URL not configured.")
                return ""
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            return client.get_secret(self.key_name).value
        except ImportError:
            logger.error("azure-keyvault-secrets library not installed.")
            return ""
        except Exception as e:
            logger.error(f"Error fetching secret from Azure: {e}")
            return ""

    def _get_atlassian_secret(self) -> str:
        # Placeholder for Atlassian Vault implementation
        logger.warning("Atlassian Vault support is not yet implemented.")
        return ""
