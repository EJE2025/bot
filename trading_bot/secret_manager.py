import os
import logging
from typing import Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None
    BotoCoreError = ClientError = Exception

logger = logging.getLogger(__name__)


def get_secret(name: str) -> Optional[str]:
    """Return a secret from AWS Secrets Manager or environment variables.

    If boto3 is not installed or the secret fetch fails, fall back to
    ``os.getenv(name)``.
    """
    if boto3 is None:
        return os.getenv(name)
    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=name)
        return response.get("SecretString")
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network errors
        logger.warning("No se pudo recuperar %s de Secrets Manager: %s", name, exc)
        return os.getenv(name)
