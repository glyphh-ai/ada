"""
Pipedream Connect execution provider.

Wraps the Pipedream Connect SDK to execute actions on behalf of users.
Credentials come from environment variables:

    PIPEDREAM_CLIENT_ID
    PIPEDREAM_CLIENT_SECRET
    PIPEDREAM_PROJECT_ID
    PIPEDREAM_ENVIRONMENT   (default: "production")
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class PipedreamProvider:
    """Execute Pipedream actions via the Connect SDK."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        project_id: str,
        environment: str = "production",
    ):
        try:
            from pipedream import Pipedream
        except ImportError:
            raise ImportError(
                "Pipedream SDK not installed. Run: pip install pipedream"
            )

        self._client = Pipedream(
            client_id=client_id,
            client_secret=client_secret,
            project_id=project_id,
            project_environment=environment,
        )
        self._project_id = project_id
        self._environment = environment

    async def execute(
        self,
        action_key: str,
        props: dict[str, Any],
        external_user_id: str,
    ) -> dict[str, Any]:
        """Execute a Pipedream action via Connect SDK.

        The SDK client is synchronous, so we call it directly
        (FastAPI runs this in a threadpool via async def).
        """
        try:
            result = self._client.actions.run(
                id=action_key,
                external_user_id=external_user_id,
                configured_props=props or {},
            )

            # Normalize the response
            return {
                "success": True,
                "exports": getattr(result, "exports", {}) or {},
                "return_value": getattr(result, "ret", None),
                "error": None,
            }

        except Exception as e:
            logger.error(
                f"Pipedream execute failed: action={action_key} "
                f"user={external_user_id} error={e}",
                exc_info=True,
            )
            return {
                "success": False,
                "exports": {},
                "return_value": None,
                "error": str(e),
            }


def create_provider(config: dict[str, Any]) -> PipedreamProvider:
    """Factory function — called by the provider registry.

    Reads credentials from env vars (not from config, for security).
    Config can override the environment (development vs production).
    """
    client_id = os.environ.get("PIPEDREAM_CLIENT_ID", "")
    client_secret = os.environ.get("PIPEDREAM_CLIENT_SECRET", "")
    project_id = os.environ.get("PIPEDREAM_PROJECT_ID", "")
    environment = config.get(
        "environment",
        os.environ.get("PIPEDREAM_ENVIRONMENT", "production"),
    )

    if not client_id or not client_secret or not project_id:
        raise ValueError(
            "Pipedream provider requires env vars: "
            "PIPEDREAM_CLIENT_ID, PIPEDREAM_CLIENT_SECRET, PIPEDREAM_PROJECT_ID"
        )

    return PipedreamProvider(
        client_id=client_id,
        client_secret=client_secret,
        project_id=project_id,
        environment=environment,
    )
