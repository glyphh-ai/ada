"""
Execution providers — bolt-on action execution for MCP models.

Each provider implements the ExecutionProvider protocol and handles
vendor-specific logic (Pipedream, Zapier, Make, etc.).  Providers
are loaded dynamically based on model config:

    execute:
      provider: pipedream   # → shared.providers.pipedream

Models without an ``execute`` section don't expose the execute tool.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ExecutionProvider(Protocol):
    """Protocol for action execution providers."""

    async def execute(
        self,
        action_key: str,
        props: dict[str, Any],
        external_user_id: str,
    ) -> dict[str, Any]:
        """Execute an action and return the result.

        Args:
            action_key: Action component ID (e.g. "slack-send-message")
            props: Configured properties for the action
            external_user_id: The end-user's ID in the caller's system

        Returns:
            Provider-specific result dict with at least:
              {"success": bool, "exports": dict, "error": str|None}
        """
        ...


# Registry of loaded providers (keyed by model config tuple)
_provider_cache: dict[tuple[str, str], ExecutionProvider] = {}


def get_provider(
    org_id: str,
    model_id: str,
    execute_config: dict[str, Any],
) -> ExecutionProvider:
    """Get or create the execution provider for a model.

    Loads the provider module from ``shared.providers.<name>`` and
    calls its ``create_provider(config)`` factory.

    Raises:
        ValueError: If the provider name is missing or the module
                    can't be loaded.
    """
    key = (org_id, model_id)
    if key in _provider_cache:
        return _provider_cache[key]

    provider_name = execute_config.get("provider", "")
    if not provider_name:
        raise ValueError("execute.provider is required in model config")

    module_path = f"shared.providers.{provider_name}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Provider '{provider_name}' not found ({module_path}). "
            f"Install its dependencies or check the provider name. {e}"
        ) from e

    factory = getattr(mod, "create_provider", None)
    if factory is None:
        raise ValueError(
            f"Provider module {module_path} must export create_provider(config)"
        )

    provider = factory(execute_config)
    _provider_cache[key] = provider
    logger.info(f"Loaded execution provider '{provider_name}' for {org_id}/{model_id}")
    return provider
