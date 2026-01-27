from __future__ import annotations

from typing import Any

from ..core import models


def resolve_mcp_config(
    db,
    *,
    model_id: str,
    endpoint_name: str,
) -> dict[str, Any] | None:
    if not model_id or not endpoint_name:
        return None
    cfg = db.query(models.ModelMcpConfig).filter(models.ModelMcpConfig.model_id == model_id).first()
    if not cfg or not isinstance(cfg.config, dict):
        return None
    endpoints = cfg.config.get("endpoints")
    if isinstance(endpoints, dict) and endpoint_name in endpoints:
        endpoint_cfg = endpoints.get(endpoint_name)
        if isinstance(endpoint_cfg, dict):
            return endpoint_cfg
        return {"endpoint_name": endpoint_name}
    stored_name = cfg.config.get("endpoint_name")
    if isinstance(stored_name, str) and stored_name == endpoint_name:
        return cfg.config
    return None
