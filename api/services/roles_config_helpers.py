from __future__ import annotations

from fastapi import HTTPException


ALLOWED_ROLE_TYPES = {"string", "number", "boolean", "date", "datetime"}
TEMPORAL_FIELDS = {
    "event_date",
    "valid_from",
    "valid_to",
    "observed_at",
    "period_start",
    "period_end",
}


def validate_roles_config(roles_config: dict) -> dict:
    if not isinstance(roles_config, dict):
        raise HTTPException(status_code=400, detail="roles_config must be an object")
    layers = roles_config.get("layers")
    if not isinstance(layers, list) or not layers:
        raise HTTPException(status_code=400, detail="roles_config.layers must be a non-empty array")

    for layer in layers:
        if not isinstance(layer, dict):
            raise HTTPException(status_code=400, detail="roles_config.layers items must be objects")
        segments = layer.get("segments")
        if not isinstance(segments, list) or not segments:
            raise HTTPException(status_code=400, detail="roles_config.layers.segments must be a non-empty array")
        for segment in segments:
            if not isinstance(segment, dict):
                raise HTTPException(status_code=400, detail="roles_config.segments items must be objects")
            roles = segment.get("roles")
            if not isinstance(roles, list) or not roles:
                raise HTTPException(status_code=400, detail="roles_config.segments.roles must be a non-empty array")
            for role_entry in roles:
                if isinstance(role_entry, str):
                    continue
                if not isinstance(role_entry, dict):
                    raise HTTPException(status_code=400, detail="roles_config roles must be strings or objects")
                role_name = role_entry.get("role")
                if not isinstance(role_name, str) or not role_name:
                    raise HTTPException(status_code=400, detail="roles_config role entries need a role string")
                role_type = role_entry.get("type", "string")
                if not isinstance(role_type, str) or role_type not in ALLOWED_ROLE_TYPES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"roles_config role '{role_name}' has invalid type '{role_type}'",
                    )
                if role_name in TEMPORAL_FIELDS and role_type not in {"date", "datetime", "string"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"roles_config temporal role '{role_name}' must be date/datetime/string",
                    )
                value_policy = role_entry.get("value_policy")
                if value_policy is not None and not isinstance(value_policy, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=f"roles_config role '{role_name}' value_policy must be an object",
                    )
    return roles_config
