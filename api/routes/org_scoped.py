"""
Org-Scoped API Routes for Glyphh Runtime (Cloud Mode).

Endpoints scoped by org_id and model_id for multi-tenant cloud deployments.
URL pattern: /{org_id}/{model_id}/...

No namespace concept — org_id and model_id are passed directly to services.
"""

import logging
import shutil
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from domains.auth.service import AuthService
from domains.mcp.server import MCPServer
from domains.query.service import QueryService
from infrastructure.config import get_settings
from shared.auth import AuthenticatedUser, get_current_user, require_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/{org_id}/{model_id}", tags=["org-scoped"])
settings = get_settings()


# Dependency injection
async def get_auth_service() -> AuthService:
    return AuthService()


async def validate_org_access(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """
    Validate org access with local mode bypass.

    Used for model lifecycle endpoints (deploy, undeploy, status, re-encode)
    where the user is operating their own CLI.
    """
    if current_user.org_id == "local-dev-org":
        return current_user

    if current_user.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization mismatch - you don't have access to this organization"
        )

    return current_user


async def validate_token_access(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(require_token),
) -> AuthenticatedUser:
    """
    Always require a valid JWT token — no local mode bypass.

    Used for data and query endpoints (listener, MCP) that external
    services like Boomi, Make.com, or agents call with API tokens.
    """
    if current_user.org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail="Organization mismatch - you don't have access to this organization"
        )

    return current_user


async def get_mcp_server(
    org_id: str,
    model_id: str,
) -> MCPServer:
    """Get MCP server for org/model."""
    from glyphh.server import model_manager
    from infrastructure.database import async_session_maker

    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    query_service = QueryService(model_manager, async_session_maker)
    auth_service = AuthService()

    return MCPServer(query_service, auth_service)



class UndeployRequest(BaseModel):
    delete_data: bool = False


async def get_model_manager():
    """Get the ModelManager instance from glyphh.server."""
    from glyphh.server import model_manager

    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager




# MCP Endpoint
@router.post("/mcp")
async def mcp_endpoint(
    org_id: str,
    model_id: str,
    request: Dict[str, Any],
    mcp_server: MCPServer = Depends(get_mcp_server),
    current_user: AuthenticatedUser = Depends(validate_token_access),
) -> Dict[str, Any]:
    """
    MCP endpoint for org-scoped model access.

    Requires a valid JWT token (even in local mode). External services
    like Boomi, Make.com, and agents use API tokens to query models.
    """
    tool_name = request.get("tool")
    arguments = request.get("arguments", {})

    if not tool_name:
        raise HTTPException(status_code=400, detail="Missing 'tool' field")

    arguments["org_id"] = org_id
    arguments["model_id"] = model_id

    response = await mcp_server.handle_tool_call(
        tool_name=tool_name,
        arguments=arguments,
        auth_token="",
    )

    return response.to_dict()


@router.get("/mcp/tools")
async def list_mcp_tools(
    org_id: str,
    model_id: str,
    mcp_server: MCPServer = Depends(get_mcp_server),
    current_user: AuthenticatedUser = Depends(validate_token_access),
) -> Dict[str, Any]:
    return {"tools": mcp_server.get_tools_list()}


# Model Lifecycle Endpoints

@router.get("/ready")
async def readiness_check(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Check if a model is deployed and ready to serve queries."""
    from glyphh.server import model_manager

    if model_manager is None:
        return {"ready": False, "status": "model_manager_not_initialized", "model_id": model_id}

    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        return {"ready": False, "status": "not_deployed", "model_id": model_id}

    is_locked = model_manager.is_model_locked(org_id, model_id)
    if is_locked:
        return {"ready": False, "status": "locked", "model_id": model_id}

    return {
        "ready": True,
        "status": "ready",
        "model_id": model_id,
        "meta_name": loaded_model.meta_name,
    }


@router.post("/model/deploy")
async def deploy_model(
    org_id: str,
    model_id: str,
    file: Optional[UploadFile] = File(None),
    model_path: Optional[str] = None,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Deploy a .glyphh model.

    Accepts either:
    - A multipart file upload of a .glyphh file (field name: file)
    - A form field model_path pointing to a filesystem path

    Supports two .glyphh formats:
    - ZIP archive (from CLI packaging): unpacked and loaded via directory path
    - Gzip JSON (from SDK GlyphhModel.to_file): loaded via GlyphhModel.from_file
    """
    from shared.exceptions import ModelLoadException, ModelIncompatibleException

    resolved_path: Optional[str] = None
    tmp_path: Optional[str] = None
    tmp_dir: Optional[str] = None

    try:
        if file is not None:
            # Multipart file upload — save to temp file
            suffix = ".glyphh"
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                content = await file.read()
                os.write(tmp_fd, content)
            finally:
                os.close(tmp_fd)
            resolved_path = tmp_path
        elif model_path is not None:
            resolved_path = model_path
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a .glyphh file upload or a model_path.",
            )

        # Detect file format by magic bytes
        with open(resolved_path, "rb") as f:
            magic = f.read(2)

        if magic == b"PK":
            # ZIP format (CLI packaging) — unpack and load from directory
            from glyphh.cli.packaging import unpack_model

            tmp_dir = tempfile.mkdtemp(prefix="glyphh_deploy_")
            model_dir = unpack_model(Path(resolved_path), dest=Path(tmp_dir) / model_id)

            loaded_model = await model_manager.load_model_from_directory(
                model_dir=model_dir,
                org_id=org_id,
                model_id=model_id,
            )
        elif magic == b"\x1f\x8b":
            # Gzip JSON format (SDK GlyphhModel.to_file)
            loaded_model = await model_manager.load_model(
                model_path=resolved_path,
                org_id=org_id,
                model_id=model_id,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unrecognized .glyphh file format (magic bytes: {magic!r}). "
                       f"Expected ZIP (CLI package) or gzip (SDK model).",
            )

        return {
            "status": "deployed",
            "model_id": model_id,
            "version": getattr(loaded_model.sdk_model, "version", "unknown"),
            "meta_name": loaded_model.meta_name,
        }

    except ModelLoadException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelIncompatibleException as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file if we created one
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)




@router.post("/model/undeploy")
async def undeploy_model(
    org_id: str,
    model_id: str,
    request: UndeployRequest = UndeployRequest(),
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Unload a model from memory, optionally deleting its data."""
    from shared.exceptions import ModelNotFoundException

    try:
        await model_manager.unload_model(
            org_id=org_id,
            model_id=model_id,
            delete_data=request.delete_data,
        )
        return {"status": "undeployed", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


@router.post("/model/re-encode")
async def re_encode_model(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Re-encode all glyphs for a deployed model."""
    from shared.exceptions import ModelNotFoundException

    # Check if model is loaded
    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")

    # Check if re-encode is already in progress
    if model_manager.is_model_locked(org_id, model_id):
        raise HTTPException(status_code=409, detail="Re-encode already in progress for this model")

    try:
        result = await model_manager.re_encode_model(
            org_id=org_id,
            model_id=model_id,
        )
        return {"job_id": result.job_id, "status": result.status}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


@router.delete("/model")
async def delete_model(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
    model_manager=Depends(get_model_manager),
) -> Dict[str, Any]:
    """Full delete: unload model + purge all data (config, glyphs, vectors, edges, procedures)."""
    from shared.exceptions import ModelNotFoundException
    from infrastructure.database import async_session_maker
    from sqlalchemy import delete as sql_delete
    from domains.procedures.models import StoredProcedureModel

    # Check if model exists
    loaded_model = await model_manager.get_model(org_id, model_id)
    if loaded_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")

    # Check if re-encode is in progress
    if model_manager.is_model_locked(org_id, model_id):
        raise HTTPException(status_code=409, detail="Cannot delete model while re-encode is in progress")

    try:
        # Unload model and delete config + glyph data
        await model_manager.unload_model(
            org_id=org_id,
            model_id=model_id,
            delete_data=True,
        )

        # Purge stored procedures
        async with async_session_maker() as session:
            await session.execute(
                sql_delete(StoredProcedureModel).where(
                    StoredProcedureModel.org_id == org_id,
                    StoredProcedureModel.model_id == model_id,
                )
            )
            await session.commit()

        return {"status": "deleted", "model_id": model_id}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=f"Model not found: {org_id}/{model_id}")


# ── Data Management Endpoints ──

@router.get("/data")
async def list_data(
    org_id: str,
    model_id: str,
    limit: int = 20,
    offset: int = 0,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """List glyphs stored for this model with pagination."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        count = await storage.count_glyphs(org_id, model_id)
        glyphs = await storage.list_glyphs(org_id, model_id, limit=limit, offset=offset)

    return {
        "total": count,
        "limit": limit,
        "offset": offset,
        "glyphs": [
            {
                "id": str(g.id),
                "concept_text": g.concept_text[:200] if g.concept_text else "",
                "metadata": g.metadata,
                "created_at": g.created_at.isoformat() + "Z" if g.created_at else None,
            }
            for g in glyphs
        ],
    }


@router.get("/data/count")
async def count_data(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Count glyphs and edges for this model."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        glyph_count = await storage.count_glyphs(org_id, model_id)
        vector_count = await storage.count_glyph_vectors(org_id, model_id)

    return {
        "glyphs": glyph_count,
        "vectors": vector_count,
        "model_id": model_id,
    }


@router.delete("/data")
async def clear_data(
    org_id: str,
    model_id: str,
    current_user: AuthenticatedUser = Depends(validate_org_access),
) -> Dict[str, Any]:
    """Clear all glyphs and edges for this model without unloading it."""
    from infrastructure.database import async_session_maker
    from domains.models.storage import GlyphStorage

    async with async_session_maker() as session:
        storage = GlyphStorage(session)
        glyphs_deleted, edges_deleted = await storage.delete_model_data(org_id, model_id)
        await session.commit()

    return {
        "status": "cleared",
        "model_id": model_id,
        "glyphs_deleted": glyphs_deleted,
        "edges_deleted": edges_deleted,
    }


# ── Chat UI ───────────────────────────────────────────────────────────────────

def _load_asset_b64(filename: str) -> str | None:
    """Load an image from glyphh-runtime/public/ as base64.

    Returns base64-encoded string, or None if not found.
    Loaded once at module import time — not per-request.
    """
    import base64
    # glyphh-runtime/public/ — the runtime's own public assets
    p = Path(__file__).resolve().parent.parent.parent / "public" / filename
    if p.exists():
        try:
            return base64.b64encode(p.read_bytes()).decode("ascii")
        except Exception:
            pass
    return None


_LOGO_B64: str | None = _load_asset_b64("glyphh-logo.png")
_FAVICON_B64: str | None = _load_asset_b64("favicon.png")


@router.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui(org_id: str, model_id: str):
    """Browser-based GPT-style chat tester. NL + GQL tabs, split results panel."""
    return HTMLResponse(content=_chat_html(org_id, model_id, settings.deployment_mode))


def _chat_html(org_id: str, model_id: str, deployment_mode: str = "local") -> str:  # noqa: E501
    logo_tag = (
        f'<img src="data:image/png;base64,{_LOGO_B64}" alt="glyphh" style="height:28px;display:block;">'
        if _LOGO_B64
        else '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <polygon points="12,2 21,7 21,17 12,22 3,17 3,7" fill="#9333ea" opacity="0.9"/>
          <polygon points="12,5 18.5,8.75 18.5,16.25 12,20 5.5,16.25 5.5,8.75" fill="none" stroke="#a855f7" stroke-width="0.75"/>
          <circle cx="12" cy="12" r="3" fill="#c084fc"/>
        </svg><span class="logo-word">glyphh</span>'''
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>glyphh · {model_id}</title>
{f'<link rel="icon" type="image/png" href="data:image/png;base64,{_FAVICON_B64}">' if _FAVICON_B64 else ''}
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0a0a;--surface:#111118;--surface-input:#141418;
  --border:#1a1a2e;--border-input:#2a2a3a;
  --text:#fff;--text-light:#e5e5e5;--text-muted:#6b7280;--placeholder:#3a3a4a;
  --purple:#9333ea;--purple-light:#a855f7;--purple-dim:rgba(147,51,234,0.12);
  --error:#ef4444;--success:#22c55e;
  --mono:"Menlo","Monaco","Courier New",monospace;
}}
html,body{{height:100%;background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;line-height:1.5}}
/* ── App shell ── */
.app{{display:flex;flex-direction:column;height:100vh;overflow:hidden}}
/* ── Header ── */
.header{{display:flex;align-items:center;justify-content:space-between;padding:0 20px;height:52px;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--surface)}}
.logo{{display:flex;align-items:center;gap:10px}}
.logo img{{display:block;height:28px;width:auto}}
.logo-word{{font-size:15px;font-weight:700;letter-spacing:-0.5px;color:var(--text)}}
.model-badge{{display:flex;align-items:center;gap:8px;padding:4px 10px;background:var(--purple-dim);border:1px solid rgba(147,51,234,0.25);border-radius:6px}}
.model-badge .name{{color:var(--purple-light);font-size:12px;font-weight:600}}
.model-badge .ver{{color:var(--text-muted);font-size:11px}}
/* ── Tabs ── */
.tabs{{display:flex;align-items:center;gap:0;border-bottom:1px solid var(--border)}}
.tab{{padding:10px 20px;font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text-muted);background:none;border:none;border-bottom:2px solid transparent;cursor:pointer;transition:color .15s,border-color .15s;margin-bottom:-1px}}
.tab:hover{{color:var(--text-light)}}
.tab.active{{color:var(--purple-light);border-bottom-color:var(--purple)}}
/* ── Body ── */
.body{{display:flex;flex:1;overflow:hidden}}
/* ── Chat panel ── */
.chat-panel{{display:flex;flex-direction:column;flex:0 0 60%;border-right:1px solid var(--border);overflow:hidden}}
.messages{{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}}
.messages::-webkit-scrollbar{{width:3px}}
.messages::-webkit-scrollbar-thumb{{background:var(--purple);border-radius:2px}}
.msg-row{{display:flex;flex-direction:column}}
.msg-row.user{{align-items:flex-end}}
.msg-row.assistant{{align-items:flex-start}}
.bubble{{max-width:80%;padding:10px 14px;border-radius:12px;font-size:13px;line-height:1.5;word-break:break-word}}
.msg-row.user .bubble{{background:var(--purple);color:#fff;border-radius:12px 12px 2px 12px}}
.msg-row.assistant .bubble{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--purple);border-radius:2px 12px 12px 12px}}
.msg-row.error .bubble{{background:var(--surface);border:1px solid rgba(239,68,68,0.3);border-left:3px solid var(--error);border-radius:2px 12px 12px 12px;color:#fca5a5}}
.meta{{margin-top:5px;font-size:10px;color:var(--text-muted);display:flex;gap:8px;flex-wrap:wrap}}
.meta .conf{{color:var(--purple-light);font-weight:600}}
.meta .pill{{background:var(--purple-dim);border:1px solid rgba(147,51,234,0.2);border-radius:4px;padding:1px 5px;font-size:10px}}
/* ── Thinking dots ── */
.thinking{{display:flex;gap:5px;align-items:center;padding:12px 16px;background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--purple);border-radius:2px 12px 12px 12px;width:fit-content}}
.dot{{width:6px;height:6px;background:var(--purple-light);border-radius:50%;animation:bounce .9s ease-in-out infinite}}
.dot:nth-child(2){{animation-delay:.15s}}
.dot:nth-child(3){{animation-delay:.3s}}
@keyframes bounce{{0%,80%,100%{{transform:translateY(0)}}40%{{transform:translateY(-6px)}}}}
/* ── Input bar ── */
.input-bar{{flex-shrink:0;padding:14px 16px;border-top:1px solid var(--border);background:var(--surface)}}
.input-wrap{{display:flex;align-items:flex-end;gap:10px;background:var(--surface-input);border:1px solid var(--border-input);border-radius:12px;padding:8px 8px 8px 14px;transition:border-color .15s}}
.input-wrap:focus-within{{border-color:var(--purple)}}
#nl-input{{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:var(--mono);font-size:13px;resize:none;line-height:1.5;caret-color:var(--purple-light)}}
#nl-input::placeholder{{color:var(--placeholder)}}
#gql-input{{flex:1;background:none;border:none;outline:none;color:var(--text);font-family:var(--mono);font-size:13px;resize:none;line-height:1.5;min-height:60px;max-height:140px;caret-color:var(--purple-light)}}
#gql-input::placeholder{{color:var(--placeholder)}}
.send-btn{{width:36px;height:36px;border-radius:50%;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s,opacity .15s;flex-shrink:0}}
.send-btn:disabled{{background:var(--border-input);cursor:default;opacity:.5}}
.send-btn:not(:disabled){{background:var(--purple)}}
.send-btn:not(:disabled):hover{{background:var(--purple-light)}}
.send-btn svg{{pointer-events:none}}
.hint{{font-size:10px;color:var(--text-muted);margin-top:6px;padding-left:2px}}
/* ── Results panel ── */
.results-panel{{flex:0 0 40%;display:flex;flex-direction:column;overflow:hidden}}
.results-header{{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--surface)}}
.results-title{{font-size:11px;font-weight:700;letter-spacing:.08em;color:var(--text-muted);text-transform:uppercase}}
.clear-btn{{background:none;border:none;cursor:pointer;color:var(--text-muted);font-size:16px;line-height:1;padding:2px 4px;border-radius:4px;transition:color .15s,background .15s;font-family:var(--mono)}}
.clear-btn:hover{{color:var(--error);background:rgba(239,68,68,.08)}}
.results-body{{flex:1;overflow-y:auto;padding:12px}}
.results-body::-webkit-scrollbar{{width:3px}}
.results-body::-webkit-scrollbar-thumb{{background:var(--purple);border-radius:2px}}
.results-empty{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:8px;color:var(--text-muted)}}
.results-empty .icon{{font-size:28px;opacity:.3}}
.results-empty .label{{font-size:11px;opacity:.5}}
/* ── Result items ── */
.result-item{{margin-bottom:8px}}
details{{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;transition:border-color .15s}}
details:hover{{border-color:rgba(147,51,234,0.35)}}
details[open]{{border-color:var(--purple)}}
summary{{padding:10px 12px;cursor:pointer;font-size:12px;font-weight:600;color:var(--text-light);display:flex;align-items:center;gap:8px;list-style:none;user-select:none}}
summary::-webkit-details-marker{{display:none}}
.summary-arrow{{color:var(--purple-light);font-size:10px;transition:transform .15s;flex-shrink:0}}
details[open] .summary-arrow{{transform:rotate(90deg)}}
.summary-label{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.summary-count{{font-size:10px;color:var(--text-muted);flex-shrink:0}}
.state-badge{{font-size:9px;font-weight:700;letter-spacing:.06em;padding:2px 6px;border-radius:4px;flex-shrink:0;text-transform:uppercase}}
.state-DONE{{background:rgba(34,197,94,.15);color:#4ade80;border:1px solid rgba(34,197,94,.25)}}
.state-ASK{{background:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.25)}}
.state-BLOCKED{{background:rgba(249,115,22,.15);color:#fb923c;border:1px solid rgba(249,115,22,.25)}}
.state-AUTH_REQUIRED{{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.25)}}
.state-ERROR{{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.25)}}
.result-pre{{margin:0;padding:10px 12px;font-size:11px;line-height:1.6;color:#c4b5fd;background:rgba(0,0,0,.3);border-top:1px solid var(--border);overflow-x:auto;white-space:pre-wrap;word-break:break-all}}
.result-pre::-webkit-scrollbar{{height:3px}}
.result-pre::-webkit-scrollbar-thumb{{background:var(--purple);border-radius:2px}}
.result-query-label{{padding:4px 12px 8px;font-size:10px;color:var(--text-muted)}}
/* ── Match result rows (similarity search) ── */
.match-row{{display:flex;align-items:center;gap:8px;padding:2px 0;font-family:var(--mono)}}
.match-pct{{color:var(--purple-light);min-width:3.8em;text-align:right;font-size:12px;font-weight:600}}
.match-bar{{color:var(--purple-light);font-size:11px;opacity:0.65;letter-spacing:0}}
.match-name{{color:var(--text-light);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <div class="header">
    <div class="logo">
      {logo_tag}
    </div>
    <div class="model-badge">
      <span class="name">{model_id}</span>
      <span class="ver">{deployment_mode}</span>
    </div>
  </div>
{"" if deployment_mode == "local" else f'''
  <!-- Token banner (non-local mode) -->
  <div style="display:flex;align-items:center;gap:10px;padding:8px 20px;background:#0d0d14;border-bottom:1px solid var(--border);flex-shrink:0">
    <span style="color:var(--text-muted);font-size:11px;white-space:nowrap">API Token</span>
    <input id="token-input" type="password" placeholder="Paste your token (glyphh token create)" autocomplete="off"
      oninput="document.getElementById('token-status').textContent=this.value?'●':'○'"
      style="flex:1;background:var(--surface-input);border:1px solid var(--border-input);border-radius:6px;padding:5px 10px;color:var(--text);font-family:var(--mono);font-size:12px;outline:none">
    <span id="token-status" style="color:var(--text-muted);font-size:14px">○</span>
  </div>'''}

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" id="tab-nl" onclick="switchTab('nl')">Natural Language</button>
    <button class="tab" id="tab-gql" onclick="switchTab('gql')">GQL</button>
  </div>

  <!-- Body -->
  <div class="body">

    <!-- Chat panel -->
    <div class="chat-panel">
      <div class="messages" id="messages">
        <div class="msg-row assistant">
          <div class="bubble">
            Hello! Ask me anything about <strong>{model_id}</strong>.<br>
            Switch to GQL tab for direct Glyph Query Language queries.
          </div>
        </div>
      </div>
      <div class="input-bar">
        <!-- NL input -->
        <div class="input-wrap" id="nl-wrap">
          <input id="nl-input" type="text" placeholder="Ask anything about your model..." autocomplete="off" oninput="syncBtn()" onkeydown="nlKey(event)">
          <button class="send-btn" id="nl-btn" disabled onclick="submit()" title="Send (Enter)">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M14 8L2 2l2.5 6L2 14l12-6z" fill="white"/>
            </svg>
          </button>
        </div>
        <!-- GQL input -->
        <div class="input-wrap" id="gql-wrap" style="display:none">
          <textarea id="gql-input" placeholder="LIST ALL LIMIT 5&#10;FIND SIMILAR TO &quot;send message to slack&quot; LIMIT 10" rows="3" oninput="syncBtn()" onkeydown="gqlKey(event)"></textarea>
          <button class="send-btn" id="gql-btn" disabled onclick="submit()" title="Send (Ctrl+Enter)">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M14 8L2 2l2.5 6L2 14l12-6z" fill="white"/>
            </svg>
          </button>
        </div>
        <div class="hint" id="input-hint">Press Enter to send</div>
      </div>
    </div>

    <!-- Results panel -->
    <div class="results-panel">
      <div class="results-header">
        <span class="results-title">Results</span>
        <button class="clear-btn" onclick="clearResults()" title="Clear results">✕</button>
      </div>
      <div class="results-body" id="results-body">
        <div class="results-empty" id="results-empty">
          <div class="icon">◈</div>
          <div class="label">Results will appear here</div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
const ORG_ID = {repr(org_id)};
const MODEL_ID = {repr(model_id)};
const IS_LOCAL = {'true' if deployment_mode == 'local' else 'false'};
let currentTab = 'nl';
let busy = false;
let resultCount = 0;

function switchTab(tab) {{
  currentTab = tab;
  document.getElementById('tab-nl').classList.toggle('active', tab === 'nl');
  document.getElementById('tab-gql').classList.toggle('active', tab === 'gql');
  document.getElementById('nl-wrap').style.display = tab === 'nl' ? 'flex' : 'none';
  document.getElementById('gql-wrap').style.display = tab === 'gql' ? 'flex' : 'none';
  document.getElementById('input-hint').textContent = tab === 'nl' ? 'Press Enter to send' : 'Press Ctrl+Enter to send';
  (tab === 'nl' ? document.getElementById('nl-input') : document.getElementById('gql-input')).focus();
  syncBtn();
}}

function getInput() {{
  const el = document.getElementById(currentTab === 'nl' ? 'nl-input' : 'gql-input');
  return el ? el.value.trim() : '';
}}

function clearInput() {{
  const el = document.getElementById(currentTab === 'nl' ? 'nl-input' : 'gql-input');
  if (el) el.value = '';
  syncBtn();
}}

function syncBtn() {{
  const hasText = getInput().length > 0;
  const nlBtn = document.getElementById('nl-btn');
  const gqlBtn = document.getElementById('gql-btn');
  if (nlBtn) nlBtn.disabled = !hasText || busy;
  if (gqlBtn) gqlBtn.disabled = !hasText || busy;
}}

function nlKey(e) {{
  if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); submit(); }}
}}

function gqlKey(e) {{
  if (e.key === 'Enter' && e.ctrlKey) {{ e.preventDefault(); submit(); }}
}}

function addMessage(role, html, meta) {{
  const msgs = document.getElementById('messages');
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = html;
  row.appendChild(bubble);
  if (meta) {{
    const m = document.createElement('div');
    m.className = 'meta';
    m.innerHTML = meta;
    row.appendChild(m);
  }}
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
  return row;
}}

function addThinking() {{
  const msgs = document.getElementById('messages');
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.innerHTML = '<div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
  return row;
}}

function removeEl(el) {{
  if (el && el.parentNode) el.parentNode.removeChild(el);
}}

function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// Walk the FactTree JSON to find Match nodes inside the "results" intermediate node
function getMatchNodes(ft) {{
  if (!ft || !ft.children) return [];
  for (const child of ft.children) {{
    if (child.description === 'results' && Array.isArray(child.children)) {{
      return child.children.filter(c => c.description && c.description.startsWith('Match'));
    }}
  }}
  return [];
}}

function summarizeResult(ft) {{
  if (!ft) return 'No result';
  // Error embedded in fact_tree
  if (ft.description === 'Query Error') {{
    const errChild = (ft.children || []).find(c => c.description === 'Error Details');
    return '⚠ ' + escHtml(String(errChild?.value || 'Query failed'));
  }}
  // Similarity search matches — render all rows like the CLI
  const matches = getMatchNodes(ft);
  if (matches.length > 0) {{
    return matches.map(m => {{
      const score = m.value?.final_score ?? 0;
      const pct = (score * 100).toFixed(1);
      const filled = Math.round(score * 12);
      const bar = '█'.repeat(filled) + '░'.repeat(12 - filled);
      const name = escHtml(m.value?.concept_text || 'match');
      return `<div class="match-row"><span class="match-pct">${{pct}}%</span><span class="match-bar">[${{bar}}]</span><span class="match-name">${{name}}</span></div>`;
    }}).join('');
  }}
  // No results
  if (ft.description === 'Similarity Search') return 'No matches found';
  // List query
  if (ft.description === 'List Query') {{
    const rNode = (ft.children || []).find(c => c.description === 'results');
    const n = rNode?.children?.length || 0;
    return n + ' glyph' + (n !== 1 ? 's' : '');
  }}
  if (ft.description) return escHtml(ft.description);
  return 'Result received';
}}

function resultLabel(ft) {{
  if (!ft) return 'empty';
  if (ft.description === 'Query Error') return 'error';
  const matches = getMatchNodes(ft);
  if (matches.length > 0) return matches.length + ' result' + (matches.length !== 1 ? 's' : '');
  const rNode = (ft?.children || []).find(c => c.description === 'results');
  if (rNode?.children?.length) return rNode.children.length + ' glyph' + (rNode.children.length !== 1 ? 's' : '');
  return 'view';
}}

function addResult(query, ft, tab, state) {{
  const body = document.getElementById('results-body');
  const empty = document.getElementById('results-empty');
  if (empty) empty.style.display = 'none';
  resultCount++;
  const item = document.createElement('div');
  item.className = 'result-item';
  const count = resultLabel(ft);
  const jsonStr = JSON.stringify(ft, null, 2);
  const stateClass = state ? 'state-badge state-' + state : '';
  const stateBadge = state ? `<span class="${{stateClass}}">${{escHtml(state)}}</span>` : '';
  item.innerHTML = `
    <details>
      <summary>
        <span class="summary-arrow">▶</span>
        ${{stateBadge}}
        <span class="summary-label">${{escHtml(query.substring(0,60))}}${{query.length>60?'…':''}}</span>
        <span class="summary-count">${{count}}</span>
      </summary>
      <div class="result-query-label">${{tab.toUpperCase()}} · #${{resultCount}}</div>
      <pre class="result-pre">${{escHtml(jsonStr)}}</pre>
    </details>`;
  body.insertBefore(item, body.firstChild);
}}

function clearResults() {{
  const body = document.getElementById('results-body');
  const empty = document.getElementById('results-empty');
  const items = body.querySelectorAll('.result-item');
  items.forEach(el => el.remove());
  resultCount = 0;
  if (empty) empty.style.display = 'flex';
}}

async function submit() {{
  const query = getInput();
  if (!query || busy) return;
  busy = true;
  syncBtn();

  addMessage('user', escHtml(query), null);
  clearInput();
  const thinking = addThinking();

  try {{
    const tool = currentTab === 'nl' ? 'nl_query' : 'gql_query';
    const args = currentTab === 'nl'
      ? {{query}}
      : {{query, enable_cache: true}};

    const tokenVal = IS_LOCAL ? '' : (document.getElementById('token-input')?.value || '');
    const headers = {{'Content-Type': 'application/json'}};
    if (tokenVal) headers['Authorization'] = `Bearer ${{tokenVal}}`;

    const res = await fetch(`/${{ORG_ID}}/${{MODEL_ID}}/mcp`, {{
      method: 'POST',
      headers,
      body: JSON.stringify({{tool, arguments: args}})
    }});

    const data = await res.json();
    removeEl(thinking);

    const ft = data.result;
    const isQueryError = ft?.description === 'Query Error';
    // State lives in content[0].data.state for NL, or synthesized for GQL
    const state = data.content?.[0]?.data?.state || (ft ? 'DONE' : null);

    if (data.isError || !res.ok) {{
      const errMsg = data.error || data.detail || 'Unknown error';
      addMessage('error', escHtml(errMsg), null);
    }} else if (isQueryError) {{
      // Error embedded in fact_tree — show as red bubble, no results panel entry
      const errChild = (ft?.children || []).find(c => c.description === 'Error Details');
      addMessage('error', escHtml(String(errChild?.value || 'Query error')), null);
    }} else {{
      const ms = typeof data.query_time_ms === 'number' ? data.query_time_ms.toFixed(1) + 'ms' : null;
      const method = data.match_method;
      const qtype = data.query_type;

      let metaParts = [];
      if (state) metaParts.push(`<span class="pill state-badge state-${{state}}">${{escHtml(state)}}</span>`);
      if (ms) metaParts.push(`<span>${{ms}}</span>`);
      if (method && method !== 'none') metaParts.push(`<span class="pill">${{escHtml(method)}}</span>`);
      if (qtype && qtype !== 'unknown') metaParts.push(`<span class="pill">${{escHtml(qtype)}}</span>`);

      if (state === 'ASK') {{
        const askData = data.content?.[0]?.data?.ask || {{}};
        const question = askData.question || 'Please clarify your request.';
        let html = escHtml(question);
        const options = askData.disambiguation_options || [];
        if (options.length > 0) {{
          html += '<ul style="margin:6px 0 0;padding-left:20px;list-style:none;">';
          for (const opt of options) {{
            const label = opt.suggestion || opt.intent || String(opt);
            html += `<li style="margin:2px 0;">• ${{escHtml(label)}}</li>`;
          }}
          html += '</ul>';
        }}
        addMessage('assistant', html, metaParts.join(''));
      }} else {{
        const summary = summarizeResult(ft);
        addMessage('assistant', summary, metaParts.join(''));
        addResult(query, ft, currentTab, state);
      }}
    }}
  }} catch(err) {{
    removeEl(thinking);
    addMessage('error', 'Network error: ' + escHtml(err.message), null);
  }}

  busy = false;
  syncBtn();
  (currentTab === 'nl' ? document.getElementById('nl-input') : document.getElementById('gql-input')).focus();
}}

// Focus on load
window.addEventListener('load', () => document.getElementById('nl-input').focus());
</script>
</body>
</html>"""

