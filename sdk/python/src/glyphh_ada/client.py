"""The client: your runtime's address and a runtime key; every call is one request to the runtime's API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

DEFAULT_TIMEOUT = 90.0
WAITING = ("needs_input", "needs_approval")
FINAL = ("done", "failed", "expired", "cancelled")

#: Sends one request: (method, url, headers, body) -> (status, body bytes). Give your own to use another HTTP library.
Transport = Callable[[str, str, Mapping[str, str], Optional[bytes]], Tuple[int, bytes]]


class AdaError(Exception):
    """The runtime refused, or could not be reached. `code` is the runtime's (E_AUTH, E_LIMIT, E_MODEL_MISSING,
    E_RUN_MISSING, E_BAD_REQUEST, ...), or E_UNREACHABLE; `status` is the HTTP status, 0 when there was none."""

    def __init__(self, code: str, message: str, status: int = 0, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.detail = detail or {}

    def __repr__(self) -> str:
        return f"AdaError({self.code!r}, {self.message!r}, status={self.status})"


@dataclass(frozen=True)
class Result:
    """What a run answered: its status, and its data (the function's answer when done; why, when failed)."""

    run_id: str
    model: str
    version: int
    function: str
    status: str
    data: Any
    step: Optional[str]
    prompt: Optional[str]
    seconds: float

    @property
    def done(self) -> bool:
        return self.status == "done"

    @property
    def waiting(self) -> bool:
        """The run asks: `needs_input` or `needs_approval`; `prompt` says what. Answer with `resume`."""
        return self.status in WAITING

    @property
    def ended(self) -> bool:
        return self.status in FINAL

    @classmethod
    def of(cls, body: Mapping[str, Any]) -> "Result":
        return cls(
            run_id=str(body["run_id"]),
            model=str(body["model"]),
            version=int(body["version"]),
            function=str(body["function"]),
            status=str(body["status"]),
            data=body.get("data"),
            step=body.get("step"),
            prompt=body.get("prompt"),
            seconds=float(body.get("seconds", 0)),
        )


def _urllib(method: str, url: str, headers: Mapping[str, str], body: Optional[bytes]) -> Tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:  # noqa: S310  # the caller's runtime
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Ada:
    """Your runtime, for one key.

        incidents = Ada().model("incidents")          # GLYPHH_RUNTIME_URL and GLYPHH_RUNTIME_KEY from the environment
        r = incidents.decide({"error": {...}})
        if r.done: apply(r.data["fix"])
        elif r.waiting: incidents.resume(r.run_id, ask(r.prompt))
    """

    def __init__(
        self, url: Optional[str] = None, key: Optional[str] = None, *, transport: Optional[Transport] = None
    ) -> None:
        self.url = (url or os.environ.get("GLYPHH_RUNTIME_URL") or "").rstrip("/")
        self.key = key or os.environ.get("GLYPHH_RUNTIME_KEY") or ""
        if not self.url or not self.key:
            raise AdaError("E_CONFIG", "the runtime's address and a runtime key are needed: GLYPHH_RUNTIME_URL, GLYPHH_RUNTIME_KEY")
        self._send = transport or _urllib

    def _call(self, method: str, path: str, body: Optional[Mapping[str, Any]] = None) -> Any:
        headers = {"authorization": f"Bearer {self.key}", "accept": "application/json"}
        raw = None
        if body is not None:
            headers["content-type"] = "application/json"
            raw = json.dumps(body).encode()
        try:
            status, answer = self._send(method, f"{self.url}{path}", headers, raw)
        except OSError as e:
            raise AdaError("E_UNREACHABLE", f"the runtime could not be reached: {e}") from e
        try:
            parsed = json.loads(answer) if answer else {}
        except ValueError:
            parsed = {}
        if status >= 400:
            err = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            code = err.get("code") or {401: "E_AUTH", 402: "E_LIMIT", 404: "E_NOT_FOUND"}.get(status, "E_RUNTIME")
            detail = {k: v for k, v in err.items() if k not in ("code", "message")}
            raise AdaError(code, err.get("message") or f"the runtime answered {status}", status, detail)
        return parsed

    # the runtime

    def models(self) -> List[Dict[str, Any]]:
        """The models deployed: each one's versions, the active one and its functions."""
        return list(self._call("GET", "/v1/models")["models"])

    def deploy(self, repo: str, commit: str, token: Optional[str] = None) -> Dict[str, Any]:
        """A commit of a model's GitHub repo (owner/name) as its next version; a token for a private repo."""
        body: Dict[str, Any] = {"repo": repo, "commit": commit}
        if token is not None:
            body["token"] = token
        return dict(self._call("POST", "/v1/models/deploy", body))

    def model(self, name: str) -> "Model":
        return Model(self, name)

    def run(self, run_id: str) -> Dict[str, Any]:
        """A run as the runtime shows it: status, function, times, the last Result."""
        return dict(self._call("GET", f"/v1/runs/{run_id}"))

    def log(self, run_id: str) -> List[Any]:
        return list(self._call("GET", f"/v1/runs/{run_id}/log")["lines"])

    def resume(self, run_id: str, answer: Any) -> Result:
        """Answer a run that waits (`needs_input`, `needs_approval`): the next Result."""
        return Result.of(self._call("POST", f"/v1/runs/{run_id}/continue", {"answer": answer}))

    def cancel(self, run_id: str) -> Result:
        return Result.of(self._call("POST", f"/v1/runs/{run_id}/cancel", {}))


class Model:
    """One model on the runtime. `call` reaches any function; the named ones are an Ada model's own."""

    def __init__(self, ada: Ada, name: str) -> None:
        self.ada, self.name = ada, name

    def call(
        self,
        function: str = "start",
        inputs: Optional[Mapping[str, Any]] = None,
        *,
        version: Optional[int] = None,
        clearance: Optional[float] = None,
    ) -> Result:
        body: Dict[str, Any] = {"model": self.name, "function": function, "inputs": dict(inputs or {})}
        if version is not None:
            body["version"] = version
        if clearance is not None:
            body["clearance"] = clearance
        return Result.of(self.ada._call("POST", "/v1/runs", body))

    def resume(self, run_id: str, answer: Any) -> Result:
        return self.ada.resume(run_id, answer)

    def record(self, data: Mapping[str, Any], outcome: Optional[str] = None) -> Result:
        inputs: Dict[str, Any] = {"data": dict(data)}
        if outcome is not None:
            inputs["outcome"] = outcome
        return self.call("record", inputs)

    def query(self, data: Mapping[str, Any], *, clearance: Optional[float] = None) -> Result:
        return self.call("query", {"data": dict(data)}, clearance=clearance)

    def facts(self, data: Mapping[str, Any], *, top: Optional[int] = None, clearance: Optional[float] = None) -> Result:
        inputs: Dict[str, Any] = {"data": dict(data)}
        if top is not None:
            inputs["top"] = top
        return self.call("facts", inputs, clearance=clearance)

    def probabilities(self, data: Mapping[str, Any], *, clearance: Optional[float] = None) -> Result:
        return self.call("probabilities", {"data": dict(data)}, clearance=clearance)

    def decide(self, data: Mapping[str, Any], *, clearance: Optional[float] = None) -> Result:
        return self.call("decide", {"data": dict(data)}, clearance=clearance)
