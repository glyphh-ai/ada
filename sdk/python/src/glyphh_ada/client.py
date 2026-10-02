"""The client: one endpoint, POST /ada, on a Glyphh API key."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

from .types import (
    Answer, Args, Check, Contract, Device, Edges, Facts, GqlResult, History, Model, Prediction, Procedure, Recorded, Records,
    Situation, Store, Trend, Weights, When,
)

DEFAULT_URL = "https://api.glyphh.ai"
DEFAULT_TIMEOUT = 30.0

#: An argument given as NULL is sent as null. One given as None is not sent.
NULL = object()

#: Sends one request: (url, headers, body, timeout seconds) -> (status, body). Give your own to use another HTTP library.
Transport = Callable[[str, Mapping[str, str], bytes, float], Tuple[int, bytes]]


class AdaError(Exception):
    """Ada refused, or could not be reached. `code` says which kind:
    E_VALIDATION, E_NOT_FOUND, E_FORBIDDEN, E_FAILED_PRECONDITION,
    E_UNAUTHENTICATED, E_PAYMENT, E_UNREACHABLE and others. `status` is the
    HTTP status, 0 when there was none."""

    def __init__(self, code: str, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def __repr__(self) -> str:
        return f"AdaError({self.code!r}, {self.message!r}, status={self.status})"


def _urllib(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> Tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310  # the URL is the caller's Ada endpoint
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Ada:
    """Ada, for one API key.

        ada = Ada()                                   # GLYPHH_API_KEY from the environment
        answer = ada.model("am_0123456789ab").query({"ticket": {"issue": {"component": "kubelet"}}})
        if answer["act"]:
            route(answer["top"])
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, *, timeout: float = DEFAULT_TIMEOUT,
                 headers: Optional[Mapping[str, str]] = None, transport: Optional[Transport] = None) -> None:
        key = api_key or os.environ.get("GLYPHH_API_KEY")
        if not key:
            raise AdaError("E_UNAUTHENTICATED", "an API key is required: pass api_key, or set GLYPHH_API_KEY")
        self._key = key
        self.base_url = (base_url or os.environ.get("GLYPHH_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self._headers = dict(headers or {})
        self._transport = transport or _urllib

    def op(self, op: str, **args: Any) -> Any:
        """One Ada operation by name: the tool's name without ``ada_``, and
        its arguments. An argument that is None is not sent; one that is
        NULL is sent as null. The typed methods call this."""
        body = json.dumps({"op": op, **{k: (None if v is NULL else v) for k, v in args.items() if v is not None}}).encode()
        headers = {"content-type": "application/json", "x-glyphh-api-key": f"Bearer {self._key}", **self._headers}
        try:
            status, raw = self._transport(f"{self.base_url}/ada", headers, body, self.timeout)
        except AdaError:
            raise
        except Exception as e:  # noqa: BLE001  # whatever the transport raises is one thing to a caller: unreachable
            raise AdaError("E_UNREACHABLE", f"Ada is unreachable: {e}") from e
        try:
            answer = json.loads(raw) if raw else None
        except ValueError:
            answer = None
        if 200 <= status < 300 and isinstance(answer, dict) and "data" in answer:
            return answer["data"]
        refused = (answer.get("error") or answer.get("detail")) if isinstance(answer, dict) else None
        said: Dict[str, Any] = refused if isinstance(refused, dict) else {}
        message = refused if isinstance(refused, str) else str(said.get("message") or said.get("error") or "") or f"Ada answered {status}"
        fallback = {401: "E_UNAUTHENTICATED", 402: "E_PAYMENT"}.get(status, "E_ADA")
        raise AdaError(str(said.get("code") or fallback), message, status)

    def model(self, model_id: str) -> "AdaModel":
        """One model, by its id (am_ and 12 hex digits)."""
        return AdaModel(self, model_id)

    def models(self) -> List[Model]:
        """The organization's models."""
        return cast(List[Model], self.op("models")["models"])

    def create_model(self, name: str, storage: str = "cloud", *, spec: Optional[Dict[str, Any]] = None,
                     weights: Optional[Weights] = None, device_id: Optional[str] = None) -> "AdaModel":
        """Create a model (org admins). With a spec it is typed; without, it
        takes any JSON. The model it returns carries what was created in `created`."""
        created = self.op("create_model", name=name, storage=storage, spec=spec, weights=weights, device_id=device_id)
        model = self.model(created["model_id"])
        model.created = cast(Model, created)
        return model

    def devices(self) -> List[Device]:
        """The registered devices that can hold a local model's records."""
        return cast(List[Device], self.op("devices")["devices"])

    def contract(self) -> Contract:
        """The JSON Schema of every answer, and what each reason means."""
        return cast(Contract, self.op("contract"))


class AdaModel:
    """One model. Every method is one call."""

    def __init__(self, ada: Ada, model_id: str) -> None:
        self._ada = ada
        self.id = model_id
        self.created: Optional[Model] = None

    def __repr__(self) -> str:
        return f"AdaModel({self.id!r})"

    def _op(self, op: str, **args: Any) -> Any:
        return self._ada.op(op, model_id=self.id, **args)

    # -- ask --

    def query(self, situation: Situation, *, weights: Optional[Weights] = None) -> Answer:
        """What situations like this one resolve to, and whether to act on
        it. Branch on ``act``; read ``reason`` when it is false."""
        return cast(Answer, self._op("query", situation=situation, weights=weights))

    def facts(self, situation: Situation, *, top: Optional[int] = None, weights: Optional[Weights] = None) -> Facts:
        """Why: the nearest situations among the wins and the failures, each as a fact tree."""
        return cast(Facts, self._op("facts", situation=situation, weights=weights, top=top))

    def check(self, situation: Situation, outcome: str, *, weights: Optional[Weights] = None) -> Check:
        """How close the nearest win and the nearest failure of one outcome are."""
        return cast(Check, self._op("check", situation=situation, outcome=outcome, weights=weights))

    # -- record --

    def record(self, situation: Situation, outcome: str, *, at: Optional[When] = None) -> Recorded:
        """A graded win: this outcome worked here. `at` is when it happened; now when left out."""
        return cast(Recorded, self._op("record", situation=situation, outcome=outcome, at=at))

    def veto(self, situation: Situation, outcome: str, *, at: Optional[When] = None) -> Recorded:
        """A graded failure: this outcome failed, or was rejected, here."""
        return cast(Recorded, self._op("veto", situation=situation, outcome=outcome, at=at))

    # -- one thing, in time --

    def history(self, situation: Situation) -> History:
        """Every record of exactly this situation, oldest first. In a model
        with key parts: every version of the thing, each with what changed."""
        return cast(History, self._op("history", situation=situation))

    def trend(self, thing: Situation) -> Trend:
        """Where one thing has been going. `thing` carries its key parts."""
        return cast(Trend, self._op("trend", situation=thing))

    def predict(self, thing: Situation, *, at: Optional[When] = None) -> Prediction:
        """One thing's next version, and what the model's other things say of it."""
        return cast(Prediction, self._op("predict", situation=thing, at=at))

    def edges(self, thing: Situation, *, level: Optional[str] = None, top: Optional[int] = None) -> Edges:
        """One thing's edges: the things nearest it at a level, the change
        between its versions, and its relations."""
        return cast(Edges, self._op("edges", situation=thing, level=level, top=top))

    # -- GQL --

    def gql(self, query: str, args: Optional[Args] = None) -> GqlResult:
        """One GQL statement, with its $parameters when it takes any."""
        return cast(GqlResult, self._op("gql", query=query, args=args))

    def call(self, name: str, args: Optional[Args] = None) -> GqlResult:
        """Run a stored procedure by name."""
        return cast(GqlResult, self._op("call", name=name, args=args))

    def procedures(self) -> List[Procedure]:
        """The model's stored procedures."""
        return cast(List[Procedure], self._op("procedures")["procedures"])

    def save_procedure(self, name: str, query: str, description: Optional[str] = None) -> Procedure:
        """Save a stored procedure, new or replaced (org admins). The statement is checked now."""
        return cast(Procedure, self._op("save_procedure", name=name, query=query, description=description))

    def delete_procedure(self, name: str) -> Dict[str, Any]:
        """Delete a stored procedure (org admins)."""
        return cast(Dict[str, Any], self._op("delete_procedure", name=name))

    # -- the model and its records --

    def info(self) -> Model:
        """The model with its record counts and cache counts."""
        found = self._op("models")["models"]
        if not found:
            raise AdaError("E_NOT_FOUND", f"no Ada model {self.id}", 404)
        return cast(Model, found[0])

    def update(self, *, name: Optional[str] = None, weights: Optional[Weights] = None, scale: Optional[Dict[str, Any]] = None,
               spec: Optional[Dict[str, Any]] = None, edges: Optional[Dict[str, Any]] = None, clear_weights: bool = False) -> Model:
        """Rename the model, or set its weights, scale, edges or (while it
        has no records) its spec (org admins). `clear_weights` removes its weights."""
        return cast(Model, self._op("update_model", name=name, weights=NULL if clear_weights else weights, scale=scale, spec=spec, edges=edges))

    def delete(self) -> Dict[str, Any]:
        """Delete the model and its records, wherever they live (org admins)."""
        return cast(Dict[str, Any], self._op("delete_model"))

    def records(self, *, store: Optional[Store] = None, offset: Optional[int] = None, limit: Optional[int] = None) -> Records:
        """The model's records, newest first."""
        return cast(Records, self._op("records", store=store, offset=offset, limit=limit))

    def edit_record(self, record_id: str, *, situation: Optional[Situation] = None, outcome: Optional[str] = None) -> Dict[str, Any]:
        """Correct a record (org admins)."""
        return cast(Dict[str, Any], self._op("edit_record", record_id=record_id, situation=situation, outcome=outcome))

    def delete_record(self, record_id: str) -> Dict[str, Any]:
        """Delete one record (org admins)."""
        return cast(Dict[str, Any], self._op("delete_record", record_id=record_id))

    def calibrate(self) -> Dict[str, Any]:
        """Fit the model's probabilities to its own records (10 or more wins)."""
        return cast(Dict[str, Any], self._op("calibrate"))

    def learn(self) -> Dict[str, Any]:
        """Learn what decides from the model's wins (org admins). A flat model learns each top-level
        key's weight, saved as its weights. A typed model learns each role's similarity weight, by
        path, saved in its spec."""
        return cast(Dict[str, Any], self._op("learn"))

    def contract(self) -> Contract:
        """The JSON Schema of every answer and, for a typed model, of its data."""
        return cast(Contract, self._op("contract"))
