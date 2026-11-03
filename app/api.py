from __future__ import annotations

import logging
import os
import pathlib
import secrets
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import db, graph, sops

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("advisory")
app = FastAPI(title="Weather Advisory Support Bot")


class RateLimiter:
    """Token bucket per client. Every visitor of the public demo shares one free-tier
    model quota, and a turn costs up to three model calls, so one script looping on
    /chat would otherwise spend the day's quota for everyone. In-process only: with
    more than one replica this belongs in the gateway or Redis."""

    def __init__(self, burst: int, per_minute: float, max_clients: int = 10_000):
        self.burst, self.rate, self.max_clients = burst, per_minute / 60, max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """0 if allowed, else seconds until the next request would be."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.pop(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            wait = 0.0 if tokens >= 1 else (1 - tokens) / self.rate
            self._buckets[key] = (tokens - 1 if tokens >= 1 else tokens, now)
            if len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
            return wait


limiter = RateLimiter(
    burst=int(os.getenv("CHAT_BURST", "5")),
    per_minute=float(os.getenv("CHAT_PER_MINUTE", "10")),
)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = rid[:64]
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


@app.exception_handler(sops.PolicyError)
def policy_error(_: Request, e: sops.PolicyError) -> JSONResponse:
    """A broken policy file should say which file and why, not a bare 500."""
    return JSONResponse(status_code=503, content={"error": f"policy file error - {e}"})


@app.exception_handler(Exception)
def unexpected(request: Request, e: Exception) -> JSONResponse:
    """Log the detail, return only an id the operator can grep for."""
    rid = getattr(request.state, "request_id", "")
    log.exception("unhandled error, request_id=%s", rid)
    return JSONResponse(
        status_code=500, content={"error": "internal error", "request_id": rid},
        headers={"x-request-id": rid},
    )


class Turn(BaseModel):
    # crypto.randomUUID() in the page; the pattern keeps arbitrary strings out of the
    # checkpoint thread ids and the audit table.
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # Long enough for any real question; short enough that nobody pastes a novel
    # into three model calls on the shared quota.
    message: str = Field(min_length=1, max_length=500)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


def client_key(request: Request) -> str:
    """Who to rate-limit. Behind Render or any single proxy every request arrives from
    the proxy's address, so one bucket would throttle all visitors together. The
    proxy appends the real peer as the last X-Forwarded-For entry; earlier entries
    are whatever the client claimed, so only the last one is trusted."""
    if forwarded := request.headers.get("x-forwarded-for"):
        return forwarded.rsplit(",", 1)[-1].strip()
    return request.client.host if request.client else "unknown"


@app.post("/chat")
def chat(turn: Turn, request: Request) -> dict:
    if wait := limiter.take(client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="too many questions, slow down a little",
            headers={"retry-after": str(max(1, round(wait)))},
        )
    return graph.ask(turn.session_id, turn.message.strip(), request.state.request_id)


@app.get("/sops")
def catalogue() -> list[dict]:
    return [
        s.model_dump(include={"id", "category", "severity", "title", "intent", "overrides"})
        for s in sops.load()
    ]


@app.get("/audit")
def audit(
    limit: int = Query(50, ge=1, le=500),
    session_id: str | None = None,
    x_audit_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> list[dict]:
    """Holds every visitor's questions, so it is closed unless AUDIT_TOKEN is set and
    supplied as `Authorization: Bearer ...` or `X-Audit-Token`. Unset means closed.
    Not accepted as a query parameter: URLs end up in proxy logs and browser history."""
    expected = os.getenv("AUDIT_TOKEN", "")
    bearer = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
    given = x_audit_token or bearer
    if not expected or not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="the audit log needs AUDIT_TOKEN")
    return db.recent(limit, session_id)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "sops": len(sops.load())}
