from __future__ import annotations

import logging
import os
import pathlib
import secrets
import uuid

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import db, graph, limits, metrics, sops
from .limits import RateLimiter  # noqa: F401  (re-exported for tests)

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("advisory")
app = FastAPI(title="Weather Advisory Support Bot")


limiter = limits.make_limiter()


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
        metrics.RATE_LIMITED.inc()
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


@app.get("/metrics")
def prometheus(authorization: str | None = Header(default=None)) -> Response:
    """Counts and timings only, no questions or places. Open unless METRICS_TOKEN is
    set, in which case the scraper sends it as a bearer token."""
    expected = os.getenv("METRICS_TOKEN", "")
    if expected and not secrets.compare_digest((authorization or "").encode(), f"Bearer {expected}".encode()):
        raise HTTPException(status_code=401, detail="metrics need METRICS_TOKEN")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
