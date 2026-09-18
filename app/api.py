from __future__ import annotations

import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import db, graph, sops

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Weather Advisory Support Bot")


@app.exception_handler(sops.PolicyError)
def policy_error(_: Request, e: sops.PolicyError) -> JSONResponse:
    """A broken policy file should say which file and why, not a bare 500."""
    return JSONResponse(status_code=503, content={"error": f"policy file error - {e}"})


class Turn(BaseModel):
    session_id: str
    message: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.post("/chat")
def chat(turn: Turn) -> dict:
    return graph.ask(turn.session_id, turn.message)


@app.get("/sops")
def catalogue() -> list[dict]:
    return [
        s.model_dump(include={"id", "category", "severity", "title", "intent", "overrides"})
        for s in sops.load()
    ]


@app.get("/audit")
def audit(limit: int = 50, session_id: str | None = None) -> list[dict]:
    return db.recent(limit, session_id)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "sops": len(sops.load())}
