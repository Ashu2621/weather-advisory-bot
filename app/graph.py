"""The LangGraph agent. Three branch points: after parse, fetch and match. Each of
the three can end in the same honest `unavailable` reply."""
from __future__ import annotations

import operator
import sqlite3
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from . import db, llm, sops
from .weather import Snapshot, WeatherUnavailable, snapshot

NO_MATCH = (
    "I don't have a policy covering that, so I won't guess. I can advise on outdoor "
    "exercise, travel, children, the elderly, pets, and casual outings."
)


class State(TypedDict, total=False):
    question: str
    turns: Annotated[list[dict], operator.add]
    location: str | None
    window: str
    activity: str | None
    snapshot: dict | None
    error: str | None
    error_kind: str | None
    candidates: list[str]
    matched: list[str]
    primary: str | None
    secondary: list[str]
    reasoning: str
    guard_tripped: list[float]
    answer: str
    outcome: str


def _history(turns: list[dict]) -> str:
    if not turns:
        return "(no earlier turns)"
    return "Earlier in this conversation:\n" + "\n".join(
        f"- user asked {t['question']!r} about {t.get('place') or 'an unknown place'}"
        f" ({t.get('window', 'now')}); policy applied: {t.get('primary') or 'none'}"
        for t in turns[-6:]
    )


# Everything a single turn computes. The checkpointer persists *all* state on a thread,
# so any of these not reset at the start of a turn silently carries the previous
# turn's value forward - e.g. a stale `primary` routing an off-topic question to
# `compose` with the last answer's citation. Only `turns` is memory.
FRESH_TURN: dict = {
    "location": None, "window": "now", "activity": None, "snapshot": None,
    "error": None, "error_kind": None, "candidates": [], "matched": [],
    "primary": None, "secondary": [], "reasoning": "", "guard_tripped": [],
    "answer": "", "outcome": "",
}


def parse(state: State) -> dict:
    turns = state.get("turns", [])
    last = next((t.get("location") for t in reversed(turns) if t.get("location")), None)
    try:
        p = llm.extract(state["question"], _history(turns), last)
    except llm.ModelUnavailable as e:
        return {**FRESH_TURN, "error": str(e), "error_kind": "model"}
    return {**FRESH_TURN, "location": p.location or last, "window": p.window, "activity": p.activity}


def fetch(state: State) -> dict:
    try:
        snap = snapshot(state["location"], state.get("window", "now"))
        return {"snapshot": snap.model_dump(), "error": None}
    except WeatherUnavailable as e:
        return {"snapshot": None, "error": str(e), "error_kind": "weather"}


def match(state: State) -> dict:
    snap = Snapshot(**state["snapshot"])
    eligible = [s for s in sops.load() if s.eligible(snap.values)]
    try:
        ids, why = llm.match(state["question"], _history(state.get("turns", [])), eligible)
    except llm.ModelUnavailable as e:
        return {"candidates": [s.id for s in eligible], "error": str(e), "error_kind": "model"}
    hits = sops.apply_overrides(eligible, [s for s in eligible if s.id in ids])
    if not hits:
        return {"candidates": [s.id for s in eligible], "matched": [], "reasoning": why}
    primary, secondary = sops.rank_matches(hits)
    return {
        "candidates": [s.id for s in eligible],
        "matched": [s.id for s in hits],
        "primary": primary.id,
        "secondary": [s.id for s in secondary],
        "reasoning": why,
    }


def compose(state: State) -> dict:
    snap = Snapshot(**state["snapshot"])
    primary = sops.by_id(state["primary"])
    extra = [sops.by_id(i) for i in state.get("secondary", [])]
    try:
        answer, bad = llm.compose(state["question"], snap, primary, extra)
        if bad:
            answer = llm.template(snap, primary, extra)
    except llm.ModelUnavailable:
        # The policy is already chosen and the numbers are real, so the deterministic
        # wording is still a correct answer - unlike a failure in parse or match.
        answer, bad = llm.template(snap, primary, extra), set()
    return {"answer": answer, "guard_tripped": sorted(bad), "outcome": "answered"}


def no_match(state: State) -> dict:
    return {"answer": NO_MATCH, "outcome": "no_match"}


def unavailable(state: State) -> dict:
    if state.get("error_kind") == "model":
        text = (
            f"I can't check my policies right now - {state['error']}. I won't guess, "
            "so I have no advice for you until that's back."
        )
    else:
        text = (
            f"I can't get live weather right now - {state['error']}. I won't guess a "
            "forecast, so I have no advice for you until the data is back."
        )
    return {"answer": text, "outcome": "unavailable"}


def need_location(state: State) -> dict:
    return {
        "answer": "Which city should I check? I need a location before I can advise.",
        "outcome": "need_location",
    }


def remember(state: State) -> dict:
    snap = state.get("snapshot") or {}
    return {
        "turns": [
            {
                "question": state["question"],
                "location": state.get("location"),
                "place": snap.get("place"),
                "window": state.get("window", "now"),
                "primary": state.get("primary"),
                "answer": state.get("answer", ""),
            }
        ]
    }


def build(checkpoint_db: str | None = db.CHECKPOINT_DB):
    g = StateGraph(State)
    for fn in (parse, fetch, match, compose, no_match, unavailable, need_location, remember):
        g.add_node(fn.__name__, fn)

    g.add_edge(START, "parse")
    g.add_conditional_edges(
        "parse",
        lambda s: "unavailable" if s.get("error") else "fetch" if s.get("location") else "need_location",
        ["unavailable", "fetch", "need_location"],
    )
    g.add_conditional_edges(
        "fetch", lambda s: "unavailable" if s.get("error") else "match",
        ["unavailable", "match"],
    )
    g.add_conditional_edges(
        "match",
        lambda s: "unavailable" if s.get("error") else "compose" if s.get("primary") else "no_match",
        ["unavailable", "compose", "no_match"],
    )
    for n in ("compose", "no_match", "unavailable", "need_location"):
        g.add_edge(n, "remember")
    g.add_edge("remember", END)

    saver = None
    if checkpoint_db:
        conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
    return g.compile(checkpointer=saver)


_graph = None


def agent():
    global _graph
    if _graph is None:
        _graph = build()
    return _graph


def ask(session_id: str, question: str) -> dict:
    out = agent().invoke(
        {"question": question}, {"configurable": {"thread_id": session_id}}
    )
    snap = out.get("snapshot") or {}
    primary = sops.by_id(out["primary"]) if out.get("primary") else None
    db.record(
        session_id=session_id,
        question=question,
        outcome=out.get("outcome", ""),
        place=snap.get("place", ""),
        window=out.get("window", ""),
        weather=snap.get("values", {}),
        candidates=out.get("candidates", []),
        matched=out.get("matched", []),
        primary_sop=out.get("primary") or "",
        severity=primary.severity if primary else "",
        guard_tripped=out.get("guard_tripped", []),
        answer=out.get("answer", ""),
    )
    return {
        "answer": out.get("answer", ""),
        "outcome": out.get("outcome", ""),
        "citation": (
            {
                "id": primary.id,
                "title": primary.title,
                "category": primary.category,
                "severity": primary.severity,
            }
            if primary
            else None
        ),
        "also": out.get("secondary", []),
        "weather": snap,
        "guard_tripped": out.get("guard_tripped", []),
    }
