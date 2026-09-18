"""Policy layer. SOPs are data; nothing here calls a weather API or a model. It knows
only the vocabulary of readings (`weather.LABELS`), so a policy naming one that does
not exist is rejected on load instead of silently never firing."""
from __future__ import annotations

import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from .weather import LABELS

SEVERITY = {"info": 0, "advisory": 1, "warning": 2, "danger": 3}
OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}
SOP_DIR = pathlib.Path(__file__).resolve().parent.parent / "sops"


class Cond(BaseModel):
    field: str
    op: Literal[">=", "<=", ">", "<", "=="]
    value: float

    def holds(self, w: dict) -> bool:
        v = w.get(self.field)
        return v is not None and OPS[self.op](v, self.value)


class SOP(BaseModel):
    id: str
    category: str
    severity: Literal["info", "advisory", "warning", "danger"]
    title: str
    intent: str
    guidance: str
    cite: list[str] = []
    overrides: bool = False
    all_of: list[Cond] = []
    any_of: list[Cond] = []

    @property
    def rank(self) -> int:
        return SEVERITY[self.severity]

    def eligible(self, w: dict) -> bool:
        """True when the live numbers satisfy this SOP's gate. No gate = always true."""
        return all(c.holds(w) for c in self.all_of) and (
            not self.any_of or any(c.holds(w) for c in self.any_of)
        )

    def cited(self, w: dict) -> list[str]:
        """The readings the reply may present. For an `any_of` gate that is only the
        clauses that actually fired - deciding which readings are *responsible* is a
        fact about the numbers, so code decides it, not the model."""
        if not self.any_of:
            return self.cite
        return list(dict.fromkeys(c.field for c in (*self.all_of, *self.any_of) if c.holds(w)))


class PolicyError(ValueError):
    """A policy file that would misbehave silently. Raised on load, never swallowed:
    a rule the team believes is live but which cannot fire is worse than an outage."""


_cache: tuple | None = None


def _parse(f: pathlib.Path) -> list[SOP]:
    try:
        raw = yaml.safe_load(f.read_text("utf-8")) or []
        out = [SOP(**r) for r in raw]
    except (yaml.YAMLError, ValidationError, TypeError) as e:
        raise PolicyError(f"{f.name}: {e}") from e
    for s in out:
        named = {c.field for c in (*s.all_of, *s.any_of)} | set(s.cite)
        if unknown := sorted(named - LABELS.keys()):
            raise PolicyError(
                f"{f.name}: SOP {s.id} names unknown reading(s) {unknown}. "
                f"Known readings: {', '.join(sorted(LABELS))}"
            )
    return out


def load(directory: pathlib.Path = SOP_DIR) -> list[SOP]:
    """Re-reads on any file add/edit, so a new SOP is live without a restart."""
    global _cache
    files = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
    stamp = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    if _cache and _cache[0] == stamp:
        return _cache[1]
    sops = [s for f in files for s in _parse(f)]
    ids = [s.id for s in sops]
    if dupes := {i for i in ids if ids.count(i) > 1}:
        raise PolicyError(f"duplicate SOP ids: {sorted(dupes)}")
    if not sops:
        raise PolicyError(f"no SOPs found in {directory}")
    _cache = (stamp, sops)
    return sops


def by_id(sop_id: str) -> SOP | None:
    return next((s for s in load() if s.id == sop_id), None)


def apply_overrides(eligible: list[SOP], intent_matched: list[SOP]) -> list[SOP]:
    """An `overrides` SOP applies on its numeric gate alone - the matcher does not get
    a vote on it, because "regardless of what was asked" is the whole point of one.

    It is still gated on the question being in-domain at all: if nothing matched on
    intent, the question is not about outdoor activity and an honest no-match beats
    volunteering a rain warning to someone asking about share prices."""
    if not intent_matched:
        return []
    forced = [s for s in eligible if s.overrides]
    seen: dict[str, SOP] = {}
    for s in (*forced, *intent_matched):
        seen.setdefault(s.id, s)
    return list(seen.values())


def rank_matches(matched: list[SOP]) -> tuple[SOP, list[SOP]]:
    """Conflict policy: an `overrides` SOP wins outright, else highest severity.
    Ties break on the order SOPs are declared. Returns (primary, secondary)."""
    ordered = sorted(
        matched, key=lambda s: (not s.overrides, -s.rank, [x.id for x in load()].index(s.id))
    )
    return ordered[0], ordered[1:]
