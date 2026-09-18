"""Audit trail. Not required by the brief - see README "Why there is a database"."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib

from dotenv import load_dotenv
from sqlalchemy import DateTime, Integer, String, Text, create_engine, desc, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA / 'audit.db'}")
CHECKPOINT_DB = str(DATA / "checkpoints.db")


class Base(DeclarativeBase):
    pass


class Decision(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.UTC))
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String(24))
    place: Mapped[str] = mapped_column(String(160), default="")
    window: Mapped[str] = mapped_column(String(16), default="")
    weather: Mapped[str] = mapped_column(Text, default="{}")
    candidates: Mapped[str] = mapped_column(Text, default="[]")
    matched: Mapped[str] = mapped_column(Text, default="[]")
    primary_sop: Mapped[str] = mapped_column(String(48), default="")
    severity: Mapped[str] = mapped_column(String(16), default="")
    guard_tripped: Mapped[str] = mapped_column(Text, default="[]")
    answer: Mapped[str] = mapped_column(Text, default="")

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts.isoformat(),
            "session_id": self.session_id,
            "question": self.question,
            "outcome": self.outcome,
            "place": self.place,
            "window": self.window,
            "weather": json.loads(self.weather),
            "candidates": json.loads(self.candidates),
            "matched": json.loads(self.matched),
            "primary_sop": self.primary_sop,
            "severity": self.severity,
            "guard_tripped": json.loads(self.guard_tripped),
            "answer": self.answer,
        }


engine = create_engine(DB_URL, future=True)
Base.metadata.create_all(engine)


def record(**kw) -> None:
    for k in ("weather", "candidates", "matched", "guard_tripped"):
        if k in kw and not isinstance(kw[k], str):
            kw[k] = json.dumps(kw[k])
    with Session(engine) as s:
        s.add(Decision(**kw))
        s.commit()


def recent(limit: int = 50, session_id: str | None = None) -> list[dict]:
    stmt = select(Decision).order_by(desc(Decision.id)).limit(limit)
    if session_id:
        stmt = stmt.where(Decision.session_id == session_id)
    with Session(engine) as s:
        return [d.as_dict() for d in s.scalars(stmt)]
