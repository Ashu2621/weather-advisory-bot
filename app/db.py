"""Audit trail. Not required by the brief - see README "Why there is a database"."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib

from dotenv import load_dotenv
from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA / 'audit.db'}")
# Hosts hand out postgres:// URLs; SQLAlchemy wants a driver in the scheme and the
# LangGraph checkpointer wants plain libpq. PG_URL is empty unless this is Postgres.
PG_URL = ""
if DB_URL.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
    PG_URL = "postgresql://" + DB_URL.split("://", 1)[1]
    DB_URL = "postgresql+psycopg://" + DB_URL.split("://", 1)[1]
CHECKPOINT_DB = str(DATA / "checkpoints.db")


class Base(DeclarativeBase):
    pass


class Decision(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    request_id: Mapped[str] = mapped_column(String(64), default="")
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
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    model_calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts.isoformat(),
            "session_id": self.session_id,
            "request_id": self.request_id,
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
            "latency_ms": self.latency_ms,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


engine = create_engine(DB_URL, future=True, pool_pre_ping=True)
# Replicas start together and each runs the schema setup; on Postgres two concurrent
# CREATE TABLEs collide in the catalog and one replica crashes. Any fixed number works
# as the lock id as long as nothing else in the database uses it.
SCHEMA_LOCK = 724_100


def add_missing_columns(eng=engine) -> None:
    """create_all never alters an existing table, so an audit.db from before a column
    was added would fail every insert. Additive columns only; anything bigger gets a
    real migration tool."""
    table = Decision.__tablename__
    with eng.begin() as conn:
        have = {c["name"] for c in inspect(conn).get_columns(table)}
        for col in Decision.__table__.columns:
            if col.name in have:
                continue
            fixed = col.default is not None and not col.default.is_callable
            default = f" DEFAULT {col.default.arg!r}" if fixed else ""
            ddl = col.type.compile(eng.dialect)
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col.name} {ddl}{default}"))


def init_schema(eng=engine) -> None:
    with eng.begin() as conn:
        if eng.dialect.name == "postgresql":
            # Transaction-scoped: released at commit, so the next replica only gets
            # the lock once the tables it is about to check for are visible.
            conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": SCHEMA_LOCK})
        Base.metadata.create_all(conn)
    add_missing_columns(eng)


init_schema()


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
