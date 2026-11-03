"""Proves the two-replica stack shares what it must. Runs inside the compose network
with the standard library only: python /ops/stack_check.py

  1. A follow-up answered by the *other* replica still knows the city (Postgres memory).
  2. The rate limit is one budget across replicas, not one per replica (Redis).
  3. Both turns are in one audit log, whichever replica wrote them (Postgres audit).
  4. Each replica exposes metrics.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import uuid

BASE = "http://nginx"
BURST = 4


def call(method: str, url: str, body: dict | None = None, headers: dict | None = None):
    req = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body else None,
        headers={"content-type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  - ' + detail if detail else ''}")
    if not ok:
        sys.exit(1)


sid = f"stack-{uuid.uuid4().hex[:8]}"
CHAT = f"{BASE}/chat"
s1, h1, b1 = call("POST", CHAT, {"session_id": sid, "message": "Is it safe to cycle in Malabo today?"})
s2, h2, b2 = call("POST", CHAT, {"session_id": sid, "message": "And what about tomorrow?"})
check("first turn answered", s1 == 200 and json.loads(b1)["outcome"] == "answered", b1[:120])
check("turns landed on different replicas", h1.get("X-Upstream") != h2.get("X-Upstream"),
      f"{h1.get('X-Upstream')} then {h2.get('X-Upstream')}")
check("follow-up on the other replica remembered the city",
      s2 == 200 and json.loads(b2)["outcome"] == "answered", b2[:120])

# Two tokens are spent above. One shared bucket of BURST leaves BURST - 2; per-replica
# buckets would leave 2 * BURST - 2.
statuses = [call("POST", f"{BASE}/chat", {"session_id": sid, "message": "in Malabo?"})[0]
            for _ in range(2 * BURST)]
allowed = statuses.count(200)
check("rate limit is shared across replicas", allowed == BURST - 2,
      f"{allowed} more allowed after 2 turns, burst {BURST}: {statuses}")

s, _, b = call("GET", f"{BASE}/audit?session_id={sid}&limit=50",
               headers={"X-Audit-Token": "stack-audit"})
rows = json.loads(b) if s == 200 else []
check("both replicas write one audit log", s == 200 and len(rows) == 2 + allowed,
      f"{len(rows)} rows")

for replica in ("app1", "app2"):
    s, _, b = call("GET", f"http://{replica}:8000/metrics")
    check(f"{replica} exposes metrics", s == 200 and "advisory_turns_total" in b)

print("stack OK")
