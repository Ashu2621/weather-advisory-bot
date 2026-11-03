"""Closed-loop load test for /chat: N virtual users, each in its own session, each
sending its next question as soon as the last one is answered.

    python loadtest/run.py --base http://nginx --users 20 --seconds 60

Run it against the compose stack (fake model and recorded weather), so the numbers
are this service's cost, not Google's latency or Open-Meteo's. Rate limits must be
raised for the run (CHAT_BURST / CHAT_PER_MINUTE), or most requests measure a 429."""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
import time
import uuid

import httpx

QUESTIONS = [
    "Is it safe to cycle in Malabo today?",
    "And what about this evening?",
    "Can I take my kids to the park in Malabo?",
    "Is it a good day for a picnic?",
]


async def user(client: httpx.AsyncClient, base: str, stop: float, out: list, outcomes: collections.Counter):
    sid = f"load-{uuid.uuid4().hex[:12]}"
    i = 0
    while time.perf_counter() < stop:
        q = QUESTIONS[i % len(QUESTIONS)]
        i += 1
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{base}/chat", json={"session_id": sid, "message": q})
            outcomes[r.json().get("outcome", r.status_code) if r.status_code == 200 else r.status_code] += 1
        except httpx.HTTPError as e:
            outcomes[type(e).__name__] += 1
        out.append(time.perf_counter() - t0)


def pct(xs: list[float], p: float) -> float:
    return statistics.quantiles(xs, n=100)[int(p) - 1] if len(xs) > 1 else xs[0]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://nginx")
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=60)
    args = ap.parse_args()

    latencies: list[float] = []
    outcomes: collections.Counter = collections.Counter()
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        await client.get(f"{args.base}/health")
        started = time.perf_counter()
        stop = started + args.seconds
        await asyncio.gather(*(user(client, args.base, stop, latencies, outcomes)
                               for _ in range(args.users)))
        elapsed = time.perf_counter() - started

    ms = [x * 1000 for x in latencies]
    report = {
        "users": args.users,
        "seconds": round(elapsed, 1),
        "requests": len(ms),
        "throughput_rps": round(len(ms) / elapsed, 1),
        "p50_ms": round(pct(ms, 50)),
        "p95_ms": round(pct(ms, 95)),
        "p99_ms": round(pct(ms, 99)),
        "max_ms": round(max(ms)),
        "outcomes": dict(outcomes),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
