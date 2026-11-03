"""Prometheus metrics. One process per container, so the default registry is enough;
each replica is scraped on its own and Prometheus sums them."""
from __future__ import annotations

from prometheus_client import Counter, Histogram

TURNS = Counter("advisory_turns_total", "Turns answered, by outcome", ["outcome"])
TURN_LATENCY = Histogram(
    "advisory_turn_latency_seconds", "Wall time of one /chat turn",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
MODEL_CALLS = Counter("advisory_model_calls_total", "Calls made to the model")
MODEL_TOKENS = Counter("advisory_model_tokens_total", "Model tokens", ["direction"])
GUARD_TRIPS = Counter("advisory_guard_trips_total", "Replies replaced because of an invented number")
RATE_LIMITED = Counter("advisory_rate_limited_total", "/chat requests refused with 429")
BUDGET_EXHAUSTED = Counter("advisory_budget_exhausted_total", "Model calls refused by the daily budget")
LIMITER_ERRORS = Counter(
    "advisory_limiter_errors_total", "Redis failures in the limiter or budget (failed open)"
)
