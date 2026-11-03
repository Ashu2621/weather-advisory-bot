# Load test results

Run on 2026-11-03 with `loadtest/run.py`, 45 s per step, closed loop (each virtual
user sends its next question as soon as the last is answered, each in its own
session, cycling through four questions including follow-ups).

**Setup:** `ops/compose.yml`, so two app replicas (one uvicorn worker each) behind
nginx, with Postgres for session memory and audit and Redis for the rate limit.
Model and weather are faked (`MODEL_PROVIDER=fake`, `WEATHER_FIXTURE`), so this
measures the service and not Gemini or Open-Meteo. Rate limits raised for the run.
Host: HP laptop, 12 cores, 14 GB RAM, Ubuntu, Docker; load generator on the same host.

## Service overhead (fake model answers instantly)

| Users | Requests | Throughput | p50 | p95 | p99 | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3,578 | 79 rps | 136 ms | 217 ms | 288 ms | 0 |
| 50 | 3,216 | 71 rps | 685 ms | 1,405 ms | 1,512 ms | 0 |
| 100 | 3,006 | 66 rps | 1,426 ms | 2,118 ms | 2,420 ms | 0 |

Two replicas saturate at about 70-80 turns/s, roughly 25-30 ms of work per turn per
replica. Past that, extra users only queue. Not profiled yet; the likely main cost
is the graph writing a Postgres checkpoint after every step (six or seven per turn).

## With model latency (300 ms per call, three calls per turn)

| Users | Requests | Throughput | p50 | p95 | p99 | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 480 | 10.5 rps | 949 ms | 972 ms | 1,116 ms | 0 |
| 50 | 2,399 | 52 rps | 939 ms | 1,027 ms | 1,131 ms | 0 |
| 100 | 3,527 | 77 rps | 1,276 ms | 1,609 ms | 1,911 ms | 0 |

With 0.9 s of model time per turn, the service adds about 40 ms at the median up to
50 concurrent users. At 100 it is back at the CPU limit above. In production the
real ceiling is the model quota, long before either of these.

## What would raise the ceiling

* More replicas: the stack check shows memory, limits and audit hold across them.
* Fewer checkpoint writes: LangGraph can persist once at the end of a turn instead
  of after every step, giving up resuming a turn half-way. Profile first.
* An async endpoint and async model client, so a slow model call holds no thread.
  With sync handlers each replica has a 40-thread pool, so about 80 turns can be
  waiting on the model at once across two replicas.
