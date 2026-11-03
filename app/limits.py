"""Rate limit and daily model budget, shared across replicas when REDIS_URL is set.

In-memory versions are kept for a single process and for tests. Both Redis versions
fail open: if Redis is down the bot keeps answering and the error is counted, because
an outage of the limiter should not become an outage of the bot."""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from collections import OrderedDict

from . import metrics

log = logging.getLogger("advisory")

# One round trip, atomic across replicas: refill by elapsed time, take a token if
# there is one, otherwise return how long until there will be.
_BUCKET = """
local s = redis.call('HMGET', KEYS[1], 't', 'ts')
local burst, rate, now = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local t = tonumber(s[1]) or burst
local ts = tonumber(s[2]) or now
t = math.min(burst, t + math.max(0, now - ts) * rate)
local wait = 0
if t >= 1 then t = t - 1 else wait = (1 - t) / rate end
redis.call('HSET', KEYS[1], 't', t, 'ts', now)
redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) + 60)
return tostring(wait)
"""


class RateLimiter:
    """Token bucket per client, in this process only."""

    def __init__(self, burst: int, per_minute: float, max_clients: int = 10_000):
        self.burst, self.rate, self.max_clients = burst, per_minute / 60, max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """0 if allowed, else seconds until the next request would be."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.pop(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            wait = 0.0 if tokens >= 1 else (1 - tokens) / self.rate
            self._buckets[key] = (tokens - 1 if tokens >= 1 else tokens, now)
            if len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
            return wait


class RedisRateLimiter:
    """Same bucket, kept in Redis so every replica draws from it."""

    def __init__(self, redis, burst: int, per_minute: float, prefix: str = "rl:"):
        self.redis, self.burst, self.rate, self.prefix = redis, burst, per_minute / 60, prefix
        self._script = redis.register_script(_BUCKET)

    def take(self, key: str) -> float:
        try:
            wait = self._script(keys=[self.prefix + key], args=[self.burst, self.rate, time.time()])
            return float(wait)
        except Exception:
            metrics.LIMITER_ERRORS.inc()
            log.warning("rate limiter unavailable, allowing request", exc_info=True)
            return 0.0


class DailyBudget:
    """Model calls allowed per UTC day across the whole deployment. The free tier's
    real limit is a daily cap shared by every visitor; hitting it mid-sentence used
    to be the first sign. With a budget set a little under the cap, the bot says
    plainly that the demo is out of calls for today and when that resets.
    `limit` 0 means no budget."""

    def __init__(self, limit: int, redis=None, clock=None):
        self.limit, self.redis = limit, redis
        self.clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self._local: dict[str, int] = {}
        self._lock = threading.Lock()

    def take(self) -> bool:
        if self.limit <= 0:
            return True
        day = self.clock().strftime("%Y-%m-%d")
        if self.redis is not None:
            try:
                key = f"budget:{day}"
                used = self.redis.incr(key)
                if used == 1:
                    self.redis.expire(key, 2 * 86400)
                return used <= self.limit
            except Exception:
                metrics.LIMITER_ERRORS.inc()
                log.warning("budget store unavailable, allowing model call", exc_info=True)
                return True
        with self._lock:
            self._local = {day: self._local.get(day, 0) + 1}
            return self._local[day] <= self.limit


def _redis():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)


_shared = _redis()


def make_limiter():
    burst = int(os.getenv("CHAT_BURST", "5"))
    per_minute = float(os.getenv("CHAT_PER_MINUTE", "10"))
    if _shared is not None:
        return RedisRateLimiter(_shared, burst, per_minute)
    return RateLimiter(burst, per_minute)


budget = DailyBudget(int(os.getenv("MODEL_DAILY_CALL_BUDGET", "0")), _shared)
