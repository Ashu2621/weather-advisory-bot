"""Stand-ins for the two external services, for load tests and the compose stack.

MODEL_PROVIDER=fake swaps Gemini for a deterministic client with a configurable delay;
WEATHER_FIXTURE=<path> serves a recorded Open-Meteo exchange instead of the network.
Neither is on by default, and the graph, guard, audit and limits all run for real -
only the two remote calls are replaced, so a load test measures this service and not
Google's or Open-Meteo's latency (or quota)."""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import typing
from types import SimpleNamespace as NS


class FakeModel:
    """Answers the three structured calls the way the real model would shape them."""

    def __init__(self, latency_ms: float | None = None, reply: str | None = None, fail=None):
        self.models = self
        self.latency = (latency_ms if latency_ms is not None
                        else float(os.getenv("FAKE_MODEL_LATENCY_MS", "0"))) / 1000
        self.reply = reply or "Conditions call for care today; follow the guidance for your plans."
        self.fail = fail

    def generate_content(self, model, contents, config):
        if self.latency:
            time.sleep(self.latency)
        if self.fail:
            raise self.fail
        from . import llm

        schema = config.response_schema
        parsed = None
        if schema is llm.Parsed:
            # "... in Malabo ..." names a city; a follow-up without one returns None,
            # so the graph has to find the city in session memory, as it does live.
            question = str(contents).rsplit("Current question:", 1)[-1]
            city = re.search(r"\bin ([A-Z][\w-]+)", question)
            parsed = llm.Parsed(location=city.group(1) if city else None, window="today")
        elif schema is not None:
            # The match schema's enum is the eligible SOPs; pick the first, as a
            # matcher that always finds something would.
            ids = typing.get_args(typing.get_args(schema.model_fields["sop_ids"].annotation)[0])
            parsed = schema(sop_ids=list(ids[:1]), reasoning="fake")
        meta = NS(prompt_token_count=100, candidates_token_count=20)
        return NS(parsed=parsed, text=self.reply, usage_metadata=meta)


class _Recorded:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_fixture: dict | None = None


def recorded(url: str) -> _Recorded | None:
    """The recorded response for this URL when WEATHER_FIXTURE is set, else None."""
    global _fixture
    path = os.getenv("WEATHER_FIXTURE")
    if not path:
        return None
    if _fixture is None:
        _fixture = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return _Recorded(_fixture["geocoding"] if "geocoding-api" in url else _fixture["forecast"])
