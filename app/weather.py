"""Open-Meteo client. Every number the bot may state originates here."""
from __future__ import annotations

import datetime as dt
from statistics import mean

import httpx
from pydantic import BaseModel

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10.0

FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "uv_index",
    "visibility",
]
DAILY = ["precipitation_sum", "precipitation_hours", "wind_gusts_10m_max", "uv_index_max"]
WINDOWS = {
    "now": None,
    "morning": (6, 11),
    "afternoon": (12, 16),
    "evening": (17, 21),
    "night": (22, 23),
    "today": (0, 23),
    "tomorrow": (0, 23),
}
# How each field is collapsed across a multi-hour window: worst case, not average.
AGG = {
    "precipitation": max,
    "precipitation_probability": max,
    "wind_speed_10m": max,
    "wind_gusts_10m": max,
    "uv_index": max,
    "visibility": min,
    "temperature_2m": mean,
    "apparent_temperature": mean,
    "relative_humidity_2m": mean,
}


# Human labels for every reading. Deliberately digit-free: the raw field names carry
# digits (wind_gusts_10m), and the grounding guard would read those as claimed figures.
LABELS = {
    "temperature_2m": "temperature",
    "apparent_temperature": "feels-like temperature",
    "relative_humidity_2m": "relative humidity",
    "precipitation": "precipitation",
    "precipitation_probability": "chance of precipitation",
    "wind_speed_10m": "wind speed",
    "wind_gusts_10m": "wind gusts",
    "uv_index": "UV index",
    "visibility": "visibility",
    "precipitation_sum_24h": "rainfall total for the day",
    "precipitation_hours_24h": "hours of rain in the day",
    "wind_gusts_max_24h": "peak wind gusts for the day",
    "uv_index_max_24h": "peak UV index for the day",
}


class WeatherUnavailable(Exception):
    """Raised for every failure mode: geocoding empty, network down, bad response."""


class Snapshot(BaseModel):
    place: str
    latitude: float
    longitude: float
    window: str
    observed_at: str
    values: dict[str, float]
    units: dict[str, str]


def _get(url: str, params: dict) -> dict:
    try:
        r = httpx.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise WeatherUnavailable(f"{url.split('/')[2]} did not respond: {e}") from e


def resolve(city: str) -> tuple[float, float, str]:
    data = _get(GEO_URL, {"name": city, "count": 1, "format": "json"})
    results = data.get("results") or []
    if not results:
        raise WeatherUnavailable(f"no location matches {city!r}")
    top = results[0]
    label = ", ".join(
        str(x) for x in (top.get("name"), top.get("admin1"), top.get("country")) if x
    )
    return top["latitude"], top["longitude"], label


def snapshot(city: str, window: str = "now") -> Snapshot:
    if window not in WINDOWS:
        window = "now"
    lat, lon, label = resolve(city)
    data = _get(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join(FIELDS),
            "hourly": ",".join(FIELDS),
            "daily": ",".join(DAILY),
            "timezone": "auto",
            "forecast_days": 2,
        },
    )
    if "current" not in data or "hourly" not in data:
        raise WeatherUnavailable("forecast response carried no weather values")

    day = 1 if window == "tomorrow" else 0
    values, observed_at = (
        _from_current(data) if window == "now" else _from_hourly(data, window, day)
    )
    values |= _daily(data, day)
    units = {k: v for k, v in data.get("current_units", {}).items() if k in values}
    units |= {
        "precipitation_sum_24h": data.get("daily_units", {}).get("precipitation_sum", "mm"),
        "precipitation_hours_24h": "h",
        "wind_gusts_max_24h": data.get("daily_units", {}).get("wind_gusts_10m_max", "km/h"),
        "uv_index_max_24h": "",
    }
    return Snapshot(
        place=label,
        latitude=lat,
        longitude=lon,
        window=window,
        observed_at=observed_at,
        values={k: round(float(v), 2) for k, v in values.items() if v is not None},
        units=units,
    )


def _from_current(data: dict) -> tuple[dict, str]:
    cur = data["current"]
    return {k: cur.get(k) for k in FIELDS}, cur.get("time", "")


def _from_hourly(data: dict, window: str, day: int) -> tuple[dict, str]:
    h = data["hourly"]
    times = [dt.datetime.fromisoformat(t) for t in h["time"]]
    target = times[0].date() + dt.timedelta(days=day)
    lo, hi = WINDOWS[window]
    idx = [i for i, t in enumerate(times) if t.date() == target and lo <= t.hour <= hi]
    if not idx:
        raise WeatherUnavailable(f"forecast does not cover {window}")
    out = {}
    for f in FIELDS:
        series = [h[f][i] for i in idx if h.get(f) and h[f][i] is not None]
        out[f] = round(AGG[f](series), 2) if series else None
    return out, f"{target.isoformat()} {lo:02d}:00-{hi:02d}:59 local"


def _daily(data: dict, day: int) -> dict:
    d = data.get("daily", {})
    pick = lambda k: (d.get(k) or [None] * (day + 1))[day]
    return {
        k: v
        for k, v in {
            "precipitation_sum_24h": pick("precipitation_sum"),
            "precipitation_hours_24h": pick("precipitation_hours"),
            "wind_gusts_max_24h": pick("wind_gusts_10m_max"),
            "uv_index_max_24h": pick("uv_index_max"),
        }.items()
        if v is not None
    }
