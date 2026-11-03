# Weather-Advisory Support Bot

A LangGraph agent that answers outdoor-activity safety questions from live Open-Meteo
data, and answers **only** from written policy. Every reply either cites a Standard
Operating Procedure or says no policy applies.

**Live demo:** https://weather-advisory-bot-5evh.onrender.com — free tier, so the first
request after 15 idle minutes can take about a minute to wake the service.

**Five minutes, in order:**

1. `pytest evals -m "not live and not remote"` — 27 cases, no key, no network, about 4 seconds.
2. With a key in `.env`: `uvicorn app.api:app`, open http://localhost:8000, ask
   *"is it safe to cycle in Bhopal today?"*
3. The live 11th SOP — [Adding the 11th SOP live](#adding-the-11th-sop-live).
4. Where each non-negotiable is enforced —
   [What is deterministic and what is the model's](#what-is-deterministic-and-what-is-the-models).
5. What broke while building it, and why — [Conflict resolution](#conflict-resolution),
   [Session memory](#session-memory), [Honest gaps](#honest-gaps).

---

## Setup

Python 3.10+ required. From the repo root:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

(Windows PowerShell: `.\.venv\Scripts\Activate.ps1`)

```bash
pip install -r requirements-dev.txt
```

(`requirements.txt` is the runtime set the deploy installs; `-dev` adds pytest and ruff.)

```bash
cp .env.example .env
```

Put a Gemini key in `.env` (free, no card, from https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

## Run

Backend and frontend are one command — FastAPI serves the chat page at `/`:

```bash
uvicorn app.api:app --reload --port 8000
```

Open **http://localhost:8000**. No build step, no npm. Opening `web/index.html` directly
as a file will not work — the page calls the API on the origin that served it — and
the page says so if you try.

### Deployment

The live demo is a Render free web service built straight from this repo:
`pip install -r requirements.txt`, then `uvicorn app.api:app --host 0.0.0.0 --port $PORT`,
with `GEMINI_API_KEY`, `GEMINI_MODEL` and `PYTHON_VERSION=3.13.1` set in the dashboard.
A merge to `main` deploys on its own once CI passes (Render auto-deploy is set to
"After CI Checks Pass"), so a red build never reaches the demo.
Two consequences of the free tier worth knowing: the disk is ephemeral, so the SQLite
audit log and session memory reset on every redeploy; and every visitor shares one
free-tier model quota, so when it runs out the bot says so rather than failing.

Or as a container (`docker build -t advisory . && docker run -p 8000:8000 --env-file .env advisory`).
The image runs as a non-root user and has a healthcheck on `/health`. CI builds it and
checks it boots on every PR.

| Endpoint | Purpose |
|---|---|
| `GET /` | chat UI |
| `POST /chat` | `{session_id, message}` → answer + citation + the readings used. Message ≤ 500 chars; per-client rate limit (429 + `retry-after`) |
| `GET /sops` | the live policy catalogue |
| `GET /audit` | decision trail, newest first. Needs `Authorization: Bearer <AUDIT_TOKEN>` or `X-Audit-Token`; closed when `AUDIT_TOKEN` is unset |
| `GET /health` | liveness + SOP count |

Every response carries an `x-request-id` (yours if you send one), and the same id is on
the turn's audit row and its log line.

### Running it for more than a demo

What a turn is allowed to cost, and what it records:

* **Rate limit** — token bucket per client IP on `/chat` (`CHAT_BURST`, default 5;
  `CHAT_PER_MINUTE`, default 10). Every visitor shares one model quota, and a turn is up
  to three model calls, so one looping script would otherwise spend the day's quota for
  everyone. Behind Render's proxy every request comes from the proxy's address, so the
  limiter keys on the last `X-Forwarded-For` entry, the one the proxy appended; earlier
  entries are whatever the client claimed. With `REDIS_URL` set, the bucket lives in
  Redis and every replica draws from it; without, it is per process.
* **Model timeouts** — each call has a 20 s HTTP timeout (`MODEL_TIMEOUT_MS`), and the
  429/503 retries share a 20 s sleep budget (`MODEL_RETRY_BUDGET_SECONDS`). Gemini's
  `retryDelay` is often a minute; honouring it held the user's request open that long.
  Past the budget the turn ends as `unavailable` and says the model is rate-limited.
* **Cost per turn** — the audit row records latency, the number of model calls and
  input/output tokens, and one JSON log line per turn carries the same numbers. That
  turns "the first thing to change under real load" (gap 5) into something measurable.
* **Bounded memory** — a session keeps its last 20 turns. The checkpointer rewrites the
  whole state on every step, so an uncapped list grows the database quadratically.
* **Geocoding is cached** for a day (512 names), which takes one of the two Open-Meteo
  calls off every follow-up turn. Forecasts are not cached: they are the facts.
* **Schema upgrades** — new audit columns are added in place on startup, so an
  `audit.db` from the previous version keeps working (`test_34`). Anything beyond
  additive columns needs a real migration tool.
* **Daily model budget** — `MODEL_DAILY_CALL_BUDGET` caps model calls per UTC day
  across all replicas (Redis when set). Set a little under the free tier's daily cap,
  the bot says "this demo has used today's model budget; it resets at 00:00 UTC"
  instead of failing mid-turn when Google cuts it off. Unset or 0 means no budget.

### Scaling out

One process with SQLite is the default because it needs nothing installed. For more
than one replica, set two variables and nothing else changes:

| Shared state | Single process | With replicas |
|---|---|---|
| Session memory (LangGraph checkpoints) | SQLite file | Postgres, `DATABASE_URL` |
| Audit log | SQLite file | Postgres, `DATABASE_URL` |
| Rate limit, daily budget | in memory | Redis, `REDIS_URL` |

`ops/compose.yml` runs two replicas behind nginx on shared Postgres and Redis, with
Prometheus, and the model and weather faked so it needs no keys. CI brings it up from
an empty database on every PR and runs `ops/stack_check.py`, which checks that a
follow-up answered by the *other* replica still knows the city, that the rate limit
is one budget across replicas rather than one each, and that both replicas write one
audit log. Bringing it up the first time found a real bug: both replicas created the
tables at once and one crashed, so schema setup now runs under a Postgres advisory
lock. If Redis goes down, the limiter and budget fail open and count the errors;
the bot keeps answering.

```bash
docker compose -f ops/compose.yml up -d --build --wait
docker compose -f ops/compose.yml run --rm check
```

### Load

Two replicas, fake model, on a 12-core laptop: about **70-80 turns/s** with **p95 217 ms
at 10 users**, zero errors at 100 users. With 300 ms per model call (0.9 s of model
time per turn), 50 users get **p95 1.03 s**, so the service adds about 40 ms on top of
the model. Full tables, method and what would raise the ceiling:
[loadtest/RESULTS.md](loadtest/RESULTS.md). The real limit in production is the
model quota, long before either number.

### Metrics and alerts

`GET /metrics` (Prometheus) counts turns by outcome, turn latency, model calls and
tokens, guard trips, 429s, budget refusals and limiter errors. It carries no question
text or place names (`test_37`); `METRICS_TOKEN` closes it. `ops/alerts.yml` has six
rules, checked with `promtool` in CI: more than 20% of turns `unavailable`, p95 above
8 s, guard trips above 10% of answers, budget exhausted, limiter degraded, replica down.

### Load tests and the stack without Google

`MODEL_PROVIDER=fake` swaps Gemini for a deterministic client
(`FAKE_MODEL_LATENCY_MS` adds a delay per call) and `WEATHER_FIXTURE=<path>` serves a
recorded Open-Meteo exchange. Both are off unless set. Everything else (graph, guard,
audit, limits) runs for real (`test_38`).

## Evals

```bash
pytest evals -v
```

Live tests skip automatically when `GEMINI_API_KEY` is unset. The same live cases
also run over HTTP against a deployed instance, using that instance's own key:

```bash
EVAL_BASE_URL=https://weather-advisory-bot-5evh.onrender.com pytest evals -m remote -v
```

`.github/workflows/live-eval.yml` does that every morning against the demo, so the
real model is checked daily without its key ever leaving Render. Results are in
[Eval results](#eval-results) below.

---

## Architecture

```
web/index.html ──POST /chat──► app/api.py ──► app/graph.py (LangGraph)
                                                   │
        ┌──────────────────────────────────────────┤
        ▼                  ▼                       ▼
   app/weather.py     app/sops.py             app/llm.py
   Open-Meteo         sops/*.yaml             Gemini
   (all numbers)      (all advice)            (language only)
                                                   │
                                            app/db.py → SQLite
                                        audit_log + graph checkpoints
```

### The graph

```
                  ┌─────────┐
      START ─────►│  parse  │ city / time window / activity  (model)
                  └────┬────┘
        ┌──────────────┼──────────────────────┐
     no city        has city            model failed
        ▼              ▼                      │
 ┌─────────────┐  ┌─────────┐                 │
 │need_location│  │  fetch  │ Open-Meteo      │
 └──────┬──────┘  └────┬────┘                 │
        │        ok    │    weather failed    │
        │       ┌──────┴───────┐              │
        │       ▼              ▼              ▼
        │  ┌─────────┐   ┌──────────────────────┐
        │  │  match  │──►│     unavailable      │
        │  └────┬────┘   └──────────┬───────────┘
        │ hit   │  none     model failed
        │  ┌────┴─────┐             │
        │  ▼          ▼             │
        │ compose   no_match        │
        │  │          │             │
        └──┴──────────┴─────────────┘
                      ▼
                ┌──────────┐
                │ remember │──► END
                └──────────┘
```

Three branch points, not decoration, and the failure path is reachable from all
three. `unavailable` is one node with two causes — the weather API (or geocoding)
failing, or the model failing — and in both the bot says which, and gives no advice.
`match → no_match` is the honest-refusal path; `parse → need_location` catches a
question with no city in it or anywhere earlier in the session, and `fetch → need_location`
asks which place is meant when the name is ambiguous ("Kochi").

A model failure in `compose` is the one place that does *not* route to `unavailable`:
by then the policy is chosen and the numbers are real, so the deterministic template is
still a correct answer.

`remember` is a join node: all four terminal paths write the turn into session state,
so a follow-up after a *failed* turn still knows what was asked.

---

## How policies are represented

**SOPs are YAML files in `sops/`, one file per category, loaded as data at request
time.** Chosen because the brief requires adding a policy with no code change and no
restart — a YAML file is the smallest thing a non-engineer can edit, diff and review,
and `app/sops.py:load()` re-reads whenever any file's mtime changes.

```yaml
- id: EX-GUST-TWOWHEEL
  category: outdoor_exercise
  severity: warning                 # info | advisory | warning | danger
  title: Gusts affecting cycling and two-wheelers
  intent: >-                        # what the *question* must be about (semantic gate)
    Cycling, biking, riding a scooter or motorcycle...
  all_of:                           # what the *weather* must be (numeric gate)
    - {field: wind_gusts_10m, op: ">", value: 40}
  guidance: >-                      # the advice. The model may not add to this.
    Treat this as a safety risk, not a comfort issue...
  cite: [wind_gusts_10m, wind_speed_10m]   # readings quoted back to the user
```

14 SOPs across 5 categories (`outdoor_exercise`, `travel`, `vulnerable_groups`,
`leisure`, `situational`), spanning all four severities.

### No gap between "calm" and "hazard"

Threshold rules leave holes wherever one rule's gate stops and the next one's starts.
The live eval found one: Bhopal had gusts of **27 km/h** — too windy for
`EX-FAVOURABLE` (gusts < 25) and nowhere near `EX-GUST-TWOWHEEL` (> 40) — so *"is it
safe to cycle to work?"* got *"I don't have a policy covering that"* on an ordinary
breezy morning. The same hole existed for UV 6–8, rain chance 30–70 %, and feels-like
temperatures above 32 °C or below 5 °C.

A no-match there is technically honest and practically useless, so the fix is
structural rather than another threshold: `EX-CAUTION-BAND` has a gate that is the
**exact logical complement** of `EX-FAVOURABLE`'s — one needs every reading in range,
the other fires when any reading is out of it — so every exercise question is covered
by exactly one of the two, plus any hazard rule, which then outranks it on severity.
`test_25` proves the property over a 2,268-point grid straddling every threshold,
rather than trusting it. The same Bhopal reading now answers: *"Outdoor exercise is
reasonable in Bhopal right now, but wind gusts of 27.0 km/h make headwind stretches
harder work and make bridges, flyovers and junctions less stable for cyclists."*

The other categories are not tiled this way: travel, vulnerable-group and leisure
questions in unremarkable weather can still get an honest no-match. That is deliberate
for now — there is no "all clear" policy for, say, walking a dog on a mild day — but it
is the next gap a policy owner should close.

### Adding the 11th SOP live

Drop a file into `sops/`, or append to an existing one. Nothing else. No restart, no
reload endpoint, no code touched — the next message picks it up, and the enum the
model chooses from is rebuilt from the files on every request.
`evals/test_evals.py::test_05_new_sop_goes_live_without_touching_code` proves it.

A mistyped policy fails loudly instead of loading and never firing. Name a reading
that does not exist and every request returns, until it is fixed:

```
policy file error - zz_demo.yaml: SOP ZZ-LIVE names unknown reading(s) ['wind_speed'].
Known readings: apparent_temperature, precipitation, ..., wind_speed_10m
```

Silently-dead rules are the worst failure a safety policy can have — everyone believes
it is live — so this deliberately takes the bot down rather than serve without it.
`test_23`.

**Where this design does not pass the "no code change" test, stated plainly:** an SOP
can gate on any of the 13 readings the fetcher already pulls, in any combination, with
no code change. A rule that needs a reading *not* yet fetched — snowfall, cloud cover,
dew point, air quality — needs one edit to `app/weather.py`: add the field to `FIELDS`
and a label to `LABELS`. That is the code that fetches weather, which the brief says
should not need touching. The alternative, fetching every field Open-Meteo offers on
every request, trades a one-line change for a much larger response on every turn; I
chose the one-line change and am flagging it rather than hiding it.

### The fuzzy SOP

`LE-PLEASANTNESS` (leisure) has **no `all_of` or `any_of` block at all**. It is gated
purely on intent, and its guidance asks for a qualitative verdict — pleasant,
acceptable with caveats, unpleasant — justified from whichever readings decide it,
with an explicit statement that this is comfort and not safety. That is the honest
shape for "is today good for a picnic": there is no threshold, so the policy does not
pretend there is one. It still constrains the model tightly — the *reasoning style*
is the policy even where a number isn't.

### Conflict resolution

Two steps, both in `app/sops.py`, both deterministic.

**First, `apply_overrides()` decides who is even in the running.** An SOP marked
`overrides: true` applies on its numeric gate alone — the intent matcher gets no vote
on it, because "regardless of what was asked" is the entire point of a situational
policy. It is still gated on the question being in-domain at all: if nothing matched
on intent, the answer stays an honest no-match rather than volunteering a rain warning
to someone asking about share prices.

> This was a bug first, and it is worth stating plainly because it is exactly the case
> the brief singles out. The override was originally offered to the matcher like any
> other SOP, and during a live run the matcher picked `EX-UV-PEAK` for a bike-ride
> question while a rain system was active and simply left the situational SOP out — so
> the policy that was supposed to outrank everything never fired at all. Routing it
> around the model is the fix. `test_02b` pins it.

**Then `rank_matches()` orders whoever is in the running:**

1. An SOP with `overrides: true` wins outright and is stated first.
2. Otherwise the highest severity wins.
3. Ties break on declaration order in the YAML.

Every other match is surfaced as a one-line "Also applies" rather than dropped —
suppressing a live gust warning because a heat warning outranked it would be the
wrong failure. **Why override beats severity:** the brief's hardest case is a rain
system whose individual numbers look unremarkable while the situational risk is high.
Severity ranking alone cannot express "this changes the frame of the whole question",
so `overrides` is a separate axis. `test_02` pins this: an *advisory* override beats a
*danger* non-override, which is the counter-intuitive case, and the danger still shows.

---

## What is deterministic and what is the model's

This split is the whole design.

| Decision | Made by | Where |
|---|---|---|
| Which city, when, what activity | model | `llm.extract` |
| What the weather is | Open-Meteo, never the model | `weather.snapshot` |
| Whether an SOP's numeric gate holds | code | `sops.SOP.eligible` |
| Which eligible SOP fits the question | model, enum-constrained | `llm.match` |
| Whether a situational SOP applies anyway | code — the model gets no vote | `sops.apply_overrides` |
| Which match wins | code | `sops.rank_matches` |
| The advice itself | YAML `guidance` | `sops/*.yaml` |
| Wording of the reply | model | `llm.compose` |
| Whether the wording is allowed to stand | code | `llm.bad_numbers` |

The model never sees a threshold, never compares a number, and never chooses an id
that isn't in the catalogue.

### The model cannot invent a policy

`app/llm.py:116` builds the response schema at request time from the SOPs that passed
their numeric gate:

```python
ids = tuple(s.id for s in candidates)
schema = create_model("Match", sop_ids=(List[Literal[ids]], ...), reasoning=(str, ...))
```

A fabricated id is not a wrong answer the prompt discourages — it is unrepresentable
in the response schema. An empty list is a first-class outcome routing to `no_match`.

### The model cannot invent a number

`app/llm.py:151`, `allowed_numbers()`, builds the permitted set for a turn: the live
readings (at 0 and 1 decimal places) plus whatever figures already appear in the cited
policy text and the location/time labels. `bad_numbers()` subtracts that set from every
numeric token in the drafted reply. `app/graph.py:106` throws the draft away and falls
back to `llm.template()` — a deterministic string built from the snapshot — whenever
anything is left over. The audit row records `guard_tripped` and the UI shows a banner,
so a suppressed hallucination is visible rather than silent.

Two deliberate choices here:

* **The user's question is excluded from the allowed set.** A figure the user typed is
  not a fact, and admitting it would let someone seed a number ("it's 45 °C here,
  right?") and have it echoed back as though the API had reported it.
* **The eval calls the same `allowed_numbers()` the bot does.** An earlier version
  recomputed the allowed set inside the test, the two definitions drifted, and a case
  failed on a number the bot had legitimately permitted. One source of truth, or the
  suite is testing something other than the guarantee.

### The model cannot answer without data

The `fetch → unavailable` edge is taken before `match` is ever reached, so on an
outage there is no snapshot, no candidate list and no compose call. Geocoding failure
and API failure raise the same `WeatherUnavailable`, so both take that one branch —
`test_15` and `test_16` cover them separately.

The model is treated as a dependency that fails too. Every failure in `app/llm.py` —
no key, a spent daily quota, a network error, a response that breaks the schema — is
raised as `ModelUnavailable`, which the graph routes to the same node. Only that type
is caught: a programming error still surfaces as an error rather than being dressed up
as an outage. Before this, a missing key or spent quota returned a bare HTTP 500 —
found by running a fresh clone without a key, the way a reviewer would. `test_24`.

---

## Session memory

LangGraph `SqliteSaver`, keyed on `thread_id` = the browser's session id. State carries
a `turns` list (question, resolved place, window, SOP applied) rather than raw message
history, because that is all a follow-up needs and it keeps the prompt small and
auditable. "What about this evening instead?" reuses the prior city and switches the
window — `test_12`. "New session" in the UI mints a fresh id, which is the reset the
brief asks for.

**Memory is `turns` and nothing else.** The checkpointer persists *every* state key on
a thread, not just the ones meant as memory, so any key a turn does not overwrite
silently carries into the next. That was a real bug here, found in the line-by-line
review before submission: after an answered question, a follow-up the policies do not
cover ("will the stock market go up?") left the previous turn's `primary` in state, the
router saw it, and the bot answered the share-price question with a heavy-rain
citation — breaking two non-negotiables at once, and only in multi-turn sessions, which
is why no single-turn case caught it. `parse`, the entry node, now resets every
per-turn key from `FRESH_TURN` in `app/graph.py`. `test_22` replays exactly that
sequence — answered, off-topic, outage — in one session, fully offline.

---

## Why there is a database

**The brief does not ask for one** — it says memory resets between sessions and warns
against sprawl. It was added anyway, for one reason: the brief frames this as a
liability problem ("we have to stand behind whatever this bot tells a user", "we should
always be able to ask why did it say that"). A citation in a chat bubble answers that
for the user; it does not answer it for the business three weeks later.

`audit_log` stores, per turn: the question, the resolved location, **the exact weather
values the decision used**, every candidate SOP, every match, the primary, the severity,
whether the number guard tripped, the final text, and what the turn cost (latency,
model calls, tokens, request id). `GET /audit` reads it back. That
makes "why did it say that" reconstructable after the fact, including what the weather
was at the time — which no live re-query can recover.

Because it holds every user's questions, `/audit` is **closed unless `AUDIT_TOKEN` is
set and supplied**, and an unset variable means closed, not open. The first live
deployment got this wrong: `/audit` was world-readable on the public URL, found by
checking the deployment rather than trusting it. The token is accepted in a header only;
the first version also took `?token=`, which puts the secret in proxy logs and browser
history. `test_27`.

SQLite so there is nothing to install; SQLAlchemy so `DATABASE_URL` can point at
Postgres without a code change (it needs a driver such as `psycopg[binary]` added to
`requirements.txt`). **SOPs are deliberately *not* in the database** — a reviewer
adding a policy should drop a file, not write SQL.

---

## Eval results

Cases 01–27: run on **2026-09-18**, `gemini-3.5-flash-lite`, live Open-Meteo.
**28 passed, 0 failed, 0 skipped, in 3 m 13 s** (`EVAL_PACE_SECONDS=4`; most of that is
free-tier rate-limit waits). Cases 28–38 were added on 2026-11-03 and are all offline;
the 27 offline cases pass in CI on every PR.

**Remote run, 2026-11-03**, against the deployed demo after those changes, real Gemini:
**9 passed, 0 failed** in 2 m 48 s (`test_remote.py`: 08–12, 14, 16, 17, and the
two-turn "which Kochi?" case). Case 13 (severe live weather) and the outage cases
15/18 stay in-process only, since they need the network stubbed. Several cases exist
because something failed first — 02b, 08, 22, 24, 26, 27, 28 and 29 — and each is
marked in the table with what it found.

| # | Case | What it checks | Pass looks like | Result |
|---|---|---|---|---|
| 01 | Numeric gate | Eligibility at/above/below threshold, and on a missing reading | UV 8.0/9.1 select, 7.9 and `{}` do not | ✅ pass |
| 02 | Conflict: override | Declared conflict policy | Advisory override beats danger; danger kept as secondary | ✅ pass |
| 02b | **Override bypasses the matcher** | The brief's hardest case | An eligible override is forced in when the matcher omits it, and becomes primary; nothing is forced when the question is out of domain | ✅ pass |
| 03 | Conflict: severity | Ranking without an override | Danger beats info | ✅ pass |
| 04 | Fuzzy SOP | Non-numeric policy is intent-gated only | Eligible at -5 °C and 45 °C | ✅ pass |
| 05 | **Hot-add SOP** | The review-call requirement | New YAML file is live, no restart, no code change | ✅ pass |
| 06 | Number guard | Non-negotiable #5 | 47 accepted, 85 rejected, policy-text numbers accepted | ✅ pass |
| 07 | Duplicate ids | Bad policy edit fails loudly | `ValueError` on load | ✅ pass |
| 08 | Clear match — cycling | Obvious in-policy question | Cited SOP's own gate holds; zero ungrounded numbers | ✅ pass — **failed once** on a 27 km/h breeze, see *No gap between "calm" and "hazard"* |
| 09 | Clear match — child outing | Second category | as above | ✅ pass |
| 10 | **Paraphrase** — "legs itching for a long spin" | Semantic, not keyword, matching | Policy cited though the question shares no SOP vocabulary | ✅ pass |
| 11 | **Paraphrase** — "Gran on the balcony with the baby" | Same, second category | as above | ✅ pass |
| 12 | Session memory | Follow-up builds on context | "this evening instead" keeps the city, switches the window | ✅ pass |
| 13 | **Severe live weather** | Grounded in real numbers | SIT-* cited and the real 24 h total appears in the reply | ✅ pass |
| 14 | No policy applies | Honest refusal | `outcome=no_match`, no citation, no advice | ✅ pass |
| 15 | API unreachable | Non-negotiable #3 | `outcome=unavailable`, no citation, no numbers | ✅ pass |
| 16 | Unresolvable location | Same failure path as an outage | `outcome=unavailable` | ✅ pass |
| 17 | **Adversarial — prompt injection** | Fabricated-policy attack | `SOP-999` never printed or cited | ✅ pass |
| 18 | Adversarial — "offline demo mode" | Failure branch cannot be argued away | Still `unavailable`, still no numbers | ✅ pass |
| 19 | **Recorded severe weather** (offline) | The severe guarantee with no network, model or season | Heavy gate holds on the recording, override is primary, real total reaches the text | ✅ pass |
| 20 | Fixture integrity | A fixture that drifts makes every assertion on it meaningless | The real capture is unedited; the derived one differs in exactly one field and says so | ✅ pass |
| 21 | **Recorded severe weather** (full graph) | Case 13 with the seasonality removed | `SIT-HEAVY-RAIN-SYSTEM` primary, `118.4` in the reply, nothing ungrounded | ✅ pass |
| 22 | **State does not leak between turns** | A previous turn's verdict never answers the next question | Answered → off-topic → outage in one session gives `no_match` then `unavailable`, neither with turn 1's citation | ✅ pass — **failed first**, see *Session memory* |
| 23 | Mistyped policy | The live-11th-SOP risk: a typo'd field used to load and never fire | Load fails naming the file, SOP, bad field and the valid ones | ✅ pass |
| 24 | **Model outage** | The model fails too: no key, spent quota, outage | `unavailable`, no citation, says what is wrong — not an HTTP 500 | ✅ pass — **failed first** (was a 500) |
| 25 | **No policy gap for exercise** | Threshold rules leave holes between gates | Over 2,268 readings straddling every threshold, exactly one of `EX-FAVOURABLE` / `EX-CAUTION-BAND` applies | ✅ pass |
| 26 | **Only readings that fired reach the reply** | An in-range reading presented as a concern — real number, so the guard cannot see it | Fact sheet and fallback carry exactly the clauses that fired; in-range gusts absent | ✅ pass — found live, see *Honest gaps* #4 |
| 27 | **Audit log is not public** | The first deployment served every visitor's questions at `/audit` | Closed with no token configured, closed with a wrong one or one in the URL, open with the right header | ✅ pass — found on the live deployment |
| 28 | **Number guard reads signs and commas** | A correct sub-zero reply was flagged as invented | `-12.4` and `24,140` grounded; a flipped sign caught; dates and time ranges not read as negatives | ✅ pass — **failed first**, see *Honest gaps* #4 |
| 29 | **Ambiguous place names** | "Kochi" answered for Japan without asking | Comparable places in different countries get a question with choices; London and "Kochi, India" answer directly | ✅ pass — closes *Honest gaps* #3 |
| 30 | Bounded session memory | The checkpoint grew with every turn | 25 turns leave the last 20 | ✅ pass |
| 31 | `/chat` input and rate limit | Public endpoint on a shared quota | 501-char message and bad session id are 422 before any model call; burst exhausted gives 429 + `retry-after` | ✅ pass |
| 32 | **Cost and trace per turn** | Real model wrapper with a fake client, no key | Audit row has 3 model calls, their tokens, latency and the echoed request id | ✅ pass |
| 33 | Retry budget | A 60 s `retryDelay` held the request open | No sleep past the budget; turn ends `unavailable` | ✅ pass |
| 34 | Old audit database | `create_all` never adds columns | New columns added in place, old rows kept | ✅ pass |
| 35 | **Shared rate limit** | Per-process buckets meant N replicas, N times the limit | Two limiters on one Redis share one burst; Redis down fails open and is counted | ✅ pass |
| 36 | Daily model budget | Free-tier cap hit mid-turn | Past the budget: `unavailable`, says when it resets, no call sent; new UTC day resets | ✅ pass |
| 37 | Metrics | No way to see health without reading rows | Turn, latency, calls, tokens counted; no question text; token closes it | ✅ pass |
| 38 | Fake mode | Load tests must not measure Google | Full graph answered with zero network calls | ✅ pass |

### Case 13 in full — the situational override on live data

Case 13 scans a global list for somewhere currently under a rain system, then asks a
question in a *different* category and checks the situational SOP still leads. On this
run it found Bangkok. Taken verbatim from the `audit_log` row:

```
question   : Is it safe to go for a bike ride in Bangkok today?
place      : Bangkok, Bangkok, Thailand      window: today
candidates : LE-PLEASANTNESS, EX-UV-PEAK, SIT-RAIN-SPELL, TR-RAIN-DELAY
primary    : SIT-RAIN-SPELL (advisory)       also: EX-UV-PEAK (warning)
readings   : precipitation_sum_24h 16.0 mm · precipitation_hours_24h 14.0 h
             precipitation_probability 95.0 % · precipitation 8.3 mm
guard      : [] (no invented numbers)

answer     : You will see rain for 14.0 h today with a total of 16.0 mm, which is
             below the heavy-rainfall category, so use a rain plan rather than
             cancelling your bike ride. Avoid unprotected outdoor exertion between
             11:00 and 16:00 due to high UV, and instead wear broad-spectrum
             sunscreen, covered arms and a hat, or reschedule your session to before
             09:00 or after 17:00. Note that UV burns skin on overcast days too.
```

Four things are visible in that one row. The rain system leads, before any
bike-specific advice. An **advisory** override outranks a **warning** — severity alone
would have got this wrong. The warning is not silently dropped; it follows as the
secondary. And `16.0` and `14.0` are the API's own figures, with the guard empty.

Note this is also the case that exposed the override bug described under *Conflict
resolution*: the matcher returned only `EX-UV-PEAK` here, and before the fix the rain
system would not have been mentioned at all.

### The recorded-fixture layer — how case 13 stops being seasonal

Case 13 is still weather-dependent. An earlier run of this suite skipped it outright,
because the scan list was India-only and the wettest place on it was Chennai at 13.3 mm
against a 15.6 mm gate. Widening the scan globally makes a hit likelier, never certain,
and the heavy tier is rarer still: **a 76-location global scan on the build date found
no 24-hour total at or above the 64.5 mm heavy gate anywhere on earth.** A suite whose
most important case depends on that is not a suite.

So the guarantee is asserted twice, and only one of the two can skip:

| | Case 19 / 21 — recorded | Case 13 — live |
|---|---|---|
| Weather source | fixture in `evals/fixtures/` | whatever the API says now |
| Can skip | never | yes, with a reason |
| Proves | the severe path works | the recording still matches reality |

`evals/fixtures/` holds two files, and the distinction between them matters:

* **`rain_spell_real.json`** — a verbatim Open-Meteo exchange for Malabo, Equatorial
  Guinea, the wettest place in that global scan at **60.6 mm**. Not one byte edited.
* **`heavy_rain_derived.json`** — built from that same capture by changing **exactly one
  value**, `daily.precipitation_sum[0]` from 60.6 to 118.4 mm, to reach the IMD "very
  heavy" band. The file's own `_note` says so in full, and **case 20 fails the build if
  the two files ever differ by more than that one field**, or if the labels are removed.

This is a test fixture, not a shortcut in the product: no event and no number is
hardcoded anywhere in `app/`, and the bot still reads whatever the API layer hands it.
Case 21's output, from the `audit_log`:

```
question   : Is it safe to go for a bike ride in Malabo today?
candidates : LE-PLEASANTNESS, EX-UV-PEAK, SIT-HEAVY-RAIN-SYSTEM,
             TR-RAIN-DELAY, TR-LOW-VISIBILITY
primary    : SIT-HEAVY-RAIN-SYSTEM (danger)   also: EX-UV-PEAK (warning)
guard      : []

answer     : An active heavy-rainfall system is over Malabo today, bringing 118.4 mm
             of rain that meets the IMD heavy-rainfall category. Avoid outdoor
             activities due to waterlogging, poor visibility, fallen branches, and
             live wiring risks, defer non-essential outdoor plans, and check the IMD
             district bulletin before going out. Avoid unprotected outdoor exertion
             between 11:00 and 16:00 …
```

Five SOPs were eligible; the situational one led regardless of the cycling question,
and the UV warning still followed. That runs on any day, in 7 seconds.

### Why the adversarial case is prompt injection

The user's question flows verbatim into two LLM calls. The highest-value attack is not
making the bot rude, it is making it **cite a policy that does not exist** — a
fabricated citation is indistinguishable from a real one to the user, and it defeats
the entire traceability guarantee the product is sold on. `test_17` attacks the matcher
directly; `test_18` attacks the failure branch, which is the other place a confident
wrong answer would be most damaging.

### Honest gaps

1. **Live weather does not sit still for a test suite** — the brief's closing question.
   Three layers answer it, and the honest part is that only the third can ever skip:

   * **Injected readings** (cases 01–07, 19): exact threshold behaviour, asserted
     against numbers the test chooses. Never drifts, never touches the network.
   * **Recorded fixtures** (19, 20, 21): a real Open-Meteo response replayed through
     the whole graph, so the severe path is proven on a calm day. Case 20 guards the
     fixtures themselves against silent drift.
   * **Live properties** (08–12, 14–18): assertions that hold *whatever* the API
     returns today — "the cited SOP's own gate holds against the numbers returned",
     "no number in the reply is ungrounded" — rather than "the answer mentions rain".

   Only case 13 depends on real severe weather existing, and it now duplicates case 21
   rather than being the sole cover for it, so a skip costs no coverage. What a
   production suite should add that this one does not have: a *library* of recorded
   responses — a cyclone, a heatwave, a fog event, a cold snap — captured as they
   occur, so every situational SOP gets an unconditional end-to-end case instead of
   just the rain ones.

2. **`SIT-HEAVY-RAIN-SYSTEM` is a proxy, not the real signal.** The brief's example is
   an IMD-flagged well-marked low-pressure area. Open-Meteo does not publish that flag,
   so the SOP keys on the IMD 24-hour rainfall categories (heavy ≥ 64.5 mm) plus
   rain-hours, which is the closest defensible thing in the available data. A real
   deployment needs the IMD district bulletin as a second source; the `overrides`
   mechanism is already the right shape to carry it, and would need no code change
   beyond a new field in `weather.snapshot`.

3. **Geocoding used to take the first result silently** — now closed, mostly.
   **"Kochi"** resolved to **Kochi, Japan**, not Kochi, Kerala; **"Sohra"** — the
   current name of Cherrapunji — to **Sohra, Saxony, Germany**. The resolver now asks
   when a same-named place in *another country* is at least 30 % the size of the top
   hit (`weather.rivals`), and the page shows the choices as buttons. The threshold is
   what stops it pestering: London, Ontario is 5 % of London, England, so "London"
   still just answers. A name with no population data on either side always asks.
   What is still open: same-country ambiguity (five Springfields in the US) is not
   asked about, and the 30 % is a judgement call, not something I measured.

4. **The number guard only sees digits.** Two known ways past it, one closed and one
   open:
   * *Closed:* the model first tried spelling figures out ("twenty-three point two
     degrees"), which sails through a digit regex. The compose prompt now requires
     digits copied from FACTS, and the fallback template is digits by construction.
     This is a prompt-level mitigation, not enforcement — a determined model could
     still spell a number out.
   * *Narrowed, not closed:* a *qualitative* fabrication. Manual testing found a
     concrete one: asked about a run in Atarra, the caution SOP fired on rain chance
     (31 %) and feels-like temperature (32.6 °C), and the model *also* warned that
     "wind gusts of 21.2 km/h make headwind stretches harder work … for cyclists".
     21.2 is a real reading, so the number guard passed it — but it is under the
     25 km/h threshold, so the policy says it is not a concern, and the user was
     running, not cycling. The fix is not a stronger prompt: `SOP.cited()` in
     `app/sops.py` computes which `any_of` clauses actually fired, and only those
     readings reach the fact sheet, so the model never sees the in-range value to
     misuse. `test_26`. What remains open is a claim with no reading behind it at
     all ("winds are picking up sharply"); only the guidance text constrains that.

   Two false positives are now fixed and worth naming, because they are the kind of
   bug that makes a guard useless. The model was quoting raw column names, and
   `wind_gusts_10m` made the guard read a claim of "10"; the fact sheet now uses
   digit-free labels and `_nums` strips field identifiers before matching. And the
   regex dropped minus signs, so in any city below zero a correct "feels like -12.4"
   was rejected as an invented 12.4 and every answer fell back to the template; it also
   read "24,140 m" as 24 and 140. Signs and thousands separators are now part of the
   number, a flipped sign is still caught, and hyphens in dates are not signs. `test_28`.

5. **Three LLM calls per turn** (extract, match, compose) against a free tier whose
   real constraint turned out to be a **per-day** cap, not a per-minute one.
   `app/llm.py:_generate` retries transient 429/503 honouring the server's
   `retryDelay`, but returns immediately on a `...PerDay...` quota id — retrying a
   daily cap only burns wall-clock, and an early version of this code spent 31 minutes
   proving it. Merging `extract` into `match` is not possible — the numeric gate needs
   the weather, which needs the location — but `extract` and `compose` could both move
   to a cheaper model. This is the first thing to change under real load, and the
   audit log now records calls and tokens per turn, so the change can be measured.

### A note on model choice

Gemini's free tier moves faster than its documentation, and two obvious picks were
dead ends before `gemini-3.5-flash-lite` stuck:

| Model | Outcome on a fresh free-tier key |
|---|---|
| `gemini-2.5-flash` | `404` — "no longer available to new users" |
| `gemini-3.6-flash` | works, but **20 requests per day**; one eval run exhausts it |
| `gemini-3.5-flash-lite` | what ships — a full eval run (~36 calls) fits comfortably |

Google no longer publishes per-model free-tier limits in the docs, so the 20/day cap
was only discoverable by hitting it. Note also that `thinking_budget` is rejected by
the Gemini 3.x models; there is no thinking config in the request at all now.

The model id is read from
`GEMINI_MODEL` and is the only provider detail outside `app/llm.py`, so moving to
another Gemini model, or to another provider entirely, touches one file.

---

## Repository layout

```
app/sops.py      policy loading, numeric gates, conflict ranking   (no API, no model)
app/weather.py   Open-Meteo geocoding + forecast, window slicing   (no model)
app/llm.py       the only module that calls a model
app/graph.py     the LangGraph agent and its branches
app/db.py        audit trail
app/api.py       FastAPI, serves the frontend too
sops/*.yaml      the policies
web/index.html   chat UI, vanilla JS, no build step
evals/           the eval suite
app/limits.py    rate limit + daily budget, in memory or Redis
app/metrics.py   Prometheus counters
app/fakes.py     fake model + recorded weather, for load tests and the stack
ops/             compose stack (2 replicas, nginx, Postgres, Redis, Prometheus), alerts
loadtest/        load generator and results
Dockerfile       the deployable image; .github/workflows/ci.yml lints, tests, builds it
```
