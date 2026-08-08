# NBA Stat Explorer

A Flask web app for exploring NBA player statistics — season averages, stat
correlations, projected stat lines, head-to-head splits, and team leaderboards
— backed by real career game logs pulled from `nba_api` and stored in
MongoDB.

## What it does

Pick a team and a player (or search by name across all 30 rosters) to
explore their career-to-date game log. Seven pages, each backed by the same
MongoDB-stored data:

| Route | What it shows |
|---|---|
| `/` (Player Stats) | Season averages, shooting splits, career highs/lows, plus three charts (points trend with rolling average, points distribution, stat correlation heatmap) |
| `/correlation` | Pearson correlation between any two stats (e.g. minutes vs. points) |
| `/predict` | A projected stat line blended from season average, last-10-game form, and a multivariable regression fit on the rest of the box score — plus the player's career hit rate against that line and two charts |
| `/matchup` | Head-to-head: a player's career averages against one specific opponent team, compared to their season averages |
| `/compare` | Two players' season averages side by side, with a comparison table and chart |
| `/splits` | Home vs. away and win vs. loss per-game averages |
| `/teams` | Whole-roster leaderboard for a chosen stat, with a bar chart of the leaders |

Every page shares a "quick search" box that autocompletes across all ~530
players regardless of team, so you don't have to know which roster someone's
on before looking them up.

The projections and correlations are for statistical exploration, not
betting advice.

## Architecture

- **Frontend/app.py** — the Flask app: all seven routes, chart generation
  (matplotlib/seaborn rendered server-side to base64 PNGs, no client-side
  charting library), and request validation.
- **mongo_store.py** — the only place that talks to MongoDB. One document per
  player: `{team, first, last, games: [...]}`, where `games` is the player's
  full career game log (same shape as the old per-player CSVs). The Mongo
  client connects lazily on first query, not at import time, so importing
  the module never makes a network call or fails just because `.env` isn't
  filled in yet.
- **ML/Multivariable.py** — the original V1 statistics module (correlation,
  regression, VIF, BIC), now reading through `mongo_store` instead of local
  CSVs. `Frontend/app.py` drives it by setting `TEAM_NAME` /
  `PLAYER_FIRST_NAME` / `PLAYER_LAST_NAME` before calling `generate_data()`.
- **Frontend/templates/** — Jinja templates sharing a common `base.html`
  layout and a `_macros.html` team/player picker macro (with the quick-search
  box) reused across every page that needs one.
- **Frontend/static/app.js** — vanilla JS: the team→player dependent
  dropdowns, the cross-page quick search, and the mobile hamburger nav.
  No frontend framework or build step.

Data flow: `nba_api` → `refresh_data.py` (writes local CSVs, mainly useful
for inspecting/diffing a refresh before it goes live) → `migrate_to_mongo.py`
(pushes those CSVs into MongoDB) → the Flask app reads exclusively from
MongoDB at runtime. The `TEAMS/` folder of CSVs is a local staging/backup
artifact, not something the running app depends on.

## Project structure

```
NBA-ML/
├── Frontend/
│   ├── app.py                  # Flask app: routes, charts, validation
│   ├── conftest.py             # puts repo root on sys.path for pytest
│   ├── requirements.txt        # duplicate of root requirements.txt
│   ├── static/
│   │   ├── app.js              # roster pickers, quick search, mobile nav
│   │   ├── styles.css
│   │   └── img/                # team logos
│   ├── templates/               # base.html + one template per route
│   └── tests/
│       └── test_app.py         # pytest suite: all routes, adversarial inputs
├── ML/
│   ├── Multivariable.py         # stats/regression, now Mongo-backed
│   ├── Descent.py, Regression.py, config.py   # V1 standalone scripts (unchanged)
├── TEAMS/                       # local CSV staging area (not read by the app)
├── mongo_store.py               # MongoDB data-access layer
├── refresh_data.py              # pulls current rosters + career logs from nba_api
├── migrate_to_mongo.py          # pushes TEAMS/ CSVs into MongoDB
├── requirements.txt
├── Procfile                     # gunicorn entry point for deployment
├── .env.example                 # documents the two required env vars
└── .gitignore                   # excludes .env and .venv
```

## Setup

**1. Install dependencies** (Python 3.10+; a virtual environment is recommended):

```bash
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

**2. Configure MongoDB.** Copy `.env.example` to `.env` and fill in your
connection details:

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-url>/?retryWrites=true&w=majority
MONGODB_DB_NAME=nba_ml
```

`.env` is gitignored — never commit real credentials.

**3. Populate the database.** If `TEAMS/` already has data (it does, checked
into this repo), just run the migration:

```bash
python migrate_to_mongo.py
```

This upserts one document per player into the `players` collection and is
safe to re-run at any time. To pull fresh rosters and career logs from
`nba_api` first (e.g. at the start of a new season):

```bash
python refresh_data.py          # writes TEAMS/<team>/GAMELOG/<player>.csv
python migrate_to_mongo.py      # pushes those into MongoDB
```

`refresh_data.py` accepts `--only "Team Name"` to refresh a single team and
`--season 2025-26` to override the auto-detected season.

## Running locally

```bash
cd Frontend
PYTHONPATH=.. python app.py
```

On Windows PowerShell, set the env var separately since it doesn't support
the `VAR=value cmd` prefix form:

```powershell
cd Frontend
$env:PYTHONPATH = ".."
python app.py
```

(`PYTHONPATH` needs to include the repo root so `app.py` can `import
mongo_store` and `from ML import Multivariable`.) The app serves on
`http://127.0.0.1:5000` by default, or the port in the `PORT` env var.

## Running tests

```bash
cd Frontend
python -m pytest -v
```

The suite (`Frontend/tests/test_app.py`) hits every route through Flask's
test client — against your real configured MongoDB, since there's no
separate test database — covering happy paths, missing/invalid input,
zero-game and one-game edge-case players, XSS/injection attempts, oversized
requests, path traversal, and security headers.

## Deployment

The app is set up for any platform that runs a `Procfile` (Heroku, Render,
Railway, etc.):

```
web: PYTHONPATH=. gunicorn --chdir Frontend app:app
```

Set `MONGODB_URI` and `MONGODB_DB_NAME` as environment variables on the
platform (not in a committed `.env`). The app binds to `0.0.0.0` and reads
`PORT` from the environment, matching what most PaaS platforms inject
automatically.

A few things worth knowing before exposing this publicly at scale:

- There's no rate limiting on the chart-generating routes (each renders 1-3
  matplotlib charts server-side per request). Fine for personal/low-traffic
  use; add something like Flask-Limiter if that changes.
- All routes are read-only — nothing a visitor does writes to the database —
  so there's no CSRF exposure to worry about.

## Data

- **Source**: `nba_api`, the community wrapper around stats.nba.com.
- **Season**: whatever `refresh_data.py` was last run against — currently
  the 2025-26 season, auto-detected from the current date (season rolls
  over every October).
- **Coverage**: all 30 teams, ~530 players, each with their full career game
  log (not just the current season) — the trend charts and predictions use
  career-to-date data, not just this year.
- **Team logos**: static images under `Frontend/static/img/`, mapped to team
  names in `Frontend/app.py`'s `TEAM_LOGOS` dict.
