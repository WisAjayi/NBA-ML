"""MongoDB-backed data access for team rosters and player game logs.

Replaces the old TEAMS/<team>/GAMELOG/<player>.csv layout with a single
"players" collection: one document per player, holding their team, name,
and their full game log as an embedded list of rows (the same shape as the
old CSVs, one dict per game).

Connection details come from environment variables, loaded from a local
.env file (see .env.example):
    MONGODB_URI      - full connection string, e.g. mongodb+srv://...
    MONGODB_DB_NAME  - database name (defaults to "nba_ml")

The client is created lazily on first actual query, not at import time, so
importing this module never makes a network call and never fails just
because .env hasn't been filled in yet.
"""
from functools import lru_cache
import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient

load_dotenv()

PLAYERS_COLLECTION = "players"


class PlayerNotFoundError(FileNotFoundError):
    """No game log is stored for this team/player combination.

    Subclasses FileNotFoundError so existing `except FileNotFoundError`
    handlers (written back when this data came from local CSV files) keep
    working unchanged.
    """


@lru_cache(maxsize=1)
def get_client():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Copy .env.example to .env in the project "
            "root and fill in your MongoDB connection details."
        )
    return MongoClient(uri, serverSelectionTimeoutMS=8000)


@lru_cache(maxsize=1)
def get_db():
    db_name = os.environ.get("MONGODB_DB_NAME", "nba_ml")
    return get_client()[db_name]


def players_collection():
    return get_db()[PLAYERS_COLLECTION]


def ensure_indexes():
    """Idempotent; safe to call every time the app or a migration script starts."""
    players_collection().create_index(
        [("team", ASCENDING), ("first", ASCENDING), ("last", ASCENDING)],
        unique=True,
        name="team_first_last",
    )


@lru_cache(maxsize=1)
def get_teams():
    return sorted(players_collection().distinct("team"))


@lru_cache(maxsize=1)
def get_rosters():
    rosters = {}
    for doc in players_collection().find({}, {"team": 1, "first": 1, "last": 1}):
        rosters.setdefault(doc["team"], []).append({"first": doc["first"], "last": doc["last"]})
    for roster in rosters.values():
        roster.sort(key=lambda p: (p["first"], p["last"]))
    return rosters


@lru_cache(maxsize=1)
def get_all_players():
    players = [
        {"team": doc["team"], "first": doc["first"], "last": doc["last"]}
        for doc in players_collection().find({}, {"team": 1, "first": 1, "last": 1})
    ]
    return sorted(players, key=lambda p: (p["first"], p["last"]))


def load_player_data(team, first, last):
    doc = players_collection().find_one({"team": team, "first": first, "last": last})
    if not doc or not doc.get("games"):
        raise PlayerNotFoundError(f"No game log stored for {first} {last} on {team}.")
    return pd.DataFrame(doc["games"])


def load_team_data(team):
    """Every player on a team in a single query -> {(first, last): DataFrame}."""
    return {
        (doc["first"], doc["last"]): pd.DataFrame(doc.get("games", []))
        for doc in players_collection().find({"team": team})
    }


def upsert_player(team, first, last, games):
    """games: list of dict rows (e.g. df.to_dict("records"))."""
    players_collection().update_one(
        {"team": team, "first": first, "last": last},
        {"$set": {"team": team, "first": first, "last": last, "games": games}},
        upsert=True,
    )
