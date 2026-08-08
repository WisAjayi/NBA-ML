"""One-time (or repeatable) migration: loads the locally-cached TEAMS/ game
logs into MongoDB, so the Flask app can run without the TEAMS/ folder.

Run this after refresh_data.py, or any time you want to push local CSV
changes into the database configured in .env.

Usage:
    python migrate_to_mongo.py
"""
import sys
from pathlib import Path

import pandas as pd

import mongo_store

try:  # player names can contain non-ASCII characters; don't crash on narrow consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
TEAMS_DIR = ROOT / "TEAMS"


def main():
    mongo_store.ensure_indexes()

    upserted = 0
    skipped = 0

    for team_dir in sorted(TEAMS_DIR.iterdir()):
        if not team_dir.is_dir():
            continue
        gamelog_dir = team_dir / "GAMELOG"
        if not gamelog_dir.is_dir():
            continue

        print(f"\n=== {team_dir.name} ===")
        for csv_file in sorted(gamelog_dir.glob("*.csv")):
            first, sep, last = csv_file.stem.partition("_")
            if not sep:
                skipped += 1
                continue

            df = pd.read_csv(csv_file)
            games = df.to_dict("records")
            mongo_store.upsert_player(team_dir.name, first, last, games)
            upserted += 1
            print(f"  {first} {last.replace('_', ' ')} -> {len(games)} games")

    print(f"\nDone. Upserted {upserted} player document(s), skipped {skipped} unrecognized file(s).")


if __name__ == "__main__":
    main()
