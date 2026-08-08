"""Refresh TEAMS/ with the current-season roster and full career game logs
for every player, pulled live via nba_api.

For each team folder already in TEAMS/:
  - fetches the current roster (CommonTeamRoster)
  - fetches each player's full career game log (PlayerGameLog, SeasonAll)
  - rewrites cred.txt with the current roster
  - removes GAMELOG csvs for players no longer on the roster

Usage:
    python refresh_data.py                       # refresh every team
    python refresh_data.py --only "Atlanta Hawks" # refresh just one team (repeatable)
    python refresh_data.py --season 2025-26       # override the detected season
"""
import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import teams as static_teams

try:  # player names can contain non-ASCII characters; don't crash on narrow consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
TEAMS_DIR = ROOT / "TEAMS"

REQUEST_DELAY = 0.6  # be polite to the free public API
RETRIES = 4

# Existing TEAMS/ folder name -> official nba_api abbreviation.
# Folder names are kept as-is (some differ from the official name, e.g. plurals)
# so the Frontend's team_logos mapping and bookmarked folder paths keep working.
FOLDER_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Maverick": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heats": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunders": "OKC",
    "Orlando Magics": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trailblazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def current_season():
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def sanitize(name):
    return name.replace(".", "").strip()


def fetch_with_retry(fn, retries=RETRIES, delay=1.5):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # network hiccups, timeouts, throttling
            last_exc = exc
            time.sleep(delay * (attempt + 1))
    raise last_exc


def refresh_team(folder_name, abbrev, team_id, season, failures):
    team_dir = TEAMS_DIR / folder_name
    gamelog_dir = team_dir / "GAMELOG"
    gamelog_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {folder_name} ({abbrev}) - season {season} ===")

    try:
        roster_df = fetch_with_retry(
            lambda: commonteamroster.CommonTeamRoster(
                team_id=team_id, season=season, timeout=30
            ).get_data_frames()[0]
        )
    except Exception as exc:
        print(f"  FAILED to fetch roster: {exc}")
        failures.append((folder_name, "ROSTER", str(exc)))
        return
    time.sleep(REQUEST_DELAY)

    current_files = set()
    cred_lines = []

    for _, row in roster_df.iterrows():
        full_name = sanitize(row["PLAYER"])
        player_id = row["PLAYER_ID"]
        filename_base = full_name.replace(" ", "_")
        current_files.add(f"{filename_base}.csv")
        cred_lines.append(f"{full_name} | {player_id}")

        try:
            gamelog_df = fetch_with_retry(
                lambda: playergamelog.PlayerGameLog(
                    player_id=player_id, season=SeasonAll.all, timeout=30
                ).get_data_frames()[0]
            )
            gamelog_df.to_csv(gamelog_dir / f"{filename_base}.csv", index=False)
            print(f"  ok    {full_name} ({len(gamelog_df)} career games)")
        except Exception as exc:
            print(f"  FAILED {full_name}: {exc}")
            failures.append((folder_name, full_name, str(exc)))

        time.sleep(REQUEST_DELAY)

    removed = 0
    for csv_file in gamelog_dir.glob("*.csv"):
        if csv_file.name not in current_files:
            csv_file.unlink()
            removed += 1
    if removed:
        print(f"  removed {removed} player(s) no longer on the roster")

    (team_dir / "cred.txt").write_text("\n".join(cred_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="Only refresh this team folder (repeatable)")
    parser.add_argument("--season", default=None, help="Override the season, e.g. 2025-26")
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up TEAMS/ first")
    args = parser.parse_args()

    season = args.season or current_season()
    team_names = args.only or list(FOLDER_TO_ABBREV.keys())

    unknown = [t for t in team_names if t not in FOLDER_TO_ABBREV]
    if unknown:
        sys.exit(f"Unknown team folder(s): {unknown}")

    if not args.no_backup:
        backup_dir = ROOT / f"TEAMS_backup_{datetime.now():%Y%m%d_%H%M%S}"
        print(f"Backing up existing TEAMS/ to {backup_dir.name}")
        shutil.copytree(TEAMS_DIR, backup_dir)

    abbrev_to_id = {t["abbreviation"]: t["id"] for t in static_teams.get_teams()}

    failures = []
    for folder_name in team_names:
        abbrev = FOLDER_TO_ABBREV[folder_name]
        refresh_team(folder_name, abbrev, abbrev_to_id[abbrev], season, failures)

    print(f"\nDone. Refreshed {len(team_names)} team(s) for season {season}.")
    if failures:
        print(f"{len(failures)} failure(s):")
        for team, who, err in failures:
            print(f"  {team} / {who}: {err}")


if __name__ == "__main__":
    main()
