import base64
import io
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

import mongo_store
from ML import Multivariable

sns.set_theme(style="whitegrid")

NAVY = "#0c1f3f"
ORANGE = "#f58426"

CORRELATION_COLUMNS = [
    "PTS", "AST", "REB", "STL", "BLK", "TOV", "MIN", "FGM", "FGA", "FG3M", "FTM", "FTA",
]

PREDICTION_FEATURE_COLUMNS = [
    "MIN", "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PTS",
]

# Columns that are an exact arithmetic function of a given target (e.g. PTS = 2*FGM + FG3M + FTM,
# REB = OREB + DREB) must be excluded as features for that target, or the model just trivially
# reconstructs the identity instead of predicting anything.
PREDICTION_EXCLUDED_COMPONENTS = {
    "PTS": {"FGM", "FGA", "FG3M", "FTM", "FTA"},
    "REB": {"OREB", "DREB"},
}

LOW_SAMPLE_THRESHOLD = 10

STAT_OPTIONS = [
    ("PTS", "Points"),
    ("AST", "Assists"),
    ("STL", "Steals"),
    ("BLK", "Blocks"),
    ("REB", "Rebounds"),
    ("TOV", "Turnovers"),
    ("MIN", "Minutes"),
]

VALID_STATS = {abbrev for abbrev, _ in STAT_OPTIONS}

CORE_STATS = [abbrev for abbrev, _ in STAT_OPTIONS]

# Folder name -> official 3-letter abbreviation, as it appears in the MATCHUP column
# (e.g. "ATL vs. BOS" / "ATL @ BOS"). Used to filter a player's log by opponent.
TEAM_ABBREVIATIONS = {
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

TEAM_LOGOS = {
    "Atlanta Hawks": "atlanta.png",
    "Boston Celtics": "boston.png",
    "Brooklyn Nets": "brooklyn.png",
    "Charlotte Hornets": "charlotte.png",
    "Chicago Bulls": "chicago.png",
    "Cleveland Cavaliers": "cleveland.png",
    "Dallas Maverick": "dallas.png",
    "Denver Nuggets": "denver.png",
    "Detroit Pistons": "detroit.png",
    "Golden State Warriors": "golden.png",
    "Houston Rockets": "houston.png",
    "Indiana Pacers": "indiana.png",
    "Los Angeles Clippers": "lac.png",
    "Los Angeles Lakers": "lal.png",
    "Memphis Grizzlies": "memphis.png",
    "Miami Heats": "miami.png",
    "Milwaukee Bucks": "milwaukee.png",
    "Minnesota Timberwolves": "minnesota.png",
    "New Orleans Pelicans": "new-orleans.png",
    "New York Knicks": "new-york.png",
    "Oklahoma City Thunders": "oklahoma.png",
    "Orlando Magics": "orlando.png",
    "Philadelphia 76ers": "philadelphia.png",
    "Phoenix Suns": "phoenix.png",
    "Portland Trailblazers": "portland.png",
    "Sacramento Kings": "sacramento.png",
    "San Antonio Spurs": "san-antonio.png",
    "Toronto Raptors": "toronto.png",
    "Utah Jazz": "utah-jazz.png",
    "Washington Wizards": "washington.png",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # these forms never legitimately need more than a few KB


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def load_player_data(team, first, last):
    Multivariable.PLAYER_FIRST_NAME = first
    Multivariable.PLAYER_LAST_NAME = last
    Multivariable.TEAM_NAME = team
    return Multivariable.generate_data()


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_charts(data):
    """Render a few matplotlib/seaborn views of a player's game log to base64 PNGs."""
    charts = {}

    chronological = data.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(chronological.index, chronological["PTS"], color=NAVY, linewidth=1, alpha=0.5, label="Points")
    rolling = chronological["PTS"].rolling(10, min_periods=1).mean()
    ax.plot(chronological.index, rolling, color=ORANGE, linewidth=2.5, label="10-game average")
    ax.set_xlabel("Game # (career-to-date)")
    ax.set_ylabel("Points")
    ax.set_title("Points per game over time")
    ax.legend(frameon=False)
    charts["trend"] = _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data["PTS"], bins=20, kde=True, color=ORANGE, ax=ax)
    ax.set_xlabel("Points")
    ax.set_title("Points distribution")
    charts["distribution"] = _fig_to_base64(fig)

    columns = [c for c in CORRELATION_COLUMNS if c in data.columns]
    corr = data[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Stat correlations")
    charts["heatmap"] = _fig_to_base64(fig)

    return charts


def _prediction_chart(values, line, stat_label):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(values, bins=20, color="#94a3b8", ax=ax)
    ax.axvline(line, color=ORANGE, linewidth=2.5, linestyle="--", label=f"Line: {line:g}")
    ax.set_xlabel(stat_label)
    ax.set_title(f"{stat_label} per game vs. the projected line")
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def _prediction_trend_chart(chronological, stat, line, stat_label):
    values = chronological[stat]
    roll5 = values.rolling(5, min_periods=1).mean()
    roll10 = values.rolling(10, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(chronological.index, values, color="#94a3b8", s=16, alpha=0.6, label=f"{stat_label} per game")
    ax.plot(chronological.index, roll5, color=NAVY, linewidth=2, label="5-game average")
    ax.plot(chronological.index, roll10, color=ORANGE, linewidth=2.5, label="10-game average")
    ax.axhline(line, color="#ef4444", linewidth=1.5, linestyle="--", label=f"Projected line ({line:g})")
    ax.set_xlabel("Game # (career-to-date)")
    ax.set_ylabel(stat_label)
    ax.set_title(f"{stat_label} per game with 5- and 10-game rolling averages")
    ax.legend(frameon=False, fontsize=8)
    return _fig_to_base64(fig)


def _grouped_bar_chart(labels, series, title, ylabel=""):
    """series: dict[series_name] -> list of values aligned with labels."""
    colors = [NAVY, ORANGE, "#10b981", "#ef4444"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    n = len(series)
    width = 0.8 / max(n, 1)
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, values, width, label=name, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def build_prediction(data, stat):
    if data.empty:
        raise ValueError("This player hasn't played any games yet, so there's nothing to project from.")

    excluded = PREDICTION_EXCLUDED_COMPONENTS.get(stat, set())
    features = [
        c for c in PREDICTION_FEATURE_COLUMNS
        if c in data.columns and c != stat and c not in excluded
    ]
    chronological = data.iloc[::-1].reset_index(drop=True)  # oldest -> newest

    season_avg = float(data[stat].mean())
    recent_avg = float(chronological[stat].tail(10).mean())

    model = LinearRegression()
    model.fit(data[features], data[stat])
    r_squared = model.score(data[features], data[stat])

    recent_features = chronological[features].tail(5).mean().to_frame().T
    model_projection = float(model.predict(recent_features)[0])

    blended = (season_avg + recent_avg + model_projection) / 3
    line = round(blended * 2) / 2  # round to the nearest half, like a sportsbook line

    total_games = len(data[stat])
    over = int((data[stat] > line).sum())
    under = int((data[stat] < line).sum())
    push = total_games - over - under

    return {
        "line": line,
        "season_avg": season_avg,
        "recent_avg": recent_avg,
        "model_projection": model_projection,
        "r_squared": r_squared,
        "over_count": over,
        "under_count": under,
        "push_count": push,
        "total_games": total_games,
        "over_pct": over / total_games * 100,
        "under_pct": under / total_games * 100,
        "small_sample": total_games < LOW_SAMPLE_THRESHOLD,
        "chart": _prediction_chart(data[stat], line, dict(STAT_OPTIONS).get(stat, stat)),
        "trend_chart": _prediction_trend_chart(
            chronological, stat, line, dict(STAT_OPTIONS).get(stat, stat)
        ),
    }


def build_splits(data):
    home_mask = data["MATCHUP"].str.contains(" vs. ", regex=False)
    win_mask = data["WL"] == "W"

    groups = {
        "Home": data[home_mask],
        "Away": data[~home_mask],
        "Wins": data[win_mask],
        "Losses": data[~win_mask],
    }

    rows = []
    for abbrev, label in STAT_OPTIONS:
        row = {"abbrev": abbrev, "label": label}
        for name, subset in groups.items():
            row[name] = float(subset[abbrev].mean()) if len(subset) else None
        rows.append(row)

    chart_stats = [abbrev for abbrev in CORE_STATS if abbrev != "MIN"]

    def averages(subset):
        return [float(subset[s].mean()) if len(subset) else 0.0 for s in chart_stats]

    home_away_chart = _grouped_bar_chart(
        chart_stats,
        {"Home": averages(groups["Home"]), "Away": averages(groups["Away"])},
        "Home vs. away averages",
    )
    win_loss_chart = _grouped_bar_chart(
        chart_stats,
        {"Wins": averages(groups["Wins"]), "Losses": averages(groups["Losses"])},
        "Win vs. loss averages",
    )

    return {
        "rows": rows,
        "home_games": int(home_mask.sum()),
        "away_games": int((~home_mask).sum()),
        "wins": int(win_mask.sum()),
        "losses": int((~win_mask).sum()),
        "home_away_chart": home_away_chart,
        "win_loss_chart": win_loss_chart,
    }


def base_context(**overrides):
    context = {
        "teams": mongo_store.get_teams(),
        "rosters": mongo_store.get_rosters(),
        "all_players": mongo_store.get_all_players(),
        "team_logos": TEAM_LOGOS,
        "stats": STAT_OPTIONS,
        "selected_team": "",
        "error": None,
        "name": None,
        "charts": None,
        "corr": None,
        "player_name": None,
        "stat1": None,
        "stat2": None,
        "target_stat": "PTS",
        "prediction": None,
        "opponent_team": "",
        "matchup_result": None,
        "selected_team_a": "",
        "selected_team_b": "",
        "compare_result": None,
        "splits_result": None,
        "dashboard_stat": "PTS",
        "dashboard": None,
    }
    context.update(overrides)
    return context


@app.route("/", methods=["GET", "POST"])
def index():
    context = base_context()

    if request.method == "POST":
        team = request.form.get("team", "").strip()
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        context["selected_team"] = team

        if not (team and first and last):
            context["error"] = "Please choose a team and a player."
        else:
            try:
                data = load_player_data(team, first, last)

                context.update(
                    name=f"{first} {last.replace('_', ' ')}",
                    points_average=data["PTS"].mean(),
                    points_assist=data["AST"].mean(),
                    points_steal=data["STL"].mean(),
                    points_block=data["BLK"].mean(),
                    points_rebound=data["REB"].mean(),
                    points_minutes=data["MIN"].mean(),
                    points_turnover=data["TOV"].mean(),
                    fgm=data["FGM"].mean(),
                    fga=data["FGA"].mean(),
                    fg3m=data["FG3M"].mean(),
                    ftm=data["FTM"].mean(),
                    fta=data["FTA"].mean(),
                    points_most=data["PTS"].max(),
                    assist_most=data["AST"].max(),
                    steal_most=data["STL"].max(),
                    block_most=data["BLK"].max(),
                    rebound_most=data["REB"].max(),
                    minutes_most=data["MIN"].max(),
                    turnover_most=data["TOV"].max(),
                    points_least=data["PTS"].min(),
                    assist_least=data["AST"].min(),
                    steal_least=data["STL"].min(),
                    block_least=data["BLK"].min(),
                    rebound_least=data["REB"].min(),
                    minutes_least=data["MIN"].min(),
                    turnover_least=data["TOV"].min(),
                    charts=build_charts(data),
                )
            except FileNotFoundError:
                context["error"] = (
                    f"No stats found for {first} {last.replace('_', ' ')} on {team}. "
                    "Try picking a player from the dropdown."
                )
            except Exception as e:
                context["error"] = f"Something went wrong: {e}"

    return render_template("index.html", **context)


@app.route("/correlation", methods=["GET", "POST"])
def correlate():
    context = base_context()

    if request.method == "POST":
        team = request.form.get("team", "").strip()
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        stat_1 = request.form.get("stat_1", "").strip()
        stat_2 = request.form.get("stat_2", "").strip()
        context["selected_team"] = team

        if not (team and first and last and stat_1 and stat_2):
            context["error"] = "Please fill in every field."
        elif stat_1 not in VALID_STATS or stat_2 not in VALID_STATS:
            context["error"] = "Please choose stats from the provided list."
        else:
            try:
                load_player_data(team, first, last)

                Multivariable.Name_Abbrev = stat_1
                Multivariable.Compare_Abbrev = stat_2

                context.update(
                    player_name=f"{first} {last.replace('_', ' ')}",
                    stat1=stat_1,
                    stat2=stat_2,
                    corr=Multivariable.corr(),
                )
            except FileNotFoundError:
                context["error"] = (
                    f"No stats found for {first} {last.replace('_', ' ')} on {team}."
                )
            except Exception as e:
                context["error"] = f"Something went wrong: {e}"

    return render_template("correlate.html", **context)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    context = base_context()

    if request.method == "POST":
        team = request.form.get("team", "").strip()
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        stat = request.form.get("stat", "PTS").strip()
        context["selected_team"] = team
        context["target_stat"] = stat

        if not (team and first and last and stat):
            context["error"] = "Please choose a team, a player, and a stat."
        elif stat not in VALID_STATS:
            context["error"] = "Please choose a stat from the provided list."
        else:
            try:
                data = load_player_data(team, first, last)

                context.update(
                    player_name=f"{first} {last.replace('_', ' ')}",
                    prediction=build_prediction(data, stat),
                )
            except FileNotFoundError:
                context["error"] = (
                    f"No stats found for {first} {last.replace('_', ' ')} on {team}."
                )
            except Exception as e:
                context["error"] = f"Something went wrong: {e}"

    return render_template("predict.html", **context)


@app.route("/matchup", methods=["GET", "POST"])
def matchup():
    context = base_context()

    if request.method == "POST":
        team = request.form.get("team", "").strip()
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        opponent_team = request.form.get("opponent_team", "").strip()
        context["selected_team"] = team
        context["opponent_team"] = opponent_team

        if not (team and first and last and opponent_team):
            context["error"] = "Please choose a team, a player, and an opponent."
        else:
            try:
                data = load_player_data(team, first, last)
                opponent_abbrev = TEAM_ABBREVIATIONS.get(opponent_team)
                vs_data = data[data["MATCHUP"].str.split().str[-1] == opponent_abbrev]

                if vs_data.empty:
                    context["error"] = (
                        f"No career games found for {first} {last.replace('_', ' ')} "
                        f"against {opponent_team}."
                    )
                else:
                    rows = [
                        {
                            "abbrev": abbrev,
                            "label": label,
                            "vs_avg": float(vs_data[abbrev].mean()),
                            "season_avg": float(data[abbrev].mean()),
                        }
                        for abbrev, label in STAT_OPTIONS
                    ]
                    for row in rows:
                        row["diff"] = row["vs_avg"] - row["season_avg"]

                    chart_stats = [r for r in rows if r["abbrev"] != "MIN"]
                    chart = _grouped_bar_chart(
                        [r["abbrev"] for r in chart_stats],
                        {
                            f"vs. {opponent_team}": [r["vs_avg"] for r in chart_stats],
                            "Season avg": [r["season_avg"] for r in chart_stats],
                        },
                        f"{first} {last.replace('_', ' ')} vs. {opponent_team}",
                    )

                    context.update(
                        player_name=f"{first} {last.replace('_', ' ')}",
                        matchup_result={
                            "games": len(vs_data),
                            "wins": int((vs_data["WL"] == "W").sum()),
                            "losses": int((vs_data["WL"] == "L").sum()),
                            "rows": rows,
                            "chart": chart,
                        },
                    )
            except FileNotFoundError:
                context["error"] = f"No stats found for {first} {last.replace('_', ' ')} on {team}."
            except Exception as e:
                context["error"] = f"Something went wrong: {e}"

    return render_template("matchup.html", **context)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    context = base_context()

    if request.method == "POST":
        team_a = request.form.get("team_a", "").strip()
        first_a = request.form.get("first_name_a", "").strip()
        last_a = request.form.get("last_name_a", "").strip()
        team_b = request.form.get("team_b", "").strip()
        first_b = request.form.get("first_name_b", "").strip()
        last_b = request.form.get("last_name_b", "").strip()
        context["selected_team_a"] = team_a
        context["selected_team_b"] = team_b

        if not (team_a and first_a and last_a and team_b and first_b and last_b):
            context["error"] = "Please choose two players to compare."
        else:
            try:
                data_a = load_player_data(team_a, first_a, last_a)
                name_a = f"{first_a} {last_a.replace('_', ' ')}"
                data_b = load_player_data(team_b, first_b, last_b)
                name_b = f"{first_b} {last_b.replace('_', ' ')}"

                rows = [
                    {
                        "abbrev": abbrev,
                        "label": label,
                        "a": float(data_a[abbrev].mean()),
                        "b": float(data_b[abbrev].mean()),
                    }
                    for abbrev, label in STAT_OPTIONS
                ]

                chart = _grouped_bar_chart(
                    [r["abbrev"] for r in rows],
                    {name_a: [r["a"] for r in rows], name_b: [r["b"] for r in rows]},
                    f"{name_a} vs. {name_b} — season averages",
                )

                context.update(
                    compare_result={
                        "name_a": name_a,
                        "name_b": name_b,
                        "team_a": team_a,
                        "team_b": team_b,
                        "rows": rows,
                        "chart": chart,
                    },
                )
            except FileNotFoundError:
                context["error"] = "Couldn't find stats for one of those players. Try picking from the dropdowns."
            except Exception as e:
                context["error"] = f"Something went wrong: {e}"

    return render_template("compare.html", **context)


@app.route("/splits", methods=["GET", "POST"])
def splits():
    context = base_context()

    if request.method == "POST":
        team = request.form.get("team", "").strip()
        first = request.form.get("first_name", "").strip()
        last = request.form.get("last_name", "").strip()
        context["selected_team"] = team

        if not (team and first and last):
            context["error"] = "Please choose a team and a player."
        else:
            try:
                data = load_player_data(team, first, last)
                if data.empty:
                    context["error"] = "This player hasn't played any games yet."
                else:
                    context.update(
                        player_name=f"{first} {last.replace('_', ' ')}",
                        splits_result=build_splits(data),
                    )
            except FileNotFoundError:
                context["error"] = f"No stats found for {first} {last.replace('_', ' ')} on {team}."
            except Exception as e:
                context["error"] = f"Something went wrong: {e}"

    return render_template("splits.html", **context)


@app.route("/teams", methods=["GET", "POST"])
def team_dashboard():
    context = base_context()

    if request.method == "POST":
        team = request.form.get("team", "").strip()
        stat = request.form.get("stat", "PTS").strip()
        context["selected_team"] = team
        context["dashboard_stat"] = stat

        if not (team and stat):
            context["error"] = "Please choose a team and a stat."
        elif stat not in VALID_STATS:
            context["error"] = "Please choose a stat from the provided list."
        else:
            try:
                team_data = mongo_store.load_team_data(team)  # one query for the whole roster
                rows = []
                for (first, last), data in team_data.items():
                    if data.empty:
                        continue
                    rows.append({
                        "name": f"{first} {last.replace('_', ' ')}",
                        "games": len(data),
                        "avg": float(data[stat].mean()),
                        "low_sample": len(data) < LOW_SAMPLE_THRESHOLD,
                    })

                if not rows:
                    context["error"] = f"No roster data found for {team}."
                else:
                    rows.sort(key=lambda r: r["avg"], reverse=True)

                    chart_rows = [r for r in rows if not r["low_sample"]][:15]
                    chart = None
                    if chart_rows:
                        fig, ax = plt.subplots(figsize=(8, max(3, len(chart_rows) * 0.4)))
                        names = [r["name"] for r in reversed(chart_rows)]
                        values = [r["avg"] for r in reversed(chart_rows)]
                        ax.barh(names, values, color=ORANGE)
                        ax.set_xlabel(dict(STAT_OPTIONS).get(stat, stat))
                        ax.set_title(f"{team} — {dict(STAT_OPTIONS).get(stat, stat)} leaders")
                        chart = _fig_to_base64(fig)

                    context.update(dashboard={"rows": rows, "chart": chart})
            except Exception as e:
                context["error"] = f"Something went wrong: {e}"

    return render_template("teams.html", **context)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
