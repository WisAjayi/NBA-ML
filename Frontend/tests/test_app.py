"""Pre-deployment test suite: exercises every route with both happy-path and
adversarial input, and checks for common web-app security/robustness issues
(error leakage, XSS escaping, path traversal, oversized input, debug mode).

Run from Frontend/:
    ../.venv/Scripts/python.exe -m pytest -v
"""
import pytest

import app as flask_app_module
import mongo_store

ROUTES = ["/", "/correlation", "/predict", "/matchup", "/compare", "/splits", "/teams"]

# Strings that should never appear in a response to an end user, regardless of
# what triggered the error - these all indicate leaked internals.
LEAK_MARKERS = [b"Traceback (most recent", b"KeyError:", b"ValueError:", b"pymongo.",
                b"mongodb+srv", b"MONGODB_URI", b'File "', b".py\", line", b"Werkzeug/"]


@pytest.fixture(scope="module")
def client():
    flask_app_module.app.config.update(TESTING=True)
    with flask_app_module.app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def sample_player():
    """A real player with a healthy game count, so happy-path assertions are meaningful."""
    players = mongo_store.get_all_players()
    assert players, "No players found in MongoDB - run migrate_to_mongo.py first"
    for p in players:
        doc = mongo_store.players_collection().find_one(
            {"team": p["team"], "first": p["first"], "last": p["last"]}, {"games": 1}
        )
        if doc and len(doc.get("games", [])) >= 20:
            return p
    return players[0]


def assert_no_leak(resp):
    for marker in LEAK_MARKERS:
        assert marker not in resp.data, f"Leaked internal detail: {marker!r} in response"


# ---------------------------------------------------------------------------
# Smoke tests: every route, every method
# ---------------------------------------------------------------------------

class TestSmoke:
    @pytest.mark.parametrize("route", ROUTES)
    def test_get_returns_200(self, client, route):
        resp = client.get(route)
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_unknown_route_is_404(self, client):
        resp = client.get("/this-route-does-not-exist")
        assert resp.status_code == 404
        assert_no_leak(resp)

    @pytest.mark.parametrize("route", ROUTES)
    def test_disallowed_method_is_405(self, client, route):
        resp = client.delete(route)
        assert resp.status_code == 405

    def test_debug_mode_is_off(self):
        assert flask_app_module.app.debug is False

    def test_static_path_traversal_blocked(self, client):
        resp = client.get("/static/..%2f..%2fapp.py")
        assert resp.status_code in (400, 403, 404)
        assert b"Flask" not in resp.data and b"import" not in resp.data

    def test_static_serves_real_file(self, client):
        resp = client.get("/static/styles.css")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_player_stats(self, client, sample_player):
        resp = client.post("/", data={
            "team": sample_player["team"],
            "first_name": sample_player["first"],
            "last_name": sample_player["last"],
        })
        assert resp.status_code == 200
        assert b"player-name" in resp.data
        assert resp.data.count(b"data:image/png;base64") == 3
        assert_no_leak(resp)

    def test_correlation(self, client, sample_player):
        resp = client.post("/correlation", data={
            "team": sample_player["team"],
            "first_name": sample_player["first"],
            "last_name": sample_player["last"],
            "stat_1": "PTS",
            "stat_2": "MIN",
        })
        assert resp.status_code == 200
        assert b"correlation-value" in resp.data
        assert_no_leak(resp)

    def test_predict(self, client, sample_player):
        resp = client.post("/predict", data={
            "team": sample_player["team"],
            "first_name": sample_player["first"],
            "last_name": sample_player["last"],
            "stat": "PTS",
        })
        assert resp.status_code == 200
        assert b"line-value" in resp.data
        assert_no_leak(resp)

    def test_matchup(self, client, sample_player):
        resp = client.post("/matchup", data={
            "team": sample_player["team"],
            "first_name": sample_player["first"],
            "last_name": sample_player["last"],
            "opponent_team": "Boston Celtics",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_compare(self, client, sample_player):
        resp = client.post("/compare", data={
            "team_a": sample_player["team"], "first_name_a": sample_player["first"], "last_name_a": sample_player["last"],
            "team_b": "Los Angeles Lakers", "first_name_b": "LeBron", "last_name_b": "James",
        })
        assert resp.status_code == 200
        assert b"compare-table" in resp.data
        assert_no_leak(resp)

    def test_splits(self, client, sample_player):
        resp = client.post("/splits", data={
            "team": sample_player["team"],
            "first_name": sample_player["first"],
            "last_name": sample_player["last"],
        })
        assert resp.status_code == 200
        assert b"splits-table" in resp.data
        assert_no_leak(resp)

    def test_team_dashboard(self, client, sample_player):
        resp = client.post("/teams", data={"team": sample_player["team"], "stat": "PTS"})
        assert resp.status_code == 200
        assert b"leaderboard-table" in resp.data
        assert_no_leak(resp)


# ---------------------------------------------------------------------------
# Missing / invalid input handling
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.parametrize("route,data", [
        ("/", {"team": "Los Angeles Lakers"}),
        ("/correlation", {"team": "Los Angeles Lakers", "first_name": "LeBron"}),
        ("/predict", {"team": "Los Angeles Lakers"}),
        ("/matchup", {"team": "Los Angeles Lakers", "first_name": "LeBron", "last_name": "James"}),
        ("/compare", {"team_a": "Los Angeles Lakers", "first_name_a": "LeBron", "last_name_a": "James"}),
        ("/splits", {"team": "Los Angeles Lakers"}),
        ("/teams", {}),
    ])
    def test_missing_fields_gives_friendly_error_not_crash(self, client, route, data):
        resp = client.post(route, data=data)
        assert resp.status_code == 200
        assert b"alert" in resp.data
        assert_no_leak(resp)

    def test_nonexistent_player(self, client):
        resp = client.post("/", data={
            "team": "Los Angeles Lakers", "first_name": "Nobody", "last_name": "Fake",
        })
        assert resp.status_code == 200
        assert b"No stats found" in resp.data
        assert_no_leak(resp)

    def test_nonexistent_team(self, client):
        resp = client.post("/", data={
            "team": "Fictional Team", "first_name": "Nobody", "last_name": "Fake",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_zero_game_player(self, client):
        resp = client.post("/", data={
            "team": "Oklahoma City Thunders", "first_name": "Thomas", "last_name": "Sorber",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_zero_game_player_predict(self, client):
        resp = client.post("/predict", data={
            "team": "Oklahoma City Thunders", "first_name": "Thomas", "last_name": "Sorber", "stat": "PTS",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_one_game_player_charts(self, client):
        resp = client.post("/", data={
            "team": "Utah Jazz", "first_name": "Hayden", "last_name": "Gray",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_one_game_player_splits(self, client):
        resp = client.post("/splits", data={
            "team": "Utah Jazz", "first_name": "Hayden", "last_name": "Gray",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)


# ---------------------------------------------------------------------------
# Adversarial: things a hostile client would try
# ---------------------------------------------------------------------------

class TestAdversarial:
    def test_xss_in_name_fields_is_escaped(self, client):
        payload = "<script>alert(document.cookie)</script>"
        resp = client.post("/", data={
            "team": "Los Angeles Lakers", "first_name": payload, "last_name": "Y",
        })
        assert resp.status_code == 200
        assert payload.encode() not in resp.data
        assert_no_leak(resp)

    def test_invalid_stat_on_correlation(self, client, sample_player):
        resp = client.post("/correlation", data={
            "team": sample_player["team"], "first_name": sample_player["first"], "last_name": sample_player["last"],
            "stat_1": "'; DROP TABLE players; --", "stat_2": "PTS",
        })
        assert resp.status_code == 200
        assert b"Please choose stats from the provided list" in resp.data
        assert b"Something went wrong" not in resp.data
        assert_no_leak(resp)

    def test_invalid_stat_on_predict(self, client, sample_player):
        resp = client.post("/predict", data={
            "team": sample_player["team"], "first_name": sample_player["first"], "last_name": sample_player["last"],
            "stat": "NOT_A_REAL_STAT",
        })
        assert resp.status_code == 200
        assert b"Please choose a stat from the provided list" in resp.data
        assert b"Something went wrong" not in resp.data
        assert_no_leak(resp)

    def test_invalid_stat_on_team_dashboard(self, client, sample_player):
        resp = client.post("/teams", data={"team": sample_player["team"], "stat": "$where"})
        assert resp.status_code == 200
        assert b"Please choose a stat from the provided list" in resp.data
        assert b"Something went wrong" not in resp.data
        assert_no_leak(resp)

    def test_matchup_same_team_as_opponent(self, client, sample_player):
        resp = client.post("/matchup", data={
            "team": sample_player["team"], "first_name": sample_player["first"], "last_name": sample_player["last"],
            "opponent_team": sample_player["team"],
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_matchup_unknown_opponent(self, client, sample_player):
        resp = client.post("/matchup", data={
            "team": sample_player["team"], "first_name": sample_player["first"], "last_name": sample_player["last"],
            "opponent_team": "Fictional Team",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_compare_same_player_twice(self, client, sample_player):
        resp = client.post("/compare", data={
            "team_a": sample_player["team"], "first_name_a": sample_player["first"], "last_name_a": sample_player["last"],
            "team_b": sample_player["team"], "first_name_b": sample_player["first"], "last_name_b": sample_player["last"],
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_long_input_does_not_crash(self, client):
        resp = client.post("/", data={
            "team": "A" * 5000, "first_name": "B" * 5000, "last_name": "C" * 5000,
        })
        assert resp.status_code in (200, 413)

    def test_oversized_body_is_rejected(self, client):
        resp = client.post("/", data={"team": "X" * 2_000_000, "first_name": "a", "last_name": "b"})
        assert resp.status_code in (200, 413)
        assert_no_leak(resp)

    def test_null_bytes_in_input(self, client):
        resp = client.post("/", data={
            "team": "Los Angeles Lakers", "first_name": "Le\x00Bron", "last_name": "James",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)

    def test_unicode_input(self, client):
        resp = client.post("/", data={
            "team": "Los Angeles Lakers", "first_name": "Lê​Brön", "last_name": "Jämes",
        })
        assert resp.status_code == 200
        assert_no_leak(resp)


# ---------------------------------------------------------------------------
# Security hardening
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    @pytest.mark.parametrize("route", ROUTES)
    def test_security_headers_present(self, client, route):
        resp = client.get(route)
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") in ("DENY", "SAMEORIGIN")
