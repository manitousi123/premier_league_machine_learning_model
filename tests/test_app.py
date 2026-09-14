"""The app is a thin shell: it loads a snapshot, serves it as JSON, and runs
the weekly refresh in the background. Everything here uses a synthetic
snapshot, so no test touches the disk or the network.
"""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import pytest

from plfootball import app as webapp
from plfootball import model, plea, track
from tests.helpers import CLUBS, calls_from, season_fixtures, tiny_results

AHEAD = [
    ("2026-09-12", "15:00", CLUBS[0], CLUBS[2], 0.62, 0.17),
    ("2026-09-12", "15:00", CLUBS[1], CLUBS[3], 0.41, 0.31),
]


def _snapshot(table, log_path, results=None) -> webapp.Snapshot:
    results = tiny_results() if results is None else results
    fitted = model.fit(table)
    played = {
        n + 1: [(d, h, a) for d, h, a in block[["date", "home_team", "away_team"]].itertuples(index=False)]
        for n, (_, block) in enumerate(results.groupby("date"))
    }
    ahead = {
        5: [("2026-09-12", CLUBS[0], CLUBS[2]), ("2026-09-12", CLUBS[1], CLUBS[3])],
        6: [("2026-09-19", CLUBS[2], CLUBS[0]), ("2026-09-19", CLUBS[3], CLUBS[1])],
    }
    return webapp.Snapshot(
        results=results,
        history=plea.run(results),
        fitted=fitted,
        coefficients=model.coefficients(fitted),
        calls=calls_from(AHEAD),
        log=track.load(log_path),
        backtest_scored=None,
        band_rates={},
        season_fixtures=season_fixtures({**played, **ahead}),
    )


def _wait(refresh: webapp.Refresh, seconds: float = 5.0) -> None:
    deadline = time.time() + seconds
    while refresh.running and time.time() < deadline:
        time.sleep(0.01)
    assert not refresh.running, "refresh did not finish"


@pytest.fixture
def harness(table, tmp_path):
    snap = _snapshot(table, tmp_path / "log.csv")
    store = webapp.Store(loader=lambda: snap)
    refresh = webapp.Refresh(store, runner=lambda: "ran")
    client = webapp.create_app(store, refresh).test_client()
    return client, store, refresh


def test_the_page_is_served(harness):
    client, _, _ = harness
    response = client.get("/")
    assert response.status_code == 200
    assert b"Premier League Match Predictor" in response.data
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_state_has_every_section(harness):
    client, _, _ = harness
    state = client.get("/api/state").get_json()
    assert set(state) >= {
        "freshness", "week", "record", "model", "rounds", "current_round", "versions", "refresh",
    }
    assert state["current_round"] == 5
    assert state["week"]["label"] == "Matchweek 5"
    assert [r["round"] for r in state["rounds"]] == [5, 6]
    assert state["refresh"] == {"running": False, "notice": None}


def test_state_is_strict_json(harness):
    """A NaN in the payload is not JSON and the browser would reject the whole page."""
    client, _, _ = harness
    raw = client.get("/api/state").data.decode()

    def reject(constant):
        pytest.fail(f"{constant} in the state payload")

    json.loads(raw, parse_constant=reject)


def test_preview_builds_calls_for_a_round_ahead(harness):
    client, _, _ = harness
    preview = client.get("/api/preview/6").get_json()
    assert preview["preview"] is True
    assert preview["label"] == "Matchweek 6"
    assert preview["n"] == 2
    assert all(not f["logged"] for f in preview["committed"] + preview["weak"])


def test_preview_refuses_what_it_cannot_build(harness):
    client, _, _ = harness
    assert client.get("/api/preview/1").status_code == 404   # already played
    assert client.get("/api/preview/99").status_code == 404  # no such round


def test_preview_without_a_fixture_list_says_why(table, tmp_path):
    snap = _snapshot(table, tmp_path / "log.csv")
    snap.season_fixtures = None
    snap.season_error = "unknown club 'Wrexham'"
    client = webapp.create_app(webapp.Store(loader=lambda: snap)).test_client()

    response = client.get("/api/preview/6")
    assert response.status_code == 404
    assert "Wrexham" in response.get_json()["error"]
    assert client.get("/api/state").get_json()["rounds_error"] == "unknown club 'Wrexham'"


# --- refresh -------------------------------------------------------------


def test_refresh_runs_once_in_the_background(table, tmp_path):
    snap = _snapshot(table, tmp_path / "log.csv")
    store = webapp.Store(loader=lambda: snap)
    gate = threading.Event()

    def runner():
        gate.wait(5)
        return "ran"

    refresh = webapp.Refresh(store, runner=runner)
    client = webapp.create_app(store, refresh).test_client()

    assert client.post("/api/refresh").status_code == 202
    assert client.post("/api/refresh").status_code == 409  # already running
    assert client.get("/api/refresh").get_json()["running"] is True

    gate.set()
    _wait(refresh)
    status = client.get("/api/refresh").get_json()
    assert status["running"] is False
    assert status["notice"]["kind"] == "warn"           # nothing new arrived
    assert "Already current" in status["notice"]["text"]


def test_refresh_reports_new_results(table, tmp_path):
    before = tiny_results(n_rounds=3)
    after = tiny_results(n_rounds=4)
    snapshots = iter([
        _snapshot(table, tmp_path / "log.csv", before),
        _snapshot(table, tmp_path / "log.csv", after),
    ])
    store = webapp.Store(loader=lambda: next(snapshots))
    refresh = webapp.Refresh(store, runner=lambda: "ran")
    client = webapp.create_app(store, refresh).test_client()

    client.get("/api/state")  # loads the first snapshot
    client.post("/api/refresh")
    _wait(refresh)

    notice = client.get("/api/refresh").get_json()["notice"]
    assert notice["kind"] == "ok"
    assert "Ingested 2 new results" in notice["text"]
    assert client.get("/api/state").get_json()["freshness"]["date"] == "2026-09-05"


def test_a_failing_refresh_reaches_the_page(table, tmp_path):
    snap = _snapshot(table, tmp_path / "log.csv")
    store = webapp.Store(loader=lambda: snap)

    def broken():
        raise RuntimeError("build_dataset.py exited with 1")

    refresh = webapp.Refresh(store, runner=broken)
    client = webapp.create_app(store, refresh).test_client()
    client.post("/api/refresh")
    _wait(refresh)

    notice = client.get("/api/refresh").get_json()["notice"]
    assert notice["kind"] == "error"
    assert "build_dataset.py" in notice["text"]
    assert client.get("/api/state").status_code == 200  # still serving


def test_a_snapshot_without_calls_still_serves(table, tmp_path):
    snap = _snapshot(table, tmp_path / "log.csv")
    snap.calls = pd.DataFrame(columns=webapp._CALL_COLUMNS)
    client = webapp.create_app(webapp.Store(loader=lambda: snap)).test_client()
    state = client.get("/api/state").get_json()
    assert state["week"]["n"] == 0
    assert state["current_round"] is None
    assert state["week"]["label"] == "Next fixtures"
