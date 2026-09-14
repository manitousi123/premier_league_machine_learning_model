"""Every number on the page comes from here, so every number gets a test.

The page itself only formats what it is given - it cannot make a number up -
which is why the tests live against these functions rather than a browser.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from plfootball import dashboard, fixtures, model, plea, track
from tests.helpers import CLUBS, calls_from, season_fixtures, tiny_results

WEEK = [
    ("2026-09-12", "15:00", "Chelsea", "Hull City", 0.63, 0.16),
    ("2026-09-12", "20:00", "Sunderland", "Arsenal", 0.18, 0.59),
    ("2026-09-13", "14:00", "Manchester United", "Manchester City", 0.36, 0.35),
    ("2026-09-14", "20:00", "Leeds United", "Newcastle United", 0.40, 0.32),
]


@pytest.fixture
def calls():
    return calls_from(WEEK)


@pytest.fixture
def results():
    return tiny_results()


def _results_to(*dates) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(list(dates)), "home_team": "A", "away_team": "B"})


def _record(rows, path, now="2026-09-10"):
    return track.record(
        calls_from(rows), results_current_to="2026-09-06",
        model_version="v2", plea_version="v2", path=path, now=now,
    )


def _settle(played, path):
    return track.settle(pd.DataFrame([
        {"date": pd.Timestamp(d), "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag}
        for d, h, a, hg, ag in played
    ]), path)


# --- helpers -------------------------------------------------------------


def test_day_labels_have_no_leading_zero():
    assert dashboard.day_label("2026-09-06") == "Sun 6 Sep"


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (["2026-09-12"], "Sat 12 Sep"),
        (["2026-09-12", "2026-09-14"], "Sat 12 – Mon 14 Sep"),
        (["2026-08-30", "2026-09-01"], "Sun 30 Aug – Tue 1 Sep"),
        ([], ""),
    ],
)
def test_date_span(dates, expected):
    assert dashboard.date_span(dates) == expected


def test_stated_is_what_each_verdict_actually_claims(calls):
    claimed = dashboard.stated(calls).round(2).tolist()
    # ordered by date/kickoff: Chelsea (home backed), Sunderland (away backed),
    # Man Utd (neither), Leeds (neither)
    assert claimed == [0.63, 0.59, 0.64, 0.60]


def test_a_contradiction_claims_nothing():
    both = calls_from([("2026-09-12", "15:00", "Arsenal", "Chelsea", 0.6, 0.55)])
    assert dashboard.stated(both).isna().all()


def test_bands_match_the_log_s_own():
    bands = dashboard.band_of(pd.Series([0.63, 0.40, 0.59, 0.51, 0.75])).tolist()
    assert bands == ["60-70%", "neither backed", "55-60%", "50-55%", "70%+"]


# --- freshness -----------------------------------------------------------


def test_fresh_when_nothing_has_kicked_off_since():
    """Eight days old, but a fortnight with no football is not stale."""
    played = season_fixtures({5: [("2026-09-19", "A", "B")]})
    fresh = dashboard.freshness(_results_to("2026-09-06"), played, now="2026-09-14")
    assert fresh["stale"] is False
    assert fresh["age_days"] == 8
    assert fresh["label"] == "Sun 6 Sep"


def test_stale_once_a_fixture_has_kicked_off_since():
    played = season_fixtures({4: [("2026-09-12", "A", "B")]})
    assert dashboard.freshness(_results_to("2026-09-06"), played, now="2026-09-14")["stale"]


def test_a_fixture_today_does_not_count_as_kicked_off():
    played = season_fixtures({4: [("2026-09-14", "A", "B")]})
    assert not dashboard.freshness(_results_to("2026-09-06"), played, now="2026-09-14")["stale"]


def test_without_a_fixture_list_the_clock_decides():
    assert not dashboard.freshness(_results_to("2026-09-06"), now="2026-09-08")["stale"]
    assert dashboard.freshness(_results_to("2026-09-06"), now="2026-09-14")["stale"]


@pytest.mark.parametrize(
    ("now", "text"),
    [("2026-09-06", "updated today"), ("2026-09-07", "1 day old"), ("2026-09-09", "3 days old")],
)
def test_age_reads_naturally(now, text):
    assert dashboard.freshness(_results_to("2026-09-06"), now=now)["age"] == text


# --- this week -----------------------------------------------------------


def test_the_week_is_sorted_into_calls_and_weak_claims(calls):
    week = dashboard.this_week(calls, label="Matchweek 4", now="2026-09-10")
    assert week["n"] == 4
    assert [f["home"] for f in week["committed"]] == ["Chelsea", "Sunderland"]
    assert [f["home"] for f in week["weak"]] == ["Manchester United", "Leeds United"]
    assert week["contradictions"] == []
    assert (week["n_calls"], week["n_home"], week["n_away"], week["n_weak"]) == (2, 1, 1, 2)
    assert week["dates"] == "Sat 12 – Mon 14 Sep"
    assert week["threshold"] == 50


def test_percentages_are_whole_numbers(calls):
    week = dashboard.this_week(calls, label="x", now="2026-09-10")
    chelsea = week["committed"][0]
    assert (chelsea["p_home"], chelsea["p_away"], chelsea["p_draw"]) == (63, 16, 21)
    assert chelsea["confidence"] == 63
    assert chelsea["home_backed"] and not chelsea["away_backed"]


def test_each_call_carries_the_backtest_rate_for_its_band(calls):
    rates = {"60-70%": 0.636, "55-60%": 0.586, "neither backed": 0.61}
    week = dashboard.this_week(calls, rates, label="x", now="2026-09-10")
    assert week["committed"][0]["band"] == "60-70%"
    assert week["committed"][0]["band_rate"] == 64
    assert week["committed"][1]["band_rate"] == 59
    assert week["weak"][0]["band_rate"] == 61


def test_a_band_with_no_backtest_figure_is_none_not_zero(calls):
    week = dashboard.this_week(calls, {}, label="x", now="2026-09-10")
    assert week["committed"][0]["band_rate"] is None


def test_logged_and_kicked_off_are_marked(calls, tmp_path):
    log = _record(WEEK[2:], tmp_path / "log.csv", now="2026-09-13")
    week = dashboard.this_week(calls, label="x", log=log, now="2026-09-13")
    by_home = {f["home"]: f for f in week["committed"] + week["weak"]}
    assert by_home["Chelsea"]["kicked_off"] and not by_home["Chelsea"]["logged"]
    assert by_home["Manchester United"]["logged"]
    assert not by_home["Leeds United"]["kicked_off"]
    assert week["n_logged"] == 2


def test_an_empty_week_renders_as_nothing_rather_than_crashing():
    empty = pd.DataFrame(columns=[
        "date", "kickoff", "home_team", "away_team", "p_home", "p_away", "p_draw",
        "home_call", "away_call", "verdict", "confidence",
    ])
    week = dashboard.this_week(empty, label="Next fixtures")
    assert week["n"] == 0 and week["dates"] == "" and week["committed"] == []


def test_preview_calls_come_out_one_per_fixture(results, table):
    ahead = season_fixtures({5: [
        ("2026-09-12", CLUBS[0], CLUBS[2]), ("2026-09-12", CLUBS[1], CLUBS[3]),
    ]})
    calls = dashboard.preview_calls(ahead, results, plea.run(results), model.fit(table))
    assert len(calls) == 2
    assert {"p_home", "p_away", "verdict", "confidence"} <= set(calls.columns)
    assert ((calls["p_home"] > 0) & (calls["p_home"] < 1)).all()


# --- rounds --------------------------------------------------------------


def _season():
    return season_fixtures({
        4: [("2026-09-12", "Chelsea", "Hull City"), ("2026-09-12", "Sunderland", "Arsenal"),
            ("2026-09-13", "Manchester United", "Manchester City"),
            ("2026-09-14", "Leeds United", "Newcastle United")],
        5: [("2026-09-19", "Arsenal", "Chelsea")],
    })


def test_the_current_round_is_found_by_pairing(calls):
    assert dashboard.current_round(calls, _season()) == 4


def test_no_round_without_a_fixture_list_or_a_match(calls):
    assert dashboard.current_round(calls, None) is None
    other = season_fixtures({1: [("2026-09-12", "Everton", "Fulham")]})
    assert dashboard.current_round(calls, other) is None


def test_calls_from_another_season_are_not_matched(calls):
    last_year = season_fixtures({4: [("2025-09-12", "Chelsea", "Hull City")]})
    assert dashboard.current_round(calls, last_year) is None


def test_rounds_view_keeps_open_rounds_and_the_current_one():
    played = pd.DataFrame([
        {"date": pd.Timestamp("2026-09-12"), "season": 2027,
         "home_team": "Chelsea", "away_team": "Hull City"},
    ])
    table = fixtures.rounds(_season(), played)
    view = dashboard.rounds_view(table, current=4)
    assert [(r["round"], r["remaining"], r["current"]) for r in view] == [(4, 3, True), (5, 1, False)]
    assert view[0]["label"] == "Matchweek 4"
    assert view[0]["dates"] == "Sat 12 – Mon 14 Sep"

    finished = pd.DataFrame([
        {"date": pd.Timestamp("2026-09-12"), "season": 2027, "home_team": h, "away_team": a}
        for _, h, a in _season().query("round == 4")[["date", "home_team", "away_team"]].itertuples(index=False)
    ])
    view = dashboard.rounds_view(fixtures.rounds(_season(), finished), current=None)
    assert [r["round"] for r in view] == [5]


# --- the backtest, scored like the log -----------------------------------


def _backtest():
    """Three fixtures, two rows each, with known probabilities and outcomes."""
    recs = []
    for date, home, away, p_home, p_away, home_won, away_won in [
        ("2020-08-01", "A", "B", 0.70, 0.10, 1, 0),   # home backed, home won  -> right
        ("2020-08-01", "C", "D", 0.20, 0.60, 0, 0),   # away backed, draw      -> wrong
        ("2020-08-08", "A", "C", 0.40, 0.30, 0, 0),   # neither, draw          -> right
    ]:
        recs.append({"date": pd.Timestamp(date), "season": 2021, "team": home, "opponent": away,
                     "is_home": 1, "target": home_won, "p_model": p_home})
        recs.append({"date": pd.Timestamp(date), "season": 2021, "team": away, "opponent": home,
                     "is_home": 0, "target": away_won, "p_model": p_away})
    return pd.DataFrame(recs)


def test_backtest_rows_are_scored_exactly_like_the_log():
    scored = dashboard.backtest_calls(_backtest())
    assert len(scored) == 3
    assert scored["outcome"].tolist() == ["home win", "draw", "draw"]
    assert scored["verdict"].tolist() == ["HOME will win", "AWAY will win", "HOME will not win"]
    assert [bool(v) for v in scored["correct"]] == [True, False, True]


def test_band_rates_cover_every_band():
    rates = dashboard.band_rates(dashboard.backtest_calls(_backtest()))
    assert set(rates) == set(track.BAND_LABELS)
    assert rates["70%+"] == 1.0
    assert rates["60-70%"] == 0.0
    assert rates["neither backed"] == 1.0
    assert rates["50-55%"] is None
    assert dashboard.band_rates(None) == {}


# --- the record ----------------------------------------------------------


def test_an_empty_record_has_zeros_and_no_surprises(tmp_path):
    rec = dashboard.record(track.load(tmp_path / "log.csv"))
    assert (rec["logged"], rec["settled"], rec["awaiting"]) == (0, 0, 0)
    assert rec["verdict_accuracy"] is None
    assert rec["bands"] == [] and rec["seasons"] == []
    json.dumps(rec, allow_nan=False)


def test_a_settled_record_scores_and_calibrates(tmp_path):
    path = tmp_path / "log.csv"
    _record(WEEK, path)
    _settle([
        ("2026-09-12", "Chelsea", "Hull City", 2, 0),               # right, claimed 63
        ("2026-09-12", "Sunderland", "Arsenal", 1, 1),              # wrong, claimed 59
        ("2026-09-13", "Manchester United", "Manchester City", 0, 0),  # right, claimed 64
    ], path)
    rec = dashboard.record(track.load(path))

    assert (rec["logged"], rec["settled"], rec["awaiting"], rec["called"]) == (4, 3, 1, 3)
    assert rec["verdict_accuracy"] == 67
    assert rec["neither_backed"] == 1
    assert rec["contradictions"] == 0
    # actual - stated, in points: (100-63) + (0-59) + (100-64) = 14, over three
    assert rec["calibration_error"] == pytest.approx(4.7, abs=0.05)

    by_band = {b["band"]: b for b in rec["bands"]}
    assert set(by_band) == set(track.BAND_LABELS)
    assert by_band["60-70%"]["predictions"] == 1
    assert by_band["60-70%"]["stated"] == 63
    assert by_band["60-70%"]["hit_rate"] == 100
    assert by_band["55-60%"]["hit_rate"] == 0
    assert by_band["70%+"]["predictions"] == 0 and by_band["70%+"]["stated"] is None
    json.dumps(rec, allow_nan=False)


def test_the_backtest_sits_beside_the_live_numbers(tmp_path):
    rec = dashboard.record(
        track.load(tmp_path / "log.csv"), dashboard.backtest_calls(_backtest())
    )
    assert rec["backtest"]["fixtures"] == 3
    assert rec["backtest"]["verdict_accuracy"] == 67
    assert rec["backtest"]["contradictions"] == 0
    assert {b["band"] for b in rec["backtest"]["bands"]} == set(track.BAND_LABELS)


def test_weeks_are_split_on_gaps_and_labelled_by_round(tmp_path):
    path = tmp_path / "log.csv"
    _record(WEEK + [("2026-09-19", "15:00", "Arsenal", "Chelsea", 0.7, 0.1)], path)
    seasons = dashboard.weeks(track.load(path), _season())

    assert [s["label"] for s in seasons] == ["2026/27"]
    labels = [w["label"] for w in seasons[0]["weeks"]]
    assert labels == ["Matchweek 5", "Matchweek 4"]   # newest first
    assert [len(w["rows"]) for w in seasons[0]["weeks"]] == [1, 4]
    assert seasons[0]["weeks"][1]["dates"] == "Sat 12 – Mon 14 Sep"
    assert seasons[0]["weeks"][1]["awaiting"] == 4


def test_weeks_fall_back_to_a_date_label_without_a_fixture_list(tmp_path):
    path = tmp_path / "log.csv"
    _record(WEEK, path)
    assert dashboard.weeks(track.load(path))[0]["weeks"][0]["label"] == "Week of Sat 12 Sep"


def test_log_rows_say_how_each_was_scored(tmp_path):
    path = tmp_path / "log.csv"
    _record(WEEK, path)
    _settle([("2026-09-12", "Chelsea", "Hull City", 2, 0),
             ("2026-09-12", "Sunderland", "Arsenal", 1, 1)], path)
    rows = dashboard.weeks(track.load(path))[0]["weeks"][0]["rows"]
    by_home = {r["home"]: r for r in rows}
    assert by_home["Chelsea"]["scored"] == "correct"
    assert by_home["Chelsea"]["result"] == "2–0"
    assert by_home["Chelsea"]["short"] == "HOME"
    assert by_home["Sunderland"]["scored"] == "wrong"
    assert by_home["Manchester United"]["scored"] == "awaiting"
    assert by_home["Manchester United"]["short"] == "NOT HOME"
    assert by_home["Manchester United"]["result"] == ""


# --- the model -----------------------------------------------------------


def test_features_are_scaled_to_the_strongest(table):
    fitted = model.fit(table)
    view = dashboard.model_view(model.coefficients(fitted), plea.run(tiny_results()), "2026-09-06",
                                calls_from(WEEK))
    assert {f["column"] for f in view["features"]} == set(model.FEATURES)
    assert view["features"][0]["relative"] == 100
    assert all(0 <= f["relative"] <= 100 for f in view["features"])
    assert all(f["name"] and f["note"] for f in view["features"])


def test_the_grid_counts_this_week(table):
    view = dashboard.model_view(
        model.coefficients(model.fit(table)), plea.run(tiny_results()), "2026-09-06", calls_from(WEEK)
    )
    assert view["grid"] == {"n": 4, "home_win": 1, "away_win": 1, "weak": 2, "contradiction": 0}


def test_elo_table_ranks_and_measures_the_last_week(results):
    history = plea.run(results)
    latest = history["date"].max()
    table = dashboard.elo_table(history, latest)

    assert [c["rank"] for c in table] == [1, 2, 3, 4]
    assert table[0]["elo"] >= table[-1]["elo"]
    assert all(0 <= c["relative"] <= 100 for c in table)
    # every club played exactly once in the last seven days, so the movement
    # is that match's own rating change
    last = history[history["date"] == latest].set_index("team")
    for club in table:
        change = last.loc[club["team"], "elo_after"] - last.loc[club["team"], "elo_before"]
        assert club["delta"] == round(float(change))


def test_everything_is_strict_json(results, table, tmp_path):
    fitted = model.fit(table)
    history = plea.run(results)
    payload = {
        "freshness": dashboard.freshness(results, _season(), now="2026-09-10"),
        "week": dashboard.this_week(calls_from(WEEK), label="Matchweek 4"),
        "record": dashboard.record(track.load(tmp_path / "log.csv"),
                                   dashboard.backtest_calls(_backtest()), _season()),
        "model": dashboard.model_view(model.coefficients(fitted), history, results["date"].max(),
                                      calls_from(WEEK)),
        "rounds": dashboard.rounds_view(fixtures.rounds(_season(), results), 4),
    }
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text
    assert not any(isinstance(v, (np.integer, np.floating)) for v in payload["model"]["grid"].values())
