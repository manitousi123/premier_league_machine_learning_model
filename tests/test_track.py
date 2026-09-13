"""The log is the only file here that cannot be rebuilt, so it is the one
place where a bug is permanent.

Two kinds of mistake matter. Losing a prediction is bad. *Gaining* one — a
forecast that quietly appears after the match it forecasts — is worse, because
it does not look like an error, it looks like a good week. Most of these tests
are about the second kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plfootball import track


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "log.csv"


def _calls(fixtures) -> pd.DataFrame:
    """Fixtures as the predictor would hand them over.

    Each entry is (date, home, away, p_home, p_away).
    """
    rows = []
    for date, home, away, p_home, p_away in fixtures:
        rows.append({
            "date": pd.Timestamp(date), "kickoff": "15:00",
            "home_team": home, "away_team": away,
            "p_home": p_home, "p_away": p_away, "p_draw": 1 - p_home - p_away,
            "home_call": "WILL win" if p_home >= 0.5 else "WILL NOT win",
            "away_call": "WILL win" if p_away >= 0.5 else "WILL NOT win",
            "verdict": (
                "NO CALL - both sides backed" if p_home >= 0.5 and p_away >= 0.5
                else "HOME will win" if p_home >= 0.5
                else "AWAY will win" if p_away >= 0.5
                else "HOME will not win"
            ),
            "confidence": max(p_home, p_away),
        })
    return pd.DataFrame(rows)


def _record(calls, path, now="2026-01-01"):
    return track.record(
        calls, results_current_to="2025-12-28",
        model_version="v2", plea_version="v2", path=path, now=now,
    )


def _results(fixtures) -> pd.DataFrame:
    """(date, home, away, home_goals, away_goals) as the results table holds them."""
    return pd.DataFrame([
        {"date": pd.Timestamp(d), "home_team": h, "away_team": a,
         "home_goals": hg, "away_goals": ag}
        for d, h, a, hg, ag in fixtures
    ])


# --- the basics ----------------------------------------------------------


def test_an_absent_log_reads_as_empty(log_path):
    log = track.load(log_path)
    assert log.empty
    assert list(log.columns) == track.COLUMNS


def test_a_recorded_prediction_survives_the_round_trip(log_path):
    calls = _calls([("2026-01-03", "Arsenal", "Chelsea", 0.62, 0.18)])
    _record(calls, log_path)

    back = track.load(log_path)
    assert len(back) == 1
    row = back.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["p_home"] == pytest.approx(0.62)
    assert row["verdict"] == "HOME will win"
    assert row["model_version"] == "v2"
    assert pd.isna(row["outcome"])


def test_the_log_remembers_how_fresh_the_data_was(log_path):
    """A bad week is much easier to explain if you know the ratings were stale."""
    _record(_calls([("2026-01-03", "Arsenal", "Chelsea", 0.62, 0.18)]), log_path)
    assert track.load(log_path).iloc[0]["results_current_to"] == pd.Timestamp("2025-12-28")


# --- refusing to gain a prediction ---------------------------------------


def test_a_fixture_that_already_kicked_off_is_not_logged(log_path):
    """The fixture list runs ahead of the results file, so this happens weekly."""
    calls = _calls([
        ("2025-12-30", "Arsenal", "Chelsea", 0.62, 0.18),   # yesterday
        ("2026-01-03", "Everton", "Fulham", 0.41, 0.30),    # still to come
    ])
    log = _record(calls, log_path, now="2026-01-01")

    assert len(log) == 1
    assert log.iloc[0]["home_team"] == "Everton"


def test_a_match_kicking_off_later_today_still_counts(log_path):
    """Kickoffs run to 8pm and the log works in whole days. Same-day is allowed."""
    calls = _calls([("2026-01-01", "Arsenal", "Chelsea", 0.62, 0.18)])
    assert len(_record(calls, log_path, now="2026-01-01")) == 1


def test_predicting_a_settled_match_is_refused(log_path):
    calls = _calls([("2026-01-03", "Arsenal", "Chelsea", 0.62, 0.18)])
    _record(calls, log_path)
    track.settle(_results([("2026-01-03", "Arsenal", "Chelsea", 2, 0)]), log_path)

    with pytest.raises(track.AlreadyPlayed, match="Arsenal v Chelsea"):
        _record(calls, log_path, now="2026-01-10")


def test_re_predicting_an_unplayed_fixture_replaces_rather_than_doubles(log_path):
    """Thursday's call with fresher form supersedes Wednesday's. It is a better
    answer to the same question, not a second question."""
    _record(_calls([("2026-01-10", "Arsenal", "Chelsea", 0.62, 0.18)]), log_path)
    _record(_calls([("2026-01-10", "Arsenal", "Chelsea", 0.55, 0.22)]), log_path,
            now="2026-01-08")

    log = track.load(log_path)
    assert len(log) == 1
    assert log.iloc[0]["p_home"] == pytest.approx(0.55)


# --- settling ------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_home", "p_away", "home_goals", "away_goals", "expected"),
    [
        (0.62, 0.18, 2, 0, True),    # called home, home won
        (0.62, 0.18, 0, 2, False),   # called home, away won
        (0.18, 0.62, 0, 2, True),    # called away, away won
        (0.41, 0.30, 1, 1, True),    # called "home will not win", it drew
        (0.41, 0.30, 3, 0, False),   # called "home will not win", home won
        (0.41, 0.30, 0, 3, True),    # called "home will not win", away won
    ],
)
def test_settling_scores_the_verdict(log_path, p_home, p_away, home_goals, away_goals, expected):
    _record(_calls([("2026-01-03", "Arsenal", "Chelsea", p_home, p_away)]), log_path)
    settled = track.settle(
        _results([("2026-01-03", "Arsenal", "Chelsea", home_goals, away_goals)]), log_path
    )
    assert bool(settled.iloc[0]["correct"]) is expected


def test_a_contradiction_is_neither_right_nor_wrong(log_path):
    """It claims nothing, so counting it as a miss would flatter never flagging one."""
    _record(_calls([("2026-01-03", "Arsenal", "Chelsea", 0.60, 0.55)]), log_path)
    settled = track.settle(
        _results([("2026-01-03", "Arsenal", "Chelsea", 2, 0)]), log_path
    )
    assert settled.iloc[0]["verdict"] == "NO CALL - both sides backed"
    assert pd.isna(settled.iloc[0]["correct"])


def test_an_unplayed_fixture_stays_unsettled(log_path):
    _record(_calls([("2026-01-03", "Arsenal", "Chelsea", 0.62, 0.18)]), log_path)
    settled = track.settle(_results([("2026-01-03", "Everton", "Fulham", 1, 0)]), log_path)
    assert pd.isna(settled.iloc[0]["outcome"])


def test_settling_twice_cannot_revise_history(log_path):
    """The guard against a later data correction quietly rewriting the record."""
    _record(_calls([("2026-01-03", "Arsenal", "Chelsea", 0.62, 0.18)]), log_path)
    first = track.settle(_results([("2026-01-03", "Arsenal", "Chelsea", 2, 0)]), log_path)

    rewritten = track.settle(
        _results([("2026-01-03", "Arsenal", "Chelsea", 0, 5)]), log_path
    )
    assert rewritten.iloc[0]["outcome"] == first.iloc[0]["outcome"] == "home win"
    assert bool(rewritten.iloc[0]["correct"]) is True


def test_settling_an_empty_log_does_nothing(log_path):
    assert track.settle(_results([("2026-01-03", "Arsenal", "Chelsea", 2, 0)]), log_path).empty


# --- reading the record back ---------------------------------------------


def _settled_week(log_path):
    calls = _calls([
        ("2026-01-03", "Arsenal", "Chelsea", 0.75, 0.10),    # 70%+,  right
        ("2026-01-03", "Everton", "Fulham", 0.64, 0.16),     # 60-70%, wrong
        ("2026-01-03", "Brentford", "Burnley", 0.57, 0.20),  # 55-60%, right
        ("2026-01-03", "Leeds United", "Luton Town", 0.40, 0.31),  # neither, right
    ])
    _record(calls, log_path)
    return track.settle(_results([
        ("2026-01-03", "Arsenal", "Chelsea", 3, 0),
        ("2026-01-03", "Everton", "Fulham", 0, 1),
        ("2026-01-03", "Brentford", "Burnley", 2, 1),
        ("2026-01-03", "Leeds United", "Luton Town", 1, 1),
    ]), log_path)


def test_the_track_record_splits_by_confidence(log_path):
    _settled_week(log_path)
    bands = track.track_record(path=log_path).set_index("band")

    assert bands.loc["70%+", "correct"] == 1
    assert bands.loc["60-70%", "correct"] == 0
    assert bands.loc["55-60%", "correct"] == 1
    assert bands.loc["neither backed", "correct"] == 1
    assert bands["predictions"].sum() == 4


def test_every_band_appears_even_when_empty(log_path):
    """A UI reading this should not have rows vanish from under it."""
    _record(_calls([("2026-01-03", "Arsenal", "Chelsea", 0.75, 0.10)]), log_path)
    track.settle(_results([("2026-01-03", "Arsenal", "Chelsea", 3, 0)]), log_path)
    assert list(track.track_record(path=log_path)["band"]) == track.BAND_LABELS


def test_the_summary_counts_what_is_and_is_not_settled(log_path):
    _settled_week(log_path)
    _record(_calls([("2026-02-01", "Arsenal", "Everton", 0.6, 0.2)]), log_path)

    facts = track.summary(path=log_path)
    assert facts["logged"] == 5
    assert facts["settled"] == 4
    assert facts["awaiting_result"] == 1
    assert facts["verdict_accuracy"] == pytest.approx(0.75)


def test_the_summary_scores_both_perspectives(log_path):
    """Each fixture contributes two rows, exactly as the walk-forward scored it,
    so the live numbers can be read against the backtest's."""
    _settled_week(log_path)
    facts = track.summary(path=log_path)

    log = track.load(log_path)
    probability = np.concatenate([log["p_home"], log["p_away"]])
    actual = np.concatenate([
        (log["outcome"] == "home win"), (log["outcome"] == "away win")
    ]).astype(int)
    assert facts["brier"] == pytest.approx(((probability - actual) ** 2).mean())


def test_an_empty_log_summarises_without_blowing_up(log_path):
    facts = track.summary(path=log_path)
    assert facts["logged"] == 0
    assert "verdict_accuracy" not in facts
    assert track.track_record(path=log_path).empty
