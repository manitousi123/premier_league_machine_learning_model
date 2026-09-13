"""A prediction row has to be built the same way a training row was.

That is the only thing really being tested here. Everything else in the project
fails loudly; this fails silently — assemble the row slightly differently and
the model is shown numbers that do not mean what it learned they meant, with no
error and no symptom beyond predictions that are quietly worse.

So the test that matters is `test_a_rebuilt_row_matches_the_training_table`,
which replays real matchweeks as though they had not happened yet and demands
the numbers come out bit-for-bit identical. The rest guard the plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plfootball import config, features, ingest, model, plea, predict

CLUBS = ["Arsenal", "Chelsea", "Everton", "Fulham"]


def _results(n_rounds: int = 4, season: int = 2021) -> pd.DataFrame:
    """A tiny but complete results frame: every club plays every round."""
    rng = np.random.default_rng(0)
    rows = []
    for week in range(n_rounds):
        date = pd.Timestamp("2020-08-01") + pd.Timedelta(days=7 * week)
        for home, away in ((CLUBS[0], CLUBS[1]), (CLUBS[2], CLUBS[3])):
            if week % 2:
                home, away = away, home
            rows.append({
                "date": date, "season": season, "kickoff": "15:00",
                "home_team": home, "away_team": away,
                "home_goals": int(rng.integers(0, 4)), "away_goals": int(rng.integers(0, 4)),
                "home_shots": int(rng.integers(5, 20)), "away_shots": int(rng.integers(5, 20)),
                "home_sot": int(rng.integers(1, 9)), "away_sot": int(rng.integers(1, 9)),
                "home_corners": int(rng.integers(0, 12)), "away_corners": int(rng.integers(0, 12)),
            })
    return pd.DataFrame(rows)


def _fixtures(date="2020-09-05", pairs=((CLUBS[0], CLUBS[2]),)) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp(date), "kickoff": "15:00", "home_team": h, "away_team": a}
        for h, a in pairs
    ])


@pytest.fixture
def built():
    results = _results()
    return predict.build_rows(_fixtures(), results, plea.run(results)), results


# --- the shape of what comes out -----------------------------------------


def test_every_fixture_becomes_two_rows(built):
    rows, _ = built
    assert len(rows) == 2
    assert set(rows["is_home"]) == {0, 1}


def test_the_two_rows_are_mirror_images(built):
    """A difference seen from the other side is the same difference negated."""
    rows, _ = built
    home = rows[rows.is_home == 1].iloc[0]
    away = rows[rows.is_home == 0].iloc[0]

    assert home["team"] == away["opponent"]
    assert home["sot_gap"] == pytest.approx(-away["sot_gap"])
    assert home["shots_gap"] == pytest.approx(-away["shots_gap"])
    assert home["elo_gap"] == pytest.approx(-away["elo_gap"])
    assert home["elo_expected"] + away["elo_expected"] == pytest.approx(1.0)


def test_the_home_side_is_favoured_between_equals():
    """Two clubs on identical ratings: the home bonus has to show up."""
    results = _results()
    rows = predict.build_rows(_fixtures(), results, plea.run(results))
    home = rows[rows.is_home == 1].iloc[0]
    assert home["elo_expected"] > 0.5


def test_the_rows_can_be_scored(built):
    """Whatever else, the model has to accept them."""
    rows, _ = built
    matrix = model.design_matrix(rows)
    assert list(matrix.columns) == model.FEATURES
    assert matrix.notna().all().all()


def test_a_club_with_no_history_gets_the_league_average_prior():
    results = _results()
    fixtures = _fixtures(pairs=[("Arsenal", "Luton Town")])
    rows = predict.build_rows(fixtures, results, plea.run(results))
    newcomer = rows[rows.team == "Luton Town"].iloc[0]
    assert newcomer["form_sot"] == pytest.approx(config.LEAGUE_AVERAGE_FORM["sot"])
    assert newcomer["form_shots"] == pytest.approx(config.LEAGUE_AVERAGE_FORM["shots"])


def test_a_club_with_no_history_starts_on_the_promoted_rating():
    results = _results()
    fixtures = _fixtures(pairs=[("Arsenal", "Luton Town")])
    rows = predict.build_rows(fixtures, results, plea.run(results))
    assert rows[rows.team == "Luton Town"].iloc[0]["own_elo"] == plea.DEFAULT.promoted_elo


def test_a_fixture_next_season_gets_the_summer_pull_back():
    results = _results()
    history = plea.run(results)
    same = predict.build_rows(_fixtures("2020-09-05"), results, history)
    next_year = predict.build_rows(_fixtures("2021-09-05"), results, history)

    arsenal_now = same[same.team == "Arsenal"].iloc[0]["own_elo"]
    arsenal_next = next_year[next_year.team == "Arsenal"].iloc[0]["own_elo"]
    expected = plea.DEFAULT.start_elo + plea.DEFAULT.carry_over * (
        arsenal_now - plea.DEFAULT.start_elo
    )
    assert arsenal_next == pytest.approx(expected)


# --- the fixture list ----------------------------------------------------


def test_already_played_fixtures_are_dropped():
    results = _results()
    stale = results[["date", "home_team", "away_team"]].head(2).assign(kickoff="15:00")
    mixed = pd.concat([stale, _fixtures()], ignore_index=True)
    assert len(predict.drop_already_played(mixed, results)) == 1


def test_download_fixtures_filters_to_the_premier_league(monkeypatch):
    csv = (
        b"Div,Date,Time,HomeTeam,AwayTeam\n"
        b"E0,12/09/2026,15:00,Arsenal,Chelsea\n"
        b"E1,12/09/2026,15:00,Millwall,Watford\n"
        b"E0,13/09/2026,14:00,Man City,Everton\n"
    )

    class Response:
        content = csv
        def raise_for_status(self):
            return None

    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: Response())
    fixtures = predict.download_fixtures()

    assert len(fixtures) == 2
    assert list(fixtures["home_team"]) == ["Arsenal", "Manchester City"]  # canonicalised


def test_an_empty_division_raises_rather_than_returning_nothing(monkeypatch):
    class Response:
        content = b"Div,Date,HomeTeam,AwayTeam\nE1,12/09/2026,Millwall,Watford\n"
        def raise_for_status(self):
            return None

    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: Response())
    with pytest.raises(predict.NoFixtures):
        predict.download_fixtures()


# --- the verdict ---------------------------------------------------------


def _scored(p_home: float, p_away: float) -> pd.DataFrame:
    rows = predict.build_rows(_fixtures(), _results(), plea.run(_results()))
    order = [p_home if is_home else p_away for is_home in rows["is_home"]]
    return predict.verdicts(rows, pd.Series(order, index=rows.index))


@pytest.mark.parametrize(
    ("p_home", "p_away", "expected"),
    [
        (0.70, 0.15, "HOME will win"),
        (0.15, 0.70, "AWAY will win"),
        (0.35, 0.30, "no confident call"),
    ],
)
def test_the_verdict_follows_the_probabilities(p_home, p_away, expected):
    assert _scored(p_home, p_away).iloc[0]["verdict"] == expected


def test_the_three_outcomes_add_to_one():
    row = _scored(0.45, 0.30).iloc[0]
    assert row["p_home"] + row["p_away"] + row["p_draw"] == pytest.approx(1.0)


def test_confidence_is_the_better_backed_side():
    assert _scored(0.22, 0.61).iloc[0]["confidence"] == pytest.approx(0.61)


def test_a_fixture_missing_its_other_half_is_noticed():
    rows = predict.build_rows(_fixtures(), _results(), plea.run(_results()))
    orphan = rows[rows.is_home == 1]
    with pytest.raises(ValueError, match="both perspectives"):
        predict.verdicts(orphan, pd.Series([0.5], index=orphan.index))


# --- the one that matters ------------------------------------------------


@pytest.mark.skipif(
    not (config.RAW / f"E0_{ingest.season_code(config.PLEA_START)}.csv").exists(),
    reason="no downloaded seasons — run scripts/build_dataset.py",
)
@pytest.mark.parametrize(
    "day", ["2015-11-07", "2019-03-09", "2023-12-26", "2025-04-19"]
)
def test_a_rebuilt_row_matches_the_training_table(day):
    """Replay a real matchweek as though it had not been played yet.

    Hide everything from that day onwards, build prediction rows for the
    fixtures, and compare against what `build_features` produced for the same
    matches with the benefit of hindsight. Every number the model reads has to
    be identical — not close, identical. Anything else means the two paths have
    drifted and the live predictions are subtly wrong.
    """
    day = pd.Timestamp(day)
    results = ingest.load_results()
    table = features.build_features(results, plea.run(results))

    before = results[results.date < day]
    fixtures = results[results.date == day][["date", "home_team", "away_team"]].assign(
        kickoff=pd.NA
    )
    rebuilt = predict.build_rows(fixtures, before, plea.run(before))

    truth = table[table.date == day]
    joined = rebuilt.merge(truth, on=["date", "team", "opponent"], suffixes=("_new", "_old"))
    assert len(joined) == len(rebuilt) > 0

    for column in model.FEATURES:
        assert np.array_equal(joined[f"{column}_new"], joined[f"{column}_old"]), column
