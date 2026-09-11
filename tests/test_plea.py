import pandas as pd
import pytest

from plfootball import plea
from plfootball.plea import PleaParams, expected_score, margin_multiplier


def _match(date, season, home, away, hg, ag):
    return {
        "date": pd.Timestamp(date),
        "kickoff": "15:00",
        "season": season,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
    }


def test_equal_ratings_expect_half_without_home_advantage():
    assert expected_score(1500, 1500, home_adv=0) == pytest.approx(0.5)


def test_home_advantage_tilts_the_expectation():
    assert expected_score(1500, 1500, home_adv=65) > 0.5


def test_four_hundred_points_is_ten_to_one():
    assert expected_score(1900, 1500, home_adv=0) == pytest.approx(10 / 11, abs=1e-6)


@pytest.mark.parametrize("gd,expected", [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (3, 1.75)])
def test_margin_multiplier(gd, expected):
    assert margin_multiplier(gd) == pytest.approx(expected)


def test_ratings_are_zero_sum():
    """Whatever one club gains, the other loses. Total rating must not drift."""
    results = pd.DataFrame([_match("2020-08-01", 2021, "Arsenal", "Chelsea", 3, 0)])
    history = plea.run(results)
    before = history["elo_before"].sum()
    after = history["elo_after"].sum()
    assert after == pytest.approx(before)


def test_winner_gains_loser_loses():
    results = pd.DataFrame([_match("2020-08-01", 2021, "Arsenal", "Chelsea", 3, 0)])
    history = plea.run(results).set_index("team")
    assert history.loc["Arsenal", "elo_after"] > history.loc["Arsenal", "elo_before"]
    assert history.loc["Chelsea", "elo_after"] < history.loc["Chelsea", "elo_before"]


def test_bigger_win_moves_ratings_further():
    narrow = plea.run(pd.DataFrame([_match("2020-08-01", 2021, "Arsenal", "Chelsea", 1, 0)]))
    wide = plea.run(pd.DataFrame([_match("2020-08-01", 2021, "Arsenal", "Chelsea", 4, 0)]))
    gain = lambda h: h.set_index("team").loc["Arsenal", "elo_after"] - 1500
    assert gain(wide) > gain(narrow)


def test_beating_a_stronger_club_is_worth_more():
    """Chelsea builds a rating on Everton, then Arsenal beats each of them.

    Both Arsenal wins are 1-0 at home, so the margin multiplier and home
    advantage are identical and the only thing that differs is who they beat.
    """
    rows = [_match(f"2020-08-{d:02d}", 2021, "Chelsea", "Everton", 3, 0) for d in (1, 2, 3)]
    rows.append(_match("2020-08-10", 2021, "Arsenal", "Everton", 1, 0))
    rows.append(_match("2020-08-20", 2021, "Arsenal", "Chelsea", 1, 0))
    history = plea.run(pd.DataFrame(rows))

    last = history[history["date"] == pd.Timestamp("2020-08-20")].set_index("team")
    beat_strong = last.loc["Arsenal", "elo_after"] - last.loc["Arsenal", "elo_before"]

    weak = history[
        (history["date"] == pd.Timestamp("2020-08-10")) & (history["team"] == "Arsenal")
    ].iloc[0]
    beat_weak = weak["elo_after"] - weak["elo_before"]

    assert beat_strong > beat_weak


def test_promoted_club_starts_on_promoted_elo():
    params = PleaParams()
    rows = [_match("2020-08-01", 2021, "Arsenal", "Chelsea", 1, 0),
            _match("2021-08-01", 2022, "Arsenal", "Brentford", 1, 0)]
    history = plea.run(pd.DataFrame(rows), params)

    brentford = history[history["team"] == "Brentford"].iloc[0]
    assert brentford["elo_before"] == pytest.approx(params.promoted_elo)


def test_carry_over_pulls_towards_the_mean():
    params = PleaParams()
    rows = [_match("2020-08-01", 2021, "Arsenal", "Chelsea", 4, 0),
            _match("2021-08-01", 2022, "Arsenal", "Chelsea", 1, 0)]
    history = plea.run(pd.DataFrame(rows), params)

    season1 = history[(history.season == 2021) & (history.team == "Arsenal")].iloc[0]
    season2 = history[(history.season == 2022) & (history.team == "Arsenal")].iloc[0]

    expected = params.start_elo + params.carry_over * (season1["elo_after"] - params.start_elo)
    assert season2["elo_before"] == pytest.approx(expected)
    assert season2["elo_before"] < season1["elo_after"]  # pulled back toward 1500


def test_history_has_two_rows_per_match():
    rows = [_match("2020-08-01", 2021, "Arsenal", "Chelsea", 1, 0),
            _match("2020-08-08", 2021, "Everton", "Fulham", 2, 2)]
    history = plea.run(pd.DataFrame(rows))
    assert len(history) == 4


def test_elo_gap_is_symmetric():
    results = pd.DataFrame([_match("2020-08-01", 2021, "Arsenal", "Chelsea", 1, 0)])
    history = plea.run(results)
    assert history["elo_gap"].sum() == pytest.approx(0.0)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        plea.run(pd.DataFrame({"date": [], "season": []}))


def test_ratings_as_of_respects_the_cutoff():
    rows = [_match("2020-08-01", 2021, "Arsenal", "Chelsea", 1, 0),
            _match("2020-09-01", 2021, "Arsenal", "Chelsea", 5, 0)]
    history = plea.run(pd.DataFrame(rows))

    early = plea.ratings_as_of(history, "2020-08-15").set_index("team")
    late = plea.ratings_as_of(history, "2020-09-15").set_index("team")

    assert early.loc["Arsenal", "elo"] < late.loc["Arsenal", "elo"]
    assert early.loc["Arsenal", "played"] == 1
