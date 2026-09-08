import pandas as pd
import pytest

from plfootball import config, plea
from plfootball.features import build_features


def _match(
    date,
    season,
    home,
    away,
    hg,
    ag,
    *,
    kickoff="15:00",
    home_shots=10,
    away_shots=10,
    home_sot=5,
    away_sot=5,
    home_corners=5,
    away_corners=5,
    home_yellows=1,
    away_yellows=1,
    home_reds=0,
    away_reds=0,
    referee="A Referee",
):
    result = "H" if hg > ag else ("A" if ag > hg else "D")
    return {
        "date": pd.Timestamp(date),
        "kickoff": kickoff,
        "season": season,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_sot": home_sot,
        "away_sot": away_sot,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "home_yellows": home_yellows,
        "away_yellows": away_yellows,
        "home_reds": home_reds,
        "away_reds": away_reds,
        "referee": referee,
    }


def _build(rows):
    """Build (results, history, features) the way the real pipeline would.

    ``history`` comes from the already-tested ``plea.run`` rather than being
    hand-rolled, so PLEA-derived columns are checked against real Elo output
    instead of a second, error-prone hand computation of the same maths.
    """
    results = pd.DataFrame(rows)
    history = plea.run(results)
    features = build_features(results, history)
    return results, history, features


def _prior_window_mean(values, window):
    """Pure-python mirror of a strictly-backward rolling mean, for hand-checking."""
    out = []
    for i in range(len(values)):
        prior = values[max(0, i - window) : i]
        out.append(sum(prior) / len(prior) if prior else None)
    return out


# --- shape ------------------------------------------------------------------


def test_shape_two_rows_per_match_sorted_by_date_then_team():
    rows = [
        _match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0),
        _match("2010-08-15", 2011, "Everton", "Fulham", 2, 2),
        _match("2010-08-21", 2011, "Chelsea", "Arsenal", 0, 3),
    ]
    _, _, features = _build(rows)

    assert len(features) == 6
    assert list(features.index) == list(range(6))

    key = list(zip(features["date"], features["team"]))
    assert key == sorted(key)

    assert features["team"].value_counts().to_dict() == {
        "Arsenal": 2,
        "Chelsea": 2,
        "Everton": 1,
        "Fulham": 1,
    }


# --- matchweek ----------------------------------------------------------------


def test_matchweek_counts_within_club_season():
    rows = [
        _match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0),  # home
        _match("2010-08-21", 2011, "Everton", "Arsenal", 1, 1),  # away
        _match("2010-08-28", 2011, "Arsenal", "Fulham", 0, 0),  # home
    ]
    _, _, features = _build(rows)

    arsenal = features[features["team"] == "Arsenal"].sort_values("date")
    assert list(arsenal["matchweek"]) == [1, 2, 3]


def test_matchweek_resets_each_season():
    rows = [
        _match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0),
        _match("2010-08-21", 2011, "Arsenal", "Everton", 1, 1),
        _match("2011-08-13", 2012, "Arsenal", "Chelsea", 2, 0),
    ]
    _, _, features = _build(rows)

    arsenal = features[features["team"] == "Arsenal"].sort_values("date")
    assert list(arsenal["matchweek"]) == [1, 2, 1]


# --- rest days ------------------------------------------------------------


def test_rest_days_since_own_previous_match():
    rows = [
        _match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0),
        _match("2010-08-21", 2011, "Arsenal", "Everton", 1, 1),  # 7 days later
        _match("2010-09-04", 2011, "Arsenal", "Fulham", 2, 0),  # 14 days later
    ]
    _, _, features = _build(rows)

    arsenal = features[features["team"] == "Arsenal"].sort_values("date").reset_index(drop=True)
    assert arsenal.loc[0, "rest_days"] == pytest.approx(config.MAX_REST_DAYS)  # no previous match -> fully rested
    assert arsenal.loc[1, "rest_days"] == pytest.approx(7)
    assert arsenal.loc[2, "rest_days"] == pytest.approx(14)


def test_rest_days_crosses_season_boundary_without_resetting():
    rows = [
        _match("2011-05-01", 2011, "Arsenal", "Chelsea", 1, 0),
        _match("2011-05-08", 2012, "Arsenal", "Everton", 2, 0),  # new season, 7 days later
    ]
    _, _, features = _build(rows)

    arsenal = features[features["team"] == "Arsenal"].sort_values("date").reset_index(drop=True)
    assert arsenal.loc[0, "rest_days"] == pytest.approx(config.MAX_REST_DAYS)  # first-ever match
    assert arsenal.loc[1, "rest_days"] == pytest.approx(7)  # real gap -- season boundary no longer resets it


def test_rest_days_caps_at_max_rest_days():
    rows = [
        _match("2011-05-01", 2011, "Arsenal", "Chelsea", 1, 0),
        _match("2011-08-13", 2012, "Arsenal", "Everton", 2, 0),  # new season, ~104 days later
    ]
    _, _, features = _build(rows)

    arsenal = features[features["team"] == "Arsenal"].sort_values("date").reset_index(drop=True)
    assert arsenal.loc[0, "rest_days"] == pytest.approx(config.MAX_REST_DAYS)  # first-ever match
    assert arsenal.loc[1, "rest_days"] == pytest.approx(config.MAX_REST_DAYS)  # ~104 real days, capped
    assert features["rest_days"].max() == config.MAX_REST_DAYS


def test_rest_days_gap_is_own_minus_opponent():
    rows = [
        _match("2010-08-10", 2011, "Arsenal", "Fulham", 2, 0),
        _match("2010-08-14", 2011, "Chelsea", "Everton", 1, 0),
        _match("2010-08-21", 2011, "Arsenal", "Chelsea", 1, 1),
    ]
    _, _, features = _build(rows)

    match3 = features[features["date"] == pd.Timestamp("2010-08-21")]
    arsenal = match3[match3["team"] == "Arsenal"].iloc[0]
    chelsea = match3[match3["team"] == "Chelsea"].iloc[0]

    assert arsenal["rest_days"] == pytest.approx(11)  # since 2010-08-10
    assert chelsea["rest_days"] == pytest.approx(7)  # since 2010-08-14
    assert arsenal["opp_rest_days"] == pytest.approx(7)
    assert chelsea["opp_rest_days"] == pytest.approx(11)
    assert arsenal["rest_days_gap"] == pytest.approx(4)
    assert chelsea["rest_days_gap"] == pytest.approx(-4)


# --- crowd ------------------------------------------------------------------


@pytest.mark.parametrize(
    "date,expected_crowd",
    [
        ("2020-06-16", 1),  # day before the window starts
        ("2020-06-17", 0),  # window start, inclusive
        ("2020-10-01", 0),  # well inside the window
        ("2021-05-16", 0),  # window end, inclusive
        ("2021-05-17", 1),  # day after the window ends
    ],
)
def test_crowd_flag_reflects_no_crowd_window(date, expected_crowd):
    rows = [_match(date, 2021, "Arsenal", "Chelsea", 1, 0)]
    _, _, features = _build(rows)
    assert features.loc[features["team"] == "Arsenal", "crowd"].iloc[0] == expected_crowd


# --- rolling form: leakage and windowing -------------------------------------


def test_form_gf_is_strictly_backward_looking_and_respects_window():
    window = config.FORM_WINDOW
    goals = list(range(1, window + 3))  # window + 2 matches, e.g. [1..7] for window=5
    opponents = [
        "Blackpool",
        "Reading",
        "Watford",
        "Sunderland",
        "Cardiff City",
        "Hull City",
        "Norwich City",
        "Middlesbrough",
        "Bolton Wanderers",
        "Leeds United",
    ][: len(goals)]
    assert len(opponents) == len(goals)

    start = pd.Timestamp("2010-08-14")
    rows = [
        _match(start + pd.Timedelta(days=7 * i), 2011, "Burnley", opp, g, 0)
        for i, (opp, g) in enumerate(zip(opponents, goals))
    ]
    _, _, features = _build(rows)

    burnley = features[features["team"] == "Burnley"].sort_values("date").reset_index(drop=True)
    expected = _prior_window_mean(goals, window)

    for i, exp in enumerate(expected):
        actual = burnley.loc[i, "form_gf"]
        if exp is None:
            fallback = config.LEAGUE_AVERAGE_FORM["gf"]
            assert actual == pytest.approx(fallback), f"match {i}: expected league-average prior {fallback}, got {actual}"
        else:
            assert actual == pytest.approx(exp), f"match {i}: expected {exp}, got {actual}"


def test_form_uses_league_average_prior_not_nan_for_first_match():
    rows = [
        _match(
            "2010-08-14",
            2011,
            "Arsenal",
            "Chelsea",
            1,
            0,
            home_shots=12,
            away_shots=8,
            home_sot=6,
            away_sot=3,
            home_corners=5,
            away_corners=4,
        )
    ]
    _, _, features = _build(rows)

    arsenal = features[features["team"] == "Arsenal"].iloc[0]
    src_by_col = {
        "form_gf": "gf",
        "form_ga": "ga",
        "form_shots": "shots",
        "form_sot": "sot",
        "form_corners": "corners",
        "form_points": "points_share",
    }
    for col, src in src_by_col.items():
        expected = config.LEAGUE_AVERAGE_FORM[src]
        val = arsenal[col]
        assert val == pytest.approx(expected), f"{col}: expected league-average prior {expected}, got {val}"

    # the ratios built from that prior form should be the ratio implied by the
    # league averages themselves, not inf or NaN
    accuracy_fallback = config.LEAGUE_AVERAGE_FORM["sot"] / config.LEAGUE_AVERAGE_FORM["shots"]
    finishing_fallback = config.LEAGUE_AVERAGE_FORM["gf"] / config.LEAGUE_AVERAGE_FORM["sot"]
    assert arsenal["form_accuracy"] == pytest.approx(accuracy_fallback)
    assert arsenal["form_finishing"] == pytest.approx(finishing_fallback)


def test_form_shots_sot_corners_and_points_use_rolling_means_of_prior_matches():
    # (opponent, goals_for, goals_against, shots, sot, corners) for Stoke City's five prior matches
    prior = [
        ("Wigan Athletic", 1, 1, 10, 4, 3),
        ("Bolton Wanderers", 0, 2, 8, 5, 6),
        ("Middlesbrough", 3, 0, 14, 7, 4),
        ("Leeds United", 2, 2, 6, 2, 5),
        ("Leicester City", 2, 1, 12, 6, 2),
    ]
    start = pd.Timestamp("2010-08-14")
    rows = [
        _match(
            start + pd.Timedelta(days=7 * i),
            2011,
            "Stoke City",
            opp,
            gf,
            ga,
            home_shots=shots,
            away_shots=shots - 1,
            home_sot=sot,
            away_sot=max(sot - 1, 0),
            home_corners=corners,
            away_corners=max(corners - 1, 0),
        )
        for i, (opp, gf, ga, shots, sot, corners) in enumerate(prior)
    ]
    # a sixth, "current" match -- its own big numbers must not leak into form
    rows.append(
        _match(
            start + pd.Timedelta(days=35),
            2011,
            "Stoke City",
            "Luton Town",
            9,
            1,
            home_shots=50,
            away_shots=5,
            home_sot=40,
            away_sot=2,
            home_corners=30,
            away_corners=1,
        )
    )

    _, _, features = _build(rows)

    stoke = features[features["team"] == "Stoke City"].sort_values("date").reset_index(drop=True)
    current = stoke.iloc[5]

    assert current["form_gf"] == pytest.approx(1.6)  # mean(1, 0, 3, 2, 2)
    assert current["form_ga"] == pytest.approx(1.2)  # mean(1, 2, 0, 2, 1)
    assert current["form_shots"] == pytest.approx(10.0)  # mean(10, 8, 14, 6, 12)
    assert current["form_sot"] == pytest.approx(4.8)  # mean(4, 5, 7, 2, 6)
    assert current["form_corners"] == pytest.approx(4.0)  # mean(3, 6, 4, 5, 2)
    # form_points is the 1.0 / 0.5 / 0.0 PLEA `actual` share, not football's 3/1/0
    assert current["form_points"] == pytest.approx(0.6)  # mean(0.5, 0.0, 1.0, 0.5, 1.0)


def test_opp_form_reflects_the_opponents_own_form_not_the_clubs():
    start = pd.Timestamp("2010-08-14")
    rows = []
    # Aston Villa build a known form: gf=2, shots=10, sot=5 in each of three warm-up wins
    for i, opp in enumerate(["Bournemouth", "Brentford", "Nottingham Forest"]):
        rows.append(
            _match(
                start + pd.Timedelta(days=7 * i),
                2011,
                "Aston Villa",
                opp,
                2,
                0,
                home_shots=10,
                away_shots=8,
                home_sot=5,
                away_sot=3,
            )
        )
    # West Bromwich Albion build a different known form: gf=5, shots=8, sot=2, playing away
    for i, opp in enumerate(["Sheffield United", "Queens Park Rangers", "Wolverhampton Wanderers"]):
        rows.append(
            _match(
                start + pd.Timedelta(days=7 * i),
                2011,
                opp,
                "West Bromwich Albion",
                0,
                5,
                home_shots=6,
                home_sot=3,
                away_shots=8,
                away_sot=2,
            )
        )
    # the match itself, after both clubs' warm-ups
    rows.append(
        _match(start + pd.Timedelta(days=28), 2011, "Aston Villa", "West Bromwich Albion", 1, 1)
    )

    _, _, features = _build(rows)

    final = features[features["date"] == start + pd.Timedelta(days=28)]
    villa = final[final["team"] == "Aston Villa"].iloc[0]
    baggies = final[final["team"] == "West Bromwich Albion"].iloc[0]

    # each club's own form matches its own warm-up run
    assert villa["form_gf"] == pytest.approx(2.0)
    assert villa["form_accuracy"] == pytest.approx(0.5)  # sot/shots = 5/10
    assert villa["form_finishing"] == pytest.approx(0.4)  # gf/sot = 2/5

    assert baggies["form_gf"] == pytest.approx(5.0)
    assert baggies["form_accuracy"] == pytest.approx(0.25)  # sot/shots = 2/8
    assert baggies["form_finishing"] == pytest.approx(2.5)  # gf/sot = 5/2

    # opp_form is genuinely the OTHER club's form, not a repeat of the club's own
    assert villa["opp_form_gf"] == pytest.approx(5.0)
    assert villa["opp_form_accuracy"] == pytest.approx(0.25)
    assert villa["opp_form_finishing"] == pytest.approx(2.5)

    assert baggies["opp_form_gf"] == pytest.approx(2.0)
    assert baggies["opp_form_accuracy"] == pytest.approx(0.5)
    assert baggies["opp_form_finishing"] == pytest.approx(0.4)


# --- ratio columns ------------------------------------------------------------


def test_ratio_columns_are_computed_from_rolling_form():
    rows = [
        _match(
            "2010-08-14",
            2011,
            "Swansea City",
            "Ipswich Town",
            2,
            0,
            home_shots=10,
            away_shots=9,
            home_sot=5,
            away_sot=4,
        ),
        _match(
            "2010-08-21",
            2011,
            "Swansea City",
            "Huddersfield Town",
            4,
            0,
            home_shots=6,
            away_shots=5,
            home_sot=3,
            away_sot=2,
        ),
        _match(
            "2010-08-28",
            2011,
            "Swansea City",
            "Hull City",
            1,
            0,
            home_shots=20,
            away_shots=1,
            home_sot=10,
            away_sot=1,
        ),
    ]
    _, _, features = _build(rows)

    swansea = features[features["team"] == "Swansea City"].sort_values("date").reset_index(drop=True)
    current = swansea.iloc[2]  # form built from matches 1 and 2 only

    assert current["form_shots"] == pytest.approx(8.0)  # mean(10, 6)
    assert current["form_sot"] == pytest.approx(4.0)  # mean(5, 3)
    assert current["form_gf"] == pytest.approx(3.0)  # mean(2, 4)
    assert current["form_accuracy"] == pytest.approx(0.5)  # 4/8
    assert current["form_finishing"] == pytest.approx(0.75)  # 3/4


def test_ratio_columns_fall_back_to_league_average_when_denominator_is_zero():
    rows = [
        _match(
            "2010-08-14",
            2011,
            "Ipswich Town",
            "Luton Town",
            1,
            0,
            home_shots=0,
            away_shots=5,
            home_sot=0,
            away_sot=2,
        ),
        _match(
            "2010-08-21",
            2011,
            "Ipswich Town",
            "Swansea City",
            0,
            0,
            home_shots=3,
            away_shots=3,
            home_sot=1,
            away_sot=1,
        ),
        _match(
            "2010-08-14",
            2011,
            "Huddersfield Town",
            "Birmingham City",
            2,
            0,
            home_shots=5,
            away_shots=4,
            home_sot=0,
            away_sot=1,
        ),
        _match(
            "2010-08-21",
            2011,
            "Huddersfield Town",
            "Blackburn Rovers",
            0,
            0,
            home_shots=3,
            away_shots=3,
            home_sot=1,
            away_sot=1,
        ),
    ]
    _, _, features = _build(rows)

    accuracy_fallback = config.LEAGUE_AVERAGE_FORM["sot"] / config.LEAGUE_AVERAGE_FORM["shots"]
    finishing_fallback = config.LEAGUE_AVERAGE_FORM["gf"] / config.LEAGUE_AVERAGE_FORM["sot"]

    zero_shots = (
        features[features["team"] == "Ipswich Town"].sort_values("date").reset_index(drop=True).iloc[1]
    )
    assert zero_shots["form_shots"] == pytest.approx(0.0)
    assert zero_shots["form_accuracy"] == pytest.approx(accuracy_fallback)  # 0 / 0 -> league-average fallback
    assert zero_shots["form_finishing"] == pytest.approx(finishing_fallback)  # gf(1) / sot(0) -> fallback

    zero_sot = (
        features[features["team"] == "Huddersfield Town"]
        .sort_values("date")
        .reset_index(drop=True)
        .iloc[1]
    )
    assert zero_sot["form_sot"] == pytest.approx(0.0)
    assert zero_sot["form_accuracy"] == pytest.approx(0.0)  # 0 / 5 is a genuine zero, not missing
    assert zero_sot["form_finishing"] == pytest.approx(finishing_fallback)  # gf(2) / sot(0) -> fallback


# --- season to date -----------------------------------------------------------


def test_ppg_uses_football_scoring_and_only_earlier_matches_this_season():
    rows = [
        _match("2010-08-14", 2011, "Crystal Palace", "Southampton", 2, 0),  # win
        _match("2010-08-21", 2011, "Crystal Palace", "West Ham United", 1, 1),  # draw
        _match("2010-08-28", 2011, "Crystal Palace", "Newcastle United", 0, 2),  # loss
        _match("2010-09-04", 2011, "Crystal Palace", "Leicester City", 3, 1),  # win
        _match("2011-08-13", 2012, "Crystal Palace", "Everton", 1, 0),  # new season
    ]
    _, _, features = _build(rows)

    palace = features[features["team"] == "Crystal Palace"].sort_values("date").reset_index(drop=True)
    assert palace["played"].tolist() == [0, 1, 2, 3, 0]

    # matchweek 1, first season in the whole dataset: no history anywhere -> promoted prior
    assert palace.loc[0, "ppg"] == pytest.approx(config.PROMOTED_PPG)

    # blend arithmetic, hand-computed: (played * this-season ppg + shrinkage * prior) / (played + shrinkage),
    # equivalently (points so far + shrinkage * prior) / (played + shrinkage)
    shrink = config.PPG_SHRINKAGE
    prior = config.PROMOTED_PPG
    assert palace.loc[1, "ppg"] == pytest.approx((3 + shrink * prior) / (1 + shrink))  # 3 pts from 1 game
    assert palace.loc[2, "ppg"] == pytest.approx((4 + shrink * prior) / (2 + shrink))  # 3+1 pts from 2 games
    assert palace.loc[3, "ppg"] == pytest.approx((4 + shrink * prior) / (3 + shrink))  # 3+1+0 pts from 3 games

    # matchweek 1 of season 2012: Crystal Palace played the whole of season 2011
    # in this same table, so the prior is their own final ppg from that season
    # (7 points -- 3+1+0+3 -- from 4 games = 1.75), not the promoted fallback
    assert palace.loc[4, "ppg"] == pytest.approx(7 / 4)


def test_ppg_and_position_fall_back_to_promoted_priors_for_a_club_new_to_the_league():
    rows = [
        _match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0),
        _match("2011-08-13", 2012, "Arsenal", "Chelsea", 1, 0),
        # Leeds United arrive in season 2013 with no history anywhere in this table,
        # even though other clubs by now have two full seasons of it
        _match("2012-08-18", 2013, "Arsenal", "Leeds United", 2, 0),
    ]
    _, _, features = _build(rows)

    leeds = features[features["team"] == "Leeds United"].iloc[0]
    assert leeds["played"] == 0
    assert leeds["ppg"] == pytest.approx(config.PROMOTED_PPG)
    assert leeds["position"] == config.PROMOTED_POSITION


def test_position_ranks_by_points_then_goal_difference():
    rows = [
        # round 1, all four matches on the same day
        _match("2010-08-14", 2011, "Manchester City", "Arsenal", 2, 0),  # City beat X, GD +2
        _match("2010-08-14", 2011, "Manchester United", "Everton", 3, 2),  # United beat Y, GD +1
        _match("2010-08-14", 2011, "Liverpool", "Fulham", 0, 0),  # genuine 3-way tie
        _match("2010-08-14", 2011, "Tottenham Hotspur", "Chelsea", 3, 0),  # Spurs beat S, GD +3
        # round 2, on separate later dates so "before this match" is unambiguous
        _match("2010-08-21", 2011, "Manchester City", "Manchester United", 1, 1),
        _match("2010-08-22", 2011, "Liverpool", "Chelsea", 2, 0),
    ]
    _, _, features = _build(rows)

    def position_on(team, date):
        row = features[(features["team"] == team) & (features["date"] == pd.Timestamp(date))]
        return row.iloc[0]["position"]

    # standings after round 1:
    # Spurs(3,+3) > City(3,+2) > United(3,+1) > {Liverpool, Fulham}(1,0) > Everton(0,-1) > Arsenal(0,-2) > Chelsea(0,-3)
    assert position_on("Manchester City", "2010-08-21") == 2
    assert position_on("Manchester United", "2010-08-21") == 3
    assert position_on("Liverpool", "2010-08-22") in (4, 5)  # tied on points, GD, and goals scored
    assert position_on("Chelsea", "2010-08-22") == 8


def test_position_is_promoted_prior_when_no_matches_played_yet():
    rows = [_match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0)]
    _, _, features = _build(rows)

    arsenal = features[features["team"] == "Arsenal"].iloc[0]
    assert arsenal["played"] == 0
    assert arsenal["position"] == config.PROMOTED_POSITION  # no history anywhere -> promoted prior


def test_position_prior_is_last_seasons_final_position_when_club_played_last_season():
    rows = [
        # round 1, all four matches on the same day -- season 2011 has only this
        # round, so it also IS the final standings for the season
        _match("2010-08-14", 2011, "Manchester City", "Arsenal", 2, 0),  # City beat X, GD +2
        _match("2010-08-14", 2011, "Manchester United", "Everton", 3, 2),  # United beat Y, GD +1
        _match("2010-08-14", 2011, "Liverpool", "Fulham", 0, 0),
        _match("2010-08-14", 2011, "Tottenham Hotspur", "Chelsea", 3, 0),  # Spurs beat S, GD +3
        # Manchester City's season 2012 opener
        _match("2011-08-13", 2012, "Manchester City", "Liverpool", 1, 0),
    ]
    _, _, features = _build(rows)

    city_opener = features[
        (features["team"] == "Manchester City") & (features["season"] == 2012)
    ].iloc[0]
    # final standings after round 1: Spurs(3,+3) > City(3,+2) > United(3,+1) > ...
    assert city_opener["played"] == 0
    assert city_opener["position"] == 2


# --- target -------------------------------------------------------------------


@pytest.mark.parametrize(
    "hg,ag,home_target,away_target",
    [
        (2, 0, 1, 0),  # home win
        (0, 2, 0, 1),  # away win
        (1, 1, 0, 0),  # draw -- 0 for both sides
    ],
)
def test_target_is_1_for_a_win_and_0_for_a_draw_or_loss(hg, ag, home_target, away_target):
    rows = [_match("2010-08-14", 2011, "Arsenal", "Chelsea", hg, ag)]
    _, _, features = _build(rows)

    home_row = features[features["team"] == "Arsenal"].iloc[0]
    away_row = features[features["team"] == "Chelsea"].iloc[0]
    assert home_row["target"] == home_target
    assert away_row["target"] == away_target


# --- fixture facts --------------------------------------------------------


def test_is_home_flag():
    rows = [_match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0)]
    _, _, features = _build(rows)

    assert features.loc[features["team"] == "Arsenal", "is_home"].iloc[0] == 1
    assert features.loc[features["team"] == "Chelsea", "is_home"].iloc[0] == 0


def test_kickoff_hour_is_not_a_feature():
    """Dropped deliberately: football-data.co.uk has no kickoff time before
    2019/20, so the column was missing for 56% of rows and its absence was a
    perfect tell for "this is an older season". A weak feature carrying a strong
    era signal is worse than no feature."""
    _, _, features = _build([_match("2020-08-01", 2021, "Arsenal", "Chelsea", 1, 0)])
    assert "kickoff_hour" not in features.columns


@pytest.mark.parametrize("date", ["2010-08-14", "2010-08-16", "2020-12-25"])
def test_day_of_week_matches_the_calendar(date):
    rows = [_match(date, 2011, "Arsenal", "Chelsea", 1, 0)]
    _, _, features = _build(rows)
    expected = pd.Timestamp(date).dayofweek
    assert features.loc[features["team"] == "Arsenal", "day_of_week"].iloc[0] == expected


# --- no mutation ----------------------------------------------------------


def test_inputs_are_not_mutated():
    rows = [
        _match("2010-08-14", 2011, "Arsenal", "Chelsea", 1, 0),
        _match("2010-08-21", 2011, "Everton", "Fulham", 2, 2),
    ]
    results = pd.DataFrame(rows)
    history = plea.run(results)
    results_before = results.copy(deep=True)
    history_before = history.copy(deep=True)

    build_features(results, history)

    pd.testing.assert_frame_equal(results, results_before)
    pd.testing.assert_frame_equal(history, history_before)
