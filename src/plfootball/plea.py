"""PLEA — Premier League Elo Algorithm.

Replays every match from August 2010 in date order. Each club carries one
number that rises after good results and falls after bad ones, by more when the
opponent was strong or the margin was wide.

The output is a *history*, not a table: one row per club per match, recording
the rating going into the match and the rating coming out. Everything else is a
query against it —

    ratings_as_of(history, "2022-03-01")   the table on any date
    season_end_table(history, 2024)        the table at the end of 2023/24

and, crucially, the model reads ``elo_before`` and never ``elo_after``.
``elo_after`` already knows the result you are trying to predict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from . import config


@dataclass(frozen=True)
class PleaParams:
    """Tuning knobs. Change these and you have a new version — record which."""

    version: str = "v2"
    k: float = 20.0  # reaction speed; higher is jumpier
    # Rating points handed to the home side.
    #
    # 65 looks wrong and is not. Over 6,110 matches the home side takes 0.5685
    # of the points, and the rating gap that *directly* implies is 47.9 — which
    # is why this was carried for three sessions as an obvious fix. It is not.
    # Swept properly against out-of-sample log loss, 48 makes the model worse
    # (z = -2.79 on the tuning seasons, -1.43 across all thirteen) and nothing
    # beats 65 by a detectable margin.
    #
    # The reason is that home_adv is a knob inside a feedback loop, not an
    # estimate of a quantity. The ratings adapt around whatever it is set to,
    # and the model downstream re-fits its own intercept and slope on
    # elo_expected anyway — so PLEA being internally biased costs nothing that
    # the regression does not simply absorb. 47.9 answers a real question. It
    # is not the question this parameter asks.
    home_adv: float = 65.0
    # Behind closed doors the advantage did not shrink, it vanished: 452
    # covid-era matches split 0.5022 against 0.5738 for the rest, a difference
    # real at z = 2.9 as a plain measurement of what happened.
    #
    # Its effect on predictions is a different matter and much weaker — even
    # restricted to the closed-door matches it only reaches z = +1.42, because
    # one season in thirteen cannot carry a result on its own. It is kept
    # because the cause is established even where the benefit is not, and
    # because those ratings feed forward: without it the covid season leaves a
    # permanent dent in every rating that carries out of it.
    no_crowd_home_adv: float = 0.0
    start_elo: float = 1500.0  # everyone, August 2010
    promoted_elo: float = 1400.0  # a club arriving from the Championship
    carry_over: float = 0.80  # between seasons: pull 20% back towards start

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = PleaParams()


def expected_score(own_elo: float, opp_elo: float, home_adv: float) -> float:
    """Probability-ish expectation of the result, 0 to 1.

    ``home_adv`` is added to the home side's rating before comparing, so pass it
    as +home_adv for the home team and -home_adv for the away team.
    """
    return 1.0 / (1.0 + 10.0 ** ((opp_elo - own_elo - home_adv) / 400.0))


def margin_multiplier(goal_difference: int) -> float:
    """Wider wins move the ratings further, with diminishing returns."""
    gd = abs(int(goal_difference))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _season_start_ratings(
    clubs: set[str],
    previous: dict[str, float],
    played_last_season: set[str],
    params: PleaParams,
) -> dict[str, float]:
    """Carry ratings into a new season.

    A club that played last season keeps most of its rating, pulled slightly
    back towards average. Anyone else — newly promoted, or returning after years
    away, where the old rating is stale — starts on ``promoted_elo``.
    """
    ratings = {}
    for club in clubs:
        if club in played_last_season:
            prior = previous[club]
            ratings[club] = params.start_elo + params.carry_over * (prior - params.start_elo)
        else:
            ratings[club] = params.promoted_elo
    return ratings


def run(results: pd.DataFrame, params: PleaParams = DEFAULT) -> pd.DataFrame:
    """Replay every match and return the rating history.

    ``results`` must be the output of ingest.load_results(): canonical club
    names, one row per match, sorted by date. Order matters — PLEA is a chain.
    """
    required = {"date", "season", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing columns: {sorted(missing)}")

    results = results.sort_values(["date", "kickoff"], kind="stable", na_position="first")

    no_crowd_from = pd.Timestamp(config.NO_CROWD_START)
    no_crowd_to = pd.Timestamp(config.NO_CROWD_END)

    ratings: dict[str, float] = {}
    played_last_season: set[str] = set()
    previous_season: int | None = None
    rows: list[dict] = []

    for season, block in results.groupby("season", sort=True):
        clubs = set(block["home_team"]) | set(block["away_team"])

        if previous_season is None:
            ratings = dict.fromkeys(clubs, params.start_elo)
        else:
            ratings = _season_start_ratings(clubs, ratings, played_last_season, params)

        for match in block.itertuples(index=False):
            home, away = match.home_team, match.away_team
            home_elo, away_elo = ratings[home], ratings[away]

            playing_to_an_empty_ground = no_crowd_from <= match.date <= no_crowd_to
            advantage = (
                params.no_crowd_home_adv if playing_to_an_empty_ground else params.home_adv
            )
            home_expected = expected_score(home_elo, away_elo, advantage)
            gd = match.home_goals - match.away_goals
            home_actual = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)

            delta = params.k * margin_multiplier(gd) * (home_actual - home_expected)
            ratings[home] = home_elo + delta
            ratings[away] = away_elo - delta

            for team, opp, is_home, elo_b, opp_b, elo_a, actual in (
                (home, away, 1, home_elo, away_elo, ratings[home], home_actual),
                (away, home, 0, away_elo, home_elo, ratings[away], 1.0 - home_actual),
            ):
                rows.append(
                    {
                        "date": match.date,
                        "season": season,
                        "team": team,
                        "opponent": opp,
                        "is_home": is_home,
                        "elo_before": elo_b,
                        "opp_elo_before": opp_b,
                        "elo_after": elo_a,
                        "elo_expected": home_expected if is_home else 1.0 - home_expected,
                        "goals_for": match.home_goals if is_home else match.away_goals,
                        "goals_against": match.away_goals if is_home else match.home_goals,
                        "actual": actual,
                        "won": int(actual == 1.0),
                    }
                )

        played_last_season = clubs
        previous_season = season

    history = pd.DataFrame(rows)
    history["elo_gap"] = history["elo_before"] - history["opp_elo_before"]
    return history.sort_values(["date", "team"], kind="stable").reset_index(drop=True)


def ratings_as_of(history: pd.DataFrame, when=None, *, active_only: bool = True) -> pd.DataFrame:
    """The rating table as it stood on a given date.

    Takes each club's most recent ``elo_after`` strictly before ``when``.
    Pass nothing for the latest available.

    ``active_only`` keeps just the clubs playing in the most recent season, so
    the table is the twenty clubs you'd expect. Turn it off to see every club
    that has ever played, each frozen at the rating it left the league on —
    useful for history, misleading as a league table.
    """
    df = history if when is None else history[history["date"] < pd.Timestamp(when)]
    if df.empty:
        return pd.DataFrame(columns=["team", "elo", "played", "last_match"])

    if active_only:
        df = df[df["season"] == df["season"].max()]

    latest = df.sort_values("date").groupby("team", as_index=False).last()
    counts = df.groupby("team").size().rename("played")

    table = (
        latest[["team", "elo_after", "date"]]
        .rename(columns={"elo_after": "elo", "date": "last_match"})
        .join(counts, on="team")
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    table.index += 1
    table.index.name = "rank"
    return table


def season_end_table(history: pd.DataFrame, season: int) -> pd.DataFrame:
    """Final ratings for the clubs that played in a given season."""
    block = history[history["season"] == season]
    if block.empty:
        raise ValueError(f"no matches for season {season}")

    latest = block.sort_values("date").groupby("team", as_index=False).last()
    table = (
        latest[["team", "elo_after"]]
        .rename(columns={"elo_after": "elo"})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    table.index += 1
    table.index.name = "rank"
    return table
