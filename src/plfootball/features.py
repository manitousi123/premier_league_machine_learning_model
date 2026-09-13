"""Builds the model's training table from match results and PLEA ratings.

One row per club per match, all of it knowable before kickoff. That last part
is the whole point of this module: a feature built from the match it describes
is a leak, and a leaked feature makes the model look brilliant in testing and
useless in production without ever telling you why. So every rolling window
here is shifted at least one match into the past before it is ever averaged,
every "season so far" number excludes the row's own match, and league position
is built from a standings snapshot taken strictly before the date in question —
not just before the row, but before every match played that day, since same-day
fixtures are simultaneous and none of them can see each other's results.

``target`` is the one deliberate exception: it is the answer being learned,
copied straight from PLEA's ``won``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Own-club rolling-form sources -> output column names.
_FORM_METRICS = {
    "gf": "form_gf",
    "ga": "form_ga",
    "shots": "form_shots",
    "sot": "form_sot",
    "corners": "form_corners",
    "points_share": "form_points",
}

_OUTPUT_COLUMNS = [
    # identity - kept for joining and inspection, not fed to the model
    "date", "season", "team", "opponent", "matchweek",
    # fixture facts, known before kickoff
    "is_home", "day_of_week",
    "rest_days", "opp_rest_days", "rest_days_gap", "crowd",
    # PLEA, copied from history
    "own_elo", "opp_elo", "elo_gap", "elo_expected",
    # rolling form, this club
    "form_gf", "form_ga", "form_shots", "form_sot", "form_corners", "form_points",
    # rolling form, the opponent, going into this same match
    "opp_form_gf", "opp_form_ga", "opp_form_shots", "opp_form_sot",
    "opp_form_corners", "opp_form_points",
    # how much more of the game this club has been creating than its opponent
    "sot_gap", "shots_gap",
    # ratios, computed from the rolling form above
    "form_accuracy", "form_finishing", "opp_form_accuracy", "opp_form_finishing",
    # season to date, before this match
    "played", "ppg", "position", "points",
    # what is still at stake, for this club and for the opponent
    "points_from_title", "points_from_top4", "points_from_safety",
    "title_live", "top4_live", "relegation_live", "stakes_live",
    "opp_points_from_top4", "opp_points_from_safety", "opp_stakes_live",
    # the answer
    "target",
]


def _to_long(results: pd.DataFrame) -> pd.DataFrame:
    """One row per match -> one row per club per match.

    Mirrors how plea.run() already looks at the data: a home row and an away
    row per fixture, each carrying that side's own shots/sot/corners. The
    opponent's copy of those numbers is never needed here directly - it shows
    up later as the opponent's own row, joined back in by date and name.
    """
    shared = ["date", "season", "kickoff"]

    home = results[shared + [
        "home_team", "away_team", "home_goals", "away_goals",
        "home_shots", "home_sot", "home_corners",
    ]].rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "gf", "away_goals": "ga",
        "home_shots": "shots", "home_sot": "sot", "home_corners": "corners",
    })
    home["is_home"] = 1

    away = results[shared + [
        "home_team", "away_team", "home_goals", "away_goals",
        "away_shots", "away_sot", "away_corners",
    ]].rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "gf", "home_goals": "ga",
        "away_shots": "shots", "away_sot": "sot", "away_corners": "corners",
    })
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["date", "team"], kind="stable").reset_index(drop=True)


def recent_form(results: pd.DataFrame, window: int = config.FORM_WINDOW) -> pd.DataFrame:
    """Each club's rolling form as it stands *after* its most recent match.

    These are exactly the numbers `_add_form` would put on that club's next
    row, which is the point: a fixture that has not been played yet needs its
    form computed the same way the training rows were, or the model is being
    shown numbers that mean something slightly different from what it learned
    on. Sharing `_to_long` and the window definition is what guarantees that.

    `.tail(window)` over a club's matches in date order is the same set the
    shifted rolling window would see from the following match - all of them if
    the club has played fewer than `window`, which mirrors `min_periods=1`. A
    club with no matches at all does not appear here; the caller fills it from
    `config.LEAGUE_AVERAGE_FORM`.
    """
    long = _to_long(results).sort_values(["team", "date"], kind="stable")
    long["points_share"] = np.select(
        [long["gf"] > long["ga"], long["gf"] == long["ga"]], [1.0, 0.5], default=0.0
    )

    recent = long.groupby("team", sort=False).tail(window)
    form = recent.groupby("team").agg(
        **{dst: (src, "mean") for src, dst in _FORM_METRICS.items()},
        form_window=("date", "size"),
        last_match=("date", "max"),
    )
    return form.reset_index()


def _merge_history(long: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach PLEA's ratings and result for each row - the pieces we don't recompute."""
    hist = history[[
        "date", "team", "opponent",
        "elo_before", "opp_elo_before", "elo_expected", "elo_gap", "actual", "won",
    ]].rename(columns={"elo_before": "own_elo", "opp_elo_before": "opp_elo"})

    merged = long.merge(hist, on=["date", "team", "opponent"], how="left")
    if merged["own_elo"].isna().any():
        raise ValueError("results and history disagree - some matches have no PLEA row")

    # Two representations of the same result: `actual` (1.0/0.5/0.0, PLEA's
    # convention, for rolling form) and real football points (3/1/0, for
    # points-per-game and the league table).
    merged["points_share"] = merged["actual"]
    merged["football_points"] = np.select(
        [merged["actual"] == 1.0, merged["actual"] == 0.5], [3, 1], default=0
    )
    return merged



def _crowd_flag(date: pd.Series) -> np.ndarray:
    """0 inside the closed-doors covid window, 1 otherwise."""
    start, end = pd.Timestamp(config.NO_CROWD_START), pd.Timestamp(config.NO_CROWD_END)
    return np.where((date >= start) & (date <= end), 0, 1)


def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Days since each club's previous match, computed across season boundaries and capped.

    Unlike form, a close-season gap of two-plus months isn't meaningful "rest",
    so beyond `config.MAX_REST_DAYS` more rest stops meaning anything and the
    value is capped there. A club's very first match ever has no previous
    match at all - treated as fully rested, the same as hitting the cap.
    """
    df = df.sort_values(["team", "date"], kind="stable")
    rest = df.groupby("team")["date"].diff().dt.days
    df["rest_days"] = rest.fillna(config.MAX_REST_DAYS).clip(upper=config.MAX_REST_DAYS)

    opp_rest = df[["date", "team", "rest_days"]].rename(
        columns={"team": "opponent", "rest_days": "opp_rest_days"}
    )
    df = df.merge(opp_rest, on=["date", "opponent"], how="left")
    df["rest_days_gap"] = df["rest_days"] - df["opp_rest_days"]
    return df


def _add_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean of the club's own last `window` matches, strictly before this one.

    `.shift(1)` moves the current match out of the window before `.rolling()`
    ever sees it - the only way to be sure nothing peeks at its own result.
    Fewer than `window` prior matches -> mean of what exists; none -> the
    frozen `config.LEAGUE_AVERAGE_FORM` prior (never 0 - a zero shot count is
    real data, not "unknown", and never NaN - a club has to start somewhere).

    The opponent's own form going into this same match is a second copy of the
    same numbers, looked up by matching this row's date against the opponent's
    own row for that date - not recomputed.
    """
    df = df.sort_values(["team", "date"], kind="stable")
    grouped = df.groupby("team", sort=False)
    for src, dst in _FORM_METRICS.items():
        df[dst] = grouped[src].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    form_cols = list(_FORM_METRICS.values())
    opp_form = df[["date", "team", *form_cols]].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in form_cols}}
    )
    df = df.merge(opp_form, on=["date", "opponent"], how="left")

    # A club's very first match ever has no prior matches to average - fill
    # from the frozen league-average prior instead of leaving it NaN. A club
    # with 1+ prior matches already got a real rolling mean above and is
    # untouched here.
    for src, dst in _FORM_METRICS.items():
        fallback = config.LEAGUE_AVERAGE_FORM[src]
        df[dst] = df[dst].fillna(fallback)
        df[f"opp_{dst}"] = df[f"opp_{dst}"].fillna(fallback)

    return df


def _add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """How much more of the game this club has been creating than its opponent.

    Shots on target is the one thing PLEA structurally cannot see. PLEA is
    built from goals, and goals are the lucky part of football: a club creating
    far more than it converts carries a rating that is too low, and it tends to
    come back. Shot counts are the repeatable half of the same story, which is
    most of what expected goals would have given us had it been obtainable.

    Taken as a difference rather than as two columns because the two only mean
    anything against each other - twelve shots is good against Liverpool and
    poor against a side camped in its own box - and because the table holds two
    mirrored rows per fixture, so a difference makes the two perspectives exact
    negatives of one another rather than two loosely related numbers.
    """
    df["sot_gap"] = df["form_sot"] - df["opp_form_sot"]
    df["shots_gap"] = df["form_shots"] - df["opp_form_shots"]
    return df


def _safe_ratio(numer: pd.Series, denom: pd.Series, fallback: float) -> pd.Series:
    """numer / denom, with a zero denominator giving `fallback` instead of inf or NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numer / denom
    return ratio.where(denom != 0, fallback)


def _add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Shot accuracy and finishing, computed from the rolling form - never from this match.

    Form is always populated now (see `_add_form`), so these are almost always
    a real ratio. The rare case of a rolling window averaging to exactly zero
    shots (or zero shots on target) falls back to the ratio implied by
    `config.LEAGUE_AVERAGE_FORM`, so the result is never inf and never NaN.
    """
    accuracy_fallback = config.LEAGUE_AVERAGE_FORM["sot"] / config.LEAGUE_AVERAGE_FORM["shots"]
    finishing_fallback = config.LEAGUE_AVERAGE_FORM["gf"] / config.LEAGUE_AVERAGE_FORM["sot"]

    df = df.copy()
    df["form_accuracy"] = _safe_ratio(df["form_sot"], df["form_shots"], accuracy_fallback)
    df["form_finishing"] = _safe_ratio(df["form_gf"], df["form_sot"], finishing_fallback)
    df["opp_form_accuracy"] = _safe_ratio(df["opp_form_sot"], df["opp_form_shots"], accuracy_fallback)
    df["opp_form_finishing"] = _safe_ratio(df["opp_form_gf"], df["opp_form_sot"], finishing_fallback)
    return df


def _add_league_context(df: pd.DataFrame) -> pd.DataFrame:
    """The league table as it stood before each match: position, points, and the
    points held by the clubs on the lines that matter.

    Built once per season from a date x club grid of that day's points and
    goals. Cumulative-summing down the date axis and then shifting by one date
    step means a date's own matches - the row's own match *and* every other
    match played that day - are excluded from the standings entering it. Same-
    day fixtures are simultaneous; none of them has "already happened" from any
    other's point of view.

    Alongside each club's own position it records the points on the title,
    top-four, safety and relegation lines that day, which is everything
    `_add_stakes` needs to work out what is still reachable.
    """
    daily = (
        df.groupby(["season", "date", "team"], as_index=False)
        .agg(
            pts=("football_points", "sum"),
            gf=("gf", "sum"),
            ga=("ga", "sum"),
            games=("football_points", "size"),
        )
    )

    lines = {
        "pts_title": config.TITLE_POSITION,
        "pts_top4": config.TOP4_POSITION,
        "pts_safety": config.SAFETY_POSITION,
        "pts_drop": config.RELEGATION_POSITION,
    }

    tables = []
    for season, block in daily.groupby("season", sort=True):
        wide = block.pivot(index="date", columns="team", values=["pts", "gf", "ga", "games"])
        wide = wide.sort_index().fillna(0.0)
        before = wide.cumsum().shift(1).fillna(0.0)
        goal_diff = before["gf"] - before["ga"]

        for match_date in before.index:
            standing = pd.DataFrame({
                "table_points": before["pts"].loc[match_date],
                "gd": goal_diff.loc[match_date],
                "gf": before["gf"].loc[match_date],
                "table_played": before["games"].loc[match_date],
            }).sort_values(["table_points", "gd", "gf"], ascending=False, kind="stable")

            # A row per club, plus that day's snapshot of the four dividing
            # lines - the same four numbers repeated down the column, because
            # every club is looking at the same table.
            points = standing["table_points"].to_numpy()
            played = standing["table_played"].to_numpy()
            slot = {name: min(pos, len(points)) - 1 for name, pos in lines.items()}

            tables.append(
                standing.assign(
                    season=season,
                    date=match_date,
                    team=standing.index,
                    position=range(1, len(standing) + 1),
                    played_drop=played[slot["pts_drop"]],
                    **{name: points[index] for name, index in slot.items()},
                ).reset_index(drop=True)
            )

    context = pd.concat(tables, ignore_index=True)
    keep = [
        "season", "date", "team", "position", "table_points", "table_played",
        "played_drop", *lines,
    ]
    return df.merge(context[keep], on=["season", "date", "team"], how="left")


def _add_stakes(df: pd.DataFrame) -> pd.DataFrame:
    """What each club still has to play for, going into this match.

    PLEA knows how good a club is. It has no idea whether that club still
    cares, and a side already safe, already relegated or already champion plays
    a very different game in May - which is exactly where the ratings lose most
    ground to the betting market.

    "Live" here means *reachable on points*: the club could still draw level
    with whoever holds that line if it won everything left. That is a necessary
    condition for the real thing rather than the whole of it - proper
    mathematical elimination depends on who still plays whom - but it needs no
    fixture list and it is honest about what it measures.

    At matchweek one every club has nought points, so every gap is zero and
    everything is live. That is correct, and needs no special case.
    """
    max_gain = 3 * (config.SEASON_MATCHES - df["table_played"])

    df["points"] = df["table_points"]
    df["points_from_title"] = df["pts_title"] - df["table_points"]
    df["points_from_top4"] = df["pts_top4"] - df["table_points"]
    # Negative means this club is above the line looking down.
    df["points_from_safety"] = df["pts_safety"] - df["table_points"]

    df["title_live"] = (df["points_from_title"] <= max_gain).astype(int)
    df["top4_live"] = (df["points_from_top4"] <= max_gain).astype(int)

    # Relegation runs the other way: not "can I get down to them" but "can they
    # still climb up to me", so it is the bottom club's remaining games that count.
    drop_ceiling = df["pts_drop"] + 3 * (config.SEASON_MATCHES - df["played_drop"])
    df["relegation_live"] = (drop_ceiling >= df["table_points"]).astype(int)

    df["stakes_live"] = df[["title_live", "top4_live", "relegation_live"]].max(axis=1)

    # The opponent's situation matters as much as our own: a dead rubber for
    # one side is not a dead rubber if the other is still fighting to stay up.
    mine = ["points_from_safety", "points_from_top4", "stakes_live"]
    opponent = df[["date", "team", *mine]].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in mine}}
    )
    return df.merge(opponent, on=["date", "opponent"], how="left")


def _season_end_table(df: pd.DataFrame) -> pd.DataFrame:
    """Final points-per-game and league position for every club, for every season.

    A pure end-of-season summary, built from that season's matches alone. By
    the time the following season kicks off every match here is finished, so
    handing this forward as next season's prior is not a leak - only feeding a
    season's numbers back into *itself* would be.
    """
    totals = df.groupby(["season", "team"], as_index=False).agg(
        pts=("football_points", "sum"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
        played=("football_points", "size"),
    )
    totals["gd"] = totals["gf"] - totals["ga"]
    totals["ppg"] = totals["pts"] / totals["played"]

    ranked = []
    for season, block in totals.groupby("season", sort=True):
        block = block.sort_values(["pts", "gd", "gf"], ascending=False, kind="stable")
        ranked.append(block.assign(position=range(1, len(block) + 1)))

    return pd.concat(ranked, ignore_index=True)[["season", "team", "ppg", "position"]]


def _add_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's cold-start priors: last season's final ppg and position.

    "Last season" means `season - 1` exactly - a club that played in the
    Premier League last season carries its own numbers forward; anyone else
    (freshly promoted, or back after enough years away that its old numbers
    are stale) falls back to the frozen `config.PROMOTED_PPG` /
    `config.PROMOTED_POSITION`, including every club in the first season on
    record, which has no "last season" at all.
    """
    season_end = _season_end_table(df)
    prior = season_end.rename(columns={"ppg": "prior_ppg", "position": "prior_position"})
    prior = prior.assign(season=prior["season"] + 1)

    df = df.merge(prior, on=["season", "team"], how="left")
    df["prior_ppg"] = df["prior_ppg"].fillna(config.PROMOTED_PPG)
    df["prior_position"] = df["prior_position"].fillna(config.PROMOTED_POSITION)
    return df


def _add_season_to_date(df: pd.DataFrame) -> pd.DataFrame:
    """`played`, `ppg` and `position` as they stood before this match, this season.

    `ppg` blends the raw this-season figure toward the club's prior (see
    `_add_priors`) with `config.PPG_SHRINKAGE` matches of weight, so a club's
    number starts exactly at the prior on matchday one and leans further on
    its own results as the season goes on - `played * this_season + shrinkage
    * prior`, all over `played + shrinkage`. `position` is filled with the
    prior only where `played == 0`; matchweek 2 onward is the real, unfilled
    table position, exactly as before.
    """
    df = df.sort_values(["season", "team", "date"], kind="stable")

    grouped = df.groupby(["season", "team"], sort=False)
    df["played"] = grouped.cumcount()

    # prior_points is played * ppg_this_season already (0 when played == 0),
    # so this needs no special-casing for the cold-start row.
    prior_points = grouped["football_points"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    df["ppg"] = (prior_points + config.PPG_SHRINKAGE * df["prior_ppg"]) / (
        df["played"] + config.PPG_SHRINKAGE
    )

    df = _add_league_context(df)
    df["position"] = df["position"].where(df["played"] > 0, df["prior_position"])
    return _add_stakes(df)


def build_features(results: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build the model's training table.

    ``results`` is ingest.load_results() and ``history`` is plea.run(results) -
    the same results, already replayed through PLEA. Returns one row per club
    per match, sorted by date then team, roughly 12,160 rows for the full 16
    seasons. Neither input is modified.
    """
    df = _to_long(results)
    df = _merge_history(df, history)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["crowd"] = _crowd_flag(df["date"])

    df = _add_rest_days(df)
    df = _add_form(df, config.FORM_WINDOW)
    df = _add_gaps(df)
    df = _add_ratios(df)
    df = _add_priors(df)
    df = _add_season_to_date(df)

    df["matchweek"] = df["played"] + 1
    df["target"] = df["won"].astype(int)

    df = df.sort_values(["date", "team"], kind="stable").reset_index(drop=True)
    return df[_OUTPUT_COLUMNS]
