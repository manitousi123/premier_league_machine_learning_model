"""Predict fixtures that have not been played yet.

Everything else in this project scores matches that already happened, where the
feature row was built by `features.build_features` from a completed result. A
fixture on Saturday has no such row, so one has to be constructed — and that is
the whole risk of this module.

**The failure this is written to avoid.** If a prediction row is assembled even
slightly differently from a training row — a form window over six matches
instead of five, a rating taken after the last match instead of before the next
one — the model is quietly shown numbers that do not mean what it learned they
meant. Nothing errors. The predictions are simply worse, for no visible reason.

So the two quantities that matter are not recomputed here. `form_sot` and
`form_shots` come from `features.recent_form`, which shares `_to_long` and the
window definition with the training path, and the ratings come from
`plea.ratings_entering`, which mirrors PLEA's own season-rollover rule. This
module's job is to put them side by side, not to work them out.

`tests/test_predict.py` closes the loop: it rebuilds rows for matches that are
already in the training table and asserts the numbers come out identical.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from . import config, features, model, plea
from .teams import canonicalise

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DIVISION = "E0"  # the Premier League, in football-data.co.uk's coding


class NoFixtures(RuntimeError):
    """The fixture list has nothing for this division — normal between seasons."""


def download_fixtures(url: str = FIXTURES_URL) -> pd.DataFrame:
    """Upcoming matches, from the same source as the results.

    One file covers every division the site tracks, so it is filtered to the
    Premier League and the club names put through the same canonical map. An
    unknown name raises here exactly as it does on ingest — a promoted club
    under an unfamiliar spelling should stop the run, not quietly predict
    itself as somebody else.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    raw = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig", on_bad_lines="skip")
    if "Div" not in raw.columns:
        raise NoFixtures(f"{url} did not return a fixture list")

    block = raw[raw["Div"] == DIVISION].dropna(subset=["HomeTeam", "AwayTeam"])
    if block.empty:
        raise NoFixtures(f"no {DIVISION} fixtures listed — probably between seasons")

    fixtures = pd.DataFrame({
        "date": pd.to_datetime(block["Date"], dayfirst=True, format="mixed"),
        "kickoff": block.get("Time", pd.Series(pd.NA, index=block.index)),
        "home_team": canonicalise(block["HomeTeam"]),
        "away_team": canonicalise(block["AwayTeam"]),
    })
    return fixtures.sort_values(["date", "kickoff"], kind="stable").reset_index(drop=True)


def drop_already_played(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Remove anything the results file already has a score for.

    The two files overlap around a matchweek in progress. A match with a result
    is not a prediction, and leaving it in would let the model be graded on
    something it could have looked up.
    """
    played = set(zip(results["date"], results["home_team"], results["away_team"]))
    keep = [
        row not in played
        for row in zip(fixtures["date"], fixtures["home_team"], fixtures["away_team"])
    ]
    return fixtures[keep].reset_index(drop=True)


def _form_lookup(results: pd.DataFrame) -> pd.DataFrame:
    """Every club's current form, indexed by club, with newcomers filled in."""
    form = features.recent_form(results).set_index("team")
    defaults = {dst: config.LEAGUE_AVERAGE_FORM[src] for src, dst in features._FORM_METRICS.items()}
    return form, defaults


def build_rows(
    fixtures: pd.DataFrame,
    results: pd.DataFrame,
    history: pd.DataFrame,
    params: plea.PleaParams = plea.DEFAULT,
) -> pd.DataFrame:
    """Two model-ready rows per fixture — one from each club's point of view.

    Mirrors the training table's shape: the same fixture appears twice, once
    with each club as `team`, and the pair is rejoined into a single verdict
    later. A club with no history at all takes the league-average form prior,
    the same one `_add_form` uses for a club's first ever match.
    """
    form, defaults = _form_lookup(results)

    def look(club: str, column: str) -> float:
        if club in form.index:
            return float(form.loc[club, column])
        return defaults[column]

    rows = []
    for fixture in fixtures.itertuples(index=False):
        season = config.season_of(fixture.date)
        ratings = plea.ratings_entering(
            history, season, [fixture.home_team, fixture.away_team], params
        )
        home_elo, away_elo = ratings[fixture.home_team], ratings[fixture.away_team]
        # Every fixture still to be played has a crowd, so the covid exception
        # in PLEA never applies here.
        home_expected = plea.expected_score(home_elo, away_elo, params.home_adv)

        for team, opponent, is_home, own_elo, opp_elo, expected in (
            (fixture.home_team, fixture.away_team, 1, home_elo, away_elo, home_expected),
            (fixture.away_team, fixture.home_team, 0, away_elo, home_elo, 1.0 - home_expected),
        ):
            rows.append({
                "date": fixture.date,
                "kickoff": fixture.kickoff,
                "season": season,
                "team": team,
                "opponent": opponent,
                "is_home": is_home,
                "own_elo": own_elo,
                "opp_elo": opp_elo,
                "elo_gap": own_elo - opp_elo,
                "elo_expected": expected,
                "form_sot": look(team, "form_sot"),
                "opp_form_sot": look(opponent, "form_sot"),
                "form_shots": look(team, "form_shots"),
                "opp_form_shots": look(opponent, "form_shots"),
            })

    built = pd.DataFrame(rows)
    built["sot_gap"] = built["form_sot"] - built["opp_form_sot"]
    built["shots_gap"] = built["form_shots"] - built["opp_form_shots"]
    return built


def verdicts(rows: pd.DataFrame, probability: pd.Series) -> pd.DataFrame:
    """The two rows of each fixture rejoined into one call.

    `p_draw` is whatever neither club claimed. It is a leftover rather than a
    prediction — the model was never trained to spot draws — so it is reported
    but not called.
    """
    scored = rows.assign(p_win=np.asarray(probability, dtype=float))

    home = scored[scored["is_home"] == 1].rename(
        columns={"team": "home_team", "opponent": "away_team", "p_win": "p_home"}
    )[["date", "kickoff", "home_team", "away_team", "p_home"]]
    away = scored[scored["is_home"] == 0].rename(
        columns={"team": "away_team", "opponent": "home_team", "p_win": "p_away"}
    )[["date", "home_team", "away_team", "p_away"]]

    merged = home.merge(away, on=["date", "home_team", "away_team"], how="inner")
    if len(merged) != len(home):
        raise ValueError(
            f"{len(home)} fixtures but only {len(merged)} found both perspectives"
        )

    merged["p_draw"] = 1.0 - merged["p_home"] - merged["p_away"]

    # The grid. Each fixture was scored twice and each answer is a yes or a no,
    # so there are four ways the pair can land — not three. The fourth, both
    # clubs backed to win the same match, is the model contradicting itself; it
    # is kept as its own outcome rather than folded into "HOME will win",
    # because a contradiction reported as a confident call is the worst of the
    # four. It has never yet fired: 0 of 4,940 test matches.
    home_backed = merged["p_home"] >= 0.5
    away_backed = merged["p_away"] >= 0.5

    merged["home_call"] = np.where(home_backed, "WILL win", "WILL NOT win")
    merged["away_call"] = np.where(away_backed, "WILL win", "WILL NOT win")
    merged["verdict"] = np.select(
        [
            home_backed & ~away_backed,
            away_backed & ~home_backed,
            home_backed & away_backed,
        ],
        ["HOME will win", "AWAY will win", "NO CALL - both sides backed"],
        default="HOME will not win",
    )

    # How far the better-backed side is from a coin flip. Across thirteen test
    # seasons a call at 60% landed 69% of the time and one at 70% landed 77%,
    # so this is worth reading as a real strength rather than decoration.
    merged["confidence"] = merged[["p_home", "p_away"]].max(axis=1)

    ordered = [
        "date", "kickoff", "home_team", "away_team",
        "p_home", "p_away", "p_draw",
        "home_call", "away_call", "verdict", "confidence",
    ]
    return merged[ordered].sort_values(["date", "kickoff"], kind="stable").reset_index(drop=True)


def grid(calls: pd.DataFrame) -> pd.DataFrame:
    """The 2x2 itself: how many fixtures landed in each quadrant.

    Rows are what the home club's own row concluded, columns what the away
    club's did. Reading it as a table rather than four labels makes the
    diagonal obvious — the two cells where the perspectives agree are the
    confident calls, and the top-right is the one that should stay empty.
    """
    counts = pd.crosstab(calls["home_call"], calls["away_call"])
    return counts.reindex(
        index=["WILL win", "WILL NOT win"],
        columns=["WILL NOT win", "WILL win"],
        fill_value=0,
    ).rename_axis(index="home", columns="away")


def next_matchweek(
    results: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Download the upcoming fixtures and call every one of them.

    The model is refitted here on the whole table rather than loaded from
    models/. It is a three-column logistic regression and takes milliseconds,
    and a file on disk can silently fall out of step with the data beside it.
    """
    from . import ingest  # local: keeps the network import off module load

    results = ingest.load_results() if results is None else results
    history = plea.run(results) if history is None else history
    if table is None:
        table = pd.read_parquet(config.PROCESSED / "features.parquet")

    fixtures = download_fixtures() if fixtures is None else fixtures
    fixtures = drop_already_played(fixtures, results)
    if fixtures.empty:
        raise NoFixtures("every listed fixture has already been played")

    rows = build_rows(fixtures, results, history)
    return verdicts(rows, model.predict(model.fit(table), rows))
