"""Every round of the season, so a matchweek ahead can be looked at.

football-data.co.uk's fixtures.csv lists only the next few days. That is the
right source for anything that gets *logged*: it is the same publisher as the
results, so a fixture's date there is the date its result will arrive under,
and `track.settle` matches on date. But it cannot show the round after next.

fixturedownload.com publishes the whole season with a round number against
every match, which is what a look-ahead needs. It is a third party, so it is
held at arm's length: nothing from it is written to the log, and the scores it
carries are ignored entirely. Results come from one place.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from . import config, ingest
from .teams import canonicalise

# The season is named by the year it starts in here, not the year it ends.
SEASON_URL = "https://fixturedownload.com/download/epl-{start_year}-GMTStandardTime.csv"

COLUMNS = ["round", "date", "kickoff", "home_team", "away_team"]


class NoSeasonFixtures(RuntimeError):
    """No fixture file on disk for this season, and the download did not work."""


def raw_path(season: int) -> Path:
    return config.RAW / f"fixtures_{ingest.season_code(season)}.csv"


def parse(raw: pd.DataFrame) -> pd.DataFrame:
    """The published table cut down to what a preview needs.

    The score column is dropped on purpose - see the module docstring. Club
    names go through the same canonical map as everything else, so an unknown
    spelling stops here rather than becoming a fixture nobody can match.
    """
    needed = {"Round Number", "Date", "Home Team", "Away Team"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"fixture file is missing columns: {sorted(missing)}")

    when = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")
    fixtures = pd.DataFrame({
        "round": pd.to_numeric(raw["Round Number"]).astype(int),
        "date": when.dt.normalize(),
        "kickoff": when.dt.strftime("%H:%M"),
        "home_team": canonicalise(raw["Home Team"]),
        "away_team": canonicalise(raw["Away Team"]),
    })
    return (
        fixtures.sort_values(["date", "kickoff", "home_team"], kind="stable")
        .reset_index(drop=True)
    )


def download(season: int | None = None) -> pd.DataFrame:
    """Fetch the season's fixtures and keep a copy on disk.

    Always hits the network. The file is thirty kilobytes, and the dates in it
    move whenever a match is picked for television - a change that cannot be
    seen from the copy on disk, so there is nothing to cache against.
    """
    season = season or config.season_of(pd.Timestamp.now())
    url = SEASON_URL.format(start_year=season - 1)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    raw = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")
    fixtures = parse(raw)  # a bad file is rejected before anything is written

    path = raw_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return fixtures


def load(season: int | None = None, *, fetch_if_missing: bool = True) -> pd.DataFrame:
    """The copy on disk, or a fresh download if there is none yet."""
    season = season or config.season_of(pd.Timestamp.now())
    path = raw_path(season)
    if path.exists():
        return parse(pd.read_csv(path, encoding="utf-8-sig"))
    if not fetch_if_missing:
        raise NoSeasonFixtures(f"no fixture file for {config.season_label(season)}")
    try:
        return download(season)
    except requests.RequestException as exc:
        raise NoSeasonFixtures(
            f"no fixture file for {config.season_label(season)} and the download failed: {exc}"
        ) from exc


def played_mask(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.Series:
    """Which of these fixtures already have a result.

    Judged against the results file, never against the fixture file's own
    score column. Matched on the two clubs rather than the date: a match moved
    for television keeps its pairing and loses its date, and each pairing
    happens once a season at that ground.
    """
    if fixtures.empty:
        return pd.Series(dtype=bool, index=fixtures.index)

    season = config.season_of(fixtures["date"].min())
    block = results[results["season"] == season] if "season" in results.columns else results
    pairs = set(zip(block["home_team"], block["away_team"]))
    return pd.Series(
        [(h, a) in pairs for h, a in zip(fixtures["home_team"], fixtures["away_team"])],
        index=fixtures.index,
    )


def unplayed(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    return fixtures[~played_mask(fixtures, results)].reset_index(drop=True)


def for_round(fixtures: pd.DataFrame, number: int) -> pd.DataFrame:
    return fixtures[fixtures["round"] == number].reset_index(drop=True)


def rounds(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per round: when it falls, and how much of it has been played."""
    if fixtures.empty:
        return pd.DataFrame(
            columns=["round", "first", "last", "fixtures", "played", "remaining"]
        )

    table = (
        fixtures.assign(played=played_mask(fixtures, results))
        .groupby("round")
        .agg(
            first=("date", "min"),
            last=("date", "max"),
            fixtures=("date", "size"),
            played=("played", "sum"),
        )
        .reset_index()
    )
    table["played"] = table["played"].astype(int)
    table["remaining"] = table["fixtures"] - table["played"]
    return table


def next_round(table: pd.DataFrame) -> int | None:
    """The lowest round with anything still to play, or None when the season is done."""
    open_rounds = table[table["remaining"] > 0]
    return None if open_rounds.empty else int(open_rounds["round"].min())
