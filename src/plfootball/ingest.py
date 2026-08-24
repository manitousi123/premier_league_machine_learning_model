"""Fetch and load Premier League results.

Source: football-data.co.uk, one CSV per season. Free, no key, no rate limit,
back to 1993. Provides everything PLEA needs (date, both clubs, both scores)
plus shots, corners and cards, which we keep because they cost nothing.

xG is *not* here — that comes from FBref later, for the 2017/18-onward training
table only.

Raw files land in data/raw/ exactly as downloaded and are never edited. Every
fix happens on the way out, in load_results().
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from . import config
from .teams import canonicalise

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Source column -> our name. Anything not listed is dropped on load.
_COLUMNS = {
    "Date": "date",
    "Time": "kickoff",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "Referee": "referee",
}

_REQUIRED = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]


def season_code(end_year: int) -> str:
    """2026 -> '2526' (the 2025/26 season)."""
    return f"{str(end_year - 1)[2:]}{str(end_year)[2:]}"


def raw_path(end_year: int):
    return config.RAW / f"E0_{season_code(end_year)}.csv"


class SeasonUnavailable(RuntimeError):
    """No published file for this season yet — normal for a season not started."""


def _looks_like_results_csv(content: bytes) -> bool:
    """Guard against the server answering with a web page.

    A season that doesn't exist yet returns HTTP 300 with an HTML "Multiple
    Choices" body, not a 404 — so raise_for_status() sails straight past it and
    we'd save a web page as a CSV. Check the content, not the status code.
    """
    head = content[:200].lstrip().lower()
    return b"hometeam" in content[:400].lower() and not head.startswith(b"<!doctype")


def download_season(end_year: int, *, force: bool = False) -> bool:
    """Download one season's raw CSV. Returns True if it hit the network."""
    path = raw_path(end_year)
    if path.exists() and not force:
        return False

    url = BASE_URL.format(code=season_code(end_year))
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    if not _looks_like_results_csv(response.content):
        raise SeasonUnavailable(f"{url} returned no results table (season not published yet)")

    path.write_bytes(response.content)
    return True


def download_all(seasons=None, *, force: bool = False) -> dict[int, str]:
    """Download every season we need. Skips what's already on disk."""
    seasons = seasons or config.PLEA_SEASONS
    outcome = {}
    for end_year in seasons:
        try:
            fetched = download_season(end_year, force=force)
            outcome[end_year] = "downloaded" if fetched else "cached"
            if fetched:
                time.sleep(0.5)  # be polite to a free service
        except SeasonUnavailable:
            raw_path(end_year).unlink(missing_ok=True)
            outcome[end_year] = "not published yet"
        except requests.HTTPError as exc:
            outcome[end_year] = f"unavailable (HTTP {exc.response.status_code})"
    return outcome


def _load_one(end_year: int) -> pd.DataFrame:
    path = raw_path(end_year)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run download_all() first")

    raw = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    present = {src: dst for src, dst in _COLUMNS.items() if src in raw.columns}
    df = raw[list(present)].rename(columns=present)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    # Some files carry blank padding rows past the end of the season.
    df = df.dropna(subset=["home_team", "away_team"])

    df["season"] = end_year
    return df


def load_results(seasons=None) -> pd.DataFrame:
    """Every match, cleaned just enough for PLEA.

    One row per match. Club names are canonical; an unknown name raises.
    Sorted strictly by date then kickoff — PLEA is order-dependent, so a match
    out of sequence corrupts every rating after it.
    """
    seasons = seasons or config.PLEA_SEASONS
    frames = [_load_one(y) for y in seasons if raw_path(y).exists()]
    if not frames:
        raise FileNotFoundError("no raw season files found — run download_all() first")

    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["home_team"] = canonicalise(df["home_team"])
    df["away_team"] = canonicalise(df["away_team"])

    for col in ("home_goals", "away_goals"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df[["home_goals", "away_goals"]] = df[["home_goals", "away_goals"]].astype(int)

    if "kickoff" not in df.columns:
        df["kickoff"] = pd.NA

    df = df.sort_values(["date", "kickoff"], kind="stable", na_position="first")
    return df.reset_index(drop=True)
