"""Small frames shared by the dashboard and app tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plfootball import predict

CLUBS = ["Arsenal", "Chelsea", "Everton", "Fulham"]


def tiny_results(n_rounds: int = 4, start: str = "2026-08-15", season: int = 2027) -> pd.DataFrame:
    """A complete little results frame: every club plays every round, weekly."""
    rng = np.random.default_rng(0)
    rows = []
    for week in range(n_rounds):
        date = pd.Timestamp(start) + pd.Timedelta(days=7 * week)
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


def calls_from(rows) -> pd.DataFrame:
    """Calls as the predictor returns them, from (date, kickoff, home, away, p_home, p_away).

    Goes through `predict.verdicts` so the verdict logic is the real one.
    """
    recs, probs = [], []
    for date, kickoff, home, away, p_home, p_away in rows:
        base = {"date": pd.Timestamp(date), "kickoff": kickoff}
        recs.append({**base, "team": home, "opponent": away, "is_home": 1})
        probs.append(p_home)
        recs.append({**base, "team": away, "opponent": home, "is_home": 0})
        probs.append(p_away)
    return predict.verdicts(pd.DataFrame(recs), pd.Series(probs))


def season_fixtures(rounds: dict) -> pd.DataFrame:
    """{round: [(date, home, away), ...]} in the shape fixtures.parse returns."""
    recs = []
    for number, matches in rounds.items():
        for date, home, away in matches:
            recs.append({
                "round": number, "date": pd.Timestamp(date), "kickoff": "15:00",
                "home_team": home, "away_team": away,
            })
    return pd.DataFrame(recs)
