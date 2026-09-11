"""A synthetic feature table, shaped exactly like the real one.

Real data would make these tests slow and, worse, would make them agree with
whatever the data happened to do that year. The table built here has a signal
deliberately planted in `elo_expected` and pure noise everywhere else, so a
test can assert that a model finds the signal and ignores the noise without
depending on sixteen seasons of football having gone a particular way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plfootball import model

CLUBS = [f"club_{i:02d}" for i in range(10)]

# Columns make_table fills deliberately. `elo_expected` carries the planted
# signal, the two shot gaps a weaker version of it, and `is_home` a real one
# (home sides win more often here, as they do in life); the rest are
# structural. Every other candidate column is pure noise, which is what lets a
# test assert that a model ignored it.
MEANINGFUL = [
    "elo_expected", "sot_gap", "shots_gap", "is_home", "crowd", "played",
]
NOISE = [c for c in model.CANDIDATES if c not in MEANINGFUL]


def make_table(seasons=(1, 2, 3), matches_per_season: int = 120, seed: int = 0) -> pd.DataFrame:
    """One row per club per match, two rows per fixture, every table column present."""
    rng = np.random.default_rng(seed)
    frames = []

    for season in seasons:
        n = matches_per_season
        dates = pd.to_datetime(f"20{50 + season:02d}-08-01") + pd.to_timedelta(
            np.arange(n) // 5, unit="D"
        )
        home = [CLUBS[i % len(CLUBS)] for i in range(n)]
        away = [CLUBS[(i + 3 + i // len(CLUBS)) % len(CLUBS)] for i in range(n)]

        # The planted signal: the home side's real chance of winning.
        chance = rng.uniform(0.12, 0.85, n)
        home_won = rng.random(n) < chance
        # Whoever didn't win the match either drew it or lost it.
        away_won = (~home_won) & (rng.random(n) < 0.55)

        for is_home in (1, 0):
            block = pd.DataFrame({c: rng.normal(size=n) for c in model.CANDIDATES})
            block["date"] = dates
            block["season"] = season
            block["team"] = home if is_home else away
            block["opponent"] = away if is_home else home
            block["matchweek"] = np.arange(n) // len(CLUBS) + 1
            block["is_home"] = is_home
            block["crowd"] = 1
            block["played"] = block["matchweek"] - 1
            # elo_expected carries the signal; a little jitter keeps it from
            # being a perfect giveaway, which no real feature ever is.
            own = chance if is_home else 1 - chance
            block["elo_expected"] = np.clip(own + rng.normal(0, 0.05, n), 0.02, 0.98)
            # the shot gaps are mirrored between the two rows of a fixture, as
            # a difference always is, and carry a weaker version of the signal
            side = 1 if is_home else -1
            block["sot_gap"] = side * (chance - 0.5) * 6 + rng.normal(0, 1.5, n)
            block["shots_gap"] = side * (chance - 0.5) * 9 + rng.normal(0, 3.0, n)
            block["target"] = (home_won if is_home else away_won).astype(int)
            frames.append(block)

    table = pd.concat(frames, ignore_index=True)
    return table[model.IDENTITY + model.CANDIDATES + [model.TARGET]].sort_values(
        ["date", "team"], kind="stable", ignore_index=True
    )


@pytest.fixture
def table() -> pd.DataFrame:
    return make_table()
