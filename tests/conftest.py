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
# signal and `is_home` a weaker real one (home sides win more often here, as
# they do in life); the rest are structural. Every other feature is pure noise,
# which is what lets a test assert that a model ignored it.
MEANINGFUL = ["elo_expected", "is_home", "crowd", "played"]
NOISE = [c for c in model.FEATURES if c not in MEANINGFUL]


def make_table(seasons=(1, 2, 3), matches_per_season: int = 120, seed: int = 0) -> pd.DataFrame:
    """One row per club per match, two rows per fixture, every FEATURES column present."""
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
            block = pd.DataFrame({c: rng.normal(size=n) for c in model.FEATURES})
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
            block["target"] = (home_won if is_home else away_won).astype(int)
            frames.append(block)

    table = pd.concat(frames, ignore_index=True)
    return table[model.IDENTITY + model.FEATURES + [model.TARGET]].sort_values(
        ["date", "team"], kind="stable", ignore_index=True
    )


@pytest.fixture
def table() -> pd.DataFrame:
    return make_table()
