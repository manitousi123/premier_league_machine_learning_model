"""Paths and project-wide constants.

Everything that needs to know *where* things live asks this module, so no other
file ever hard-codes a path.
"""

from __future__ import annotations

from pathlib import Path

# --- paths ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # exactly as downloaded, never hand-edited
PROCESSED = DATA / "processed"  # cleaned, feature-engineered
BENCHMARK = DATA / "benchmark"  # frozen eval set, committed to git

MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments" / "runs"

for _d in (RAW, PROCESSED, BENCHMARK, MODELS, EXPERIMENTS):
    _d.mkdir(parents=True, exist_ok=True)


# --- seasons -------------------------------------------------------------

# A season is named by the calendar year it *ends* in: 2011 is 2010/11.
PLEA_START = 2011  # ratings replay from here — results only, no xG needed
TRAIN_START = 2018  # model training starts here — first season with xG
CURRENT_SEASON = 2027  # 2026/27, in progress

PLEA_SEASONS = list(range(PLEA_START, CURRENT_SEASON + 1))
TRAIN_SEASONS = list(range(TRAIN_START, CURRENT_SEASON + 1))

# Walk-forward: two seasons of burn-in, then test each season in turn.
TEST_SEASONS = list(range(TRAIN_START + 2, CURRENT_SEASON))


def season_label(end_year: int) -> str:
    """2011 -> '2010/11'."""
    return f"{end_year - 1}/{str(end_year)[2:]}"


# --- known baselines -----------------------------------------------------
# Used by the sanity checks in quality.py. If the data disagrees with these,
# something upstream is broken.

HOME_WIN_RATE = 0.45
AWAY_WIN_RATE = 0.30
DRAW_RATE = 0.25
SANITY_TOLERANCE = 0.04
