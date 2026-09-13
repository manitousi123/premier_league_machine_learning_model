"""Paths and project-wide constants.

Everything that needs to know *where* things live asks this module, so no other
file ever hard-codes a path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- paths ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # exactly as downloaded, never hand-edited
PROCESSED = DATA / "processed"  # cleaned, feature-engineered
BENCHMARK = DATA / "benchmark"  # frozen eval set, committed to git
# The prediction log is the one thing here that cannot be rebuilt. Every
# other file under data/ can be regenerated from the raw results; a record of
# what we said before a match was played cannot be. It is tracked in git.
PREDICTIONS = DATA / "predictions"

MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments" / "runs"

for _d in (RAW, PROCESSED, BENCHMARK, PREDICTIONS, MODELS, EXPERIMENTS):
    _d.mkdir(parents=True, exist_ok=True)


# --- seasons -------------------------------------------------------------

# A season is named by the calendar year it *ends* in: 2011 is 2010/11.
PLEA_START = 2011  # ratings replay from here
TRAIN_START = 2011  # model trains on the same range — see note below
CURRENT_SEASON = 2027  # 2026/27, not published yet

# We looked at starting in 2017/18 to pick up expected goals, but xG isn't
# available from a source we can use. Shots on target carries most of the same
# signal (+0.417 vs xG's +0.451 against points won), and staying at 2010/11
# nearly doubles the training data. If xG ever arrives it joins on as an extra
# optional column — no restructuring needed.

PLEA_SEASONS = list(range(PLEA_START, CURRENT_SEASON + 1))
TRAIN_SEASONS = list(range(TRAIN_START, CURRENT_SEASON + 1))

# Walk-forward: three seasons of burn-in, then test each season in turn.
BURN_IN_SEASONS = 3
TEST_SEASONS = list(range(TRAIN_START + BURN_IN_SEASONS, CURRENT_SEASON))

# --- rolling form ---
FORM_WINDOW = 5  # matches, always strictly before the one being predicted

# --- cold-start priors -------------------------------------------------
# What a club's numbers are assumed to be before it has any history to go on.
# Measured from 2010/11-2025/26 and then frozen: recomputing them from the data
# being processed would leak, and would make output depend on the seasons loaded.

PPG_SHRINKAGE = 5.0        # matches of prior weight in the ppg blend
PROMOTED_PPG = 1.07        # promoted clubs' average, 65 cases
PROMOTED_POSITION = 15     # promoted clubs' median finish

LEAGUE_AVERAGE_FORM = {
    "gf": 1.40,
    "ga": 1.40,
    "shots": 12.74,
    "sot": 4.89,
    "corners": 5.30,
    "points_share": 0.50,
}

MAX_REST_DAYS = 14         # beyond a fortnight, more rest stops meaning anything

# --- stakes: what a club still has to play for ---------------------------
# PLEA knows how good a club is but not whether it still cares. A side already
# safe, already relegated, or already champion plays differently in May, and
# that is where the ratings lose most ground to the betting market.

SEASON_MATCHES = 38     # matches each club plays
SEASON_FIXTURES = 380   # matches in the whole division: 20 clubs x 38 / 2

TITLE_POSITION = 1        # the club whose points you must reach to win it
TOP4_POSITION = 4         # Champions League cut-off
SAFETY_POSITION = 17      # the last club above the drop
RELEGATION_POSITION = 18  # the first club below it

# --- covid: matches played to empty or near-empty grounds ---
# 2019/20 resumed behind closed doors on 17 Jun 2020; crowds only returned for
# the last two matchweeks of 2020/21. 452 matches, and home advantage went to
# roughly zero across them. They stay in the data with a flag rather than being
# deleted — removing mid-season matches would punch holes in the rolling-form
# windows of the matches either side.
NO_CROWD_START = "2020-06-17"
NO_CROWD_END = "2021-05-16"


def season_label(end_year: int) -> str:
    """2011 -> '2010/11'."""
    return f"{end_year - 1}/{str(end_year)[2:]}"


def season_of(date) -> int:
    """Which season a date falls in, named by the year it ends in.

    The league year turns over in the summer, so anything from July onwards
    belongs to the season ending the following year: 2 Aug 2026 is 2026/27,
    and so is 2 May 2027.
    """
    date = pd.Timestamp(date)
    return date.year + 1 if date.month >= 7 else date.year


# --- known baselines -----------------------------------------------------
# Used by the sanity checks in quality.py. If the data disagrees with these,
# something upstream is broken.

HOME_WIN_RATE = 0.45
AWAY_WIN_RATE = 0.30
DRAW_RATE = 0.25
SANITY_TOLERANCE = 0.04
