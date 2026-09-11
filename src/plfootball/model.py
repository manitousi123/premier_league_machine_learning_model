"""Trains the win / not-win model and turns a table row into a probability.

The table is the whole specification. Every column named in ``FEATURES`` goes
in exactly as ``features.parquet`` holds it — nothing scaled, nothing encoded,
nothing dropped on the way. A tree doesn't care about units, and every column
is already numeric and gap-free by construction.

What the forest actually learns, in one sentence: given rows already labelled
won / didn't-win, each tree searches every column at every threshold for the
split that best separates the two, keeps the best one, and repeats on each
half; four hundred trees do this on different random slices of the rows and
columns, and the share voting "win" is the probability that comes out. No
football goes in. Only which numbers moved together.

``min_samples_leaf`` is the knob that matters. Left at scikit-learn's default
of 1 a forest will keep splitting until every leaf is a single match, which is
memorising rather than learning: it scores beautifully on data it has seen and
returns confident nonsense on data it hasn't. Holding leaves to 25 rows means
every probability it quotes is backed by at least 25 real matches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Columns that identify a row rather than describe it. Kept in the table for
# joining and inspection, never shown to the model: a club name would let it
# learn "Manchester City usually win" instead of why they usually win, and a
# date would let it learn the calendar. `matchweek` is left out too, being
# `played + 1` — the same column twice under two names.
IDENTITY = ["date", "season", "team", "opponent", "matchweek"]

TARGET = "target"

# The model's input contract, pinned by hand. If features.py grows a column it
# does not silently become a feature — a test compares this list against the
# table and fails, which is the moment to decide whether the new column belongs.
FEATURES = [
    # the fixture itself
    "is_home", "day_of_week", "rest_days", "opp_rest_days", "rest_days_gap", "crowd",
    # PLEA
    "own_elo", "opp_elo", "elo_gap", "elo_expected",
    # how this club has been playing
    "form_gf", "form_ga", "form_shots", "form_sot", "form_corners", "form_points",
    # how the opponent has been playing
    "opp_form_gf", "opp_form_ga", "opp_form_shots", "opp_form_sot",
    "opp_form_corners", "opp_form_points",
    # ratios the tree would otherwise have to approximate one split at a time
    "form_accuracy", "form_finishing", "opp_form_accuracy", "opp_form_finishing",
    # season to date
    "played", "ppg", "position", "points",
    # what is still at stake - the one thing here PLEA structurally cannot know
    "points_from_title", "points_from_top4", "points_from_safety",
    "title_live", "top4_live", "relegation_live", "stakes_live",
    "opp_points_from_top4", "opp_points_from_safety", "opp_stakes_live",
]


@dataclass(frozen=True)
class ModelParams:
    """Everything that decides what the model becomes. Recorded alongside it."""

    version: str = "v1"
    n_estimators: int = 400
    # Chosen on the 2013/14-2018/19 folds only, so the seasons the model is
    # finally reported on had no say in picking them. The honest finding is
    # that it barely matters: every setting tried landed within 0.012 log loss
    # of every other. See README, "What the backtest found".
    min_samples_leaf: int = 50
    max_features: str | float = 0.4
    random_state: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = ModelParams()


class TableMismatch(ValueError):
    """The table doesn't carry what the model was trained on. Fix the table."""


def design_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """The FEATURES columns, in order, checked.

    Order matters: a forest identifies columns by position, so a table with the
    right names in the wrong order would train and predict happily on shuffled
    meanings. Selecting by `FEATURES` fixes the order at both ends.
    """
    missing = [c for c in FEATURES if c not in table.columns]
    if missing:
        raise TableMismatch(f"table is missing {len(missing)} feature column(s): {missing}")

    matrix = table[FEATURES]
    blank = matrix.columns[matrix.isna().any()].tolist()
    if blank:
        raise TableMismatch(
            f"{blank} contain NaN — features.py is meant to leave no gaps, so this is a bug there"
        )
    return matrix


def fit(table: pd.DataFrame, params: ModelParams = DEFAULT) -> RandomForestClassifier:
    """Train on every row of `table`. Whatever you pass is what it learns from.

    This function has no idea which seasons it is being handed, and that is
    deliberate — keeping the train/test split outside the model is what makes
    the walk-forward in backtest.py the only place it can go wrong.
    """
    if TARGET not in table.columns:
        raise TableMismatch(f"table has no '{TARGET}' column — nothing to learn from")

    forest = RandomForestClassifier(
        n_estimators=params.n_estimators,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        random_state=params.random_state,
        n_jobs=-1,
    )
    forest.fit(design_matrix(table), table[TARGET].astype(int))
    forest.plea_params_ = params  # travels with the file, so a saved model says what made it
    return forest


def predict(model: RandomForestClassifier, table: pd.DataFrame) -> pd.Series:
    """Probability that each row's club wins that match.

    One number per row, so a fixture gets two: the home club's chance from its
    own row and the away club's from its. See backtest.match_view for the pair
    put back together.
    """
    prob = model.predict_proba(design_matrix(table))[:, 1]
    return pd.Series(prob, index=table.index, name="p_win")


def importance(model: RandomForestClassifier) -> pd.DataFrame:
    """Each feature's share of the splitting work, largest first.

    Read this as "how often the forest reached for this column", not "how much
    this column matters". It flatters columns with many distinct values, which
    simply offer more thresholds to try; backtest.py's permutation check is the
    honest version.
    """
    return (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def save(model: RandomForestClassifier, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: Path) -> RandomForestClassifier:
    return joblib.load(Path(path))
