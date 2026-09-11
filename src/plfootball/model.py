"""Trains the win / not-win model and turns a table row into a probability.

Three columns go in. PLEA's expectation for the fixture, and how much more of
the game this club has been creating than its opponent — shots on target, and
shots. Out comes the probability that this club wins this match.

**Why so few.** The table offers forty-two columns and this reads three of
them, which was not the plan. It is what the walk-forward measured: dropping
any other group of columns changed nothing anyone could detect, and the whole
table together was no better than PLEA on its own. Only the shot counts ever
cleared the bar. A column that earns nothing is not free — it is one more thing
for the model to mistake for signal.

**Why a regression and not the forest this started as.** Also measured. A
random forest on all forty-two columns scored 0.5866; this scores 0.5830; the
same three columns handed to a forest scored 0.5892, which is *worse than doing
nothing*. The relationship here is smooth and close to monotone — more shots on
target than your opponent, better chance of winning — and a regression spends
one coefficient on that. A forest has to build the same curve out of staircase
steps, and on a signal this weak the noise in the steps costs more than the
flexibility is worth.

**Why gaps rather than both sides separately.** A club's own shot count partly
measures who it happened to play. The difference cancels that, and it keeps the
two rows of a fixture exact mirrors of each other.

The columns are scaled before fitting because a regression's coefficients are
in the units of their column, and `elo_expected` runs 0 to 1 while `shots_gap`
runs about -15 to +15. Scaling makes the penalty fall on all three equally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# Columns that identify a row rather than describe it. Kept in the table for
# joining and inspection, never offered to a model: a club name would let it
# learn "Manchester City usually win" instead of why they usually win, and a
# date would let it learn the calendar. `matchweek` is left out too, being
# `played + 1` — the same column twice under two names.
IDENTITY = ["date", "season", "team", "opponent", "matchweek"]

TARGET = "target"

# Everything the table offers a model to choose from — the menu, not the meal.
# Pinned by hand so that a new column in features.py does not quietly become an
# input: a test compares this list against the real table and fails, which is
# the moment to decide whether the new column belongs.
CANDIDATES = [
    # the fixture itself
    "is_home", "day_of_week", "rest_days", "opp_rest_days", "rest_days_gap", "crowd",
    # PLEA
    "own_elo", "opp_elo", "elo_gap", "elo_expected",
    # how this club has been playing
    "form_gf", "form_ga", "form_shots", "form_sot", "form_corners", "form_points",
    # how the opponent has been playing
    "opp_form_gf", "opp_form_ga", "opp_form_shots", "opp_form_sot",
    "opp_form_corners", "opp_form_points",
    # the two of those taken against each other
    "sot_gap", "shots_gap",
    # ratios a tree would otherwise have to approximate one split at a time
    "form_accuracy", "form_finishing", "opp_form_accuracy", "opp_form_finishing",
    # season to date
    "played", "ppg", "position", "points",
    # what is still at stake
    "points_from_title", "points_from_top4", "points_from_safety",
    "title_live", "top4_live", "relegation_live", "stakes_live",
    "opp_points_from_top4", "opp_points_from_safety", "opp_stakes_live",
]

# What the shipped model actually reads. Chosen on the 2013/14-2018/19 folds
# alone, so the seasons it is finally reported on had no say in picking it.
FEATURES = ["elo_expected", "sot_gap", "shots_gap"]


@dataclass(frozen=True)
class ModelParams:
    """Everything that decides what the model becomes. Recorded alongside it."""

    version: str = "v2"
    # Smaller C means a stronger pull of the coefficients toward zero. Left at
    # the default: with three columns and eleven thousand rows there is very
    # little to overfit.
    penalty_c: float = 1.0
    max_iter: int = 2000

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = ModelParams()


class TableMismatch(ValueError):
    """The table doesn't carry what the model was trained on. Fix the table."""


def design_matrix(table: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """The model's columns, in order, checked.

    Order matters: a fitted model identifies columns by position, so a table
    with the right names in the wrong order would train and predict happily on
    shuffled meanings. Selecting by `FEATURES` fixes the order at both ends.
    """
    columns = columns or FEATURES
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise TableMismatch(f"table is missing {len(missing)} column(s): {missing}")

    matrix = table[columns]
    blank = matrix.columns[matrix.isna().any()].tolist()
    if blank:
        raise TableMismatch(
            f"{blank} contain NaN — features.py is meant to leave no gaps, so this is a bug there"
        )
    return matrix


def fit(table: pd.DataFrame, params: ModelParams = DEFAULT) -> Pipeline:
    """Train on every row of `table`. Whatever you pass is what it learns from.

    This function has no idea which seasons it is being handed, and that is
    deliberate — keeping the train/test split outside the model is what makes
    the walk-forward in backtest.py the only place it can go wrong.
    """
    if TARGET not in table.columns:
        raise TableMismatch(f"table has no '{TARGET}' column — nothing to learn from")

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=params.penalty_c, max_iter=params.max_iter),
    )
    pipeline.fit(design_matrix(table), table[TARGET].astype(int))
    pipeline.plea_params_ = params  # travels with the file, so a saved model says what made it
    return pipeline


def predict(fitted: Pipeline, table: pd.DataFrame) -> pd.Series:
    """Probability that each row's club wins that match.

    One number per row, so a fixture gets two: the home club's chance from its
    own row and the away club's from its. See backtest.match_view for the pair
    put back together.
    """
    prob = fitted.predict_proba(design_matrix(table))[:, 1]
    return pd.Series(prob, index=table.index, name="p_win")


def coefficients(fitted: Pipeline) -> pd.DataFrame:
    """What the model learned, in the open.

    One number per column, and because the columns were scaled first they are
    directly comparable: the largest is the one doing the most work. `odds_x`
    is the same number as a multiplier — how the odds of winning move for a one
    standard deviation rise in that column, everything else held still.

    This is the part a forest could never give you. Four hundred trees have no
    summary; three coefficients are the whole model.
    """
    weights = fitted.named_steps["logisticregression"].coef_[0]
    return (
        pd.DataFrame({
            "feature": FEATURES,
            "coefficient": weights,
            "odds_x": np.exp(weights),
        })
        .assign(strength=lambda d: d["coefficient"].abs())
        .sort_values("strength", ascending=False, ignore_index=True)
        .drop(columns="strength")
    )


def save(fitted: Pipeline, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)
    return path


def load(path: Path) -> Pipeline:
    return joblib.load(Path(path))
