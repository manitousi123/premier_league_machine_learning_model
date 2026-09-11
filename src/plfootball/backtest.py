"""Walk-forward evaluation: does the model beat the things it is built from?

A model's score on data it was trained on is worthless — it can always recite
what it has already seen. The only honest question is how it does on a season
it has never met, so that is the only question asked here.

**Walk-forward.** Train on every season before season S, predict S, record the
result, move on to S+1. Thirteen rounds after three seasons of burn-in. The
model therefore always predicts forward in time, exactly as it would in
production, and never sees a match played after the one it is predicting.

This is why an odd/even split was rejected: training on 2016 to predict 2015
lets the model learn who was about to be good. It scores better and means less.

**The bars.** Three of them, each fitted on the same training seasons as the
model so nothing gets an unfair look at the future:

* ``base_rate`` — one number, the training win rate, for every row. The floor.
  Anything that fails to beat this has learned nothing at all.
* ``home_or_away`` — two numbers, the training win rate at home and away.
  "Always back the home side", written as a probability.
* ``plea_only`` — PLEA's expected points share mapped to a win probability by a
  one-column logistic regression. The real bar: if the whole table cannot beat
  one rating, the rest of the columns are not paying rent.

**What is measured.** Accuracy is reported because people ask for it, but it is
the least useful number here — it throws away the difference between "60% sure"
and "99% sure", and a model that never predicts an away win can still score
62%. Log loss and Brier score both grade the probability itself. The
calibration error asks the question that actually matters: when it says 70%,
does it happen 70% of the time?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from . import config, model

# --- the bars ------------------------------------------------------------
# Each takes the training seasons and the test season and returns one
# probability per test row. Every one of them is fitted on `train` alone.


def base_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The training win rate, quoted for every row regardless of who is playing."""
    return np.full(len(test), train[model.TARGET].mean())


def home_or_away(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The training win rate for home rows, and for away rows. Nothing else."""
    rates = train.groupby("is_home")[model.TARGET].mean()
    return test["is_home"].map(rates).to_numpy(dtype=float)


def plea_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """PLEA alone, converted from expected points share into a win probability.

    `elo_expected` counts a draw as half a win, so it cannot be read as a win
    probability directly — it is systematically too high. One logistic
    regression on that single column learns the conversion from the training
    seasons, which is the fairest possible version of "just use the ratings".
    """
    fitted = LogisticRegression().fit(train[["elo_expected"]], train[model.TARGET])
    return fitted.predict_proba(test[["elo_expected"]])[:, 1]


def fitted(train: pd.DataFrame, test: pd.DataFrame, params=model.DEFAULT) -> np.ndarray:
    """The model itself, retrained from scratch on each fold's training seasons."""
    return model.predict(model.fit(train, params), test).to_numpy()


CONTENDERS = {
    "base_rate": base_rate,
    "home_or_away": home_or_away,
    "plea_only": plea_only,
    "model": fitted,
}


# --- scoring -------------------------------------------------------------


def calibration_error(actual: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Mean gap between what was promised and what happened, weighted by bucket size.

    Sort the predictions into ten buckets and ask, in each, what fraction
    actually won. A perfectly calibrated model's buckets sit on the diagonal.

    A contender that quotes one probability for everything (base_rate, or
    home_or_away inside a single season) cannot be bucketed at all — there are
    no distinct values to sort. That is one bucket holding everything, and the
    gap is simply promised minus delivered.
    """
    actual = np.asarray(actual, dtype=float)
    prob = np.asarray(prob, dtype=float)
    if len(np.unique(prob)) < 2:
        return float(abs(prob.mean() - actual.mean()))

    frame = pd.DataFrame({"actual": actual, "prob": prob})
    frame["bucket"] = pd.qcut(frame["prob"], bins, labels=False, duplicates="drop")
    grouped = frame.groupby("bucket")
    gap = (grouped["prob"].mean() - grouped["actual"].mean()).abs()
    return float(np.average(gap, weights=grouped.size()))


def score(actual: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Every metric for one set of predictions."""
    actual = np.asarray(actual, dtype=int)
    prob = np.asarray(prob, dtype=float)
    constant = len(np.unique(prob)) == 1
    return {
        "n": len(actual),
        "accuracy": accuracy_score(actual, prob >= 0.5),
        "log_loss": log_loss(actual, prob, labels=[0, 1]),
        "brier": brier_score_loss(actual, prob),
        # A single repeated probability has no ranking to grade, so AUC is
        # undefined for base_rate rather than bad.
        "auc": np.nan if constant else roc_auc_score(actual, prob),
        "calibration": calibration_error(actual, prob),
    }


# --- the walk ------------------------------------------------------------


def walk_forward(
    table: pd.DataFrame,
    seasons: list[int] | None = None,
    *,
    contenders: dict | None = None,
    params: model.ModelParams = model.DEFAULT,
    verbose: bool = False,
) -> pd.DataFrame:
    """Predict each test season using only the seasons before it.

    Returns one row per predicted match-perspective, carrying the true answer
    and every contender's probability side by side, so all of them are graded
    on identical rows.
    """
    contenders = contenders or CONTENDERS
    available = sorted(table["season"].unique())
    seasons = seasons or [s for s in config.TEST_SEASONS if s in available]

    folds = []
    for test_season in seasons:
        train = table[table["season"] < test_season]
        test = table[table["season"] == test_season]
        if train.empty or test.empty:
            continue

        fold = test[["date", "season", "team", "opponent", "is_home", model.TARGET]].copy()
        for name, contender in contenders.items():
            fold[f"p_{name}"] = (
                contender(train, test, params) if name == "model" else contender(train, test)
            )
        folds.append(fold)

        if verbose:
            span = f"{train['season'].min()}-{test_season - 1}"
            print(
                f"  {config.season_label(test_season)}  trained on {span} "
                f"({len(train):,} rows) -> {len(test):,} predictions"
            )

    if not folds:
        raise ValueError(f"no season in {seasons} has any earlier season to train on")
    return pd.concat(folds, ignore_index=True)


def _contender_columns(predictions: pd.DataFrame) -> list[str]:
    return [c[2:] for c in predictions.columns if c.startswith("p_")]


def summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Every contender scored over all test seasons at once."""
    actual = predictions[model.TARGET].to_numpy()
    rows = [
        {"contender": name, **score(actual, predictions[f"p_{name}"].to_numpy())}
        for name in _contender_columns(predictions)
    ]
    return pd.DataFrame(rows).sort_values("log_loss", ignore_index=True)


def by_season(predictions: pd.DataFrame, metric: str = "log_loss") -> pd.DataFrame:
    """One metric per contender, season by season — a steady edge or one lucky year?"""
    names = _contender_columns(predictions)
    rows = []
    for season, block in predictions.groupby("season"):
        actual = block[model.TARGET].to_numpy()
        row = {"season": config.season_label(season)}
        row.update({n: score(actual, block[f"p_{n}"].to_numpy())[metric] for n in names})
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(
    predictions: pd.DataFrame, name: str = "model", bins: int = 10
) -> pd.DataFrame:
    """Promised versus delivered, bucket by bucket. The table behind the number."""
    frame = predictions[[model.TARGET]].copy()
    frame["prob"] = predictions[f"p_{name}"]
    frame["bucket"] = pd.qcut(frame["prob"], bins, labels=False, duplicates="drop")

    out = (
        frame.groupby("bucket")
        .agg(rows=("prob", "size"), predicted=("prob", "mean"), actual=(model.TARGET, "mean"))
        .reset_index(drop=True)
    )
    out["gap"] = out["predicted"] - out["actual"]
    return out


def row_log_loss(actual: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Log loss for each row on its own, rather than averaged away."""
    actual = np.asarray(actual, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-15, 1 - 1e-15)
    return -(actual * np.log(prob) + (1 - actual) * np.log(1 - prob))


def paired_test(predictions: pd.DataFrame, challenger: str, incumbent: str) -> dict[str, float]:
    """Is the challenger's log-loss edge real, or is it the size of the noise?

    Two models graded on the same rows can be compared row by row, which is far
    more sensitive than comparing two averages: the hard matches are hard for
    both, and pairing cancels that out.

    The rows are not independent, though — every fixture appears twice, once
    from each club's side, and those two rows always agree about who was hard
    to predict. Treating them as two observations would halve the standard
    error and manufacture significance. So the differences are averaged within
    each match first, and the match is the unit of evidence.

    Returns a negative `mean_gain` when the challenger is worse. `z` beyond
    about ±2 is the usual threshold for "not chance".
    """
    delta = row_log_loss(predictions[model.TARGET], predictions[f"p_{incumbent}"]) - row_log_loss(
        predictions[model.TARGET], predictions[f"p_{challenger}"]
    )
    # Sorting the pair makes the key identical from either club's side.
    pair = np.sort(predictions[["team", "opponent"]].to_numpy().astype(str), axis=1)
    key = predictions["date"].astype(str) + "|" + pair[:, 0] + "|" + pair[:, 1]
    per_match = pd.Series(delta, index=predictions.index).groupby(key.to_numpy()).mean()

    n = len(per_match)
    stderr = per_match.std(ddof=1) / np.sqrt(n)
    return {
        "matches": n,
        "mean_gain": float(per_match.mean()),
        "stderr": float(stderr),
        "z": float(per_match.mean() / stderr) if stderr else np.nan,
    }


def match_view(predictions: pd.DataFrame, name: str = "model") -> pd.DataFrame:
    """The two rows of a fixture put back together into one verdict.

    Each match was scored twice — once from the home club's side, once from the
    away club's. Rejoining them gives the four-way call the project is for, plus
    a free consistency check: the two win probabilities and the implied draw
    chance have to add to 1, so `p_home + p_away` above 1 means the model has
    contradicted itself.
    """
    column = f"p_{name}"
    home = predictions[predictions["is_home"] == 1].rename(
        columns={"team": "home_team", "opponent": "away_team", column: "p_home",
                 model.TARGET: "home_won"}
    )[["date", "season", "home_team", "away_team", "p_home", "home_won"]]

    away = predictions[predictions["is_home"] == 0].rename(
        columns={"team": "away_team", "opponent": "home_team", column: "p_away",
                 model.TARGET: "away_won"}
    )[["date", "home_team", "away_team", "p_away", "away_won"]]

    merged = home.merge(away, on=["date", "home_team", "away_team"], how="inner")
    if len(merged) != len(home):
        raise ValueError(
            f"{len(home)} home rows but only {len(merged)} matched an away row — "
            "the two perspectives of some fixture disagree on date or club names"
        )

    merged["p_draw"] = 1.0 - merged["p_home"] - merged["p_away"]
    merged["verdict"] = np.where(
        merged["p_home"] >= 0.5,
        "HOME will win",
        np.where(merged["p_away"] >= 0.5, "AWAY will win", "HOME will not win"),
    )
    merged["outcome"] = np.where(
        merged["home_won"] == 1,
        "home win",
        np.where(merged["away_won"] == 1, "away win", "draw"),
    )
    return merged


def permutation_ranking(
    trained, test: pd.DataFrame, *, repeats: int = 5, random_state: int = 0
) -> pd.DataFrame:
    """How much worse the model gets when one column is shuffled into nonsense.

    The honest counterpart to a coefficient: rather than reading what the model
    says it weighs, this destroys a column and measures what actually breaks. A
    feature that can be scrambled without hurting anything is not being used,
    whatever its coefficient claims.
    """
    result = permutation_importance(
        trained,
        model.design_matrix(test),
        test[model.TARGET].astype(int),
        scoring="neg_log_loss",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": model.FEATURES,
            "log_loss_cost": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("log_loss_cost", ascending=False, ignore_index=True)
