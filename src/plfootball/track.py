"""A permanent record of what was predicted, and what then happened.

Every other file this project writes can be thrown away and rebuilt from the
raw results. This one cannot. A prediction is only evidence if it was written
down *before* the match, so the log is append-mostly, lives in git, and is the
only place where the model is held to account against reality rather than
against a backtest.

The distinction matters more than it sounds. The walk-forward says the model
should get 62% of verdicts right; whether it actually does, week after week, on
fixtures nobody had seen when the code was written, is a different claim. Only
this file can settle it — and only if it is kept honestly.

Two rules enforce that honesty:

* A fixture that already has a result recorded can never be predicted again.
  A forecast written after the fact is not a forecast, and one of those in the
  log would quietly inflate everything computed from it.
* Re-predicting an *unplayed* fixture replaces the earlier call rather than
  adding a second. Running the predictor again on Thursday with fresher form is
  a better answer to the same question, not a new question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

LOG_PATH = config.PREDICTIONS / "log.csv"

# What identifies a fixture. Two clubs and a date - the same key the results
# table and the feature table both join on.
KEY = ["date", "home_team", "away_team"]

_PREDICTION_COLUMNS = [
    "predicted_at",       # when we committed to this
    "results_current_to", # how fresh the ratings were when we did
    *KEY,
    "kickoff",
    "p_home", "p_away", "p_draw",
    "home_call", "away_call", "verdict", "confidence",
    "model_version", "plea_version",
]

_OUTCOME_COLUMNS = [
    "home_goals", "away_goals", "outcome", "correct", "settled_at",
]

COLUMNS = _PREDICTION_COLUMNS + _OUTCOME_COLUMNS

_DATES = ["date", "predicted_at", "results_current_to", "settled_at"]

_TEXT = [
    "kickoff", "home_call", "away_call", "verdict",
    "model_version", "plea_version", "outcome", "correct",
]


class AlreadyPlayed(ValueError):
    """Refusing to log a prediction for a match that already has a result."""


def still_to_come(calls: pd.DataFrame, now=None) -> pd.Series:
    """Which of these fixtures had not been played when the call was made.

    The results file lags the fixture list by a few days, so a run on Sunday
    will happily offer a "prediction" for Saturday's matches: the model has not
    seen those results, so nothing leaks, and the call is genuinely blind. It
    is still not a forecast. A record that counts it as one is quietly
    measuring something easier than the thing it claims to measure.

    Same-day is allowed. Kickoffs run from lunchtime to late evening and the
    log works in whole days, so refusing them would throw away legitimate
    calls to guard against a few hours.
    """
    now = (pd.Timestamp.now() if now is None else pd.Timestamp(now)).normalize()
    return pd.to_datetime(calls["date"]).dt.normalize() >= now


def empty() -> pd.DataFrame:
    """A log with the right shape and nothing in it."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def load(path: Path | None = None) -> pd.DataFrame:
    """Read the log, or an empty one if nothing has been recorded yet."""
    path = Path(path or LOG_PATH)
    if not path.exists():
        return empty()

    log = pd.read_csv(path)
    for column in _DATES:
        if column in log.columns:
            log[column] = pd.to_datetime(log[column], errors="coerce")

    log = log.reindex(columns=COLUMNS)
    # An empty or all-blank column reads back as float64, which then refuses a
    # string. Pin the text columns to object so settle() can write into them.
    for column in _TEXT:
        log[column] = log[column].astype("object")
    return log


def save(log: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path or LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = log.reindex(columns=COLUMNS).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for column in ("predicted_at", "results_current_to", "settled_at"):
        out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in ("p_home", "p_away", "p_draw", "confidence"):
        out[column] = pd.to_numeric(out[column]).round(4)

    out.sort_values(KEY, kind="stable").to_csv(path, index=False)
    return path


def record(
    calls: pd.DataFrame,
    *,
    results_current_to,
    model_version: str,
    plea_version: str,
    path: Path | None = None,
    now=None,
) -> pd.DataFrame:
    """Write this week's calls into the log and return the whole thing.

    Raises `AlreadyPlayed` if any fixture already carries a result — see the
    module docstring for why that is a refusal rather than an overwrite.
    """
    log = load(path)
    # Floored to the second, which is what the CSV stores. Without this a
    # microsecond timestamp cannot be written into a column already read
    # back from disk at second precision.
    now = (pd.Timestamp.now() if now is None else pd.Timestamp(now)).floor("s")

    offered = calls.copy()
    offered["date"] = pd.to_datetime(offered["date"])

    # Checked before anything is filtered. A fixture that already has a result
    # in the log means something upstream is feeding stale fixtures, and that
    # deserves to be said out loud rather than quietly dropped along with the
    # ordinary already-kicked-off ones below.
    settled = log[log["outcome"].notna()]
    clash = offered.merge(settled[KEY], on=KEY, how="inner")
    if not clash.empty:
        names = ", ".join(f"{r.home_team} v {r.away_team}" for r in clash.itertuples())
        raise AlreadyPlayed(
            f"{len(clash)} fixture(s) already have a result recorded: {names}. "
            "A prediction made after the match is not a prediction."
        )

    # Anything already kicked off is dropped rather than logged. See
    # still_to_come: the call is blind, but it is not a forecast.
    fresh = offered[still_to_come(offered, now)].copy()
    if fresh.empty:
        return log

    fresh["predicted_at"] = now
    fresh["results_current_to"] = pd.Timestamp(results_current_to)
    fresh["model_version"] = model_version
    fresh["plea_version"] = plea_version

    # An unplayed fixture predicted again supersedes the earlier call.
    superseded = log.merge(fresh[KEY].assign(_drop=1), on=KEY, how="left")
    kept = log[superseded["_drop"].isna().to_numpy()]

    combined = pd.concat([kept, fresh.reindex(columns=COLUMNS)], ignore_index=True)
    save(combined, path)
    return combined.sort_values(KEY, kind="stable").reset_index(drop=True)


def verdict_held(verdict: pd.Series, outcome: pd.Series) -> pd.Series:
    """Did the call turn out right?

    A contradiction claims nothing, so it is neither right nor wrong — left as
    NA rather than counted as a miss, which would flatter the alternative of
    never flagging one.
    """
    held = pd.Series(pd.NA, index=verdict.index, dtype="object")
    held[verdict == "HOME will win"] = (outcome == "home win")[verdict == "HOME will win"]
    held[verdict == "AWAY will win"] = (outcome == "away win")[verdict == "AWAY will win"]
    held[verdict == "HOME will not win"] = (outcome != "home win")[
        verdict == "HOME will not win"
    ]
    return held


def settle(results: pd.DataFrame, path: Path | None = None, now=None) -> pd.DataFrame:
    """Fill in what actually happened, for any logged prediction now played.

    Safe to run as often as you like: a row that already has an outcome is left
    exactly as it was, so a re-run can never revise history.
    """
    log = load(path)
    if log.empty:
        return log

    # Floored to the second, which is what the CSV stores. Without this a
    # microsecond timestamp cannot be written into a column already read
    # back from disk at second precision.
    now = (pd.Timestamp.now() if now is None else pd.Timestamp(now)).floor("s")
    played = results[[*KEY, "home_goals", "away_goals"]].copy()
    played["date"] = pd.to_datetime(played["date"])

    log["date"] = pd.to_datetime(log["date"])
    merged = log.merge(played, on=KEY, how="left", suffixes=("", "_actual"))
    for column in ("outcome", "correct"):
        merged[column] = merged[column].astype("object")
    for column in ("home_goals", "away_goals"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    # Only rows that are unsettled *and* now have a result.
    newly = merged["outcome"].isna() & merged["home_goals_actual"].notna()
    if newly.any():
        home_goals = merged.loc[newly, "home_goals_actual"]
        away_goals = merged.loc[newly, "away_goals_actual"]
        outcome = np.select(
            [home_goals > away_goals, home_goals < away_goals],
            ["home win", "away win"],
            default="draw",
        )
        merged.loc[newly, "home_goals"] = home_goals
        merged.loc[newly, "away_goals"] = away_goals
        merged.loc[newly, "outcome"] = outcome
        merged.loc[newly, "correct"] = verdict_held(
            merged.loc[newly, "verdict"], pd.Series(outcome, index=home_goals.index)
        )
        merged.loc[newly, "settled_at"] = now

    updated = merged.drop(columns=["home_goals_actual", "away_goals_actual"])
    save(updated, path)
    return updated.sort_values(KEY, kind="stable").reset_index(drop=True)


# --- reading the record back ---------------------------------------------

BANDS = [0.0, 0.50, 0.55, 0.60, 0.70, 1.0]
# The lowest band is not "no call" - "HOME will not win" is a real claim and
# is scored like any other. It is the band where neither side was backed.
BAND_LABELS = ["neither backed", "50-55%", "55-60%", "60-70%", "70%+"]


def track_record(log: pd.DataFrame | None = None, path: Path | None = None) -> pd.DataFrame:
    """Hit rate by confidence band, over every prediction that has been settled.

    The bands are the point. An overall accuracy mixes the fixtures the model
    was sure about with the ones it shrugged at, and the whole claim being
    tested is that its confidence means something — that the 70% calls really
    do land more often than the 55% ones.
    """
    log = load(path) if log is None else log
    done = log[log["outcome"].notna()].copy()
    if done.empty:
        return pd.DataFrame(columns=["band", "predictions", "correct", "hit_rate"])

    done["confidence"] = pd.to_numeric(done["confidence"])
    done["band"] = pd.cut(
        done["confidence"], BANDS, labels=BAND_LABELS, include_lowest=True, right=False
    )

    scoreable = done[done["correct"].notna()].copy()
    scoreable["correct"] = scoreable["correct"].astype(bool)

    by_band = (
        scoreable.groupby("band", observed=False)
        .agg(predictions=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_band["hit_rate"] = (by_band["correct"] / by_band["predictions"]).where(
        by_band["predictions"] > 0
    )
    return by_band


def summary(log: pd.DataFrame | None = None, path: Path | None = None) -> dict:
    """The headline numbers, and the backtest figures they should be read against.

    Every prediction contributes two rows to the probability scores — the home
    club's chance against whether it won, and the away club's against whether
    it did — which is exactly how the walk-forward scored things, so the two are
    directly comparable.
    """
    log = load(path) if log is None else log
    done = log[log["outcome"].notna()].copy()

    result = {
        "logged": len(log),
        "settled": len(done),
        "awaiting_result": int(log["outcome"].isna().sum()),
    }
    if done.empty:
        return result

    scoreable = done[done["correct"].notna()]
    result["verdict_accuracy"] = float(scoreable["correct"].astype(bool).mean())
    result["called"] = len(scoreable)
    result["no_call_rate"] = float((done["verdict"] == "HOME will not win").mean())

    probability = np.concatenate([
        pd.to_numeric(done["p_home"]).to_numpy(),
        pd.to_numeric(done["p_away"]).to_numpy(),
    ])
    actual = np.concatenate([
        (done["outcome"] == "home win").to_numpy(),
        (done["outcome"] == "away win").to_numpy(),
    ]).astype(int)

    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    result["log_loss"] = float(
        -(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)).mean()
    )
    result["brier"] = float(((probability - actual) ** 2).mean())

    # What the walk-forward promised, for comparison.
    result["backtest_log_loss"] = 0.5821
    result["backtest_verdict_accuracy"] = 0.622
    return result
