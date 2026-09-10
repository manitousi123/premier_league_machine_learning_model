"""The evaluation has to be harder to fool than the model is.

If the backtest is wrong the model looks good and nobody finds out, so these
tests lean hardest on the two places that would hide a mistake: whether the
walk-forward ever trains on the future, and whether the paired test can be
talked into calling noise a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plfootball import backtest, model

from .conftest import make_table


def flat(value: float):
    """A contender that quotes the same probability for everything."""
    return lambda train, test: np.full(len(test), value)


# --- the leak ------------------------------------------------------------


def test_walk_forward_never_trains_on_the_test_season(table):
    """The one thing that would invalidate every number in the project."""
    seen = []

    def spy(train, test):
        seen.append((test["season"].iloc[0], train["season"].max()))
        return np.full(len(test), 0.5)

    backtest.walk_forward(table, [2, 3], contenders={"spy": spy})

    assert len(seen) == 2
    for test_season, newest_training_season in seen:
        assert newest_training_season < test_season


def test_walk_forward_never_trains_on_a_row_it_will_predict(table):
    """Stronger than the season check: no individual row appears on both sides."""

    def spy(train, test):
        overlap = set(map(tuple, train[["date", "team"]].to_numpy())) & set(
            map(tuple, test[["date", "team"]].to_numpy())
        )
        assert not overlap
        return np.full(len(test), 0.5)

    backtest.walk_forward(table, [2, 3], contenders={"spy": spy})


def test_walk_forward_predicts_every_row_of_every_test_season(table):
    predictions = backtest.walk_forward(table, [2, 3], contenders={"flat": flat(0.5)})
    assert len(predictions) == len(table[table["season"].isin([2, 3])])


def test_walk_forward_skips_a_season_with_nothing_before_it(table):
    predictions = backtest.walk_forward(table, [1, 2], contenders={"flat": flat(0.5)})
    assert set(predictions["season"]) == {2}


def test_walk_forward_refuses_when_no_season_can_be_tested(table):
    with pytest.raises(ValueError, match="earlier season"):
        backtest.walk_forward(table, [1], contenders={"flat": flat(0.5)})


# --- the bars ------------------------------------------------------------


def test_base_rate_quotes_the_training_win_rate(table):
    train, test = table[table["season"] < 3], table[table["season"] == 3]
    prob = backtest.base_rate(train, test)
    assert np.allclose(prob, train[model.TARGET].mean())


def test_home_or_away_quotes_one_rate_per_venue(table):
    train, test = table[table["season"] < 3], table[table["season"] == 3]
    prob = backtest.home_or_away(train, test)
    assert len(np.unique(prob)) == 2
    home_rate = train[train["is_home"] == 1][model.TARGET].mean()
    assert np.allclose(prob[test["is_home"].to_numpy() == 1], home_rate)


def test_plea_only_uses_nothing_but_elo_expected(table):
    """Scramble every other column and its predictions must not move."""
    train, test = table[table["season"] < 3], table[table["season"] == 3]
    before = backtest.plea_only(train, test)

    rng = np.random.default_rng(0)
    noisy = test.copy()
    for column in model.FEATURES:
        if column != "elo_expected":
            noisy[column] = rng.permutation(noisy[column].to_numpy())

    assert np.allclose(before, backtest.plea_only(train, noisy))


# --- scoring -------------------------------------------------------------


def test_perfect_predictions_score_perfectly():
    actual = np.array([1, 0, 1, 0])
    result = backtest.score(actual, actual.astype(float))
    assert result["accuracy"] == 1.0
    assert result["brier"] == pytest.approx(0.0)
    assert result["log_loss"] < 1e-9


def test_auc_is_undefined_rather_than_bad_for_a_constant_prediction():
    result = backtest.score(np.array([1, 0, 1, 0]), np.full(4, 0.5))
    assert np.isnan(result["auc"])


def test_calibration_error_is_zero_when_promises_are_kept():
    """Ten rows at 30%, three of which win. Exactly as advertised."""
    actual = np.array([1, 1, 1] + [0] * 7)
    assert backtest.calibration_error(actual, np.full(10, 0.3), bins=1) == pytest.approx(0.0)


def test_calibration_error_catches_overconfidence():
    actual = np.array([1] * 3 + [0] * 7)
    assert backtest.calibration_error(actual, np.full(10, 0.9), bins=1) == pytest.approx(0.6)


def test_calibration_error_survives_a_constant_prediction():
    """qcut cannot bucket one repeated value — this used to divide by zero."""
    actual = np.array([1, 0, 1, 0])
    assert backtest.calibration_error(actual, np.full(4, 0.5)) == pytest.approx(0.0)


# --- the paired test -----------------------------------------------------


def _two_contenders(table, a: float, b: float) -> pd.DataFrame:
    return backtest.walk_forward(
        table, [2, 3], contenders={"a": flat(a), "b": flat(b)}
    )


def test_a_model_compared_with_itself_gains_exactly_nothing(table):
    predictions = _two_contenders(table, 0.4, 0.4)
    result = backtest.paired_test(predictions, "a", "b")
    assert result["mean_gain"] == pytest.approx(0.0)


def test_a_genuinely_better_model_gains(table):
    """0.38 is near the true rate; 0.05 is badly wrong. The gap must be real."""
    predictions = _two_contenders(table, 0.38, 0.05)
    result = backtest.paired_test(predictions, "a", "b")
    assert result["mean_gain"] > 0
    assert result["z"] > 2


def test_the_sign_flips_when_the_arguments_swap(table):
    predictions = _two_contenders(table, 0.38, 0.05)
    forward = backtest.paired_test(predictions, "a", "b")["mean_gain"]
    backward = backtest.paired_test(predictions, "b", "a")["mean_gain"]
    assert forward == pytest.approx(-backward)


def test_the_unit_of_evidence_is_the_match_not_the_row(table):
    """Two rows per fixture. Counting them separately would halve the stderr
    and turn noise into significance."""
    predictions = _two_contenders(table, 0.38, 0.30)
    result = backtest.paired_test(predictions, "a", "b")
    assert result["matches"] == len(predictions) // 2


# --- putting the fixture back together -----------------------------------


def test_match_view_rejoins_both_perspectives(table):
    predictions = backtest.walk_forward(table, [2, 3], contenders={"flat": flat(0.4)})
    matches = backtest.match_view(predictions, "flat")
    assert len(matches) == len(predictions) // 2


def test_match_view_draw_probability_is_the_leftover(table):
    predictions = backtest.walk_forward(table, [2, 3], contenders={"flat": flat(0.4)})
    matches = backtest.match_view(predictions, "flat")
    total = matches["p_home"] + matches["p_away"] + matches["p_draw"]
    assert np.allclose(total, 1.0)


def test_match_view_records_the_outcome_that_happened(table):
    predictions = backtest.walk_forward(table, [2, 3], contenders={"flat": flat(0.4)})
    matches = backtest.match_view(predictions, "flat")
    assert set(matches["outcome"]) <= {"home win", "away win", "draw"}
    assert (matches.loc[matches["home_won"] == 1, "outcome"] == "home win").all()
    assert (matches.loc[matches["away_won"] == 1, "outcome"] == "away win").all()
    # Both clubs cannot win the same match.
    assert not ((matches["home_won"] == 1) & (matches["away_won"] == 1)).any()


def test_match_view_notices_when_a_fixture_loses_its_other_half(table):
    predictions = backtest.walk_forward(table, [2, 3], contenders={"flat": flat(0.4)})
    orphaned = predictions.drop(predictions[predictions["is_home"] == 0].index[:1])
    with pytest.raises(ValueError, match="matched an away row"):
        backtest.match_view(orphaned, "flat")


# --- end to end ----------------------------------------------------------


def test_the_forest_beats_a_coin_flip_on_data_it_has_never_seen():
    """The whole apparatus, run for real on the planted signal."""
    predictions = backtest.walk_forward(make_table(seasons=(1, 2, 3, 4)), [3, 4])
    summary = backtest.summary(predictions).set_index("model")
    assert summary.loc["forest", "auc"] > 0.65
    assert summary.loc["forest", "log_loss"] < summary.loc["base_rate", "log_loss"]
