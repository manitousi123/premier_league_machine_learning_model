"""What the model must never do quietly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plfootball import config, features, ingest, model, plea

from .conftest import NOISE, make_table

# --- the input contract ---------------------------------------------------


def test_features_identity_and_target_account_for_every_column(table):
    """No column of the table is unclassified.

    This is the test that fires when features.py grows a column: it is neither
    a known feature nor a known identity column, so someone has to decide which
    it is rather than letting it drift silently into the model.
    """
    claimed = set(model.FEATURES) | set(model.IDENTITY) | {model.TARGET}
    assert claimed == set(table.columns)


def test_no_identity_column_is_also_a_feature():
    assert not set(model.IDENTITY) & set(model.FEATURES)


def test_the_answer_is_not_a_feature():
    assert model.TARGET not in model.FEATURES


def test_matchweek_is_kept_out_because_played_already_says_it():
    assert "matchweek" in model.IDENTITY
    assert "played" in model.FEATURES


@pytest.mark.parametrize("banned", ["team", "opponent", "date", "season"])
def test_the_model_cannot_see_who_or_when(banned):
    """Club names and dates would let it learn reputations and calendars."""
    assert banned not in model.FEATURES


# --- design_matrix --------------------------------------------------------


def test_design_matrix_fixes_column_order(table):
    shuffled = table[list(reversed(table.columns))]
    assert list(model.design_matrix(shuffled).columns) == model.FEATURES


def test_design_matrix_rejects_a_missing_column(table):
    with pytest.raises(model.TableMismatch, match="elo_expected"):
        model.design_matrix(table.drop(columns=["elo_expected"]))


def test_design_matrix_rejects_nan(table):
    holed = table.copy()
    holed.loc[0, "form_gf"] = np.nan
    with pytest.raises(model.TableMismatch, match="form_gf"):
        model.design_matrix(holed)


def test_fit_rejects_a_table_with_no_answer(table):
    with pytest.raises(model.TableMismatch, match="target"):
        model.fit(table.drop(columns=[model.TARGET]))


# --- fitting and predicting ----------------------------------------------


def test_predictions_are_probabilities(table):
    prob = model.predict(model.fit(table), table)
    assert len(prob) == len(table)
    assert prob.between(0.0, 1.0).all()


def test_predictions_keep_the_table_index(table):
    subset = table.iloc[50:150]
    prob = model.predict(model.fit(table), subset)
    assert prob.index.equals(subset.index)


def test_the_model_finds_the_planted_signal(table):
    """elo_expected is the only column that means anything in the fixture.

    Asserted as a comparison rather than a threshold: what matters is that the
    forest tracks the one real column far more closely than any of the 28
    noise columns, not that it clears some particular correlation.
    """
    prob = model.predict(model.fit(table), table)
    tracking = {c: abs(np.corrcoef(prob, table[c])[0, 1]) for c in [*NOISE, "elo_expected"]}
    signal = tracking.pop("elo_expected")

    assert signal > 0.7
    assert signal > 3 * max(tracking.values())


def test_the_model_is_deterministic(table):
    a = model.predict(model.fit(table), table)
    b = model.predict(model.fit(table), table)
    pd.testing.assert_series_equal(a, b)


def test_min_samples_leaf_is_not_left_at_one():
    """A leaf of 1 memorises single matches and quotes them back as certainty."""
    assert model.DEFAULT.min_samples_leaf >= 10


# --- importance and persistence ------------------------------------------


def test_importance_covers_every_feature_and_sums_to_one(table):
    ranking = model.importance(model.fit(table))
    assert list(ranking["feature"].sort_values()) == sorted(model.FEATURES)
    assert ranking["importance"].sum() == pytest.approx(1.0)


def test_save_and_load_round_trips(table, tmp_path):
    fitted = model.fit(table)
    path = model.save(fitted, tmp_path / "nested" / "forest.joblib")
    reloaded = model.load(path)

    pd.testing.assert_series_equal(
        model.predict(fitted, table), model.predict(reloaded, table)
    )
    assert reloaded.plea_params_ == model.DEFAULT


# --- against the real table ----------------------------------------------


@pytest.mark.skipif(
    not (config.RAW / f"E0_{ingest.season_code(config.PLEA_START)}.csv").exists(),
    reason="no downloaded seasons — run scripts/build_dataset.py",
)
def test_the_real_table_satisfies_the_contract():
    """The synthetic fixture is shaped like the real table. Prove it still is."""
    results = ingest.load_results()
    real = features.build_features(results, plea.run(results))
    synthetic = make_table()
    assert set(real.columns) == set(synthetic.columns)
    model.design_matrix(real)  # raises if a column is missing or holed
