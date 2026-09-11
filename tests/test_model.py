"""What the model must never do quietly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plfootball import config, features, ingest, model, plea

from .conftest import NOISE, make_table

# --- the input contract ---------------------------------------------------


def test_candidates_identity_and_target_account_for_every_column(table):
    """No column of the table is unclassified.

    This is the test that fires when features.py grows a column: it is neither
    a known candidate nor a known identity column, so someone has to decide
    which it is rather than letting it drift silently into reach of the model.
    """
    claimed = set(model.CANDIDATES) | set(model.IDENTITY) | {model.TARGET}
    assert claimed == set(table.columns)


def test_the_model_only_reads_columns_the_table_offers():
    assert set(model.FEATURES) <= set(model.CANDIDATES)


def test_no_identity_column_is_also_a_candidate():
    assert not set(model.IDENTITY) & set(model.CANDIDATES)


def test_the_answer_is_not_a_feature():
    assert model.TARGET not in model.FEATURES


def test_matchweek_is_kept_out_because_played_already_says_it():
    assert "matchweek" in model.IDENTITY
    assert "played" in model.CANDIDATES


@pytest.mark.parametrize("banned", ["team", "opponent", "date", "season"])
def test_the_model_cannot_see_who_or_when(banned):
    """Club names and dates would let it learn reputations and calendars."""
    assert banned not in model.CANDIDATES


# --- design_matrix --------------------------------------------------------


def test_design_matrix_fixes_column_order(table):
    shuffled = table[list(reversed(table.columns))]
    assert list(model.design_matrix(shuffled).columns) == model.FEATURES


def test_design_matrix_rejects_a_missing_column(table):
    with pytest.raises(model.TableMismatch, match="elo_expected"):
        model.design_matrix(table.drop(columns=["elo_expected"]))


def test_design_matrix_rejects_nan(table):
    holed = table.copy()
    holed.loc[0, "sot_gap"] = np.nan
    with pytest.raises(model.TableMismatch, match="sot_gap"):
        model.design_matrix(holed)


def test_design_matrix_can_be_asked_for_other_columns(table):
    """Ablation work needs to build a matrix from a column set of its own."""
    picked = ["elo_expected", "form_gf"]
    assert list(model.design_matrix(table, picked).columns) == picked


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
    """elo_expected is the strongest planted signal in the fixture.

    Asserted as a comparison rather than a threshold: what matters is that the
    model tracks the real column far more closely than any of the noise
    columns, not that it clears some particular correlation.
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


def test_the_columns_are_scaled_before_fitting(table):
    """Without scaling the penalty falls hardest on whichever column has the
    largest units, which is an accident of measurement, not a decision."""
    assert "standardscaler" in model.fit(table).named_steps


# --- what it learned, and persistence -------------------------------------


def test_coefficients_report_every_feature(table):
    learned = model.coefficients(model.fit(table))
    assert list(learned["feature"].sort_values()) == sorted(model.FEATURES)
    # odds_x is the coefficient as a multiplier, so the two must agree.
    assert np.allclose(np.log(learned["odds_x"]), learned["coefficient"])


def test_creating_more_than_the_opponent_raises_the_odds(table):
    """The sign has to come out right, or the model is fitting something else."""
    learned = model.coefficients(model.fit(table)).set_index("feature")
    assert learned.loc["sot_gap", "coefficient"] > 0
    assert learned.loc["elo_expected", "coefficient"] > 0


def test_save_and_load_round_trips(table, tmp_path):
    fitted = model.fit(table)
    path = model.save(fitted, tmp_path / "nested" / "model.joblib")
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
