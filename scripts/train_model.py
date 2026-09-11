"""Backtest the model, then train the one that gets kept.

    .venv/Scripts/python.exe scripts/train_model.py

Runs the walk-forward across every test season, grades the model against the
three bars, and only then trains a final model on all of it and writes it to
models/. The order is deliberate: the score comes from seasons the model never
saw, and the shipped model is trained on everything precisely because it will
never be scored again.

Reads data/processed/features.parquet — run scripts/build_dataset.py first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from plfootball import backtest, config, model


def _rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true", help="backtest only, write nothing")
    args = parser.parse_args()

    path = config.PROCESSED / "features.parquet"
    if not path.exists():
        print(f"{path} missing — run scripts/build_dataset.py first", file=sys.stderr)
        return 1

    table = pd.read_parquet(path)
    seasons = sorted(table["season"].unique())
    print(f"{len(table):,} rows x {len(table.columns)} columns, "
          f"{config.season_label(seasons[0])} to {config.season_label(seasons[-1])}")
    print(f"{len(model.CANDIDATES)} of them a model could use, "
          f"{len(model.IDENTITY)} identify the row and 1 is the answer")
    print(f"the shipped model reads {len(model.FEATURES)}: {', '.join(model.FEATURES)}")

    _rule(f"1. WALK-FORWARD  (model {model.DEFAULT.version})")
    for key, value in model.DEFAULT.as_dict().items():
        print(f"  {key:<18} {value}")
    print()
    predictions = backtest.walk_forward(table, verbose=True)

    _rule("2. THE BARS")
    print("Every contender graded on identical rows. Lower log loss is better;")
    print("accuracy is shown because people ask for it, not because it says much.")
    print()
    print(backtest.summary(predictions).round(4).to_string(index=False))

    _rule("3. IS THE DIFFERENCE REAL?")
    print("Paired per match, because a fixture's two rows are not two independent")
    print("observations. |z| below about 2 means the gap is the size of the noise.")
    print()
    for name in backtest._contender_columns(predictions):
        if name == "plea_only":
            continue
        result = backtest.paired_test(predictions, name, "plea_only")
        verdict = "REAL" if abs(result["z"]) >= 2 else "noise"
        print(f"  {name:<14} vs plea_only   {result['mean_gain']:+.5f} "
              f"+/- {result['stderr']:.5f}   z = {result['z']:+5.2f}   {verdict}")

    _rule("4. SEASON BY SEASON  (log loss)")
    print(backtest.by_season(predictions).round(4).to_string(index=False))

    _rule("5. CALIBRATION - when it says 70%, does it happen 70% of the time?")
    print(backtest.calibration_table(predictions).round(4).to_string(index=False))

    _rule("6. THE WHOLE MODEL, IN THREE NUMBERS")
    print("Columns are scaled before fitting, so these are directly comparable.")
    print("odds_x is the same number as a multiplier: how the odds of winning move")
    print("for a one standard deviation rise in that column.")
    print()
    # The last *complete* season, not simply the newest: a season three
    # matchweeks old has too few rows for scrambling a column to mean anything.
    last = max(s for s in config.TEST_SEASONS if s in seasons)
    trained = model.fit(table[table["season"] < last])
    print(model.coefficients(trained).round(4).to_string(index=False))

    print()
    print("The check on those: how much log loss worsens when each column is")
    print(f"scrambled on {config.season_label(last)}. At or below zero means unused.")
    print()
    ranking = backtest.permutation_ranking(trained, table[table["season"] == last])
    print(ranking.round(5).to_string(index=False))

    _rule("7. THE FOUR-WAY CALL")
    matches = backtest.match_view(predictions)
    incoherent = int((matches["p_draw"] < 0).sum())
    print(f"  {len(matches):,} matches, both perspectives rejoined")
    print(f"  {incoherent} where p_home + p_away exceeded 1 (the model contradicting itself)")
    print()
    print(matches["verdict"].value_counts().to_string())
    print()
    hit = (
        ((matches["verdict"] == "HOME will win") & (matches["outcome"] == "home win"))
        | ((matches["verdict"] == "AWAY will win") & (matches["outcome"] == "away win"))
        | ((matches["verdict"] == "HOME will not win") & (matches["outcome"] != "home win"))
    )
    print(f"  verdict correct on {hit.mean():.1%} of matches")

    if args.no_save:
        print("\n--no-save: nothing written")
        return 0

    _rule("8. THE MODEL THAT GETS KEPT")
    final = model.fit(table)
    out = model.save(final, config.MODELS / f"model_{model.DEFAULT.version}.joblib")
    predictions.to_parquet(config.PROCESSED / "backtest_predictions.parquet", index=False)
    print(f"  trained on all {len(table):,} rows -> {out.relative_to(config.ROOT)}")
    print("  predictions      -> data/processed/backtest_predictions.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
