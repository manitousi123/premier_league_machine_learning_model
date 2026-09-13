"""Call every fixture in the upcoming matchweek.

    .venv/Scripts/python.exe scripts/predict_matchweek.py

Downloads the fixture list, builds a row for each side of each match the same
way the training rows were built, and prints the verdict. Writes the same thing
to data/processed/predictions.csv.

Needs data/processed/features.parquet — run scripts/build_dataset.py first, and
re-run it after each matchweek so the ratings and form are current.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from plfootball import config, ingest, model, plea, predict


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true", help="print only, write nothing")
    args = parser.parse_args()

    path = config.PROCESSED / "features.parquet"
    if not path.exists():
        print(f"{path} missing — run scripts/build_dataset.py first", file=sys.stderr)
        return 1

    results = ingest.load_results()
    history = plea.run(results)
    table = pd.read_parquet(path)

    try:
        calls = predict.next_matchweek(results, history, table=table)
    except predict.NoFixtures as exc:
        print(f"nothing to predict: {exc}")
        return 0

    latest = results["date"].max()
    _rule(f"{len(calls)} FIXTURES - results current to {latest:%a %d %b %Y}")
    print("Confidence is the better-backed side's probability. Across thirteen test")
    print("seasons a call at 60% landed 69% of the time, and one at 70% landed 77%.")
    print()

    for row in calls.itertuples(index=False):
        when = f"{row.date:%a %d %b}"
        kickoff = "" if pd.isna(row.kickoff) else str(row.kickoff)
        print(
            f"  {when}  {kickoff:>5}   {row.home_team:>24}  v  {row.away_team:<26}"
            f"{row.verdict:<18} {row.confidence:>5.0%}"
        )

    _rule("THE SHAPE OF THE MATCHWEEK")
    counts = calls["verdict"].value_counts()
    for verdict, n in counts.items():
        print(f"  {verdict:<20} {n:>2}")
    print()
    print(f"  strongest call   {calls.loc[calls['confidence'].idxmax(), 'home_team']} v "
          f"{calls.loc[calls['confidence'].idxmax(), 'away_team']} "
          f"at {calls['confidence'].max():.0%}")
    print(f"  closest fixture  {calls.loc[calls['confidence'].idxmin(), 'home_team']} v "
          f"{calls.loc[calls['confidence'].idxmin(), 'away_team']} "
          f"at {calls['confidence'].min():.0%}")

    _rule("WHAT IT READ")
    print("Three columns, and the model is the three numbers beside them.")
    print()
    print(model.coefficients(model.fit(table)).round(4).to_string(index=False))

    if args.no_save:
        print("\n--no-save: nothing written")
        return 0

    out = config.PROCESSED / "predictions.csv"
    calls.to_csv(out, index=False)
    print(f"\nwritten to {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
