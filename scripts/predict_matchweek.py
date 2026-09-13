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

CELL = 38  # characters per quadrant - wide enough for the longest fixture line


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _fixture_lines(block: pd.DataFrame, empty_note: str) -> list[str]:
    """One line per fixture in a quadrant: the pair, then both probabilities."""
    if block.empty:
        return [f"  {empty_note}"]
    return [
        f"  {row.home_team[:12]:<12} v {row.away_team[:12]:<12} "
        f"{row.p_home:.0%}/{row.p_away:.0%}"
        for row in block.itertuples(index=False)
    ]


def _draw_grid(calls: pd.DataFrame) -> None:
    """The four quadrants side by side, with the fixtures that landed in each."""
    home = calls["p_home"] >= 0.5
    away = calls["p_away"] >= 0.5

    quadrants = [
        [("HOME team WILL win", calls[home & ~away], "none"),
         ("NO CALL - both backed", calls[home & away], "none, as it should be")],
        [("HOME team WILL NOT win", calls[~home & ~away], "none"),
         ("AWAY team WILL win", calls[~home & away], "none")],
    ]
    row_labels = ["HOME WILL WIN", "HOME WILL NOT WIN"]
    bar = "+" + "-" * CELL + "+" + "-" * CELL + "+"
    pad = " " * 20

    header = f"{pad} {'AWAY WILL NOT WIN':<{CELL + 1}}{'AWAY WILL WIN'}"
    print(header.rstrip())
    print(pad + bar)
    for label, pair in zip(row_labels, quadrants):
        blocks = []
        for title, block, note in pair:
            lines = [f"{title}  ({len(block)})"] + _fixture_lines(block, note)
            blocks.append(lines)

        height = max(len(b) for b in blocks)
        for i in range(height):
            left = blocks[0][i] if i < len(blocks[0]) else ""
            right = blocks[1][i] if i < len(blocks[1]) else ""
            shown = label if i == 0 else ""
            print(f"{shown:<20}|{left:<{CELL}}|{right:<{CELL}}|")
        print(pad + bar)


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
            f"  {when}  {kickoff:>5}   {row.home_team:>24} {row.p_home:>4.0%}"
            f"  v  {row.p_away:<4.0%} {row.away_team:<26}{row.verdict}"
        )

    _rule("THE 2x2 GRID")
    print("Every fixture is scored twice, once from each club's side. Each answer is")
    print("a yes or a no, so the pair lands in one of four places. Percentages are")
    print("home/away chance of winning.")
    print()
    _draw_grid(calls)
    print()
    print(predict.grid(calls).to_string())
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
    # Four decimals is already far more precision than a football forecast has.
    rounded = calls.copy()
    numbers = ["p_home", "p_away", "p_draw", "confidence"]
    rounded[numbers] = rounded[numbers].round(4)
    rounded.to_csv(out, index=False)
    print(f"\nwritten to {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
