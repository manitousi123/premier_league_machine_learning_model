"""Download results, run the quality gate, build the PLEA rating history.

    .venv/Scripts/python.exe scripts/build_plea.py

Writes data/processed/plea_history.parquet and data/processed/results.parquet.
Safe to re-run: cached raw files are not re-downloaded unless --force.
"""

from __future__ import annotations

import argparse
import sys

from plfootball import config, ingest, plea, quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download every season")
    args = parser.parse_args()

    print("=" * 66)
    print("1. DOWNLOAD")
    print("=" * 66)
    for season, status in ingest.download_all(force=args.force).items():
        print(f"  {config.season_label(season)}  {status}")

    print()
    print("=" * 66)
    print("2. LOAD + STRUCTURAL CHECKS")
    print("=" * 66)
    results = ingest.load_results()
    print(f"  {len(results):,} matches, {results['date'].min():%Y-%m-%d} to "
          f"{results['date'].max():%Y-%m-%d}\n")
    quality.report(quality.check_results(results))

    print()
    print("=" * 66)
    print("3. SANITY CHECKS")
    print("=" * 66)
    quality.report(quality.check_sanity(results))

    print()
    print("=" * 66)
    print(f"4. PLEA {plea.DEFAULT.version}")
    print("=" * 66)
    for key, value in plea.DEFAULT.as_dict().items():
        print(f"  {key:<14} {value}")
    print()

    history = plea.run(results)
    print(f"  {len(history):,} rating rows\n")
    quality.report(quality.check_plea(history))

    results.to_parquet(config.PROCESSED / "results.parquet", index=False)
    history.to_parquet(config.PROCESSED / "plea_history.parquet", index=False)

    print()
    print("=" * 66)
    print(f"CURRENT RATINGS — {config.season_label(int(history['season'].max()))}")
    print("=" * 66)
    table = plea.ratings_as_of(history)
    print(table.assign(elo=table["elo"].round(1))[["team", "elo", "played"]].to_string())

    print("\nwritten to data/processed/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
