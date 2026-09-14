"""Run the local web app.

    .venv/Scripts/python.exe scripts/serve.py [--port 8765] [--open]

Needs data/processed/features.parquet - run scripts/build_dataset.py first.
The page's REFRESH button runs build_dataset.py and predict_matchweek.py for
you after that.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from plfootball import config
from plfootball.app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args()

    if not (config.PROCESSED / "features.parquet").exists():
        print("no dataset yet - run scripts/build_dataset.py first", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.port}"
    print(f"Premier League Match Predictor -> {url}")
    app = create_app()
    if args.open:
        webbrowser.open(url)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
