"""The local web app: one page, three tabs, one button.

Flask serves the page and a few JSON endpoints. Every number comes from
`dashboard`; this module loads the files, remembers what it loaded, and runs
the refresh. Refresh runs the two scripts a person would otherwise run by
hand, as subprocesses, so the page and the command line can never drift apart.

    .venv/Scripts/python.exe scripts/serve.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory

from . import config, dashboard, fixtures, ingest, model, plea, track
from .teams import UnknownClubError

UI = Path(__file__).parent / "ui"
SCRIPTS = config.ROOT / "scripts"

# The weekly run, in order. Settling happens inside the second script.
REFRESH_STEPS = ["build_dataset.py", "predict_matchweek.py"]

_CALL_COLUMNS = [
    "date", "kickoff", "home_team", "away_team", "p_home", "p_away", "p_draw",
    "home_call", "away_call", "verdict", "confidence",
]


@dataclass
class Snapshot:
    """Everything the page reads, loaded together so the tabs agree."""

    results: pd.DataFrame
    history: pd.DataFrame
    fitted: object
    coefficients: pd.DataFrame
    calls: pd.DataFrame
    log: pd.DataFrame
    backtest_scored: pd.DataFrame | None
    band_rates: dict
    season_fixtures: pd.DataFrame | None
    season_error: str | None = None

    @property
    def latest(self) -> pd.Timestamp:
        return pd.Timestamp(self.results["date"].max())


def load_snapshot() -> Snapshot:
    """Read what the scripts wrote. Raises if the dataset has never been built."""
    table_path = config.PROCESSED / "features.parquet"
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} missing - run scripts/build_dataset.py first")

    results = ingest.load_results()
    history = plea.run(results)
    fitted = model.fit(pd.read_parquet(table_path))

    calls_path = config.PROCESSED / "predictions.csv"
    calls = (
        pd.read_csv(calls_path, parse_dates=["date"])
        if calls_path.exists()
        else pd.DataFrame(columns=_CALL_COLUMNS)
    )

    backtest_path = config.PROCESSED / "backtest_predictions.parquet"
    scored = (
        dashboard.backtest_calls(pd.read_parquet(backtest_path))
        if backtest_path.exists()
        else None
    )

    season, season_error = None, None
    try:
        season = fixtures.load()
    except (fixtures.NoSeasonFixtures, UnknownClubError, ValueError) as exc:
        # Loud, but not fatal: the look-ahead is a convenience built on a
        # third-party file. The page still works without it.
        season_error = str(exc)
        print(f"season fixture list unavailable: {exc}", file=sys.stderr)

    return Snapshot(
        results=results,
        history=history,
        fitted=fitted,
        coefficients=model.coefficients(fitted),
        calls=calls,
        log=track.load(),
        backtest_scored=scored,
        band_rates=dashboard.band_rates(scored),
        season_fixtures=season,
        season_error=season_error,
    )


class Store:
    """Holds the current snapshot and swaps in a new one after a refresh."""

    def __init__(self, loader=load_snapshot):
        self._loader = loader
        self._snapshot: Snapshot | None = None
        self._lock = threading.Lock()

    def get(self) -> Snapshot:
        if self._snapshot is None:
            self.reload()
        return self._snapshot  # type: ignore[return-value]

    def reload(self) -> Snapshot:
        fresh = self._loader()
        with self._lock:
            self._snapshot = fresh
        return fresh


def run_scripts() -> str:
    """The weekly run, exactly as it is done by hand."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    output = []
    for name in REFRESH_STEPS:
        done = subprocess.run(
            [sys.executable, str(SCRIPTS / name)],
            cwd=config.ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        output.append(done.stdout)
        if done.returncode != 0:
            tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-8:])
            raise RuntimeError(f"{name} exited with {done.returncode}:\n{tail}")

    try:
        fixtures.download()
    except (requests.RequestException, UnknownClubError, ValueError) as exc:
        output.append(f"season fixture list not refreshed: {exc}")
    return "\n".join(output)


class Refresh:
    """The weekly run as a background job the page can start and watch."""

    def __init__(self, store: Store, runner=run_scripts):
        self.store = store
        self.runner = runner
        self.running = False
        self.notice: dict | None = None
        self.output = ""
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.notice = None
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            before = self.store.get().latest
            self.output = self.runner()
            after = self.store.reload()
            new = int((after.results["date"] > before).sum())
            if new:
                plural = "s" if new != 1 else ""
                self.notice = {
                    "kind": "ok",
                    "text": (
                        f"Ingested {new} new result{plural}, now current to "
                        f"{dashboard.day_label(after.latest)}. Ratings and this week's "
                        "card rebuilt."
                    ),
                }
            else:
                self.notice = {
                    "kind": "warn",
                    "text": (
                        f"Already current to {dashboard.day_label(before)} - the source "
                        "has nothing newer yet."
                    ),
                }
        except Exception as exc:  # noqa: BLE001 - anything here must reach the page
            self.notice = {"kind": "error", "text": f"Refresh failed: {exc}"}
        finally:
            self.running = False

    def status(self) -> dict:
        return {"running": self.running, "notice": self.notice}


def build_state(snap: Snapshot, refresh_status: dict) -> dict:
    current = dashboard.current_round(snap.calls, snap.season_fixtures)
    label = f"Matchweek {current}" if current else "Next fixtures"

    rounds: list = []
    if snap.season_fixtures is not None:
        rounds = dashboard.rounds_view(
            fixtures.rounds(snap.season_fixtures, snap.results), current
        )

    return {
        "freshness": dashboard.freshness(snap.results, snap.season_fixtures),
        "week": dashboard.this_week(snap.calls, snap.band_rates, label=label, log=snap.log),
        "record": dashboard.record(snap.log, snap.backtest_scored, snap.season_fixtures),
        "model": dashboard.model_view(
            snap.coefficients, snap.history, snap.latest, snap.calls
        ),
        "rounds": rounds,
        "rounds_error": snap.season_error,
        "current_round": current,
        "versions": {"model": model.DEFAULT.version, "plea": plea.DEFAULT.version},
        "refresh": refresh_status,
    }


def create_app(store: Store | None = None, refresh: Refresh | None = None) -> Flask:
    store = store or Store()
    refresh = refresh or Refresh(store)
    app = Flask(__name__, static_folder=str(UI), static_url_path="")

    @app.get("/")
    def index():
        return send_from_directory(UI, "index.html")

    @app.get("/api/state")
    def state():
        return jsonify(build_state(store.get(), refresh.status()))

    @app.get("/api/preview/<int:number>")
    def preview(number: int):
        snap = store.get()
        if snap.season_fixtures is None:
            return jsonify({"error": snap.season_error or "no season fixture list"}), 404

        block = fixtures.for_round(snap.season_fixtures, number)
        if block.empty:
            return jsonify({"error": f"no matchweek {number}"}), 404
        remaining = fixtures.unplayed(block, snap.results)
        if remaining.empty:
            return jsonify({"error": f"matchweek {number} has been played"}), 404

        calls = dashboard.preview_calls(remaining, snap.results, snap.history, snap.fitted)
        return jsonify(dashboard.this_week(
            calls, snap.band_rates, label=f"Matchweek {number}", log=snap.log, preview=True
        ))

    @app.post("/api/refresh")
    def start_refresh():
        started = refresh.start()
        return jsonify({**refresh.status(), "started": started}), (202 if started else 409)

    @app.get("/api/refresh")
    def refresh_status():
        return jsonify(refresh.status())

    return app
