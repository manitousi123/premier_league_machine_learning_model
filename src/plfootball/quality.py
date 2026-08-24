"""Data-quality gate and sanity checks.

Two different jobs, both of which fail loudly.

**Structural checks** catch a broken download: a missing club, a duplicated
fixture, a season with 379 matches.

**Sanity checks** catch something worse — data that is structurally perfect and
quietly wrong. The model cannot tell you it is cheating: build the rolling-form
window the wrong way round and it will score 85% and look like a triumph. These
checks are the only place we compare the table against things we already know to
be true, so they are the only defence.

Both run on every ingest, forever.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config


class QualityError(AssertionError):
    """The data disagrees with something we know. Investigate, don't suppress."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name:<42} {self.detail}"


def _near(value: float, target: float, tol: float = config.SANITY_TOLERANCE) -> bool:
    return abs(value - target) <= tol


def check_results(results: pd.DataFrame, *, complete_seasons_only: bool = True) -> list[Check]:
    """Structural checks on the match table."""
    checks: list[Check] = []

    counts = results.groupby("season").size()
    complete = counts[counts == 380].index
    bad = counts[counts != 380]
    checks.append(
        Check(
            "380 matches per season",
            bad.empty,
            "all seasons complete" if bad.empty else f"wrong count: {bad.to_dict()}",
        )
    )

    scope = results[results["season"].isin(complete)] if complete_seasons_only else results

    clubs = scope.groupby("season")["home_team"].nunique()
    off = clubs[clubs != 20]
    checks.append(
        Check("20 clubs per season", off.empty, "ok" if off.empty else f"wrong: {off.to_dict()}")
    )

    played = (
        pd.concat([scope[["season", "home_team"]].rename(columns={"home_team": "team"}),
                   scope[["season", "away_team"]].rename(columns={"away_team": "team"})])
        .groupby(["season", "team"]).size()
    )
    off = played[played != 38]
    checks.append(
        Check(
            "38 matches per club",
            off.empty,
            "ok" if off.empty else f"{len(off)} club-seasons wrong, e.g. {off.head(3).to_dict()}",
        )
    )

    dupes = results.duplicated(subset=["date", "home_team", "away_team"]).sum()
    checks.append(Check("no duplicate fixtures", dupes == 0, f"{dupes} duplicates"))

    same = (results["home_team"] == results["away_team"]).sum()
    checks.append(Check("no club plays itself", same == 0, f"{same} self-matches"))

    return checks


def check_sanity(results: pd.DataFrame) -> list[Check]:
    """Does the data agree with things we already know about football?"""
    checks: list[Check] = []
    mix = results["result"].value_counts(normalize=True)

    for code, label, target in (
        ("H", "home win rate", config.HOME_WIN_RATE),
        ("A", "away win rate", config.AWAY_WIN_RATE),
        ("D", "draw rate", config.DRAW_RATE),
    ):
        rate = float(mix.get(code, 0.0))
        checks.append(
            Check(label, _near(rate, target), f"{rate:.1%} (expect ~{target:.0%})")
        )

    return checks


def check_plea(history: pd.DataFrame) -> list[Check]:
    """Is PLEA producing ratings that mean anything?"""
    checks: list[Check] = []

    favoured = history[history["elo_gap"] > 0]["actual"].mean()
    checks.append(
        Check(
            "higher-Elo side scores more",
            0.58 <= favoured <= 0.70,
            f"{favoured:.3f} points share (expect 0.60-0.66)",
        )
    )

    error = (history["elo_expected"] - history["actual"]).mean()
    checks.append(
        Check("PLEA is unbiased overall", abs(error) < 0.02, f"mean error {error:+.4f}")
    )

    # The leakage tripwire. Nothing legitimate predicts a football match this
    # well on its own — if something does, it has seen the result.
    best = float(history[["elo_expected"]].corrwith(history["actual"]).abs().max())
    checks.append(
        Check(
            "no single column is suspiciously predictive",
            best < 0.70,
            f"strongest correlation {best:.3f} (leakage if > 0.70)",
        )
    )

    return checks


def report(checks: list[Check], *, raise_on_fail: bool = True) -> list[Check]:
    """Print every check, then raise if any failed."""
    for check in checks:
        print(check)

    failed = [c for c in checks if not c.passed]
    if failed and raise_on_fail:
        raise QualityError(
            f"{len(failed)} check(s) failed:\n"
            + "\n".join(f"  - {c.name}: {c.detail}" for c in failed)
        )
    return failed
