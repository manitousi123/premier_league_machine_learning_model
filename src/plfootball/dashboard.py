"""What the page shows, worked out as plain data.

Every function here returns dicts and lists that `json.dumps` accepts, and
none of them reads a file or touches the network. The app loads the frames and
hands them in; this module decides what they mean. That split is what lets the
page be tested without a browser or a server, and it keeps every number on
screen traceable to a function with a test behind it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, model, plea, predict, track

# The line both clubs are judged against. It is not adjustable: the verdicts
# in the log were made at this line, the backtest was scored at it, and a page
# that let it slide would be comparing calls made under different rules.
THRESHOLD = 0.5

VERDICT_SHORT = {
    "HOME will win": "HOME",
    "AWAY will win": "AWAY",
    "HOME will not win": "NOT HOME",
    "NO CALL - both sides backed": "BOTH",
}

FEATURE_NOTES = {
    "elo_expected": (
        "Rating expectation",
        "PLEA's expected score for this club, from the two ratings and home advantage.",
    ),
    "sot_gap": (
        "Shots-on-target gap",
        f"Own shots on target per match minus the opponent's, last {config.FORM_WINDOW}.",
    ),
    "shots_gap": (
        "Total shots gap",
        f"Own shots per match minus the opponent's, last {config.FORM_WINDOW}.",
    ),
}


# --- small helpers -------------------------------------------------------


def _now(now) -> pd.Timestamp:
    return (pd.Timestamp.now() if now is None else pd.Timestamp(now)).normalize()


def day_label(ts) -> str:
    """'Sun 6 Sep' - no leading zero, no year."""
    ts = pd.Timestamp(ts)
    return f"{ts:%a} {ts.day} {ts:%b}"


def _short(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {ts:%b}"


def date_span(dates) -> str:
    """'Sat 12 – Mon 14 Sep', or a single day when that is all there is."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize().dropna()
    if dates.empty:
        return ""
    first, last = dates.min(), dates.max()
    if first == last:
        return day_label(first)
    if first.month == last.month:
        return f"{first:%a} {first.day} – {day_label(last)}"
    return f"{day_label(first)} – {day_label(last)}"


def _pct(x) -> int:
    return round(float(x) * 100)


def _pct_or_none(x):
    if x is None or pd.isna(x):
        return None
    return _pct(x)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def band_of(confidence) -> pd.Series:
    """The same bands the log is scored in, as plain strings."""
    return pd.cut(
        pd.to_numeric(confidence),
        track.BANDS,
        labels=track.BAND_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)


def stated(frame: pd.DataFrame) -> pd.Series:
    """The probability each verdict actually claims.

    A backed side claims its own chance. "HOME will not win" claims everything
    except a home win, which is 1 - p_home. A contradiction claims nothing.
    """
    p_home = pd.to_numeric(frame["p_home"]).to_numpy(dtype=float)
    p_away = pd.to_numeric(frame["p_away"]).to_numpy(dtype=float)
    verdict = frame["verdict"].to_numpy()
    claimed = np.select(
        [
            verdict == "HOME will win",
            verdict == "AWAY will win",
            verdict == "HOME will not win",
        ],
        [p_home, p_away, 1.0 - p_home],
        default=np.nan,
    )
    return pd.Series(claimed, index=frame.index)


# --- freshness -----------------------------------------------------------


def freshness(results: pd.DataFrame, season_fixtures: pd.DataFrame | None = None, now=None) -> dict:
    """How current the results are, and whether that is a problem.

    Stale does not mean old. A fortnight with no football is a fortnight in
    which nothing could have arrived. Stale means a fixture has kicked off
    since the last result came in - so the fixture list decides, when there is
    one, and the clock is only the fallback.
    """
    now = _now(now)
    latest = pd.Timestamp(results["date"].max()).normalize()
    age = int((now - latest).days)

    if season_fixtures is not None and not season_fixtures.empty:
        dates = pd.to_datetime(season_fixtures["date"]).dt.normalize()
        stale = bool(((dates > latest) & (dates < now)).any())
    else:
        stale = age > 4

    if age <= 0:
        age_text = "updated today"
    elif age == 1:
        age_text = "1 day old"
    else:
        age_text = f"{age} days old"

    return {
        "date": latest.strftime("%Y-%m-%d"),
        "label": day_label(latest),
        "age_days": age,
        "age": age_text,
        "stale": stale,
    }


# --- this week -----------------------------------------------------------


def this_week(
    calls: pd.DataFrame,
    band_rates: dict | None = None,
    *,
    label: str,
    log: pd.DataFrame | None = None,
    preview: bool = False,
    now=None,
) -> dict:
    """One matchweek's calls, sorted into the three kinds the page shows.

    `band_rates` is what the backtest said about calls of each strength, so a
    63% call can be shown next to how often 60-70% calls have actually landed.
    `log` marks which of these fixtures are on the record.
    """
    now = _now(now)
    band_rates = band_rates or {}

    logged: set = set()
    if log is not None and not log.empty:
        logged = set(zip(
            pd.to_datetime(log["date"]).dt.normalize(), log["home_team"], log["away_team"]
        ))

    frame = calls.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=[
            "date", "kickoff", "home_team", "away_team", "p_home", "p_away", "p_draw",
            "verdict", "confidence",
        ])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["band"] = band_of(frame["confidence"]) if not frame.empty else ""

    rows = []
    for r in frame.itertuples(index=False):
        rows.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "day": f"{r.date:%a}",
            "time": "" if pd.isna(r.kickoff) else str(r.kickoff),
            "home": r.home_team,
            "away": r.away_team,
            "p_home": _pct(r.p_home),
            "p_away": _pct(r.p_away),
            "p_draw": _pct(r.p_draw),
            "home_backed": bool(float(r.p_home) >= THRESHOLD),
            "away_backed": bool(float(r.p_away) >= THRESHOLD),
            "verdict": r.verdict,
            "short": VERDICT_SHORT.get(r.verdict, r.verdict),
            "confidence": _pct(r.confidence),
            "band": r.band,
            "band_rate": _pct_or_none(band_rates.get(r.band)),
            "logged": (r.date, r.home_team, r.away_team) in logged,
            "kicked_off": bool(r.date < now),
        })

    committed = [x for x in rows if x["verdict"] in ("HOME will win", "AWAY will win")]
    weak = [x for x in rows if x["verdict"] == "HOME will not win"]
    both = [x for x in rows if x["verdict"] == "NO CALL - both sides backed"]

    return {
        "label": label,
        "preview": preview,
        "dates": date_span(frame["date"]),
        "n": len(rows),
        "n_calls": len(committed),
        "n_home": sum(x["verdict"] == "HOME will win" for x in committed),
        "n_away": sum(x["verdict"] == "AWAY will win" for x in committed),
        "n_weak": len(weak),
        "n_contradictions": len(both),
        "n_logged": sum(x["logged"] for x in rows),
        "threshold": _pct(THRESHOLD),
        "committed": committed,
        "weak": weak,
        "contradictions": both,
    }


def preview_calls(
    round_fixtures: pd.DataFrame,
    results: pd.DataFrame,
    history: pd.DataFrame,
    fitted,
) -> pd.DataFrame:
    """Calls for a round that has not come up yet, on today's ratings and form.

    Built by the same code as the logged calls, so the only thing that makes
    it a preview is that nobody writes it down.
    """
    rows = predict.build_rows(round_fixtures, results, history)
    return predict.verdicts(rows, model.predict(fitted, rows))


def current_round(calls: pd.DataFrame, season_fixtures: pd.DataFrame | None) -> int | None:
    """Which round the committed calls belong to, looked up by club pairing."""
    if season_fixtures is None or season_fixtures.empty or calls.empty:
        return None
    if config.season_of(pd.to_datetime(calls["date"]).min()) != config.season_of(
        pd.to_datetime(season_fixtures["date"]).min()
    ):
        return None

    lookup = {
        (h, a): int(n)
        for h, a, n in zip(
            season_fixtures["home_team"], season_fixtures["away_team"], season_fixtures["round"]
        )
    }
    found = [lookup.get((h, a)) for h, a in zip(calls["home_team"], calls["away_team"])]
    found = [n for n in found if n is not None]
    if not found:
        return None
    return int(pd.Series(found).mode().iloc[0])


def rounds_view(table: pd.DataFrame, current: int | None = None) -> list[dict]:
    """The rounds a person can page to: anything still open, plus the current one."""
    out = []
    for r in table.itertuples(index=False):
        number = int(r.round)
        if r.remaining == 0 and number != current:
            continue
        out.append({
            "round": number,
            "label": f"Matchweek {number}",
            "dates": date_span(pd.Series([r.first, r.last])),
            "fixtures": int(r.fixtures),
            "played": int(r.played),
            "remaining": int(r.remaining),
            "current": number == current,
        })
    return out


# --- the track record ----------------------------------------------------


def backtest_calls(backtest: pd.DataFrame, column: str = "p_model") -> pd.DataFrame:
    """The walk-forward's rows put through the same verdict and scoring as the
    live log, so the two can be read against each other band for band."""
    rows = backtest.assign(kickoff=pd.NA)
    calls = predict.verdicts(rows, backtest[column])

    home = backtest[backtest["is_home"] == 1][["date", "team", "opponent", "target"]].rename(
        columns={"team": "home_team", "opponent": "away_team", "target": "home_won"}
    )
    away = backtest[backtest["is_home"] == 0][["date", "team", "opponent", "target"]].rename(
        columns={"team": "away_team", "opponent": "home_team", "target": "away_won"}
    )
    merged = calls.merge(home, on=track.KEY).merge(away, on=track.KEY)
    merged["outcome"] = np.select(
        [merged["home_won"] == 1, merged["away_won"] == 1],
        ["home win", "away win"],
        default="draw",
    )
    merged["correct"] = track.verdict_held(merged["verdict"], merged["outcome"])
    return merged.drop(columns=["home_won", "away_won"])


def band_rates(scored: pd.DataFrame | None) -> dict:
    """Hit rate per confidence band, keyed by the band's label."""
    if scored is None or scored.empty:
        return {}
    table = track.track_record(log=scored)
    return {
        str(r.band): (None if pd.isna(r.hit_rate) else float(r.hit_rate))
        for r in table.itertuples(index=False)
    }


def _bands(scored: pd.DataFrame) -> list[dict]:
    """Per band: how many, how many right, and what was claimed on average."""
    table = track.track_record(log=scored)
    if table.empty:
        return []
    frame = scored.assign(stated=stated(scored), band=band_of(scored["confidence"]))
    claimed = frame[frame["correct"].notna()].groupby("band")["stated"].mean()
    return [
        {
            "band": str(r.band),
            "predictions": int(r.predictions),
            "correct": int(r.correct),
            "hit_rate": _pct_or_none(r.hit_rate),
            "stated": _pct_or_none(claimed.get(str(r.band))),
        }
        for r in table.itertuples(index=False)
    ]


def _log_row(r) -> dict:
    settled = not pd.isna(r.outcome)
    if not settled:
        scored = "awaiting"
    elif pd.isna(r.correct):
        scored = "unscored"
    else:
        scored = "correct" if _as_bool(r.correct) else "wrong"

    return {
        "date": _short(r.date),
        "iso": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
        "fixture": f"{r.home_team} v {r.away_team}",
        "home": r.home_team,
        "away": r.away_team,
        "p_home": _pct(r.p_home),
        "p_away": _pct(r.p_away),
        "home_backed": bool(float(r.p_home) >= THRESHOLD),
        "away_backed": bool(float(r.p_away) >= THRESHOLD),
        "verdict": r.verdict,
        "short": VERDICT_SHORT.get(r.verdict, r.verdict),
        "result": f"{int(r.home_goals)}–{int(r.away_goals)}" if settled else "",
        "scored": scored,
    }


def _round_lookup(season_fixtures: pd.DataFrame | None) -> dict:
    if season_fixtures is None or season_fixtures.empty:
        return {}
    dates = pd.to_datetime(season_fixtures["date"])
    return {
        (config.season_of(d), h, a): int(n)
        for d, h, a, n in zip(
            dates, season_fixtures["home_team"], season_fixtures["away_team"],
            season_fixtures["round"],
        )
    }


CONFIDENT = ("HOME will win", "AWAY will win")


def _tally(rows: list[dict]) -> dict:
    """How many of these rows have been scored, and how many held."""
    settled = [r for r in rows if r["scored"] in ("correct", "wrong")]
    return {"called": len(settled), "correct": sum(r["scored"] == "correct" for r in settled)}


def _split(scored: pd.DataFrame) -> dict:
    """The two kinds of claim, scored separately.

    A backed side is the confident call; "HOME will not win" is the weak one.
    They are different bets - the weak claim is a double chance and pays
    accordingly - so one accuracy figure mixing them says little about either.
    """
    held = scored[scored["correct"].notna()]

    def part(block: pd.DataFrame) -> dict:
        n = len(block)
        right = int(block["correct"].map(_as_bool).sum()) if n else 0
        return {"called": n, "correct": right, "hit_rate": _pct(right / n) if n else None}

    return {
        "confident": part(held[held["verdict"].isin(CONFIDENT)]),
        "weak": part(held[held["verdict"] == "HOME will not win"]),
    }


def weeks(log: pd.DataFrame, season_fixtures: pd.DataFrame | None = None) -> list[dict]:
    """The log broken into matchweeks, grouped by season, newest first.

    The log does not store a round number - the fixture list it is fed from
    does not carry one - so a week is a run of fixtures with no gap of more
    than two days between them. A Friday-to-Monday round holds together; a
    midweek round three days later starts a new one. Where the season's
    fixture list is available the round number is looked up for the label.
    """
    if log.empty:
        return []

    frame = log.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["date", "kickoff", "home_team"], kind="stable").reset_index(
        drop=True
    )
    frame["week"] = (frame["date"].diff().dt.days.fillna(0) > 2).cumsum()
    frame["season"] = frame["date"].map(config.season_of)
    rounds = _round_lookup(season_fixtures)

    seasons = []
    for season, block in frame.groupby("season", sort=True):
        out = []
        for _, wk in block.groupby("week", sort=True):
            rows = [_log_row(r) for r in wk.itertuples(index=False)]
            numbers = {
                rounds.get((int(season), r.home_team, r.away_team))
                for r in wk.itertuples(index=False)
            } - {None}
            label = (
                f"Matchweek {numbers.pop()}" if len(numbers) == 1
                else f"Week of {day_label(wk['date'].min())}"
            )
            scored = [row["scored"] for row in rows]
            out.append({
                "label": label,
                "dates": date_span(wk["date"]),
                "correct": scored.count("correct"),
                "wrong": scored.count("wrong"),
                "awaiting": scored.count("awaiting"),
                "unscored": scored.count("unscored"),
                "confident": _tally([r for r in rows if r["verdict"] in CONFIDENT]),
                "weak": _tally([r for r in rows if r["verdict"] == "HOME will not win"]),
                "rows": rows,
            })
        seasons.append({"label": config.season_label(int(season)), "weeks": out[::-1]})
    return seasons[::-1]


def record(
    log: pd.DataFrame,
    backtest_scored: pd.DataFrame | None = None,
    season_fixtures: pd.DataFrame | None = None,
) -> dict:
    """The headline numbers, the bands, and every prediction by week.

    Alongside each live figure sits the backtest's, computed the same way over
    the same bands. The live log is small for a long time, and a number with
    nothing to read it against is just a number.
    """
    facts = track.summary(log=log)
    done = log[log["outcome"].notna()].copy()

    out = {
        "logged": int(facts["logged"]),
        "settled": int(facts["settled"]),
        "awaiting": int(facts["awaiting_result"]),
        "called": int(facts.get("called", 0)),
        "verdict_accuracy": _pct_or_none(facts.get("verdict_accuracy")),
        "log_loss": facts.get("log_loss"),
        "brier": facts.get("brier"),
        "neither_backed": int((done["verdict"] == "HOME will not win").sum()),
        "contradictions": int((log["verdict"] == "NO CALL - both sides backed").sum()),
        "calibration_error": None,
        **_split(done),
        "bands": [],
        "seasons": weeks(log, season_fixtures),
        "backtest": None,
    }

    if not done.empty:
        claim = stated(done)
        scoreable = done[done["correct"].notna() & claim.notna()]
        if not scoreable.empty:
            hit = scoreable["correct"].map(_as_bool).astype(float)
            out["calibration_error"] = round(
                float((hit - claim[scoreable.index]).mean() * 100), 1
            )
        out["bands"] = _bands(done)

    if backtest_scored is not None and not backtest_scored.empty:
        held = backtest_scored["correct"].dropna().map(_as_bool)
        out["backtest"] = {
            "fixtures": len(backtest_scored),
            "verdict_accuracy": _pct_or_none(held.mean()) if len(held) else None,
            "log_loss": facts.get("backtest_log_loss"),
            **_split(backtest_scored),
            "contradictions": int(
                (backtest_scored["verdict"] == "NO CALL - both sides backed").sum()
            ),
            "bands": _bands(backtest_scored),
        }
    return out


# --- the model -----------------------------------------------------------


def elo_table(history: pd.DataFrame, latest, days: int = 7) -> list[dict]:
    """The current ratings, and how far each has moved in the last `days`."""
    table = plea.ratings_as_of(history)
    if table.empty:
        return []

    since = pd.Timestamp(latest).normalize() - pd.Timedelta(days=days)
    recent = history[history["date"] > since].sort_values("date", kind="stable")
    moved = recent.groupby("team").agg(first=("elo_before", "first"), last=("elo_after", "last"))
    delta = moved["last"] - moved["first"]

    lo = float(table["elo"].min()) - 40
    span = (float(table["elo"].max()) - lo) or 1.0
    return [
        {
            "rank": int(r.Index),
            "team": r.team,
            "elo": round(float(r.elo)),
            "played": int(r.played),
            "relative": round(float(r.elo - lo) / span * 100),
            "delta": round(float(delta.get(r.team, 0.0))),
        }
        for r in table.itertuples()
    ]


def model_view(
    coefficients: pd.DataFrame,
    history: pd.DataFrame,
    latest,
    calls: pd.DataFrame,
) -> dict:
    top = float(coefficients["coefficient"].abs().max()) or 1.0
    features = []
    for r in coefficients.itertuples(index=False):
        name, note = FEATURE_NOTES.get(r.feature, (r.feature, ""))
        features.append({
            "column": r.feature,
            "name": name,
            "note": note,
            "coefficient": round(float(r.coefficient), 4),
            "odds_x": round(float(r.odds_x), 2),
            "relative": round(abs(float(r.coefficient)) / top * 100),
        })

    verdict = calls["verdict"] if not calls.empty else pd.Series(dtype=object)
    grid = {
        "n": len(calls),
        "home_win": int((verdict == "HOME will win").sum()),
        "away_win": int((verdict == "AWAY will win").sum()),
        "weak": int((verdict == "HOME will not win").sum()),
        "contradiction": int((verdict == "NO CALL - both sides backed").sum()),
    }

    return {
        "columns": list(model.FEATURES),
        "features": features,
        "grid": grid,
        "elo": elo_table(history, latest),
    }
