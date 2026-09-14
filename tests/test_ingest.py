"""The downloader's job is to notice when the file on disk is out of date.

Every other season is finished and frozen; the one being played now is the only
one that changes, and it is the only one that matters for a prediction. A
stale copy of it does not raise - the pipeline runs, the table builds, the
predictor prints a grid. It just prints last week's grid again, and last week's
calls never settle. These tests are about that one silent case.
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from plfootball import config, ingest


@pytest.fixture(autouse=True)
def raw_dir(tmp_path, monkeypatch):
    """Keep every test off the real data/raw."""
    monkeypatch.setattr(config, "RAW", tmp_path)
    return tmp_path


def _write_season(end_year: int, played: int) -> None:
    """A raw file in football-data.co.uk's shape, with `played` results in it."""
    rows = [
        {
            "Div": "E0",
            "Date": f"{(i % 28) + 1:02d}/09/2026",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 1,
            "FTAG": 0,
            "FTR": "H",
        }
        for i in range(played)
    ]
    pd.DataFrame(rows).to_csv(ingest.raw_path(end_year), index=False)


def _served(content: bytes):
    """A stand-in for requests.get that records whether it was called."""
    calls = []

    class Response:
        def __init__(self):
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            pass

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    return get, calls


_A_SEASON = (
    b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    b"E0,13/09/2026,Arsenal,Chelsea,2,1,H\n"
)


# --- is the file on disk the whole season? -------------------------------


def test_a_finished_season_is_complete():
    _write_season(2026, config.SEASON_FIXTURES)
    assert ingest.is_complete(2026)


def test_a_season_in_progress_is_not_complete():
    _write_season(2027, 30)
    assert not ingest.is_complete(2027)


def test_a_missing_file_is_not_complete():
    assert not ingest.is_complete(2027)


def test_fixtures_without_scores_do_not_count_as_played():
    """Some files carry the rest of the season as blank rows."""
    played = pd.DataFrame({
        "Div": ["E0"] * config.SEASON_FIXTURES,
        "Date": ["13/09/2026"] * config.SEASON_FIXTURES,
        "HomeTeam": ["Arsenal"] * config.SEASON_FIXTURES,
        "AwayTeam": ["Chelsea"] * config.SEASON_FIXTURES,
        "FTHG": [1] * 30 + [None] * (config.SEASON_FIXTURES - 30),
        "FTAG": [0] * 30 + [None] * (config.SEASON_FIXTURES - 30),
        "FTR": ["H"] * 30 + [None] * (config.SEASON_FIXTURES - 30),
    })
    played.to_csv(ingest.raw_path(2027), index=False)
    assert not ingest.is_complete(2027)


def test_an_unreadable_file_is_not_complete():
    ingest.raw_path(2027).write_bytes(b"\xff\xfe not a csv")
    assert not ingest.is_complete(2027)


# --- what that means for downloading -------------------------------------


def test_a_finished_season_is_not_downloaded_again(monkeypatch):
    _write_season(2026, config.SEASON_FIXTURES)
    get, calls = _served(_A_SEASON)
    monkeypatch.setattr(requests, "get", get)

    assert ingest.download_season(2026) is False
    assert calls == []


def test_the_season_in_progress_is_always_downloaded_again(monkeypatch):
    """The bug this file exists for: a part-season on disk was treated as done,
    so results stopped arriving the week after the first run."""
    _write_season(2027, 30)
    get, calls = _served(_A_SEASON)
    monkeypatch.setattr(requests, "get", get)

    assert ingest.download_season(2027) is True
    assert len(calls) == 1
    assert ingest.raw_path(2027).read_bytes() == _A_SEASON


def test_a_missing_season_is_downloaded(monkeypatch):
    get, calls = _served(_A_SEASON)
    monkeypatch.setattr(requests, "get", get)

    assert ingest.download_season(2027) is True
    assert len(calls) == 1


def test_force_refetches_even_a_finished_season(monkeypatch):
    _write_season(2026, config.SEASON_FIXTURES)
    get, calls = _served(_A_SEASON)
    monkeypatch.setattr(requests, "get", get)

    assert ingest.download_season(2026, force=True) is True
    assert len(calls) == 1


def test_a_web_page_is_never_saved_as_a_season(monkeypatch):
    """An unpublished season answers HTTP 300 with HTML, not 404."""
    get, _ = _served(b"<!DOCTYPE html><html><body>Multiple Choices</body></html>")
    monkeypatch.setattr(requests, "get", get)

    with pytest.raises(ingest.SeasonUnavailable):
        ingest.download_season(2028)


def test_a_stale_part_season_survives_a_network_failure(monkeypatch):
    """Better last week's data than none. download_all must not delete it."""
    _write_season(2027, 30)

    def boom(url, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", boom)
    outcome = ingest.download_all(seasons=[2027])

    assert "network error" in outcome[2027]
    assert ingest.raw_path(2027).exists()
    assert ingest.is_complete(2027) is False
