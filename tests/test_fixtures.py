"""The season fixture list is a third-party file, and is held at arm's length.

What matters: club names go through the canonical map (an unknown one
raises), round numbers come through, dates are read day-first, the score
column is ignored, and "played" is judged against the results file rather
than against the fixture file's own scores.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
import requests

from plfootball import config, fixtures
from plfootball.teams import UnknownClubError

RAW = """Match Number,Round Number,Date,Location,Home Team,Away Team,Result
1,1,21/08/2026 20:00,Emirates Stadium,Arsenal,Coventry,3 - 0
2,1,22/08/2026 15:00,MKM Stadium,Hull,Man Utd,2 - 0
3,2,29/08/2026 15:00,Old Trafford,Man Utd,Arsenal,
4,2,30/08/2026 14:00,Coventry Building Society Arena,Coventry,Hull,
"""


def _raw() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(RAW))


def _results(*played) -> pd.DataFrame:
    """(date, home, away) for each match the results file has a score for."""
    return pd.DataFrame([
        {"date": pd.Timestamp(d), "season": 2027, "home_team": h, "away_team": a,
         "home_goals": 1, "away_goals": 0}
        for d, h, a in played
    ])


class _Response:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200

    def raise_for_status(self):
        pass


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    return tmp_path


# --- parsing -------------------------------------------------------------


def test_parse_canonicalises_names_and_keeps_rounds():
    parsed = fixtures.parse(_raw())
    assert list(parsed.columns) == fixtures.COLUMNS
    assert set(parsed["away_team"]) == {
        "Coventry City", "Manchester United", "Arsenal", "Hull City"
    }
    assert parsed["round"].tolist() == [1, 1, 2, 2]


def test_parse_reads_dates_day_first():
    parsed = fixtures.parse(_raw())
    assert parsed["date"].iloc[0] == pd.Timestamp("2026-08-21")
    assert parsed["kickoff"].iloc[0] == "20:00"


def test_parse_drops_the_score_column():
    assert "Result" not in fixtures.parse(_raw()).columns


def test_an_unknown_club_raises():
    raw = _raw()
    raw.loc[0, "Home Team"] = "Wrexham"
    with pytest.raises(UnknownClubError):
        fixtures.parse(raw)


def test_a_file_missing_columns_raises():
    with pytest.raises(ValueError, match="Round Number"):
        fixtures.parse(_raw().drop(columns=["Round Number"]))


# --- what counts as played -----------------------------------------------


def test_played_is_judged_by_the_results_file_not_the_score_column():
    """Both round-one matches carry a score in the file; only one is in our results."""
    parsed = fixtures.parse(_raw())
    mask = fixtures.played_mask(parsed, _results(("2026-08-21", "Arsenal", "Coventry City")))
    assert mask.tolist() == [True, False, False, False]


def test_a_rescheduled_match_still_counts_as_played():
    """Moved four days for television: same pairing, different date."""
    parsed = fixtures.parse(_raw())
    mask = fixtures.played_mask(parsed, _results(("2026-08-25", "Arsenal", "Coventry City")))
    assert bool(mask.iloc[0]) is True


def test_rounds_count_played_and_remaining():
    parsed = fixtures.parse(_raw())
    table = fixtures.rounds(parsed, _results(("2026-08-21", "Arsenal", "Coventry City")))
    first = table.set_index("round").loc[1]
    assert (first["fixtures"], first["played"], first["remaining"]) == (2, 1, 1)
    assert first["first"] == pd.Timestamp("2026-08-21")
    assert fixtures.next_round(table) == 1


def test_next_round_skips_finished_rounds():
    parsed = fixtures.parse(_raw())
    table = fixtures.rounds(parsed, _results(
        ("2026-08-21", "Arsenal", "Coventry City"),
        ("2026-08-22", "Hull City", "Manchester United"),
    ))
    assert fixtures.next_round(table) == 2


def test_next_round_is_none_once_the_season_is_done():
    parsed = fixtures.parse(_raw())
    table = fixtures.rounds(parsed, _results(
        ("2026-08-21", "Arsenal", "Coventry City"),
        ("2026-08-22", "Hull City", "Manchester United"),
        ("2026-08-29", "Manchester United", "Arsenal"),
        ("2026-08-30", "Coventry City", "Hull City"),
    ))
    assert fixtures.next_round(table) is None


def test_for_round_and_unplayed():
    parsed = fixtures.parse(_raw())
    block = fixtures.for_round(parsed, 2)
    assert len(block) == 2
    left = fixtures.unplayed(block, _results(("2026-08-29", "Manchester United", "Arsenal")))
    assert left["home_team"].tolist() == ["Coventry City"]


# --- fetching ------------------------------------------------------------


def test_download_writes_the_file_and_returns_it_parsed(raw_dir, monkeypatch):
    asked = []

    def get(url, **kwargs):
        asked.append(url)
        return _Response(RAW.encode())

    monkeypatch.setattr(requests, "get", get)
    parsed = fixtures.download(2027)

    assert len(parsed) == 4
    assert "epl-2026" in asked[0]  # 2026/27 is named by its starting year there
    assert fixtures.raw_path(2027).exists()


def test_a_bad_file_is_never_written(raw_dir, monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda url, **k: _Response(b"<!doctype html><html>nope</html>")
    )
    with pytest.raises(ValueError):
        fixtures.download(2027)
    assert not fixtures.raw_path(2027).exists()


def test_load_uses_the_disk_copy_without_the_network(raw_dir, monkeypatch):
    fixtures.raw_path(2027).write_text(RAW, encoding="utf-8")

    def boom(url, **kwargs):
        raise AssertionError("should not have gone to the network")

    monkeypatch.setattr(requests, "get", boom)
    assert len(fixtures.load(2027)) == 4


def test_load_reports_a_missing_file_when_offline(raw_dir, monkeypatch):
    def offline(url, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", offline)
    with pytest.raises(fixtures.NoSeasonFixtures):
        fixtures.load(2027)
    with pytest.raises(fixtures.NoSeasonFixtures):
        fixtures.load(2027, fetch_if_missing=False)
