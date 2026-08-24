import pandas as pd
import pytest

from plfootball.teams import (
    CANONICAL_CLUBS,
    UnknownClubError,
    canonical,
    canonicalise,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Man United", "Manchester United"),
        ("Man Utd", "Manchester United"),
        ("Manchester Utd", "Manchester United"),
        ("Manchester United", "Manchester United"),
        ("Wolves", "Wolverhampton Wanderers"),
        ("Brighton", "Brighton and Hove Albion"),
        ("Nott'm Forest", "Nottingham Forest"),
        ("Nott'ham Forest", "Nottingham Forest"),
        ("QPR", "Queens Park Rangers"),
        ("Sheffield Utd", "Sheffield United"),
        ("West Brom", "West Bromwich Albion"),
        ("Spurs", "Tottenham Hotspur"),
    ],
)
def test_known_aliases(raw, expected):
    assert canonical(raw) == expected


def test_normalisation_is_forgiving():
    assert canonical("  man  united ") == "Manchester United"
    assert canonical("MAN UNITED") == "Manchester United"
    assert canonical("Nott’m Forest") == "Nottingham Forest"  # curly apostrophe


def test_canonical_names_map_to_themselves():
    for club in CANONICAL_CLUBS:
        assert canonical(club) == club


def test_unknown_club_raises():
    with pytest.raises(UnknownClubError, match="Real Madrid"):
        canonical("Real Madrid")


def test_series_reports_every_bad_name_at_once():
    s = pd.Series(["Arsenal", "Barcelona", "Man Utd", "Ajax"])
    with pytest.raises(UnknownClubError) as exc:
        canonicalise(s)
    assert "Ajax" in str(exc.value)
    assert "Barcelona" in str(exc.value)


def test_series_round_trip():
    s = pd.Series(["Man Utd", "Wolves", "Brighton"])
    assert list(canonicalise(s)) == [
        "Manchester United",
        "Wolverhampton Wanderers",
        "Brighton and Hove Albion",
    ]
