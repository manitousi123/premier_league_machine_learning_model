"""Canonical club names.

Every data source spells clubs differently. football-data.co.uk says "Man
United", FBref's team column says "Manchester United" and its opponent column
says "Manchester Utd". PLEA is a chain — each match updates a rating that feeds
the next one — so if one club arrives under two names it grows two half-finished
rating histories and every number downstream is wrong.

So: one canonical name per club, applied to every team column at load time, and
an unknown name raises instead of passing through. Silent pass-through is how
you lose a club for a whole season without noticing.
"""

from __future__ import annotations

import re

import pandas as pd


class UnknownClubError(ValueError):
    """A club name that isn't in the map. Add it to ALIASES, don't work around it."""


# Canonical name -> every spelling seen in the wild.
# The canonical name is itself always a valid alias.
_ALIASES: dict[str, list[str]] = {
    "Arsenal": [],
    "Aston Villa": ["Villa"],
    "Birmingham City": ["Birmingham"],
    "Blackburn Rovers": ["Blackburn"],
    "Blackpool": [],
    "Bolton Wanderers": ["Bolton"],
    "Bournemouth": ["AFC Bournemouth"],
    "Brentford": [],
    "Brighton and Hove Albion": ["Brighton", "Brighton & Hove Albion"],
    "Burnley": [],
    "Cardiff City": ["Cardiff"],
    "Chelsea": [],
    "Coventry City": ["Coventry"],
    "Crystal Palace": [],
    "Everton": [],
    "Fulham": [],
    "Huddersfield Town": ["Huddersfield"],
    "Hull City": ["Hull"],
    "Ipswich Town": ["Ipswich"],
    "Leeds United": ["Leeds", "Leeds Utd"],
    "Leicester City": ["Leicester"],
    "Liverpool": [],
    "Luton Town": ["Luton"],
    "Manchester City": ["Man City"],
    "Manchester United": ["Man United", "Man Utd", "Manchester Utd"],
    "Middlesbrough": ["Middlesboro"],
    "Newcastle United": ["Newcastle", "Newcastle Utd", "Newcastle Untited"],
    "Norwich City": ["Norwich"],
    "Nottingham Forest": ["Nott'm Forest", "Nott'ham Forest", "Nottm Forest"],
    "Queens Park Rangers": ["QPR", "Queens Park Rgs"],
    "Reading": [],
    "Sheffield United": ["Sheffield Utd", "Sheff United", "Sheff Utd"],
    "Southampton": [],
    "Stoke City": ["Stoke"],
    "Sunderland": [],
    "Swansea City": ["Swansea"],
    "Tottenham Hotspur": ["Tottenham", "Spurs"],
    "Watford": [],
    "West Bromwich Albion": ["West Brom", "West Bromwich"],
    "West Ham United": ["West Ham"],
    "Wigan Athletic": ["Wigan"],
    "Wolverhampton Wanderers": ["Wolves", "Wolverhampton"],
}

CANONICAL_CLUBS: frozenset[str] = frozenset(_ALIASES)


def _normalise(name: str) -> str:
    """Fold away the differences that never carry meaning."""
    name = str(name).strip().lower()
    name = name.replace("’", "'")  # curly apostrophe -> plain
    name = name.replace(".", "")
    return re.sub(r"\s+", " ", name)


_LOOKUP: dict[str, str] = {}
for _canon, _alts in _ALIASES.items():
    for _variant in (_canon, *_alts):
        _key = _normalise(_variant)
        if _key in _LOOKUP and _LOOKUP[_key] != _canon:
            raise RuntimeError(f"alias {_variant!r} maps to two clubs")
        _LOOKUP[_key] = _canon


def canonical(name: str) -> str:
    """Map any known spelling to the canonical club name.

    Raises UnknownClubError on anything unrecognised — including a newly
    promoted club, which is the point: add it to _ALIASES deliberately.
    """
    try:
        return _LOOKUP[_normalise(name)]
    except KeyError:
        raise UnknownClubError(
            f"unknown club {name!r}. Add it to _ALIASES in teams.py — "
            "do not strip or skip the row."
        ) from None


def canonicalise(names: pd.Series) -> pd.Series:
    """Canonicalise a whole column, reporting every bad name at once."""
    unknown = sorted({n for n in names.dropna().unique() if _normalise(n) not in _LOOKUP})
    if unknown:
        raise UnknownClubError(
            f"{len(unknown)} unknown club name(s): {unknown}. "
            "Add them to _ALIASES in teams.py."
        )
    return names.map(canonical)
