"""The US stock market's calendar (NYSE and Nasdaq share it), 2026-11-26: full-day holidays and the
13:00 ET early closes, computed from the exchanges' standing rules so no yearly list has to be kept.

Without it every weekday looked like a trading day: on Thanksgiving 2026 the holdings showed
"pre-market" and the regular close of the day before instead of that day's after-hours price. A
holiday now reads as closed, exactly like a weekend.

Unscheduled closures (a national day of mourning, a storm) are not known in advance and are not
covered; on such a day Nasdaq's own "Closed" status still wins where a quote carries it.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import FrozenSet

REGULAR_CLOSE_HOUR = 16
EARLY_CLOSE_HOUR = 13
POST_HOURS = 4                      # after-hours trading runs until four hours after the close


def _easter(year: int) -> date:
    """Easter Sunday (Gregorian; the anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7          # noqa: E741 - the algorithm's own name
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` (Monday 0) of the month; n = -1 for the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = (date(year + month // 12, month % 12 + 1, 1)) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """A fixed-date holiday on a Saturday is taken on the Friday, on a Sunday on the Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=32)
def holidays(year: int) -> FrozenSet[date]:
    """The full-day market holidays of the year."""
    out = set()
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        out.add(new_year + timedelta(days=1))
    elif new_year.weekday() < 5:            # on a Saturday it is not moved to the Friday before
        out.add(new_year)
    out.add(_nth_weekday(year, 1, 0, 3))    # Martin Luther King Jr. Day
    out.add(_nth_weekday(year, 2, 0, 3))    # Washington's Birthday
    out.add(_easter(year) - timedelta(days=2))   # Good Friday
    out.add(_nth_weekday(year, 5, 0, -1))   # Memorial Day
    if year >= 2022:
        out.add(_observed(date(year, 6, 19)))    # Juneteenth
    out.add(_observed(date(year, 7, 4)))    # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1))    # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving
    out.add(_observed(date(year, 12, 25)))  # Christmas
    return frozenset(out)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


@lru_cache(maxsize=32)
def early_closes(year: int) -> FrozenSet[date]:
    """Days the regular session ends at 13:00 ET: the day after Thanksgiving, and July 3 and
    Christmas Eve when they fall Monday to Thursday."""
    out = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}
    for d in (date(year, 7, 3), date(year, 12, 24)):
        if d.weekday() <= 3 and is_trading_day(d):
            out.add(d)
    return frozenset(out)


def close_hour(d: date) -> int:
    """The hour (ET) the regular session of a trading day ends."""
    return EARLY_CLOSE_HOUR if d in early_closes(d.year) else REGULAR_CLOSE_HOUR


def post_end_hour(d: date) -> int:
    """The hour (ET) after-hours trading of a trading day ends: 20:00, or 17:00 after an early close."""
    return close_hour(d) + POST_HOURS


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d
