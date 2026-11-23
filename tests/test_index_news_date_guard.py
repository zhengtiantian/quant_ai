"""Unit tests for news_v1 Qdrant indexer date-range validation (tools/index_news.py)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import index_news  # noqa: E402

FIXED_NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def test_to_date8_and_to_int_date_handle_dual_formats():
    assert index_news.to_date8("20261123") == "20261123"
    assert index_news.to_date8("20261123154500") == "20261123"
    assert index_news.to_date8("invalid") == ""
    assert index_news.to_int_date("20261123") == 20261123
    assert index_news.to_int_date("20261123154500") == 20261123
    assert index_news.to_int_date(None) is None


def test_is_valid_date_int_bounds_and_calendar_validation():
    assert index_news.is_valid_date_int(20150101, now=FIXED_NOW) is True
    assert index_news.is_valid_date_int(20261124, now=FIXED_NOW) is True   # today + 1d
    assert index_news.is_valid_date_int(20261125, now=FIXED_NOW) is False  # > today + 1d
    assert index_news.is_valid_date_int(20141231, now=FIXED_NOW) is False
    assert index_news.is_valid_date_int(19350901, now=FIXED_NOW) is False
    assert index_news.is_valid_date_int(20340719, now=FIXED_NOW) is False
    assert index_news.is_valid_date_int(20201345, now=FIXED_NOW) is False  # invalid month/day
    assert index_news.is_valid_date_int(20210229, now=FIXED_NOW) is False  # non-leap Feb 29
    assert index_news.is_valid_date_int(None, now=FIXED_NOW) is False
    assert index_news.is_valid_date_int(0, now=FIXED_NOW) is False
    assert index_news.is_valid_date_int(True, now=FIXED_NOW) is False
