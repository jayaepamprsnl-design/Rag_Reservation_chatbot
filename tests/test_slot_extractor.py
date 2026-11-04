"""
Unit tests for app/chat/slot_extractor.py.

Run with:  pytest tests/test_slot_extractor.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.chat.slot_extractor import Slots, extract_slots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today(h: int = 25, minute: int = 0) -> datetime:
    """Return a datetime N hours in the future."""
    return datetime.now().replace(second=0, microsecond=0) + timedelta(hours=h)


# ---------------------------------------------------------------------------
# Customer name extraction
# ---------------------------------------------------------------------------


class TestNameExtraction:

    def test_my_name_is(self):
        s = extract_slots("my name is jaya kumari i want to book a spot")
        assert s.customer_name == "jaya kumari"

    def test_my_name_is_first_only(self):
        s = extract_slots("my name is Alice")
        assert s.customer_name == "Alice"

    def test_i_am(self):
        s = extract_slots("I am John Smith and I need parking")
        assert s.customer_name == "John Smith"

    def test_im_prefix(self):
        s = extract_slots("I'm Mary Jane, I want to book a slot")
        assert s.customer_name == "Mary Jane"

    def test_called_prefix(self):
        s = extract_slots("called Bob Jones")
        assert s.customer_name == "Bob Jones"

    def test_no_name_returns_none(self):
        s = extract_slots("I want to park at central tomorrow")
        assert s.customer_name is None

    def test_name_stops_at_i_want(self):
        s = extract_slots("my name is jaya kumari i want to park")
        # Must not include "i want to park"
        assert s.customer_name == "jaya kumari"

    def test_date_word_not_captured_as_name(self):
        s = extract_slots("book for 4th nov 2026 from 2pm")
        # "4th nov" starts with digit — should not be a name
        assert s.customer_name is None

    def test_lot_keyword_not_captured_as_name(self):
        s = extract_slots("reserve for central parking")
        assert s.customer_name is None


# ---------------------------------------------------------------------------
# Car plate extraction
# ---------------------------------------------------------------------------


class TestPlateExtraction:

    def test_hyphenated_plate(self):
        s = extract_slots("my plate is AB-1234")
        assert s.car_plate == "AB-1234"

    def test_no_hyphen_plate(self):
        s = extract_slots("plate XY9900")
        assert s.car_plate == "XY9900"

    def test_year_not_captured_as_plate(self):
        s = extract_slots("I want to park on 4th nov 2026")
        assert s.car_plate is None

    def test_time_not_captured_as_plate(self):
        s = extract_slots("from 2pm till 5pm")
        assert s.car_plate is None


# ---------------------------------------------------------------------------
# Lot hint extraction
# ---------------------------------------------------------------------------


class TestLotHintExtraction:

    def test_central(self):
        s = extract_slots("I want to park at central")
        assert s.lot_hint == "central"

    def test_eastside(self):
        s = extract_slots("eastside parking please")
        assert s.lot_hint == "eastside"

    def test_airport(self):
        s = extract_slots("I need airport parking")
        assert s.lot_hint == "airport"

    def test_lot_id(self):
        s = extract_slots("lot 2 please")
        assert s.lot_hint == "2"

    def test_no_lot_hint(self):
        s = extract_slots("my name is Alice Smith")
        assert s.lot_hint is None


# ---------------------------------------------------------------------------
# Date / time extraction — the key regression test
# ---------------------------------------------------------------------------


class TestStartEndExtraction:

    def _anchor(self) -> datetime:
        """Fixed reference: 2026-11-02 12:00 so tests are deterministic."""
        return datetime(2026, 11, 2, 12, 0)

    def test_jaya_full_sentence(self):
        """The original bug report sentence."""
        today = datetime(2026, 11, 2, 12, 0)
        s = extract_slots(
            "my name is jaya kumari i want to book a spot "
            "for 4th nov 2026 from 2pm till 3hrs",
            today=today,
        )
        assert s.customer_name == "jaya kumari"
        assert s.start_time is not None
        assert s.start_time.date() == datetime(2026, 11, 4).date()
        assert s.start_time.hour == 14
        assert s.end_time is not None
        assert s.end_time.hour == 17   # 2pm + 3h = 5pm

    def test_explicit_till_time(self):
        today = datetime(2026, 11, 2, 12, 0)
        s = extract_slots("from 2pm till 5pm on 4th nov 2026", today=today)
        assert s.start_time is not None
        assert s.start_time.hour == 14
        assert s.end_time is not None
        assert s.end_time.hour == 17

    def test_duration_for_hours(self):
        today = datetime(2026, 11, 2, 12, 0)
        s = extract_slots("from 10am for 2 hours on 4th nov 2026", today=today)
        assert s.start_time is not None and s.start_time.hour == 10
        assert s.end_time is not None and s.end_time.hour == 12

    def test_duration_90_minutes(self):
        today = datetime(2026, 11, 2, 12, 0)
        s = extract_slots("from 3pm for 90 minutes on 5th nov 2026", today=today)
        assert s.start_time is not None and s.start_time.hour == 15
        assert s.end_time is not None
        assert (s.end_time - s.start_time).seconds == 90 * 60

    def test_past_date_returns_none(self):
        today = datetime(2026, 11, 2, 12, 0)
        s = extract_slots("from 10am on 1st jan 2020", today=today)
        assert s.start_time is None

    def test_no_times_returns_none(self):
        s = extract_slots("my name is Alice Smith")
        assert s.start_time is None
        assert s.end_time is None


# ---------------------------------------------------------------------------
# Combined / all-fields-at-once
# ---------------------------------------------------------------------------


class TestCombined:

    def test_all_except_plate(self):
        today = datetime(2026, 11, 2, 12, 0)
        s = extract_slots(
            "my name is Alice Smith, I want to park at central "
            "on 4th nov 2026 from 10am till 2pm",
            today=today,
        )
        assert s.customer_name == "Alice Smith"
        assert s.lot_hint == "central"
        assert s.start_time is not None and s.start_time.hour == 10
        assert s.end_time is not None and s.end_time.hour == 14

    def test_name_and_plate_only(self):
        s = extract_slots("my name is Bob Jones, plate AB-1234")
        assert s.customer_name == "Bob Jones"
        assert s.car_plate == "AB-1234"
        assert s.start_time is None

    def test_no_slots_empty_result(self):
        s = extract_slots("hello")
        assert s == Slots()
