"""Reservation state machine — one session per session_id, in-process memory.

State flow
----------
IDLE → ASK_NAME → ASK_CAR_PLATE → ASK_LOT → ASK_START → ASK_END → CONFIRM → DONE

Design notes
------------
* At every active step the user may type "cancel" / "quit" to abort.
* ASK_START accepts a full range ("tomorrow 10am to 2pm") which skips ASK_END.
* Availability and conflict checks happen before the CONFIRM summary is shown.
* The DB write happens only on "confirm" — no partial records created.

Public API
----------
start(session_id)                         -> str           first greeting
prefill_name(session_id, name)            -> str | None    skip name step
advance(session_id, user_input, db)       -> ReplyResult
get_state(session_id)                     -> ReservationState
reset(session_id)                         -> None
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import dateparser
from sqlalchemy.orm import Session

from app.core.logging_config import logger
from app.db.models import ParkingLot
from app.db.repository import (
    ValidationError,
    check_conflicts,
    create_reservation,
    get_availability,
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ReservationState(str, Enum):
    IDLE = "IDLE"
    ASK_NAME = "ASK_NAME"
    ASK_CAR_PLATE = "ASK_CAR_PLATE"
    ASK_LOT = "ASK_LOT"
    ASK_START = "ASK_START"
    ASK_END = "ASK_END"
    CONFIRM = "CONFIRM"
    DONE = "DONE"


@dataclass
class ReplyResult:
    message: str
    state: ReservationState
    reservation_id: int | None = None
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Per-session data (private)
# ---------------------------------------------------------------------------


@dataclass
class _SessionData:
    state: ReservationState = ReservationState.IDLE
    name: str = ""
    car_plate: str = ""
    lot_id: int | None = None
    lot_name: str = ""
    lot_hourly: float = 0.0
    lot_daily: float = 0.0
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    reservation_id: int | None = None


_sessions: dict[str, _SessionData] = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANCEL_WORDS = frozenset(
    {"cancel", "quit", "stop", "exit", "abort", "never mind", "nevermind"}
)
_CONFIRM_WORDS = frozenset(
    {"yes", "y", "confirm", "ok", "sure", "yep", "yeah", "book", "proceed"}
)

_PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,8}[A-Z0-9]$", re.IGNORECASE)
_MIN_ADVANCE_MINUTES = 60
_MIN_DURATION_MINUTES = 30

_DP_FUTURE = {"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False}

# Range-detection patterns (applied in order; first match wins)
# Pattern A: "[WORD_DATE(s)] from [TIME] to [TIME]"  e.g. "tomorrow from 10am to 2pm"
_RANGE_A = re.compile(r"^((?:\w+\s+)+?)from\s+(.+?)\s+to\s+(.+)$", re.I)
# Pattern B: "from [DATETIME] to [DATETIME]"
_RANGE_B = re.compile(r"^from\s+(.+?)\s+to\s+(.+)$", re.I)
# Pattern C: "[DATETIME] to [DATETIME]"  e.g. "2026-11-15 10:00 to 14:00"
_RANGE_C = re.compile(r"^(.+?)\s+to\s+(.+)$", re.I)

# ---------------------------------------------------------------------------
# Date/time helpers
# ---------------------------------------------------------------------------


def _tomorrow_str() -> str:
    """Return tomorrow's date as YYYY-MM-DD."""
    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


def _mask_plate(plate: str) -> str:
    """ABC-1234 → ABC****34  (show first 3 and last 2 chars)."""
    if len(plate) <= 4:
        return plate[0] + "***"
    return plate[:3] + "****" + plate[-2:]


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%a %d %b %Y, %H:%M")


def _duration_str(start: datetime, end: datetime) -> str:
    total_mins = int((end - start).total_seconds()) // 60
    h, m = divmod(total_mins, 60)
    return f"{h}h {m:02d}m"


def _estimated_cost(
    start: datetime, end: datetime, hourly: float, daily: float
) -> str:
    hours = (end - start).total_seconds() / 3600
    cost = min(math.ceil(hours / 24) * daily, hours * hourly) if hours > 24 else hours * hourly
    return f"~${cost:.2f}"


def _build_summary(data: _SessionData) -> str:
    assert data.start_dt and data.end_dt
    sep = "-" * 44
    lines = [
        "",
        "  BOOKING SUMMARY",
        f"  {sep}",
        f"  Customer  :  {data.name}",
        f"  Plate     :  {_mask_plate(data.car_plate)}",
        f"  Facility  :  {data.lot_name}",
        f"  Arrives   :  {_fmt_dt(data.start_dt)}",
        f"  Departs   :  {_fmt_dt(data.end_dt)}",
        f"  Duration  :  {_duration_str(data.start_dt, data.end_dt)}",
        f"  Est. cost :  {_estimated_cost(data.start_dt, data.end_dt, data.lot_hourly, data.lot_daily)}",
        f"  {sep}",
        "",
        '  Reply "confirm" to book, or "cancel" to start over.',
        "",
    ]
    return "\n".join(lines)


def _lot_menu(lots: list[ParkingLot]) -> str:
    lines = ["Which facility would you like?", ""]
    for lot in lots:
        avail_pct = lot.available_spaces / max(lot.total_spaces, 1) * 100
        status = "available" if lot.available_spaces > 0 else "FULL"
        lines.append(
            f"  {lot.id}.  {lot.name:<22}  ${lot.hourly_price:.2f}/hr  "
            f"({lot.available_spaces}/{lot.total_spaces} spaces — {status})"
        )
    lines += ["", "  Type the number (1/2/3) or part of the name."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _is_cancel(text: str) -> bool:
    return text.strip().lower() in _CANCEL_WORDS


def _validate_name(text: str) -> tuple[str, str]:
    cleaned = " ".join(text.split())
    if len(cleaned) < 2:
        return "", "Name is too short. Please enter your full name."
    if not re.search(r"[A-Za-z]", cleaned):
        return "", "Please enter a name that contains letters."
    return cleaned, ""


def _validate_plate(text: str) -> tuple[str, str]:
    cleaned = text.strip().upper()
    if not _PLATE_RE.match(cleaned):
        return "", (
            f"'{text}' doesn't look like a valid plate number. "
            "Use letters and digits, with an optional hyphen — e.g. ABC-1234 or XY9900."
        )
    return cleaned, ""


def _resolve_lot(text: str, lots: list[ParkingLot]) -> Optional[ParkingLot]:
    t = text.strip().lower()
    if t.isdigit():
        return next((l for l in lots if l.id == int(t)), None)
    for lot in lots:
        if t in lot.name.lower():
            return lot
    for keywords, fragment in [
        (["central", "downtown"], "central"),
        (["eastside", "east"],    "eastside"),
        (["airport", "connector", "transit"], "airport"),
    ]:
        if any(kw in t for kw in keywords):
            return next((l for l in lots if fragment in l.name.lower()), None)
    return None


def _parse_dt(text: str, reference: Optional[datetime] = None) -> Optional[datetime]:
    settings = dict(_DP_FUTURE)
    if reference:
        settings["RELATIVE_BASE"] = reference
    return dateparser.parse(text, settings=settings)


def _validate_start(
    text: str, lot: ParkingLot
) -> tuple[Optional[datetime], str]:
    now = datetime.now()
    dt = _parse_dt(text)

    # If the input looks like a bare time ("10am", "14:30") and it's already
    # past today, try anchoring to tomorrow.
    if dt is not None and dt <= now + timedelta(minutes=_MIN_ADVANCE_MINUTES):
        tomorrow_base = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
        dt2 = _parse_dt(text, reference=tomorrow_base)
        if dt2 and dt2 > now + timedelta(minutes=_MIN_ADVANCE_MINUTES):
            dt = dt2

    if dt is None:
        tomorrow = _tomorrow_str()
        return None, (
            "I couldn't understand that date and time.\n"
            f"Try: '{tomorrow} 10:00'  or  'tomorrow at 9am'  or  '10am' (assumes tomorrow)."
        )
    if dt <= now + timedelta(minutes=_MIN_ADVANCE_MINUTES):
        return None, (
            f"Arrival must be at least {_MIN_ADVANCE_MINUTES} minutes from now. "
            "Please choose a future time."
        )
    opens, closes = lot.opens_at, lot.closes_at
    if not (opens <= dt.time() <= closes):
        return None, (
            f"{lot.name} is open {opens.strftime('%H:%M')} – {closes.strftime('%H:%M')}. "
            "Please choose an arrival time within those hours."
        )
    return dt, ""


def _validate_end(
    text: str, start_dt: datetime, lot: ParkingLot
) -> tuple[Optional[datetime], str]:
    # Parse with start_dt as reference so relative expressions work correctly.
    dt = _parse_dt(text, reference=start_dt)

    # Re-anchor to start date if result falls before start (e.g. "2pm" when
    # start is tomorrow morning).
    if dt is not None and dt <= start_dt:
        dt = start_dt.replace(
            hour=dt.hour, minute=dt.minute, second=0, microsecond=0
        )

    # Last resort: parse as bare time and graft onto the start date.
    if dt is None:
        bare = dateparser.parse(text, settings={"RETURN_AS_TIMEZONE_AWARE": False})
        if bare:
            dt = start_dt.replace(
                hour=bare.hour, minute=bare.minute, second=0, microsecond=0
            )

    if dt is None:
        return None, (
            "I couldn't understand that departure time.\n"
            "Try: '14:00', '2pm', or '2026-11-15 14:00'."
        )
    if dt <= start_dt:
        return None, (
            f"Departure must be after arrival ({_fmt_dt(start_dt)}). Please try again."
        )
    if (dt - start_dt).total_seconds() < _MIN_DURATION_MINUTES * 60:
        return None, (
            f"Minimum booking is {_MIN_DURATION_MINUTES} minutes. "
            "Please choose a later departure time."
        )
    opens, closes = lot.opens_at, lot.closes_at
    if not (opens <= dt.time() <= closes):
        return None, (
            f"{lot.name} closes at {closes.strftime('%H:%M')}. "
            "Please choose a departure time before closing."
        )
    return dt, ""


# ---------------------------------------------------------------------------
# Date-time range parsing  ("tomorrow 10am to 2pm" → start + end at once)
# ---------------------------------------------------------------------------


def _parse_range(
    text: str, lot: ParkingLot
) -> tuple[Optional[datetime], Optional[datetime], str]:
    """Try to parse text as a start/end range.

    Handles:
        "tomorrow from 10am to 2pm"
        "tomorrow 10am to 2pm"
        "2026-11-15 10:00 to 14:00"
        "from 10am to 2pm"

    Returns (start_dt, end_dt, error_msg).
    Returns (None, None, '') when no range pattern is found.
    Returns (None, None, error_msg) when a range is found but validation fails.
    """
    # Quick guard — don't bother unless " to " is present
    if not re.search(r"\bto\b", text, re.I):
        return None, None, ""

    start_raw: str
    end_raw: str
    date_prefix: str = ""

    # Pattern A: "[word_date] from [time] to [time]"
    # e.g. "tomorrow from 10am to 2pm"
    # end_raw is the bare time only — _validate_end will anchor it to the
    # start date automatically, keeping both times on the same calendar day.
    m = _RANGE_A.match(text)
    if m:
        date_prefix = m.group(1).strip()
        start_raw   = f"{date_prefix} {m.group(2).strip()}"
        end_raw     = m.group(3).strip()  # bare time — anchored to start date
    else:
        # Pattern B: "from [datetime] to [datetime]"
        m = _RANGE_B.match(text)
        if m:
            start_raw = m.group(1).strip()
            end_raw   = m.group(2).strip()
        else:
            # Pattern C: "[datetime] to [datetime]"
            m = _RANGE_C.match(text)
            if not m:
                return None, None, ""
            start_raw = m.group(1).strip()
            end_raw   = m.group(2).strip()

    start_dt, start_err = _validate_start(start_raw, lot)
    if start_err:
        return None, None, start_err

    end_dt, end_err = _validate_end(end_raw, start_dt, lot)
    if end_err:
        return None, None, end_err

    return start_dt, end_dt, ""


# ---------------------------------------------------------------------------
# Availability check + CONFIRM transition (shared by ASK_START and ASK_END)
# ---------------------------------------------------------------------------


def _check_and_confirm(
    session_id: str,
    data: _SessionData,
    lot: ParkingLot,
    db: Session,
) -> ReplyResult:
    """Check availability then show the booking summary, or redirect to ASK_LOT."""
    avail = get_availability(db, data.lot_id)
    if avail["available_spaces"] <= 0:
        return _no_spaces_reply(data, lot.name, db)

    conflicts = check_conflicts(db, data.lot_id, data.start_dt, data.end_dt)
    if len(conflicts) >= lot.total_spaces:
        return _no_spaces_reply(data, lot.name, db, time_conflict=True)

    data.state = ReservationState.CONFIRM
    return ReplyResult(message=_build_summary(data), state=ReservationState.CONFIRM)


def _no_spaces_reply(
    data: _SessionData,
    lot_name: str,
    db: Session,
    time_conflict: bool = False,
) -> ReplyResult:
    reason = "fully booked for that time window" if time_conflict else "fully booked"
    data.lot_id = None
    data.lot_name = ""
    data.start_dt = None
    data.end_dt = None
    data.state = ReservationState.ASK_LOT
    lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
    return ReplyResult(
        message=(
            f"Sorry — {lot_name} is {reason}.\n"
            "Please choose a different facility:\n\n" + _lot_menu(lots)
        ),
        state=ReservationState.ASK_LOT,
    )


# ---------------------------------------------------------------------------
# Cancel helper
# ---------------------------------------------------------------------------


def _do_cancel(session_id: str) -> ReplyResult:
    reset(session_id)
    return ReplyResult(
        message=(
            "Reservation cancelled. No problem at all!\n"
            "Feel free to start a new booking whenever you're ready."
        ),
        state=ReservationState.IDLE,
        cancelled=True,
    )


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------


def _handle_ask_name(
    session_id: str, data: _SessionData, text: str, db: Session
) -> ReplyResult:
    name, err = _validate_name(text)
    if err:
        return ReplyResult(
            message=f"{err}\n\nWhat is your full name?",
            state=ReservationState.ASK_NAME,
        )
    data.name = name
    data.state = ReservationState.ASK_CAR_PLATE
    return ReplyResult(
        message=f"Nice to meet you, {name.split()[0]}! What is your vehicle's licence plate?\n(e.g. ABC-1234 or XY9900)",
        state=ReservationState.ASK_CAR_PLATE,
    )


def _handle_ask_plate(
    session_id: str, data: _SessionData, text: str, db: Session
) -> ReplyResult:
    plate, err = _validate_plate(text)
    if err:
        return ReplyResult(message=err, state=ReservationState.ASK_CAR_PLATE)
    data.car_plate = plate
    data.state = ReservationState.ASK_LOT
    lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
    return ReplyResult(message=_lot_menu(lots), state=ReservationState.ASK_LOT)


def _handle_ask_lot(
    session_id: str, data: _SessionData, text: str, db: Session
) -> ReplyResult:
    lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
    lot = _resolve_lot(text, lots)
    if lot is None:
        return ReplyResult(
            message=(
                f"I couldn't match '{text}' to a facility.\n"
                f"Please type a number (1–3) or part of the name.\n\n"
                + _lot_menu(lots)
            ),
            state=ReservationState.ASK_LOT,
        )
    data.lot_id    = lot.id
    data.lot_name  = lot.name
    data.lot_hourly = lot.hourly_price
    data.lot_daily  = lot.daily_price
    data.state = ReservationState.ASK_START
    tomorrow = _tomorrow_str()
    return ReplyResult(
        message=(
            f"Got it — {lot.name} "
            f"(open {lot.opens_at.strftime('%H:%M')}–{lot.closes_at.strftime('%H:%M')}).\n\n"
            "When would you like to park?\n\n"
            "You can give just an arrival time, or a full range in one go:\n"
            f"  •  '{tomorrow} 10:00'\n"
            f"  •  'tomorrow 9am'\n"
            f"  •  'tomorrow 10am to 2pm'       ← arrival + departure together\n"
            f"  •  '{tomorrow} 10:00 to 14:00'"
        ),
        state=ReservationState.ASK_START,
    )


def _handle_ask_start(
    session_id: str, data: _SessionData, text: str, db: Session
) -> ReplyResult:
    lot = db.get(ParkingLot, data.lot_id)

    # ── Try to parse as a full range first ────────────────────────────────────
    start_dt, end_dt, range_err = _parse_range(text, lot)

    if range_err:
        # Range pattern detected but validation failed — surface the error
        return ReplyResult(
            message=f"{range_err}\n\nPlease try again or provide just the arrival time.",
            state=ReservationState.ASK_START,
        )

    if start_dt is not None and end_dt is not None:
        # Both arrival and departure parsed — skip ASK_END entirely
        data.start_dt = start_dt
        data.end_dt   = end_dt
        return _check_and_confirm(session_id, data, lot, db)

    # ── Single start time ──────────────────────────────────────────────────────
    dt, err = _validate_start(text, lot)
    if err:
        tomorrow = _tomorrow_str()
        return ReplyResult(
            message=(
                f"{err}\n\n"
                "Tip: you can provide arrival + departure in one message, e.g.:\n"
                f"  'tomorrow 10am to 2pm'\n"
                f"  '{tomorrow} 10:00 to 14:00'"
            ),
            state=ReservationState.ASK_START,
        )

    data.start_dt = dt
    data.state = ReservationState.ASK_END
    tomorrow = _tomorrow_str()
    return ReplyResult(
        message=(
            f"Arrival set to {_fmt_dt(dt)}.\n\n"
            "When would you like to depart?\n"
            "(e.g. '14:00', '2pm', or a full date-time)"
        ),
        state=ReservationState.ASK_END,
    )


def _handle_ask_end(
    session_id: str, data: _SessionData, text: str, db: Session
) -> ReplyResult:
    lot = db.get(ParkingLot, data.lot_id)
    dt, err = _validate_end(text, data.start_dt, lot)
    if err:
        return ReplyResult(message=err, state=ReservationState.ASK_END)
    data.end_dt = dt
    return _check_and_confirm(session_id, data, lot, db)


def _handle_confirm(
    session_id: str, data: _SessionData, text: str, db: Session
) -> ReplyResult:
    if text.lower() not in _CONFIRM_WORDS:
        return ReplyResult(
            message='Please type "confirm" to complete the booking, or "cancel" to start over.',
            state=ReservationState.CONFIRM,
        )
    try:
        res = create_reservation(
            db=db,
            lot_id=data.lot_id,
            customer_name=data.name,
            car_plate=data.car_plate,
            start_time=data.start_dt,
            end_time=data.end_dt,
        )
    except ValidationError as exc:
        # Race condition — try again with a different facility
        lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
        data.lot_id = None; data.lot_name = ""
        data.start_dt = None; data.end_dt = None
        data.state = ReservationState.ASK_LOT
        return ReplyResult(
            message=(
                f"Booking failed: {exc}\n"
                "Please choose a facility to try again:\n\n" + _lot_menu(lots)
            ),
            state=ReservationState.ASK_LOT,
        )

    data.reservation_id = res.id
    data.state = ReservationState.DONE
    logger.info("Reservation confirmed: id=%d session=%s", res.id, session_id)
    return ReplyResult(
        message=(
            f"Your reservation is confirmed!\n\n"
            f"  Booking reference : CPK-{res.id:06d}\n"
            f"  Facility          : {data.lot_name}\n"
            f"  Arriving          : {_fmt_dt(data.start_dt)}\n\n"
            "We look forward to seeing you. Have a great visit!"
        ),
        state=ReservationState.DONE,
        reservation_id=res.id,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    ReservationState.ASK_NAME:      _handle_ask_name,
    ReservationState.ASK_CAR_PLATE: _handle_ask_plate,
    ReservationState.ASK_LOT:       _handle_ask_lot,
    ReservationState.ASK_START:     _handle_ask_start,
    ReservationState.ASK_END:       _handle_ask_end,
    ReservationState.CONFIRM:       _handle_confirm,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start(session_id: str) -> str:
    """Create (or reset) a session and return the opening question."""
    _sessions[session_id] = _SessionData(state=ReservationState.ASK_NAME)
    logger.debug("Reservation session started: %s", session_id)
    return (
        "I'd be happy to help you make a reservation!\n"
        "First, could I get your full name please?"
    )


def prefill_name(session_id: str, name: str) -> str | None:
    """Pre-populate the name from the initial booking request.

    If the session is in ASK_NAME and *name* is valid, stores the name,
    advances to ASK_CAR_PLATE, and returns the next prompt.
    Returns None if the session is in a different state or name is invalid.
    """
    data = _sessions.get(session_id)
    if not data or data.state != ReservationState.ASK_NAME:
        return None
    validated, err = _validate_name(name)
    if err:
        return None
    data.name = validated
    data.state = ReservationState.ASK_CAR_PLATE
    first = validated.split()[0]
    return (
        f"Got it — booking for {validated}.\n\n"
        f"Hi {first}! What is your vehicle's licence plate?\n"
        "(e.g. ABC-1234 or XY9900)"
    )


def get_state(session_id: str) -> ReservationState:
    return _sessions.get(session_id, _SessionData()).state


def reset(session_id: str) -> None:
    _sessions[session_id] = _SessionData(state=ReservationState.IDLE)


def advance(session_id: str, user_input: str, db: Session) -> ReplyResult:
    """Process *user_input* for *session_id* and return the next reply."""
    data = _sessions.get(session_id)
    if data is None or data.state == ReservationState.IDLE:
        start(session_id)
        data = _sessions[session_id]
        text = user_input.strip()
        if not text:
            return ReplyResult(
                message="I'd be happy to help! Could I get your full name please?",
                state=ReservationState.ASK_NAME,
            )
        return _handle_ask_name(session_id, data, text, db)

    if data.state == ReservationState.DONE:
        return ReplyResult(
            message=(
                f"Your booking CPK-{data.reservation_id:06d} is already confirmed!\n"
                "Type 'cancel' or start a new chat if you need a different reservation."
            ),
            state=ReservationState.DONE,
            reservation_id=data.reservation_id,
        )

    text = user_input.strip()
    if _is_cancel(text):
        return _do_cancel(session_id)

    handler = _HANDLERS.get(data.state)
    if handler is None:
        return ReplyResult(
            message="Something went wrong. Type 'cancel' to start over.",
            state=data.state,
        )
    return handler(session_id, data, text, db)
