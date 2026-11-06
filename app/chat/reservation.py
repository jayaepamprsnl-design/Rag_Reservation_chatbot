"""Reservation state machine — one session per session_id, in-process memory.

State flow
----------
IDLE → ASK_NAME → ASK_CAR_PLATE → ASK_LOT → ASK_START → ASK_END → CONFIRM
     → AWAITING_APPROVAL  (admin must decide)
     → DONE  (approved)  |  IDLE  (refused / expired / user cancels)

Design notes
------------
* At every active step the user may type "cancel" / "quit" to abort.
* ASK_START accepts a full range ("tomorrow 10am to 2pm") which skips ASK_END.
* Availability and conflict checks happen before the CONFIRM summary is shown.
* The DB write (create_pending_reservation) happens on "confirm"; spaces are
  only decremented when the admin approves (approve_reservation).

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
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import dateparser
from sqlalchemy.orm import Session

from app.core.logging_config import logger
from app.db.models import LotReservation, ParkingLot, ReservationStatus
from app.db.repository import (
    ValidationError,
    check_conflicts,
    create_pending_reservation,
    get_availability,
    get_lot,
    get_reservations_by_session,
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ReservationState(str, Enum):
    IDLE              = "IDLE"
    ASK_NAME          = "ASK_NAME"
    ASK_CAR_PLATE     = "ASK_CAR_PLATE"
    ASK_LOT           = "ASK_LOT"
    ASK_START         = "ASK_START"
    ASK_END           = "ASK_END"
    CONFIRM           = "CONFIRM"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE              = "DONE"


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

# States where the user is actively filling a booking field.
# Imported in chat.py but also needed here for multi-slot extraction.
_COLLECTING_STATES = frozenset({
    ReservationState.ASK_NAME,
    ReservationState.ASK_CAR_PLATE,
    ReservationState.ASK_LOT,
    ReservationState.ASK_START,
    ReservationState.ASK_END,
})

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
        return next((lt for lt in lots if lt.id == int(t)), None)
    for lot in lots:
        if t in lot.name.lower():
            return lot
    for keywords, fragment in [
        (["central", "downtown"], "central"),
        (["eastside", "east"],    "eastside"),
        (["airport", "connector", "transit"], "airport"),
    ]:
        if any(kw in t for kw in keywords):
            return next((lt for lt in lots if fragment in lt.name.lower()), None)
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
    # If lot/times were pre-filled by slot extraction, skip ahead.
    next_st = _next_missing_state(data)
    if next_st is None:
        lot = db.get(ParkingLot, data.lot_id)
        return _check_and_confirm(session_id, data, lot, db)
    if next_st not in (ReservationState.ASK_LOT, ReservationState.ASK_CAR_PLATE):
        data.state = next_st
        return _prompt_for_next_field(data, next_st, db)
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

    # If start/end were pre-filled by multi-slot extraction, skip ahead.
    if data.start_dt is not None and data.end_dt is not None:
        return _check_and_confirm(session_id, data, lot, db)
    if data.start_dt is not None:
        data.state = ReservationState.ASK_END
        return ReplyResult(
            message=(
                f"Got it — {lot.name}. Arrival set to {_fmt_dt(data.start_dt)}.\n\n"
                "When would you like to depart?\n"
                "(e.g. '14:00', '2pm', or a full date-time)"
            ),
            state=ReservationState.ASK_END,
        )

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
            f"  •  'tomorrow 10am to 2pm'       \n"
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

    # Create the reservation in PENDING state — spaces not decremented yet.
    try:
        res = create_pending_reservation(
            db=db,
            lot_id=data.lot_id,
            customer_name=data.name,
            car_plate=data.car_plate,
            start_time=data.start_dt,
            end_time=data.end_dt,
            session_id=session_id,
        )
    except ValidationError as exc:
        lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
        data.lot_id = None
        data.lot_name = ""
        data.start_dt = None
        data.end_dt = None
        data.state = ReservationState.ASK_LOT
        return ReplyResult(
            message=(
                f"Could not create your request: {exc}\n"
                "Please choose a facility to try again:\n\n" + _lot_menu(lots)
            ),
            state=ReservationState.ASK_LOT,
        )

    data.reservation_id = res.id
    data.state = ReservationState.AWAITING_APPROVAL
    logger.info("Reservation pending: id=%d session=%s", res.id, session_id)

    # Escalate to admin through all configured notification channels.
    channel_ok = _escalate_to_admin(res, data, session_id, db)

    if channel_ok:
        msg = (
            f"Your parking request has been sent to the administrator!\n\n"
            f"  Booking reference : CPK-{res.id:06d}\n"
            f"  Facility          : {data.lot_name}\n"
            f"  Arrives           : {_fmt_dt(data.start_dt)}\n"
            f"  Departs           : {_fmt_dt(data.end_dt)}\n"
            f"  Status            : awaiting admin approval\n\n"
            "I'll notify you as soon as a decision is made. "
            "You can ask \"what's the status?\" at any time."
        )
    else:
        msg = (
            f"Your parking request has been queued "
            f"(reference CPK-{res.id:06d}), but our notification system is "
            "temporarily unavailable. An administrator will review it shortly. "
            "You can check the status by asking \"what's my booking status?\"."
        )

    return ReplyResult(
        message=msg,
        state=ReservationState.AWAITING_APPROVAL,
        reservation_id=res.id,
    )


# ---------------------------------------------------------------------------
# Helpers for AWAITING_APPROVAL
# ---------------------------------------------------------------------------

def _escalate_to_admin(
    res: LotReservation, data: _SessionData, session_id: str, db: Session
) -> bool:
    """Send escalation through all configured channels. Returns True if ≥1 succeeded."""
    try:
        from app.agents.contracts import AvailabilitySnapshot, ReservationRequest
        from app.agents.channels.registry import build_registry

        lot = get_lot(db, res.lot_id)
        hours = (res.end_time - res.start_time).total_seconds() / 3600

        request = ReservationRequest(
            reservation_id=res.id,
            customer_name=res.customer_name,
            car_plate_masked=_mask_plate(res.car_plate),
            lot_name=lot.name,
            start_time=res.start_time,
            end_time=res.end_time,
            price_estimate=f"~${hours * lot.hourly_price:.2f}",
            availability_snapshot=AvailabilitySnapshot(
                total_spaces=lot.total_spaces,
                available_spaces=lot.available_spaces,
                is_open=lot.available_spaces > 0,
            ),
            session_id=session_id,
        )

        results = build_registry().send_all(request, db)
        return any(r.success for r in results)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "reservation|escalation_failed session=%s error_type=%s",
            session_id, type(exc).__name__,
        )
        return False


def _deliver_decision_if_ready(
    session_id: str, data: _SessionData, res: LotReservation, db: Session
) -> Optional[ReplyResult]:
    """
    If the reservation has been decided and not yet notified, deliver the
    notification and update in-memory state.  Returns None if still PENDING.
    """
    from app.agents.bus import get_bus

    if res.status == ReservationStatus.PENDING:
        return None

    # Already delivered via another channel (e.g. /sessions/.../notifications)
    if res.notified_user:
        if res.status == ReservationStatus.CONFIRMED:
            data.state = ReservationState.DONE
        else:
            reset(session_id)
        return None

    bus = get_bus()

    if res.status == ReservationStatus.CONFIRMED:
        data.state = ReservationState.DONE
        bus.mark_notified(res.id, db)
        lot_name = data.lot_name or f"Facility #{res.lot_id}"
        return ReplyResult(
            message=(
                f"Great news! Your parking request has been APPROVED!\n\n"
                f"  Booking reference : CPK-{res.id:06d}\n"
                f"  Facility          : {lot_name}\n"
                f"  Arrives           : {_fmt_dt(res.start_time)}\n"
                f"  Departs           : {_fmt_dt(res.end_time)}\n\n"
                "We look forward to seeing you. Have a great visit!"
            ),
            state=ReservationState.DONE,
            reservation_id=res.id,
        )

    if res.status == ReservationStatus.REFUSED:
        reason = res.admin_decision_reason or "no reason provided"
        bus.mark_notified(res.id, db)
        reset(session_id)
        return ReplyResult(
            message=(
                f"We're sorry — your parking request CPK-{res.id:06d} "
                f"was not approved.\n\n"
                f"  Reason : {reason}\n\n"
                "Would you like to book at a different time or facility? "
                "Just say \"I'd like to make a reservation\" to start again."
            ),
            state=ReservationState.IDLE,
            cancelled=True,
        )

    if res.status == ReservationStatus.EXPIRED:
        bus.mark_notified(res.id, db)
        reset(session_id)
        return ReplyResult(
            message=(
                f"Your parking request CPK-{res.id:06d} has expired "
                "(no admin decision was received within the required timeframe).\n\n"
                "Would you like to make a new reservation?"
            ),
            state=ReservationState.IDLE,
            cancelled=True,
        )

    if res.status == ReservationStatus.CANCELLED:
        reset(session_id)
        return ReplyResult(
            message="Your parking request has been cancelled.",
            state=ReservationState.IDLE,
            cancelled=True,
        )

    return None


def _answer_status_query(
    data: _SessionData, res_id: Optional[int], db: Session
) -> ReplyResult:
    if not res_id:
        return ReplyResult(
            message="I don't have an active reservation request for your session.",
            state=ReservationState.AWAITING_APPROVAL,
        )
    res = db.get(LotReservation, res_id)
    if res is None:
        return ReplyResult(
            message=f"Reservation CPK-{res_id:06d} could not be found.",
            state=ReservationState.IDLE,
        )
    status_val = res.status.value if hasattr(res.status, "value") else str(res.status)
    reason_line = (
        f"\n  Reason   : {res.admin_decision_reason}"
        if res.admin_decision_reason else ""
    )
    return ReplyResult(
        message=(
            f"Reservation CPK-{res_id:06d} is currently: {status_val.upper()}"
            f"{reason_line}\n\n"
            "The administrator is reviewing your request. I'll notify you here "
            "as soon as a decision is made."
        ),
        state=ReservationState.AWAITING_APPROVAL,
        reservation_id=res_id,
    )


def _cancel_pending_request(
    session_id: str, data: _SessionData, res_id: Optional[int], db: Session
) -> ReplyResult:
    if res_id:
        res = db.get(LotReservation, res_id)
        if res and res.status == ReservationStatus.PENDING:
            res.status = ReservationStatus.CANCELLED
            res.admin_decision_reason = "Cancelled by customer"
            res.decided_at = datetime.utcnow()
            db.commit()
            logger.info("Pending reservation cancelled by customer: id=%d", res_id)
    reset(session_id)
    return ReplyResult(
        message=(
            "Your parking request has been cancelled. "
            "No problem — you can make a new reservation whenever you're ready!"
        ),
        state=ReservationState.IDLE,
        cancelled=True,
    )


def _handle_awaiting_approval(
    session_id: str, data: _SessionData, text: str, db: Session
) -> ReplyResult:
    """
    Handle user messages while the reservation awaits an admin decision.

    Checks the DB first so it handles both the live case and server-restart
    recovery (in-memory state rebuilt from DB in advance()).
    """
    res_id = data.reservation_id

    if res_id:
        res = db.get(LotReservation, res_id)
        if res:
            decision = _deliver_decision_if_ready(session_id, data, res, db)
            if decision:
                return decision

    t = text.strip().lower()

    # User wants to cancel
    if _is_cancel(t) or "cancel" in t:
        return _cancel_pending_request(session_id, data, res_id, db)

    # Status / progress query
    if any(kw in t for kw in (
        "status", "pending", "decision", "approved", "refused",
        "waiting", "update", "booking", "reservation", "request",
    )):
        return _answer_status_query(data, res_id, db)

    # User tries to start a new reservation while one is waiting
    if any(kw in t for kw in ("reserve", "book", "new reservation", "parking slot")):
        ref = f"CPK-{res_id:06d}" if res_id else "a pending request"
        return ReplyResult(
            message=(
                f"You already have {ref} awaiting admin approval. "
                "Please wait for the decision before starting a new booking, "
                "or type \"cancel my request\" to withdraw it."
            ),
            state=ReservationState.AWAITING_APPROVAL,
            reservation_id=res_id,
        )

    # Generic hold message
    ref = f"CPK-{res_id:06d}" if res_id else "your request"
    return ReplyResult(
        message=(
            f"Your reservation {ref} is still awaiting admin approval. "
            "I'll notify you as soon as there's a decision!\n\n"
            "You can ask \"what's the status?\" or type \"cancel\" "
            "to withdraw your request."
        ),
        state=ReservationState.AWAITING_APPROVAL,
        reservation_id=res_id,
    )


# ---------------------------------------------------------------------------
# Multi-slot helpers
# ---------------------------------------------------------------------------

def _next_missing_state(data: _SessionData) -> Optional[ReservationState]:
    """Return the first state whose required field is still empty, or None."""
    if not data.name:
        return ReservationState.ASK_NAME
    if not data.car_plate:
        return ReservationState.ASK_CAR_PLATE
    if data.lot_id is None:
        return ReservationState.ASK_LOT
    if data.start_dt is None:
        return ReservationState.ASK_START
    if data.end_dt is None:
        return ReservationState.ASK_END
    return None


def _prompt_for_next_field(
    data: _SessionData, state: ReservationState, db: Session
) -> ReplyResult:
    """Return the question prompt for *state* without consuming user input."""
    if state == ReservationState.ASK_NAME:
        return ReplyResult(
            message="What is your full name?",
            state=ReservationState.ASK_NAME,
        )
    if state == ReservationState.ASK_CAR_PLATE:
        first = data.name.split()[0] if data.name else "there"
        return ReplyResult(
            message=(
                f"Nice to meet you, {first}! "
                "What is your vehicle's licence plate?\n"
                "(e.g. ABC-1234 or XY9900)"
            ),
            state=ReservationState.ASK_CAR_PLATE,
        )
    if state == ReservationState.ASK_LOT:
        lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
        return ReplyResult(message=_lot_menu(lots), state=ReservationState.ASK_LOT)
    if state == ReservationState.ASK_START:
        lot = db.get(ParkingLot, data.lot_id)
        tomorrow = _tomorrow_str()
        return ReplyResult(
            message=(
                f"Got it — {lot.name} "
                f"(open {lot.opens_at.strftime('%H:%M')}"
                f"–{lot.closes_at.strftime('%H:%M')}).\n\n"
                "When would you like to arrive?\n\n"
                f"  •  '{tomorrow} 10:00'\n"
                f"  •  'tomorrow 9am'\n"
                f"  •  'tomorrow 10am to 2pm'  (arrival + departure together)"
            ),
            state=ReservationState.ASK_START,
        )
    if state == ReservationState.ASK_END:
        return ReplyResult(
            message=(
                f"Arrival set to {_fmt_dt(data.start_dt)}.\n\n"
                "When would you like to depart?\n"
                "(e.g. '14:00', '2pm', or a full date-time)"
            ),
            state=ReservationState.ASK_END,
        )
    return ReplyResult(
        message="Something went wrong. Type 'cancel' to start over.",
        state=state,
    )


def _prefill_from_slots(
    data: _SessionData, raw_text: str, db: Session
) -> bool:
    """
    Run the slot extractor on *raw_text* and fill any empty fields in *data*.

    Uses existing validators so invalid values are silently dropped.
    Returns True if at least one new field was filled.
    """
    from app.chat.slot_extractor import extract_slots

    slots = extract_slots(raw_text)
    filled = False
    now = datetime.now()

    if slots.customer_name and not data.name:
        name, err = _validate_name(slots.customer_name)
        if not err:
            data.name = name
            filled = True

    if slots.car_plate and not data.car_plate:
        plate, err = _validate_plate(slots.car_plate)
        if not err:
            data.car_plate = plate
            filled = True

    if slots.lot_hint and data.lot_id is None:
        lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
        lot = _resolve_lot(slots.lot_hint, lots)
        if lot:
            data.lot_id    = lot.id
            data.lot_name  = lot.name
            data.lot_hourly = lot.hourly_price
            data.lot_daily  = lot.daily_price
            filled = True

    if slots.start_time and data.start_dt is None:
        if slots.start_time > now + timedelta(minutes=_MIN_ADVANCE_MINUTES):
            # If the lot is already known, also check opening hours.
            if data.lot_id is not None:
                lot = db.get(ParkingLot, data.lot_id)
                if lot and (lot.opens_at <= slots.start_time.time() <= lot.closes_at):
                    data.start_dt = slots.start_time
                    filled = True
                # else: outside opening hours — let the normal handler surface the error
            else:
                data.start_dt = slots.start_time
                filled = True

    if slots.end_time and data.end_dt is None and data.start_dt is not None:
        min_end = data.start_dt + timedelta(minutes=_MIN_DURATION_MINUTES)
        if slots.end_time > min_end:
            data.end_dt = slots.end_time
            filled = True

    return filled


def _is_name_answer(text: str) -> bool:
    """
    Return True when text looks like a name answer rather than a booking command.

    Prevents storing "I want to book a parking space" as the customer's name.
    """
    words = text.split()
    if len(words) > 5:
        return False
    _cmd_words = {
        "i", "want", "book", "reserve", "parking", "space", "slot",
        "a", "an", "the", "to", "need", "help", "please", "hi", "hello",
    }
    if sum(1 for w in words if w.lower() in _cmd_words) >= 2:
        return False
    return True


def _apply_slots_and_advance(
    session_id: str, data: _SessionData, raw_text: str, db: Session
) -> Optional[ReplyResult]:
    """
    Try slot extraction; if anything was filled, skip ahead to the next
    missing field and return its prompt.  Returns None when nothing was
    extracted (caller should fall through to the normal per-state handler).
    """
    if not _prefill_from_slots(data, raw_text, db):
        return None

    next_state = _next_missing_state(data)
    if next_state is None:
        lot = db.get(ParkingLot, data.lot_id)
        return _check_and_confirm(session_id, data, lot, db)

    data.state = next_state
    return _prompt_for_next_field(data, next_state, db)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    ReservationState.ASK_NAME:          _handle_ask_name,
    ReservationState.ASK_CAR_PLATE:     _handle_ask_plate,
    ReservationState.ASK_LOT:           _handle_ask_lot,
    ReservationState.ASK_START:         _handle_ask_start,
    ReservationState.ASK_END:           _handle_ask_end,
    ReservationState.CONFIRM:           _handle_confirm,
    ReservationState.AWAITING_APPROVAL: _handle_awaiting_approval,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start(session_id: str) -> str:
    """Create (or reset) a session and return the opening question.

    If a reservation is already AWAITING_APPROVAL for this session, returns
    a reminder instead of resetting — protecting the pending request.
    """
    existing = _sessions.get(session_id)
    if existing and existing.state == ReservationState.AWAITING_APPROVAL:
        ref = f"CPK-{existing.reservation_id:06d}" if existing.reservation_id else "your request"
        return (
            f"You already have {ref} waiting for admin approval. "
            "I'll let you know the decision. "
            "Type \"what's the status?\" to check, or \"cancel\" to withdraw."
        )
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
    """Return the current ReservationState for *session_id* (IDLE if unknown)."""
    return _sessions.get(session_id, _SessionData()).state


def reset(session_id: str) -> None:
    """Reset *session_id* to the IDLE state, discarding all in-memory data."""
    _sessions[session_id] = _SessionData(state=ReservationState.IDLE)


def _recover_session_from_db(session_id: str, db: Session) -> Optional[_SessionData]:
    """
    After a server restart the in-memory _sessions dict is empty.
    Check the DB for PENDING reservations or undelivered decisions for this
    session and, if found, restore the session to AWAITING_APPROVAL.
    """
    rows = get_reservations_by_session(db, session_id)
    candidates = [
        r for r in rows
        if r.status == ReservationStatus.PENDING
        or (r.decided_at is not None and not r.notified_user
            and r.status in (
                ReservationStatus.CONFIRMED,
                ReservationStatus.REFUSED,
                ReservationStatus.EXPIRED,
            ))
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda r: r.created_at)
    recovered = _SessionData(state=ReservationState.AWAITING_APPROVAL)
    recovered.reservation_id = latest.id
    logger.info(
        "session_recovered_from_db session=%s reservation_id=%d",
        session_id, latest.id,
    )
    return recovered


def advance(session_id: str, user_input: str, db: Session) -> ReplyResult:
    """Process *user_input* for *session_id* and return the next reply."""
    data = _sessions.get(session_id)

    # Server-restart recovery: if no in-memory state, check DB for anything
    # that needs to be tracked (pending approval or undelivered decision).
    if data is None or data.state == ReservationState.IDLE:
        recovered = _recover_session_from_db(session_id, db)
        if recovered:
            data = recovered
            _sessions[session_id] = data

    if data is None or data.state == ReservationState.IDLE:
        start(session_id)
        data = _sessions[session_id]
        text = user_input.strip()
        if not text:
            return ReplyResult(
                message="I'd be happy to help! Could I get your full name please?",
                state=ReservationState.ASK_NAME,
            )
        # Apply multi-slot extraction to the very first booking message so
        # information given upfront (name, date, time, lot) is not wasted.
        result = _apply_slots_and_advance(session_id, data, text, db)
        if result is not None:
            return result
        # Nothing extracted from the text.  Only pass it to _handle_ask_name if
        # it looks like a name answer (short, no booking command words).
        # If the text is a booking sentence like "I want to book a space", show
        # the greeting so the user knows to provide their name.
        if _is_name_answer(text):
            return _handle_ask_name(session_id, data, text, db)
        return ReplyResult(
            message=(
                "I'd be happy to help you make a reservation!\n"
                "First, could I get your full name please?"
            ),
            state=ReservationState.ASK_NAME,
        )

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

    # AWAITING_APPROVAL must reach its own handler first so it can cancel
    # the PENDING DB row.  The global _do_cancel would reset in-memory state
    # without touching the DB, leaving an orphaned PENDING reservation.
    handler = _HANDLERS.get(data.state)

    if _is_cancel(text) and data.state != ReservationState.AWAITING_APPROVAL:
        return _do_cancel(session_id)

    # Multi-slot extraction for all collecting states: if the user volunteers
    # multiple pieces of information at once, fill them and skip ahead.
    if data.state in _COLLECTING_STATES:
        result = _apply_slots_and_advance(session_id, data, text, db)
        if result is not None:
            return result

    if handler is None:
        return ReplyResult(
            message="Something went wrong. Type 'cancel' to start over.",
            state=data.state,
        )
    return handler(session_id, data, text, db)
