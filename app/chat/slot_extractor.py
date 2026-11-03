"""
Multi-slot extractor for the reservation booking flow.

extract_slots(text, today=None) -> Slots

Extracts whatever is clearly present in a free-form user message.
Fields not found stay None — never invents values.

Supported patterns
------------------
customer_name : "my name is X", "I am X", "I'm X", "called X"
car_plate     : ABC-1234, AB1234, XY9900 (alphanumeric, 3-10 chars)
lot_hint      : "central", "eastside", "airport", "lot 1", "lot 2", "lot 3"
start_time    : "4th nov 2026", "tomorrow", "from 2pm", "14:00",
                "2026-11-04 14:00"  (dateparser, PREFER_DATES_FROM=future)
end_time      : "till 5pm", "until 17:00", "to 5:30pm", "by 5pm"
duration      : "for 3 hours", "3hrs", "90 minutes", "1.5h"
                → end_time = start_time + duration when start is known
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import dateparser

_DP_SETTINGS = {"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False}

_MIN_FUTURE_MINUTES = 30   # reject datetimes < 30 min in future

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Slots:
    customer_name: Optional[str] = None
    car_plate:     Optional[str] = None
    lot_hint:      Optional[str] = None
    start_time:    Optional[datetime] = None
    end_time:      Optional[datetime] = None


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Name — explicit introductions only (high-precision, stops at stop words)
_NAME_MY_NAME = re.compile(
    r"my\s+name\s+is\s+"
    r"([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+){0,2})",
    re.I,
)
_NAME_IAM = re.compile(
    r"\bI(?:'m|\s+am)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
    re.I,
)
_NAME_CALLED = re.compile(
    r"\bcalled\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,1})",
    re.I,
)
_NAME_FOR = re.compile(
    r"\b(?:for|reserve\s+for|book(?:ing)?\s+for|slot\s+for|parking\s+for)\s+"
    r"([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+){0,2})",
    re.I,
)

# Words that should stop name capture or cause rejection
_NAME_STOP = frozenset(
    "i a an the my your his her this that want would like to book reserve park "
    "and or but so then tomorrow today monday tuesday wednesday thursday friday "
    "saturday sunday jan feb mar apr may jun jul aug sep oct nov dec "
    "january february march april june july august september october november december "
    "central eastside airport spot space slot parking lot one two three four five".split()
)

# Plate — conservative: avoid matching dates like "2026" or times like "2pm"
_PLATE_BODY = re.compile(
    r"\b([A-Z]{2,3}[-\s]?[0-9]{3,4}|[A-Z]{1,3}[0-9]{2,4}[A-Z]{0,3})\b",
    re.I,
)
_PLATE_VALID = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,8}[A-Z0-9]$", re.I)
# Known date/time-like strings to avoid false plate matches
_DATE_LIKE = re.compile(r"^\d{4}$|^\d{1,2}[ap]m$|^\d{1,2}:\d{2}$", re.I)

# Words that can never be the letter-part of a plate number
_PLATE_REJECT_WORDS = frozenset(
    # Month names
    "jan feb mar apr may jun jul aug sep oct nov dec "
    "january february march april june july august september october november december "
    # Common prepositions / particles that appear before years: "on 2026", "by 2026"
    "on by at in to of is it as or up go no so".split()
)
# Still keep _MONTHS as alias for backward compat within this module
_MONTHS = _PLATE_REJECT_WORDS

# Lot keywords → canonical name used by _resolve_lot
_LOT_KEYWORDS = {
    "central": "central", "downtown": "central",
    "eastside": "eastside", "east": "eastside",
    "airport": "airport", "transit": "airport", "connector": "airport",
}
_LOT_ID_RE = re.compile(r"\blot\s*([123])\b", re.I)

# Duration: "for 3 hours", "3hrs", "90 minutes", "1.5h"
_DUR_RE = re.compile(
    r"(?:\bfor\s+)?(\d+(?:\.\d+)?)\s*(hours?|hrs?|h(?=\s|$)|minutes?|mins?(?=\s|$))",
    re.I,
)

# End-time anchor — "till/until/by/..." followed by a time (am/pm OR 24h colon)
_END_TIME_RE = re.compile(
    r"\b(?:till|until|ending\s+at|by|depart(?:s|ure)?\s+at)\s+"
    r"(\d{1,2}:\d{2}|\d{1,2}(?::\d{2})?\s*[ap]m)",
    re.I,
)
# "to X" end-time: ISO datetime, 24h time, or am/pm time
# Negative lookbehind prevents matching "to book", "to park", "to reserve"
_END_TO_RE = re.compile(
    r"\bto\s+"
    r"(\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\s+\d{1,2}:\d{2}"   # ISO: 2026-11-04 14:00
    r"|\d{1,2}:\d{2}(?!\d)"                                 # 24h: 14:00
    r"|\d{1,2}\s*[ap]m)",                                   # am/pm: 2pm
    re.I,
)

# Start-time anchor: "from 2pm", "arriving at 10am", "at 14:00"
_START_ANCHOR = re.compile(
    r"\b(?:from|arriving\s+at|starting\s+at)\s+"
    r"(\d{1,2}(?::\d{2})?\s*(?:[ap]m)?)",
    re.I,
)

# Patterns for explicit date extraction (used to avoid passing noisy text to dateparser)
_ISO_DATE_RE = re.compile(r"\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b")
_ORDINAL_DATE_RE = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"(?:\s+\d{4})?)\b",
    re.I,
)
_RELATIVE_DATE_RE = re.compile(r"\b(tomorrow|today|next\s+\w+)\b", re.I)
# Bare time: "14:00" or "2pm"
_BARE_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}|\d{1,2}\s*[ap]m)\b", re.I)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(text: str, reference: Optional[datetime] = None) -> Optional[datetime]:
    settings = dict(_DP_SETTINGS)
    if reference:
        settings["RELATIVE_BASE"] = reference
    try:
        return dateparser.parse(text, settings=settings)
    except Exception:
        return None


def _extract_name(text: str) -> Optional[str]:
    """Return a name if clearly stated, else None."""
    for pattern in (_NAME_MY_NAME, _NAME_IAM, _NAME_CALLED):
        m = pattern.search(text)
        if m:
            words = m.group(1).split()
            kept = []
            for w in words:
                if w.lower() in _NAME_STOP:
                    break
                kept.append(w)
            candidate = " ".join(kept)
            if len(candidate) >= 2:
                return candidate

    # "for X" — lowest priority, only if the candidate looks like a name
    m = _NAME_FOR.search(text)
    if m:
        words = m.group(1).split()
        kept = []
        for w in words:
            if w.lower() in _NAME_STOP:
                break
            kept.append(w)
        candidate = " ".join(kept)
        if len(candidate) >= 2 and not any(kw in candidate.lower() for kw in _LOT_KEYWORDS):
            return candidate
    return None


def _extract_plate(text: str) -> Optional[str]:
    for m in _PLATE_BODY.finditer(text):
        raw = m.group(0).strip()
        candidate = raw.upper().replace(" ", "-")
        if _DATE_LIKE.match(raw):
            continue
        if len(candidate) < 4 or len(candidate) > 10:
            continue
        # Reject if the letter part is a common word (month, preposition, etc.)
        alpha_part = re.sub(r"[0-9\-]", "", candidate).lower()
        if alpha_part in _PLATE_REJECT_WORDS:
            continue
        # Reject if the numeric part is a 4-digit year (2000-2099)
        num_part = re.sub(r"[A-Za-z\-]", "", candidate)
        if re.match(r"^20\d{2}$", num_part):
            continue
        if _PLATE_VALID.match(candidate):
            return candidate
    return None


def _extract_lot_hint(text: str) -> Optional[str]:
    t = text.lower()
    for kw, canonical in _LOT_KEYWORDS.items():
        # Use word-boundary check
        if re.search(r"\b" + kw + r"\b", t):
            return canonical
    m = _LOT_ID_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_duration_minutes(text: str) -> Optional[int]:
    m = _DUR_RE.search(text)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("h"):
        return max(1, int(value * 60))
    return max(1, int(value))


def _strip_name(text: str, name: Optional[str]) -> str:
    """Remove name-related phrases from text before date parsing."""
    if not name:
        return text
    cleaned = text
    # Remove "my name is X", "I am X", "I'm X", "called X"
    cleaned = re.sub(r"my\s+name\s+is\s+" + re.escape(name), " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\bI(?:'m|\s+am)\s+" + re.escape(name), " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\bcalled\s+" + re.escape(name), " ", cleaned, flags=re.I)
    # Also remove the bare name if it appears early in text
    cleaned = re.sub(r"\b" + re.escape(name) + r"\b", " ", cleaned, flags=re.I)
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_slots(text: str, today: Optional[datetime] = None) -> Slots:
    """
    Extract booking slots from a free-form user message.

    Returns a Slots dataclass; any field not found stays None.
    Never raises — unknown/ambiguous text simply yields empty slots.
    """
    now = today or datetime.now()
    result = Slots()

    # ── Name ──────────────────────────────────────────────────────────────────
    result.customer_name = _extract_name(text)

    # ── Plate ─────────────────────────────────────────────────────────────────
    result.car_plate = _extract_plate(text)

    # ── Lot hint ──────────────────────────────────────────────────────────────
    result.lot_hint = _extract_lot_hint(text)

    # ── Duration ──────────────────────────────────────────────────────────────
    duration_mins = _extract_duration_minutes(text)

    # ── End time anchor ───────────────────────────────────────────────────────
    # Search original text for end_raw, then remove anchor from work via regex
    # substitution (not positional slice) so name-stripping doesn't shift offsets.
    end_raw: Optional[str] = None
    m_end_orig = _END_TIME_RE.search(text) or _END_TO_RE.search(text)
    if m_end_orig:
        end_raw = m_end_orig.group(1).strip()

    # ── Build working text for start parsing ─────────────────────────────────
    work = _strip_name(text, result.customer_name)
    if end_raw:
        work = _END_TIME_RE.sub(" ", work)
        work = _END_TO_RE.sub(" ", work)
    if duration_mins is not None:
        work = _DUR_RE.sub(" ", work)
    # Remove stray "till/until" that may remain after duration removal
    work = re.sub(r"\b(?:till|until)\s*$", " ", work.strip(), flags=re.I)

    # ── Start time ────────────────────────────────────────────────────────────
    start_dt: Optional[datetime] = None

    # Guard: a very short working text (e.g. just "1" or "ok") is almost
    # certainly not a date expression — skip parsing to avoid dateparser
    # interpreting single digits as day-of-month ordinals.
    _work_clean = work.strip()
    if len(_work_clean) < 5:
        pass   # leave start_dt as None
    else:
        # Extract date and time components separately, then combine.
        # Passing noisy text (like "plate AB-1234, ...") directly to dateparser
        # often returns None; clean "DATE TIME" pairs parse reliably.
        _iso_m   = _ISO_DATE_RE.search(_work_clean)
        _ord_m   = _ORDINAL_DATE_RE.search(_work_clean)
        _rel_m   = _RELATIVE_DATE_RE.search(_work_clean)
        _anc_m   = _START_ANCHOR.search(_work_clean)
        _time_m  = _BARE_TIME_RE.search(_work_clean)

        # Pick best date string
        date_str = None
        if _iso_m:
            date_str = _iso_m.group(1)
        elif _ord_m:
            date_str = _ord_m.group(1)
        elif _rel_m:
            date_str = _rel_m.group(1)

        # Pick best time string (anchor preferred over bare time)
        time_str = None
        if _anc_m:
            time_str = _anc_m.group(1).strip()
        elif _time_m:
            time_str = _time_m.group(1).strip()

        def _try(s: str) -> bool:
            nonlocal start_dt
            dt = _parse_dt(s, reference=now)
            if dt and dt > now + timedelta(minutes=_MIN_FUTURE_MINUTES):
                start_dt = dt
                return True
            return False

        if date_str and time_str:
            _try(f"{date_str} {time_str}")
        if start_dt is None and date_str:
            _try(date_str)
        if start_dt is None and time_str:
            _try(time_str)
        # Last resort: full working text (may work for simple inputs)
        if start_dt is None:
            _try(_work_clean)

    # Validate year is within a sensible booking window.
    # Rejects:
    #  • past dates ("1st jan 2020" → year < now.year)
    #  • implausibly far-future spurious parses ("OP-2222" → year 2222)
    #  • text-explicit past year that dateparser overrode ("1st jan 2020 from 10am"
    #    → dateparser returns tomorrow 10am, ignoring "2020")
    if start_dt is not None:
        yr = start_dt.year
        if yr < now.year:
            start_dt = None  # extracted year is past
        elif yr > now.year + 5:
            start_dt = None  # implausibly far future (spurious parse)
        elif yr == now.year and start_dt.date() < now.date():
            start_dt = None  # this year but already past date

    # Secondary check: if the *original text* contains an explicit year that is
    # past, reject even if dateparser chose a different (future) year.
    if start_dt is not None:
        _expl_yr = re.search(r"\b(20\d{2})\b", text)
        if _expl_yr and int(_expl_yr.group(1)) < now.year:
            start_dt = None

    result.start_time = start_dt

    # ── End time ──────────────────────────────────────────────────────────────
    if end_raw and start_dt:
        end_dt = _parse_dt(end_raw, reference=start_dt)
        if end_dt:
            if end_dt <= start_dt:
                # Anchor to same date as start
                bare = _parse_dt(end_raw)
                if bare:
                    end_dt = start_dt.replace(
                        hour=bare.hour, minute=bare.minute, second=0, microsecond=0
                    )
            if end_dt and end_dt > start_dt:
                result.end_time = end_dt

    # Duration wins over explicit end if end not found
    if result.end_time is None and duration_mins and start_dt:
        result.end_time = start_dt + timedelta(minutes=duration_mins)

    return result
