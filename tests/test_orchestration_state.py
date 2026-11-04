"""app/orchestration/state.py and the reservation snapshot used for checkpointing."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from app.chat import reservation as resv
from app.chat.reservation import ReservationState
from app.orchestration import state as S


def test_append_messages_keeps_only_recent_turns():
    msgs = [{"role": "user", "content": str(i)} for i in range(S._MAX_MESSAGES + 7)]
    out = S.append_messages(msgs[:5], msgs[5:])
    assert len(out) == S._MAX_MESSAGES
    assert out[-1]["content"] == str(S._MAX_MESSAGES + 6)
    assert S.append_messages(None, None) == []


def test_reservation_snapshot_roundtrip_restores_a_half_finished_booking():
    sid = f"snap-{uuid.uuid4().hex[:6]}"
    start = datetime.now() + timedelta(days=3)
    resv._sessions[sid] = resv._SessionData(
        state=ReservationState.ASK_END, name="Jane Doe", car_plate="ABC-1234",
        lot_id=1, lot_name="OrchLot", lot_hourly=8.0, lot_daily=40.0, start_dt=start,
    )
    snap = resv.export_session(sid)
    assert snap["state"] == "ASK_END" and snap["start_dt"] == start.isoformat()

    resv._sessions.pop(sid)                    # "restart": in-memory state is gone
    assert resv.get_state(sid) == ReservationState.IDLE
    assert resv.import_session(sid, snap) is True
    restored = resv._sessions[sid]
    assert (restored.name, restored.car_plate, restored.lot_id, restored.start_dt) == (
        "Jane Doe", "ABC-1234", 1, start)
    resv._sessions.pop(sid)


def test_import_never_overwrites_a_live_session_and_idle_exports_nothing():
    sid = f"live-{uuid.uuid4().hex[:6]}"
    assert resv.export_session(sid) is None
    resv._sessions[sid] = resv._SessionData(state=ReservationState.ASK_LOT, name="Live User")
    assert resv.import_session(sid, {"state": "ASK_CAR_PLATE", "name": "Stale"}) is False
    assert resv._sessions[sid].name == "Live User"
    assert resv.import_session("nobody", {"state": "NOT_A_STATE"}) is False
    resv._sessions.pop(sid)
