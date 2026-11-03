from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import LotReservation, Reservation
from app.db.session import get_db

router = APIRouter(prefix="/reservations", tags=["reservations"])

# Separate router for session-scoped endpoints
sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])


class ReservationOut(BaseModel):
    id: int
    customer_name: str
    car_number: str
    date: str
    start_time: str
    end_time: str

    model_config = {"from_attributes": True}


@router.get("/", response_model=List[ReservationOut])
def list_reservations(db: Session = Depends(get_db)) -> List[ReservationOut]:
    rows = db.query(Reservation).order_by(Reservation.created_at.desc()).all()
    return [
        ReservationOut(
            id=r.id,
            customer_name=r.customer_name,
            car_number=r.car_number,
            date=str(r.date),
            start_time=str(r.start_time),
            end_time=str(r.end_time),
        )
        for r in rows
    ]


@router.get("/{reservation_id}", response_model=ReservationOut)
def get_reservation(reservation_id: int, db: Session = Depends(get_db)) -> ReservationOut:
    r = db.query(Reservation).filter(Reservation.id == reservation_id).first()
    if r is None:
        raise HTTPException(status_code=404, detail="Reservation not found")
    return ReservationOut(
        id=r.id,
        customer_name=r.customer_name,
        car_number=r.car_number,
        date=str(r.date),
        start_time=str(r.start_time),
        end_time=str(r.end_time),
    )


# ---------------------------------------------------------------------------
# GET /reservations/{id}/status  — approval-workflow status (LotReservation)
# ---------------------------------------------------------------------------


class ReservationStatusOut(BaseModel):
    id: int
    status: str
    admin_decision_reason: Optional[str] = None
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None


@router.get("/{reservation_id}/status", response_model=ReservationStatusOut)
def get_lot_reservation_status(
    reservation_id: int,
    db: Session = Depends(get_db),
) -> ReservationStatusOut:
    """Return the current approval-workflow status of a LotReservation."""
    r = db.get(LotReservation, reservation_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"Reservation {reservation_id} not found")
    return ReservationStatusOut(
        id=r.id,
        status=r.status.value if hasattr(r.status, "value") else str(r.status),
        admin_decision_reason=r.admin_decision_reason,
        decided_at=r.decided_at.isoformat() if r.decided_at else None,
        decided_by=r.decided_by,
    )


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/notifications
# ---------------------------------------------------------------------------


class NotificationOut(BaseModel):
    reservation_id: int
    decision: str
    reason: Optional[str] = None
    message: str


class NotificationsOut(BaseModel):
    count: int
    notifications: List[NotificationOut]


def _build_user_message(r: LotReservation) -> str:
    """Build a user-facing notification message from a decided LotReservation."""
    status = r.status.value if hasattr(r.status, "value") else str(r.status)
    if status == "confirmed":
        return (
            f"Great news! Your parking reservation #{r.id} has been approved. "
            "We look forward to seeing you."
        )
    reason = r.admin_decision_reason or "no reason given"
    return (
        f"We're sorry — your parking reservation #{r.id} was not approved "
        f"({reason}). Please try booking again."
    )


@sessions_router.get(
    "/{session_id}/notifications",
    response_model=NotificationsOut,
    summary="Fetch and mark delivered pending user notifications",
)
def get_session_notifications(
    session_id: str,
    db: Session = Depends(get_db),
) -> NotificationsOut:
    """
    Return all decided reservations for *session_id* that have not yet been
    delivered to the user, and mark them as delivered (notified_user=True).
    """
    rows = (
        db.query(LotReservation)
        .filter(
            LotReservation.session_id == session_id,
            LotReservation.decided_at.isnot(None),
            LotReservation.notified_user == False,  # noqa: E712
        )
        .all()
    )

    notifications: list[NotificationOut] = []
    for r in rows:
        status = r.status.value if hasattr(r.status, "value") else str(r.status)
        notifications.append(NotificationOut(
            reservation_id=r.id,
            decision=status,
            reason=r.admin_decision_reason,
            message=_build_user_message(r),
        ))
        r.notified_user = True

    if rows:
        db.commit()

    return NotificationsOut(count=len(notifications), notifications=notifications)
