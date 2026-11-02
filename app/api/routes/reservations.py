from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Reservation
from app.db.session import get_db

router = APIRouter(prefix="/reservations", tags=["reservations"])


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
