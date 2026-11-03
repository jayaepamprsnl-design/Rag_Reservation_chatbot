#!/usr/bin/env python3
"""Seed SQLite with parking lots, spaces, working hours, and prices."""
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import (
    ApprovalRequest, LotReservation, ParkingLot, Price, ParkingSpace, WorkingHours,
)
from app.db.session import SessionLocal, create_tables


def seed() -> None:
    create_tables()
    db = SessionLocal()
    try:
        # Delete child tables before parents to respect FK ordering
        for model in (ApprovalRequest, LotReservation, ParkingLot, ParkingSpace, WorkingHours, Price):
            db.query(model).delete()

        # --- Parking lots -------------------------------------------------
        lots = [
            ParkingLot(
                name="CityPark Central",
                address="123 Central Avenue, Downtown, City DC 10001",
                total_spaces=350,
                available_spaces=287,
                hourly_price=8.00,
                daily_price=45.00,
                opens_at=time(6, 0),
                closes_at=time(22, 0),
            ),
            ParkingLot(
                name="CityPark Eastside",
                address="45 Eastside Boulevard, East Business District, City DC 10024",
                total_spaces=220,
                available_spaces=194,
                hourly_price=5.00,
                daily_price=28.00,
                opens_at=time(6, 0),
                closes_at=time(22, 0),
            ),
            ParkingLot(
                name="CityPark Airport",
                address="1 Airport Connector Road, Transit Zone, City DC 10088",
                total_spaces=480,
                available_spaces=431,
                hourly_price=6.00,
                daily_price=35.00,
                opens_at=time(0, 0),
                closes_at=time(23, 59),
            ),
        ]
        db.add_all(lots)

        # --- Zone spaces (legacy model) ------------------------------------
        spaces = (
            [ParkingSpace(code=f"A{i:02d}", zone="A", is_available=True, vehicle_type="car") for i in range(1, 11)]
            + [ParkingSpace(code=f"B{i:02d}", zone="B", is_available=True, vehicle_type="car") for i in range(1, 11)]
            + [ParkingSpace(code=f"C{i:02d}", zone="C", is_available=True, vehicle_type="motorcycle") for i in range(1, 6)]
        )
        db.add_all(spaces)

        # --- Working hours (legacy model) ----------------------------------
        for day in range(7):
            db.add(WorkingHours(day_of_week=day, open_time=time(6, 0), close_time=time(22, 0)))

        # --- Prices (legacy model) -----------------------------------------
        db.add_all([
            Price(zone="A", vehicle_type="car",        rate_per_hour=8.00, currency="USD"),
            Price(zone="B", vehicle_type="car",        rate_per_hour=5.00, currency="USD"),
            Price(zone="C", vehicle_type="motorcycle", rate_per_hour=2.50, currency="USD"),
        ])

        db.commit()
        print(
            "Database seeded: 3 parking lots, 25 spaces, "
            "7 working-hour rows, 3 price rows."
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
