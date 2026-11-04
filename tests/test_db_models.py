from datetime import date, time

from app.db.models import ParkingSpace, Price, Reservation, WorkingHours


def test_parking_space_persists(db_session):
    space = ParkingSpace(code="T01", zone="A", is_available=True, vehicle_type="car")
    db_session.add(space)
    db_session.flush()
    assert space.id is not None
    assert space.code == "T01"


def test_reservation_links_to_space(db_session):
    space = ParkingSpace(code="T02", zone="B", is_available=True)
    db_session.add(space)
    db_session.flush()

    r = Reservation(
        customer_name="John Doe",
        car_number="ABC-123",
        space_id=space.id,
        date=date(2026, 1, 15),
        start_time=time(9, 0),
        end_time=time(11, 0),
    )
    db_session.add(r)
    db_session.flush()
    assert r.id is not None
    assert r.space_id == space.id


def test_working_hours_persists(db_session):
    wh = WorkingHours(day_of_week=0, open_time=time(6, 0), close_time=time(22, 0))
    db_session.add(wh)
    db_session.flush()
    assert wh.id is not None


def test_price_persists(db_session):
    p = Price(zone="A", vehicle_type="car", rate_per_hour=5.0, currency="USD")
    db_session.add(p)
    db_session.flush()
    assert p.id is not None
    assert p.rate_per_hour == 5.0


def test_reservation_without_space(db_session):
    r = Reservation(
        customer_name="Alice",
        car_number="ZZZ-000",
        date=date(2026, 3, 1),
        start_time=time(10, 0),
        end_time=time(12, 0),
    )
    db_session.add(r)
    db_session.flush()
    assert r.space_id is None
