from datetime import datetime

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, String, Time,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class ParkingLot(Base):
    __tablename__ = "parking_lots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    address = Column(String(200), nullable=False)
    total_spaces = Column(Integer, nullable=False)
    available_spaces = Column(Integer, nullable=False)
    hourly_price = Column(Float, nullable=False)
    daily_price = Column(Float, nullable=False)
    opens_at = Column(Time, nullable=False)
    closes_at = Column(Time, nullable=False)

    lot_reservations = relationship("LotReservation", back_populates="lot")


class LotReservation(Base):
    __tablename__ = "lot_reservations"

    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(100), nullable=False)
    car_plate = Column(String(20), nullable=False)
    lot_id = Column(Integer, ForeignKey("parking_lots.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String(20), nullable=False, default="confirmed")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    lot = relationship("ParkingLot", back_populates="lot_reservations")


class ParkingSpace(Base):
    __tablename__ = "parking_spaces"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, nullable=False)
    zone = Column(String(50), nullable=False)
    is_available = Column(Boolean, default=True, nullable=False)
    vehicle_type = Column(String(20), default="car", nullable=False)

    reservations = relationship("Reservation", back_populates="space")


class WorkingHours(Base):
    __tablename__ = "working_hours"

    id = Column(Integer, primary_key=True)
    day_of_week = Column(Integer, nullable=False)   # 0 = Monday … 6 = Sunday
    open_time = Column(Time, nullable=False)
    close_time = Column(Time, nullable=False)


class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True)
    zone = Column(String(50), nullable=False)
    vehicle_type = Column(String(20), default="car", nullable=False)
    rate_per_hour = Column(Float, nullable=False)
    currency = Column(String(5), default="USD", nullable=False)


class Reservation(Base):
    __tablename__ = "reservations"

    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(100), nullable=False)
    car_number = Column(String(20), nullable=False)
    space_id = Column(Integer, ForeignKey("parking_spaces.id"), nullable=True)
    date = Column(Date, nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    space = relationship("ParkingSpace", back_populates="reservations")
