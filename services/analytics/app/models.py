"""SQLAlchemy models for the analytics microservice."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

    settings = relationship("UserSettings", back_populates="user", uselist=False)
    trades = relationship("Trade", back_populates="user")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    locale = Column(String, default="es-ES")
    theme = Column(String, default="pastel")
    max_risk = Column(Float, default=0.05)
    notifications_enabled = Column(Boolean, default=True)

    user = relationship("User", back_populates="settings")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    pnl = Column(Float, default=0.0)
    open_time = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime)

    user = relationship("User", back_populates="trades")
