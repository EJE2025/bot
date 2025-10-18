"""Authentication helpers and user model for the dashboard."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from flask_login import UserMixin
from sqlalchemy import Boolean, Column, DateTime, Integer, String, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash

from trading_bot import config
from trading_bot.db import Base, SessionLocal

logger = logging.getLogger(__name__)


class User(Base, UserMixin):
    __tablename__ = "dashboard_users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    last_login_at = Column(DateTime, nullable=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password: str) -> bool:
        if not password:
            return False
        try:
            return check_password_hash(self.password_hash, password)
        except ValueError:
            return False

    def get_id(self) -> str:
        return str(self.id)


def _get_user_by_username(session: Session, username: str) -> Optional[User]:
    if not username:
        return None
    stmt = select(User).where(User.username == username)
    return session.execute(stmt).scalar_one_or_none()


def load_user(user_id: str) -> Optional[User]:
    if not user_id:
        return None
    session = SessionLocal()
    try:
        try:
            numeric_id = int(user_id)
        except (TypeError, ValueError):
            return None
        user = session.get(User, numeric_id)
        if not user or not user.is_active:
            return None
        session.expunge(user)
        return user
    finally:
        session.close()


def authenticate(username: str, password: str) -> Optional[User]:
    session = SessionLocal()
    try:
        user = _get_user_by_username(session, username)
        if not user or not user.is_active:
            return None
        if not user.verify_password(password):
            return None
        user.last_login_at = datetime.utcnow()
        session.commit()
        session.refresh(user)
        session.expunge(user)
        return user
    except IntegrityError:
        session.rollback()
        logger.exception("Error de integridad autenticando al usuario '%s'", username)
        return None
    finally:
        session.close()


def ensure_default_admin() -> bool:
    """Create a default dashboard user if credentials are provided via config."""

    username = config.DASHBOARD_ADMIN_USERNAME.strip()
    password = config.DASHBOARD_ADMIN_PASSWORD
    password_hash = config.DASHBOARD_ADMIN_PASSWORD_HASH

    if not username or (not password and not password_hash):
        return False

    session = SessionLocal()
    created = False
    try:
        existing = _get_user_by_username(session, username)
        if existing:
            return False

        user = User(username=username, is_active=True)
        if password_hash:
            user.password_hash = password_hash
        else:
            user.set_password(password)

        session.add(user)
        session.commit()
        created = True
        logger.info("Usuario administrador predeterminado '%s' creado para el dashboard", username)
    except IntegrityError:
        session.rollback()
        logger.exception("No se pudo crear el usuario administrador predeterminado")
    finally:
        session.close()

    return created


def has_any_user() -> bool:
    session = SessionLocal()
    try:
        stmt = select(User.id).limit(1)
        return session.execute(stmt).first() is not None
    finally:
        session.close()


__all__ = ["User", "authenticate", "ensure_default_admin", "has_any_user", "load_user"]

