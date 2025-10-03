"""GraphQL schema for the analytics service."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import graphene

from ..database import SessionLocal
from ..models import Trade, UserSettings


def _serialize_trade(trade: Trade) -> Trade:
    return trade


def _serialize_settings(settings: UserSettings) -> UserSettings:
    return settings


class TradeType(graphene.ObjectType):
    id = graphene.ID(required=True)
    symbol = graphene.String()
    pnl = graphene.Float()
    side = graphene.String()
    quantity = graphene.Float()
    open_time = graphene.DateTime()
    close_time = graphene.DateTime()


class UserSettingsType(graphene.ObjectType):
    locale = graphene.String()
    theme = graphene.String()
    max_risk = graphene.Float()
    notifications_enabled = graphene.Boolean()


class Query(graphene.ObjectType):
    trades = graphene.List(
        TradeType,
        symbol=graphene.String(),
        start=graphene.DateTime(),
        end=graphene.DateTime(),
        user_id=graphene.Int(),
    )
    user_settings = graphene.Field(UserSettingsType, user_id=graphene.Int(required=True))

    @staticmethod
    def resolve_trades(
        info,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        user_id: Optional[int] = None,
    ) -> List[Trade]:
        session = SessionLocal()
        try:
            query = session.query(Trade)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if start:
                query = query.filter(Trade.open_time >= start)
            if end:
                query = query.filter(Trade.close_time <= end)
            if user_id:
                query = query.filter(Trade.user_id == user_id)
            return [_serialize_trade(trade) for trade in query.all()]
        finally:
            session.close()

    @staticmethod
    def resolve_user_settings(info, user_id: int) -> Optional[UserSettings]:
        session = SessionLocal()
        try:
            settings = session.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                return None
            return _serialize_settings(settings)
        finally:
            session.close()


schema = graphene.Schema(query=Query)
