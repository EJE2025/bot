"""Simple trading engine microservice handling order intake and execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator


class OrderModel(BaseModel):
    id: int
    user_id: int
    symbol: str
    side: str
    quantity: float
    price: float

    @validator("side")
    def validate_side(cls, value: str) -> str:
        if value.lower() not in {"buy", "sell"}:
            raise ValueError("Side must be 'buy' or 'sell'")
        return value.lower()


@dataclass
class Order:
    id: int
    user_id: int
    symbol: str
    side: str
    quantity: float
    price: float


@dataclass
class OrderBook:
    orders: Dict[int, Order] = field(default_factory=dict)

    def submit(self, order: Order) -> Order:
        if order.id in self.orders:
            raise ValueError("Order already exists")
        self.orders[order.id] = order
        return order

    def cancel(self, order_id: int) -> Order:
        try:
            return self.orders.pop(order_id)
        except KeyError as exc:
            raise ValueError("Order not found") from exc

    def list(self) -> List[Order]:
        return list(self.orders.values())


app = FastAPI(title="Trading Engine")
book = OrderBook()


@app.post("/orders", response_model=OrderModel)
def create_order(order: OrderModel) -> Order:
    record = Order(**order.dict())
    try:
        stored = book.submit(record)
        return OrderModel(**stored.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/orders/{order_id}", response_model=OrderModel)
def cancel_order(order_id: int) -> Order:
    try:
        removed = book.cancel(order_id)
        return OrderModel(**removed.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/orders", response_model=List[OrderModel])
def list_orders() -> List[OrderModel]:
    return [OrderModel(**order.__dict__) for order in book.list()]
