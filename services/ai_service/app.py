"""Flask application exposing AI-powered endpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from flask import Flask, jsonify, request

from .chatgpt import ask_gpt


@dataclass
class MetricSummary:
    pnl_total: float
    win_rate: float
    open_positions: int
    risk: float

    @classmethod
    def from_payload(cls, payload: Dict[str, float]) -> "MetricSummary":
        return cls(
            pnl_total=float(payload.get("pnl_total", 0.0)),
            win_rate=float(payload.get("win_rate", 0.0)),
            open_positions=int(payload.get("open_positions", 0)),
            risk=float(payload.get("risk", 0.0)),
        )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.post("/api/ai/report")
    def generate_report():
        summary = MetricSummary.from_payload(request.json or {})
        messages = [
            {
                "role": "system",
                "content": "Eres un experto analista de trading. Responde en español.",
            },
            {
                "role": "user",
                "content": (
                    "Genera un informe breve y claro sobre el rendimiento de hoy:\n"
                    f"PnL total: {summary.pnl_total}\n"
                    f"Win rate: {summary.win_rate}\n"
                    f"Operaciones abiertas: {summary.open_positions}\n"
                    f"Riesgo total: {summary.risk}"
                ),
            },
        ]
        report = ask_gpt(messages)
        return jsonify({"report": report})

    @app.post("/api/ai/chat")
    def chat():
        payload = request.json or {}
        content = payload.get("message", "")
        summary = payload.get("summary", {})
        pnl = summary.get("pnl", 0.0)
        win_rate = summary.get("win_rate", 0.0)
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente de trading que proporciona respuestas claras y concisas en español.",
            },
            {
                "role": "user",
                "content": (
                    f"Mis métricas actuales: PnL={pnl}, win rate={win_rate}. "
                    f"{content}"
                ).strip(),
            },
        ]
        answer = ask_gpt(messages)
        return jsonify({"answer": answer})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
