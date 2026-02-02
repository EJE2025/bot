from __future__ import annotations

from typing import Any, Dict, Optional
import importlib.util
import json
import os

OpenAI = None
if importlib.util.find_spec("openai") is not None:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore[no-redef]


def generate_llm_explanation(
    analysis: Dict[str, Any],
    model: str,
    *,
    timeout: float = 15.0,
    max_tokens: int = 600,
) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(timeout=timeout)

    schema = {
        "type": "object",
        "properties": {
            "titulo": {"type": "string"},
            "resumen": {"type": "string"},
            "puntos_clave": {"type": "array", "items": {"type": "string"}},
            "niveles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tipo": {"type": "string", "enum": ["support", "resistance"]},
                        "desde": {"type": "number"},
                        "hasta": {"type": "number"},
                        "toques": {"type": "integer"},
                    },
                    "required": ["tipo", "desde", "hasta", "toques"],
                    "additionalProperties": False,
                },
            },
            "idea_trade": {
                "type": ["object", "null"],
                "properties": {
                    "direccion": {"type": "string", "enum": ["long", "short"]},
                    "entrada": {"type": "number"},
                    "stop": {"type": "number"},
                    "take_profit": {"type": "number"},
                    "rr": {"type": "number"},
                },
                "required": ["direccion", "entrada", "stop", "take_profit", "rr"],
                "additionalProperties": False,
            },
            "nota_riesgo": {"type": "string"},
        },
        "required": ["titulo", "resumen", "puntos_clave", "niveles", "idea_trade", "nota_riesgo"],
        "additionalProperties": False,
    }

    system = (
        "Eres un analista técnico. Responde SOLO usando el JSON dado. "
        "No inventes precios, niveles ni eventos. Si falta información, dilo explícitamente. "
        "Escribe en español, conciso y claro."
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(analysis, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ta_explanation",
                "strict": True,
                "schema": schema,
            }
        },
        max_output_tokens=max_tokens,
    )

    try:
        return json.loads(response.output_text)
    except Exception:
        return None
