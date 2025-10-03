"""Utility wrapper to interact with OpenAI's ChatGPT API."""
from __future__ import annotations

import os
from typing import Iterable, Mapping

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt(
    messages: Iterable[Mapping[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> str:
    """Send a conversational prompt to the ChatGPT API."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"].strip()
