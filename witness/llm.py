"""Thin client for a local Ollama server: embeddings + structured (JSON-schema) chat."""
import os
from typing import List, Type, TypeVar

import httpx
from pydantic import BaseModel

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3.5:27b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

T = TypeVar("T", bound=BaseModel)
_client = httpx.Client(base_url=OLLAMA, timeout=300)


def embed(texts: List[str]) -> List[List[float]]:
    r = _client.post("/api/embed", json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    return r.json()["embeddings"]


def chat_structured(messages: List[dict], schema: Type[T], temperature: float = 0.4) -> T:
    """Ask the model to answer in the Pydantic schema, then validate it.

    Ollama constrains generation to the JSON schema; Pydantic still validates,
    because a well-formed JSON can still break our own rules (see witness.py).
    """
    r = _client.post(
        "/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        },
    )
    r.raise_for_status()
    return schema.model_validate_json(r.json()["message"]["content"])
