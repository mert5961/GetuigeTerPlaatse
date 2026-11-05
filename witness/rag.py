"""Retrieval-augmented generation, built by hand (no framework).

The four steps of RAG, each one small function:
  1. chunk   - split a document into pieces (here: one memory / checklist item per line)
  2. embed   - turn each piece into a vector with an embedding model
  3. search  - embed the question and find the closest pieces (cosine similarity)
  4. ground  - give only those pieces to the LLM (done in witness.py / debrief.py)
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .llm import embed

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "embeddings.json"
LINE = re.compile(r"^- \[(?P<id>[A-Z]\d+)\] (?P<text>.+)$")


@dataclass
class Chunk:
    id: str
    text: str
    section: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


# 1. chunk ------------------------------------------------------------------
def chunk_markdown(path: Path) -> List[Chunk]:
    """One chunk per '- [ID] text' line; keep the section heading as context."""
    chunks, section = [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        m = LINE.match(line.strip())
        if m:
            chunks.append(Chunk(m["id"], m["text"], section))
    return chunks


# 2. embed (with a small on-disk cache so restarts are instant) --------------
def _load_cache() -> Dict[str, List[float]]:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def embed_texts(texts: List[str], prefix: str) -> np.ndarray:
    cache = _load_cache()
    keys = [hashlib.sha1((prefix + t).encode()).hexdigest() for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    if missing:
        vectors = embed([prefix + t for _, t in missing])
        for (k, _), v in zip(missing, vectors):
            cache[k] = v
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    m = np.array([cache[k] for k in keys], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)  # unit length -> dot product = cosine


# 3. search -----------------------------------------------------------------
class Index:
    """A tiny in-memory vector store: a matrix of unit vectors plus the chunks."""

    # nomic-embed-text expects these task prefixes on documents and queries
    DOC, QUERY = "search_document: ", "search_query: "

    def __init__(self, path: Path):
        self.chunks = chunk_markdown(path)
        self.matrix = embed_texts([f"{c.section}: {c.text}" for c in self.chunks], self.DOC)

    def search(self, query: str, k: int = 4) -> List[Hit]:
        q = embed_texts([query], self.QUERY)[0]
        scores = self.matrix @ q
        best = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]

    def get(self, chunk_id: str) -> Chunk:
        return next(c for c in self.chunks if c.id == chunk_id)
