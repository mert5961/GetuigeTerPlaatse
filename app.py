"""FastAPI app: a chat with the witness, and a debrief at the end.

Run:  .venv/bin/uvicorn app:app --reload   then open http://localhost:8000
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from witness.debrief import debrief
from witness.llm import CHAT_MODEL, EMBED_MODEL
from witness.models import ChatRequest, ChatResponse, DebriefRequest, DebriefResponse
from witness.witness import Witness

app = FastAPI(title="SceneWitness")
witness = Witness()  # builds the vector index once at startup
STATIC = Path(__file__).parent / "static"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info() -> dict:
    return {"chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL, "memories": len(witness.index.chunks)}


@app.post("/api/ask", response_model=ChatResponse)
def ask(req: ChatRequest) -> ChatResponse:
    return witness.answer(req.history, req.question)


@app.post("/api/debrief", response_model=DebriefResponse)
def run_debrief(req: DebriefRequest) -> DebriefResponse:
    return debrief(req.history)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
