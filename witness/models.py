"""Pydantic models: the contract between the LLM, the API and the UI."""
from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class WitnessReply(BaseModel):
    reply: str = Field(description="What Anouk says, in first person, 1-4 short sentences.")
    used_memories: List[str] = Field(
        description="IDs (like W05) of the memories this reply is based on. Empty if she does not know."
    )
    mood: Literal["calm", "nervous", "upset", "confused"]


class Turn(BaseModel):
    role: Literal["officer", "witness"]
    text: str
    memories: List[str] = []  # for witness turns: the memory ids her reply was based on


class ChatRequest(BaseModel):
    history: List[Turn] = []
    question: str


class Source(BaseModel):
    id: str
    text: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    mood: str
    used_memories: List[str]
    retrieved: List[Source]
    retries: int


class ChecklistResult(BaseModel):
    id: str = Field(description="Checklist item id, like C05")
    covered: bool
    evidence: str = Field(description="The officer's question that covered it, or empty if missed")


class DebriefLLM(BaseModel):
    items: List[ChecklistResult]
    strengths: List[str] = Field(description="1-3 short points the officer did well")
    advice: List[str] = Field(description="1-3 short, concrete tips for next time")


class DebriefRequest(BaseModel):
    history: List[Turn]


class DebriefResponse(BaseModel):
    score: str
    checklist: Dict[str, str]
    items: List[ChecklistResult]
    unlocked: List[str]
    never_unlocked: List[Source]
    strengths: List[str]
    advice: List[str]
