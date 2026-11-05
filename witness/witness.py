"""The witness: retrieve her relevant memories, let the LLM answer only from them, validate."""
from pathlib import Path
from typing import List

from .llm import chat_structured
from .models import ChatResponse, Source, Turn, WitnessReply
from .rag import Hit, Index

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_RETRIES = 2

PERSONA = """You are Anouk Visser, 67, a neighbour who just witnessed the aftermath of a break-in.
A police officer who has just arrived is questioning you on the landing. Stay in character.

Rules:
- You may ONLY state facts that appear in YOUR MEMORIES below. Never invent names, times, colours or details.
- If the answer is not in your memories, say you don't know or didn't notice - naturally, in character.
- Answer only what was asked. Ignore memories that do not answer the question; a good officer has to ask.
- Never say that is all you know or saw; just stop talking.
- Keep events apart: tonight's events are not the same as memories from other days. Never merge them.
- Speak plainly, 1-4 short sentences, a little shaken. No stage directions.
- In used_memories list the IDs of every memory you used. Use [] if you answered from none.

YOUR MEMORIES (the only things you know):
{memories}"""


class Witness:
    def __init__(self) -> None:
        self.index = Index(DATA / "case_file.md")

    def retrieve(self, history: List[Turn], question: str, k: int) -> List[Hit]:
        """Search with the question alone AND with the previous question as context, then merge.

        Context helps follow-ups ("what colour was it?") but can drag a new topic off course
        ("Are you hurt?" + "What happened?"), so each memory keeps its best score from either search.
        """
        last = [t.text for t in history if t.role == "officer"][-1:]
        queries = [question] + ([f"{last[0]} {question}"] if last else [])
        best = {}
        for q in queries:
            for h in self.index.search(q, k):
                if h.chunk.id not in best or h.score > best[h.chunk.id].score:
                    best[h.chunk.id] = h
        return sorted(best.values(), key=lambda h: -h.score)[:k]

    def answer(self, history: List[Turn], question: str, k: int = 5) -> ChatResponse:
        hits = self.retrieve(history, question, k)
        allowed = {h.chunk.id for h in hits}
        memories = "\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)

        messages = [{"role": "system", "content": PERSONA.format(memories=memories)}]
        for t in history[-8:]:
            messages.append({"role": "user" if t.role == "officer" else "assistant", "content": t.text})
        messages.append({"role": "user", "content": question})

        # Guardrail: the JSON schema guarantees the *shape*; this loop checks the *content*.
        retries = 0
        reply = chat_structured(messages, WitnessReply)
        while not set(reply.used_memories) <= allowed and retries < MAX_RETRIES:
            retries += 1
            bad = sorted(set(reply.used_memories) - allowed)
            messages.append({"role": "assistant", "content": reply.model_dump_json()})
            messages.append({
                "role": "user",
                "content": f"(System check: {bad} are not in your memories. Answer again using only "
                           f"{sorted(allowed)}, or say you don't know.)",
            })
            reply = chat_structured(messages, WitnessReply, temperature=0.2)

        used = [m for m in reply.used_memories if m in allowed]
        return ChatResponse(
            reply=reply.reply,
            mood=reply.mood,
            used_memories=used,
            retrieved=[Source(id=h.chunk.id, text=h.chunk.text, score=round(h.score, 3)) for h in hits],
            retries=retries,
        )
