"""The debrief: judge the officer's questions against the checklist, and show what stayed hidden."""
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

from .llm import chat_structured
from .models import ChecklistResult, DebriefLLM, DebriefResponse, Source, Turn
from .rag import chunk_markdown

DATA = Path(__file__).resolve().parent.parent / "data"

# Memories that matter for the investigation (not just small talk about who she is).
KEY_MEMORIES = {f"W{i:02d}" for i in range(4, 22)}

PROMPT = """You are a calm, fair police trainer reviewing a trainee's first questioning of a witness.

CHECKLIST:
{checklist}

THE TRAINEE'S QUESTIONS, in order:
{questions}

For EVERY checklist item (all {n}), decide if the trainee's questions covered it. Be strict but fair:
an item counts only if a question clearly addressed it. As evidence, quote the question that covered it,
or leave evidence empty. Then give 1-3 strengths and 1-3 concrete tips for next time."""


def _quotes_a_question(evidence: str, questions: List[str]) -> bool:
    """True if the evidence is (close to) one of the officer's real questions."""
    e = evidence.strip().strip('"').lower()
    if not e:
        return False
    return any(e in q.lower() or q.lower() in e or SequenceMatcher(None, e, q.lower()).ratio() > 0.6
               for q in questions)


def debrief(history: List[Turn]) -> DebriefResponse:
    checklist = chunk_markdown(DATA / "checklist.md")
    memories = chunk_markdown(DATA / "case_file.md")
    questions = [t.text for t in history if t.role == "officer"]

    result = chat_structured(
        [{
            "role": "user",
            "content": PROMPT.format(
                checklist="\n".join(f"[{c.id}] {c.text}" for c in checklist),
                questions="\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)) or "(none)",
                n=len(checklist),
            ),
        }],
        DebriefLLM,
        temperature=0.1,
    )

    # Validation: exactly one result per checklist item, in checklist order; unknown ids dropped.
    by_id = {r.id: r for r in result.items}
    items = [by_id.get(c.id, ChecklistResult(id=c.id, covered=False, evidence="")) for c in checklist]
    # A "covered" item must quote a real question; otherwise it was a hallucinated pass.
    for it in items:
        it.evidence = re.sub(r'^\s*"?\d+\.\s*', "", it.evidence).strip().strip('"')  # drop "3. " numbering
        if it.covered and not _quotes_a_question(it.evidence, questions):
            it.covered, it.evidence = False, ""

    unlocked = sorted({m for t in history if t.role == "witness" for m in t.memories})
    hidden = [Source(id=m.id, text=m.text, score=0) for m in memories
              if m.id in KEY_MEMORIES and m.id not in unlocked]
    covered = sum(it.covered for it in items)
    return DebriefResponse(
        score=f"{covered}/{len(items)}",
        checklist={c.id: c.text for c in checklist},
        items=items,
        unlocked=unlocked,
        never_unlocked=hidden,
        strengths=result.strengths,
        advice=result.advice,
    )
