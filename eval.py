"""Evaluation: is retrieval finding the right memories, and does the witness stay honest?

  .venv/bin/python eval.py            # retrieval only (fast, embeddings only)
  .venv/bin/python eval.py --llm      # + witness answers (slow, uses the chat model)
"""
import sys

from witness.witness import Witness

# question -> memory ids that SHOULD be retrieved (any one of them counts as a hit)
RETRIEVAL = [
    ("What happened?", {"W04", "W05"}),
    ("What time did you hear the noise?", {"W04"}),
    ("When did you call 112?", {"W14"}),
    ("What was he wearing?", {"W06"}),
    ("How tall was he? Anything special about how he moved?", {"W07"}),
    ("Did you see his face?", {"W09"}),
    ("Which way did he go?", {"W08"}),
    ("How did he leave the area?", {"W10"}),
    ("Did you touch anything?", {"W12", "W15", "W16"}),
    ("Did you go into the flat?", {"W11", "W12", "W16"}),
    ("Did you move anything in there?", {"W15"}),
    ("Has anyone else been inside?", {"W19"}),
    ("Are there any cameras?", {"W17"}),
    ("Have you noticed anything unusual in the last few days?", {"W18"}),
    ("How is the lighting on the stairs?", {"W21"}),
    ("Do you wear glasses?", {"W20"}),
    ("Are you hurt?", {"W22"}),
    ("Who lives in 3A?", {"W03"}),
    ("What is your name and address?", {"W01"}),
]

# questions whose answer is NOT in the case file: she must not invent anything
UNKNOWN = [
    "What was the licence plate of the moped?",
    "What colour were his shoes?",
    "Did he say anything to you?",
    "What was the burglar's name?",
    "Did the man say anything tonight?",
]


# phrases that show she admits not knowing (a simple, transparent check; read the CHECK lines yourself)
DONT_KNOW = ["don't know", "do not know", "didn't see", "did not see", "didn't get", "didn't hear", "not sure",
             "no idea", "couldn't see", "didn't notice", "can't say", "never saw", "didn't say", "no, he"]


def retrieval(w: Witness, k: int) -> float:
    hits = 0
    for q, expected in RETRIEVAL:
        got = [h.chunk.id for h in w.index.search(q, k)]
        ok = bool(expected & set(got))
        hits += ok
        if not ok:
            print(f"  miss@{k}: {q!r} expected {sorted(expected)} got {got}")
    return hits / len(RETRIEVAL)


def honesty(w: Witness) -> None:
    print("\nUnknown facts (she must admit she doesn't know, not invent):")
    for q in UNKNOWN:
        r = w.answer([], q)
        admits = any(p in r.reply.lower() for p in DONT_KNOW)
        flag = "OK " if admits else "CHECK"
        print(f"  [{flag}] {q}\n         -> {r.reply}  {r.used_memories}")


if __name__ == "__main__":
    w = Witness()
    for k in (1, 3, 4, 5):
        print(f"recall@{k}: {retrieval(w, k):.0%}")
    if "--llm" in sys.argv:
        honesty(w)
