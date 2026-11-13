"""Lesson 4: why RAG. The same model, with and without the rule book in its prompt.

    python learn/04_why_rag.py

A small model's built-in knowledge is out of date (Bangladesh's 2026 budget changed the
thresholds) and it will confidently make numbers up. Put the right paragraph in the prompt
and the same model answers correctly. That is the whole idea of Retrieval-Augmented Generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from taxbot import llm
from taxbot.knowledge import Retriever

QUESTION = "In Bangladesh for 2026-27, what is the tax-free income threshold for women, and the minimum tax?"
MODEL = llm.CHAT_MODEL

print(f"Q: {QUESTION}\n\n--- Without RAG (model memory only) ---")
text, _ = llm.chat([{"role": "user", "content": QUESTION}], model=MODEL)
print(text.strip())

hits = Retriever().search(QUESTION, k=2)
context = "\n\n".join(h["text"] for h in hits)
print("\n--- Retrieved sections ---")
for h in hits:
    print(f"  {h['score']}  {h['id']}")

print("\n--- With RAG ---")
text, _ = llm.chat([
    {"role": "system", "content": "Answer only from the CONTEXT. Copy numbers exactly."},
    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {QUESTION}"},
], model=MODEL)
print(text.strip())
print("\nCompare with taxbot/data/tax_rules.json: thresholds.woman_or_senior and minimum_tax.regular.")
