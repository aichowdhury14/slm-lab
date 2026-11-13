"""Lesson 2: embeddings. Text in, a list of ~1000 numbers out. Similar meaning -> similar numbers.

    python learn/02_embeddings.py

This is how RAG finds the right paragraph: not by matching words, but by matching meaning,
even across languages (a Bangla question can find an English rule).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from taxbot import llm

query = "করমুক্ত আয়সীমা কত?"  # "What is the tax-free income limit?"
candidates = [
    "The tax-free income threshold for a general taxpayer is Tk 4,00,000.",
    "The return filing deadline for individuals is 30 November.",
    "Investment in savings certificates gives a tax rebate.",
    "Dhaka is the capital of Bangladesh.",
]

vectors = np.array(llm.embed([query] + candidates))
print(f"Each text became a vector of {vectors.shape[1]} numbers. First five of the query: {vectors[0][:5].round(3)}")
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
scores = vectors[1:] @ vectors[0]  # cosine similarity: 1 = same meaning, ~0 = unrelated

print(f"\nQuery: {query}")
for score, text in sorted(zip(scores, candidates), reverse=True):
    print(f"  {score:.3f}  {text}")
print("\nTry your own query, or swap in another embedding model (e.g. nomic-embed-text) and compare.")
