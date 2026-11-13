"""Lesson 1: talk to a small language model running on this PC, and measure its speed.

    python learn/01_hello_slm.py

Things to notice:
- The first call is slow: the model file is loaded from disk into RAM. Later calls are faster.
- "prompt tokens/s" (reading) is much faster than "output tokens/s" (writing, one token at a time).
- The 1.7B model is roughly 2x faster than the 4B one. Is its answer worse? That trade-off is
  the core decision in any SLM product.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from taxbot import llm

QUESTION = "In two sentences: what is a small language model, and why run one on a laptop?"

for model in ("qwen3:1.7b", "qwen3:4b"):
    print(f"\n=== {model} ===")
    try:
        text, st = llm.chat([{"role": "user", "content": QUESTION}], model=model)
    except RuntimeError as e:  # most often "out-of-memory": close apps or enlarge the page file
        print(e)
        continue
    print(text.strip())
    print(f"\n[{st['prompt_tokens']} prompt tokens in {st['prompt_seconds']}s, "
          f"{st['output_tokens']} output tokens at {st['tokens_per_second']} tok/s, {st['total_seconds']}s total]")
