"""Lesson 5: where does the time go? Profile each pipeline step before optimising anything.

    python learn/05_where_time_goes.py

For every model call Ollama reports three costs:
  load    reading the model from disk into RAM (only when it isn't already loaded)
  prompt  reading the input tokens (fast, parallel)
  output  writing tokens one at a time (slow; cost grows with answer length)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from taxbot.bot import TaxBot

QUESTIONS = [
    "My salary is 80k a month. How much income tax do I pay?",
    "What is the minimum tax if my income is above the threshold?",
]

bot = TaxBot()
for q in QUESTIONS:
    t0 = time.time()
    first_token = None
    for ev in bot.ask(q):
        if ev["type"] == "meta":
            meta_at = time.time() - t0
        elif ev["type"] == "token" and first_token is None:
            first_token = time.time() - t0
        elif ev["type"] == "done":
            total = time.time() - t0
            stats = ev["stats"]
    print(f"\nQ: {q}")
    print(f"  calculator card shown after {meta_at:5.1f}s, first word after {first_token:5.1f}s, done after {total:5.1f}s")
    for step, st in stats.items():
        if st:
            print(f"  {step:8} load {st['load_seconds']:5.1f}s | prompt {st['prompt_tokens']:4} tok in "
                  f"{st['prompt_seconds']:5.1f}s | output {st['output_tokens']:4} tok in {st['output_seconds']:5.1f}s "
                  f"({st['tokens_per_second']} tok/s)")
