"""Lesson 3: structured output. Force the model to answer in a JSON shape your code can use.

    python learn/03_structured_output.py

Ollama turns the JSON schema into a grammar, so the model *cannot* produce invalid JSON.
It can still produce wrong values, though, which is why the bot only asks it to COPY amounts
and lets Python (taxbot/amounts.py) convert "1.2 lakh" into 120000.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from taxbot import llm
from taxbot.amounts import parse_amount
from taxbot.bot import extract

QUESTIONS = [
    "I'm a woman, salary 1.2 lakh per month, I put 2 lakh into sanchayapatra. How much tax?",
    "আমার বছরে বেতন ১২ লাখ টাকা, কর কত?",
    "What's the deadline to file my return?",
]

print("A) Asking the model to do the conversion itself (no schema, free text):")
text, _ = llm.chat([{"role": "user", "content":
                     f"How many taka is the monthly salary in this sentence, as a plain number? {QUESTIONS[0]}"}],
                   model="qwen3:1.7b", temperature=0)
print("  ", text.strip()[:200])

print("\nB) Schema extraction + Python conversion (what TaxBot does):")
for q in QUESTIONS:
    fields, st = extract(q, model="qwen3:1.7b")
    print(f"\nQ: {q}\n   {json.dumps(fields, ensure_ascii=False)}")
    print(f"   salary in taka -> {parse_amount(fields['salary_text'])}  ({st['total_seconds']}s)")
