"""Ask TaxBot from the terminal.

    python ask.py "My salary is 80k a month, how much tax?"
    python ask.py --model qwen3:1.7b --debug "আমার মাসিক বেতন ৫০ হাজার টাকা, কত কর দিতে হবে?"
"""

import argparse
import json
import sys

from taxbot.bot import TaxBot

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a code page without Bangla

p = argparse.ArgumentParser()
p.add_argument("question", nargs="+")
p.add_argument("--model")
p.add_argument("--debug", action="store_true", help="show extracted fields, sources and speed")
p.add_argument("--explain", action="store_true", help="let the model explain calculations (slower)")
args = p.parse_args()

for ev in TaxBot(args.model).ask(" ".join(args.question), explain=args.explain):
    if ev["type"] == "token":
        print(ev["text"], end="", flush=True)
    elif args.debug and ev["type"] == "meta":
        print("fields:", json.dumps(ev["fields"], ensure_ascii=False))
        print("tax:", ev["calc"]["result"]["net_tax"] if ev["calc"] else None)
        for s in ev["sources"]:
            print(f"  source {s['score']}: {s['id']}")
        print()
    elif args.debug and ev["type"] == "done":
        for step, st in ev["stats"].items():
            print(f"\n[{step}] load {st['load_seconds']}s, {st['prompt_tokens']} prompt tokens, "
                  f"{st['output_tokens']} output tokens at {st['tokens_per_second']} tok/s, "
                  f"{st['total_seconds']}s total", end="")
print()
