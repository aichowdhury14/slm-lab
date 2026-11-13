"""Score TaxBot on a fixed question set so you can compare models and prompt changes with numbers.

    python eval/run_eval.py                      # default model
    python eval/run_eval.py --model qwen3:1.7b

Checks per question:
  calc     tax questions: the engine got the right inputs (net tax matches); other questions: no calculation
  source   the expected rule section is in the top 3 retrieved
  answer   the reply text contains the key number or phrase
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # show progress even when piped to a file

from taxbot.amounts import normalize
from taxbot.bot import TaxBot
from taxbot.tax import TaxInput, compute_tax

p = argparse.ArgumentParser()
p.add_argument("--model")
p.add_argument("--limit", type=int, help="only the first N questions (quick check)")
args = p.parse_args()

cases = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))[: args.limit]
bot = TaxBot(args.model)
rows = []
for i, case in enumerate(cases, 1):
    t0 = time.time()
    out = bot.answer(case["q"])
    secs = time.time() - t0
    answer = normalize(out["text"])
    got_tax = out["calc"]["result"]["net_tax"] if out["calc"] else None

    if "tax" in case:
        want = compute_tax(TaxInput(**case["tax"])).net_tax
        calc_ok = got_tax == want
        answer_ok = f"{want:,}".replace(",", "") in answer or want == 0 and ("0" in answer or "no tax" in answer)
        source_ok = None
    else:
        want = None
        calc_ok = got_tax is None
        answer_ok = any(normalize(s) in answer for s in case["answer_any"])
        source_ok = any(case["source"].lower() in s["id"].lower() for s in out["sources"])

    rows.append({"q": case["q"], "calc_ok": calc_ok, "source_ok": source_ok, "answer_ok": answer_ok,
                 "want_tax": want, "got_tax": got_tax, "seconds": round(secs, 1),
                 "fields": out["fields"], "answer": out["text"]})
    mark = lambda ok: "-" if ok is None else ("✓" if ok else "✗")
    print(f"{i:2}. calc {mark(calc_ok)} source {mark(source_ok)} answer {mark(answer_ok)} "
          f"{secs:5.1f}s  {case['q'][:60]}")
    if not calc_ok:
        print(f"      tax: wanted {want}, got {got_tax}; fields {json.dumps(out['fields'], ensure_ascii=False)}")

def pct(key):
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{sum(vals)}/{len(vals)}"

model = bot.model
print(f"\n{model}: calc {pct('calc_ok')}, source {pct('source_ok')}, answer {pct('answer_ok')}, "
      f"avg {sum(r['seconds'] for r in rows) / len(rows):.1f}s per question")
out_dir = ROOT / "eval" / "results"
out_dir.mkdir(exist_ok=True)
(out_dir / f"{model.replace(':', '_')}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
