"""The TaxBot pipeline. Each step plays to a strength:

  1. extract   (SLM + JSON schema)  read the question, copy out amounts and the taxpayer category
  2. calculate (plain Python)       exact tax from the tested engine; no model arithmetic
  3. retrieve  (embeddings)         find the rule sections that answer the question
  4. answer    (SLM, streamed)      explain using only the calculation and the retrieved text
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from . import llm
from .amounts import appears_in, find_amounts, parse_amount, tk
from .knowledge import CATEGORY_LABELS, Retriever
from .tax import CATEGORIES, TaxInput, compute_tax, load_rules

# Rule sections shown as sources when the calculator answers (no search needed).
CALC_SECTIONS = ("Salary exemption", "Tax rates and slabs", "Investment tax rebate")

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "wants_tax_calculation": {"type": "boolean"},
        "salary_text": {"type": "string"},
        "salary_period": {"type": "string", "enum": ["monthly", "annual", "unknown"]},
        "bonus_text": {"type": "string"},
        "investment_text": {"type": "string"},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "parent_of_disabled": {"type": "boolean"},
        "new_taxpayer": {"type": "boolean"},
    },
    "required": ["wants_tax_calculation", "salary_text", "salary_period", "bonus_text",
                 "investment_text", "category", "parent_of_disabled", "new_taxpayer"],
}

EXTRACT_PROMPT = """You read questions about Bangladesh salary income tax and fill in a form.
Copy amounts EXACTLY as written in the question (e.g. "80k", "1.2 lakh", "৫০ হাজার", "Tk 9,00,000"). Use "" when not mentioned. Never calculate or convert.
- wants_tax_calculation: true if they want to know how much tax THEY pay on a specific salary.
- salary_period: "monthly" if the salary is per month (মাসিক, প্রতি মাসে, /month), "annual" if per year (বছরে, yearly), else "unknown".
- bonus_text: total festival bonus amount if given.
- investment_text: amount invested in DPS, savings certificates (সঞ্চয়পত্র), provident fund, shares, etc.
- category: woman_or_senior for women or age 65+; disabled; third_gender; freedom_fighter for war-wounded freedom fighters or July warriors; otherwise general.
- parent_of_disabled: true only if they are the parent/guardian of a person with disability.
- new_taxpayer: true only if they say it is their first time paying tax / first return."""

ANSWER_PROMPT = """You are TaxBot, a careful assistant for Bangladesh salary income tax (Finance Act 2026).
Rules:
- Use ONLY the CALCULATION and SOURCES below. If they do not answer the question, say you don't know and suggest checking nbr.gov.bd or a tax practitioner.
- Copy every number exactly as written in CALCULATION or SOURCES. Never do your own arithmetic.
- Be brief: 2-4 sentences. Cite sources like [1].
- {language}"""

BANGLA = re.compile(r"[ঀ-৿]")
# Small models tend to switch yes/no flags on for no reason, so a flag only counts with supporting words.
NEW_TAXPAYER_WORDS = re.compile(r"first[- ]time|first (?:return|year)|new taxpayer|never (?:paid|filed)|প্রথম|নতুন করদাতা", re.I)
DISABILITY_WORDS = re.compile(r"disab|প্রতিবন্ধী", re.I)
PARENT_WORDS = re.compile(r"parent|guardian|child|son|daughter|সন্তান|ছেলে|মেয়ে|অভিভাবক", re.I)


def language_rule(question: str) -> str:
    if BANGLA.search(question):
        return "The question is in Bangla: answer in simple Bangla (বাংলায় উত্তর দিন). Keep numbers as given."
    return "Answer in English."


_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def bn(x: float) -> str:
    return tk(x).replace("Tk ", "").translate(_TO_BN) + " টাকা"


def pct(rate: float, bangla: bool = False) -> str:
    s = f"{rate * 100:g}%"
    return s.translate(_TO_BN) if bangla else s


def headline(calc: dict, bangla: bool) -> str:
    """The verdict sentence, written by code so the most important line can't be wrong."""
    r = calc["result"]
    if bangla:
        if r["net_tax"] == 0:
            return (f"আপনাকে কোনো আয়কর দিতে হবে না: আপনার করযোগ্য আয় ({bn(r['taxable_income'])}) "
                    f"করমুক্ত আয়সীমার ({bn(r['threshold'])}) মধ্যে।")
        return f"আনুমানিক আয়কর: বছরে {bn(r['net_tax'])} (মাসে প্রায় {bn(r['net_tax'] / 12)})।"
    if r["net_tax"] == 0:
        return (f"No income tax is payable: your taxable income ({tk(r['taxable_income'])}) is within "
                f"the tax-free limit ({tk(r['threshold'])}).")
    return f"Estimated income tax: {tk(r['net_tax'])} for the year (about {tk(r['net_tax'] / 12)} a month)."


def explanation(calc: dict, bangla: bool) -> str:
    """How the number was worked out, written by code: instant, exact, and fluent in both languages."""
    inp, r = calc["input"], calc["result"]
    cap = load_rules()["employment_exemption"]["cap"]
    slabs = r["slab_lines"]
    if bangla:
        out = [f"বার্ষিক মোট বেতন {bn(r['gross'])}। এর এক-তৃতীয়াংশ (সর্বোচ্চ {bn(cap)}) করমুক্ত, অর্থাৎ "
               f"{bn(r['exemption'])}; তাই করযোগ্য আয় {bn(r['taxable_income'])}।"]
        if slabs:
            steps = "; ".join(f"{bn(s['amount'])}-এর ওপর {pct(s['rate'], True)} = {bn(s['tax'])}" for s in slabs)
            out.append(f"প্রথম {bn(r['threshold'])} করমুক্ত। বাকি অংশে স্তরভিত্তিক কর: {steps}। "
                       f"মোট {bn(r['gross_tax'])}।")
        if r["rebate"]:
            out.append(f"{bn(inp['eligible_investment'])} বিনিয়োগের জন্য কর রেয়াত {bn(r['rebate'])}।")
        if r["minimum_tax_applied"]:
            out.append(f"আয় করমুক্ত সীমার বেশি হওয়ায় ন্যূনতম কর {bn(r['net_tax'])} প্রযোজ্য।")
        elif not r["rebate"] and r["max_rebate"]:
            out.append(f"পরামর্শ: ডিপিএস, সঞ্চয়পত্র বা প্রভিডেন্ট ফান্ডে {bn(r['investment_for_max_rebate'])} "
                       f"বিনিয়োগ করলে কর সর্বোচ্চ {bn(r['max_rebate'])} কমতে পারে।")
        if calc["assumed_period"]:
            kind = "মাসিক" if calc["assumed_period"] == "monthly" else "বার্ষিক"
            out.append(f"মাসিক না বার্ষিক বলা হয়নি, তাই এটি {kind} বেতন ধরা হয়েছে।")
        if not calc["bonus"]:
            out.append("উৎসব বোনাস ধরা হয়নি; বোনাস পেলে তা উল্লেখ করে আবার জিজ্ঞেস করুন।")
        return "\n\n".join(out)

    out = [f"Annual gross salary is {tk(r['gross'])}. One-third of it (up to {tk(cap)}) is exempt, "
           f"{tk(r['exemption'])}, so taxable income is {tk(r['taxable_income'])}."]
    if slabs:
        steps = "; ".join(f"{pct(s['rate'])} on {tk(s['amount'])} = {tk(s['tax'])}" for s in slabs)
        out.append(f"The first {tk(r['threshold'])} is tax-free. The rest is taxed in slabs: {steps}. "
                   f"Total {tk(r['gross_tax'])}.")
    if r["rebate"]:
        out.append(f"Investing {tk(inp['eligible_investment'])} gives a rebate of {tk(r['rebate'])}.")
    if r["minimum_tax_applied"]:
        out.append(f"Because income is above the tax-free limit, the minimum tax of {tk(r['net_tax'])} applies.")
    elif not r["rebate"] and r["max_rebate"]:
        out.append(f"Tip: investing {tk(r['investment_for_max_rebate'])} in DPS, savings certificates or "
                   f"provident fund could cut your tax by up to {tk(r['max_rebate'])}.")
    out += calc["assumptions"]
    return "\n\n".join(out)


def extract(question: str, model: str | None = None) -> tuple[dict, dict]:
    raw, st = llm.chat(
        [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": question}],
        model=model, schema=EXTRACT_SCHEMA, temperature=0, max_tokens=300,
    )
    try:
        return json.loads(raw), st
    except json.JSONDecodeError:  # hit the token cap mid-JSON (e.g. a whitespace loop): treat as "no fields"
        return {}, st


def calculate(fields: dict, question: str) -> dict | None:
    """Turn extracted fields into a tax result, or None if there is nothing to calculate."""
    salary_text = fields.get("salary_text", "")
    if not parse_amount(salary_text):
        # qwen3:1.7b once copied "বার্ষিক" (the word "annual") instead of "৯ লাখ". If the question
        # holds exactly one amount, that must be the salary.
        amounts = find_amounts(question)
        salary_text = amounts[0] if len(amounts) == 1 else ""
    if not appears_in(salary_text, question):  # the model invented a salary that isn't in the question
        return None
    amount = parse_amount(salary_text)
    if not amount:
        return None

    assumptions = []
    period = assumed_period = fields.get("salary_period", "unknown")
    if period == "unknown":
        period = assumed_period = "monthly" if amount < 300_000 else "annual"
        assumptions.append(f"Assumed {tk(amount)} is a {period} salary.")
    else:
        assumed_period = None
    annual = amount * 12 if period == "monthly" else amount

    bonus = parse_amount(fields.get("bonus_text")) if appears_in(fields.get("bonus_text"), question) else None
    if bonus:
        annual += bonus
    else:
        assumptions.append("No festival bonus included; add it if you get one.")
    investment = (parse_amount(fields.get("investment_text"))
                  if appears_in(fields.get("investment_text"), question) else None) or 0

    category = fields.get("category") if fields.get("category") in CATEGORIES else "general"
    parent = bool(fields.get("parent_of_disabled")) and bool(
        DISABILITY_WORDS.search(question) and PARENT_WORDS.search(question))
    new_taxpayer = bool(fields.get("new_taxpayer")) and bool(NEW_TAXPAYER_WORDS.search(question))
    inp = TaxInput(annual_salary=annual, category=category, parent_of_disabled=parent,
                   eligible_investment=investment, new_taxpayer=new_taxpayer)
    res = compute_tax(inp)
    return {"input": inp.__dict__, "monthly_salary": amount if period == "monthly" else None,
            "bonus": bonus or 0, "assumptions": assumptions, "assumed_period": assumed_period,
            "result": res.to_dict()}


def calculation_text(calc: dict) -> str:
    inp, r = calc["input"], calc["result"]
    lines = []
    if calc["monthly_salary"]:
        lines.append(f"- Monthly salary: {tk(calc['monthly_salary'])}")
    if calc["bonus"]:
        lines.append(f"- Festival bonus: {tk(calc['bonus'])}")
    lines += [
        f"- Taxpayer category: {CATEGORY_LABELS[inp['category']]}",
        f"- Annual gross salary: {tk(r['gross'])}",
        f"- Salary exemption (lower of 1/3 or Tk 5,00,000): {tk(r['exemption'])}",
        f"- Taxable income: {tk(r['taxable_income'])}",
        f"- Tax-free threshold: {tk(r['threshold'])}",
    ]
    lines += [f"- Slab {line['rate'] * 100:g}% on {tk(line['amount'])} = {tk(line['tax'])}" for line in r["slab_lines"]]
    lines.append(f"- Tax before rebate: {tk(r['gross_tax'])}")
    if inp["eligible_investment"]:
        lines.append(f"- Investment {tk(inp['eligible_investment'])} gives rebate: {tk(r['rebate'])}")
    elif r["max_rebate"]:
        lines.append(f"- Investing {tk(r['investment_for_max_rebate'])} would cut tax by up to {tk(r['max_rebate'])}")
    if r["minimum_tax_applied"]:
        lines.append("- Minimum tax applies")
    lines += [f"- TAX PAYABLE FOR THE YEAR: {tk(r['net_tax'])} (about {tk(r['net_tax'] / 12)} a month)",
              f"- Effective tax rate: {r['effective_rate'] * 100:.1f}%"]
    lines += [f"- Note: {a}" for a in calc["assumptions"]]
    return "\n".join(lines)


class TaxBot:
    def __init__(self, model: str | None = None):
        self.model = model or llm.CHAT_MODEL
        self.retriever = Retriever()

    def ask(self, question: str, model: str | None = None, explain: bool = False) -> Iterator[dict]:
        """Streams events: {"type": "meta", ...}, many {"type": "token"}, then {"type": "done"}.

        Speed comes from skipping work: a question with no amount in it can't be a tax calculation,
        so it skips extraction; a calculation is answered entirely by code unless explain=True
        asks the model to write the explanation (slower, and weaker in Bangla on small models).
        """
        model = model or self.model
        bangla = bool(BANGLA.search(question))
        fields, extract_stats, calc = {}, None, None
        if find_amounts(question):
            fields, extract_stats = extract(question, model)
            calc = calculate(fields, question) if fields.get("wants_tax_calculation") else None

        if calc and not explain:
            sources = self.retriever.sections(CALC_SECTIONS)
        else:
            sources = self.retriever.search(question, k=3 if calc else 2)
        yield {"type": "meta", "model": model, "fields": fields, "calc": calc,
               "sources": [{k: s[k] for k in ("id", "heading", "source", "score")} for s in sources]}

        if calc and not explain:
            yield {"type": "token", "text": headline(calc, bangla) + "\n\n" + explanation(calc, bangla)}
            yield {"type": "done", "stats": {"extract": extract_stats}}
            return

        context = "SOURCES:\n" + "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, 1))
        task = f"QUESTION: {question}"
        if calc:
            context = "CALCULATION (exact, from the official-rules calculator):\n" + calculation_text(calc) + "\n\n" + context
            first = headline(calc, bangla)
            yield {"type": "token", "text": first + "\n\n"}
            task += (f"\n\nThe answer already starts with: \"{first}\"\nContinue with a brief explanation "
                     f"of how it was worked out (2-4 sentences). Do not repeat that sentence.")
        messages = [
            {"role": "system", "content": ANSWER_PROMPT.format(language=language_rule(question))},
            {"role": "user", "content": f"{context}\n\n{task}"},
        ]
        # Small models often repeat the headline despite being told not to, so hold back the
        # opening until we can see whether it is a repeat, and drop it if so.
        pending = "" if calc else None
        for ev in llm.chat_stream(messages, model=model, max_tokens=250):
            if "text" in ev and pending is None:
                yield {"type": "token", "text": ev["text"]}
            elif "text" in ev:
                pending += ev["text"]
                if len(pending) > len(first) or "\n" in pending.lstrip():
                    if pending.lstrip().startswith(first[:20]):
                        pending = pending.lstrip()[len(first):] if pending.lstrip().startswith(first) \
                            else pending.lstrip().partition("\n")[2]
                    yield {"type": "token", "text": pending.lstrip()}
                    pending = None
            else:
                if pending:
                    yield {"type": "token", "text": pending}
                stats = {"extract": extract_stats, "answer": ev["stats"]}
                yield {"type": "done", "stats": {k: v for k, v in stats.items() if v}}

    def answer(self, question: str, model: str | None = None, explain: bool = False) -> dict:
        """Non-streaming convenience wrapper: collects the whole reply."""
        out = {"text": ""}
        for ev in self.ask(question, model, explain):
            if ev["type"] == "token":
                out["text"] += ev["text"]
            elif ev["type"] == "meta":
                out.update(ev)
            else:
                out["stats"] = ev["stats"]
        return out
