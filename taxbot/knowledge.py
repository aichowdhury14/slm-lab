"""Retrieval (the "R" in RAG): split documents into chunks, embed them, search by meaning.

The index is just two files: index/chunks.json (text) and index/vectors.npy (one row per chunk).
For a few hundred chunks a NumPy dot product is instant; a vector database only pays off at
tens of thousands.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from . import INDEX_DIR, KNOWLEDGE_DIR, llm
from .amounts import tk
from .tax import TaxInput, compute_tax, load_rules, tax_free_monthly_salary

CATEGORY_LABELS = {
    "general": "General taxpayer (সাধারণ করদাতা)",
    "woman_or_senior": "Women and people aged 65+ (নারী ও ৬৫ বছর বা তার বেশি বয়সী)",
    "disabled": "Persons with disability (প্রতিবন্ধী ব্যক্তি)",
    "third_gender": "Third gender (তৃতীয় লিঙ্গ)",
    "freedom_fighter": "Gazetted war-wounded freedom fighters and July warriors (যুদ্ধাহত মুক্তিযোদ্ধা ও জুলাই যোদ্ধা)",
}


def rules_document() -> str:
    """Render tax_rules.json (plus worked examples from the real engine) as Markdown."""
    r = load_rules()
    ex, rb, mt = r["employment_exemption"], r["rebate"], r["minimum_tax"]
    out = [f"# Salary income tax rules: {r['label']}", "", r["applies_to"], ""]

    out += ["## Tax-free income threshold (করমুক্ত আয়সীমা)", "",
            "No tax is due on taxable income up to the threshold for your category:", ""]
    for cat, amount in r["thresholds"].items():
        monthly = tax_free_monthly_salary(cat)
        out.append(f"- {CATEGORY_LABELS[cat]}: {tk(amount)} per year "
                   f"(salary up to about {tk(monthly)} a month is tax-free)")
    out += ["", f"A parent or legal guardian of a person with disability gets an extra "
                f"{tk(r['parent_of_disabled_extra'])} on their threshold.", ""]

    out += ["## Tax rates and slabs (করহার)", "",
            "Taxable income above the threshold is taxed in slabs:", ""]
    labels = ["First", "Next", "Next", "Next"]
    for i, s in enumerate(r["slabs"]):
        pct = f"{s['rate'] * 100:g}%"
        out.append(f"- {labels[i]} {tk(s['width'])}: {pct}" if s["width"] else f"- Everything above that: {pct}")
    out.append("")

    out += ["## Salary exemption (বেতন আয়ে অব্যাহতি)", "",
            f"Before applying the slabs, salaried people deduct the lower of one-third of gross salary "
            f"or {tk(ex['cap'])}. Taxable income = gross salary minus this exemption. "
            f"Gross salary means all cash pay for the year, including festival bonuses.", ""]

    out += ["## Investment tax rebate (বিনিয়োগ কর রেয়াত)", "",
            f"Eligible investments (for example DPS, savings certificates/sanchayapatra, your provident "
            f"fund contribution, listed shares) reduce tax. The rebate is the lowest of: "
            f"{rb['income_rate'] * 100:g}% of taxable income, {rb['investment_rate'] * 100:g}% of the "
            f"eligible investment, or {tk(rb['cap'])}. It can never be more than the tax itself.", ""]

    out += ["## Minimum tax (ন্যূনতম কর)", "",
            f"If your taxable income is above the threshold, you pay at least {tk(mt['regular'])}, "
            f"even if the slab tax after rebate is lower. A first-time (new) taxpayer pays at least "
            f"{tk(mt['new_taxpayer'])}. If income is at or below the threshold, no tax is due.", ""]

    out += ["## Worked examples (উদাহরণ)", ""]
    for monthly in (50_000, 80_000, 150_000):
        res = compute_tax(TaxInput(annual_salary=monthly * 12))
        out.append(f"- Monthly salary {tk(monthly)} (annual {tk(monthly * 12)}): exemption "
                   f"{tk(res.exemption)}, taxable {tk(res.taxable_income)}, tax before rebate "
                   f"{tk(res.gross_tax)}, tax payable with no investment {tk(res.net_tax)} "
                   f"(about {tk(res.net_tax / 12)} a month).")
    out.append("")

    out += ["## Sources and confidence", ""] + [f"- {n}" for n in r["confidence_notes"]]
    out += [f"- Source: {s['name']} ({s['date']}) {s['url']}" for s in r["sources"]]
    return "\n".join(out) + "\n"


def load_documents() -> dict[str, str]:
    docs = {"tax-rules-2026.md (generated)": rules_document()}
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8")
    return docs


def chunk(name: str, text: str) -> list[dict]:
    """One chunk per '## ' section, prefixed with the document title so it makes sense alone."""
    title = text.splitlines()[0].lstrip("# ").strip()
    chunks = []
    for part in re.split(r"(?m)^## ", text)[1:]:
        heading, _, body = part.partition("\n")
        chunks.append({
            "id": f"{name}#{heading.strip()}",
            "source": name,
            "heading": heading.strip(),
            "text": f"{title} > {heading.strip()}\n{body.strip()}",
        })
    return chunks


def build_index(verbose: bool = True) -> list[dict]:
    chunks = [c for name, text in load_documents().items() for c in chunk(name, text)]
    vectors = np.array(llm.embed([c["text"] for c in chunks]), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # unit length -> dot product = cosine
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    meta = {"embed_model": llm.EMBED_MODEL, "fingerprint": _fingerprint(chunks), "chunks": chunks}
    (INDEX_DIR / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print(f"Indexed {len(chunks)} chunks with {llm.EMBED_MODEL} -> {INDEX_DIR}")
    return chunks


def _fingerprint(chunks: list[dict]) -> str:
    return hashlib.sha256("".join(c["text"] for c in chunks).encode()).hexdigest()[:16]


class Retriever:
    def __init__(self):
        path = INDEX_DIR / "chunks.json"
        if not path.exists():
            build_index()
        meta = json.loads(path.read_text(encoding="utf-8"))
        current = [c for name, text in load_documents().items() for c in chunk(name, text)]
        if meta["fingerprint"] != _fingerprint(current) or meta["embed_model"] != llm.EMBED_MODEL:
            print("Knowledge changed since the last index; rebuilding.")
            build_index()
            meta = json.loads(path.read_text(encoding="utf-8"))
        self.chunks = meta["chunks"]
        self.vectors = np.load(INDEX_DIR / "vectors.npy")

    def sections(self, headings: tuple[str, ...]) -> list[dict]:
        """Chunks whose heading starts with one of these: a lookup, no embedding model needed."""
        return [{**c, "score": None} for h in headings for c in self.chunks if c["heading"].startswith(h)]

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = np.array(llm.embed([query])[0], dtype=np.float32)
        scores = self.vectors @ (q / np.linalg.norm(q))
        best = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in best]
