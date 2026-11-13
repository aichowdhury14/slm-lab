"""Pipeline tests with the model replaced by fakes, so they run instantly and without Ollama."""

import pytest

from taxbot import bot as botmod
from taxbot.bot import TaxBot


class FakeRetriever:
    chunk = {"id": "rules#x", "heading": "x", "source": "rules", "score": 1.0, "text": "rules text"}

    def search(self, query, k=3):
        return [self.chunk]

    def sections(self, headings):
        return [{**self.chunk, "score": None}]


def never(*args, **kwargs):
    raise AssertionError("this step should have been skipped")


def make_bot(monkeypatch, fields, reply_pieces=None):
    monkeypatch.setattr(botmod, "extract", (lambda q, model=None: (fields, {})) if fields is not None else never)
    monkeypatch.setattr(botmod.llm, "chat_stream", never if reply_pieces is None else
                        lambda messages, model=None, **kw: iter([{"text": p} for p in reply_pieces] + [{"stats": {"x": 1}}]))
    b = TaxBot.__new__(TaxBot)
    b.model, b.retriever = "fake", FakeRetriever()
    return b


FIELDS = {"wants_tax_calculation": True, "salary_text": "80k", "salary_period": "monthly", "bonus_text": "",
          "investment_text": "", "category": "general", "parent_of_disabled": False, "new_taxpayer": False}
HEADLINE = "Estimated income tax: Tk 24,000 for the year (about Tk 2,000 a month)."
Q = "My salary is 80k a month, how much tax?"


def test_calculation_is_answered_by_code_without_generation(monkeypatch):
    out = make_bot(monkeypatch, FIELDS).answer(Q)  # chat_stream would raise if called
    assert out["text"].startswith(HEADLINE)
    assert "taxable income is Tk 6,40,000" in out["text"]
    assert "10% on Tk 2,40,000 = Tk 24,000" in out["text"]


def test_bangla_calculation_explanation_uses_bangla_digits(monkeypatch):
    fields = {**FIELDS, "salary_text": "৫০ হাজার"}
    out = make_bot(monkeypatch, fields).answer("আমার মাসিক বেতন ৫০ হাজার টাকা, কত কর দিতে হবে?")
    assert out["text"].startswith("আপনাকে কোনো আয়কর দিতে হবে না")
    assert "৬,০০,০০০ টাকা" in out["text"] and "0" not in out["text"]


def test_question_without_amount_skips_extraction(monkeypatch):
    out = make_bot(monkeypatch, None, ["The deadline is ", "30 November [1]."]).answer("When is the filing deadline?")
    assert out["calc"] is None and out["text"] == "The deadline is 30 November [1]."


def test_explain_mode_drops_the_models_repeat_of_the_headline(monkeypatch):
    b = make_bot(monkeypatch, FIELDS, [HEADLINE[:30], HEADLINE[30:], "\n", "Taxable income is Tk 6,40,000."])
    out = b.answer(Q, explain=True)
    assert out["text"] == HEADLINE + "\n\nTaxable income is Tk 6,40,000."


def test_explain_mode_keeps_explanation_without_repeat(monkeypatch):
    b = make_bot(monkeypatch, FIELDS, ["Your taxable income ", "is Tk 6,40,000 after the exemption."])
    out = b.answer(Q, explain=True)
    assert out["text"] == HEADLINE + "\n\nYour taxable income is Tk 6,40,000 after the exemption."


@pytest.mark.parametrize("q, has_amount", [
    ("What is the tax-free limit for 2026-27?", False),
    ("I earn 80k", True),
    ("বেতন ১২ লাখ", True),
])
def test_amount_router(q, has_amount):
    assert bool(botmod.find_amounts(q)) == has_amount
