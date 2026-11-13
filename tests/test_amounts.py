import pytest

from taxbot.amounts import appears_in, parse_amount, tk
from taxbot.bot import calculate


@pytest.mark.parametrize("text, expected", [
    ("80k", 80_000),
    ("1.2 lakh", 120_000),
    ("Tk 9,00,000", 900_000),
    ("৫০ হাজার", 50_000),
    ("১২ লাখ", 1_200_000),
    ("50000 monthly", 50_000),   # 'm' of 'monthly' must not mean million
    ("2 crore", 20_000_000),
    ("50000tk", 50_000),
    ("", None),
    ("no number", None),
])
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


def test_tk_uses_lakh_grouping():
    assert tk(1_200_000) == "Tk 12,00,000"
    assert tk(45_000) == "Tk 45,000"
    assert tk(999) == "Tk 999"
    assert tk(25_000_000) == "Tk 2,50,00,000"


def test_appears_in_rejects_invented_numbers():
    q = "My salary is 80,000 per month"
    assert appears_in("80000", q)
    assert appears_in("80k", "I earn 80k")
    assert not appears_in("90000", q)
    assert appears_in("৮০ হাজার", "আমার বেতন ৮০ হাজার")


def _fields(**kw):
    base = {"wants_tax_calculation": True, "salary_text": "", "salary_period": "unknown", "bonus_text": "",
            "investment_text": "", "category": "general", "parent_of_disabled": False, "new_taxpayer": False}
    return {**base, **kw}


def test_calculate_monthly_with_bonus_and_investment():
    q = "Salary 1 lakh per month, bonus 2 lakh, invested 1.5 lakh in DPS"
    calc = calculate(_fields(salary_text="1 lakh", salary_period="monthly", bonus_text="2 lakh",
                             investment_text="1.5 lakh"), q)
    assert calc["result"]["gross"] == 1_400_000
    assert calc["input"]["eligible_investment"] == 150_000


def test_calculate_ignores_hallucinated_salary():
    assert calculate(_fields(salary_text="50000"), "How is income tax calculated?") is None


def test_calculate_falls_back_to_the_only_amount_in_question():
    q = "আমি একজন যুদ্ধাহত মুক্তিযোদ্ধা, আমার বার্ষিক বেতন ৯ লাখ টাকা। আমার কর কত?"
    calc = calculate(_fields(salary_text="বার্ষিক", salary_period="annual", category="freedom_fighter"), q)
    assert calc["result"]["gross"] == 900_000 and calc["result"]["net_tax"] == 5_000
    # two amounts in the question: too ambiguous to guess
    assert calculate(_fields(salary_text="monthly"), "Salary 80k, bonus 40k") is None


def test_calculate_drops_unsupported_flags():
    # qwen3:1.7b really did this: new_taxpayer=true for "My salary is 80k a month"
    calc = calculate(_fields(salary_text="51000", salary_period="monthly", new_taxpayer=True,
                             parent_of_disabled=True), "My salary is 51000 a month, tax?")
    assert not calc["input"]["new_taxpayer"] and not calc["input"]["parent_of_disabled"]
    calc = calculate(_fields(salary_text="51000", salary_period="monthly", new_taxpayer=True),
                     "First time taxpayer, salary 51000 a month")
    assert calc["input"]["new_taxpayer"] and calc["result"]["net_tax"] == 1_000


def test_calculate_guesses_period_and_says_so():
    calc = calculate(_fields(salary_text="80000"), "I get 80000, what's my tax")
    assert calc["result"]["gross"] == 960_000
    assert any("monthly" in a for a in calc["assumptions"])
