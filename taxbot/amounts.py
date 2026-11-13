"""Money parsing and formatting in Bangladeshi style.

Small models are unreliable at arithmetic ("1.2 lakh" -> 12000 is a classic slip), so the
model only copies the amount text out of the question and this module converts it.
"""

from __future__ import annotations

import re

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_UNITS = {
    "k": 1e3, "thousand": 1e3, "হাজার": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "লাখ": 1e5, "লক্ষ": 1e5,
    "m": 1e6, "million": 1e6,
    "crore": 1e7, "crores": 1e7, "কোটি": 1e7,
}
_AMOUNT = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(?:(lakhs?|lacs?|crores?|thousand|million|k|m)(?![a-z])|(হাজার|লাখ|লক্ষ|কোটি))?"
)


def normalize(text: str) -> str:
    """Bangla digits -> ASCII, lower case, drop thousands separators."""
    return text.translate(_BN_DIGITS).lower().replace(",", "")


def parse_amount(text: str | None) -> float | None:
    """'1.2 lakh' -> 120000, '৫০ হাজার' -> 50000, 'Tk 80,000' -> 80000. None if no number."""
    if not text:
        return None
    m = _AMOUNT.search(normalize(str(text)))
    if not m:
        return None
    unit = m.group(2) or m.group(3)
    return float(m.group(1)) * (_UNITS[unit] if unit else 1)


_FISCAL_YEAR = re.compile(r"\b(?:19|20)\d\d\s*[-–/]\s*\d{2,4}\b")


def find_amounts(text: str) -> list[str]:
    """Every amount-looking phrase in the text, e.g. ['9 লাখ']. Years like 2026 or 2026-27 don't count."""
    text = _FISCAL_YEAR.sub(" ", normalize(text))
    return [m.group(0).strip() for m in _AMOUNT.finditer(text)
            if m.group(2) or m.group(3) or not 1990 <= float(m.group(1)) <= 2099]


def appears_in(amount_text: str | None, question: str) -> bool:
    """Guard against hallucination: the digits the model extracted must really be in the question."""
    if not amount_text:
        return False
    m = _AMOUNT.search(normalize(amount_text))
    return bool(m) and m.group(1) in normalize(question)


def tk(x: float) -> str:
    """12,00,000-style grouping (lakh/crore), e.g. tk(1200000) -> 'Tk 12,00,000'."""
    n = round(x)
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = [head[max(0, i - 2):i] for i in range(len(head), 0, -2)]
        s = ",".join(reversed(groups)) + "," + tail
    return ("-" if n < 0 else "") + "Tk " + s
