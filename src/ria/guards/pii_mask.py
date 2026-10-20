"""Regex masking of emails, street addresses and phone numbers in text.

This is the last line of defence for personal data: `graph.lifecycle.finalize` masks every reply, and `save_report` masks an
analysis before storing it. The primary protection is upstream (the SQL guard blocks personal columns), so these
patterns are tuned to avoid false positives on ordinary numbers such as IDs and SKUs. No dependencies on the rest of the package.
"""
from __future__ import annotations

import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A phone needs a separator, parentheses around the area code, or a leading +.
# Bare digit runs (IDs, SKUs) do not match.
PHONE = re.compile(
    r"(?<![\w.])(?:\(\d{3}\)[\s.-]?|\+\d{1,3}[\s.-]?\d{3}[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]?\d{4}(?![\w.])"
)
STREET_TYPES = (
    "Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
    "Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy"
)
# The street type must end the phrase (end of text, newline, |, or . , ; : )),
# or be followed by an Apt/Suite/Unit/# suffix. Otherwise "Best Way units" matches.
ADDRESS = re.compile(
    rf"\b\d{{1,6}}\s+(?:[A-Z][A-Za-z']+\s+){{1,4}}(?:{STREET_TYPES})\b"
    rf"(?:\s+(?:Apt|Suite|Unit|#)\s*\w+|(?=\s*(?:\Z|[\n|.,;:)]))\.?)"
)

_RULES = [(EMAIL, "[email]"), (ADDRESS, "[address]"), (PHONE, "[phone]")]


def mask_pii(text: str) -> tuple[str, int]:
    """Replace emails, addresses and phone numbers with `[email]`, `[address]` and `[phone]`; returns the masked text and the number of replacements."""
    total = 0
    for pattern, token in _RULES:
        text, n = pattern.subn(token, text)
        total += n
    return text, total
