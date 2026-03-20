"""PII detection and redaction filter.

Detects and optionally redacts:
- Email addresses
- Public IP addresses
- Chinese phone numbers
- Chinese ID card numbers
- Bank card numbers
"""

import re
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

# Email pattern
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# IPv4 public IP (exclude private ranges and loopback)
_IP_RE = re.compile(
    r"\b(?!10\.)(?!127\.)(?!172\.(?:1[6-9]|2\d|3[01])\.)(?!192\.168\.)"
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}\b"
)

# Chinese phone number: 1[3-9]XXXXXXXXX
_CN_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# Chinese ID card: 18 digits (last may be X)
_CN_ID_RE = re.compile(
    r"(?<!\d)[1-9]\d{5}"  # region code
    r"(?:19|20)\d{2}"     # year
    r"(?:0[1-9]|1[0-2])"  # month
    r"(?:0[1-9]|[12]\d|3[01])"  # day
    r"\d{3}[\dXx](?!\d)"  # sequence + check digit
)

# Bank card number: 16-19 digits
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")

_PII_PATTERNS = [
    ("email", _EMAIL_RE, "email@example.com"),
    ("ip", _IP_RE, "0.0.0.0"),
    ("cn_phone", _CN_PHONE_RE, "1XXXXXXXXXX"),
    ("cn_id", _CN_ID_RE, "XXXXXXXXXXXXXXXXXX"),
    ("bank_card", _BANK_CARD_RE, "XXXXXXXXXXXXXXXX"),
]


@register_filter("pii")
class PIIFilter(BaseFilter):
    """PII detection and redaction filter.

    Modes:
        - detect: Report PII found, but keep the document as-is.
        - redact: Replace PII with placeholder values.
    """

    def __init__(
        self,
        text_field: str = "text",
        mode: str = "redact",
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        if mode not in ("detect", "redact"):
            raise ValueError(f"PII mode must be 'detect' or 'redact', got {mode!r}")
        self.mode = mode

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        found: dict[str, int] = {}

        for pii_type, pattern, replacement in _PII_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = len(matches)
                if self.mode == "redact":
                    text = pattern.sub(replacement, text)

        if self.mode == "redact" and found:
            doc[self.text_field] = text

        # In detect mode, we always keep the doc but report findings
        # In redact mode, we keep the doc with redacted text
        info = {"filter": self.name, "pii_found": found} if found else {}
        return True, info
