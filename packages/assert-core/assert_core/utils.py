"""Utility functions shared across assert-core consumers."""

import re
from typing import Dict, Tuple

# PII pattern registry: label -> (compiled regex, replacement)
_PII_PATTERNS: Dict[str, Tuple[re.Pattern, str]] = {
    "email": (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
        "[EMAIL]",
    ),
    "phone": (
        # Matches +1 (555) 123-4567, 555-123-4567, (555)123-4567, 5551234567
        re.compile(
            r"(?:\+?1[-.\s]?)?"
            r"(?:\(?\d{3}\)?[-.\s]?)?"
            r"\d{3}[-.\s]?\d{4}\b"
        ),
        "[PHONE]",
    ),
    "ssn": (
        re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        "[SSN]",
    ),
    "credit_card": (
        # 13-19 digit sequences that look like card numbers (groups of 4)
        re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        ),
        "[CREDIT_CARD]",
    ),
    "ip_address": (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP_ADDRESS]",
    ),
}


def detect_and_mask_pii(text: str) -> str:
    """
    Detect and replace PII patterns in text with labelled placeholders.

    Detects and masks the following PII types (in order):
    - Email addresses       → [EMAIL]
    - Credit card numbers   → [CREDIT_CARD]
    - Social security numbers → [SSN]
    - Phone numbers         → [PHONE]
    - IPv4 addresses        → [IP_ADDRESS]

    Args:
        text: Input text that may contain PII.

    Returns:
        Text with all detected PII replaced by ``[TYPE]`` placeholders.

    Example::

        >>> detect_and_mask_pii("Contact alice@example.com or call 555-867-5309")
        'Contact [EMAIL] or call [PHONE]'
    """
    if not text:
        return text

    # Apply patterns in a fixed order so broader patterns don't clobber narrower ones.
    # credit_card before ssn avoids partial SSN matches inside card numbers.
    apply_order = ["email", "credit_card", "ssn", "phone", "ip_address"]
    result = text
    for key in apply_order:
        pattern, replacement = _PII_PATTERNS[key]
        result = pattern.sub(replacement, result)
    return result
