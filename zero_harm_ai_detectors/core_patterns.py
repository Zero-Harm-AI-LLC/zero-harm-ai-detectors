"""
Shared regex patterns, validators, and utilities for PII/secrets detection.

This module is the single source of truth for all structured-data patterns
used by both the regex (free) and AI (paid) detection tiers. Changes here
automatically propagate to both tiers.

Design principles:
    - One pattern definition per data type, used everywhere
    - Validators (Luhn, entropy) live next to the patterns that need them
    - Redaction logic is shared so both tiers produce identical output
    - No detector *classes* here — just patterns and pure functions

File: zero_harm_ai_detectors/core_patterns.py
"""
from __future__ import annotations

import hashlib
import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Redaction
# ============================================================

class RedactionStrategy(str, Enum):
    """How to redact detected sensitive information.

    Shared by both regex and AI tiers so that the same strategy
    enum can be used regardless of detection mode.
    """
    MASK_ALL = "mask_all"
    MASK_LAST4 = "mask_last4"
    HASH = "hash"
    TOKEN = "token"  # [REDACTED_TYPE] replacement tokens


def apply_redaction(value: str, strategy: RedactionStrategy, detection_type: str = "") -> str:
    """Apply a redaction strategy to a single matched value.

    Args:
        value: The original matched text (e.g. "john@example.com")
        strategy: Which redaction method to use
        detection_type: Used for TOKEN strategy (e.g. "EMAIL" → "[REDACTED_EMAIL]")

    Returns:
        The redacted replacement string
    """
    if strategy == RedactionStrategy.MASK_ALL:
        return "*" * len(value)

    if strategy == RedactionStrategy.MASK_LAST4:
        keep = 4 if len(value) >= 4 else len(value)
        return "*" * (len(value) - keep) + value[-keep:]

    if strategy == RedactionStrategy.HASH:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    if strategy == RedactionStrategy.TOKEN:
        tag = f"_{detection_type}" if detection_type else ""
        return f"[REDACTED{tag}]"

    return value


def redact_spans(
    text: str,
    spans: List[Dict[str, Any]],
    strategy: RedactionStrategy = RedactionStrategy.MASK_ALL,
) -> str:
    """Redact all matched spans in *text*.

    Works with the standard span format ``{"span": str, "start": int, "end": int}``.
    Also accepts an optional ``"type"`` key used by the TOKEN strategy.

    Spans may come from *any* detector — PII, secrets, or harmful-content.
    """
    # Process from right-to-left so earlier indices stay valid.
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        start, end = s["start"], s["end"]
        original = text[start:end]
        det_type = s.get("type", "")
        text = text[:start] + apply_redaction(original, strategy, det_type) + text[end:]
    return text


# ============================================================
# Luhn check (credit-card validation)
# ============================================================

def luhn_check(number: str) -> bool:
    """Validate a number string using the Luhn algorithm.

    Accepts strings with non-digit separators (dashes, spaces) which
    are stripped before validation.

    Returns ``True`` if the checksum passes **and** the digit count is
    in the valid range for payment cards (12–19 digits).
    """
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 12 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ============================================================
# Shannon entropy (for secrets detection)
# ============================================================

def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string in bits per character.

    Reference values:
        - English prose  ≈ 3.5 – 4.0 bits/char
        - Random base64  ≈ 5.5 – 6.0 bits/char
        - Random hex      ≈ 4.0 bits/char
    """
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


# Minimum entropy (bits/char) for a string to be considered "secret-like"
MIN_SECRET_ENTROPY: float = 4.5


# ============================================================
# Structured PII Patterns
# ============================================================
# Each pattern is a compiled regex.  Both tiers import these directly
# instead of defining their own copies.

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# Phone: requires at least one separator between groups to avoid
# matching continuous digit strings (e.g. credit-card fragments).
PHONE_RE = re.compile(
    r"(?:(?:\+1[-.\s]?)?\(?(?:\d{3})\)?[-.\s]\d{3}[-.\s]\d{4})\b"
)

SSN_RE = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"
)

CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,19}\b")


# ============================================================
# Secrets Patterns  (three-tier system)
# ============================================================
# Tier 1 — Structured prefixes: high confidence, no context needed
# Tier 2 — AWS secret keys: context + entropy required
# Tier 3 — Generic secrets: context + entropy + length required

# ── Tier 1: Structured secret prefixes ────────────────────────

STRUCTURED_SECRET_PATTERNS: List[re.Pattern] = [
    # OpenAI legacy/standard keys
    re.compile(r"\bsk-[A-Za-z0-9]{32,64}\b"),
    # OpenAI project-scoped keys
    re.compile(r"\bsk-proj-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),
    # OpenAI org-scoped keys
    re.compile(r"\bsk-org-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),
    # OpenAI legacy format with known middle segment
    re.compile(r"\bsk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b"),
    # AWS Access Key ID (always starts with AKIA or ASIA)
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # Google API Key (always starts with AIza)
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    # Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-, xoxs-)
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,100}\b"),
    # JWT (three base64url segments separated by dots)
    re.compile(
        r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
    ),
    # Stripe keys
    re.compile(r"\bsk_(?:live|test)_[0-9a-zA-Z]{24,}\b"),
    # GitHub Personal Access Token
    re.compile(r"\bghp_[0-9A-Za-z]{36}\b"),
    # GitHub Fine-grained PAT
    re.compile(r"\bgithub_pat_[0-9A-Za-z_]{36,}\b"),
    # GitLab tokens
    re.compile(r"\bglpat-[0-9A-Za-z\-]{20,}\b"),
    # npm tokens
    re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
]

# ── Tier 2: AWS Secret Access Key ─────────────────────────────
# Exactly 40 chars of base64-ish characters, but ONLY with context.
# Uses lookaround instead of \b since / and + break word boundaries.

AWS_SECRET_KEY_RE = re.compile(
    r"(?<![0-9a-zA-Z/+])[0-9a-zA-Z/+]{40}(?![0-9a-zA-Z/+])"
)

# ── Tier 3: Generic high-entropy string ────────────────────────
# 20-64 chars, only flagged when keyword context AND high entropy.

GENERIC_SECRET_RE = re.compile(r"\b[A-Za-z0-9/+=_\-]{20,64}\b")

# ── Context keywords (shared by Tier 2 & 3) ───────────────────

SECRET_CONTEXT_KEYWORDS_RE = re.compile(
    r"("
    r"secret[_\s-]?key|access[_\s-]?key|api[_\s-]?key|api[_\s-]?secret|"
    r"private[_\s-]?key|auth[_\s-]?token|bearer|credentials?|password|passwd|"
    r"aws[_\s-]?secret|client[_\s-]?secret|signing[_\s-]?key|"
    r"encryption[_\s-]?key|master[_\s-]?key|service[_\s-]?key|"
    r"secret|token|apikey|api_key"
    r")",
    re.IGNORECASE,
)

# How many characters before/after a match to search for keywords
SECRET_CONTEXT_WINDOW: int = 80


def has_secret_context(text: str, start: int, end: int) -> bool:
    """Return ``True`` if *text* near ``[start:end]`` contains secret keywords."""
    ctx_start = max(0, start - SECRET_CONTEXT_WINDOW)
    ctx_end = min(len(text), end + SECRET_CONTEXT_WINDOW)
    context = text[ctx_start:ctx_end]
    return bool(SECRET_CONTEXT_KEYWORDS_RE.search(context))


def find_secrets(text: str) -> List[Dict[str, Any]]:
    """Run the full three-tier secrets scan on *text*.

    Returns a list of ``{"span": str, "start": int, "end": int}`` dicts —
    the same format used by both the regex SecretsDetector and the AI
    pipeline's secrets component.
    """
    findings: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(start: int, end: int, span: str) -> None:
        key = (start, end)
        if key not in seen:
            seen.add(key)
            findings.append({"span": span, "start": start, "end": end})

    # Tier 1 — structured prefixes (high confidence)
    for pat in STRUCTURED_SECRET_PATTERNS:
        for m in pat.finditer(text):
            _add(m.start(), m.end(), m.group())

    # Tier 2 — AWS secret key (context + entropy)
    for m in AWS_SECRET_KEY_RE.finditer(text):
        if has_secret_context(text, m.start(), m.end()):
            if shannon_entropy(m.group()) >= MIN_SECRET_ENTROPY:
                _add(m.start(), m.end(), m.group())

    # Tier 3 — generic secrets (context + entropy + length)
    for m in GENERIC_SECRET_RE.finditer(text):
        span = m.group()
        if (m.start(), m.end()) in seen:
            continue
        if len(span) < 20:
            continue
        if has_secret_context(text, m.start(), m.end()):
            if shannon_entropy(span) >= MIN_SECRET_ENTROPY:
                _add(m.start(), m.end(), span)

    return findings


# ============================================================
# Harmful-content cue words (used by AI tier for score boosting)
# ============================================================

THREAT_CUES_RE = re.compile(
    r"\b(kill|hurt|stab|shoot|burn|bomb|beat|rape|destroy|attack|threaten|lynch)\b",
    re.IGNORECASE,
)
