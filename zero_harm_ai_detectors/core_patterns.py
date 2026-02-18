"""
Shared patterns, validators, and unified result types for Zero Harm AI Detectors.

This module is the single source of truth for:
- All regex patterns (used by both tiers)
- Validators (Luhn, entropy)
- Redaction logic
- Detection and DetectionResult classes (unified across tiers)

Design: AI is ONLY used for hard-to-detect content:
- Person names (30% regex → 90% AI)
- Locations (AI only)
- Organizations (AI only)  
- Harmful content (improved accuracy with AI)

Regex is ALWAYS used for structured data (95%+ accuracy):
- Email, phone, SSN, credit card, secrets, etc.

File: zero_harm_ai_detectors/core_patterns.py
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# Redaction Strategies
# ============================================================

class RedactionStrategy(str, Enum):
    """How to redact detected sensitive information."""
    MASK_ALL = "mask_all"      # ****************
    MASK_LAST4 = "mask_last4"  # ************.com
    HASH = "hash"              # SHA-256 hex
    TOKEN = "token"            # [REDACTED_EMAIL]


def apply_redaction(value: str, strategy: RedactionStrategy, detection_type: str = "") -> str:
    """Apply a redaction strategy to a single matched value."""
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
    strategy: RedactionStrategy = RedactionStrategy.TOKEN,
) -> str:
    """Redact all matched spans in text (right-to-left to preserve indices)."""
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        start, end = s["start"], s["end"]
        det_type = s.get("type", "")
        text = text[:start] + apply_redaction(text[start:end], strategy, det_type) + text[end:]
    return text


# ============================================================
# Validators
# ============================================================

def luhn_check(number: str) -> bool:
    """Validate credit card using Luhn algorithm."""
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


def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy in bits per character."""
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


MIN_SECRET_ENTROPY: float = 4.5


# ============================================================
# Detection Types
# ============================================================

class DetectionType(str, Enum):
    """All detection types across both tiers."""
    # PII - Regex-based (95%+ accuracy, used by both tiers)
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    DOB = "DOB"
    DRIVERS_LICENSE = "DRIVERS_LICENSE"
    MEDICAL_RECORD_NUMBER = "MEDICAL_RECORD_NUMBER"
    ADDRESS = "ADDRESS"
    
    # PII - AI-enhanced (paid tier uses AI, free tier uses weak regex)
    PERSON = "PERSON"              # 30% regex → 90% AI
    LOCATION = "LOCATION"          # AI only (free tier skips)
    ORGANIZATION = "ORGANIZATION"  # AI only (free tier skips)
    
    # Secrets - Always regex (both tiers)
    API_KEY = "API_KEY"
    SECRET = "SECRET"
    
    # Harmful - AI-enhanced (paid tier uses transformer)
    HARMFUL_CONTENT = "HARMFUL_CONTENT"


# ============================================================
# Structured PII Patterns (always regex - 95%+ accuracy)
# ============================================================

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE_RE = re.compile(
    r"(?:(?:\+1[-.\s]?)?\(?(?:\d{3})\)?[-.\s]\d{3}[-.\s]\d{4})\b"
)

SSN_RE = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"
)

CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,19}\b")

BANK_ACCOUNT_KEYWORDS_RE = re.compile(r"\b(account|acct|routing|iban|aba|ach|bank)\b", re.I)
BANK_ACCOUNT_DIGITS_RE = re.compile(r"\b\d{6,17}\b")

DOB_PATTERNS = [
    re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19\d\d|20\d\d)\b"),
    re.compile(r"\b(19\d\d|20\d\d)-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b"),
    re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{1,2}),\s*(19\d\d|20\d\d)\b",
        re.I,
    ),
]

DL_KEYWORDS_RE = re.compile(
    r"\b(driver'?s?\s+license|dl\s*#?|dmv|lic#|license\s*no\.?|dlnum)\b", re.I
)
DL_TOKEN_RE = re.compile(r"\b[A-Z0-9]{6,12}\b", re.I)

MRN_KEYWORDS_RE = re.compile(r"\b(MRN|medical\s*record\s*(no\.?|number)?)\b", re.I)
MRN_DIGITS_RE = re.compile(r"\b[A-Z]?\d{6,12}[A-Z]?\b", re.I)

ADDRESS_STREET_RE = re.compile(
    r"\b\d{1,6}\s+(?:(?:N|S|E|W|NE|NW|SE|SW)\s+)?[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,3}\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|"
    r"Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Trl|Trail|Hwy|Highway)\b",
    re.IGNORECASE,
)
ADDRESS_POBOX_RE = re.compile(r"\bP\.?\s*O\.?\s*Box\s*\d+\b", re.IGNORECASE)

PERSON_NAME_RE = re.compile(
    r"(?:Name:\s*)?\b([A-Z][a-zA-Z]{1,}\s+[a-zA-Z]{1,}(?:\s+[a-zA-Z]{1,})?)\b"
)
PERSON_NAME_EXCLUDES = {
    "Street", "Avenue", "Road", "Company", "Corp", "LLC", "Inc",
    "Boulevard", "Lane", "Court", "Circle", "Way", "Place", "Highway",
    "North", "South", "East", "West",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}


# ============================================================
# Secrets Patterns (three-tier - always regex)
# ============================================================

STRUCTURED_SECRET_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bsk-[A-Za-z0-9]{32,64}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-org-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,100}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\bsk_(?:live|test)_[0-9a-zA-Z]{24,}\b"),
    re.compile(r"\bghp_[0-9A-Za-z]{36}\b"),
    re.compile(r"\bgithub_pat_[0-9A-Za-z_]{36,}\b"),
    re.compile(r"\bglpat-[0-9A-Za-z\-]{20,}\b"),
    re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
]

AWS_SECRET_KEY_RE = re.compile(r"(?<![0-9a-zA-Z/+])[0-9a-zA-Z/+]{40}(?![0-9a-zA-Z/+])")
GENERIC_SECRET_RE = re.compile(r"\b[A-Za-z0-9/+=_\-]{20,64}\b")

SECRET_CONTEXT_KEYWORDS_RE = re.compile(
    r"(secret[_\s-]?key|access[_\s-]?key|api[_\s-]?key|api[_\s-]?secret|"
    r"private[_\s-]?key|auth[_\s-]?token|bearer|credentials?|password|passwd|"
    r"aws[_\s-]?secret|client[_\s-]?secret|signing[_\s-]?key|"
    r"encryption[_\s-]?key|master[_\s-]?key|service[_\s-]?key|"
    r"secret|token|apikey|api_key)",
    re.IGNORECASE,
)
SECRET_CONTEXT_WINDOW: int = 80


def has_secret_context(text: str, start: int, end: int) -> bool:
    """Check if text near [start:end] contains secret keywords."""
    ctx_start = max(0, start - SECRET_CONTEXT_WINDOW)
    ctx_end = min(len(text), end + SECRET_CONTEXT_WINDOW)
    return bool(SECRET_CONTEXT_KEYWORDS_RE.search(text[ctx_start:ctx_end]))


def find_secrets(text: str) -> List[Dict[str, Any]]:
    """Run three-tier secrets scan."""
    findings: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(start: int, end: int, span: str) -> None:
        key = (start, end)
        if key not in seen:
            seen.add(key)
            findings.append({
                "span": span,
                "start": start,
                "end": end,
                "type": DetectionType.API_KEY.value,
                "confidence": 0.99,
            })

    for pat in STRUCTURED_SECRET_PATTERNS:
        for m in pat.finditer(text):
            _add(m.start(), m.end(), m.group())

    for m in AWS_SECRET_KEY_RE.finditer(text):
        if has_secret_context(text, m.start(), m.end()):
            if shannon_entropy(m.group()) >= MIN_SECRET_ENTROPY:
                _add(m.start(), m.end(), m.group())

    for m in GENERIC_SECRET_RE.finditer(text):
        span = m.group()
        if (m.start(), m.end()) in seen or len(span) < 20:
            continue
        if has_secret_context(text, m.start(), m.end()):
            if shannon_entropy(span) >= MIN_SECRET_ENTROPY:
                _add(m.start(), m.end(), span)

    return findings


# ============================================================
# Harmful Content Patterns (regex fallback)
# ============================================================

THREAT_CUES_RE = re.compile(
    r"\b(kill|hurt|stab|shoot|burn|bomb|beat|rape|destroy|attack|threaten|lynch)\b",
    re.IGNORECASE,
)

TOXIC_RE = re.compile(
    r'\b(fuck|shit|damn|hell|ass|bitch|bastard|crap|piss|whore|slut|'
    r'dickhead|asshole|dumbass|jackass|moron|idiot|stupid|'
    r'hate|hated|hates|hating|despise|despises|loathe|loathes)\b',
    re.IGNORECASE
)

INSULT_RE = re.compile(
    r'\b(stupid|idiot|moron|dumb|retard|loser|pathetic|worthless|'
    r'useless|garbage|trash|scum|vermin|disgusting|repulsive|'
    r'ugly|fat|pig|slob|freak|creep|weirdo|psycho|crazy|insane)\b',
    re.IGNORECASE
)

IDENTITY_HATE_RE = re.compile(
    r'\b(fag|faggot|dyke|tranny|nigger|nigga|chink|gook|spic|wetback|'
    r'kike|beaner|raghead|towelhead|terrorist|nazi|supremacist)\b',
    re.IGNORECASE
)

OBSCENE_RE = re.compile(
    r'\b(cock|dick|penis|pussy|vagina|cunt|tits|boobs|sex|porn|'
    r'masturbate|orgasm|fuck|screw|bang|hump|cum)\b',
    re.IGNORECASE
)


# ============================================================
# Unified Detection Result Classes
# ============================================================

@dataclass
class Detection:
    """A single detection result - same structure for both tiers."""
    type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "span": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class DetectionResult:
    """Unified result from detection - identical structure for both tiers."""
    original_text: str
    redacted_text: str
    detections: List[Detection]
    tier: str = "free"  # "free" or "paid"
    harmful: bool = False
    harmful_scores: Dict[str, float] = field(default_factory=dict)
    severity: str = "none"  # "none", "low", "medium", "high"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary format."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for det in self.detections:
            if det.type not in grouped:
                grouped[det.type] = []
            grouped[det.type].append(det.to_dict())
        
        return {
            "original": self.original_text,
            "redacted": self.redacted_text,
            "detections": grouped,
            "detection_count": len(self.detections),
            "tier": self.tier,
            "harmful": self.harmful,
            "harmful_scores": self.harmful_scores,
            "severity": self.severity,
        }
    
    def to_legacy_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to v0.1.x compatible format (just grouped detections)."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for det in self.detections:
            if det.type not in grouped:
                grouped[det.type] = []
            grouped[det.type].append({
                "span": det.text,
                "start": det.start,
                "end": det.end,
                "confidence": det.confidence,
            })
        return grouped
    
    def get_pii(self) -> List[Detection]:
        """Get only PII detections."""
        pii_types = {
            DetectionType.EMAIL.value, DetectionType.PHONE.value,
            DetectionType.SSN.value, DetectionType.CREDIT_CARD.value,
            DetectionType.BANK_ACCOUNT.value, DetectionType.DOB.value,
            DetectionType.DRIVERS_LICENSE.value, DetectionType.MEDICAL_RECORD_NUMBER.value,
            DetectionType.ADDRESS.value, DetectionType.PERSON.value,
            DetectionType.LOCATION.value, DetectionType.ORGANIZATION.value,
        }
        return [d for d in self.detections if d.type in pii_types]
    
    def get_secrets(self) -> List[Detection]:
        """Get only secret detections."""
        secret_types = {DetectionType.API_KEY.value, DetectionType.SECRET.value}
        return [d for d in self.detections if d.type in secret_types]
