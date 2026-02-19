"""
Core patterns, validators, and result types shared across detection modes.

This module contains:
- Regex patterns for structured data detection
- Validators (Luhn, entropy)
- Detection and DetectionResult dataclasses
- Redaction utilities

File: zero_harm_ai_detectors/core_patterns.py
"""
import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Detection Types
# ============================================================

class DetectionType(Enum):
    """Types of detectable content."""
    # PII
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    DOB = "DOB"
    DRIVERS_LICENSE = "DRIVERS_LICENSE"
    MEDICAL_RECORD_NUMBER = "MEDICAL_RECORD_NUMBER"
    ADDRESS = "ADDRESS"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    # Secrets
    API_KEY = "API_KEY"
    SECRET = "SECRET"
    # Harmful
    HARMFUL = "HARMFUL"


class RedactionStrategy(Enum):
    """Redaction strategies."""
    MASK_ALL = "mask_all"       # ********
    MASK_LAST4 = "mask_last4"   # ****1234
    HASH = "hash"               # [HASH:a1b2c3...]
    TOKEN = "token"             # [REDACTED_EMAIL]


# ============================================================
# Regex Patterns
# ============================================================

# Email pattern
EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Phone patterns (US formats)
PHONE_RE = re.compile(
    r'(?:\+?1[-.\s]?)?'  # Optional country code
    r'(?:'
    r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'  # (123) 456-7890 or 123-456-7890
    r'|'
    r'\d{3}[-.\s]\d{4}'  # 456-7890
    r')\b'
)

# SSN pattern
SSN_RE = re.compile(
    r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
)

# Credit card pattern (major cards)
CREDIT_CARD_RE = re.compile(
    r'\b(?:'
    r'4[0-9]{12}(?:[0-9]{3})?|'           # Visa
    r'5[1-5][0-9]{14}|'                    # Mastercard
    r'3[47][0-9]{13}|'                     # Amex
    r'6(?:011|5[0-9]{2})[0-9]{12}|'        # Discover
    r'(?:2131|1800|35\d{3})\d{11}'         # JCB
    r')\b'
    r'|'
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'  # Generic 16-digit with separators
)

# Bank account (basic patterns)
BANK_ACCOUNT_RE = re.compile(
    r'\b(?:'
    r'(?:account|acct)[-\s#:]*\d{8,17}'
    r'|'
    r'(?:routing|aba)[-\s#:]*\d{9}'
    r')\b',
    re.IGNORECASE
)

# Date of birth patterns
DOB_RE = re.compile(
    r'\b(?:'
    r'(?:0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}'  # MM/DD/YYYY
    r'|'
    r'(?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])'  # YYYY-MM-DD
    r'|'
    r'(?:0?[1-9]|[12]\d|3[01])[/\-](?:0?[1-9]|1[0-2])[/\-](?:19|20)\d{2}'  # DD/MM/YYYY
    r')\b'
)

# Driver's license (US state patterns - simplified)
DRIVERS_LICENSE_RE = re.compile(
    r'\b(?:'
    r'[A-Z]\d{7,8}'           # Common format: A1234567
    r'|'
    r'\d{1,3}[-\s]?\d{2,3}[-\s]?\d{4}'  # Numeric formats
    r')\b'
)

# Medical Record Number
MRN_RE = re.compile(
    r'\b(?:MRN|MR#|Medical Record)[-\s#:]*[A-Z0-9]{6,12}\b',
    re.IGNORECASE
)

# Address patterns (US)
ADDRESS_RE = re.compile(
    r'\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,4}'
    r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Place|Pl)'
    r'\.?\b',
    re.IGNORECASE
)

# Person name patterns (limited accuracy ~30-40%)
PERSON_NAME_RE = re.compile(
    r'\b(?:'
    r'(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
    r'|'
    r'[A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+'
    r')\b'
)

# ============================================================
# Secrets Patterns (Three-Tier Detection)
# ============================================================

# Tier 1: Structured prefixes (auto-detect)
STRUCTURED_SECRET_PREFIXES = [
    # OpenAI
    (r'sk-[A-Za-z0-9]{32,}', 'openai_api_key'),
    (r'sk-proj-[A-Za-z0-9\-_]{32,}', 'openai_project_key'),
    # AWS
    (r'AKIA[0-9A-Z]{16}', 'aws_access_key'),
    # GitHub
    (r'ghp_[A-Za-z0-9]{36,}', 'github_pat'),
    (r'gho_[A-Za-z0-9]{36,}', 'github_oauth'),
    (r'ghu_[A-Za-z0-9]{36,}', 'github_user'),
    (r'ghs_[A-Za-z0-9]{36,}', 'github_server'),
    (r'ghr_[A-Za-z0-9]{36,}', 'github_refresh'),
    # Stripe
    (r'sk_live_[A-Za-z0-9]{24,}', 'stripe_live_key'),
    (r'sk_test_[A-Za-z0-9]{24,}', 'stripe_test_key'),
    (r'rk_live_[A-Za-z0-9]{24,}', 'stripe_restricted'),
    # Slack
    (r'xox[baprs]-[A-Za-z0-9\-]{10,}', 'slack_token'),
    # Google
    (r'AIza[A-Za-z0-9\-_]{35}', 'google_api_key'),
    # Twilio
    (r'SK[a-f0-9]{32}', 'twilio_api_key'),
    # SendGrid
    (r'SG\.[A-Za-z0-9\-_]{22}\.[A-Za-z0-9\-_]{43}', 'sendgrid_api_key'),
    # npm
    (r'npm_[A-Za-z0-9]{36}', 'npm_token'),
    # PyPI
    (r'pypi-[A-Za-z0-9]{32,}', 'pypi_token'),
    # Anthropic
    (r'sk-ant-[A-Za-z0-9\-_]{32,}', 'anthropic_api_key'),
]

# Compile structured patterns
STRUCTURED_SECRET_PATTERNS = [
    (re.compile(r'\b' + pattern + r'\b'), name)
    for pattern, name in STRUCTURED_SECRET_PREFIXES
]

# Tier 2: AWS Secret Access Key (needs context + entropy)
AWS_SECRET_RE = re.compile(
    r'(?:aws[_\-]?secret[_\-]?(?:access[_\-]?)?key|'
    r'secret[_\-]?access[_\-]?key)\s*[=:]\s*["\']?'
    r'([A-Za-z0-9+/]{40})["\']?',
    re.IGNORECASE
)

# Tier 3: Generic secrets (needs context + entropy)
SECRET_CONTEXT_KEYWORDS = [
    'password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 'apikey',
    'api-key', 'auth', 'credential', 'private_key', 'privatekey', 'access_key',
    'accesskey', 'secret_key', 'secretkey', 'bearer', 'jwt',
]

GENERIC_SECRET_RE = re.compile(
    r'(?:' + '|'.join(SECRET_CONTEXT_KEYWORDS) + r')'
    r'\s*[=:]\s*["\']?'
    r'([A-Za-z0-9+/=\-_]{16,})["\']?',
    re.IGNORECASE
)

# ============================================================
# Harmful Content Patterns
# ============================================================

HARMFUL_PATTERNS = {
    'insult': re.compile(
        r'\b(?:stupid|idiot|moron|dumb|loser|pathetic|worthless)\b',
        re.IGNORECASE
    ),
    'threat': re.compile(
        r'\b(?:kill|murder|hurt|destroy|attack|beat)\s+(?:you|him|her|them)\b',
        re.IGNORECASE
    ),
    'hate': re.compile(
        r'\b(?:hate|despise|disgusting|revolting)\s+(?:you|them|all)\b',
        re.IGNORECASE
    ),
    'profanity': re.compile(
        r'\b(?:fuck|shit|damn|ass|bitch|bastard)\b',
        re.IGNORECASE
    ),
}


# ============================================================
# Validators
# ============================================================

def luhn_check(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    # Luhn algorithm
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    
    return checksum % 10 == 0


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        prob = count / length
        entropy -= prob * math.log2(prob)
    
    return entropy


# ============================================================
# Result Types
# ============================================================

@dataclass
class Detection:
    """Single detection result."""
    type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class DetectionResult:
    """Unified result from detection - identical structure for both modes."""
    original_text: str
    redacted_text: str
    detections: List[Detection]
    mode: str = "regex"  # "regex" or "ai"
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
            "mode": self.mode,
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


# ============================================================
# Redaction Utilities
# ============================================================

def apply_redaction(
    text: str,
    detection_type: str,
    strategy: RedactionStrategy = RedactionStrategy.TOKEN,
) -> str:
    """Apply redaction to a detected value."""
    if strategy == RedactionStrategy.MASK_ALL:
        return "*" * len(text)
    elif strategy == RedactionStrategy.MASK_LAST4:
        if len(text) > 4:
            return "*" * (len(text) - 4) + text[-4:]
        return "*" * len(text)
    elif strategy == RedactionStrategy.HASH:
        hash_val = hashlib.sha256(text.encode()).hexdigest()[:12]
        return f"[HASH:{hash_val}]"
    else:  # TOKEN
        return f"[REDACTED_{detection_type}]"


def redact_spans(
    text: str,
    detections: List[Detection],
    strategy: RedactionStrategy = RedactionStrategy.TOKEN,
) -> str:
    """Redact all detected spans in text."""
    if not detections:
        return text
    
    # Sort by start position (descending) to replace from end
    sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)
    
    result = text
    for det in sorted_detections:
        replacement = apply_redaction(det.text, det.type, strategy)
        result = result[:det.start] + replacement + result[det.end:]
    
    return result


def find_secrets(text: str) -> List[Dict[str, Any]]:
    """Find secrets using three-tier detection."""
    secrets = []
    
    # Tier 1: Structured prefixes
    for pattern, secret_type in STRUCTURED_SECRET_PATTERNS:
        for match in pattern.finditer(text):
            secrets.append({
                "span": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": secret_type,
                "method": "structured_prefix",
            })
    
    # Tier 2: AWS Secret Access Key
    for match in AWS_SECRET_RE.finditer(text):
        secret_value = match.group(1)
        if shannon_entropy(secret_value) > 4.0:
            secrets.append({
                "span": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": "aws_secret_key",
                "method": "context_entropy",
            })
    
    # Tier 3: Generic secrets with context
    for match in GENERIC_SECRET_RE.finditer(text):
        secret_value = match.group(1)
        if shannon_entropy(secret_value) > 3.5 and len(secret_value) >= 16:
            secrets.append({
                "span": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": "generic_secret",
                "method": "context_entropy",
            })
    
    return secrets
