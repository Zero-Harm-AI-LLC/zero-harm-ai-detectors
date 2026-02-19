"""
Regex-based detection (mode='regex').

Fast pattern matching for structured data with 95%+ accuracy on:
- Email, phone, SSN, credit card, bank account
- Dates of birth, driver's license, medical record numbers
- Addresses, secrets/API keys
- Person names (~30-40% accuracy - use AI mode for better)

File: zero_harm_ai_detectors/regex_detectors.py
"""
from typing import Any, Dict, List, Optional

from .core_patterns import (
    Detection,
    DetectionResult,
    DetectionType,
    RedactionStrategy,
    # Patterns
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    BANK_ACCOUNT_RE,
    DOB_RE,
    DRIVERS_LICENSE_RE,
    MRN_RE,
    ADDRESS_RE,
    PERSON_NAME_RE,
    HARMFUL_PATTERNS,
    _SEVERITY_ORDER,
    _max_severity,
    # Functions
    luhn_check,
    find_secrets,
    redact_spans,
)


# ============================================================
# Individual Detectors
# ============================================================

def detect_emails(text: str) -> List[Detection]:
    """Detect email addresses."""
    detections = []
    for match in EMAIL_RE.finditer(text):
        detections.append(Detection(
            type=DetectionType.EMAIL.value,
            text=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.99,
            metadata={"method": "regex"},
        ))
    return detections


def detect_phones(text: str) -> List[Detection]:
    """Detect phone numbers."""
    detections = []
    for match in PHONE_RE.finditer(text):
        phone = match.group()
        # Filter out numbers that are too short
        digits = ''.join(c for c in phone if c.isdigit())
        if len(digits) >= 7:
            detections.append(Detection(
                type=DetectionType.PHONE.value,
                text=phone,
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                metadata={"method": "regex"},
            ))
    return detections


def detect_ssns(text: str) -> List[Detection]:
    """Detect Social Security Numbers."""
    detections = []
    for match in SSN_RE.finditer(text):
        ssn = match.group()
        digits = ''.join(c for c in ssn if c.isdigit())
        if digits[:3] not in ('000', '666') and not digits.startswith('9'):
            detections.append(Detection(
                type=DetectionType.SSN.value,
                text=ssn,
                start=match.start(),
                end=match.end(),
                confidence=0.98,
                metadata={"method": "regex"},
            ))
    return detections


def detect_credit_cards(text: str) -> List[Detection]:
    """Detect credit card numbers with Luhn validation."""
    detections = []
    for match in CREDIT_CARD_RE.finditer(text):
        card = match.group()
        digits = ''.join(c for c in card if c.isdigit())
        if luhn_check(digits):
            detections.append(Detection(
                type=DetectionType.CREDIT_CARD.value,
                text=card,
                start=match.start(),
                end=match.end(),
                confidence=0.99,
                metadata={"method": "regex", "luhn_valid": True},
            ))
    return detections


def detect_bank_accounts(text: str) -> List[Detection]:
    """Detect bank account and routing numbers."""
    detections = []
    for match in BANK_ACCOUNT_RE.finditer(text):
        detections.append(Detection(
            type=DetectionType.BANK_ACCOUNT.value,
            text=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.90,
            metadata={"method": "regex"},
        ))
    return detections


def detect_dob(text: str) -> List[Detection]:
    """Detect dates of birth."""
    detections = []
    for match in DOB_RE.finditer(text):
        detections.append(Detection(
            type=DetectionType.DOB.value,
            text=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.85,
            metadata={"method": "regex"},
        ))
    return detections


def detect_drivers_licenses(text: str) -> List[Detection]:
    """Detect driver's license numbers."""
    detections = []
    for match in DRIVERS_LICENSE_RE.finditer(text):
        detections.append(Detection(
            type=DetectionType.DRIVERS_LICENSE.value,
            text=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.75,
            metadata={"method": "regex"},
        ))
    return detections


def detect_mrn(text: str) -> List[Detection]:
    """Detect medical record numbers."""
    detections = []
    for match in MRN_RE.finditer(text):
        detections.append(Detection(
            type=DetectionType.MEDICAL_RECORD_NUMBER.value,
            text=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.90,
            metadata={"method": "regex"},
        ))
    return detections


def detect_addresses(text: str) -> List[Detection]:
    """Detect street addresses."""
    detections = []
    for match in ADDRESS_RE.finditer(text):
        detections.append(Detection(
            type=DetectionType.ADDRESS.value,
            text=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.85,
            metadata={"method": "regex"},
        ))
    return detections


def detect_person_names_regex(text: str) -> List[Detection]:
    """
    Detect person names using regex (~30-40% accuracy).

    For better accuracy, use AI mode (mode='ai').
    """
    detections = []
    for match in PERSON_NAME_RE.finditer(text):
        detections.append(Detection(
            type=DetectionType.PERSON.value,
            text=match.group(),
            start=match.start(),
            end=match.end(),
            confidence=0.35,  # Low confidence - regex isn't great for names
            metadata={"method": "regex"},
        ))
    return detections


def detect_secrets_regex(text: str) -> List[Detection]:
    """Detect secrets and API keys using three-tier detection."""
    detections = []
    secrets = find_secrets(text)

    for secret in secrets:
        det_type = (
            DetectionType.API_KEY.value
            if "api" in secret["type"] or "key" in secret["type"]
            else DetectionType.SECRET.value
        )
        detections.append(Detection(
            type=det_type,
            text=secret["span"],
            start=secret["start"],
            end=secret["end"],
            confidence=0.95,
            metadata={
                "method": "regex",
                "secret_type": secret["type"],
                "detection_method": secret["method"],
            },
        ))

    return detections


def detect_harmful_regex(text: str) -> Dict[str, Any]:
    """
    Detect harmful content using regex patterns.

    Uses multi-factor severity logic:
    - identity_hate or threat_phrases → always 'high'
    - Multiple threat words OR high total match count → 'high'
    - Any threat word or mid-range total → 'medium'
    - Low-count insults/toxic/obscene → 'low'

    Returns:
        Dict with:
            harmful (bool): whether any harmful content was found
            severity (str): "none" | "low" | "medium" | "high"
            scores (dict): per-category normalised scores (0.0–1.0)
    """
    counts: Dict[str, int] = {}
    for category, pattern in HARMFUL_PATTERNS.items():
        counts[category] = len(pattern.findall(text))

    total_matches = sum(counts.values())

    if total_matches == 0:
        return {"harmful": False, "severity": "none", "scores": {}}

    # --- Multi-factor severity determination ---
    severity = "low"  # default when something was found

    # Identity hate speech or explicit threat phrases are always high severity
    if counts.get("identity_hate", 0) > 0 or counts.get("threat_phrases", 0) > 0:
        severity = "high"

    # Multiple threat words or large total hit count → high
    elif counts.get("threat", 0) >= 2 or total_matches >= 6:
        severity = "high"

    # Any single threat word or moderate hit count → medium
    elif counts.get("threat", 0) >= 1 or total_matches >= 4:
        severity = "medium"

    # Multiple obscene terms → medium
    elif counts.get("obscene", 0) >= 2:
        severity = "medium"

    # Normalised scores (0.0–1.0) per category — only include non-zero categories
    scores: Dict[str, float] = {}
    score_weights = {
        "toxic":         0.15,
        "threat":        0.20,
        "threat_phrases": 0.25,
        "insult":        0.15,
        "identity_hate": 0.30,
        "obscene":       0.15,
    }
    for category, count in counts.items():
        if count > 0:
            weight = score_weights.get(category, 0.15)
            scores[category] = min(1.0, 0.3 + count * weight)

    return {
        "harmful": True,
        "severity": severity,
        "scores": scores,
    }


# ============================================================
# Main Detection Function
# ============================================================

def detect_all_regex(
    text: str,
    detect_pii: bool = True,
    detect_secrets: bool = True,
    detect_harmful: bool = True,
    redaction_strategy: RedactionStrategy = RedactionStrategy.TOKEN,
) -> DetectionResult:
    """
    Run all regex-based detections.

    Args:
        text: Input text to scan
        detect_pii: Whether to detect PII
        detect_secrets: Whether to detect secrets
        detect_harmful: Whether to detect harmful content
        redaction_strategy: How to redact detected content

    Returns:
        DetectionResult with all findings
    """
    if not text:
        return DetectionResult(
            original_text="",
            redacted_text="",
            detections=[],
            mode="regex",
        )

    all_detections: List[Detection] = []

    # PII detection
    if detect_pii:
        all_detections.extend(detect_emails(text))
        all_detections.extend(detect_phones(text))
        all_detections.extend(detect_ssns(text))
        all_detections.extend(detect_credit_cards(text))
        all_detections.extend(detect_bank_accounts(text))
        all_detections.extend(detect_dob(text))
        all_detections.extend(detect_drivers_licenses(text))
        all_detections.extend(detect_mrn(text))
        all_detections.extend(detect_addresses(text))
        all_detections.extend(detect_person_names_regex(text))

    # Secrets detection
    if detect_secrets:
        all_detections.extend(detect_secrets_regex(text))

    # Harmful content detection
    is_harmful = False
    harmful_scores: Dict[str, float] = {}
    severity = "none"

    if detect_harmful:
        harmful_result = detect_harmful_regex(text)
        is_harmful = harmful_result["harmful"]
        harmful_scores = harmful_result["scores"]
        severity = harmful_result["severity"]

    # Remove duplicates (same span)
    seen = set()
    unique_detections = []
    for det in all_detections:
        key = (det.start, det.end, det.type)
        if key not in seen:
            seen.add(key)
            unique_detections.append(det)

    # Redact
    redacted = redact_spans(text, unique_detections, redaction_strategy)

    return DetectionResult(
        original_text=text,
        redacted_text=redacted,
        detections=unique_detections,
        mode="regex",
        harmful=is_harmful,
        harmful_scores=harmful_scores,
        severity=severity,
    )



