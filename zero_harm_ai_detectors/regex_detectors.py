"""
Regex-based detection (mode='regex').

Fast pattern matching for structured data with 95%+ accuracy on:
- Email, phone, SSN, credit card, bank account
- Dates of birth, driver's license, medical record numbers
- Addresses, secrets/API keys
- Person names (~30-40% accuracy - use AI mode for better)

Every public function validates its input before processing.
Internal _*_raw helpers accept pre-validated text and skip re-validation
so that detect_all_regex() doesn't pay the validation cost N times.

File: zero_harm_ai_detectors/regex_detectors.py
"""
from typing import Any, Dict, List

from .core_patterns import (
    Detection,
    DetectionResult,
    DetectionType,
    RedactionStrategy,
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
    luhn_check,
    find_secrets,
    redact_spans,
)
from .input_validation import validate_input, REGEX_MODE_CONFIG


# ============================================================
# Public individual detectors
# Each validates input at entry before running any regex.
# ============================================================

def detect_emails(text: str) -> List[Detection]:
    """Detect email addresses. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_emails_raw(text)


def detect_phones(text: str) -> List[Detection]:
    """Detect phone numbers. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_phones_raw(text)


def detect_ssns(text: str) -> List[Detection]:
    """Detect Social Security Numbers. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_ssns_raw(text)


def detect_credit_cards(text: str) -> List[Detection]:
    """Detect credit card numbers with Luhn validation. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_credit_cards_raw(text)


def detect_bank_accounts(text: str) -> List[Detection]:
    """Detect bank account and routing numbers. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_bank_accounts_raw(text)


def detect_dob(text: str) -> List[Detection]:
    """Detect dates of birth. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_dob_raw(text)


def detect_drivers_licenses(text: str) -> List[Detection]:
    """Detect driver's license numbers. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_drivers_licenses_raw(text)


def detect_mrn(text: str) -> List[Detection]:
    """Detect medical record numbers. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_mrn_raw(text)


def detect_addresses(text: str) -> List[Detection]:
    """Detect street addresses. Validates input before processing."""
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_addresses_raw(text)


def detect_person_names_regex(text: str) -> List[Detection]:
    """
    Detect person names using regex (~30-40% accuracy).

    Validates input before processing.
    For better accuracy use AI mode: detect(text, mode='ai').
    """
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_person_names_raw(text)


def detect_secrets_regex(text: str) -> List[Detection]:
    """
    Detect secrets and API keys using three-tier detection.

    Validates input before processing.
    """
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_secrets_raw(text)


def detect_harmful_regex(text: str) -> Dict[str, Any]:
    """
    Detect harmful content using regex patterns.

    Validates input before processing.

    Multi-factor severity rules:
    - identity_hate or threat_phrases present → always 'high'
    - 2+ threat words or 6+ total matches   → 'high'
    - 1  threat word  or 4+ total matches   → 'medium'
    - 2+ obscene terms                       → 'medium'
    - Any other match                        → 'low'

    Returns:
        Dict with keys:
            harmful  (bool) — whether any harmful content was found
            severity (str)  — "none" | "low" | "medium" | "high"
            scores   (dict) — per-category normalised scores (0.0–1.0)
    """
    text = validate_input(text, REGEX_MODE_CONFIG)
    return _detect_harmful_raw(text)


# ============================================================
# Orchestrating function
# Validates once at entry, then calls raw helpers to avoid
# paying the validation cost for every individual detector.
# ============================================================

def detect_all_regex(
    text: str,
    detect_pii: bool = True,
    detect_secrets: bool = True,
    detect_harmful: bool = True,
    redaction_strategy: RedactionStrategy = RedactionStrategy.TOKEN,
) -> DetectionResult:
    """
    Run all regex-based detections in a single pass.

    Validates input once at entry; individual raw helpers called internally
    receive the already-validated string and skip re-validation.

    Args:
        text:               Input text to scan.
        detect_pii:         Whether to detect PII.
        detect_secrets:     Whether to detect secrets/API keys.
        detect_harmful:     Whether to detect harmful content.
        redaction_strategy: How to replace detected content.

    Returns:
        DetectionResult containing all findings.

    Raises:
        InputValidationError: If text is None or looks like binary data.
        InputTooLongError:    If text exceeds REGEX_MODE_CONFIG.max_length.
    """
    text = validate_input(text, REGEX_MODE_CONFIG)

    if not text:
        return DetectionResult(
            original_text="",
            redacted_text="",
            detections=[],
            mode="regex",
        )

    all_detections: List[Detection] = []

    if detect_pii:
        all_detections.extend(_detect_emails_raw(text))
        all_detections.extend(_detect_phones_raw(text))
        all_detections.extend(_detect_ssns_raw(text))
        all_detections.extend(_detect_credit_cards_raw(text))
        all_detections.extend(_detect_bank_accounts_raw(text))
        all_detections.extend(_detect_dob_raw(text))
        all_detections.extend(_detect_drivers_licenses_raw(text))
        all_detections.extend(_detect_mrn_raw(text))
        all_detections.extend(_detect_addresses_raw(text))
        all_detections.extend(_detect_person_names_raw(text))

    if detect_secrets:
        all_detections.extend(_detect_secrets_raw(text))

    is_harmful = False
    harmful_scores: Dict[str, float] = {}
    severity = "none"

    if detect_harmful:
        harmful_result = _detect_harmful_raw(text)
        is_harmful = harmful_result["harmful"]
        harmful_scores = harmful_result["scores"]
        severity = harmful_result["severity"]

    # Deduplicate by (start, end, type)
    seen: set = set()
    unique_detections: List[Detection] = []
    for det in all_detections:
        key = (det.start, det.end, det.type)
        if key not in seen:
            seen.add(key)
            unique_detections.append(det)

    return DetectionResult(
        original_text=text,
        redacted_text=redact_spans(text, unique_detections, redaction_strategy),
        detections=unique_detections,
        mode="regex",
        harmful=is_harmful,
        harmful_scores=harmful_scores,
        severity=severity,
    )


# ============================================================
# Private raw helpers
# Accept pre-validated text — called only by detect_all_regex()
# or by the public wrappers above after validation.
# Not exported in __all__.
# ============================================================

def _detect_emails_raw(text: str) -> List[Detection]:
    return [
        Detection(type=DetectionType.EMAIL.value, text=m.group(),
                  start=m.start(), end=m.end(), confidence=0.99,
                  metadata={"method": "regex"})
        for m in EMAIL_RE.finditer(text)
    ]


def _detect_phones_raw(text: str) -> List[Detection]:
    detections = []
    for m in PHONE_RE.finditer(text):
        phone = m.group()
        if len(''.join(c for c in phone if c.isdigit())) >= 7:
            detections.append(Detection(
                type=DetectionType.PHONE.value, text=phone,
                start=m.start(), end=m.end(), confidence=0.95,
                metadata={"method": "regex"},
            ))
    return detections


def _detect_ssns_raw(text: str) -> List[Detection]:
    detections = []
    for m in SSN_RE.finditer(text):
        ssn = m.group()
        digits = ''.join(c for c in ssn if c.isdigit())
        if digits[:3] not in ('000', '666') and not digits.startswith('9'):
            detections.append(Detection(
                type=DetectionType.SSN.value, text=ssn,
                start=m.start(), end=m.end(), confidence=0.98,
                metadata={"method": "regex"},
            ))
    return detections


def _detect_credit_cards_raw(text: str) -> List[Detection]:
    detections = []
    for m in CREDIT_CARD_RE.finditer(text):
        card = m.group()
        digits = ''.join(c for c in card if c.isdigit())
        if luhn_check(digits):
            detections.append(Detection(
                type=DetectionType.CREDIT_CARD.value, text=card,
                start=m.start(), end=m.end(), confidence=0.99,
                metadata={"method": "regex", "luhn_valid": True},
            ))
    return detections


def _detect_bank_accounts_raw(text: str) -> List[Detection]:
    return [
        Detection(type=DetectionType.BANK_ACCOUNT.value, text=m.group(),
                  start=m.start(), end=m.end(), confidence=0.90,
                  metadata={"method": "regex"})
        for m in BANK_ACCOUNT_RE.finditer(text)
    ]


def _detect_dob_raw(text: str) -> List[Detection]:
    return [
        Detection(type=DetectionType.DOB.value, text=m.group(),
                  start=m.start(), end=m.end(), confidence=0.85,
                  metadata={"method": "regex"})
        for m in DOB_RE.finditer(text)
    ]


def _detect_drivers_licenses_raw(text: str) -> List[Detection]:
    return [
        Detection(type=DetectionType.DRIVERS_LICENSE.value, text=m.group(),
                  start=m.start(), end=m.end(), confidence=0.75,
                  metadata={"method": "regex"})
        for m in DRIVERS_LICENSE_RE.finditer(text)
    ]


def _detect_mrn_raw(text: str) -> List[Detection]:
    return [
        Detection(type=DetectionType.MEDICAL_RECORD_NUMBER.value, text=m.group(),
                  start=m.start(), end=m.end(), confidence=0.90,
                  metadata={"method": "regex"})
        for m in MRN_RE.finditer(text)
    ]


def _detect_addresses_raw(text: str) -> List[Detection]:
    return [
        Detection(type=DetectionType.ADDRESS.value, text=m.group(),
                  start=m.start(), end=m.end(), confidence=0.85,
                  metadata={"method": "regex"})
        for m in ADDRESS_RE.finditer(text)
    ]


def _detect_person_names_raw(text: str) -> List[Detection]:
    return [
        Detection(type=DetectionType.PERSON.value, text=m.group(),
                  start=m.start(), end=m.end(), confidence=0.35,
                  metadata={"method": "regex"})
        for m in PERSON_NAME_RE.finditer(text)
    ]


def _detect_secrets_raw(text: str) -> List[Detection]:
    detections = []
    for secret in find_secrets(text):
        det_type = (
            DetectionType.API_KEY.value
            if "api" in secret["type"] or "key" in secret["type"]
            else DetectionType.SECRET.value
        )
        detections.append(Detection(
            type=det_type, text=secret["span"],
            start=secret["start"], end=secret["end"], confidence=0.95,
            metadata={
                "method": "regex",
                "secret_type": secret["type"],
                "detection_method": secret["method"],
            },
        ))
    return detections


def _detect_harmful_raw(text: str) -> Dict[str, Any]:
    counts: Dict[str, int] = {
        cat: len(pattern.findall(text))
        for cat, pattern in HARMFUL_PATTERNS.items()
    }
    total_matches = sum(counts.values())

    if total_matches == 0:
        return {"harmful": False, "severity": "none", "scores": {}}

    if counts.get("identity_hate", 0) > 0 or counts.get("threat_phrases", 0) > 0:
        severity = "high"
    elif counts.get("threat", 0) >= 2 or total_matches >= 6:
        severity = "high"
    elif counts.get("threat", 0) >= 1 or total_matches >= 4:
        severity = "medium"
    elif counts.get("obscene", 0) >= 2:
        severity = "medium"
    else:
        severity = "low"

    score_weights = {
        "toxic": 0.15, "threat": 0.20, "threat_phrases": 0.25,
        "insult": 0.15, "identity_hate": 0.30, "obscene": 0.15,
    }
    scores = {
        cat: min(1.0, 0.3 + count * score_weights.get(cat, 0.15))
        for cat, count in counts.items()
        if count > 0
    }
    return {"harmful": True, "severity": severity, "scores": scores}
