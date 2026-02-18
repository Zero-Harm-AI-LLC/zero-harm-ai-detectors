"""
Regex-based detection for the FREE tier.

This module provides fast, pattern-based detection for:
- Structured PII (email, phone, SSN, credit card, etc.) - 95%+ accuracy
- Secrets/API keys (three-tier detection) - 95%+ accuracy
- Person names (regex fallback) - 30-40% accuracy
- Harmful content (regex fallback) - basic detection

For improved person name and harmful content detection, use paid tier (AI).

File: zero_harm_ai_detectors/regex_detectors.py
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .input_validation import validate_input
from .core_patterns import (
    # Redaction
    RedactionStrategy,
    apply_redaction,
    redact_spans,
    # Validators
    luhn_check,
    # Detection types
    DetectionType,
    Detection,
    DetectionResult,
    # PII patterns
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    BANK_ACCOUNT_KEYWORDS_RE,
    BANK_ACCOUNT_DIGITS_RE,
    DOB_PATTERNS,
    DL_KEYWORDS_RE,
    DL_TOKEN_RE,
    MRN_KEYWORDS_RE,
    MRN_DIGITS_RE,
    ADDRESS_STREET_RE,
    ADDRESS_POBOX_RE,
    PERSON_NAME_RE,
    PERSON_NAME_EXCLUDES,
    # Secrets
    find_secrets,
    # Harmful patterns
    THREAT_CUES_RE,
    TOXIC_RE,
    INSULT_RE,
    IDENTITY_HATE_RE,
    OBSCENE_RE,
)


CONTEXT_WINDOW = 30


def _get_context(text: str, start: int, end: int, window: int = CONTEXT_WINDOW) -> str:
    """Get surrounding context for a match."""
    return text[max(0, start - window):min(len(text), end + window)]


# ============================================================
# PII Detection Functions
# ============================================================

def detect_emails(text: str) -> List[Detection]:
    """Detect email addresses."""
    return [
        Detection(
            type=DetectionType.EMAIL.value,
            text=m.group(),
            start=m.start(),
            end=m.end(),
            confidence=1.0,
            metadata={"method": "regex"},
        )
        for m in EMAIL_RE.finditer(text)
    ]


def detect_phones(text: str) -> List[Detection]:
    """Detect phone numbers."""
    return [
        Detection(
            type=DetectionType.PHONE.value,
            text=m.group(),
            start=m.start(),
            end=m.end(),
            confidence=1.0,
            metadata={"method": "regex"},
        )
        for m in PHONE_RE.finditer(text)
    ]


def detect_ssns(text: str) -> List[Detection]:
    """Detect Social Security Numbers."""
    return [
        Detection(
            type=DetectionType.SSN.value,
            text=m.group(),
            start=m.start(),
            end=m.end(),
            confidence=1.0,
            metadata={"method": "regex"},
        )
        for m in SSN_RE.finditer(text)
    ]


def detect_credit_cards(text: str) -> List[Detection]:
    """Detect credit card numbers (with Luhn validation)."""
    results = []
    for m in CREDIT_CARD_RE.finditer(text):
        raw = m.group()
        digits_only = re.sub(r"\D", "", raw)
        if luhn_check(digits_only):
            results.append(Detection(
                type=DetectionType.CREDIT_CARD.value,
                text=raw,
                start=m.start(),
                end=m.end(),
                confidence=0.95,
                metadata={"method": "regex+luhn"},
            ))
    return results


def detect_bank_accounts(text: str) -> List[Detection]:
    """Detect bank account numbers (context-dependent)."""
    results = []
    for m in BANK_ACCOUNT_DIGITS_RE.finditer(text):
        ctx = _get_context(text, m.start(), m.end())
        if BANK_ACCOUNT_KEYWORDS_RE.search(ctx):
            # Exclude if it matches SSN pattern
            if not re.fullmatch(r"\d{9}", m.group()):
                results.append(Detection(
                    type=DetectionType.BANK_ACCOUNT.value,
                    text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.85,
                    metadata={"method": "regex+context"},
                ))
    return results


def detect_dob(text: str) -> List[Detection]:
    """Detect dates of birth."""
    results = []
    for pat in DOB_PATTERNS:
        for m in pat.finditer(text):
            results.append(Detection(
                type=DetectionType.DOB.value,
                text=m.group(),
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                metadata={"method": "regex"},
            ))
    return results


def detect_drivers_licenses(text: str) -> List[Detection]:
    """Detect driver's license numbers (context-dependent)."""
    results = []
    ca_re = re.compile(r"\b[A-Z]\d{7}\b")
    ny_re = re.compile(r"\b([A-Z]\d{7}|\d{9}|\d{8})\b")
    tx_re = re.compile(r"\b\d{8}\b")
    
    for m in DL_TOKEN_RE.finditer(text):
        ctx = _get_context(text, m.start(), m.end())
        token = m.group()
        
        if (DL_KEYWORDS_RE.search(ctx) or 
            ca_re.fullmatch(token) or 
            ny_re.fullmatch(token) or
            tx_re.fullmatch(token)):
            results.append(Detection(
                type=DetectionType.DRIVERS_LICENSE.value,
                text=token,
                start=m.start(),
                end=m.end(),
                confidence=0.80,
                metadata={"method": "regex+context"},
            ))
    return results


def detect_mrn(text: str) -> List[Detection]:
    """Detect medical record numbers (context-dependent)."""
    results = []
    for m in MRN_DIGITS_RE.finditer(text):
        ctx = _get_context(text, m.start(), m.end())
        if MRN_KEYWORDS_RE.search(ctx):
            results.append(Detection(
                type=DetectionType.MEDICAL_RECORD_NUMBER.value,
                text=m.group(),
                start=m.start(),
                end=m.end(),
                confidence=0.85,
                metadata={"method": "regex+context"},
            ))
    return results


def detect_addresses(text: str) -> List[Detection]:
    """Detect street addresses and P.O. boxes."""
    results = []
    
    for m in ADDRESS_STREET_RE.finditer(text):
        results.append(Detection(
            type=DetectionType.ADDRESS.value,
            text=m.group(),
            start=m.start(),
            end=m.end(),
            confidence=0.85,
            metadata={"method": "regex"},
        ))
    
    for m in ADDRESS_POBOX_RE.finditer(text):
        results.append(Detection(
            type=DetectionType.ADDRESS.value,
            text=m.group(),
            start=m.start(),
            end=m.end(),
            confidence=0.90,
            metadata={"method": "regex"},
        ))
    
    return results


def detect_person_names_regex(text: str) -> List[Detection]:
    """
    Detect person names using regex (30-40% accuracy).
    
    Note: For better accuracy (85-95%), use the paid tier with AI.
    """
    results = []
    
    for m in PERSON_NAME_RE.finditer(text):
        if m.lastindex and m.lastindex >= 1:
            value = m.group(1).strip()
            start, end = m.start(1), m.end(1)
        else:
            value = m.group().strip()
            start, end = m.start(), m.end()
        
        # Filter out false positives
        words = value.split()
        if len(words) >= 2 and words[-1] in PERSON_NAME_EXCLUDES:
            continue
        
        ctx = _get_context(text, start, end, window=20)
        
        # Skip if looks like address
        if re.search(r"\b\d{3,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd)\b", ctx, re.I):
            continue
        
        # Skip if looks like email context
        if re.search(r"@\w+\.\w+|email\s*:|contact\s*information", ctx, re.I):
            continue
        
        results.append(Detection(
            type=DetectionType.PERSON.value,
            text=value,
            start=start,
            end=end,
            confidence=0.35,  # Low confidence for regex-based name detection
            metadata={"method": "regex", "accuracy_note": "Use paid tier for 85-95% accuracy"},
        ))
    
    return results


# ============================================================
# Secrets Detection
# ============================================================

def detect_secrets_regex(text: str) -> List[Detection]:
    """Detect secrets/API keys using three-tier regex scan."""
    findings = find_secrets(text)
    return [
        Detection(
            type=finding.get("type", DetectionType.API_KEY.value),
            text=finding["span"],
            start=finding["start"],
            end=finding["end"],
            confidence=finding.get("confidence", 0.99),
            metadata={"method": "regex"},
        )
        for finding in findings
    ]


# ============================================================
# Harmful Content Detection (Regex Fallback)
# ============================================================

def detect_harmful_regex(text: str) -> tuple[bool, Dict[str, float], str, List[str]]:
    """
    Detect harmful content using regex patterns.
    
    Returns: (is_harmful, scores, severity, active_labels)
    
    Note: For more accurate detection, use the paid tier with transformer models.
    """
    toxic_count = len(TOXIC_RE.findall(text))
    threat_count = len(THREAT_CUES_RE.findall(text))
    insult_count = len(INSULT_RE.findall(text))
    hate_count = len(IDENTITY_HATE_RE.findall(text))
    obscene_count = len(OBSCENE_RE.findall(text))
    
    total = toxic_count + threat_count + insult_count + hate_count + obscene_count
    
    if total == 0:
        return False, {}, "none", []
    
    # Generate scores (heuristic)
    scores = {
        "toxic": min(1.0, 0.3 + toxic_count * 0.15),
        "threat": min(1.0, 0.4 + threat_count * 0.2),
        "insult": min(1.0, 0.3 + insult_count * 0.15),
        "identity_hate": min(1.0, 0.5 + hate_count * 0.3),
        "obscene": min(1.0, 0.3 + obscene_count * 0.15),
    }
    
    # Calculate severity
    if hate_count > 0 or threat_count >= 2 or total >= 5:
        severity = "high"
    elif threat_count >= 1 or total >= 3 or obscene_count >= 2:
        severity = "medium"
    else:
        severity = "low"
    
    # Active labels
    active_labels = []
    if toxic_count > 0:
        active_labels.append("toxic")
    if threat_count > 0:
        active_labels.append("threat")
    if insult_count > 0:
        active_labels.append("insult")
    if hate_count > 0:
        active_labels.append("identity_hate")
    if obscene_count > 0:
        active_labels.append("obscene")
    
    return True, scores, severity, active_labels


# ============================================================
# Combined Detection Function (Free Tier)
# ============================================================

def detect_all_regex(
    text: str,
    detect_pii: bool = True,
    detect_secrets: bool = True,
    detect_harmful: bool = True,
    detect_names: bool = True,
    redaction_strategy: RedactionStrategy = RedactionStrategy.TOKEN,
) -> DetectionResult:
    """
    Run all regex-based detection (FREE tier).
    
    Args:
        text: Input text to scan
        detect_pii: Whether to detect PII
        detect_secrets: Whether to detect secrets/API keys
        detect_harmful: Whether to detect harmful content
        detect_names: Whether to attempt person name detection (low accuracy)
        redaction_strategy: How to redact detected content
    
    Returns:
        DetectionResult with all findings
    """
    text = validate_input(text, mode="regex")
    
    if not text:
        return DetectionResult(
            original_text="",
            redacted_text="",
            detections=[],
            tier="free",
        )
    
    all_detections: List[Detection] = []
    
    # PII Detection
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
        
        if detect_names:
            all_detections.extend(detect_person_names_regex(text))
    
    # Secrets Detection
    if detect_secrets:
        all_detections.extend(detect_secrets_regex(text))
    
    # Harmful Content Detection
    is_harmful = False
    harmful_scores: Dict[str, float] = {}
    severity = "none"
    
    if detect_harmful:
        is_harmful, harmful_scores, severity, active_labels = detect_harmful_regex(text)
        if is_harmful:
            all_detections.append(Detection(
                type=DetectionType.HARMFUL_CONTENT.value,
                text=text,
                start=0,
                end=len(text),
                confidence=max(harmful_scores.values()) if harmful_scores else 0.5,
                metadata={
                    "method": "regex",
                    "labels": active_labels,
                    "scores": harmful_scores,
                },
            ))
    
    # Remove overlapping detections
    all_detections = _remove_overlaps(all_detections)
    
    # Redact text
    redacted = _redact_detections(text, all_detections, redaction_strategy)
    
    return DetectionResult(
        original_text=text,
        redacted_text=redacted,
        detections=all_detections,
        tier="free",
        harmful=is_harmful,
        harmful_scores=harmful_scores,
        severity=severity,
    )


def _remove_overlaps(detections: List[Detection]) -> List[Detection]:
    """Remove overlapping detections, keeping higher confidence."""
    if not detections:
        return []
    
    sorted_dets = sorted(detections, key=lambda x: (x.start, -x.confidence))
    result: List[Detection] = []
    
    for det in sorted_dets:
        overlaps = False
        for existing in result:
            if not (det.end <= existing.start or det.start >= existing.end):
                overlaps = True
                if det.confidence > existing.confidence:
                    result.remove(existing)
                    result.append(det)
                break
        if not overlaps:
            result.append(det)
    
    return sorted(result, key=lambda x: x.start)


def _redact_detections(
    text: str,
    detections: List[Detection],
    strategy: RedactionStrategy,
) -> str:
    """Redact all detections from text."""
    sorted_dets = sorted(detections, key=lambda x: x.start, reverse=True)
    
    result = text
    for det in sorted_dets:
        if det.type == DetectionType.HARMFUL_CONTENT.value:
            continue
        replacement = apply_redaction(det.text, strategy, det.type)
        result = result[:det.start] + replacement + result[det.end:]
    
    return result


# ============================================================
# Legacy API Compatibility
# ============================================================

def detect_pii(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Legacy API: Detect PII and return grouped dictionary.
    
    Returns: Dict mapping type -> list of span dicts
    """
    result = detect_all_regex(text, detect_secrets=False, detect_harmful=False)
    return result.to_legacy_dict()


def detect_secrets(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Legacy API: Detect secrets and return grouped dictionary.
    
    Returns: Dict with 'SECRETS' key if found
    """
    text = validate_input(text, mode="regex")
    detections = detect_secrets_regex(text)
    
    if detections:
        return {
            "SECRETS": [
                {"span": d.text, "start": d.start, "end": d.end, "confidence": d.confidence}
                for d in detections
            ]
        }
    return {}


def detect_harmful(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Legacy API: Detect harmful content and return grouped dictionary.
    """
    text = validate_input(text, mode="regex")
    is_harmful, scores, severity, labels = detect_harmful_regex(text)
    
    if is_harmful:
        return {
            "HARMFUL_CONTENT": [{
                "span": text,
                "start": 0,
                "end": len(text),
                "severity": severity,
                "labels": labels,
                "scores": scores,
            }]
        }
    return {}


def redact_text(
    text: str,
    detections: Dict[str, List[Dict[str, Any]]],
    strategy: str = "token",
) -> str:
    """
    Legacy API: Redact detected spans from text.
    """
    try:
        strat = RedactionStrategy(strategy)
    except ValueError:
        strat = RedactionStrategy.TOKEN
    
    spans = []
    for det_type, span_list in detections.items():
        for span in span_list:
            spans.append({
                "start": span["start"],
                "end": span["end"],
                "type": det_type,
            })
    
    return redact_spans(text, spans, strat)
