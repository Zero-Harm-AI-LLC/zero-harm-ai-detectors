"""
Zero Harm AI Detectors - Privacy & Content Safety Detection Library

Two detection modes:
- REGEX mode: Fast pattern-based detection (1-5ms), 95%+ accuracy on structured data
- AI mode: Transformer-based detection (50-200ms), enhanced accuracy for names/locations/orgs

Usage:
    from zero_harm_ai_detectors import detect
    
    # Regex mode (default) - fast, great for structured data
    result = detect("Email: john@example.com")
    
    # AI mode - better for names, locations, organizations
    result = detect("Contact John Smith at Microsoft", mode="ai")
    
    # Both return identical DetectionResult format!

File: zero_harm_ai_detectors/__init__.py
"""
from typing import Any, Dict, List, Optional

# ============================================================
# Core Exports (always available)
# ============================================================

from .core_patterns import (
    # Result types (unified across modes)
    Detection,
    DetectionResult,
    DetectionType,
    # Redaction
    RedactionStrategy,
    apply_redaction,
    redact_spans,
    # Validators
    luhn_check,
    shannon_entropy,
    # Pattern exports (for advanced users)
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    find_secrets,
)

from .input_validation import (
    validate_input,
    validate_input_soft,
    InputConfig,
    InputValidationError,
    InputTooLongError,
    REGEX_MODE_CONFIG,
    AI_MODE_CONFIG,
    API_MODE_CONFIG,
)

# ============================================================
# Regex Mode (always available)
# ============================================================

from .regex_detectors import (
    # Main function
    detect_all_regex,
    # Individual detectors
    detect_emails,
    detect_phones,
    detect_ssns,
    detect_credit_cards,
    detect_bank_accounts,
    detect_dob,
    detect_drivers_licenses,
    detect_mrn,
    detect_addresses,
    detect_person_names_regex,
    detect_secrets_regex,
    detect_harmful_regex,
    # Legacy API
    detect_pii as detect_pii_legacy,
    detect_secrets as detect_secrets_legacy,
    detect_harmful as detect_harmful_legacy,
    redact_text,
)

# ============================================================
# AI Mode (optional - requires transformers/torch)
# ============================================================

try:
    from .ai_detectors import (
        AI_AVAILABLE,
        check_ai_available,
        AIConfig,
        AIPipeline,
        NERDetector,
        HarmfulContentDetector,
        detect_all_ai,
        get_pipeline,
    )
except ImportError:
    AI_AVAILABLE = False
    check_ai_available = lambda: False
    AIConfig = None
    AIPipeline = None
    NERDetector = None
    HarmfulContentDetector = None
    detect_all_ai = None
    get_pipeline = None


# ============================================================
# Unified API (main entry point)
# ============================================================

def detect(
    text: str,
    mode: str = "regex",
    detect_pii: bool = True,
    detect_secrets: bool = True,
    detect_harmful: bool = True,
    redaction_strategy: str = "token",
    ai_config: Optional["AIConfig"] = None,
) -> DetectionResult:
    """
    Unified detection function - main entry point for the library.
    
    Args:
        text: Input text to scan
        mode: "regex" (fast, pattern-based) or "ai" (slower, transformer-based)
        detect_pii: Whether to detect PII
        detect_secrets: Whether to detect secrets/API keys
        detect_harmful: Whether to detect harmful content
        redaction_strategy: "token", "mask_all", "mask_last4", or "hash"
        ai_config: Optional AIConfig for AI mode customization
    
    Returns:
        DetectionResult with identical structure for both modes:
        - original_text: Original input
        - redacted_text: Text with sensitive content redacted
        - detections: List of Detection objects
        - mode: "regex" or "ai"
        - harmful: Boolean if harmful content detected
        - harmful_scores: Dict of harm category scores
        - severity: "none", "low", "medium", or "high"
    
    Mode comparison:
        | Feature          | regex        | ai           |
        |------------------|--------------|--------------|
        | Speed            | 1-5ms        | 50-200ms     |
        | Email/Phone/SSN  | 95%+ ✓       | 95%+ ✓       |
        | Person Names     | ~30%         | ~90% ✓       |
        | Locations        | ❌           | ~85% ✓       |
        | Organizations    | ❌           | ~80% ✓       |
        | Dependencies     | regex only   | transformers |
    
    Example:
        # Regex mode (default) - fast, pattern-based
        result = detect("Email: john@example.com")
        print(result.detections)  # [Detection(EMAIL, ...)]
        
        # AI mode - enhanced for names/locations/orgs
        result = detect("Contact John Smith at Microsoft", mode="ai")
        print(result.detections)  # [Detection(PERSON, ...), Detection(ORGANIZATION, ...)]
    
    Raises:
        ValueError: If mode is not "regex" or "ai"
        ImportError: If mode="ai" but AI dependencies not installed
    """
    if mode not in ("regex", "ai"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'regex' or 'ai'.")
    
    try:
        strategy = RedactionStrategy(redaction_strategy)
    except ValueError:
        strategy = RedactionStrategy.TOKEN
    
    if mode == "ai":
        if not AI_AVAILABLE:
            raise ImportError(
                "AI mode requires additional dependencies.\n"
                "Install with: pip install 'zero_harm_ai_detectors[ai]'\n"
                "Or use regex mode: detect(text, mode='regex')"
            )
        
        if ai_config:
            pipeline = AIPipeline(ai_config)
        else:
            pipeline = get_pipeline()
        
        return pipeline.detect(
            text,
            detect_pii=detect_pii,
            detect_secrets=detect_secrets,
            detect_harmful=detect_harmful,
            redaction_strategy=strategy,
        )
    
    else:  # regex mode
        return detect_all_regex(
            text,
            detect_pii=detect_pii,
            detect_secrets=detect_secrets,
            detect_harmful=detect_harmful,
            redaction_strategy=strategy,
        )


def detect_pii(text: str, mode: str = "regex") -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect PII only (legacy-compatible API).
    
    Args:
        text: Input text to scan
        mode: "regex" or "ai"
    
    Returns:
        Grouped dictionary format for backward compatibility.
    """
    result = detect(text, mode=mode, detect_secrets=False, detect_harmful=False)
    return result.to_legacy_dict()


def detect_secrets(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect secrets only (always uses regex - 95%+ accuracy).
    
    Returns grouped dictionary format.
    """
    return detect_secrets_legacy(text)


def detect_harmful(text: str, mode: str = "regex") -> Dict[str, Any]:
    """
    Detect harmful content.
    
    Args:
        text: Input text
        mode: "regex" (keyword patterns) or "ai" (transformer model)
    
    Returns:
        Dict with harmful content analysis
    """
    result = detect(text, mode=mode, detect_pii=False, detect_secrets=False)
    
    return {
        "harmful": result.harmful,
        "severity": result.severity,
        "scores": result.harmful_scores,
    }


# ============================================================
# Version
# ============================================================

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.3.0"


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Main API
    "detect",
    "detect_pii",
    "detect_secrets", 
    "detect_harmful",
    
    # Result types
    "Detection",
    "DetectionResult",
    "DetectionType",
    
    # Redaction
    "RedactionStrategy",
    "apply_redaction",
    "redact_spans",
    "redact_text",
    
    # Validation
    "validate_input",
    "validate_input_soft",
    "InputConfig",
    "InputValidationError",
    "InputTooLongError",
    
    # Regex mode
    "detect_all_regex",
    "detect_emails",
    "detect_phones",
    "detect_ssns",
    "detect_credit_cards",
    "detect_bank_accounts",
    "detect_dob",
    "detect_drivers_licenses",
    "detect_mrn",
    "detect_addresses",
    "detect_person_names_regex",
    "detect_secrets_regex",
    "detect_harmful_regex",
    
    # AI mode (None if not available)
    "AI_AVAILABLE",
    "check_ai_available",
    "AIConfig",
    "AIPipeline",
    "detect_all_ai",
    "get_pipeline",
    
    # Utilities
    "luhn_check",
    "shannon_entropy",
    "find_secrets",
    
    # Version
    "__version__",
]
