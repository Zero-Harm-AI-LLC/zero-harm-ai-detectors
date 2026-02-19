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
from typing import Optional

# ============================================================
# Core Exports (always available)
# ============================================================

from .core_patterns import (
    # Result types
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
# Unified API
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
        DetectionResult with:
            original_text (str)   - Original input
            redacted_text (str)   - Text with sensitive content replaced
            detections (list)     - List of Detection objects
            mode (str)            - "regex" or "ai"
            harmful (bool)        - Whether harmful content was found
            harmful_scores (dict) - Per-category harm scores
            severity (str)        - "none" | "low" | "medium" | "high"

    Raises:
        ValueError: If mode is not "regex" or "ai"
        ImportError: If mode="ai" but AI dependencies are not installed
    """
    if mode not in ("regex", "ai"):
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'regex' or 'ai'.")

    # Validate at the top-level entry point.  The individual detectors also
    # validate, but doing it here gives a single clear error message and
    # avoids the AI pipeline receiving bad input before we can report it.
    _config = AI_MODE_CONFIG if mode == "ai" else REGEX_MODE_CONFIG
    text = validate_input(text, _config)

    try:
        strategy = RedactionStrategy(redaction_strategy)
    except ValueError:
        strategy = RedactionStrategy.TOKEN

    if mode == "ai":
        if not AI_AVAILABLE:
            raise ImportError(
                "AI mode requires additional dependencies.\n"
                "Install with: pip install 'zero_harm_ai_detectors[ai]'\n"
                "Or use the default regex mode: detect(text, mode='regex')"
            )
        pipeline = AIPipeline(ai_config) if ai_config else get_pipeline()
        return pipeline.detect(
            text,
            detect_pii=detect_pii,
            detect_secrets=detect_secrets,
            detect_harmful=detect_harmful,
            redaction_strategy=strategy,
        )

    return detect_all_regex(
        text,
        detect_pii=detect_pii,
        detect_secrets=detect_secrets,
        detect_harmful=detect_harmful,
        redaction_strategy=strategy,
    )


# ============================================================
# Version
# ============================================================

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.0.0"


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Main entry point
    "detect",

    # Result types
    "Detection",
    "DetectionResult",
    "DetectionType",

    # Redaction
    "RedactionStrategy",
    "apply_redaction",
    "redact_spans",

    # Input validation
    "validate_input",
    "validate_input_soft",
    "InputConfig",
    "InputValidationError",
    "InputTooLongError",

    # Regex mode - individual detectors
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

    # AI mode (None if dependencies not installed)
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
