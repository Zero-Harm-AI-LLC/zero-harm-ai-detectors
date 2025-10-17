"""
Zero Harm AI Detectors - AI-powered PII and harmful content detection

File: zero_harm_ai_detectors/__init__.py

Usage:
    # NEW API (Recommended)
    from zero_harm_ai_detectors import ZeroHarmPipeline
    pipeline = ZeroHarmPipeline()
    result = pipeline.detect("Contact john@example.com")
    
    # OLD API (Still works!)
    from zero_harm_ai_detectors import detect_pii, detect_secrets
    pii = detect_pii("Contact john@example.com")
"""

# ==================== NEW AI-BASED PIPELINE ====================
try:
    from .ai_detectors import (
        ZeroHarmPipeline,
        PipelineConfig,
        RedactionStrategy,
        DetectionType,
        Detection,
        PipelineResult,
        AIPIIDetector,
        SecretsDetector,
        HarmfulContentDetector,
        detect_all,
        get_pipeline
    )
    AI_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI detection not available: {e}")
    print("Install transformers and torch: pip install transformers torch")
    AI_DETECTION_AVAILABLE = False
    
    # Create dummy classes for when AI isn't available
    ZeroHarmPipeline = None
    PipelineConfig = None
    RedactionStrategy = None
    DetectionType = None
    Detection = None
    PipelineResult = None
    AIPIIDetector = None
    HarmfulContentDetector = None
    detect_all = None
    get_pipeline = None


# ==================== LEGACY REGEX-BASED DETECTORS ====================
from .detectors import (
    detect_pii as detect_pii_legacy,
    detect_secrets as detect_secrets_legacy,
    redact_text as redact_text_legacy,
    default_detectors,
    EmailDetector,
    PhoneDetector,
    SSNDetector,
    CreditCardDetector,
    BankAccountDetector,
    DOBDetector,
    DriversLicenseDetector,
    MRNDetector,
    PersonNameDetector,
    AddressDetector,
    SecretsDetector as SecretsDetectorLegacy,
    RedactionStrategy as RedactionStrategyLegacy
)

from .harmful_detectors import (
    HarmfulTextDetector as HarmfulTextDetectorLegacy,
    DetectionConfig
)

# Export HarmfulTextDetector for backward compatibility
HarmfulTextDetector = HarmfulTextDetectorLegacy

# If AI not available, use legacy RedactionStrategy
if not AI_DETECTION_AVAILABLE:
    RedactionStrategy = RedactionStrategyLegacy
    # Create a simple SecretsDetector reference
    SecretsDetector = SecretsDetectorLegacy


# ==================== SMART API WRAPPER ====================
def detect_pii(text: str, use_ai: bool = True, **kwargs):
    """
    Detect PII in text
    
    Args:
        text: Input text
        use_ai: If True, use AI-based detection (more accurate).
                If False, use legacy regex detection (faster).
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of detected PII grouped by type
    """
    if use_ai and AI_DETECTION_AVAILABLE:
        pipeline = get_pipeline()
        return pipeline.detect_pii_legacy(text)
    else:
        return detect_pii_legacy(text, **kwargs)


def detect_secrets(text: str, use_ai: bool = False, **kwargs):
    """
    Detect secrets and API keys in text
    
    Args:
        text: Input text
        use_ai: If True, use AI pipeline. If False, use regex (recommended for secrets).
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of detected secrets
    """
    if use_ai and AI_DETECTION_AVAILABLE:
        pipeline = get_pipeline()
        return pipeline.detect_secrets_legacy(text)
    else:
        return detect_secrets_legacy(text, **kwargs)


def redact_text(text: str, findings: dict, strategy: str = "mask_all", **kwargs):
    """
    Redact sensitive information from text
    
    Args:
        text: Original text
        findings: Dictionary of detections from detect_pii or detect_secrets
        strategy: Redaction strategy ("mask_all", "mask_last4", "hash", "token")
        
    Returns:
        Redacted text
    """
    return redact_text_legacy(text, findings, strategy, **kwargs)


def detect_harmful(text: str, use_ai: bool = True, **kwargs):
    """
    Detect harmful content in text
    
    Args:
        text: Input text
        use_ai: If True, use AI pipeline (recommended)
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with harmful content analysis
    """
    if use_ai and AI_DETECTION_AVAILABLE:
        pipeline = get_pipeline()
        _, scores, severity, active_labels = pipeline.harmful_detector.detect(text)
        is_harmful = severity in ["medium", "high"]
        return {
            "text": text,
            "harmful": is_harmful,
            "severity": severity,
            "active_labels": active_labels,
            "scores": scores
        }
    else:
        detector = HarmfulTextDetectorLegacy(**kwargs)
        return detector.detect(text)


# ==================== UNIFIED DETECTION FUNCTION ====================
def detect_all_threats(
    text: str,
    detect_pii: bool = True,
    detect_secrets: bool = True,
    detect_harmful: bool = True,
    redaction_strategy: str = "token",
    use_ai: bool = True
):
    """
    One-stop function to detect all threats in text
    
    Args:
        text: Input text to analyze
        detect_pii: Check for PII
        detect_secrets: Check for secrets/API keys
        detect_harmful: Check for harmful content
        redaction_strategy: How to redact ("token", "mask_all", "mask_last4", "hash")
        use_ai: Use AI models (more accurate) or regex (faster)
        
    Returns:
        Dictionary with all results
    """
    if use_ai and AI_DETECTION_AVAILABLE:
        pipeline = get_pipeline()
        
        try:
            from .ai_detectors import RedactionStrategy
            strategy = RedactionStrategy(redaction_strategy)
        except (ValueError, ImportError):
            strategy = RedactionStrategy.TOKEN
        
        result = pipeline.detect(
            text,
            redaction_strategy=strategy,
            detect_pii=detect_pii,
            detect_secrets=detect_secrets,
            detect_harmful=detect_harmful
        )
        return result.to_dict()
    else:
        detections = {}
        
        if detect_pii:
            pii = detect_pii_legacy(text)
            detections.update(pii)
        
        if detect_secrets:
            secrets = detect_secrets_legacy(text)
            detections.update(secrets)
        
        if detections:
            redacted = redact_text_legacy(text, detections, redaction_strategy)
        else:
            redacted = text
        
        is_harmful = False
        severity = "low"
        harmful_scores = {}
        
        if detect_harmful:
            try:
                detector = HarmfulTextDetectorLegacy()
                harm_result = detector.detect(text)
                is_harmful = harm_result['harmful']
                severity = harm_result['severity']
                harmful_scores = harm_result['scores']
            except ImportError:
                # If transformers not available, skip harmful detection
                pass
        
        return {
            "original": text,
            "redacted": redacted,
            "detections": detections,
            "harmful": is_harmful,
            "severity": severity,
            "harmful_scores": harmful_scores
        }


# ==================== VERSION ====================
try:
    from ._version import version as __version__
except ImportError:
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"


# ==================== EXPORTS ====================
__all__ = [
    # NEW API (Recommended)
    'ZeroHarmPipeline',
    'PipelineConfig',
    'RedactionStrategy',
    'DetectionType',
    'Detection',
    'PipelineResult',
    'detect_all_threats',
    'get_pipeline',
    'AI_DETECTION_AVAILABLE',
    
    # Unified convenience functions
    'detect_pii',
    'detect_secrets',
    'detect_harmful',
    'redact_text',
    
    # Legacy classes (for backward compatibility)
    'EmailDetector',
    'PhoneDetector',
    'SSNDetector',
    'CreditCardDetector',
    'BankAccountDetector',
    'DOBDetector',
    'DriversLicenseDetector',
    'MRNDetector',
    'PersonNameDetector',
    'AddressDetector',
    'HarmfulTextDetector',  # Added for backward compatibility
    'HarmfulTextDetectorLegacy',
    'DetectionConfig',
    'default_detectors',
]