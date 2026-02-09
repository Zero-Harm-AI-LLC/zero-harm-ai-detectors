"""
Zero Harm AI Detectors - Freemium Detection Library

Two detection modes:
- REGEX MODE: Fast pattern-based detection (free tier)
- AI MODE: Transformer-based detection (paid tier)

Usage:
    from zero_harm_ai_detectors import ZeroHarmDetector
    
    # Regex mode (free)
    detector = ZeroHarmDetector(mode='regex')
    result = detector.detect(text)
    
    # AI mode (paid)
    detector = ZeroHarmDetector(mode='ai')
    result = detector.detect(text)

File: zero_harm_ai_detectors/__init__.py
"""

# ==================== REGEX MODE (FREE TIER) ====================
from .regex_detectors import (
    detect_pii as detect_pii_regex,
    detect_secrets as detect_secrets_regex,
    redact_text,
    RedactionStrategy as RegexRedactionStrategy,
    # Individual detectors
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
    SecretsDetector,
)

from .regex_harmful_detectors import (
    detect_harmful as detect_harmful_regex,
    DetectionConfig,
)


# ==================== AI MODE (PAID TIER) ====================
try:
    from .ai_detectors import (
        ZeroHarmPipeline as AIPipeline,
        PipelineConfig,
        RedactionStrategy as AIRedactionStrategy,
        DetectionType,
        Detection,
        PipelineResult,
        AIPIIDetector,
        HarmfulContentDetector as AIHarmfulDetector,
        detect_all as detect_all_ai,
        get_pipeline,
    )
    AI_MODE_AVAILABLE = True
except ImportError:
    # AI mode not available - only regex mode
    AI_MODE_AVAILABLE = False
    AIPipeline = None
    PipelineConfig = None
    AIRedactionStrategy = None
    DetectionType = None
    Detection = None
    PipelineResult = None
    AIPIIDetector = None
    AIHarmfulDetector = None
    detect_all_ai = None
    get_pipeline = None


# ==================== UNIFIED DETECTOR ====================
class ZeroHarmDetector:
    """
    Unified detector supporting both regex and AI modes
    
    REGEX MODE (Free Tier):
    - Fast: 1-5ms per text
    - Accuracy: 95%+ for emails/phones/SSN, 30-40% for names
    - Cost: $0.01/scan
    - Use: High-volume, cost-sensitive scanning
    
    AI MODE (Paid Tier):
    - Slower: 50-200ms per text
    - Accuracy: 99%+ for structured data, 85-95% for names
    - Cost: $0.11/scan
    - Use: Production-grade, accuracy-critical scanning
    
    Example:
        # Regex mode
        detector = ZeroHarmDetector(mode='regex')
        result = detector.detect("Contact john@example.com")
        
        # AI mode
        detector = ZeroHarmDetector(mode='ai')
        result = detector.detect("Contact John Smith at Microsoft")
    """
    
    VALID_MODES = ['regex', 'ai']
    
    def __init__(self, mode: str = 'regex', config: PipelineConfig = None):
        """
        Initialize detector
        
        Args:
            mode: 'regex' for pattern-based (free) or 'ai' for transformer-based (paid)
            config: Optional PipelineConfig for AI mode
        
        Raises:
            ValueError: If mode is not 'regex' or 'ai'
            ImportError: If mode='ai' but transformers/torch not installed
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'regex' or 'ai'."
            )
        
        self.mode = mode
        self.tier = 'paid' if mode == 'ai' else 'free'
        
        if mode == 'ai':
            if not AI_MODE_AVAILABLE:
                raise ImportError(
                    "AI mode requires additional packages.\n"
                    "Install with: pip install 'zero_harm_ai_detectors[ai]'\n"
                    "Or use regex mode: ZeroHarmDetector(mode='regex')"
                )
            self._pipeline = AIPipeline(config)
        else:
            # Regex mode - no initialization needed
            self._pipeline = None
    
    def detect(
        self,
        text: str,
        redaction_strategy: str = None,
        detect_pii: bool = True,
        detect_secrets: bool = True,
        detect_harmful: bool = True
    ):
        """
        Detect sensitive content in text
        
        Args:
            text: Input text to analyze
            redaction_strategy: How to redact ('token', 'mask_all', 'mask_last4', 'hash')
            detect_pii: Whether to detect PII
            detect_secrets: Whether to detect secrets/API keys
            detect_harmful: Whether to detect harmful content
        
        Returns:
            AI mode: PipelineResult object
            Regex mode: Dictionary with detection results
        """
        if self.mode == 'ai':
            return self._detect_ai(
                text, redaction_strategy, 
                detect_pii, detect_secrets, detect_harmful
            )
        else:
            return self._detect_regex(
                text, redaction_strategy,
                detect_pii, detect_secrets, detect_harmful
            )
    
    def _detect_ai(self, text, redaction_strategy, detect_pii, detect_secrets, detect_harmful):
        """AI mode detection"""
        strategy = redaction_strategy or 'token'
        
        # Convert string to enum
        from .ai_detectors import RedactionStrategy as RS
        strategy_map = {
            'token': RS.TOKEN,
            'mask_all': RS.MASK_ALL,
            'mask_last4': RS.MASK_LAST4,
            'hash': RS.HASH
        }
        rs = strategy_map.get(strategy, RS.TOKEN)
        
        return self._pipeline.detect(
            text,
            redaction_strategy=rs,
            detect_pii=detect_pii,
            detect_secrets=detect_secrets,
            detect_harmful=detect_harmful
        )
    
    def _detect_regex(self, text, redaction_strategy, detect_pii, detect_secrets, detect_harmful):
        """Regex mode detection"""
        detections = {}
        
        # PII detection
        if detect_pii:
            pii = detect_pii_regex(text)
            detections.update(pii)
        
        # Secrets detection
        if detect_secrets:
            secrets = detect_secrets_regex(text)
            detections.update(secrets)
        
        # Harmful content detection
        is_harmful = False
        severity = 'low'
        harmful_scores = {}
        
        if detect_harmful:
            harmful = detect_harmful_regex(text)
            if harmful:
                is_harmful = True
                # Extract severity and scores from harmful result
                if isinstance(harmful, dict) and 'HARMFUL_CONTENT' in harmful:
                    harm_data = harmful['HARMFUL_CONTENT'][0]
                    severity = harm_data.get('severity', 'low')
                    harmful_scores = harm_data.get('scores', {})
                    detections.update(harmful)
        
        # Redaction
        strategy = redaction_strategy or 'mask_all'
        if detections:
            redacted = redact_text(text, detections, strategy)
        else:
            redacted = text
        
        return {
            'original': text,
            'redacted': redacted,
            'detections': detections,
            'harmful': is_harmful,
            'severity': severity,
            'harmful_scores': harmful_scores,
            'mode': 'regex',
            'tier': 'free',
            'upgrade_available': self._should_show_upgrade(detections)
        }
    
    def _should_show_upgrade(self, detections):
        """Check if upgrade prompt should be shown"""
        # Show upgrade if person names detected (low accuracy in regex mode)
        has_person_names = 'PERSON_NAME' in detections
        return has_person_names or len(detections) > 5


# ==================== CONVENIENCE FUNCTIONS ====================
def detect(text: str, mode: str = 'regex', **kwargs):
    """
    Quick detection with specified mode
    
    Args:
        text: Input text
        mode: 'regex' (free, fast) or 'ai' (paid, accurate)
        **kwargs: Additional arguments for detect()
    
    Returns:
        Detection results
    
    Example:
        # Regex mode
        result = detect("john@example.com", mode='regex')
        
        # AI mode
        result = detect("John Smith", mode='ai')
    """
    detector = ZeroHarmDetector(mode=mode)
    return detector.detect(text, **kwargs)


def detect_pii(text: str, mode: str = 'regex'):
    """
    Detect PII with specified mode
    
    Args:
        text: Input text
        mode: 'regex' or 'ai'
    
    Returns:
        Dictionary of detected PII by type
    """
    if mode == 'ai' and AI_MODE_AVAILABLE:
        pipeline = get_pipeline()
        result = pipeline.detect(text, detect_secrets=False, detect_harmful=False)
        
        # Convert to common format
        grouped = {}
        for det in result.detections:
            if det.type not in grouped:
                grouped[det.type] = []
            grouped[det.type].append({
                'span': det.text,
                'start': det.start,
                'end': det.end,
                'confidence': det.confidence
            })
        return grouped
    else:
        return detect_pii_regex(text)


def detect_secrets(text: str, mode: str = 'regex'):
    """
    Detect secrets with specified mode
    
    Args:
        text: Input text
        mode: 'regex' or 'ai'
    
    Returns:
        Dictionary of detected secrets
    """
    if mode == 'ai' and AI_MODE_AVAILABLE:
        pipeline = get_pipeline()
        result = pipeline.detect(text, detect_pii=False, detect_harmful=False)
        
        if result.detections:
            return {
                'SECRETS': [
                    {
                        'span': det.text,
                        'start': det.start,
                        'end': det.end
                    }
                    for det in result.detections
                ]
            }
        return {}
    else:
        return detect_secrets_regex(text)


def detect_harmful(text: str, mode: str = 'regex'):
    """
    Detect harmful content with specified mode
    
    Args:
        text: Input text
        mode: 'regex' or 'ai'
    
    Returns:
        Dictionary with harmful content analysis
    """
    if mode == 'ai' and AI_MODE_AVAILABLE:
        pipeline = get_pipeline()
        result = pipeline.detect(text, detect_pii=False, detect_secrets=False)
        
        return {
            'harmful': result.harmful,
            'severity': result.severity,
            'scores': result.harmful_scores
        }
    else:
        return detect_harmful_regex(text)


# ==================== VERSION ====================
try:
    from ._version import version as __version__
except ImportError:
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = '0.3.0'


# ==================== EXPORTS ====================
__all__ = [
    # Main API
    'ZeroHarmDetector',
    'detect',
    'detect_pii',
    'detect_secrets',
    'detect_harmful',
    'redact_text',
    
    # AI Mode (None if not available)
    'AIPipeline',
    'PipelineConfig',
    'AIRedactionStrategy',
    'DetectionType',
    'Detection',
    'PipelineResult',
    'AIPIIDetector',
    'AIHarmfulDetector',
    'get_pipeline',
    
    # Regex Mode (always available)
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
    'SecretsDetector',
    'RegexHarmfulDetector',
    'RegexRedactionStrategy',
    'DetectionConfig',
    
    # Metadata
    'AI_MODE_AVAILABLE',
    '__version__',
]
