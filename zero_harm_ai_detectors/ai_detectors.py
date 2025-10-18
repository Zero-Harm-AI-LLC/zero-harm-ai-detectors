"""
AI-powered PII and harmful content detection pipeline
Uses transformer models for more reliable detection than regex
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
import torch

# ==================== Configuration ====================
@dataclass
class PipelineConfig:
    """Configuration for the Zero Harm AI pipeline"""
    # PII Detection
    pii_model: str = "dslim/bert-base-NER"  # Or "Jean-Baptiste/roberta-large-ner-english"
    pii_threshold: float = 0.7
    pii_aggregation_strategy: str = "simple"  # "simple", "first", "average", "max"
    
    # Harmful Content Detection  
    harmful_model: str = "unitary/multilingual-toxic-xlm-roberta"
    harmful_threshold_per_label: float = 0.5
    harmful_overall_threshold: float = 0.5
    threat_min_score_on_cue: float = 0.6
    
    # Secrets Detection (still use regex for API keys - they're well-structured)
    use_regex_for_secrets: bool = True
    
    # General
    device: str = "cpu"  # "cpu" or "cuda"


class RedactionStrategy(str, Enum):
    """How to redact detected sensitive information"""
    MASK_ALL = "mask_all"
    MASK_LAST4 = "mask_last4"
    TOKEN = "token"  # Use [REDACTED_TYPE] tokens
    HASH = "hash"


class DetectionType(str, Enum):
    """Types of sensitive content that can be detected"""
    # PII Types
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    DATE = "DATE"
    ADDRESS = "ADDRESS"
    
    # Secret Types
    API_KEY = "API_KEY"
    TOKEN = "TOKEN"
    PASSWORD = "PASSWORD"
    
    # Harmful Content
    TOXIC = "TOXIC"
    THREAT = "THREAT"
    INSULT = "INSULT"
    OBSCENE = "OBSCENE"
    IDENTITY_HATE = "IDENTITY_HATE"


# ==================== Result Classes ====================
@dataclass
class Detection:
    """A single detection result"""
    type: str
    text: str
    start: int
    end: int
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "span": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }


@dataclass
class PipelineResult:
    """Complete result from the Zero Harm pipeline"""
    original_text: str
    redacted_text: str
    detections: List[Detection]
    harmful: bool
    harmful_scores: Optional[Dict[str, float]] = None
    severity: str = "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with old API"""
        # Group detections by type
        grouped = {}
        for det in self.detections:
            if det.type not in grouped:
                grouped[det.type] = []
            grouped[det.type].append(det.to_dict())
        
        return {
            "original": self.original_text,
            "redacted": self.redacted_text,
            "detections": grouped,
            "harmful": self.harmful,
            "harmful_scores": self.harmful_scores or {},
            "severity": self.severity
        }


# ==================== AI-Based PII Detector ====================
class AIPIIDetector:
    """AI-powered PII detector using transformer models"""
    
    # Mapping from NER labels to our detection types
    LABEL_MAPPING = {
        "PER": DetectionType.PERSON,
        "PERSON": DetectionType.PERSON,
        "LOC": DetectionType.LOCATION,
        "LOCATION": DetectionType.LOCATION,
        "ORG": DetectionType.ORGANIZATION,
        "ORGANIZATION": DetectionType.ORGANIZATION,
        "DATE": DetectionType.DATE,
        "TIME": DetectionType.DATE,
    }
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = 0 if config.device == "cuda" and torch.cuda.is_available() else -1
        
        # Initialize NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=config.pii_model,
            tokenizer=config.pii_model,
            aggregation_strategy=config.pii_aggregation_strategy,
            device=self.device
        )
        
        # Additional regex patterns for structured PII (emails, phones, etc.)
        self.email_pattern = re.compile(r'\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b', re.UNICODE)
        self.phone_pattern = re.compile(r'(?:(?:\+1[-.\s]?)?\(?(?:\d{3})\)?[-.\s]?\d{3}[-.\s]?\d{4})\b')
        self.ssn_pattern = re.compile(r'\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b')
        self.credit_card_pattern = re.compile(r'\b(?:\d[ -]?){13,19}\b')
        
    def detect(self, text: str) -> List[Detection]:
        """Detect PII in text using AI and structured patterns"""
        detections = []
        
        # 1. AI-based NER detection
        try:
            ner_results = self.ner_pipeline(text)
            for entity in ner_results:
                if entity['score'] >= self.config.pii_threshold:
                    # Map entity type to our detection type
                    entity_type = entity['entity_group'].upper()
                    detection_type = self.LABEL_MAPPING.get(entity_type, entity_type)
                    
                    detections.append(Detection(
                        type=detection_type.value if isinstance(detection_type, DetectionType) else detection_type,
                        text=entity['word'].strip(),
                        start=entity['start'],
                        end=entity['end'],
                        confidence=float(entity['score']),
                        metadata={"method": "ner", "original_label": entity_type}
                    ))
        except Exception as e:
            print(f"Warning: NER detection failed: {e}")
        
        # 2. Structured pattern detection (emails, phones, etc.)
        # These are reliable with regex since they have well-defined formats
        
        # Emails
        for match in self.email_pattern.finditer(text):
            detections.append(Detection(
                type=DetectionType.EMAIL.value,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=1.0,
                metadata={"method": "regex"}
            ))
        
        # Phones
        for match in self.phone_pattern.finditer(text):
            detections.append(Detection(
                type=DetectionType.PHONE.value,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=1.0,
                metadata={"method": "regex"}
            ))
        
        # SSN
        for match in self.ssn_pattern.finditer(text):
            detections.append(Detection(
                type=DetectionType.SSN.value,
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=1.0,
                metadata={"method": "regex"}
            ))
        
        # Credit Cards (with Luhn check)
        for match in self.credit_card_pattern.finditer(text):
            digits_only = re.sub(r'\D', '', match.group())
            if self._luhn_check(digits_only):
                detections.append(Detection(
                    type=DetectionType.CREDIT_CARD.value,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    metadata={"method": "regex+luhn"}
                ))
        
        # Remove duplicates and overlaps
        detections = self._remove_overlaps(detections)
        
        return detections
    
    @staticmethod
    def _luhn_check(number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
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
    
    @staticmethod
    def _remove_overlaps(detections: List[Detection]) -> List[Detection]:
        """Remove overlapping detections, keeping the one with higher confidence"""
        if not detections:
            return []
        
        # Sort by start position, then by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: (x.start, -x.confidence))
        
        result = []
        for det in sorted_dets:
            # Check if this detection overlaps with any in result
            overlaps = False
            for existing in result:
                if not (det.end <= existing.start or det.start >= existing.end):
                    # They overlap
                    overlaps = True
                    # Replace if new one has higher confidence
                    if det.confidence > existing.confidence:
                        result.remove(existing)
                        result.append(det)
                    break
            
            if not overlaps:
                result.append(det)
        
        return sorted(result, key=lambda x: x.start)


# ==================== Secrets Detector ====================
class SecretsDetector:
    """Detector for API keys, tokens, and credentials"""
    
    PATTERNS = {
        "OPENAI_KEY": re.compile(r'\bsk-[A-Za-z0-9]{32,64}\b'),
        "OPENAI_PROJECT_KEY": re.compile(r'\bsk-proj-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b'),
        "OPENAI_ORG_KEY": re.compile(r'\bsk-org-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b'),
        "AWS_ACCESS_KEY": re.compile(r'\b(AKI|ASI)A[0-9A-Z]{16}\b'),
        "AWS_SECRET_KEY": re.compile(r'\b[0-9a-zA-Z/+]{40}\b'),
        "GOOGLE_API_KEY": re.compile(r'\bAIza[0-9A-Za-z\-_]{35}\b'),
        "SLACK_TOKEN": re.compile(r'\bxox[baprs]-[0-9A-Za-z-]{10,100}\b'),
        "JWT": re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b'),
        "STRIPE_KEY": re.compile(r'\bsk_(live|test)_[0-9a-zA-Z]{24}\b'),
        "GITHUB_TOKEN": re.compile(r'\bghp_[0-9A-Za-z]{36}\b'),
    }
    
    def detect(self, text: str) -> List[Detection]:
        """Detect secrets and API keys"""
        detections = []
        
        for secret_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                detections.append(Detection(
                    type=DetectionType.API_KEY.value,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.99,
                    metadata={"secret_type": secret_type, "method": "regex"}
                ))
        
        return detections


# ==================== Harmful Content Detector ====================
class HarmfulContentDetector:
    """Detector for toxic, threatening, and harmful content"""
    
    THREAT_CUES = re.compile(
        r'\b(kill|hurt|stab|shoot|burn|bomb|beat|rape|destroy|attack|threaten|lynch)\b',
        re.IGNORECASE
    )
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = 0 if config.device == "cuda" and torch.cuda.is_available() else -1
        
        # Initialize classification pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(config.harmful_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.harmful_model)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            function_to_apply="sigmoid",
            device=self.device
        )
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        self.labels = [self.id2label[i] for i in sorted(self.id2label.keys())]
    
    def detect(self, text: str) -> Tuple[bool, Dict[str, float], str, List[str]]:
        """
        Detect harmful content
        
        Returns:
            (is_harmful, scores, severity, active_labels)
        """
        # Get model scores
        raw_scores = self.pipeline(text)[0]
        scores = {item['label'].strip(): float(item['score']) for item in raw_scores}
        
        # Apply rules boost for threats
        if self.THREAT_CUES.search(text):
            for label in scores:
                if label.lower() == 'threat':
                    scores[label] = max(scores[label], self.config.threat_min_score_on_cue)
        
        # Determine active labels
        active_labels = [
            label for label, score in scores.items()
            if score >= self.config.harmful_threshold_per_label
        ]
        
        # Determine if harmful overall
        is_harmful = any(
            score >= self.config.harmful_overall_threshold
            for score in scores.values()
        )
        
        # Calculate severity
        active_scores = sorted([s for s in scores.values() if s >= self.config.harmful_threshold_per_label], reverse=True)
        severity = "low"
        if active_scores:
            max_score = active_scores[0]
            if max_score >= 0.85:
                severity = "high"
            elif max_score >= 0.6:
                severity = "medium"
        
        return is_harmful, scores, severity, active_labels


# ==================== Unified Zero Harm AI Pipeline ====================
class ZeroHarmPipeline:
    """
    Unified pipeline for detecting PII, secrets, and harmful content.
    
    This is the main class you should use - it combines all detection
    capabilities into one easy-to-use interface.
    
    Example:
        pipeline = ZeroHarmPipeline()
        result = pipeline.detect("Email me at john@example.com")
        print(result.redacted_text)
        print(result.detections)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the Zero Harm AI pipeline
        
        Args:
            config: Optional configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
        
        # Initialize detectors
        print("Loading PII detector...")
        self.pii_detector = AIPIIDetector(self.config)
        
        print("Loading secrets detector...")
        self.secrets_detector = SecretsDetector()
        
        print("Loading harmful content detector...")
        self.harmful_detector = HarmfulContentDetector(self.config)
        
        print("✅ Zero Harm AI Pipeline ready!")
    
    def detect(
        self,
        text: str,
        redaction_strategy: RedactionStrategy = RedactionStrategy.TOKEN,
        detect_pii: bool = True,
        detect_secrets: bool = True,
        detect_harmful: bool = True
    ) -> PipelineResult:
        """
        Run complete detection pipeline on text
        
        Args:
            text: Input text to analyze
            redaction_strategy: How to redact sensitive content
            detect_pii: Whether to detect PII
            detect_secrets: Whether to detect secrets
            detect_harmful: Whether to detect harmful content
            
        Returns:
            PipelineResult with all detections and redacted text
        """
        all_detections = []
        
        # 1. PII Detection
        if detect_pii:
            pii_detections = self.pii_detector.detect(text)
            all_detections.extend(pii_detections)
        
        # 2. Secrets Detection
        if detect_secrets:
            secret_detections = self.secrets_detector.detect(text)
            all_detections.extend(secret_detections)
        
        # 3. Harmful Content Detection
        is_harmful = False
        harmful_scores = {}
        severity = "low"
        active_labels = []
        
        if detect_harmful:
            is_harmful, harmful_scores, severity, active_labels = self.harmful_detector.detect(text)
            
            # If harmful, add as detection covering whole text
            if is_harmful:
                all_detections.append(Detection(
                    type="HARMFUL_CONTENT",
                    text=text,
                    start=0,
                    end=len(text),
                    confidence=max(harmful_scores.values()),
                    metadata={
                        "severity": severity,
                        "labels": active_labels,
                        "scores": harmful_scores
                    }
                ))
        
        # 4. Redact text
        redacted_text = self._redact_text(text, all_detections, redaction_strategy)
        
        return PipelineResult(
            original_text=text,
            redacted_text=redacted_text,
            detections=all_detections,
            harmful=is_harmful,
            harmful_scores=harmful_scores,
            severity=severity
        )
    
    def _redact_text(
        self,
        text: str,
        detections: List[Detection],
        strategy: RedactionStrategy
    ) -> str:
        """Redact sensitive content from text"""
        # Sort detections by start position (reversed for easier replacement)
        sorted_dets = sorted(detections, key=lambda x: x.start, reverse=True)
        
        result = text
        for det in sorted_dets:
            # Skip harmful content detection (it covers entire text)
            if det.type == "HARMFUL_CONTENT":
                continue
            
            original = result[det.start:det.end]
            
            if strategy == RedactionStrategy.TOKEN:
                replacement = f"[REDACTED_{det.type}]"
            elif strategy == RedactionStrategy.MASK_ALL:
                replacement = "*" * len(original)
            elif strategy == RedactionStrategy.MASK_LAST4:
                if len(original) >= 4:
                    replacement = "*" * (len(original) - 4) + original[-4:]
                else:
                    replacement = "*" * len(original)
            elif strategy == RedactionStrategy.HASH:
                import hashlib
                replacement = hashlib.sha256(original.encode()).hexdigest()
            else:
                replacement = "[REDACTED]"
            
            result = result[:det.start] + replacement + result[det.end:]
        
        return result
    
    # ========== Backward Compatibility Methods ==========
    
    def detect_pii_legacy(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy API compatible with old detect_pii function"""
        detections = self.pii_detector.detect(text)
        
        # Group by type
        grouped = {}
        for det in detections:
            if det.type not in grouped:
                grouped[det.type] = []
            grouped[det.type].append({
                "span": det.text,
                "start": det.start,
                "end": det.end,
                "confidence": det.confidence
            })
        
        return grouped
    
    def detect_secrets_legacy(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy API compatible with old detect_secrets function"""
        detections = self.secrets_detector.detect(text)
        
        if detections:
            return {
                "SECRETS": [
                    {
                        "span": det.text,
                        "start": det.start,
                        "end": det.end
                    }
                    for det in detections
                ]
            }
        return {}


# ==================== Convenience Functions ====================

# Global pipeline instance (lazy loaded)
_global_pipeline: Optional[ZeroHarmPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> ZeroHarmPipeline:
    """Get or create the global pipeline instance"""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = ZeroHarmPipeline(config)
    return _global_pipeline


def detect_all(text: str, redaction_strategy: str = "token") -> Dict[str, Any]:
    """
    Convenience function: detect everything in one call
    
    Args:
        text: Input text
        redaction_strategy: "token", "mask_all", "mask_last4", or "hash"
        
    Returns:
        Dictionary with all results
    """
    pipeline = get_pipeline()
    strategy = RedactionStrategy(redaction_strategy)
    result = pipeline.detect(text, redaction_strategy=strategy)
    return result.to_dict()


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Example 1: Simple usage
    print("=" * 60)
    print("Example 1: Simple Detection")
    print("=" * 60)
    
    pipeline = ZeroHarmPipeline()
    text = "Contact John Smith at john.smith@email.com or call 555-123-4567. API key: sk-abc123def456."
    
    result = pipeline.detect(text)
    print(f"Original: {result.original_text}")
    print(f"Redacted: {result.redacted_text}")
    print(f"Found {len(result.detections)} items:")
    for det in result.detections:
        print(f"  - {det.type}: '{det.text}' (confidence: {det.confidence:.2f})")
    
    print("\n" + "=" * 60)
    print("Example 2: Harmful Content")
    print("=" * 60)
    
    harmful_text = "I hate you and want to hurt you"
    result = pipeline.detect(harmful_text)
    print(f"Text: {harmful_text}")
    print(f"Harmful: {result.harmful}")
    print(f"Severity: {result.severity}")
    if result.harmful_scores:
        print(f"Top scores:")
        for label, score in sorted(result.harmful_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {label}: {score:.3f}")
