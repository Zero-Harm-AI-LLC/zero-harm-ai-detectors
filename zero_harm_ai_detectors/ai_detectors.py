"""
AI-enhanced detection (mode='ai').

Uses transformer models for improved accuracy on:
- Person names: 30% regex → 85-95% AI
- Locations: Not available in regex → 80-90% AI
- Organizations: Not available in regex → 75-85% AI
- Harmful content: Better contextual understanding

Structured data (email, phone, SSN, secrets) still uses regex (95%+ accuracy).

Requirements:
    pip install zero_harm_ai_detectors[ai]

File: zero_harm_ai_detectors/ai_detectors.py
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core_patterns import (
    Detection,
    DetectionResult,
    DetectionType,
    RedactionStrategy,
    redact_spans,
)
from .regex_detectors import (
    detect_emails,
    detect_phones,
    detect_ssns,
    detect_credit_cards,
    detect_bank_accounts,
    detect_dob,
    detect_drivers_licenses,
    detect_mrn,
    detect_addresses,
    detect_secrets_regex,
)


# ============================================================
# Check AI Availability
# ============================================================

def check_ai_available() -> bool:
    """Check if AI dependencies are available."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


AI_AVAILABLE = check_ai_available()


# ============================================================
# AI Configuration
# ============================================================

@dataclass
class AIConfig:
    """Configuration for AI models."""
    ner_model: str = "dslim/bert-base-NER"
    harmful_model: str = "unitary/multilingual-toxic-xlm-roberta"
    ner_threshold: float = 0.70
    harmful_threshold: float = 0.5
    device: str = "cpu"  # "cpu" or "cuda"
    max_length: int = 512


# ============================================================
# NER Detector (Names, Locations, Organizations)
# ============================================================

class NERDetector:
    """Named Entity Recognition using transformers."""
    
    # Map NER labels to our detection types
    LABEL_MAP = {
        "PER": DetectionType.PERSON.value,
        "LOC": DetectionType.LOCATION.value,
        "ORG": DetectionType.ORGANIZATION.value,
        "B-PER": DetectionType.PERSON.value,
        "I-PER": DetectionType.PERSON.value,
        "B-LOC": DetectionType.LOCATION.value,
        "I-LOC": DetectionType.LOCATION.value,
        "B-ORG": DetectionType.ORGANIZATION.value,
        "I-ORG": DetectionType.ORGANIZATION.value,
    }
    
    def __init__(self, config: Optional[AIConfig] = None):
        if not AI_AVAILABLE:
            raise ImportError(
                "AI dependencies not available. "
                "Install with: pip install 'zero_harm_ai_detectors[ai]'"
            )
        
        self.config = config or AIConfig()
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load the NER pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "ner",
                model=self.config.ner_model,
                aggregation_strategy="simple",
                device=0 if self.config.device == "cuda" else -1,
            )
        return self._pipeline
    
    def detect(self, text: str) -> List[Detection]:
        """Detect named entities in text."""
        if not text.strip():
            return []
        
        detections = []
        
        try:
            # Run NER
            results = self.pipeline(text[:self.config.max_length])
            
            for entity in results:
                label = entity.get("entity_group", entity.get("entity", ""))
                det_type = self.LABEL_MAP.get(label)
                
                if det_type and entity["score"] >= self.config.ner_threshold:
                    detections.append(Detection(
                        type=det_type,
                        text=entity["word"],
                        start=entity["start"],
                        end=entity["end"],
                        confidence=entity["score"],
                        metadata={"method": "ai_ner", "model": self.config.ner_model},
                    ))
        except Exception as e:
            # Fall back gracefully on errors
            pass
        
        return detections


# ============================================================
# Harmful Content Detector
# ============================================================

class HarmfulContentDetector:
    """Harmful content detection using transformers."""
    
    def __init__(self, config: Optional[AIConfig] = None):
        if not AI_AVAILABLE:
            raise ImportError(
                "AI dependencies not available. "
                "Install with: pip install 'zero_harm_ai_detectors[ai]'"
            )
        
        self.config = config or AIConfig()
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load the classification pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.config.harmful_model,
                top_k=None,
                device=0 if self.config.device == "cuda" else -1,
            )
        return self._pipeline
    
    def detect(self, text: str) -> Dict[str, Any]:
        """Detect harmful content in text."""
        if not text.strip():
            return {"harmful": False, "severity": "none", "scores": {}}
        
        try:
            results = self.pipeline(text[:self.config.max_length])
            
            # Process results
            scores = {}
            max_score = 0.0
            
            if results and len(results) > 0:
                for item in results[0] if isinstance(results[0], list) else results:
                    label = item["label"].lower()
                    score = item["score"]
                    scores[label] = score
                    if score > max_score and label != "neutral":
                        max_score = score
            
            is_harmful = max_score >= self.config.harmful_threshold
            
            # Determine severity
            if max_score >= 0.8:
                severity = "high"
            elif max_score >= 0.6:
                severity = "medium"
            elif max_score >= self.config.harmful_threshold:
                severity = "low"
            else:
                severity = "none"
            
            return {
                "harmful": is_harmful,
                "severity": severity,
                "scores": scores,
            }
        except Exception as e:
            return {"harmful": False, "severity": "none", "scores": {}}


# ============================================================
# AI Pipeline (Combines NER + Harmful + Regex)
# ============================================================

class AIPipeline:
    """
    Complete AI detection pipeline.
    
    Uses AI for: person names, locations, organizations, harmful content.
    Uses regex for: email, phone, SSN, credit card, secrets (95%+ accuracy).
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self._ner_detector = None
        self._harmful_detector = None
    
    @property
    def ner_detector(self) -> NERDetector:
        """Lazy load NER detector."""
        if self._ner_detector is None:
            self._ner_detector = NERDetector(self.config)
        return self._ner_detector
    
    @property
    def harmful_detector(self) -> HarmfulContentDetector:
        """Lazy load harmful content detector."""
        if self._harmful_detector is None:
            self._harmful_detector = HarmfulContentDetector(self.config)
        return self._harmful_detector
    
    def detect_structured_pii(self, text: str) -> List[Detection]:
        """Detect structured PII using regex (high accuracy)."""
        detections = []
        detections.extend(detect_emails(text))
        detections.extend(detect_phones(text))
        detections.extend(detect_ssns(text))
        detections.extend(detect_credit_cards(text))
        detections.extend(detect_bank_accounts(text))
        detections.extend(detect_dob(text))
        detections.extend(detect_drivers_licenses(text))
        detections.extend(detect_mrn(text))
        detections.extend(detect_addresses(text))
        return detections
    
    def detect(
        self,
        text: str,
        detect_pii: bool = True,
        detect_secrets: bool = True,
        detect_harmful: bool = True,
        redaction_strategy: RedactionStrategy = RedactionStrategy.TOKEN,
    ) -> DetectionResult:
        """
        Run full AI-enhanced detection.
        
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
                mode="ai",
            )
        
        all_detections: List[Detection] = []
        
        # PII detection
        if detect_pii:
            # Structured PII via regex (high accuracy)
            all_detections.extend(self.detect_structured_pii(text))
            
            # Names, locations, orgs via AI NER
            all_detections.extend(self.ner_detector.detect(text))
        
        # Secrets detection (always regex - 95%+ accuracy)
        if detect_secrets:
            all_detections.extend(detect_secrets_regex(text))
        
        # Harmful content detection
        is_harmful = False
        harmful_scores: Dict[str, float] = {}
        severity = "none"
        
        if detect_harmful:
            harmful_result = self.harmful_detector.detect(text)
            is_harmful = harmful_result["harmful"]
            harmful_scores = harmful_result["scores"]
            severity = harmful_result["severity"]
        
        # Remove duplicates
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
            mode="ai",
            harmful=is_harmful,
            harmful_scores=harmful_scores,
            severity=severity,
        )


# ============================================================
# Module-level Pipeline Instance
# ============================================================

_default_pipeline: Optional[AIPipeline] = None


def get_pipeline(config: Optional[AIConfig] = None) -> AIPipeline:
    """Get or create the default AI pipeline."""
    global _default_pipeline
    
    if config is not None:
        return AIPipeline(config)
    
    if _default_pipeline is None:
        _default_pipeline = AIPipeline()
    
    return _default_pipeline


def detect_all_ai(
    text: str,
    detect_pii: bool = True,
    detect_secrets: bool = True,
    detect_harmful: bool = True,
    redaction_strategy: RedactionStrategy = RedactionStrategy.TOKEN,
    config: Optional[AIConfig] = None,
) -> DetectionResult:
    """
    Convenience function for AI detection.
    
    Args:
        text: Input text
        detect_pii: Whether to detect PII
        detect_secrets: Whether to detect secrets
        detect_harmful: Whether to detect harmful content
        redaction_strategy: Redaction strategy
        config: Optional AI configuration
    
    Returns:
        DetectionResult
    """
    pipeline = get_pipeline(config)
    return pipeline.detect(
        text,
        detect_pii=detect_pii,
        detect_secrets=detect_secrets,
        detect_harmful=detect_harmful,
        redaction_strategy=redaction_strategy,
    )
