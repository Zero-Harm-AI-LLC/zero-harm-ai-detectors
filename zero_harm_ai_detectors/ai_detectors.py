"""
AI-enhanced detection (mode='ai').

Uses transformer models for improved accuracy on:
- Person names: 30% regex → 85-95% AI
- Locations: not available in regex → 80-90% AI
- Organisations: not available in regex → 75-85% AI
- Harmful content: better contextual understanding

Structured data (email, phone, SSN, secrets) still uses regex (95%+ accuracy).

Every public method validates its input before processing.

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
from .input_validation import validate_input, AI_MODE_CONFIG
from .regex_detectors import (
    _detect_emails_raw,
    _detect_phones_raw,
    _detect_ssns_raw,
    _detect_credit_cards_raw,
    _detect_bank_accounts_raw,
    _detect_dob_raw,
    _detect_drivers_licenses_raw,
    _detect_mrn_raw,
    _detect_addresses_raw,
    _detect_secrets_raw,
)


# ============================================================
# Availability check
# ============================================================

def check_ai_available() -> bool:
    """Check if AI dependencies are installed."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


AI_AVAILABLE = check_ai_available()


# ============================================================
# Configuration
# ============================================================

@dataclass
class AIConfig:
    """Configuration for AI models."""
    ner_model: str = "dslim/bert-base-NER"
    harmful_model: str = "unitary/multilingual-toxic-xlm-roberta"
    ner_threshold: float = 0.70
    harmful_threshold: float = 0.5
    device: str = "cpu"   # "cpu" or "cuda"
    max_length: int = 512


# ============================================================
# NER Detector
# ============================================================

class NERDetector:
    """Named Entity Recognition using transformers."""

    LABEL_MAP = {
        "PER":   DetectionType.PERSON.value,
        "LOC":   DetectionType.LOCATION.value,
        "ORG":   DetectionType.ORGANIZATION.value,
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
        """Lazy-load the NER pipeline."""
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
        """
        Detect named entities in text.

        Validates input before processing.
        """
        text = validate_input(text, AI_MODE_CONFIG)
        if not text.strip():
            return []

        detections = []
        try:
            results = self.pipeline(text[: self.config.max_length])
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
        except Exception:
            pass  # Degrade gracefully; caller gets an empty list

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
        """Lazy-load the classification pipeline."""
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
        """
        Detect harmful content in text.

        Validates input before processing.

        Returns:
            Dict with keys: harmful (bool), severity (str), scores (dict)
        """
        text = validate_input(text, AI_MODE_CONFIG)
        if not text.strip():
            return {"harmful": False, "severity": "none", "scores": {}}

        try:
            results = self.pipeline(text[: self.config.max_length])

            scores: Dict[str, float] = {}
            max_score = 0.0

            items = results[0] if isinstance(results[0], list) else results
            for item in items:
                label = item["label"].lower()
                score = item["score"]
                scores[label] = score
                if label != "neutral" and score > max_score:
                    max_score = score

            is_harmful = max_score >= self.config.harmful_threshold

            if max_score >= 0.8:
                severity = "high"
            elif max_score >= 0.6:
                severity = "medium"
            elif is_harmful:
                severity = "low"
            else:
                severity = "none"

            return {"harmful": is_harmful, "severity": severity, "scores": scores}

        except Exception:
            return {"harmful": False, "severity": "none", "scores": {}}


# ============================================================
# AI Pipeline
# ============================================================

class AIPipeline:
    """
    Complete AI detection pipeline.

    Uses AI for: person names, locations, organisations, harmful content.
    Uses regex for: email, phone, SSN, credit card, secrets (95%+ accuracy).

    Input is validated once at the top of detect(); individual components
    receive the already-validated string.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self._ner_detector: Optional[NERDetector] = None
        self._harmful_detector: Optional[HarmfulContentDetector] = None

    @property
    def ner_detector(self) -> NERDetector:
        if self._ner_detector is None:
            self._ner_detector = NERDetector(self.config)
        return self._ner_detector

    @property
    def harmful_detector(self) -> HarmfulContentDetector:
        if self._harmful_detector is None:
            self._harmful_detector = HarmfulContentDetector(self.config)
        return self._harmful_detector

    def _detect_structured_pii(self, text: str) -> List[Detection]:
        """Run regex-based structured PII detection on pre-validated text."""
        detections: List[Detection] = []
        detections.extend(_detect_emails_raw(text))
        detections.extend(_detect_phones_raw(text))
        detections.extend(_detect_ssns_raw(text))
        detections.extend(_detect_credit_cards_raw(text))
        detections.extend(_detect_bank_accounts_raw(text))
        detections.extend(_detect_dob_raw(text))
        detections.extend(_detect_drivers_licenses_raw(text))
        detections.extend(_detect_mrn_raw(text))
        detections.extend(_detect_addresses_raw(text))
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

        Validates input at entry; sub-components receive pre-validated text.

        Args:
            text:               Input text to scan.
            detect_pii:         Whether to detect PII.
            detect_secrets:     Whether to detect secrets/API keys.
            detect_harmful:     Whether to detect harmful content.
            redaction_strategy: How to replace detected content.

        Returns:
            DetectionResult with all findings.

        Raises:
            InputValidationError: If text is None or looks like binary data.
            InputTooLongError:    If text exceeds AI_MODE_CONFIG.max_length.
        """
        text = validate_input(text, AI_MODE_CONFIG)

        if not text:
            return DetectionResult(
                original_text="",
                redacted_text="",
                detections=[],
                mode="ai",
            )

        all_detections: List[Detection] = []

        if detect_pii:
            all_detections.extend(self._detect_structured_pii(text))
            # NER detector validates internally but receives already-clean text
            all_detections.extend(self.ner_detector.detect(text))

        if detect_secrets:
            all_detections.extend(_detect_secrets_raw(text))

        is_harmful = False
        harmful_scores: Dict[str, float] = {}
        severity = "none"

        if detect_harmful:
            harmful_result = self.harmful_detector.detect(text)
            is_harmful = harmful_result["harmful"]
            harmful_scores = harmful_result["scores"]
            severity = harmful_result["severity"]

        # Deduplicate
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
            mode="ai",
            harmful=is_harmful,
            harmful_scores=harmful_scores,
            severity=severity,
        )


# ============================================================
# Module-level singleton + convenience function
# ============================================================

_default_pipeline: Optional[AIPipeline] = None


def get_pipeline(config: Optional[AIConfig] = None) -> AIPipeline:
    """Return (or create) the shared default AIPipeline."""
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

    Validates input via AIPipeline.detect().

    Args:
        text:               Input text to scan.
        detect_pii:         Whether to detect PII.
        detect_secrets:     Whether to detect secrets/API keys.
        detect_harmful:     Whether to detect harmful content.
        redaction_strategy: Redaction strategy.
        config:             Optional AIConfig.

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
