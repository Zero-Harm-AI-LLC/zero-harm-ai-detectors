"""
AI-powered PII and harmful content detection pipeline.

Uses transformer models for more reliable detection than regex alone,
combined with shared regex patterns from core_patterns for structured data.

File: zero_harm_ai_detectors/ai_detectors.py
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
import re

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch

from .input_validation import validate_input

# ── Shared patterns and utilities from core_patterns ───────────
from .core_patterns import (
    # Redaction
    RedactionStrategy,
    apply_redaction,
    # Validators
    luhn_check,
    # PII patterns (same ones used by regex tier)
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    # Secrets (full three-tier scan)
    find_secrets,
    # Harmful content
    THREAT_CUES_RE,
)

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class PipelineConfig:
    """Configuration for the Zero Harm AI pipeline."""

    # PII Detection
    pii_model: str = "dslim/bert-base-NER"
    pii_threshold: float = 0.7
    pii_aggregation_strategy: str = "simple"

    # Harmful Content Detection
    harmful_model: str = "unitary/multilingual-toxic-xlm-roberta"
    harmful_threshold_per_label: float = 0.5
    harmful_overall_threshold: float = 0.5
    threat_min_score_on_cue: float = 0.6

    # Secrets Detection (uses shared regex from core_patterns)
    use_regex_for_secrets: bool = True

    # General
    device: str = "cpu"


class DetectionType(str, Enum):
    """Types of sensitive content that can be detected."""

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
    """A single detection result."""

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
            "metadata": self.metadata or {},
        }


@dataclass
class PipelineResult:
    """Complete result from the Zero Harm pipeline."""

    original_text: str
    redacted_text: str
    detections: List[Detection]
    harmful: bool
    harmful_scores: Optional[Dict[str, float]] = None
    severity: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
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
            "severity": self.severity,
        }


# ==================== AI-Based PII Detector ====================

class AIPIIDetector:
    """AI-powered PII detector using transformer models + shared regex patterns.

    The NER model handles names, locations, organizations, dates.
    Shared regex patterns from core_patterns handle structured PII
    (emails, phones, SSN, credit cards) — identical to the regex tier.
    """

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
            device=self.device,
        )

    def detect(self, text: str) -> List[Detection]:
        """Detect PII in text using AI + shared regex patterns."""
        text = validate_input(text, mode="ai")
        detections: List[Detection] = []

        # 1. AI-based NER detection (names, locations, orgs, dates)
        try:
            ner_results = self.ner_pipeline(text)
            for entity in ner_results:
                if entity["score"] >= self.config.pii_threshold:
                    entity_type = entity["entity_group"].upper()
                    detection_type = self.LABEL_MAPPING.get(entity_type, entity_type)

                    detections.append(
                        Detection(
                            type=(
                                detection_type.value
                                if isinstance(detection_type, DetectionType)
                                else detection_type
                            ),
                            text=entity["word"].strip(),
                            start=entity["start"],
                            end=entity["end"],
                            confidence=float(entity["score"]),
                            metadata={"method": "ner", "original_label": entity_type},
                        )
                    )
        except Exception as e:
            logger.warning("NER detection failed: %s", e)

        # 2. Structured PII via shared patterns from core_patterns
        #    These are the SAME patterns used by the regex tier.

        for match in EMAIL_RE.finditer(text):
            detections.append(
                Detection(
                    type=DetectionType.EMAIL.value,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    metadata={"method": "regex"},
                )
            )

        for match in PHONE_RE.finditer(text):
            detections.append(
                Detection(
                    type=DetectionType.PHONE.value,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    metadata={"method": "regex"},
                )
            )

        for match in SSN_RE.finditer(text):
            detections.append(
                Detection(
                    type=DetectionType.SSN.value,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    metadata={"method": "regex"},
                )
            )

        for match in CREDIT_CARD_RE.finditer(text):
            digits_only = re.sub(r"\D", "", match.group())
            if luhn_check(digits_only):
                detections.append(
                    Detection(
                        type=DetectionType.CREDIT_CARD.value,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                        metadata={"method": "regex+luhn"},
                    )
                )

        # Remove duplicates and overlaps
        detections = self._remove_overlaps(detections)
        return detections

    @staticmethod
    def _remove_overlaps(detections: List[Detection]) -> List[Detection]:
        """Remove overlapping detections, keeping the one with higher confidence."""
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


# ==================== Secrets Detector ====================

class SecretsDetector:
    """Secrets detector that delegates to the shared three-tier scan.

    Uses ``core_patterns.find_secrets()`` — the same entropy-checked,
    context-aware scan used by the regex tier. Returns ``Detection``
    objects for integration with the AI pipeline.
    """

    def detect(self, text: str) -> List[Detection]:
        """Detect secrets and API keys using the shared scan."""
        findings = find_secrets(text)
        return [
            Detection(
                type=DetectionType.API_KEY.value,
                text=f["span"],
                start=f["start"],
                end=f["end"],
                confidence=0.99,
                metadata={"method": "regex"},
            )
            for f in findings
        ]


# ==================== Harmful Content Detector ====================

class HarmfulContentDetector:
    """Detector for toxic, threatening, and harmful content."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = 0 if config.device == "cuda" and torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(config.harmful_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.harmful_model)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            function_to_apply="sigmoid",
            device=self.device,
        )

        self.id2label = self.model.config.id2label
        self.labels = [self.id2label[i] for i in sorted(self.id2label.keys())]

    def detect(self, text: str) -> Tuple[bool, Dict[str, float], str, List[str]]:
        """Detect harmful content.

        Returns:
            (is_harmful, scores, severity, active_labels)
        """
        raw_scores = self.pipeline(text)[0]
        scores = {item["label"].strip(): float(item["score"]) for item in raw_scores}

        # Apply rules boost for threats using shared cue pattern
        if THREAT_CUES_RE.search(text):
            for label in scores:
                if label.lower() == "threat":
                    scores[label] = max(scores[label], self.config.threat_min_score_on_cue)

        active_labels = [
            label
            for label, score in scores.items()
            if score >= self.config.harmful_threshold_per_label
        ]

        is_harmful = any(
            score >= self.config.harmful_overall_threshold for score in scores.values()
        )

        active_scores = sorted(
            [s for s in scores.values() if s >= self.config.harmful_threshold_per_label],
            reverse=True,
        )
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
    """Unified pipeline for detecting PII, secrets, and harmful content.

    Example:
        pipeline = ZeroHarmPipeline()
        result = pipeline.detect("Email me at john@example.com")
        print(result.redacted_text)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        logger.info("Loading PII detector...")
        self.pii_detector = AIPIIDetector(self.config)

        logger.info("Loading secrets detector...")
        self.secrets_detector = SecretsDetector()

        logger.info("Loading harmful content detector...")
        self.harmful_detector = HarmfulContentDetector(self.config)

        logger.info("Zero Harm AI Pipeline ready")

    def detect(
        self,
        text: str,
        redaction_strategy: RedactionStrategy = RedactionStrategy.TOKEN,
        detect_pii: bool = True,
        detect_secrets: bool = True,
        detect_harmful: bool = True,
    ) -> PipelineResult:
        """Run the complete detection pipeline on text."""
        all_detections: List[Detection] = []

        # 1. PII Detection
        if detect_pii:
            all_detections.extend(self.pii_detector.detect(text))

        # 2. Secrets Detection
        if detect_secrets:
            all_detections.extend(self.secrets_detector.detect(text))

        # 3. Harmful Content Detection
        is_harmful = False
        harmful_scores: Dict[str, float] = {}
        severity = "low"

        if detect_harmful:
            is_harmful, harmful_scores, severity, active_labels = (
                self.harmful_detector.detect(text)
            )
            if is_harmful:
                all_detections.append(
                    Detection(
                        type="HARMFUL_CONTENT",
                        text=text,
                        start=0,
                        end=len(text),
                        confidence=max(harmful_scores.values()),
                        metadata={
                            "severity": severity,
                            "labels": active_labels,
                            "scores": harmful_scores,
                        },
                    )
                )

        # 4. Redact text using shared redaction logic
        redacted_text = self._redact_text(text, all_detections, redaction_strategy)

        return PipelineResult(
            original_text=text,
            redacted_text=redacted_text,
            detections=all_detections,
            harmful=is_harmful,
            harmful_scores=harmful_scores,
            severity=severity,
        )

    def _redact_text(
        self,
        text: str,
        detections: List[Detection],
        strategy: RedactionStrategy,
    ) -> str:
        """Redact sensitive content from text using shared apply_redaction()."""
        sorted_dets = sorted(detections, key=lambda x: x.start, reverse=True)

        result = text
        for det in sorted_dets:
            if det.type == "HARMFUL_CONTENT":
                continue

            original = result[det.start : det.end]
            replacement = apply_redaction(original, strategy, det.type)
            result = result[: det.start] + replacement + result[det.end :]

        return result

    # ========== Backward Compatibility Methods ==========

    def detect_pii_legacy(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy API compatible with old detect_pii function."""
        detections = self.pii_detector.detect(text)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for det in detections:
            if det.type not in grouped:
                grouped[det.type] = []
            grouped[det.type].append(
                {
                    "span": det.text,
                    "start": det.start,
                    "end": det.end,
                    "confidence": det.confidence,
                }
            )
        return grouped

    def detect_secrets_legacy(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Legacy API compatible with old detect_secrets function."""
        detections = self.secrets_detector.detect(text)
        if detections:
            return {
                "SECRETS": [
                    {"span": det.text, "start": det.start, "end": det.end}
                    for det in detections
                ]
            }
        return {}


# ==================== Convenience Functions ====================

_global_pipeline: Optional[ZeroHarmPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> ZeroHarmPipeline:
    """Get or create the global pipeline instance."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = ZeroHarmPipeline(config)
    return _global_pipeline


def detect_all(text: str, redaction_strategy: str = "token") -> Dict[str, Any]:
    """Convenience function: detect everything in one call."""
    p = get_pipeline()
    strategy = RedactionStrategy(redaction_strategy)
    result = p.detect(text, redaction_strategy=strategy)
    return result.to_dict()
