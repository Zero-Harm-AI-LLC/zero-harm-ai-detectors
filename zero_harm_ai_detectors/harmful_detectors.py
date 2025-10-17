"""
Legacy harmful content detector (v0.1.x)

⚠️  DEPRECATED: This detector is kept for backward compatibility only.

For new code, use the AI-powered unified pipeline:
    from zero_harm_ai_detectors import ZeroHarmPipeline
    pipeline = ZeroHarmPipeline()
    result = pipeline.detect(text)

This legacy detector will be maintained but not enhanced. It provides:
1. Backward compatibility with v0.1.x code
2. Fallback when transformers is not installed
3. Standalone harmful content detection

File: zero_harm_ai_detectors/harmful_detectors.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import regex as re

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

# Default model
DEFAULT_MODEL = "unitary/multilingual-toxic-xlm-roberta"

THREAT_CUES = re.compile(
    r"\b(kill|hurt|stab|shoot|burn|bomb|beat|rape|destroy|attack|threaten|lynch)\b", re.I
)

@dataclass
class DetectionConfig:
    """
    Configuration for legacy HarmfulTextDetector
    
    ⚠️  DEPRECATED: Use PipelineConfig with ZeroHarmPipeline instead
    
    This is kept for backward compatibility with v0.1.x code.
    """
    model_name: str = DEFAULT_MODEL
    threshold_per_label: float = 0.5
    overall_threshold: float = 0.5
    threat_min_score_on_cue: float = 0.6


class HarmfulTextDetector:
    """
    Legacy harmful content detector
    
    ⚠️  DEPRECATED: Use ZeroHarmPipeline from ai_detectors.py instead
    
    This class is kept for backward compatibility. For new code, use:
    
        from zero_harm_ai_detectors import ZeroHarmPipeline
        pipeline = ZeroHarmPipeline()
        result = pipeline.detect(text)
    
    Example (legacy usage):
        detector = HarmfulTextDetector()
        result = detector.detect("I hate you")
        print(result['harmful'])  # True
    """
    
    def __init__(self, config: DetectionConfig = DetectionConfig()):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for HarmfulTextDetector. "
                "Install with: pip install transformers torch"
            )
        
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            function_to_apply="sigmoid"
        )
        
        self.id2label: Dict[int, str] = self.model.config.id2label
        self.labels: List[str] = [self.id2label[i] for i in sorted(self.id2label.keys())]

    def _rules_boost(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply rule-based boosting for threat detection"""
        if THREAT_CUES.search(text):
            for k in scores:
                if k.lower() == "threat":
                    scores[k] = max(scores[k], self.config.threat_min_score_on_cue)
        return scores

    def score(self, text: str) -> Dict[str, float]:
        """Get scores for all harmful content labels"""
        raw = self.pipe(text)[0]
        scores = {item["label"]: float(item["score"]) for item in raw}
        scores = {k.strip(): v for k, v in scores.items()}
        scores = self._rules_boost(text, scores)
        return scores

    def detect(self, text: str) -> Dict:
        """
        Detect harmful content in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with:
                - text: Original text
                - harmful: Boolean indicating if content is harmful
                - severity: "low", "medium", or "high"
                - active_labels: List of detected harmful labels
                - scores: Dictionary of all label scores
        """
        scores = self.score(text)

        # Active labels
        active = {
            lbl: s for lbl, s in scores.items()
            if s >= self.config.threshold_per_label
        }

        # Overall harmful determination
        harmful = any(
            s >= self.config.overall_threshold
            for lbl, s in scores.items()
        )

        # Severity calculation
        active_scores = sorted([s for l, s in scores.items()], reverse=True)
        severity = "low"
        if active_scores:
            p90 = active_scores[max(0, math.floor(0.9 * (len(active_scores)-1)))]
            if p90 >= 0.85:
                severity = "high"
            elif p90 >= 0.6:
                severity = "medium"

        # Sorted scores
        top = sorted(
            [(lbl, float(score)) for lbl, score in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "text": text,
            "harmful": harmful,
            "severity": severity,
            "active_labels": list(active.keys()),
            "scores": {lbl: round(sc, 4) for lbl, sc in top}
        }
