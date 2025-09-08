# file: harmful_text_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math, regex as re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Default multi-label model trained on Jigsaw categories.
# This model commonly exposes labels like:
# ["toxic","severe_toxic","obscene","threat","insult","identity_hate"].
DEFAULT_MODEL = "unitary/multilingual-toxic-xlm-roberta"

THREAT_CUES = re.compile(
    r"\b(kill|hurt|stab|shoot|burn|bomb|beat|rape|destroy|attack|threaten|lynch)\b", re.I
)

@dataclass
class DetectionConfig:
    model_name: str = DEFAULT_MODEL
    # Score above which a category is considered present
    threshold_per_label: float = 0.5
    # If any label exceeds this, mark overall harmful=True
    overall_threshold: float = 0.5
    # Optional boost when threat patterns are present (applied as max(existing, boost))
    threat_min_score_on_cue: float = 0.6

class HarmfulTextDetector:
    def __init__(self, config: DetectionConfig = DetectionConfig()):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,            # returns list of {label, score}
            function_to_apply="sigmoid"        # multi-label style
        )
        # Build label list from model config (robust to model variants)
        self.id2label: Dict[int, str] = self.model.config.id2label
        self.labels: List[str] = [self.id2label[i] for i in sorted(self.id2label.keys())]

    def _rules_boost(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        # Threat cue → minimum score for "threat" if label exists
        if THREAT_CUES.search(text):
            for k in scores:
                if k.lower() == "threat":
                    scores[k] = max(scores[k], self.config.threat_min_score_on_cue)

        return scores

    def score(self, text: str) -> Dict[str, float]:
        # Get model scores (multi-label sigmoid). Pipeline returns [[{label,score}...]]
        raw = self.pipe(text)[0]
        scores = {item["label"]: float(item["score"]) for item in raw}
        # Normalize label casing (make consistent)
        scores = {k.strip(): v for k, v in scores.items()}

        # Optional rules to improve recall for certain classes
        scores = self._rules_boost(text, scores)
        return scores

    def detect(self, text: str) -> Dict:
        scores = self.score(text)

        # Decide which labels are "active"
        active = {
            lbl: s for lbl, s in scores.items()
            if s >= self.config.threshold_per_label
        }

        harmful = any(
            s >= self.config.overall_threshold
            for lbl, s in scores.items()
        )

        # Severity heuristic (P90 of active scores)
        active_scores = sorted([s for l, s in scores.items()], reverse=True)
        severity = "low"
        if active_scores:
            p90 = active_scores[max(0, math.floor(0.9 * (len(active_scores)-1)))]
            if p90 >= 0.85: severity = "high"
            elif p90 >= 0.6: severity = "medium"

        # Build a tidy result
        top = sorted(
            [(lbl, float(score)) for lbl, score in scores.items()],
            key=lambda x: x[1], reverse=True
        )

        return {
            "text": text,
            "harmful": harmful,
            "severity": severity,
            "active_labels": list(active.keys()),
            "scores": {lbl: round(sc, 4) for lbl, sc in top}
        }