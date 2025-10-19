from __future__ import annotations

"""
Lightweight harmful content detector using regex patterns (v0.1.x compatible)

This detector provides fast, regex-based harmful content detection without
requiring AI models. It's designed for:
1. Backward compatibility with v0.1.x code
2. Fast detection without loading transformer models
3. Low-resource environments

For more accurate AI-based detection, use ZeroHarmPipeline:
    from zero_harm_ai_detectors import ZeroHarmPipeline
    pipeline = ZeroHarmPipeline()
    result = pipeline.detect(text)

File: zero_harm_ai_detectors/harmful_detectors.py
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Any

# ==================== Configuration ====================
@dataclass
class DetectionConfig:
    """
    Configuration for HarmfulTextDetector
    
    For backward compatibility with v0.1.x code.
    """
    threshold_per_label: float = 0.5  # Not used in regex mode, kept for compatibility
    overall_threshold: float = 0.5     # Not used in regex mode, kept for compatibility
    threat_min_score_on_cue: float = 0.6  # Not used in regex mode, kept for compatibility


# ==================== Pattern Definitions ====================
class HarmfulPatterns:
    """Regex patterns for detecting harmful content"""
    
    # Toxic/Offensive Language
    TOXIC = re.compile(
        r'\b(fuck|shit|damn|hell|ass|bitch|bastard|crap|piss|whore|slut|'
        r'dickhead|asshole|dumbass|jackass|moron|idiot|stupid)\b',
        re.IGNORECASE
    )
    
    # Threats and Violence
    THREAT = re.compile(
        r'\b(kill|murder|hurt|harm|stab|shoot|attack|destroy|beat|punch|'
        r'kick|slap|hit|strike|assault|rape|torture|mutilate|strangle|'
        r'choke|bomb|explode|burn|lynch|hang|drown|suffocate|poison)\b',
        re.IGNORECASE
    )
    
    # Threat phrases (more specific)
    THREAT_PHRASES = re.compile(
        r'\b(going to (kill|hurt|harm|attack|destroy)|'
        r'i will (kill|hurt|harm|attack|destroy)|'
        r'i\'ll (kill|hurt|harm|attack|destroy)|'
        r'gonna (kill|hurt|harm|attack|destroy)|'
        r'want (you|to) dead|'
        r'should (kill|die|be killed)|'
        r'deserve to (die|be killed|suffer)|'
        r'hope you (die|get killed|suffer))\b',
        re.IGNORECASE
    )
    
    # Insults and Personal Attacks
    INSULT = re.compile(
        r'\b(stupid|idiot|moron|dumb|retard|loser|pathetic|worthless|'
        r'useless|garbage|trash|scum|vermin|disgusting|repulsive|'
        r'ugly|fat|pig|slob|freak|creep|weirdo|psycho|crazy|insane)\b',
        re.IGNORECASE
    )
    
    # Hate Speech - Identity-based attacks
    IDENTITY_HATE = re.compile(
        r'\b(fag|faggot|dyke|tranny|nigger|nigga|chink|gook|spic|wetback|'
        r'kike|beaner|raghead|towelhead|terrorist|nazi|supremacist)\b',
        re.IGNORECASE
    )
    
    # Obscene/Sexual Content
    OBSCENE = re.compile(
        r'\b(cock|dick|penis|pussy|vagina|cunt|tits|boobs|sex|porn|'
        r'masturbate|orgasm|fuck|screw|bang|hump|cum)\b',
        re.IGNORECASE
    )
    
    # Intensity boosters (increase severity)
    INTENSITY = re.compile(
        r'\b(fucking|really|very|extremely|totally|completely|absolutely|'
        r'so much|hate|despise|loathe)\b',
        re.IGNORECASE
    )


# ==================== Detector Class ====================
class HarmfulTextDetector:
    """
    Lightweight harmful content detector using regex patterns
    
    This detector provides fast, regex-based detection compatible with
    detect_pii and detect_secrets API format.
    
    Example:
        detector = HarmfulTextDetector()
        findings = detector.detect("I hate you, you stupid idiot!")
        # Returns: {'HARMFUL_CONTENT': [{'span': '...', 'start': 0, 'end': 31, ...}]}
    """
    
    def __init__(self, config: DetectionConfig = None):
        """
        Initialize the detector
        
        Args:
            config: Optional configuration (kept for backward compatibility)
        """
        self.config = config or DetectionConfig()
        self.patterns = HarmfulPatterns()
    
    def _calculate_severity(
        self, 
        text: str,
        toxic_count: int,
        threat_count: int,
        insult_count: int,
        hate_count: int,
        obscene_count: int
    ) -> str:
        """
        Calculate severity based on pattern matches
        
        Returns:
            "low", "medium", or "high"
        """
        total_matches = toxic_count + threat_count + insult_count + hate_count + obscene_count
        
        # High severity conditions
        if hate_count > 0:  # Any hate speech is high severity
            return "high"
        if threat_count >= 2:  # Multiple threats
            return "high"
        if total_matches >= 5:  # Many harmful terms
            return "high"
        
        # Medium severity conditions
        if threat_count >= 1:  # Any threat
            return "medium"
        if total_matches >= 3:  # Several harmful terms
            return "medium"
        if obscene_count >= 2:  # Multiple obscene terms
            return "medium"
        
        # Low severity (but still harmful)
        if total_matches > 0:
            return "low"
        
        return "low"
    
    def _get_active_labels(
        self,
        toxic_count: int,
        threat_count: int,
        insult_count: int,
        hate_count: int,
        obscene_count: int
    ) -> List[str]:
        """Get list of active harmful content labels"""
        labels = []
        if toxic_count > 0:
            labels.append("toxic")
        if threat_count > 0:
            labels.append("threat")
        if insult_count > 0:
            labels.append("insult")
        if hate_count > 0:
            labels.append("identity_hate")
        if obscene_count > 0:
            labels.append("obscene")
        return labels
    
    def _generate_scores(
        self,
        toxic_count: int,
        threat_count: int,
        insult_count: int,
        hate_count: int,
        obscene_count: int
    ) -> Dict[str, float]:
        """
        Generate pseudo-confidence scores based on pattern matches
        
        These are heuristic scores for compatibility with AI detector format
        """
        # Base scores (0.0-1.0)
        scores = {
            "toxic": min(1.0, 0.3 + (toxic_count * 0.15)),
            "threat": min(1.0, 0.4 + (threat_count * 0.2)),
            "insult": min(1.0, 0.3 + (insult_count * 0.15)),
            "identity_hate": min(1.0, 0.5 + (hate_count * 0.3)),
            "obscene": min(1.0, 0.3 + (obscene_count * 0.15))
        }
        
        return scores
    
    def detect(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect harmful content in text
        
        Returns format compatible with detect_pii and detect_secrets:
        {
            "HARMFUL_CONTENT": [
                {
                    "span": "entire text with harmful content",
                    "start": 0,
                    "end": length,
                    "severity": "low|medium|high",
                    "labels": ["toxic", "threat", ...],
                    "scores": {"toxic": 0.5, "threat": 0.7, ...},
                    "harmful": True  # Added for backward compatibility
                }
            ]
        }
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with HARMFUL_CONTENT key if harmful content detected
        """
        # Count matches for each category
        toxic_matches = self.patterns.TOXIC.findall(text)
        threat_matches = self.patterns.THREAT.findall(text)
        threat_phrase_matches = self.patterns.THREAT_PHRASES.findall(text)
        insult_matches = self.patterns.INSULT.findall(text)
        hate_matches = self.patterns.IDENTITY_HATE.findall(text)
        obscene_matches = self.patterns.OBSCENE.findall(text)
        
        toxic_count = len(toxic_matches)
        threat_count = len(threat_matches) + len(threat_phrase_matches)
        insult_count = len(insult_matches)
        hate_count = len(hate_matches)
        obscene_count = len(obscene_matches)
        
        # Check if any harmful content detected
        total_matches = toxic_count + threat_count + insult_count + hate_count + obscene_count
        
        if total_matches == 0:
            return {}
        
        # Calculate severity and get labels
        severity = self._calculate_severity(
            text, toxic_count, threat_count, insult_count, hate_count, obscene_count
        )
        active_labels = self._get_active_labels(
            toxic_count, threat_count, insult_count, hate_count, obscene_count
        )
        scores = self._generate_scores(
            toxic_count, threat_count, insult_count, hate_count, obscene_count
        )
        
        # Return in detect_pii/detect_secrets compatible format
        return {
            "HARMFUL_CONTENT": [
                {
                    "span": text,
                    "start": 0,
                    "end": len(text),
                    "severity": severity,
                    "labels": active_labels,
                    "scores": scores,
                    "harmful": True,  # For backward compatibility
                    "match_counts": {
                        "toxic": toxic_count,
                        "threat": threat_count,
                        "insult": insult_count,
                        "identity_hate": hate_count,
                        "obscene": obscene_count
                    }
                }
            ]
        }
    
    def score(self, text: str) -> Dict[str, float]:
        """
        Get scores for all harmful content labels (legacy compatibility)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of label scores
        """
        result = self.detect(text)
        
        if not result:
            return {
                "toxic": 0.0,
                "threat": 0.0,
                "insult": 0.0,
                "identity_hate": 0.0,
                "obscene": 0.0
            }
        
        return result["HARMFUL_CONTENT"][0]["scores"]


# ==================== Standalone Detection Function ====================
def detect_harmful(text: str, config: DetectionConfig = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Standalone function to detect harmful content (compatible with detect_pii/detect_secrets)
    
    Args:
        text: Input text to analyze
        config: Optional configuration
        
    Returns:
        Dictionary with HARMFUL_CONTENT key if harmful content detected
        
    Example:
        findings = detect_harmful("I hate you!")
        # Returns: {'HARMFUL_CONTENT': [{'span': 'I hate you!', ...}]}
    """
    detector = HarmfulTextDetector(config)
    return detector.detect(text)
