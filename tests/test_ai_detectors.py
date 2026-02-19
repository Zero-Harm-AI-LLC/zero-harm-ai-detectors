"""
Tests for AI-enhanced detection (mode='ai').

These tests are skipped if AI dependencies are not installed.

File: tests/test_ai_detectors.py
"""
import pytest
from zero_harm_ai_detectors import (
    detect,
    AI_AVAILABLE,
    DetectionResult,
)

# Skip all tests in this module if AI is not available
pytestmark = pytest.mark.skipif(
    not AI_AVAILABLE,
    reason="AI dependencies not installed (pip install zero_harm_ai_detectors[ai])"
)


class TestAIMode:
    """Tests for AI mode detection."""
    
    def test_ai_mode_returns_detection_result(self):
        result = detect("Contact John Smith", mode="ai")
        assert isinstance(result, DetectionResult)
        assert result.mode == "ai"
    
    def test_ai_mode_detects_person_names(self):
        result = detect("Please contact John Smith for assistance", mode="ai")
        persons = [d for d in result.detections if d.type == "PERSON"]
        # AI should detect at least one person
        assert len(persons) >= 1
    
    def test_ai_mode_detects_locations(self):
        result = detect("Our office is in New York City", mode="ai")
        locations = [d for d in result.detections if d.type == "LOCATION"]
        # AI should detect the location
        assert len(locations) >= 1
    
    def test_ai_mode_detects_organizations(self):
        result = detect("I work at Microsoft", mode="ai")
        orgs = [d for d in result.detections if d.type == "ORGANIZATION"]
        # AI should detect the organization
        assert len(orgs) >= 1
    
    def test_ai_mode_still_detects_structured_pii(self):
        result = detect("Email: test@example.com, Phone: 555-123-4567", mode="ai")
        types = {d.type for d in result.detections}
        # Structured PII should still be detected (via regex)
        assert "EMAIL" in types
        assert "PHONE" in types
    
    def test_ai_mode_detects_secrets(self):
        result = detect("API key: sk-1234567890abcdef1234567890abcdef", mode="ai")
        # Secrets should be detected (via regex)
        types = {d.type for d in result.detections}
        assert "API_KEY" in types or "SECRET" in types
    
    def test_ai_mode_harmful_detection(self):
        result = detect("I hate you so much!", mode="ai", detect_pii=False, detect_secrets=False)
        # AI harmful detection should work
        assert result.harmful is True or result.severity != "none"
    
    def test_ai_mode_clean_text(self):
        result = detect("Hello, how are you today?", mode="ai", detect_pii=False, detect_secrets=False)
        assert result.harmful is False or result.severity == "none"
    
    def test_ai_mode_mixed_content(self):
        text = """
        Contact: John Smith
        Email: john.smith@example.com
        Company: Acme Corporation
        Location: San Francisco, CA
        """
        result = detect(text, mode="ai")
        
        types = {d.type for d in result.detections}
        # Should detect both AI entities and structured PII
        assert "EMAIL" in types  # Regex
        assert "PERSON" in types or "ORGANIZATION" in types  # AI
    
    def test_ai_mode_redaction(self):
        result = detect("Contact John Smith at john@example.com", mode="ai")
        # Should have redacted text
        assert "[REDACTED_" in result.redacted_text


class TestAIConfig:
    """Tests for AI configuration."""
    
    def test_custom_config(self):
        from zero_harm_ai_detectors import AIConfig
        
        config = AIConfig(
            ner_threshold=0.9,
            device="cpu",
        )
        
        result = detect(
            "Contact John Smith",
            mode="ai",
            ai_config=config,
        )
        
        assert result.mode == "ai"
    
    def test_high_threshold_reduces_detections(self):
        from zero_harm_ai_detectors import AIConfig
        
        text = "Contact John Smith at Microsoft"
        
        # Low threshold
        config_low = AIConfig(ner_threshold=0.5)
        result_low = detect(text, mode="ai", ai_config=config_low)
        
        # High threshold
        config_high = AIConfig(ner_threshold=0.99)
        result_high = detect(text, mode="ai", ai_config=config_high)
        
        # High threshold should have same or fewer detections
        assert len(result_high.detections) <= len(result_low.detections)


class TestAIPipeline:
    """Tests for AIPipeline class."""
    
    def test_get_pipeline(self):
        from zero_harm_ai_detectors import get_pipeline
        
        pipeline = get_pipeline()
        assert pipeline is not None
    
    def test_pipeline_detect(self):
        from zero_harm_ai_detectors import get_pipeline
        
        pipeline = get_pipeline()
        result = pipeline.detect("Contact John Smith at john@example.com")
        
        assert isinstance(result, DetectionResult)
        assert result.mode == "ai"


class TestAIvsRegexComparison:
    """Tests comparing AI and regex mode results."""
    
    def test_both_modes_return_same_structure(self):
        text = "Contact john@example.com"
        
        result_regex = detect(text, mode="regex")
        result_ai = detect(text, mode="ai")
        
        # Same structure
        assert hasattr(result_regex, 'original_text')
        assert hasattr(result_ai, 'original_text')
        assert hasattr(result_regex, 'redacted_text')
        assert hasattr(result_ai, 'redacted_text')
        assert hasattr(result_regex, 'detections')
        assert hasattr(result_ai, 'detections')
        assert hasattr(result_regex, 'mode')
        assert hasattr(result_ai, 'mode')
    
    def test_ai_detects_more_person_names(self):
        text = "Please contact John Smith or Sarah Johnson for help"
        
        result_regex = detect(text, mode="regex")
        result_ai = detect(text, mode="ai")
        
        regex_persons = [d for d in result_regex.detections if d.type == "PERSON"]
        ai_persons = [d for d in result_ai.detections if d.type == "PERSON"]
        
        # AI should typically detect more or equal person names
        # (This is a soft assertion since regex might catch some)
        assert len(ai_persons) >= len(regex_persons) or len(ai_persons) > 0
    
    def test_structured_pii_same_in_both_modes(self):
        text = "Email: test@example.com, SSN: 123-45-6789"
        
        result_regex = detect(text, mode="regex")
        result_ai = detect(text, mode="ai")
        
        regex_emails = [d for d in result_regex.detections if d.type == "EMAIL"]
        ai_emails = [d for d in result_ai.detections if d.type == "EMAIL"]
        
        # Both should detect the same emails (both use regex for structured)
        assert len(regex_emails) == len(ai_emails)


class TestNERDetector:
    """Tests for NERDetector class."""
    
    def test_ner_detector_creation(self):
        from zero_harm_ai_detectors import NERDetector, AIConfig
        
        config = AIConfig()
        detector = NERDetector(config)
        assert detector is not None
    
    def test_ner_detector_detect(self):
        from zero_harm_ai_detectors import NERDetector
        
        detector = NERDetector()
        results = detector.detect("John Smith works at Google in Seattle")
        
        assert isinstance(results, list)
        # Should detect at least one entity
        assert len(results) > 0


class TestHarmfulContentDetector:
    """Tests for HarmfulContentDetector class."""
    
    def test_harmful_detector_creation(self):
        from zero_harm_ai_detectors import HarmfulContentDetector, AIConfig
        
        config = AIConfig()
        detector = HarmfulContentDetector(config)
        assert detector is not None
    
    def test_harmful_detector_detect(self):
        from zero_harm_ai_detectors import HarmfulContentDetector
        
        detector = HarmfulContentDetector()
        result = detector.detect("I hate you!")
        
        assert isinstance(result, dict)
        assert "harmful" in result
        assert "severity" in result
        assert "scores" in result
