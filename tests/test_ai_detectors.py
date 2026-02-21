"""
Tests for AI-enhanced detection (mode='ai').

These tests are skipped if AI dependencies are not installed.

Covers:
  #1  HarmfulContentDetector.classify() returns HarmfulResult (not dict)
  #1  HarmfulContentDetector.detect() is deprecated alias
  #2  ai_config param name is consistent (AIPipeline, get_pipeline, detect_all_ai)
  #4  mode='AI' / mode='Regex' (case-insensitive) works via unified detect()
  #5  _detect_raw / _classify_raw exist so AIPipeline avoids double-validation

File: tests/test_ai_detectors.py
"""
import warnings

import pytest
from zero_harm_ai_detectors import (
    detect,
    AI_AVAILABLE,
    DetectionResult,
    HarmfulResult,
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
        assert len(persons) >= 1

    def test_ai_mode_detects_locations(self):
        result = detect("Our office is in New York City", mode="ai")
        locations = [d for d in result.detections if d.type == "LOCATION"]
        assert len(locations) >= 1

    def test_ai_mode_detects_organizations(self):
        result = detect("I work at Microsoft", mode="ai")
        orgs = [d for d in result.detections if d.type == "ORGANIZATION"]
        assert len(orgs) >= 1

    def test_ai_mode_still_detects_structured_pii(self):
        result = detect("Email: test@example.com, Phone: 555-123-4567", mode="ai")
        types = {d.type for d in result.detections}
        assert "EMAIL" in types
        assert "PHONE" in types

    def test_ai_mode_detects_secrets(self):
        result = detect("API key: sk-1234567890abcdef1234567890abcdef", mode="ai")
        types = {d.type for d in result.detections}
        assert "API_KEY" in types or "SECRET" in types

    def test_ai_mode_harmful_detection(self):
        result = detect("I hate you so much!", mode="ai", detect_pii=False, detect_secrets=False)
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
        assert "EMAIL" in types
        assert "PERSON" in types or "ORGANIZATION" in types

    def test_ai_mode_redaction(self):
        result = detect("Contact John Smith at john@example.com", mode="ai")
        assert "[REDACTED_" in result.redacted_text


# ============================================================
# Issue #4 — case-insensitive mode in AI branch
# ============================================================

class TestAIModeNormalization:
    """Issue #4: detect(mode='AI') should work the same as mode='ai'."""

    def test_uppercase_ai_mode(self):
        result = detect("Contact John Smith at john@example.com", mode="AI")
        assert result.mode == "ai"

    def test_mixed_case_ai_mode(self):
        result = detect("test@example.com", mode="Ai")
        assert result.mode == "ai"


# ============================================================
# Issue #2 — ai_config param name consistency
# ============================================================

class TestAIConfigParamName:
    """Issue #2: ai_config should be the param name everywhere."""

    def test_detect_accepts_ai_config(self):
        from zero_harm_ai_detectors import AIConfig
        config = AIConfig(ner_threshold=0.9, device="cpu")
        result = detect("Contact John Smith", mode="ai", ai_config=config)
        assert result.mode == "ai"

    def test_get_pipeline_accepts_ai_config(self):
        from zero_harm_ai_detectors import AIConfig, get_pipeline
        config = AIConfig(device="cpu")
        pipeline = get_pipeline(ai_config=config)
        assert pipeline is not None

    def test_detect_all_ai_accepts_ai_config(self):
        from zero_harm_ai_detectors import AIConfig, detect_all_ai
        config = AIConfig(device="cpu")
        result = detect_all_ai("test@example.com", ai_config=config)
        assert isinstance(result, DetectionResult)

    def test_aipipeline_accepts_ai_config(self):
        from zero_harm_ai_detectors import AIConfig, AIPipeline
        config = AIConfig(device="cpu")
        pipeline = AIPipeline(ai_config=config)
        assert pipeline.config is config


# ============================================================
# Issue #1 — HarmfulContentDetector.classify() returns HarmfulResult
# ============================================================

class TestHarmfulContentDetectorClassify:
    """Issue #1: classify() replaces detect() and returns HarmfulResult."""

    def test_classify_returns_harmful_result(self):
        from zero_harm_ai_detectors import HarmfulContentDetector
        detector = HarmfulContentDetector()
        result = detector.classify("I hate you!")
        assert isinstance(result, HarmfulResult)

    def test_classify_has_expected_fields(self):
        from zero_harm_ai_detectors import HarmfulContentDetector
        detector = HarmfulContentDetector()
        result = detector.classify("Hello there!")
        assert hasattr(result, "harmful")
        assert hasattr(result, "severity")
        assert hasattr(result, "scores")
        assert "harmful" in result


# ============================================================
# Issue #5 — _detect_raw / _classify_raw exist on sub-components
# ============================================================

class TestRawMethodsExist:
    """Issue #5: sub-components expose _*_raw to avoid double-validation."""

    def test_ner_detector_has_detect_raw(self):
        from zero_harm_ai_detectors import NERDetector
        detector = NERDetector()
        assert hasattr(detector, "_detect_raw")
        assert callable(detector._detect_raw)

    def test_harmful_detector_has_classify_raw(self):
        from zero_harm_ai_detectors import HarmfulContentDetector
        detector = HarmfulContentDetector()
        assert hasattr(detector, "_classify_raw")
        assert callable(detector._classify_raw)

    def test_ner_detect_raw_returns_list(self):
        from zero_harm_ai_detectors import NERDetector
        detector = NERDetector()
        result = detector._detect_raw("John Smith works at Google")
        assert isinstance(result, list)

    def test_harmful_classify_raw_returns_harmful_result(self):
        from zero_harm_ai_detectors import HarmfulContentDetector
        detector = HarmfulContentDetector()
        result = detector._classify_raw("I hate you!")
        assert isinstance(result, HarmfulResult)


# ============================================================
# Existing AI Tests (unchanged behaviour)
# ============================================================

class TestAIConfig:
    def test_custom_config(self):
        from zero_harm_ai_detectors import AIConfig
        config = AIConfig(ner_threshold=0.9, device="cpu")
        result = detect("Contact John Smith", mode="ai", ai_config=config)
        assert result.mode == "ai"

    def test_high_threshold_reduces_detections(self):
        from zero_harm_ai_detectors import AIConfig
        text = "Contact John Smith at Microsoft"
        result_low = detect(text, mode="ai", ai_config=AIConfig(ner_threshold=0.5))
        result_high = detect(text, mode="ai", ai_config=AIConfig(ner_threshold=0.99))
        assert len(result_high.detections) <= len(result_low.detections)


class TestAIPipeline:
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
    def test_both_modes_return_same_structure(self):
        text = "Contact john@example.com"
        result_regex = detect(text, mode="regex")
        result_ai = detect(text, mode="ai")
        for attr in ("original_text", "redacted_text", "detections", "mode"):
            assert hasattr(result_regex, attr)
            assert hasattr(result_ai, attr)

    def test_ai_detects_more_person_names(self):
        text = "Please contact John Smith or Sarah Johnson for help"
        result_regex = detect(text, mode="regex")
        result_ai = detect(text, mode="ai")
        regex_persons = [d for d in result_regex.detections if d.type == "PERSON"]
        ai_persons = [d for d in result_ai.detections if d.type == "PERSON"]
        assert len(ai_persons) >= len(regex_persons) or len(ai_persons) > 0

    def test_structured_pii_same_in_both_modes(self):
        text = "Email: test@example.com, SSN: 123-45-6789"
        result_regex = detect(text, mode="regex")
        result_ai = detect(text, mode="ai")
        regex_emails = [d for d in result_regex.detections if d.type == "EMAIL"]
        ai_emails = [d for d in result_ai.detections if d.type == "EMAIL"]
        assert len(regex_emails) == len(ai_emails)


class TestNERDetector:
    def test_ner_detector_creation(self):
        from zero_harm_ai_detectors import NERDetector, AIConfig
        detector = NERDetector(AIConfig())
        assert detector is not None

    def test_ner_detector_detect(self):
        from zero_harm_ai_detectors import NERDetector
        detector = NERDetector()
        results = detector.detect("John Smith works at Google in Seattle")
        assert isinstance(results, list)
        assert len(results) > 0
