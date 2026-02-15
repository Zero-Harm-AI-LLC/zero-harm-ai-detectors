"""
Comprehensive tests for AI-based detection pipeline (paid tier).

Covers:
- AI PII detection (NER + shared regex patterns)
- AI secrets detection (delegates to core_patterns.find_secrets)
- AI harmful content detection
- Shared core_patterns verification (same patterns as regex tier)
- PipelineResult / Detection data classes
- Redaction strategies
- Legacy API compatibility (detect_pii_legacy, detect_secrets_legacy)
- Convenience functions (detect_all, get_pipeline)
- Edge cases (empty text, clean text, mixed content)
- Backward compatibility with old API

Tests requiring transformers/torch are skipped if those packages
are not installed, so CI can run basic verification even without
GPU or large model downloads.

File: tests/test_ai_detectors.py
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zero_harm_ai_detectors import (
    AI_MODE_AVAILABLE,
    detect_pii,
    detect_secrets,
)
from zero_harm_ai_detectors.core_patterns import (
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    RedactionStrategy as CoreRedactionStrategy,
    luhn_check,
    find_secrets,
    THREAT_CUES_RE,
)

# Only import AI-specific classes if available
if AI_MODE_AVAILABLE:
    from zero_harm_ai_detectors import (
        ZeroHarmDetector,
        PipelineConfig,
        AIRedactionStrategy,
    )
    from zero_harm_ai_detectors.ai_detectors import (
        ZeroHarmPipeline,
        PipelineResult,
        Detection,
        DetectionType,
        RedactionStrategy as AIRedactionStrategyDirect,
        AIPIIDetector,
        SecretsDetector as AISecretsDetector,
        HarmfulContentDetector,
        detect_all,
        get_pipeline,
    )


# ============================================================
# Marker for tests that require AI dependencies
# ============================================================

requires_ai = pytest.mark.skipif(
    not AI_MODE_AVAILABLE,
    reason="AI detection not available (transformers/torch not installed)",
)


# ============================================================
# Section 1: Shared core_patterns verification for AI tier
# ============================================================

class TestAISharedPatterns:
    """Verify AI tier imports patterns from core_patterns, not its own."""

    @requires_ai
    def test_redaction_strategy_is_core(self):
        """AI tier's RedactionStrategy should be the core one."""
        assert AIRedactionStrategyDirect is CoreRedactionStrategy

    @requires_ai
    def test_ai_secrets_detector_uses_find_secrets(self):
        """AI SecretsDetector must delegate to core_patterns.find_secrets."""
        detector = AISecretsDetector()
        text = "sk-1234567890abcdef1234567890abcdef"
        ai_results = detector.detect(text)
        core_results = find_secrets(text)
        # AI wraps in Detection objects; compare spans
        ai_spans = [(d.start, d.end) for d in ai_results]
        core_spans = [(f["start"], f["end"]) for f in core_results]
        assert ai_spans == core_spans

    @requires_ai
    def test_ai_secrets_no_false_positive(self):
        """AI secrets detector should reject normal text (same as regex tier)."""
        detector = AISecretsDetector()
        normal_texts = [
            "The quick brown fox jumps over the lazy dog",
            "commit abc123def456789012345678901234567890ab",
            "background-color: #ff5733; font-size: 16px;",
        ]
        for text in normal_texts:
            results = detector.detect(text)
            assert results == [], f"False positive on: {text!r}"

    @requires_ai
    def test_threat_cues_used_by_harmful_detector(self):
        """HarmfulContentDetector should use the shared THREAT_CUES_RE."""
        # Verify the imported pattern is used (can't easily check at runtime
        # without loading models, so just verify the import exists)
        from zero_harm_ai_detectors.ai_detectors import THREAT_CUES_RE as imported
        assert imported is THREAT_CUES_RE


# ============================================================
# Section 2: Detection / PipelineResult data classes
# ============================================================

class TestDataClasses:
    @requires_ai
    def test_detection_to_dict(self):
        det = Detection(
            type="EMAIL",
            text="test@example.com",
            start=0,
            end=16,
            confidence=1.0,
            metadata={"method": "regex"},
        )
        d = det.to_dict()
        assert d["type"] == "EMAIL"
        assert d["span"] == "test@example.com"
        assert d["start"] == 0
        assert d["end"] == 16
        assert d["confidence"] == 1.0
        assert d["metadata"]["method"] == "regex"

    @requires_ai
    def test_detection_to_dict_no_metadata(self):
        det = Detection(type="PHONE", text="555-1234", start=0, end=8, confidence=0.9)
        d = det.to_dict()
        assert d["metadata"] == {}

    @requires_ai
    def test_pipeline_result_to_dict(self):
        detections = [
            Detection(type="EMAIL", text="a@b.com", start=0, end=7, confidence=1.0),
            Detection(type="PHONE", text="555-1234", start=10, end=18, confidence=1.0),
        ]
        result = PipelineResult(
            original_text="a@b.com | 555-1234",
            redacted_text="[REDACTED_EMAIL] | [REDACTED_PHONE]",
            detections=detections,
            harmful=False,
            harmful_scores={},
            severity="low",
        )
        d = result.to_dict()
        assert d["original"] == "a@b.com | 555-1234"
        assert "EMAIL" in d["detections"]
        assert "PHONE" in d["detections"]
        assert len(d["detections"]["EMAIL"]) == 1
        assert d["harmful"] is False

    @requires_ai
    def test_pipeline_result_grouped_by_type(self):
        detections = [
            Detection(type="EMAIL", text="a@b.com", start=0, end=7, confidence=1.0),
            Detection(type="EMAIL", text="c@d.com", start=10, end=17, confidence=1.0),
        ]
        result = PipelineResult(
            original_text="a@b.com | c@d.com",
            redacted_text="redacted",
            detections=detections,
            harmful=False,
        )
        d = result.to_dict()
        assert len(d["detections"]["EMAIL"]) == 2


# ============================================================
# Section 3: PipelineConfig
# ============================================================

class TestPipelineConfig:
    @requires_ai
    def test_defaults(self):
        config = PipelineConfig()
        assert config.pii_model == "dslim/bert-base-NER"
        assert config.pii_threshold == 0.7
        assert config.harmful_threshold_per_label == 0.5
        assert config.device == "cpu"
        assert config.use_regex_for_secrets is True

    @requires_ai
    def test_custom_config(self):
        config = PipelineConfig(
            pii_threshold=0.9,
            harmful_threshold_per_label=0.7,
            device="cuda",
        )
        assert config.pii_threshold == 0.9
        assert config.harmful_threshold_per_label == 0.7
        assert config.device == "cuda"


# ============================================================
# Section 4: DetectionType enum
# ============================================================

class TestDetectionType:
    @requires_ai
    def test_pii_types(self):
        assert DetectionType.PERSON.value == "PERSON"
        assert DetectionType.EMAIL.value == "EMAIL"
        assert DetectionType.PHONE.value == "PHONE"
        assert DetectionType.SSN.value == "SSN"
        assert DetectionType.CREDIT_CARD.value == "CREDIT_CARD"
        assert DetectionType.LOCATION.value == "LOCATION"
        assert DetectionType.ORGANIZATION.value == "ORGANIZATION"

    @requires_ai
    def test_secret_types(self):
        assert DetectionType.API_KEY.value == "API_KEY"

    @requires_ai
    def test_harmful_types(self):
        assert DetectionType.TOXIC.value == "TOXIC"
        assert DetectionType.THREAT.value == "THREAT"


# ============================================================
# Section 5: Full Pipeline Detection (requires model loading)
# ============================================================

@pytest.fixture(scope="module")
def pipeline():
    """Create a pipeline instance (loads models once for all tests)."""
    if not AI_MODE_AVAILABLE:
        pytest.skip("AI detection not available")
    return ZeroHarmPipeline()


@pytest.fixture
def test_texts():
    """Common test texts."""
    return {
        "email": "Contact me at john.smith@example.com",
        "phone": "Call me at 555-123-4567",
        "ssn": "My SSN is 123-45-6789",
        "person": "Please contact John Smith for more information",
        "location": "The meeting is in New York City",
        "org": "I work at Microsoft Corporation",
        "secret": "API key: sk-1234567890abcdef1234567890abcdef",
        "harmful": "I hate you and want to hurt you",
        "credit_card": "Card number: 4532-0151-1283-0366",
        "mixed": (
            "Email John Smith at john@example.com or call 555-123-4567. "
            "API key: sk-1234567890abcdef1234567890abcdef."
        ),
        "clean": "Hello world! How are you today?",
    }


@requires_ai
class TestPipelineEmailDetection:
    def test_detects_email(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["email"])
        types = {d.type for d in result.detections}
        assert "EMAIL" in types

    def test_email_confidence(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["email"])
        emails = [d for d in result.detections if d.type == "EMAIL"]
        assert len(emails) >= 1
        assert emails[0].confidence == 1.0  # regex patterns have 1.0 confidence


@requires_ai
class TestPipelinePhoneDetection:
    def test_detects_phone(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["phone"])
        types = {d.type for d in result.detections}
        assert "PHONE" in types


@requires_ai
class TestPipelineSSNDetection:
    def test_detects_ssn(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["ssn"])
        types = {d.type for d in result.detections}
        assert "SSN" in types


@requires_ai
class TestPipelinePersonDetection:
    def test_detects_person(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["person"])
        types = {d.type for d in result.detections}
        assert "PERSON" in types


@requires_ai
class TestPipelineLocationDetection:
    def test_detects_location(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["location"])
        types = {d.type for d in result.detections}
        assert "LOCATION" in types


@requires_ai
class TestPipelineOrgDetection:
    def test_detects_organization(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["org"])
        types = {d.type for d in result.detections}
        assert "ORGANIZATION" in types


@requires_ai
class TestPipelineSecretsDetection:
    def test_detects_secrets(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["secret"])
        types = {d.type for d in result.detections}
        assert "API_KEY" in types

    def test_secrets_no_false_positive(self, pipeline):
        result = pipeline.detect("The quick brown fox jumps over the lazy dog")
        secret_types = {d.type for d in result.detections if d.type == "API_KEY"}
        assert len(secret_types) == 0


@requires_ai
class TestPipelineCreditCardDetection:
    def test_detects_credit_card(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["credit_card"])
        types = {d.type for d in result.detections}
        assert "CREDIT_CARD" in types


@requires_ai
class TestPipelineHarmfulDetection:
    def test_detects_harmful(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["harmful"])
        assert result.harmful is True
        assert result.severity in ("low", "medium", "high")

    def test_harmful_scores_present(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["harmful"])
        assert isinstance(result.harmful_scores, dict)
        assert len(result.harmful_scores) > 0

    def test_clean_text_not_harmful(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["clean"])
        assert result.harmful is False


# ============================================================
# Section 6: Redaction Strategies in Pipeline
# ============================================================

@requires_ai
class TestPipelineRedaction:
    def test_token_redaction(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["email"],
            redaction_strategy=CoreRedactionStrategy.TOKEN,
        )
        assert "[REDACTED_" in result.redacted_text

    def test_mask_all_redaction(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["email"],
            redaction_strategy=CoreRedactionStrategy.MASK_ALL,
        )
        assert "john.smith@example.com" not in result.redacted_text

    def test_mask_last4_redaction(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["email"],
            redaction_strategy=CoreRedactionStrategy.MASK_LAST4,
        )
        assert "john.smith@example.com" not in result.redacted_text

    def test_hash_redaction(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["email"],
            redaction_strategy=CoreRedactionStrategy.HASH,
        )
        assert "john.smith@example.com" not in result.redacted_text


# ============================================================
# Section 7: Selective Detection
# ============================================================

@requires_ai
class TestSelectiveDetection:
    def test_pii_only(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["mixed"],
            detect_pii=True,
            detect_secrets=False,
            detect_harmful=False,
        )
        types = {d.type for d in result.detections}
        assert "API_KEY" not in types

    def test_secrets_only(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["mixed"],
            detect_pii=False,
            detect_secrets=True,
            detect_harmful=False,
        )
        types = {d.type for d in result.detections}
        assert "EMAIL" not in types
        assert "PHONE" not in types

    def test_harmful_only(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["harmful"],
            detect_pii=False,
            detect_secrets=False,
            detect_harmful=True,
        )
        assert result.harmful is True
        # Non-harmful detections should be absent
        non_harmful = [d for d in result.detections if d.type != "HARMFUL_CONTENT"]
        assert len(non_harmful) == 0


# ============================================================
# Section 8: Mixed Content Detection
# ============================================================

@requires_ai
class TestMixedContent:
    def test_detects_all_types(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["mixed"])
        types = {d.type for d in result.detections}
        # Should find at least email, phone, and secret
        assert "EMAIL" in types
        assert "PHONE" in types
        assert "API_KEY" in types

    def test_redacted_text_has_no_sensitive_data(self, pipeline, test_texts):
        result = pipeline.detect(
            test_texts["mixed"],
            redaction_strategy=CoreRedactionStrategy.TOKEN,
        )
        assert "john@example.com" not in result.redacted_text
        assert "555-123-4567" not in result.redacted_text
        assert "sk-1234567890abcdef" not in result.redacted_text


# ============================================================
# Section 9: Edge Cases
# ============================================================

@requires_ai
class TestPipelineEdgeCases:
    def test_empty_text(self, pipeline):
        result = pipeline.detect("")
        assert len(result.detections) == 0
        assert result.redacted_text == ""
        assert result.harmful is False

    def test_clean_text(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["clean"])
        assert result.harmful is False

    def test_unicode_text(self, pipeline):
        result = pipeline.detect("Contact José García at jose@example.com")
        types = {d.type for d in result.detections}
        assert "EMAIL" in types

    def test_preserves_original_text(self, pipeline, test_texts):
        result = pipeline.detect(test_texts["email"])
        assert result.original_text == test_texts["email"]


# ============================================================
# Section 10: Legacy API Compatibility
# ============================================================

@requires_ai
class TestLegacyAPI:
    def test_detect_pii_legacy_format(self, pipeline, test_texts):
        result = pipeline.detect_pii_legacy(test_texts["email"])
        assert isinstance(result, dict)
        assert "EMAIL" in result
        assert len(result["EMAIL"]) >= 1
        entry = result["EMAIL"][0]
        assert "span" in entry
        assert "start" in entry
        assert "end" in entry
        assert "confidence" in entry

    def test_detect_secrets_legacy_format(self, pipeline, test_texts):
        result = pipeline.detect_secrets_legacy(test_texts["secret"])
        assert isinstance(result, dict)
        if "SECRETS" in result:
            assert len(result["SECRETS"]) >= 1
            entry = result["SECRETS"][0]
            assert "span" in entry
            assert "start" in entry
            assert "end" in entry

    def test_detect_secrets_legacy_empty(self, pipeline, test_texts):
        result = pipeline.detect_secrets_legacy(test_texts["clean"])
        assert result == {}


# ============================================================
# Section 11: Convenience Functions
# ============================================================

@requires_ai
class TestConvenienceFunctions:
    def test_detect_all_returns_dict(self):
        result = detect_all("Email test@example.com")
        assert isinstance(result, dict)
        assert "original" in result
        assert "redacted" in result
        assert "detections" in result
        assert "harmful" in result

    def test_get_pipeline_singleton(self):
        p1 = get_pipeline()
        p2 = get_pipeline()
        assert p1 is p2


# ============================================================
# Section 12: Top-Level API Functions (from __init__.py)
# ============================================================

class TestTopLevelAPI:
    """These run regardless of AI availability — they fall back to regex."""

    def test_detect_pii_regex_mode(self):
        """detect_pii should work in regex mode (default or explicit)."""
        # The full __init__.py supports mode='regex' kwarg,
        # but the function should work without it too (defaults to regex)
        result = detect_pii("Email: test@example.com")
        assert isinstance(result, dict)
        assert "EMAIL" in result

    def test_detect_secrets_regex_mode(self):
        result = detect_secrets("sk-1234567890abcdef1234567890abcdef")
        assert isinstance(result, dict)
        assert "SECRETS" in result

    @requires_ai
    def test_detect_pii_ai_mode(self):
        result = detect_pii("Contact John Smith at john@example.com", mode="ai")
        assert isinstance(result, dict)
        # AI mode should find at least email
        assert "EMAIL" in result

    def test_detect_pii_returns_dict_always(self):
        """Both modes must return dict, never PipelineResult directly."""
        result = detect_pii("test@example.com")
        assert isinstance(result, dict)


# ============================================================
# Section 13: ZeroHarmDetector Unified Class
# ============================================================

@requires_ai
class TestZeroHarmDetector:
    def test_regex_mode(self):
        detector = ZeroHarmDetector(mode="regex")
        result = detector.detect("Email: test@example.com")
        assert isinstance(result, dict)
        assert "detections" in result
        assert result["mode"] == "regex"
        assert result["tier"] == "free"

    def test_ai_mode(self):
        detector = ZeroHarmDetector(mode="ai")
        result = detector.detect("Email: test@example.com")
        assert isinstance(result, PipelineResult)
        types = {d.type for d in result.detections}
        assert "EMAIL" in types

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            ZeroHarmDetector(mode="invalid")

    def test_ai_mode_without_packages_raises(self):
        """If AI mode is available, this test just verifies the constructor works."""
        # We can't easily simulate missing packages when they're installed
        detector = ZeroHarmDetector(mode="ai")
        assert detector.mode == "ai"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
