"""
Comprehensive tests for AI-based detection pipeline

File: tests/test_ai_detectors.py
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zero_harm_ai_detectors import (
    AI_DETECTION_AVAILABLE,
    detect_all_threats,
    detect_pii,
    detect_secrets,
)

# Only import AI-specific classes if available
if AI_DETECTION_AVAILABLE:
    from zero_harm_ai_detectors import (
        ZeroHarmPipeline,
        PipelineConfig,
        RedactionStrategy,
    )


@pytest.fixture
def pipeline():
    """Create a pipeline instance for testing"""
    if not AI_DETECTION_AVAILABLE:
        pytest.skip("AI detection not available")
    return ZeroHarmPipeline()


@pytest.fixture
def test_texts():
    """Common test texts"""
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
        "mixed": "Email John Smith at john@example.com or call 555-123-4567. API key: sk-1234567890abcdef1234567890abcdef."
    }


# ==================== Basic Detection Tests ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_email_detection(pipeline, test_texts):
    """Test email detection"""
    result = pipeline.detect(test_texts["email"])
    
    assert len(result.detections) > 0
    email_dets = [d for d in result.detections if d.type == "EMAIL"]
    assert len(email_dets) == 1
    assert "john.smith@example.com" in email_dets[0].text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_phone_detection(pipeline, test_texts):
    """Test phone number detection"""
    result = pipeline.detect(test_texts["phone"])
    
    phone_dets = [d for d in result.detections if d.type == "PHONE"]
    assert len(phone_dets) == 1
    assert "555-123-4567" in phone_dets[0].text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_ssn_detection(pipeline, test_texts):
    """Test SSN detection"""
    result = pipeline.detect(test_texts["ssn"])
    
    ssn_dets = [d for d in result.detections if d.type == "SSN"]
    assert len(ssn_dets) == 1
    assert "123-45-6789" in ssn_dets[0].text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_person_name_detection(pipeline, test_texts):
    """Test person name detection with AI"""
    result = pipeline.detect(test_texts["person"])
    
    person_dets = [d for d in result.detections if d.type == "PERSON"]
    assert len(person_dets) >= 1, f"Expected person detection, found: {result.detections}"
    
    detected_text = " ".join([d.text for d in person_dets])
    assert "John" in detected_text or "Smith" in detected_text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_location_detection(pipeline, test_texts):
    """Test location detection (AI feature)"""
    result = pipeline.detect(test_texts["location"])
    
    loc_dets = [d for d in result.detections if d.type == "LOCATION"]
    assert len(loc_dets) >= 1
    detected_text = " ".join([d.text for d in loc_dets])
    assert "New York" in detected_text or "York" in detected_text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_organization_detection(pipeline, test_texts):
    """Test organization detection (AI feature)"""
    result = pipeline.detect(test_texts["org"])
    
    org_dets = [d for d in result.detections if d.type == "ORGANIZATION"]
    assert len(org_dets) >= 1
    detected_text = " ".join([d.text for d in org_dets])
    assert "Microsoft" in detected_text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_secret_detection(pipeline, test_texts):
    """Test API key/secret detection"""
    result = pipeline.detect(test_texts["secret"])
    
    secret_dets = [d for d in result.detections if d.type == "API_KEY"]
    assert len(secret_dets) == 1
    assert "sk-" in secret_dets[0].text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_harmful_content_detection(pipeline, test_texts):
    """Test harmful content detection"""
    result = pipeline.detect(test_texts["harmful"])
    
    assert result.harmful is True
    assert result.severity in ["low", "medium", "high"]
    assert len(result.harmful_scores) > 0


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_credit_card_detection(pipeline, test_texts):
    """Test credit card detection"""
    result = pipeline.detect(test_texts["credit_card"])
    
    cc_dets = [d for d in result.detections if d.type == "CREDIT_CARD"]
    assert len(cc_dets) == 1


# ==================== Redaction Tests ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_token_redaction(pipeline, test_texts):
    """Test TOKEN redaction strategy"""
    result = pipeline.detect(
        test_texts["email"],
        redaction_strategy=RedactionStrategy.TOKEN
    )
    
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "john.smith@example.com" not in result.redacted_text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_mask_all_redaction(pipeline, test_texts):
    """Test MASK_ALL redaction strategy"""
    result = pipeline.detect(
        test_texts["phone"],
        redaction_strategy=RedactionStrategy.MASK_ALL
    )
    
    assert "*" in result.redacted_text
    assert "555-123-4567" not in result.redacted_text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_mask_last4_redaction(pipeline, test_texts):
    """Test MASK_LAST4 redaction strategy"""
    result = pipeline.detect(
        test_texts["phone"],
        redaction_strategy=RedactionStrategy.MASK_LAST4
    )
    
    assert "4567" in result.redacted_text or "67" in result.redacted_text


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_hash_redaction(pipeline, test_texts):
    """Test HASH redaction strategy"""
    result = pipeline.detect(
        test_texts["email"],
        redaction_strategy=RedactionStrategy.HASH
    )
    
    assert any(c in result.redacted_text for c in "0123456789abcdef")
    assert "john.smith@example.com" not in result.redacted_text


# ==================== Mixed Content Tests ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_mixed_content_detection(pipeline, test_texts):
    """Test detection of multiple types in one text"""
    result = pipeline.detect(test_texts["mixed"])
    
    detected_types = {d.type for d in result.detections}
    
    assert "EMAIL" in detected_types or "PERSON" in detected_types
    assert "PHONE" in detected_types
    assert "API_KEY" in detected_types
    assert len(result.detections) >= 3


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_multiple_same_type(pipeline):
    """Test detecting multiple instances of same type"""
    text = "Email john@example.com or jane@example.com"
    result = pipeline.detect(text)
    
    email_dets = [d for d in result.detections if d.type == "EMAIL"]
    assert len(email_dets) == 2


# ==================== Confidence Score Tests ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_confidence_scores(pipeline, test_texts):
    """Test that detections include confidence scores"""
    result = pipeline.detect(test_texts["email"])
    
    for detection in result.detections:
        assert hasattr(detection, 'confidence')
        assert 0.0 <= detection.confidence <= 1.0


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_high_confidence_filtering(pipeline, test_texts):
    """Test filtering by confidence threshold"""
    result = pipeline.detect(test_texts["mixed"])
    
    high_conf = [d for d in result.detections if d.confidence >= 0.9]
    assert len(high_conf) > 0


# ==================== Configuration Tests ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_custom_config():
    """Test custom pipeline configuration"""
    config = PipelineConfig(
        pii_threshold=0.8,
        harmful_threshold_per_label=0.6,
        harmful_overall_threshold=0.7
    )
    
    pipeline = ZeroHarmPipeline(config)
    
    assert pipeline.config.pii_threshold == 0.8
    assert pipeline.config.harmful_threshold_per_label == 0.6


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_selective_detection(pipeline, test_texts):
    """Test enabling/disabling detection types"""
    result = pipeline.detect(
        test_texts["mixed"],
        detect_pii=True,
        detect_secrets=False,
        detect_harmful=False
    )
    
    detected_types = {d.type for d in result.detections}
    assert "API_KEY" not in detected_types
    assert result.harmful is False


# ==================== Legacy API Tests ====================
def test_legacy_detect_pii(test_texts):
    """Test backward compatible detect_pii function"""
    # This should work with or without AI
    results = detect_pii(test_texts["email"], use_ai=False)
    
    assert isinstance(results, dict)
    assert "EMAIL" in results
    assert len(results["EMAIL"]) == 1


def test_legacy_detect_secrets(test_texts):
    """Test backward compatible detect_secrets function"""
    results = detect_secrets(test_texts["secret"])
    
    assert isinstance(results, dict)
    assert "SECRETS" in results
    assert len(results["SECRETS"]) == 1


def test_detect_all_threats_function(test_texts):
    """Test convenience function detect_all_threats"""
    # This should work with or without AI (falls back to regex)
    result = detect_all_threats(test_texts["mixed"], use_ai=False)
    
    assert "original" in result
    assert "redacted" in result
    assert "detections" in result
    assert "harmful" in result
    assert len(result["detections"]) > 0


# ==================== Edge Cases ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_empty_text(pipeline):
    """Test with empty text"""
    result = pipeline.detect("")
    
    assert len(result.detections) == 0
    assert result.redacted_text == ""


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_no_sensitive_content(pipeline):
    """Test with text containing no sensitive data"""
    result = pipeline.detect("Hello world! How are you today?")
    assert result.harmful is False


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_unicode_content(pipeline):
    """Test with unicode characters"""
    text = "Contact José at josé@example.com"
    result = pipeline.detect(text)
    
    email_dets = [d for d in result.detections if d.type == "EMAIL"]
    assert len(email_dets) >= 1


# ==================== Performance Tests ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_pipeline_reuse(pipeline, test_texts):
    """Test that pipeline can be reused efficiently"""
    for _ in range(5):
        result = pipeline.detect(test_texts["email"])
        assert len(result.detections) > 0


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_batch_consistency(pipeline, test_texts):
    """Test that same input gives consistent results"""
    results = []
    
    for _ in range(3):
        result = pipeline.detect(test_texts["email"])
        results.append(result)
    
    detection_counts = [len(r.detections) for r in results]
    assert len(set(detection_counts)) == 1


# ==================== Integration Tests ====================
@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_result_to_dict(pipeline, test_texts):
    """Test conversion of result to dictionary"""
    result = pipeline.detect(test_texts["mixed"])
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert "original" in result_dict
    assert "redacted" in result_dict
    assert "detections" in result_dict
    assert isinstance(result_dict["detections"], dict)


@pytest.mark.skipif(not AI_DETECTION_AVAILABLE, reason="AI detection not available")
def test_detection_to_dict(pipeline, test_texts):
    """Test detection to_dict method"""
    result = pipeline.detect(test_texts["email"])
    
    if result.detections:
        det_dict = result.detections[0].to_dict()
        
        assert "type" in det_dict
        assert "span" in det_dict
        assert "start" in det_dict
        assert "end" in det_dict
        assert "confidence" in det_dict


# ==================== Error Handling Tests ====================
def test_use_ai_flag():
    """Test use_ai flag for backward compatibility"""
    text = "Email: john@example.com"
    
    # When AI not available, both should use regex
    result_ai = detect_pii(text, use_ai=True)
    result_regex = detect_pii(text, use_ai=False)
    
    assert "EMAIL" in result_ai
    assert "EMAIL" in result_regex


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])