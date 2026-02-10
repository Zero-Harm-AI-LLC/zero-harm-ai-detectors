"""
Comprehensive tests for AI-based detection pipeline

File: tests/test_ai_detectors.py
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zero_harm_ai_detectors import (
    AI_MODE_AVAILABLE,
    detect_pii,
    detect_secrets,
)

# Only import AI-specific classes if available
if AI_MODE_AVAILABLE:
    from zero_harm_ai_detectors import (
        ZeroHarmDetector,
        PipelineConfig,
        AIRedactionStrategy,
    )


@pytest.fixture
def pipeline():
    """Create a pipeline instance for testing"""
    if not AI_MODE_AVAILABLE:
        pytest.skip("AI detection not available")
    return ZeroHarmDetector(mode='ai')


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
@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_email_detection(pipeline, test_texts):
    """Test email detection"""
    result = pipeline.detect(test_texts["email"])
    
    assert len(result['detections']) > 0
    assert 'EMAIL' in result['detections']


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_phone_detection(pipeline, test_texts):
    """Test phone number detection"""
    result = pipeline.detect(test_texts["phone"])
    
    assert 'PHONE' in result['detections']


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_ssn_detection(pipeline, test_texts):
    """Test SSN detection"""
    result = pipeline.detect(test_texts["ssn"])
    
    assert 'SSN' in result['detections']


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_person_name_detection(pipeline, test_texts):
    """Test person name detection with AI"""
    result = pipeline.detect(test_texts["person"])
    
    assert 'PERSON' in result['detections']


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_location_detection(pipeline, test_texts):
    """Test location detection (AI feature)"""
    result = pipeline.detect(test_texts["location"])
    
    assert 'LOCATION' in result['detections']


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_organization_detection(pipeline, test_texts):
    """Test organization detection (AI feature)"""
    result = pipeline.detect(test_texts["org"])
    
    assert 'ORGANIZATION' in result['detections']


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_secret_detection(pipeline, test_texts):
    """Test API key/secret detection"""
    result = pipeline.detect(test_texts["secret"])
    
    assert 'SECRETS' in result['detections']


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_harmful_content_detection(pipeline, test_texts):
    """Test harmful content detection"""
    result = pipeline.detect(test_texts["harmful"])
    
    assert result['harmful'] is True
    assert result['severity'] in ["low", "medium", "high"]


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_credit_card_detection(pipeline, test_texts):
    """Test credit card detection"""
    result = pipeline.detect(test_texts["credit_card"])
    
    assert 'CREDIT_CARD' in result['detections']


# ==================== Legacy API Tests ====================
def test_legacy_detect_pii(test_texts):
    """Test backward compatible detect_pii function"""
    # This should work with or without AI
    results = detect_pii(test_texts["email"], mode='regex')
    
    assert isinstance(results, dict)
    assert "EMAIL" in results
    assert len(results["EMAIL"]) == 1


def test_legacy_detect_secrets(test_texts):
    """Test backward compatible detect_secrets function"""
    results = detect_secrets(test_texts["secret"])
    
    assert isinstance(results, dict)
    assert "SECRETS" in results
    assert len(results["SECRETS"]) == 1


# ==================== Edge Cases ====================
@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_empty_text(pipeline):
    """Test with empty text"""
    result = pipeline.detect("")
    
    assert len(result['detections']) == 0
    assert result['redacted'] == ""


@pytest.mark.skipif(not AI_MODE_AVAILABLE, reason="AI detection not available")
def test_no_sensitive_content(pipeline):
    """Test with text containing no sensitive data"""
    result = pipeline.detect("Hello world! How are you today?")
    assert result['harmful'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
