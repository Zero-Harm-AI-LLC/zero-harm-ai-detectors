import pytest
import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detectors import detect_pii, detect_secrets, HarmfulTextDetector, DetectionConfig

def test_detects_email_and_ssn():
    """Test that email and SSN detection works"""
    text = "Contact me at alice@example.com. SSN 123-45-6789."
    pii = detect_pii(text)
    assert "EMAIL" in pii
    assert "SSN" in pii
    assert len(pii["EMAIL"]) == 1
    assert len(pii["SSN"]) == 1
    assert pii["EMAIL"][0]["span"] == "alice@example.com"

def test_detects_secret_key():
    """Test that secret key detection works"""
    text = "api_key=sk-1234567890abcdef1234567890abcdef"
    sec = detect_secrets(text)
    assert "SECRETS" in sec  # Note: should be "SECRETS" not "SECRET"
    assert len(sec["SECRETS"]) == 1

def test_phone_detection():
    """Test phone number detection"""
    text = "Call me at 555-123-4567"
    pii = detect_pii(text)
    assert "PHONE" in pii
    assert pii["PHONE"][0]["span"] == "555-123-4567"

def test_credit_card_detection():
    """Test credit card detection with valid Luhn checksum"""
    text = "My card is 4532015112830366"  # Valid test card number
    pii = detect_pii(text)
    assert "CREDIT_CARD" in pii

def test_person_name_detection():
    """Test person name detection"""
    text = "My name is John Smith"
    pii = detect_pii(text)
    # Note: This might not work due to "My name is" exclusion pattern
    # Let's test with a different format
    text2 = "Contact John Smith for more information"
    pii2 = detect_pii(text2)
    assert "PERSON_NAME" in pii2

def test_detection_config():
    """Test DetectionConfig dataclass"""
    config = DetectionConfig()
    assert config.threshold_per_label == 0.5
    assert config.overall_threshold == 0.5
    assert config.threat_min_score_on_cue == 0.6

def test_detection_config_custom():
    """Test DetectionConfig with custom values"""
    config = DetectionConfig(
        threshold_per_label=0.7,
        overall_threshold=0.8,
        threat_min_score_on_cue=0.9
    )
    assert config.threshold_per_label == 0.7
    assert config.overall_threshold == 0.8
    assert config.threat_min_score_on_cue == 0.9

# Note: More tests can be added for HarmfulTextDetector, but they may require