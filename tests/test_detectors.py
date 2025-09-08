
from app.detectors import detect_pii, detect_secrets

def test_detects_email_and_ssn():
    text = "Contact me at alice@example.com. SSN 123-45-6789."
    pii = detect_pii(text)
    assert "EMAIL" in pii
    assert "SSN" in pii

def test_detects_secret_key():
    text = "api_key=sk_test_1234567890abcdef"
    sec = detect_secrets(text)
    assert "SECRET" in sec
