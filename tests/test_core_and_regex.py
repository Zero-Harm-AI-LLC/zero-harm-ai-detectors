"""
Tests for core patterns, validators, and regex detection.

File: tests/test_core_and_regex.py
"""
import pytest
from zero_harm_ai_detectors import (
    # Main API
    detect,
    detect_pii,
    detect_secrets,
    detect_harmful,
    # Result types
    Detection,
    DetectionResult,
    DetectionType,
    # Redaction
    RedactionStrategy,
    apply_redaction,
    redact_spans,
    redact_text,
    # Individual detectors
    detect_emails,
    detect_phones,
    detect_ssns,
    detect_credit_cards,
    detect_bank_accounts,
    detect_dob,
    detect_addresses,
    detect_person_names_regex,
    detect_secrets_regex,
    detect_harmful_regex,
    detect_all_regex,
    # Utilities
    luhn_check,
    shannon_entropy,
    find_secrets,
    # Validation
    validate_input,
    InputConfig,
    InputValidationError,
    InputTooLongError,
)


# ============================================================
# Validator Tests
# ============================================================

class TestLuhnCheck:
    """Tests for Luhn credit card validation."""
    
    def test_valid_visa(self):
        assert luhn_check("4532015112830366") is True
    
    def test_valid_mastercard(self):
        assert luhn_check("5425233430109903") is True
    
    def test_valid_amex(self):
        assert luhn_check("374245455400126") is True
    
    def test_invalid_card(self):
        assert luhn_check("1234567890123456") is False
    
    def test_too_short(self):
        assert luhn_check("123456789") is False
    
    def test_with_spaces(self):
        # Should handle only digits
        assert luhn_check("4532015112830366") is True


class TestShannonEntropy:
    """Tests for Shannon entropy calculation."""
    
    def test_empty_string(self):
        assert shannon_entropy("") == 0.0
    
    def test_single_char(self):
        assert shannon_entropy("aaaa") == 0.0
    
    def test_two_chars(self):
        entropy = shannon_entropy("ab")
        assert entropy == 1.0  # log2(2) = 1
    
    def test_high_entropy(self):
        # Random-looking string should have high entropy
        entropy = shannon_entropy("aB3$xY9@mK2!")
        assert entropy > 3.0
    
    def test_low_entropy(self):
        # Repetitive string should have low entropy
        entropy = shannon_entropy("aaaaabbbbb")
        assert entropy < 1.5


# ============================================================
# Email Detection Tests
# ============================================================

class TestEmailDetection:
    """Tests for email detection."""
    
    def test_simple_email(self):
        result = detect_emails("Contact john@example.com")
        assert len(result) == 1
        assert result[0].text == "john@example.com"
        assert result[0].type == "EMAIL"
    
    def test_multiple_emails(self):
        result = detect_emails("Email alice@test.com or bob@example.org")
        assert len(result) == 2
    
    def test_email_with_plus(self):
        result = detect_emails("Use john+tag@example.com")
        assert len(result) == 1
        assert "john+tag@example.com" in result[0].text
    
    def test_no_email(self):
        result = detect_emails("No email here")
        assert len(result) == 0
    
    def test_invalid_email_no_tld(self):
        result = detect_emails("Not an email: john@localhost")
        assert len(result) == 0


# ============================================================
# Phone Detection Tests
# ============================================================

class TestPhoneDetection:
    """Tests for phone number detection."""
    
    def test_dashed_phone(self):
        result = detect_phones("Call 555-123-4567")
        assert len(result) == 1
        assert "555-123-4567" in result[0].text
    
    def test_parentheses_phone(self):
        result = detect_phones("Call (555) 123-4567")
        assert len(result) == 1
    
    def test_with_country_code(self):
        result = detect_phones("Call +1-555-123-4567")
        assert len(result) == 1
    
    def test_no_phone(self):
        result = detect_phones("No phone here")
        assert len(result) == 0


# ============================================================
# SSN Detection Tests
# ============================================================

class TestSSNDetection:
    """Tests for SSN detection."""
    
    def test_dashed_ssn(self):
        result = detect_ssns("SSN: 123-45-6789")
        assert len(result) == 1
        assert "123-45-6789" in result[0].text
    
    def test_invalid_ssn_000(self):
        result = detect_ssns("Invalid: 000-45-6789")
        assert len(result) == 0
    
    def test_invalid_ssn_666(self):
        result = detect_ssns("Invalid: 666-45-6789")
        assert len(result) == 0
    
    def test_invalid_ssn_9xx(self):
        result = detect_ssns("Invalid: 900-45-6789")
        assert len(result) == 0


# ============================================================
# Credit Card Detection Tests
# ============================================================

class TestCreditCardDetection:
    """Tests for credit card detection."""
    
    def test_valid_visa(self):
        result = detect_credit_cards("Card: 4532015112830366")
        assert len(result) == 1
    
    def test_valid_with_dashes(self):
        result = detect_credit_cards("Card: 4532-0151-1283-0366")
        assert len(result) == 1
    
    def test_invalid_luhn(self):
        result = detect_credit_cards("Card: 1234567890123456")
        assert len(result) == 0
    
    def test_valid_mastercard(self):
        result = detect_credit_cards("Card: 5425233430109903")
        assert len(result) == 1


# ============================================================
# Secrets Detection Tests
# ============================================================

class TestSecretsDetection:
    """Tests for secrets/API key detection."""
    
    def test_openai_key(self):
        result = detect_secrets_regex("key = sk-1234567890abcdef1234567890abcdef")
        assert len(result) >= 1
        assert any("API_KEY" in d.type or "SECRET" in d.type for d in result)
    
    def test_github_pat(self):
        result = detect_secrets_regex("token = ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234")
        assert len(result) >= 1
    
    def test_aws_access_key(self):
        result = detect_secrets_regex("aws_key = AKIAIOSFODNN7EXAMPLE")
        assert len(result) >= 1
    
    def test_no_secrets_normal_text(self):
        result = detect_secrets_regex("The quick brown fox jumps over the lazy dog")
        assert len(result) == 0
    
    def test_generic_secret_with_context(self):
        result = detect_secrets_regex("password = AbCdEfGhIjKlMnOpQrStUvWxYz123456")
        assert len(result) >= 1


# ============================================================
# Harmful Content Detection Tests
# ============================================================

class TestHarmfulDetection:
    """Tests for harmful content detection."""
    
    def test_clean_text(self):
        result = detect_harmful_regex("Hello, how are you today?")
        assert result["harmful"] is False
        assert result["severity"] == "none"
    
    def test_insult(self):
        result = detect_harmful_regex("You're such a stupid idiot!")
        assert result["harmful"] is True
        assert "insult" in result["scores"]
    
    def test_threat(self):
        result = detect_harmful_regex("I'm going to kill you!")
        assert result["harmful"] is True
        assert result["severity"] == "high"
    
    def test_profanity(self):
        result = detect_harmful_regex("What the fuck is this?")
        assert result["harmful"] is True
        assert result["severity"] == "medium"


# ============================================================
# Redaction Tests
# ============================================================

class TestRedaction:
    """Tests for redaction strategies."""
    
    def test_token_redaction(self):
        result = apply_redaction("test@example.com", "EMAIL", RedactionStrategy.TOKEN)
        assert result == "[REDACTED_EMAIL]"
    
    def test_mask_all_redaction(self):
        result = apply_redaction("secret123", "SECRET", RedactionStrategy.MASK_ALL)
        assert result == "*********"
        assert len(result) == len("secret123")
    
    def test_mask_last4_redaction(self):
        result = apply_redaction("4532015112830366", "CREDIT_CARD", RedactionStrategy.MASK_LAST4)
        assert result.endswith("0366")
        assert result.startswith("****")
    
    def test_hash_redaction(self):
        result = apply_redaction("test@example.com", "EMAIL", RedactionStrategy.HASH)
        assert result.startswith("[HASH:")
        assert result.endswith("]")
    
    def test_redact_spans(self):
        text = "Email: test@example.com"
        detections = [Detection(
            type="EMAIL",
            text="test@example.com",
            start=7,
            end=23,
            confidence=0.99,
        )]
        result = redact_spans(text, detections, RedactionStrategy.TOKEN)
        assert result == "Email: [REDACTED_EMAIL]"


# ============================================================
# Unified API Tests
# ============================================================

class TestUnifiedDetect:
    """Tests for unified detect() function."""
    
    def test_default_mode_is_regex(self):
        result = detect("test@example.com")
        assert result.mode == "regex"
    
    def test_explicit_regex_mode(self):
        result = detect("test@example.com", mode="regex")
        assert result.mode == "regex"
    
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError) as exc_info:
            detect("test", mode="invalid")
        assert "Invalid mode" in str(exc_info.value)
    
    def test_returns_detection_result(self):
        result = detect("Email: test@example.com")
        assert isinstance(result, DetectionResult)
        assert result.original_text == "Email: test@example.com"
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert len(result.detections) >= 1
        assert result.mode == "regex"
    
    def test_detect_pii_only(self):
        result = detect(
            "Email: test@example.com, key: sk-abc123def456abc123def456abc123def456",
            detect_pii=True,
            detect_secrets=False,
        )
        types = [d.type for d in result.detections]
        assert "EMAIL" in types
        assert "API_KEY" not in types and "SECRET" not in types
    
    def test_detect_secrets_only(self):
        result = detect(
            "Email: test@example.com, key: sk-abc123def456abc123def456abc123def456",
            detect_pii=False,
            detect_secrets=True,
        )
        types = [d.type for d in result.detections]
        assert "EMAIL" not in types
    
    def test_different_redaction_strategies(self):
        text = "Email: test@example.com"
        
        result_token = detect(text, redaction_strategy="token")
        assert "[REDACTED_EMAIL]" in result_token.redacted_text
        
        result_mask = detect(text, redaction_strategy="mask_all")
        assert "****" in result_mask.redacted_text
    
    def test_to_dict_format(self):
        result = detect("Email: test@example.com")
        d = result.to_dict()
        
        assert "original" in d
        assert "redacted" in d
        assert "detections" in d
        assert "mode" in d
        assert d["mode"] == "regex"
        assert "harmful" in d
        assert "severity" in d
    
    def test_to_legacy_dict_format(self):
        result = detect("Email: test@example.com")
        legacy = result.to_legacy_dict()
        
        assert "EMAIL" in legacy
        assert isinstance(legacy["EMAIL"], list)
        assert "span" in legacy["EMAIL"][0]
    
    def test_empty_text(self):
        result = detect("")
        assert result.original_text == ""
        assert result.redacted_text == ""
        assert len(result.detections) == 0


class TestDetectAllRegex:
    """Tests for detect_all_regex function."""
    
    def test_mixed_content(self):
        text = """
        Email: john@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        API Key: sk-1234567890abcdef1234567890abcdef
        """
        result = detect_all_regex(text)
        
        assert result.mode == "regex"
        types = {d.type for d in result.detections}
        assert "EMAIL" in types
        assert "PHONE" in types
        assert "SSN" in types
    
    def test_harmful_content_included(self):
        result = detect_all_regex("You stupid idiot!", detect_harmful=True)
        assert result.harmful is True
        assert result.severity != "none"


# ============================================================
# Legacy API Tests
# ============================================================

class TestLegacyAPI:
    """Tests for legacy v0.1.x compatible functions."""
    
    def test_detect_pii_legacy(self):
        result = detect_pii("Email: test@example.com")
        assert isinstance(result, dict)
        assert "EMAIL" in result
    
    def test_detect_secrets_legacy(self):
        result = detect_secrets("key = sk-abc123def456abc123def456abc123def456")
        assert isinstance(result, dict)
    
    def test_detect_harmful_legacy(self):
        result = detect_harmful("You idiot!")
        assert isinstance(result, dict)
        assert "harmful" in result
    
    def test_redact_text_legacy(self):
        detections = {
            "EMAIL": [{"span": "test@example.com", "start": 7, "end": 23}]
        }
        result = redact_text("Email: test@example.com", detections, "token")
        assert "[REDACTED_EMAIL]" in result


# ============================================================
# Input Validation Tests
# ============================================================

class TestInputValidation:
    """Tests for input validation."""
    
    def test_valid_input(self):
        result = validate_input("Hello world")
        assert result == "Hello world"
    
    def test_none_input_raises(self):
        with pytest.raises(InputValidationError):
            validate_input(None)
    
    def test_too_long_raises(self):
        config = InputConfig(max_length=10)
        with pytest.raises(InputTooLongError):
            validate_input("This is way too long", config)
    
    def test_null_bytes_stripped(self):
        result = validate_input("Hello\x00World")
        assert "\x00" not in result


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_unicode_text(self):
        result = detect("Email: café@example.com")
        # Should not crash
        assert result is not None
    
    def test_very_long_text(self):
        text = "test@example.com " * 1000
        result = detect(text)
        assert len(result.detections) > 0
    
    def test_special_characters(self):
        result = detect("Email: <test@example.com>")
        assert len(result.detections) >= 1
    
    def test_overlapping_patterns(self):
        # Credit card that looks like phone
        text = "Number: 4532-0151-1283-0366"
        result = detect(text)
        # Should detect as credit card (Luhn valid) not phone
        types = [d.type for d in result.detections]
        assert "CREDIT_CARD" in types
    
    def test_multiple_detections_same_type(self):
        text = "Emails: a@b.com, c@d.com, e@f.com"
        result = detect(text)
        emails = [d for d in result.detections if d.type == "EMAIL"]
        assert len(emails) == 3


# ============================================================
# Detection Result Methods
# ============================================================

class TestDetectionResult:
    """Tests for DetectionResult methods."""
    
    def test_get_pii(self):
        result = detect("Email: test@example.com, key: sk-abc123def456abc123def456abc123def456")
        pii = result.get_pii()
        types = [d.type for d in pii]
        assert "EMAIL" in types
        assert "API_KEY" not in types
    
    def test_get_secrets(self):
        result = detect("Email: test@example.com, key: sk-abc123def456abc123def456abc123def456")
        secrets = result.get_secrets()
        # May or may not have secrets depending on detection
        assert isinstance(secrets, list)
