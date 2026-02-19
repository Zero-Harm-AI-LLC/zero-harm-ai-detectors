"""
Tests for core patterns, validators, and regex detection.

File: tests/test_core_and_regex.py
"""
import pytest
from zero_harm_ai_detectors import (
    # Main API
    detect,
    # Result types
    Detection,
    DetectionResult,
    DetectionType,
    # Redaction
    RedactionStrategy,
    apply_redaction,
    redact_spans,
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


class TestShannonEntropy:
    def test_empty_string(self):
        assert shannon_entropy("") == 0.0

    def test_single_char(self):
        assert shannon_entropy("aaaa") == 0.0

    def test_two_chars(self):
        assert shannon_entropy("ab") == 1.0

    def test_high_entropy(self):
        assert shannon_entropy("aB3$xY9@mK2!") > 3.0

    def test_low_entropy(self):
        assert shannon_entropy("aaaaabbbbb") < 1.5


# ============================================================
# Email Detection
# ============================================================

class TestEmailDetection:
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
        assert detect_emails("No email here") == []

    def test_invalid_email_no_tld(self):
        assert detect_emails("Not an email: john@localhost") == []


# ============================================================
# Phone Detection
# ============================================================

class TestPhoneDetection:
    def test_dashed_phone(self):
        result = detect_phones("Call 555-123-4567")
        assert len(result) == 1
        assert "555-123-4567" in result[0].text

    def test_parentheses_phone(self):
        assert len(detect_phones("Call (555) 123-4567")) == 1

    def test_with_country_code(self):
        assert len(detect_phones("Call +1-555-123-4567")) == 1

    def test_no_phone(self):
        assert detect_phones("No phone here") == []


# ============================================================
# SSN Detection
# ============================================================

class TestSSNDetection:
    def test_dashed_ssn(self):
        result = detect_ssns("SSN: 123-45-6789")
        assert len(result) == 1
        assert "123-45-6789" in result[0].text

    def test_invalid_ssn_000(self):
        assert detect_ssns("Invalid: 000-45-6789") == []

    def test_invalid_ssn_666(self):
        assert detect_ssns("Invalid: 666-45-6789") == []

    def test_invalid_ssn_9xx(self):
        assert detect_ssns("Invalid: 900-45-6789") == []


# ============================================================
# Credit Card Detection
# ============================================================

class TestCreditCardDetection:
    def test_valid_visa(self):
        assert len(detect_credit_cards("Card: 4532015112830366")) == 1

    def test_valid_with_dashes(self):
        assert len(detect_credit_cards("Card: 4532-0151-1283-0366")) == 1

    def test_invalid_luhn(self):
        assert detect_credit_cards("Card: 1234567890123456") == []

    def test_valid_mastercard(self):
        assert len(detect_credit_cards("Card: 5425233430109903")) == 1


# ============================================================
# Secrets Detection
# ============================================================

class TestSecretsDetection:
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
        assert detect_secrets_regex("The quick brown fox jumps over the lazy dog") == []

    def test_generic_secret_with_context(self):
        result = detect_secrets_regex("password = AbCdEfGhIjKlMnOpQrStUvWxYz123456")
        assert len(result) >= 1


# ============================================================
# Harmful Content Detection
# ============================================================

class TestHarmfulDetection:
    def test_clean_text(self):
        result = detect_harmful_regex("Hello, how are you today?")
        assert result["harmful"] is False
        assert result["severity"] == "none"

    def test_insult_is_low(self):
        result = detect_harmful_regex("You're such a loser!")
        assert result["harmful"] is True
        assert result["severity"] == "low"

    def test_threat_phrase_is_high(self):
        result = detect_harmful_regex("I'm going to kill you!")
        assert result["harmful"] is True
        assert result["severity"] == "high"

    def test_identity_hate_is_high(self):
        result = detect_harmful_regex("You are a faggot")
        assert result["harmful"] is True
        assert result["severity"] == "high"

    def test_profanity_alone_is_low(self):
        result = detect_harmful_regex("What the fuck is this?")
        assert result["harmful"] is True
        assert result["severity"] == "low"

    def test_multiple_insults_escalate_to_medium(self):
        result = detect_harmful_regex("I hate you, you stupid idiot!")
        assert result["harmful"] is True
        assert result["severity"] == "medium"

    def test_scores_present_when_harmful(self):
        result = detect_harmful_regex("You idiot!")
        assert isinstance(result["scores"], dict)
        assert len(result["scores"]) > 0

    def test_scores_empty_when_clean(self):
        result = detect_harmful_regex("Nice weather today.")
        assert result["scores"] == {}


# ============================================================
# Redaction
# ============================================================

class TestRedaction:
    def test_token_redaction(self):
        assert apply_redaction("test@example.com", "EMAIL", RedactionStrategy.TOKEN) == "[REDACTED_EMAIL]"

    def test_mask_all_redaction(self):
        result = apply_redaction("secret123", "SECRET", RedactionStrategy.MASK_ALL)
        assert result == "*********"

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
        assert redact_spans(text, detections, RedactionStrategy.TOKEN) == "Email: [REDACTED_EMAIL]"


# ============================================================
# Unified detect() API
# ============================================================

class TestUnifiedDetect:
    def test_default_mode_is_regex(self):
        assert detect("test@example.com").mode == "regex"

    def test_explicit_regex_mode(self):
        assert detect("test@example.com", mode="regex").mode == "regex"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            detect("test", mode="invalid")

    def test_returns_detection_result(self):
        result = detect("Email: test@example.com")
        assert isinstance(result, DetectionResult)
        assert result.original_text == "Email: test@example.com"
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert len(result.detections) >= 1

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

    def test_redaction_strategies(self):
        text = "Email: test@example.com"
        assert "[REDACTED_EMAIL]" in detect(text, redaction_strategy="token").redacted_text
        assert "****" in detect(text, redaction_strategy="mask_all").redacted_text

    def test_to_dict_format(self):
        d = detect("Email: test@example.com").to_dict()
        assert all(k in d for k in ("original", "redacted", "detections", "mode", "harmful", "severity"))
        assert d["mode"] == "regex"

    def test_empty_text(self):
        result = detect("")
        assert result.original_text == ""
        assert result.redacted_text == ""
        assert result.detections == []


class TestDetectAllRegex:
    def test_mixed_content(self):
        text = """
        Email: john@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        API Key: sk-1234567890abcdef1234567890abcdef
        """
        result = detect_all_regex(text)
        types = {d.type for d in result.detections}
        assert "EMAIL" in types
        assert "PHONE" in types
        assert "SSN" in types

    def test_harmful_content_included(self):
        result = detect_all_regex("You stupid idiot!", detect_harmful=True)
        assert result.harmful is True
        assert result.severity != "none"


# ============================================================
# Input Validation
# ============================================================

class TestInputValidation:
    def test_valid_input(self):
        assert validate_input("Hello world") == "Hello world"

    def test_none_input_raises(self):
        with pytest.raises(InputValidationError):
            validate_input(None)

    def test_too_long_raises(self):
        with pytest.raises(InputTooLongError):
            validate_input("This is way too long", InputConfig(max_length=10))

    def test_null_bytes_stripped(self):
        assert "\x00" not in validate_input("Hello\x00World")


# ============================================================
# DetectionResult helpers
# ============================================================

class TestDetectionResult:
    def test_get_pii(self):
        result = detect("Email: test@example.com, key: sk-abc123def456abc123def456abc123def456")
        pii = result.get_pii()
        assert any(d.type == "EMAIL" for d in pii)
        assert all(d.type != "API_KEY" for d in pii)

    def test_get_secrets(self):
        result = detect("key: sk-abc123def456abc123def456abc123def456")
        assert isinstance(result.get_secrets(), list)

    def test_to_dict_detections_grouped(self):
        result = detect("Emails: a@b.com, c@d.com")
        d = result.to_dict()
        assert "EMAIL" in d["detections"]
        assert len(d["detections"]["EMAIL"]) == 2


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_unicode_text(self):
        assert detect("Email: café@example.com") is not None

    def test_special_characters(self):
        result = detect("Email: <test@example.com>")
        assert len(result.detections) >= 1

    def test_overlapping_patterns(self):
        # Valid Luhn card should be detected as CREDIT_CARD, not phone
        result = detect("Number: 4532-0151-1283-0366")
        types = [d.type for d in result.detections]
        assert "CREDIT_CARD" in types

    def test_multiple_detections_same_type(self):
        result = detect("Emails: a@b.com, c@d.com, e@f.com")
        emails = [d for d in result.detections if d.type == "EMAIL"]
        assert len(emails) == 3
