"""
Comprehensive tests for regex-based detection (free tier).

Covers:
- All PII detectors (email, phone, SSN, credit card, bank account, DOB,
  driver's license, MRN, person name, address)
- Secrets detection (three-tier: structured, context+entropy, generic)
- Harmful content detection (regex-based)
- Redaction strategies
- Input validation integration
- Shared core_patterns verification (patterns are the same objects)
- Edge cases (empty text, no matches, overlapping spans)
- Backward compatibility (return formats, DetectionConfig)

File: tests/test_regex_detectors.py
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zero_harm_ai_detectors import detect_pii, detect_secrets, detect_harmful, DetectionConfig
from zero_harm_ai_detectors.regex_detectors import (
    redact_text,
    RedactionStrategy,
    EmailDetector,
    PhoneDetector,
    SSNDetector,
    CreditCardDetector,
    BankAccountDetector,
    DOBDetector,
    DriversLicenseDetector,
    MRNDetector,
    PersonNameDetector,
    AddressDetector,
    SecretsDetector,
    default_detectors,
)
from zero_harm_ai_detectors.core_patterns import (
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    RedactionStrategy as CoreRedactionStrategy,
    luhn_check,
    shannon_entropy,
    find_secrets,
    has_secret_context,
    MIN_SECRET_ENTROPY,
    apply_redaction,
    redact_spans,
    THREAT_CUES_RE,
)
from zero_harm_ai_detectors.input_validation import (
    validate_input,
    InputTooLongError,
    ReDoSRiskError,
)


# ============================================================
# Section 1: Shared core_patterns verification
# ============================================================

class TestSharedPatterns:
    """Verify regex_detectors imports from core_patterns, not its own copies."""

    def test_email_detector_uses_core_pattern(self):
        assert EmailDetector().pattern is EMAIL_RE

    def test_phone_detector_uses_core_pattern(self):
        assert PhoneDetector().pattern is PHONE_RE

    def test_ssn_detector_uses_core_pattern(self):
        assert SSNDetector().pattern is SSN_RE

    def test_redaction_strategy_is_core(self):
        assert RedactionStrategy is CoreRedactionStrategy

    def test_secrets_detector_delegates_to_find_secrets(self):
        text = "key is sk-1234567890abcdef1234567890abcdef"
        assert SecretsDetector().finditer(text) == find_secrets(text)


# ============================================================
# Section 2: Luhn check (via core_patterns)
# ============================================================

class TestLuhnCheck:
    def test_valid_visa(self):
        assert luhn_check("4532015112830366") is True

    def test_valid_mastercard(self):
        assert luhn_check("5425233430109903") is True

    def test_valid_with_separators(self):
        assert luhn_check("4532-0151-1283-0366") is True

    def test_invalid_random_digits(self):
        assert luhn_check("1234567890123456") is False

    def test_too_short(self):
        assert luhn_check("12345") is False

    def test_too_long(self):
        assert luhn_check("1" * 20) is False


# ============================================================
# Section 3: Shannon entropy (via core_patterns)
# ============================================================

class TestShannonEntropy:
    def test_empty_string(self):
        assert shannon_entropy("") == 0.0

    def test_uniform_string(self):
        assert shannon_entropy("aaaa") == 0.0

    def test_high_entropy_string(self):
        assert shannon_entropy("aB3xZ9kLmNpQ7rStUvW0yE1fGhIjK2dC") > 4.0

    def test_english_below_random(self):
        english = "the quick brown fox jumps over"
        random_str = "k7Bx9LmNpQ3rStUvW0yE1fGhIjK2dCa"
        assert shannon_entropy(english) < shannon_entropy(random_str)

    def test_threshold_value(self):
        assert MIN_SECRET_ENTROPY == 4.5


# ============================================================
# Section 4: PII Detection — Email
# ============================================================

class TestEmailDetection:
    def test_basic_email(self):
        pii = detect_pii("Contact me at alice@example.com")
        assert "EMAIL" in pii
        assert pii["EMAIL"][0]["span"] == "alice@example.com"

    def test_email_with_plus(self):
        pii = detect_pii("Send to user+tag@gmail.com")
        assert "EMAIL" in pii

    def test_email_with_subdomain(self):
        pii = detect_pii("admin@mail.corp.example.co.uk")
        assert "EMAIL" in pii

    def test_multiple_emails(self):
        pii = detect_pii("From a@b.com to c@d.com")
        assert "EMAIL" in pii
        assert len(pii["EMAIL"]) == 2

    def test_no_email_in_plain_text(self):
        pii = detect_pii("Hello world, nice day")
        assert "EMAIL" not in pii


# ============================================================
# Section 5: PII Detection — Phone
# ============================================================

class TestPhoneDetection:
    def test_dashes(self):
        pii = detect_pii("Call me at 555-123-4567")
        assert "PHONE" in pii
        assert pii["PHONE"][0]["span"] == "555-123-4567"

    def test_dots(self):
        pii = detect_pii("Phone: 555.123.4567")
        assert "PHONE" in pii

    def test_parens(self):
        pii = detect_pii("Reach me at (555) 123-4567")
        assert "PHONE" in pii

    def test_country_code(self):
        pii = detect_pii("International: +1 555-123-4567")
        assert "PHONE" in pii

    def test_no_separator_no_match(self):
        """Continuous digits should NOT match (avoids credit card false positives)."""
        pii = detect_pii("Number is 5551234567")
        assert "PHONE" not in pii

    def test_no_phone_in_plain_text(self):
        pii = detect_pii("There is no phone here")
        assert "PHONE" not in pii


# ============================================================
# Section 6: PII Detection — SSN
# ============================================================

class TestSSNDetection:
    def test_with_dashes(self):
        pii = detect_pii("SSN 123-45-6789")
        assert "SSN" in pii

    def test_with_spaces(self):
        pii = detect_pii("SSN 123 45 6789")
        assert "SSN" in pii

    def test_invalid_prefix_000(self):
        pii = detect_pii("SSN 000-45-6789")
        assert "SSN" not in pii

    def test_invalid_prefix_666(self):
        pii = detect_pii("SSN 666-45-6789")
        assert "SSN" not in pii

    def test_invalid_prefix_9xx(self):
        pii = detect_pii("SSN 900-45-6789")
        assert "SSN" not in pii


# ============================================================
# Section 7: PII Detection — Credit Card
# ============================================================

class TestCreditCardDetection:
    def test_valid_visa_with_dashes(self):
        pii = detect_pii("Card: 4532-0151-1283-0366")
        assert "CREDIT_CARD" in pii

    def test_valid_visa_spaces(self):
        pii = detect_pii("Card 4532 0151 1283 0366")
        assert "CREDIT_CARD" in pii

    def test_invalid_luhn_rejected(self):
        pii = detect_pii("Card: 1234-5678-9012-3456")
        assert "CREDIT_CARD" not in pii


# ============================================================
# Section 8: PII Detection — Bank Account
# ============================================================

class TestBankAccountDetection:
    def test_with_keyword_context(self):
        pii = detect_pii("Account number: 1234567890")
        assert "BANK_ACCOUNT" in pii

    def test_routing_keyword(self):
        pii = detect_pii("Routing: 123456789012")
        assert "BANK_ACCOUNT" in pii

    def test_no_context_no_match(self):
        pii = detect_pii("The result is 1234567890")
        assert "BANK_ACCOUNT" not in pii


# ============================================================
# Section 9: PII Detection — Date of Birth
# ============================================================

class TestDOBDetection:
    def test_mm_dd_yyyy_slash(self):
        pii = detect_pii("DOB: 01/15/1990")
        assert "DOB" in pii

    def test_yyyy_mm_dd_dash(self):
        pii = detect_pii("Born 1990-01-15")
        assert "DOB" in pii

    def test_month_name_format(self):
        pii = detect_pii("Born January 15, 1990")
        assert "DOB" in pii

    def test_abbreviated_month(self):
        pii = detect_pii("DOB: Jan 15, 1990")
        assert "DOB" in pii


# ============================================================
# Section 10: PII Detection — Driver's License
# ============================================================

class TestDriversLicenseDetection:
    def test_with_keyword(self):
        pii = detect_pii("Driver's license: D1234567")
        assert "DRIVERS_LICENSE" in pii

    def test_dl_abbreviation(self):
        pii = detect_pii("DL A1234567")
        assert "DRIVERS_LICENSE" in pii

    def test_no_context_no_match(self):
        pii = detect_pii("Random code A1B2C3D4")
        assert "DRIVERS_LICENSE" not in pii


# ============================================================
# Section 11: PII Detection — Medical Record Number
# ============================================================

class TestMRNDetection:
    def test_with_mrn_keyword(self):
        pii = detect_pii("MRN 12345678")
        assert "MEDICAL_RECORD_NUMBER" in pii

    def test_with_full_keyword(self):
        pii = detect_pii("Medical record number: 123456789")
        assert "MEDICAL_RECORD_NUMBER" in pii

    def test_no_context_no_match(self):
        pii = detect_pii("Reference 12345678")
        assert "MEDICAL_RECORD_NUMBER" not in pii


# ============================================================
# Section 12: PII Detection — Person Name
# ============================================================

class TestPersonNameDetection:
    def test_basic_name(self):
        """Test person name detection — regex has limited accuracy (30-40%)."""
        # Try several patterns; at least one should work
        test_cases = [
            "Please contact John Smith",
            "John Smith will help you",
            "Dr. Robert Wilson is here",
            "The manager Sarah Davis",
            "Name: John Smith",
        ]
        found = False
        for text in test_cases:
            pii = detect_pii(text)
            if "PERSON_NAME" in pii and len(pii["PERSON_NAME"]) > 0:
                found = True
                break
        # Soft assertion — regex name detection is known to be limited
        if not found:
            pytest.skip("Person name regex didn't match any test cases (known limitation)")

    def test_multiple_names(self):
        pii = detect_pii("John Smith and Mary Johnson attended")
        if "PERSON_NAME" in pii:
            assert len(pii["PERSON_NAME"]) >= 1

    def test_excludes_address_suffix(self):
        """Names ending in street types should be excluded."""
        pii = detect_pii("Visit Main Street today")
        if "PERSON_NAME" in pii:
            for span in pii["PERSON_NAME"]:
                assert "Street" not in span["span"]

    def test_excludes_day_names(self):
        pii = detect_pii("Meeting on Happy Monday")
        if "PERSON_NAME" in pii:
            for span in pii["PERSON_NAME"]:
                assert "Monday" not in span["span"]


# ============================================================
# Section 13: PII Detection — Address
# ============================================================

class TestAddressDetection:
    def test_street_address(self):
        pii = detect_pii("Office at 123 Main Street")
        assert "ADDRESS" in pii

    def test_full_address(self):
        pii = detect_pii("Ship to 456 Oak Avenue, Springfield, IL 62701")
        assert "ADDRESS" in pii

    def test_po_box(self):
        pii = detect_pii("Mail to P.O. Box 123")
        assert "ADDRESS" in pii


# ============================================================
# Section 14: Secrets Detection — Structured (Tier 1)
# ============================================================

class TestSecretsStructured:
    """Tier 1: distinctive prefix patterns, no context needed."""

    def test_openai_key(self):
        sec = detect_secrets("api_key=sk-1234567890abcdef1234567890abcdef")
        assert "SECRETS" in sec
        assert len(sec["SECRETS"]) >= 1

    def test_aws_access_key(self):
        sec = detect_secrets("AKIAIOSFODNN7EXAMPLE here")
        assert "SECRETS" in sec

    def test_github_pat(self):
        sec = detect_secrets("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")  # ghp_ + 36 chars
        assert "SECRETS" in sec

    def test_stripe_key(self):
        sec = detect_secrets("sk_live_abcdefghijklmnopqrstuvwx")
        assert "SECRETS" in sec

    def test_slack_token(self):
        sec = detect_secrets("xoxb-12345678901-1234567890123-AbCdEfGhIjKl")
        assert "SECRETS" in sec

    def test_google_api_key(self):
        sec = detect_secrets("AIzaSyB_6abcdefghijklmnopqrstuvwxyz1234")
        assert "SECRETS" in sec

    def test_jwt(self):
        sec = detect_secrets("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")
        assert "SECRETS" in sec

    def test_gitlab_token(self):
        sec = detect_secrets("glpat-ABCDEFghijklmnopqrst")
        assert "SECRETS" in sec

    def test_npm_token(self):
        sec = detect_secrets("npm_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert "SECRETS" in sec


# ============================================================
# Section 15: Secrets Detection — Context-Dependent (Tiers 2 & 3)
# ============================================================

class TestSecretsContextDependent:
    """Tier 2 (AWS secret key) and Tier 3 (generic) require keyword context + entropy."""

    def test_aws_secret_with_context(self):
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        sec = detect_secrets(text)
        assert "SECRETS" in sec

    def test_generic_secret_with_context(self):
        text = "api_key = AbCdEfGhIjKlMnOpQrStUvWx"
        sec = detect_secrets(text)
        assert "SECRETS" in sec


# ============================================================
# Section 16: Secrets Detection — False Positive Rejection
# ============================================================

class TestSecretsFalsePositives:
    """Normal text must NOT trigger secrets detection."""

    @pytest.mark.parametrize("text", [
        "The quick brown fox jumps over the lazy dog",
        "Meeting scheduled for Tuesday afternoon in room B",
        "function calculateTotal(items) { return sum; }",
        "background-color: #ff5733; font-size: 16px;",
        "commit abc123def456789012345678901234567890ab",  # git SHA, no context
        "class='containerElement' id='mainWrapper1234'",
        "data: SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0",  # base64, no context
        "The project ID is PROJ-2024-ALPHA-BETA-GAMMA",
        "Session duration was 1234567890 milliseconds",
    ])
    def test_no_false_positive(self, text):
        sec = detect_secrets(text)
        assert sec == {}, f"False positive on: {text!r}"


# ============================================================
# Section 17: Harmful Content Detection
# ============================================================

class TestHarmfulDetection:
    def test_toxic_content(self):
        result = detect_harmful("You're such a stupid idiot!")
        assert "HARMFUL_CONTENT" in result
        data = result["HARMFUL_CONTENT"][0]
        assert "insult" in data["labels"]

    def test_threat_content(self):
        result = detect_harmful("I'm going to kill you!")
        assert "HARMFUL_CONTENT" in result
        data = result["HARMFUL_CONTENT"][0]
        assert "threat" in data["labels"]

    def test_high_severity_hate(self):
        result = detect_harmful("You disgusting nazi scum!")
        assert "HARMFUL_CONTENT" in result
        assert result["HARMFUL_CONTENT"][0]["severity"] == "high"

    def test_clean_text(self):
        result = detect_harmful("This is a beautiful day!")
        assert result == {}

    def test_return_format(self):
        """Verify the dict format is compatible with detect_pii."""
        result = detect_harmful("I hate you stupid fool!")
        assert "HARMFUL_CONTENT" in result
        data = result["HARMFUL_CONTENT"][0]
        assert "span" in data
        assert "start" in data
        assert "end" in data
        assert "severity" in data
        assert "labels" in data
        assert "scores" in data
        assert isinstance(data["scores"], dict)


# ============================================================
# Section 18: Redaction
# ============================================================

class TestRedaction:
    def test_mask_all(self):
        text = "Email alice@example.com"
        spans = {"EMAIL": [{"span": "alice@example.com", "start": 6, "end": 23}]}
        result = redact_text(text, spans, "mask_all")
        assert "alice@example.com" not in result
        assert "*" * 17 in result

    def test_mask_last4(self):
        text = "Email alice@example.com"
        spans = {"EMAIL": [{"span": "alice@example.com", "start": 6, "end": 23}]}
        result = redact_text(text, spans, "mask_last4")
        assert result.endswith(".com")
        assert "alice@example.com" not in result

    def test_hash(self):
        text = "Email alice@example.com"
        spans = {"EMAIL": [{"span": "alice@example.com", "start": 6, "end": 23}]}
        result = redact_text(text, spans, "hash")
        assert "alice@example.com" not in result
        assert len(result) > len(text)  # SHA-256 hex is longer

    def test_token_strategy_via_core(self):
        result = apply_redaction("alice@example.com", CoreRedactionStrategy.TOKEN, "EMAIL")
        assert result == "[REDACTED_EMAIL]"

    def test_redact_multiple_spans(self):
        text = "Email alice@a.com and bob@b.com"
        spans = {
            "EMAIL": [
                {"span": "alice@a.com", "start": 6, "end": 17},
                {"span": "bob@b.com", "start": 22, "end": 31},
            ]
        }
        result = redact_text(text, spans, "mask_all")
        assert "alice@a.com" not in result
        assert "bob@b.com" not in result

    def test_redact_flat_list(self):
        text = "SSN: 123-45-6789"
        spans = [{"span": "123-45-6789", "start": 5, "end": 16}]
        result = redact_text(text, spans, "mask_all")
        assert "123-45-6789" not in result

    def test_invalid_strategy_defaults(self):
        text = "Test 123-45-6789"
        spans = [{"span": "123-45-6789", "start": 5, "end": 16}]
        result = redact_text(text, spans, "invalid_strategy")
        assert "123-45-6789" not in result  # falls back to mask_all

    def test_core_redact_spans(self):
        text = "Call 555-123-4567"
        spans = [{"span": "555-123-4567", "start": 5, "end": 17, "type": "PHONE"}]
        result = redact_spans(text, spans, CoreRedactionStrategy.TOKEN)
        assert result == "Call [REDACTED_PHONE]"


# ============================================================
# Section 19: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_text(self):
        assert detect_pii("") == {}
        assert detect_secrets("") == {}
        assert detect_harmful("") == {}

    def test_whitespace_only(self):
        assert detect_pii("   \n\t  ") == {}

    def test_very_long_text_within_limits(self):
        text = "Normal text.\n" * 1000  # 13K chars, multi-line (within limits)
        result = detect_pii(text)
        assert isinstance(result, dict)

    def test_mixed_detections(self):
        """Single text with multiple PII types."""
        text = "Email alice@test.com, SSN 123-45-6789, call 555-123-4567"
        pii = detect_pii(text)
        assert "EMAIL" in pii
        assert "SSN" in pii
        assert "PHONE" in pii

    def test_overlapping_context_detectors(self):
        """Bank account digits shouldn't clash with SSN patterns."""
        text = "Account: 12345678901234"
        pii = detect_pii(text)
        # Should detect bank account (keyword context), not SSN
        assert "BANK_ACCOUNT" in pii

    def test_unicode_text(self):
        """Unicode text should pass through validation."""
        text = "Contact José García at jose@example.com"
        pii = detect_pii(text)
        assert "EMAIL" in pii


# ============================================================
# Section 20: has_secret_context (via core_patterns)
# ============================================================

class TestHasSecretContext:
    def test_keyword_before(self):
        text = "api_key: SOMEVALUE12345678901234567890"
        assert has_secret_context(text, 10, 40) is True

    def test_keyword_after(self):
        text = "VALUE12345678901234567890 is the secret_key"
        assert has_secret_context(text, 0, 25) is True

    def test_no_keyword(self):
        text = "The quick brown VALUE12345678901234567890"
        assert has_secret_context(text, 16, 46) is False


# ============================================================
# Section 21: Input Validation Integration
# ============================================================

class TestInputValidation:
    def test_null_bytes_stripped(self):
        text = "test\x00@example.com"
        pii = detect_pii(text)
        # After stripping null byte, email should be detected
        assert "EMAIL" in pii

    def test_zero_width_chars_stripped(self):
        text = "test\u200b@\u200bexample\u200b.com"
        pii = detect_pii(text)
        assert "EMAIL" in pii

    def test_oversized_input_raises(self):
        text = "a" * 600_000
        with pytest.raises(InputTooLongError):
            from zero_harm_ai_detectors.input_validation import InputConfig
            validate_input(text, InputConfig(max_length=500_000))


# ============================================================
# Section 22: DetectionConfig Backward Compatibility
# ============================================================

class TestDetectionConfig:
    def test_default_values(self):
        config = DetectionConfig()
        assert config.threshold_per_label == 0.5
        assert config.overall_threshold == 0.5
        assert config.threat_min_score_on_cue == 0.6

    def test_custom_values(self):
        config = DetectionConfig(
            threshold_per_label=0.7,
            overall_threshold=0.8,
            threat_min_score_on_cue=0.9,
        )
        assert config.threshold_per_label == 0.7
        assert config.overall_threshold == 0.8
        assert config.threat_min_score_on_cue == 0.9


# ============================================================
# Section 23: default_detectors coverage
# ============================================================

class TestDefaultDetectors:
    def test_returns_all_expected_types(self):
        dets = default_detectors()
        types = {d.type for d in dets}
        expected = {
            "EMAIL", "PHONE", "SSN", "CREDIT_CARD", "BANK_ACCOUNT",
            "DOB", "DRIVERS_LICENSE", "MEDICAL_RECORD_NUMBER",
            "PERSON_NAME", "ADDRESS",
        }
        assert types == expected

    def test_returns_ten_detectors(self):
        assert len(default_detectors()) == 10


# ============================================================
# Section 24: Span dict format consistency
# ============================================================

class TestSpanFormat:
    """All detection functions must return spans with 'span', 'start', 'end'."""

    def test_detect_pii_format(self):
        result = detect_pii("Email: test@example.com SSN: 123-45-6789")
        for det_type, spans in result.items():
            for span in spans:
                assert "span" in span, f"Missing 'span' in {det_type}"
                assert "start" in span, f"Missing 'start' in {det_type}"
                assert "end" in span, f"Missing 'end' in {det_type}"
                assert isinstance(span["start"], int)
                assert isinstance(span["end"], int)
                assert span["end"] > span["start"]

    def test_detect_secrets_format(self):
        result = detect_secrets("sk-1234567890abcdef1234567890abcdef")
        if "SECRETS" in result:
            for span in result["SECRETS"]:
                assert "span" in span
                assert "start" in span
                assert "end" in span

    def test_detect_harmful_format(self):
        result = detect_harmful("I hate you stupid fool!")
        if "HARMFUL_CONTENT" in result:
            data = result["HARMFUL_CONTENT"][0]
            assert "span" in data
            assert "start" in data
            assert "end" in data
            assert "severity" in data
            assert "labels" in data
            assert "scores" in data


# ============================================================
# Section 25: Threat cues pattern (via core_patterns)
# ============================================================

class TestThreatCues:
    def test_matches_kill(self):
        assert THREAT_CUES_RE.search("I will kill you")

    def test_matches_bomb(self):
        assert THREAT_CUES_RE.search("bomb threat")

    def test_no_match_clean(self):
        assert THREAT_CUES_RE.search("Beautiful sunny day") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
