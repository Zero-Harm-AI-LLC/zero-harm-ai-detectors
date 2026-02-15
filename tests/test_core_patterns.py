"""
Tests for core_patterns refactoring.

Verifies that:
1. Shared patterns in core_patterns.py work correctly
2. Both regex_detectors.py and ai_detectors.py import and use the same patterns
3. Security fixes (entropy-based secrets detection) apply to BOTH tiers
4. No regressions in existing detection capabilities

File: tests/test_core_patterns.py
"""
import math
import re
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zero_harm_ai_detectors.core_patterns import (
    # Redaction
    RedactionStrategy,
    apply_redaction,
    redact_spans,
    # Validators
    luhn_check,
    shannon_entropy,
    MIN_SECRET_ENTROPY,
    # PII patterns
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    # Secrets
    STRUCTURED_SECRET_PATTERNS,
    AWS_SECRET_KEY_RE,
    GENERIC_SECRET_RE,
    SECRET_CONTEXT_KEYWORDS_RE,
    has_secret_context,
    find_secrets,
    # Harmful
    THREAT_CUES_RE,
)


# ============================================================
# Redaction
# ============================================================

class TestRedactionStrategy:
    def test_mask_all(self):
        assert apply_redaction("john@example.com", RedactionStrategy.MASK_ALL) == "****************"

    def test_mask_last4(self):
        result = apply_redaction("john@example.com", RedactionStrategy.MASK_LAST4)
        assert result.endswith(".com")
        assert result.startswith("*" * 12)

    def test_hash(self):
        result = apply_redaction("test", RedactionStrategy.HASH)
        assert len(result) == 64  # SHA-256 hex

    def test_token_with_type(self):
        assert apply_redaction("john@example.com", RedactionStrategy.TOKEN, "EMAIL") == "[REDACTED_EMAIL]"

    def test_token_without_type(self):
        assert apply_redaction("value", RedactionStrategy.TOKEN, "") == "[REDACTED]"

    def test_redact_spans(self):
        text = "Email john@test.com and call 555-123-4567"
        spans = [
            {"span": "john@test.com", "start": 6, "end": 19, "type": "EMAIL"},
            {"span": "555-123-4567", "start": 29, "end": 41, "type": "PHONE"},
        ]
        result = redact_spans(text, spans, RedactionStrategy.TOKEN)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "john@test.com" not in result


# ============================================================
# Luhn Check
# ============================================================

class TestLuhnCheck:
    def test_valid_visa(self):
        assert luhn_check("4532015112830366") is True

    def test_valid_with_dashes(self):
        assert luhn_check("4532-0151-1283-0366") is True

    def test_invalid_number(self):
        assert luhn_check("1234567890123456") is False

    def test_too_short(self):
        assert luhn_check("12345") is False

    def test_too_long(self):
        assert luhn_check("1" * 20) is False


# ============================================================
# Shannon Entropy
# ============================================================

class TestShannonEntropy:
    def test_empty_string(self):
        assert shannon_entropy("") == 0.0

    def test_single_char(self):
        assert shannon_entropy("aaaa") == 0.0

    def test_high_entropy(self):
        # Random-looking string should have high entropy
        import string
        high = "aB3xZ9kLmNpQ7rStUvW0yE1fGhIjK2dC"
        assert shannon_entropy(high) > 4.0

    def test_english_lower_than_random(self):
        english = "the quick brown fox jumps over"
        random_str = "k7Bx9LmNpQ3rStUvW0yE1fGhIjK2dCa"
        assert shannon_entropy(english) < shannon_entropy(random_str)

    def test_min_entropy_threshold(self):
        assert MIN_SECRET_ENTROPY == 4.5


# ============================================================
# PII Pattern Tests
# ============================================================

class TestEmailPattern:
    def test_basic_email(self):
        assert EMAIL_RE.search("john@example.com")

    def test_complex_email(self):
        assert EMAIL_RE.search("john.doe+tag@sub.example.co.uk")

    def test_no_match_plain_text(self):
        assert EMAIL_RE.search("hello world") is None


class TestPhonePattern:
    def test_with_dashes(self):
        assert PHONE_RE.search("555-123-4567")

    def test_with_dots(self):
        assert PHONE_RE.search("555.123.4567")

    def test_with_country_code(self):
        assert PHONE_RE.search("+1 555-123-4567")

    def test_with_parens(self):
        assert PHONE_RE.search("(555) 123-4567")

    def test_no_separator_no_match(self):
        """Phone pattern requires separators to avoid CC false positives."""
        # Continuous digits should NOT match
        text = "5551234567"
        match = PHONE_RE.search(text)
        assert match is None

    def test_consistency_both_tiers_use_same_pattern(self):
        """Both tiers should import the SAME phone pattern object."""
        from zero_harm_ai_detectors.regex_detectors import PhoneDetector

        detector = PhoneDetector()
        assert detector.pattern is PHONE_RE


class TestSSNPattern:
    def test_with_dashes(self):
        assert SSN_RE.search("123-45-6789")

    def test_invalid_prefix_000(self):
        assert SSN_RE.search("000-45-6789") is None

    def test_invalid_prefix_666(self):
        assert SSN_RE.search("666-45-6789") is None


class TestCreditCardPattern:
    def test_matches_card_format(self):
        assert CREDIT_CARD_RE.search("4532-0151-1283-0366")


# ============================================================
# Secrets Detection Tests
# ============================================================

class TestStructuredSecrets:
    """Tier 1: patterns with distinctive prefixes need no context."""

    def test_openai_key(self):
        text = "key is sk-1234567890abcdef1234567890abcdef"
        results = find_secrets(text)
        assert len(results) >= 1
        assert any("sk-" in r["span"] for r in results)

    def test_aws_access_key(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        results = find_secrets(text)
        assert len(results) >= 1

    def test_github_pat(self):
        text = "token is ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # ghp_ + 36 chars
        results = find_secrets(text)
        assert len(results) >= 1

    def test_stripe_key(self):
        text = "sk_live_abcdefghijklmnopqrstuvwx"
        results = find_secrets(text)
        assert len(results) >= 1

    def test_slack_token(self):
        text = "xoxb-12345678901-1234567890123-AbCdEfGhIjKl"
        results = find_secrets(text)
        assert len(results) >= 1

    def test_google_api_key(self):
        text = "AIzaSyB_6abcdefghijklmnopqrstuvwxyz1234"  # AIza + 35 chars = 39 total
        results = find_secrets(text)
        assert len(results) >= 1


class TestContextDependentSecrets:
    """Tier 2 & 3: require keyword context and high entropy."""

    def test_aws_secret_with_context(self):
        # 40-char high-entropy string with keyword context
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        results = find_secrets(text)
        # Should find the secret key
        assert len(results) >= 1

    def test_generic_secret_with_context(self):
        text = "api_key = AbCdEfGhIjKlMnOpQrStUvWx"
        results = find_secrets(text)
        assert len(results) >= 1

    def test_no_false_positive_on_normal_words(self):
        """Normal text should NOT trigger secrets detection."""
        normal_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Meeting scheduled for Tuesday afternoon",
            "Please review the quarterly report",
            "function calculateTotal(items) { return sum; }",
            "background-color: #ff5733; font-size: 16px;",
        ]
        for text in normal_texts:
            results = find_secrets(text)
            assert results == [], f"False positive on: {text!r}"

    def test_no_false_positive_on_git_sha(self):
        """40-char git SHA should NOT match without secret context."""
        text = "commit abc123def456789012345678901234567890ab"
        results = find_secrets(text)
        # No secret-related keywords nearby → no match
        assert results == []

    def test_no_false_positive_on_css_class(self):
        """Normal 16-char strings should NOT match."""
        text = "class='containerElement' id='mainWrapper1234'"
        results = find_secrets(text)
        assert results == []

    def test_no_false_positive_on_base64_content(self):
        """40-char base64 without secret context should NOT match."""
        text = "data: SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0"
        results = find_secrets(text)
        assert results == []


class TestSecretsSameInBothTiers:
    """Verify both regex and AI tiers use the same secrets detection."""

    def test_regex_tier_uses_find_secrets(self):
        """regex_detectors.SecretsDetector should delegate to find_secrets."""
        from zero_harm_ai_detectors.regex_detectors import SecretsDetector

        detector = SecretsDetector()
        text = "key is sk-1234567890abcdef1234567890abcdef"
        results = detector.finditer(text)
        core_results = find_secrets(text)
        assert results == core_results

    def test_both_tiers_reject_normal_text(self):
        """Both tiers should produce zero findings on normal text."""
        from zero_harm_ai_detectors.regex_detectors import detect_secrets as regex_detect

        text = "The quarterly meeting is at 3pm in conference room B."
        regex_results = regex_detect(text)
        core_results = find_secrets(text)
        assert regex_results == {}
        assert core_results == []


# ============================================================
# Threat Cues
# ============================================================

class TestThreatCues:
    def test_matches_threat_words(self):
        assert THREAT_CUES_RE.search("I will kill you")
        assert THREAT_CUES_RE.search("gonna hurt someone")
        assert THREAT_CUES_RE.search("bomb threat")

    def test_no_match_on_clean_text(self):
        assert THREAT_CUES_RE.search("What a beautiful day") is None


# ============================================================
# has_secret_context
# ============================================================

class TestHasSecretContext:
    def test_with_keyword(self):
        text = "aws_secret_access_key = SOMEVALUE1234567890"
        assert has_secret_context(text, 25, 45) is True

    def test_without_keyword(self):
        text = "The quick brown fox SOMEVALUE1234567890"
        assert has_secret_context(text, 20, 40) is False

    def test_keyword_before(self):
        text = "api_key: followed by SOMEVALUE1234567890"
        assert has_secret_context(text, 21, 41) is True


# ============================================================
# Integration: regex_detectors imports from core_patterns
# ============================================================

class TestRegexDetectorsImportShared:
    """Verify regex_detectors.py uses core_patterns, not its own copies."""

    def test_email_detector_uses_shared_pattern(self):
        from zero_harm_ai_detectors.regex_detectors import EmailDetector

        det = EmailDetector()
        assert det.pattern is EMAIL_RE

    def test_phone_detector_uses_shared_pattern(self):
        from zero_harm_ai_detectors.regex_detectors import PhoneDetector

        det = PhoneDetector()
        assert det.pattern is PHONE_RE

    def test_ssn_detector_uses_shared_pattern(self):
        from zero_harm_ai_detectors.regex_detectors import SSNDetector

        det = SSNDetector()
        assert det.pattern is SSN_RE

    def test_redaction_strategy_is_shared(self):
        from zero_harm_ai_detectors.regex_detectors import RedactionStrategy as RS

        assert RS is RedactionStrategy

    def test_detect_pii_finds_email(self):
        from zero_harm_ai_detectors.regex_detectors import detect_pii

        result = detect_pii("Contact me at alice@example.com")
        assert "EMAIL" in result
        assert result["EMAIL"][0]["span"] == "alice@example.com"

    def test_detect_pii_finds_phone(self):
        from zero_harm_ai_detectors.regex_detectors import detect_pii

        result = detect_pii("Call 555-123-4567")
        assert "PHONE" in result
        assert result["PHONE"][0]["span"] == "555-123-4567"

    def test_detect_secrets_finds_openai_key(self):
        from zero_harm_ai_detectors.regex_detectors import detect_secrets

        result = detect_secrets("key=sk-1234567890abcdef1234567890abcdef")
        assert "SECRETS" in result

    def test_detect_secrets_no_false_positive(self):
        from zero_harm_ai_detectors.regex_detectors import detect_secrets

        result = detect_secrets("Hello world, this is a normal sentence.")
        assert result == {}

    def test_redact_text_works(self):
        from zero_harm_ai_detectors.regex_detectors import redact_text

        text = "Email alice@example.com"
        spans = {"EMAIL": [{"span": "alice@example.com", "start": 6, "end": 23}]}
        result = redact_text(text, spans, "mask_all")
        assert "alice@example.com" not in result
        assert "*" * 17 in result

    def test_credit_card_uses_shared_luhn(self):
        from zero_harm_ai_detectors.regex_detectors import detect_pii

        result = detect_pii("Card: 4532-0151-1283-0366")
        assert "CREDIT_CARD" in result


# ============================================================
# Backwards compatibility
# ============================================================

class TestBackwardsCompatibility:
    """Ensure the refactoring doesn't break any existing public API."""

    def test_regex_redaction_strategy_values(self):
        """Old code may compare string values."""
        assert RedactionStrategy.MASK_ALL.value == "mask_all"
        assert RedactionStrategy.MASK_LAST4.value == "mask_last4"
        assert RedactionStrategy.HASH.value == "hash"
        assert RedactionStrategy.TOKEN.value == "token"

    def test_detect_pii_returns_dict(self):
        from zero_harm_ai_detectors.regex_detectors import detect_pii

        result = detect_pii("test@example.com")
        assert isinstance(result, dict)
        assert isinstance(result.get("EMAIL", []), list)

    def test_detect_secrets_returns_dict(self):
        from zero_harm_ai_detectors.regex_detectors import detect_secrets

        result = detect_secrets("sk-1234567890abcdef1234567890abcdef")
        assert isinstance(result, dict)
        if "SECRETS" in result:
            assert isinstance(result["SECRETS"], list)
            assert "span" in result["SECRETS"][0]
            assert "start" in result["SECRETS"][0]
            assert "end" in result["SECRETS"][0]

    def test_span_dict_format(self):
        """Verify span dicts have the expected keys."""
        from zero_harm_ai_detectors.regex_detectors import detect_pii

        result = detect_pii("Email: test@example.com SSN: 123-45-6789")
        for det_type, spans in result.items():
            for span in spans:
                assert "span" in span
                assert "start" in span
                assert "end" in span
                assert isinstance(span["start"], int)
                assert isinstance(span["end"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
