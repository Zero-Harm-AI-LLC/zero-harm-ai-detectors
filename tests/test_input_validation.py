"""
Tests for input_validation.py

File: tests/test_input_validation.py
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from input_validation import (
    InputConfig,
    InputTooLongError,
    InputValidationError,
    LineTooLongError,
    ReDoSRiskError,
    TooManyLinesError,
    ValidationResult,
    validate_input,
    validate_input_soft,
    AI_MODE_CONFIG,
    API_MODE_CONFIG,
    REGEX_MODE_CONFIG,
)


# ==================== Basic Validation ====================

class TestBasicValidation:
    """Tests for basic input handling."""

    def test_empty_string(self):
        assert validate_input("") == ""

    def test_normal_text(self):
        text = "Hello, my email is john@example.com"
        assert validate_input(text) == text

    def test_multiline_text(self):
        text = "Line 1\nLine 2\nLine 3"
        assert validate_input(text) == text

    def test_unicode_text(self):
        text = "Héllo wörld, café résumé"
        result = validate_input(text)
        assert "café" in result

    def test_type_error_on_bytes(self):
        with pytest.raises(TypeError, match="Expected str"):
            validate_input(b"hello bytes")

    def test_type_error_on_int(self):
        with pytest.raises(TypeError, match="Expected str"):
            validate_input(42)

    def test_type_error_on_none(self):
        with pytest.raises(TypeError, match="Expected str"):
            validate_input(None)


# ==================== Size Limits ====================

class TestSizeLimits:
    """Tests for text length enforcement."""

    def test_text_within_limit(self):
        config = InputConfig(max_length=100)
        text = "a" * 100
        assert validate_input(text, config) == text

    def test_text_exceeds_limit_raises(self):
        config = InputConfig(max_length=100)
        text = "a" * 101
        with pytest.raises(InputTooLongError) as exc_info:
            validate_input(text, config)
        assert exc_info.value.actual == 101
        assert exc_info.value.limit == 100

    def test_text_exceeds_limit_truncates(self):
        config = InputConfig(max_length=100, truncate_on_overflow=True)
        text = "a" * 200
        result = validate_input(text, config)
        assert len(result) == 100

    def test_ai_mode_uses_stricter_limit(self):
        config = InputConfig(max_length=100_000, max_length_ai=10_000, max_line_length=100_000)
        # Use varied content so ReDoS doesn't trigger
        base = "Hello world testing input validation. "
        text = base * (20_000 // len(base) + 1)
        text = text[:20_000]
        # Regex mode: fine
        validate_input(text, config, mode="regex")
        # AI mode: too long
        with pytest.raises(InputTooLongError):
            validate_input(text, config, mode="ai")

    def test_line_too_long_raises(self):
        config = InputConfig(max_line_length=50)
        text = "a" * 51
        with pytest.raises(LineTooLongError) as exc_info:
            validate_input(text, config)
        assert exc_info.value.limit == 50

    def test_line_too_long_truncates(self):
        config = InputConfig(max_line_length=50, truncate_on_overflow=True)
        text = "short line\n" + "x" * 100 + "\nanother short"
        result = validate_input(text, config)
        lines = result.split("\n")
        assert len(lines[1]) == 50

    def test_too_many_lines_raises(self):
        config = InputConfig(max_lines=10)
        text = "\n".join(f"line {i}" for i in range(11))
        with pytest.raises(TooManyLinesError):
            validate_input(text, config)

    def test_too_many_lines_truncates(self):
        config = InputConfig(max_lines=10, truncate_on_overflow=True)
        text = "\n".join(f"line {i}" for i in range(20))
        result = validate_input(text, config)
        assert result.count("\n") == 9  # 10 lines = 9 newlines


# ==================== Character Sanitization ====================

class TestCharacterSanitization:
    """Tests for dangerous character removal."""

    def test_null_bytes_removed(self):
        text = "hello\x00world"
        result = validate_input(text)
        assert "\x00" not in result
        assert result == "helloworld"

    def test_null_bytes_kept_when_disabled(self):
        config = InputConfig(strip_null_bytes=False, strip_control_chars=False)
        text = "hello\x00world"
        result = validate_input(text, config)
        assert "\x00" in result

    def test_control_chars_removed(self):
        # Bell, backspace, escape, form feed
        text = "hello\x07\x08\x1b\x0cworld"
        result = validate_input(text)
        assert result == "helloworld"

    def test_tabs_newlines_preserved(self):
        text = "hello\tworld\nfoo\rbar"
        result = validate_input(text)
        assert "\t" in result
        assert "\n" in result
        assert "\r" in result

    def test_zero_width_chars_removed(self):
        # Zero-width space between "John" and "Smith" to evade name detection
        text = "John\u200bSmith"
        result = validate_input(text)
        assert result == "JohnSmith"

    def test_zero_width_joiner_removed(self):
        text = "secret\u200dkey\u200c=abc123"
        result = validate_input(text)
        assert result == "secretkey=abc123"

    def test_bom_removed(self):
        text = "\ufeffHello world"
        result = validate_input(text)
        assert result == "Hello world"

    def test_unicode_normalization(self):
        # é as e + combining accent vs precomposed é
        decomposed = "caf\u0065\u0301"   # e + combining acute
        result = validate_input(decomposed)
        assert result == "café"  # NFC normalized

    def test_unicode_normalization_disabled(self):
        config = InputConfig(normalize_unicode=False)
        decomposed = "caf\u0065\u0301"
        result = validate_input(decomposed, config)
        # Should keep decomposed form
        assert len(result) == 5  # c-a-f-e-combining


# ==================== ReDoS Protection ====================

class TestReDoSProtection:
    """Tests for regex denial-of-service prevention."""

    def test_repeated_chars_raises(self):
        config = InputConfig(max_repeated_chars=100)
        text = "a" * 10_000  # Way more than 100 repeated
        with pytest.raises(ReDoSRiskError, match="repeated"):
            validate_input(text, config)

    def test_repeated_chars_under_limit_ok(self):
        config = InputConfig(max_repeated_chars=100)
        text = "a" * 50 + " " + "b" * 50
        result = validate_input(text, config)
        assert result == text

    def test_repeated_pattern_raises(self):
        config = InputConfig(max_repeated_pattern_length=500)
        text = "ab" * 1000  # 2000 chars of repeated "ab"
        with pytest.raises(ReDoSRiskError, match="repeated"):
            validate_input(text, config)

    def test_repeated_chars_truncated(self):
        config = InputConfig(
            max_repeated_chars=100,
            truncate_on_overflow=True,
        )
        text = "hello " + "a" * 500 + " world"
        result = validate_input(text, config)
        # The run of a's should be collapsed to max_repeated_chars
        assert len(result) < len(text)
        assert "hello" in result
        assert "world" in result
        # The 'a' run should be at most max_repeated_chars
        import re
        longest_a_run = max(len(m.group()) for m in re.finditer(r"a+", result))
        assert longest_a_run <= config.max_repeated_chars

    def test_normal_text_not_flagged(self):
        """Real-world text should never trigger ReDoS checks."""
        text = (
            "Dear Mr. Johnson,\n\n"
            "Please find attached the quarterly report for Q3 2024. "
            "The analysis shows a 15% increase in revenue compared to "
            "the previous quarter. Key highlights include:\n\n"
            "- Revenue: $2.3M (up from $2.0M)\n"
            "- Customer acquisition: 450 new accounts\n"
            "- Churn rate decreased to 3.2%\n\n"
            "Best regards,\nSarah Williams\nVP of Operations"
        )
        result = validate_input(text)
        assert result == text


# ==================== Evasion Prevention ====================

class TestEvasionPrevention:
    """Test that adversarial inputs can't bypass detection."""

    def test_zero_width_in_email(self):
        """Zero-width chars in an email shouldn't evade detection."""
        # Attacker inserts ZWS inside email to break regex
        evasion = "john\u200b@\u200bexample\u200b.com"
        result = validate_input(evasion)
        assert result == "john@example.com"

    def test_zero_width_in_ssn(self):
        """Zero-width chars in SSN shouldn't evade detection."""
        evasion = "123\u200c-\u200d45\u200b-6789"
        result = validate_input(evasion)
        assert result == "123-45-6789"

    def test_zero_width_in_api_key(self):
        """Zero-width chars in API key shouldn't evade detection."""
        evasion = "sk-\u200b1234567890abcdef\u200c1234567890abcdef"
        result = validate_input(evasion)
        assert result == "sk-1234567890abcdef1234567890abcdef"

    def test_null_byte_in_secret(self):
        """Null bytes shouldn't split a secret across string boundaries."""
        evasion = "api_key=sk-12345678\x0090abcdef1234567890abcdef"
        result = validate_input(evasion)
        assert "\x00" not in result

    def test_bidi_override_attack(self):
        """Bidi override chars used to visually disguise text."""
        # RIGHT-TO-LEFT OVERRIDE to make text appear different
        evasion = "safe\u202enoitceted\u202c text"
        result = validate_input(evasion)
        assert "\u202e" not in result
        assert "\u202c" not in result


# ==================== Soft Validation ====================

class TestSoftValidation:
    """Tests for validate_input_soft (non-raising mode)."""

    def test_valid_input(self):
        result = validate_input_soft("hello world")
        assert result.is_valid
        assert result.text == "hello world"
        assert not result.was_truncated
        assert len(result.warnings) == 0

    def test_oversized_truncated(self):
        config = InputConfig(max_length=50)
        result = validate_input_soft("a" * 100, config)
        assert len(result.text) == 50
        assert result.was_truncated
        assert len(result.warnings) > 0

    def test_non_string_returns_empty(self):
        result = validate_input_soft(42)
        assert not result.is_valid
        assert result.text == ""

    def test_sanitized_input(self):
        result = validate_input_soft("hello\x00\u200bworld")
        assert result.was_sanitized
        assert result.text == "helloworld"


# ==================== Config Presets ====================

class TestConfigPresets:
    """Test that preset configs are sensible."""

    def test_regex_config_allows_large_input(self):
        # Use varied multiline content (no repeated chars, no long lines)
        line = "The quick brown fox jumps over the lazy dog. "  # 46 chars
        lines = [line * 100 for _ in range(80)]  # 80 lines of ~4600 chars = ~368K
        text = "\n".join(lines)
        result = validate_input(text, REGEX_MODE_CONFIG, mode="regex")
        assert len(result) == len(text)

    def test_ai_config_rejects_large_input(self):
        text = "x" * 60_000
        with pytest.raises(InputTooLongError):
            validate_input(text, AI_MODE_CONFIG, mode="ai")

    def test_api_config_truncates(self):
        # Use multiline varied content to hit total limit, not line limit
        line = "The quick brown fox jumps over the lazy dog. "  # 46 chars
        # 200 chars per line, 1000 lines = 200K chars total
        lines = [(line * 5)[:200] for _ in range(1000)]
        text = "\n".join(lines)
        assert len(text) > API_MODE_CONFIG.max_length  # Confirm it exceeds
        result = validate_input(text, API_MODE_CONFIG, mode="regex")
        assert len(result) <= API_MODE_CONFIG.max_length


# ==================== Integration Sketch ====================

class TestIntegrationPoints:
    """Show how validation plugs into each public entry point."""

    def test_detect_pii_entry_point(self):
        """Simulate what detect_pii should do."""
        user_input = "Contact\u200b john@example.com\x00"
        clean = validate_input(user_input, mode="regex")
        assert clean == "Contact john@example.com"
        # Now safe to pass to regex detectors

    def test_ai_pipeline_entry_point(self):
        """Simulate what ZeroHarmDetector(mode='ai').detect should do."""
        user_input = "John Smith at Microsoft"
        clean = validate_input(user_input, mode="ai")
        assert clean == user_input

    def test_api_endpoint_entry_point(self):
        """Simulate a Flask/FastAPI endpoint."""
        user_input = "a" * 500_000  # Huge input
        result = validate_input_soft(user_input, API_MODE_CONFIG)
        assert len(result.text) <= API_MODE_CONFIG.max_length
        assert result.was_truncated


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
