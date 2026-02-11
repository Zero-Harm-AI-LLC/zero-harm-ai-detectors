"""
Input validation and sanitization for Zero Harm AI Detectors

Prevents:
- Denial-of-service via oversized inputs
- ReDoS via pathological regex inputs
- Memory exhaustion from transformer models on huge texts
- Null byte injection and control character attacks
- Unicode normalization attacks

Usage:
    from zero_harm_ai_detectors.input_validation import validate_input, InputConfig

    # Uses sensible defaults
    clean_text = validate_input(user_text)

    # Custom limits
    config = InputConfig(max_length=50_000, max_line_length=5_000)
    clean_text = validate_input(user_text, config)

File: zero_harm_ai_detectors/input_validation.py
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


# ==================== Configuration ====================

@dataclass
class InputConfig:
    """
    Configurable limits for input validation.

    All limits have safe defaults. Override per-mode:
    - Regex mode can handle larger inputs (fast)
    - AI mode should use tighter limits (memory + latency)
    """

    # ── Size limits ──────────────────────────────────────────
    max_length: int = 100_000
    """Maximum text length in characters. Default 100K covers most
    real-world documents. Transformer models start to struggle above
    ~50K chars and regex ReDoS risk grows with length."""

    max_length_ai: int = 50_000
    """Stricter limit for AI mode. Transformer tokenizers typically
    truncate at 512 tokens (~2K chars) per chunk anyway, but we need
    to prevent OOM during pre-processing and batching."""

    max_line_length: int = 10_000
    """Maximum single-line length. Lines longer than this are often
    base64 blobs or minified code — they cause regex backtracking."""

    max_lines: int = 50_000
    """Maximum number of lines. Prevents memory exhaustion from
    line-splitting operations."""

    # ── Content limits ───────────────────────────────────────
    max_repeated_chars: int = 100
    """Maximum consecutive identical characters. Strings like
    'aaaa...' (1M chars) cause catastrophic backtracking in
    patterns like \\b\\w+\\b."""

    max_repeated_pattern_length: int = 500
    """Maximum length of a repeated short pattern (e.g., 'ab' * 10000).
    These are common ReDoS payloads."""

    # ── Character policy ─────────────────────────────────────
    strip_null_bytes: bool = True
    """Remove \\x00 bytes which can cause C-level string truncation
    in downstream libraries (sqlite, some tokenizers)."""

    strip_control_chars: bool = True
    """Remove ASCII control characters (0x00-0x1F, 0x7F) except
    tab (0x09), newline (0x0A), and carriage return (0x0D)."""

    normalize_unicode: bool = True
    """Apply NFC normalization to prevent homoglyph attacks and
    ensure consistent matching (e.g., é as single char vs e + ◌́)."""

    strip_zero_width: bool = True
    """Remove zero-width characters (ZWJ, ZWNJ, ZWSP, BOM) that
    can be used to evade pattern detection by splitting tokens."""

    # ── Behavior ─────────────────────────────────────────────
    truncate_on_overflow: bool = False
    """If True, silently truncate oversized input instead of raising.
    Useful for best-effort processing in API endpoints."""

    raise_on_invalid: bool = True
    """If True, raise InputValidationError. If False, return
    sanitized text with warnings in metadata."""


# ── Presets for common use cases ─────────────────────────────

REGEX_MODE_CONFIG = InputConfig(
    max_length=500_000,      # Regex is fast, can handle more
    max_length_ai=50_000,
    max_line_length=10_000,
    truncate_on_overflow=False,
)

AI_MODE_CONFIG = InputConfig(
    max_length=50_000,       # Tighter for transformers
    max_length_ai=50_000,
    max_line_length=5_000,
    truncate_on_overflow=False,
)

API_MODE_CONFIG = InputConfig(
    max_length=100_000,
    max_length_ai=50_000,
    max_line_length=10_000,
    truncate_on_overflow=True,  # Don't crash the API
    raise_on_invalid=False,
)


# ==================== Exceptions ====================

class InputValidationError(ValueError):
    """Raised when input fails validation checks."""

    def __init__(self, message: str, field: str = "text", limit: Optional[int] = None):
        self.field = field
        self.limit = limit
        super().__init__(message)


class InputTooLongError(InputValidationError):
    """Input exceeds maximum allowed length."""

    def __init__(self, actual: int, limit: int, mode: str = ""):
        mode_str = f" for {mode} mode" if mode else ""
        super().__init__(
            f"Input length {actual:,} exceeds maximum {limit:,} characters{mode_str}. "
            f"Truncate your input or process in chunks.",
            field="text",
            limit=limit,
        )
        self.actual = actual


class LineTooLongError(InputValidationError):
    """A single line exceeds maximum allowed length."""

    def __init__(self, line_num: int, actual: int, limit: int):
        super().__init__(
            f"Line {line_num} has {actual:,} characters, exceeding the "
            f"{limit:,} character line limit. This may indicate binary or "
            f"encoded content that should be decoded first.",
            field="text",
            limit=limit,
        )


class TooManyLinesError(InputValidationError):
    """Input has too many lines."""

    def __init__(self, actual: int, limit: int):
        super().__init__(
            f"Input has {actual:,} lines, exceeding the {limit:,} line limit.",
            field="text",
            limit=limit,
        )


class ReDoSRiskError(InputValidationError):
    """Input contains patterns likely to cause catastrophic regex backtracking."""

    def __init__(self, detail: str):
        super().__init__(
            f"Input contains a pattern that could cause slow regex processing: {detail}. "
            f"This looks like encoded or binary content rather than natural text.",
            field="text",
        )


# ==================== Sanitization Helpers ====================

# Zero-width characters used to evade detection
_ZERO_WIDTH_CHARS = re.compile(
    "["
    "\u200b"  # Zero-width space
    "\u200c"  # Zero-width non-joiner
    "\u200d"  # Zero-width joiner
    "\u2060"  # Word joiner
    "\ufeff"  # BOM / zero-width no-break space
    "\u00ad"  # Soft hyphen (invisible in most renderers)
    "\u034f"  # Combining grapheme joiner
    "\u061c"  # Arabic letter mark
    "\u115f"  # Hangul choseong filler
    "\u1160"  # Hangul jungseong filler
    "\u17b4"  # Khmer vowel inherent aq
    "\u17b5"  # Khmer vowel inherent aa
    "\u180e"  # Mongolian vowel separator
    "\u2000-\u200f"  # Various spaces and direction marks
    "\u202a-\u202e"  # Bidi controls
    "\u2066-\u2069"  # Bidi isolate controls
    "\ufff9-\ufffb"  # Interlinear annotations
    "]+"
)

# Control characters (keep \t, \n, \r)
_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# Repeated single character (e.g., 'aaaaaaa...')
_REPEATED_SINGLE = re.compile(r"(.)\1{99,}")

# Repeated short pattern (e.g., 'abababab...')
_REPEATED_SHORT_PATTERN = re.compile(r"(.{1,4})\1{49,}")


def _sanitize_characters(text: str, config: InputConfig) -> str:
    """Remove dangerous characters based on config."""
    if config.strip_null_bytes:
        text = text.replace("\x00", "")

    if config.strip_control_chars:
        text = _CONTROL_CHARS.sub("", text)

    if config.strip_zero_width:
        text = _ZERO_WIDTH_CHARS.sub("", text)

    if config.normalize_unicode:
        text = unicodedata.normalize("NFC", text)

    return text


def _check_redos_risk(text: str, config: InputConfig) -> Optional[str]:
    """
    Check for patterns that cause catastrophic regex backtracking.

    Returns a description of the risk, or None if safe.
    """
    # Check for long runs of repeated characters
    match = _REPEATED_SINGLE.search(text)
    if match:
        char = repr(match.group(1))
        length = len(match.group(0))
        if length > config.max_repeated_chars:
            return f"{length} repeated {char} characters"

    # Check for repeated short patterns
    match = _REPEATED_SHORT_PATTERN.search(text)
    if match:
        pattern = repr(match.group(1))
        length = len(match.group(0))
        if length > config.max_repeated_pattern_length:
            return f"pattern {pattern} repeated {length} chars"

    return None


# ==================== Validation Result ====================

@dataclass
class ValidationResult:
    """Result of input validation, used when raise_on_invalid=False."""
    text: str
    is_valid: bool
    was_truncated: bool = False
    was_sanitized: bool = False
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ==================== Main Validation Function ====================

def validate_input(
    text: str,
    config: Optional[InputConfig] = None,
    mode: str = "regex",
) -> str:
    """
    Validate and sanitize input text.

    This should be called at every public entry point before any
    regex or transformer processing.

    Args:
        text: Raw input text from user
        config: Validation configuration (uses defaults if None)
        mode: 'regex' or 'ai' — affects size limits

    Returns:
        Sanitized text safe for processing

    Raises:
        InputValidationError: If input fails validation and
            config.raise_on_invalid is True
        TypeError: If text is not a string
    """
    if config is None:
        config = InputConfig()

    # ── Type check ───────────────────────────────────────────
    if not isinstance(text, str):
        raise TypeError(
            f"Expected str, got {type(text).__name__}. "
            f"Decode bytes with text.decode('utf-8') first."
        )

    # ── Empty input fast path ────────────────────────────────
    if not text:
        return ""

    # ── Character sanitization (before size checks) ──────────
    original_len = len(text)
    text = _sanitize_characters(text, config)

    # ── Size limits ──────────────────────────────────────────
    max_len = config.max_length_ai if mode == "ai" else config.max_length

    if len(text) > max_len:
        if config.truncate_on_overflow:
            text = text[:max_len]
        elif config.raise_on_invalid:
            raise InputTooLongError(
                actual=len(text), limit=max_len, mode=mode
            )

    # ── Line limits ──────────────────────────────────────────
    lines = text.split("\n")

    if len(lines) > config.max_lines:
        if config.truncate_on_overflow:
            lines = lines[: config.max_lines]
            text = "\n".join(lines)
        elif config.raise_on_invalid:
            raise TooManyLinesError(
                actual=len(lines), limit=config.max_lines
            )

    for i, line in enumerate(lines, 1):
        if len(line) > config.max_line_length:
            if config.truncate_on_overflow:
                lines[i - 1] = line[: config.max_line_length]
            elif config.raise_on_invalid:
                raise LineTooLongError(
                    line_num=i, actual=len(line), limit=config.max_line_length
                )

    if config.truncate_on_overflow:
        text = "\n".join(lines)

    # ── ReDoS risk check ─────────────────────────────────────
    redos_detail = _check_redos_risk(text, config)
    if redos_detail:
        if config.truncate_on_overflow:
            # Collapse repeated chars to max allowed
            text = _REPEATED_SINGLE.sub(
                lambda m: m.group(1) * min(len(m.group(0)), config.max_repeated_chars),
                text,
            )
            text = _REPEATED_SHORT_PATTERN.sub(
                lambda m: m.group(1) * min(
                    len(m.group(0)) // len(m.group(1)),
                    config.max_repeated_pattern_length // len(m.group(1)),
                ),
                text,
            )
        elif config.raise_on_invalid:
            raise ReDoSRiskError(redos_detail)

    return text


def validate_input_soft(
    text: str,
    config: Optional[InputConfig] = None,
    mode: str = "regex",
) -> ValidationResult:
    """
    Validate input and return a result object instead of raising.

    Always sanitizes and truncates, collecting warnings instead
    of raising errors. Useful for API endpoints that want
    best-effort processing.

    Args:
        text: Raw input text
        config: Validation configuration
        mode: 'regex' or 'ai'

    Returns:
        ValidationResult with sanitized text and any warnings
    """
    if config is None:
        config = InputConfig(truncate_on_overflow=True, raise_on_invalid=False)
    else:
        # Override to non-raising mode
        config = InputConfig(
            max_length=config.max_length,
            max_length_ai=config.max_length_ai,
            max_line_length=config.max_line_length,
            max_lines=config.max_lines,
            max_repeated_chars=config.max_repeated_chars,
            max_repeated_pattern_length=config.max_repeated_pattern_length,
            strip_null_bytes=config.strip_null_bytes,
            strip_control_chars=config.strip_control_chars,
            normalize_unicode=config.normalize_unicode,
            strip_zero_width=config.strip_zero_width,
            truncate_on_overflow=True,
            raise_on_invalid=False,
        )

    warnings = []
    was_truncated = False

    if not isinstance(text, str):
        return ValidationResult(
            text="",
            is_valid=False,
            warnings=[f"Expected str, got {type(text).__name__}"],
        )

    original = text
    text = validate_input(text, config, mode)

    if len(text) < len(original):
        was_truncated = True
        warnings.append(
            f"Input was truncated/sanitized from {len(original):,} to {len(text):,} chars"
        )

    return ValidationResult(
        text=text,
        is_valid=len(warnings) == 0,
        was_truncated=was_truncated,
        was_sanitized=text != original,
        warnings=warnings,
    )
