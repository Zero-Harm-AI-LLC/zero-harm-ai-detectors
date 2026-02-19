"""
Input validation for DoS/ReDoS protection.

All public text-accepting APIs pass their input through validate_input()
before running any regex or ML processing.

File: zero_harm_ai_detectors/input_validation.py
"""
from dataclasses import dataclass
from typing import Optional


class InputValidationError(Exception):
    """Raised when input fails a validation check."""
    pass


class InputTooLongError(InputValidationError):
    """Raised when input exceeds the configured maximum length."""
    pass


@dataclass
class InputConfig:
    """
    Validation constraints for a specific call site.

    Attributes:
        max_length:      Maximum total character count.
        max_line_length: Maximum characters on a single line.
        strip_null_bytes: Replace null bytes with empty string.
        reject_binary:   Raise if the text looks like binary data.
    """
    max_length: int = 100_000
    max_line_length: int = 10_000
    strip_null_bytes: bool = True
    reject_binary: bool = True


# ---------------------------------------------------------------------------
# Preset configs — import these at each call site instead of hard-coding limits
# ---------------------------------------------------------------------------

# Regex mode: generous limits, regex is fast and handles large inputs safely
REGEX_MODE_CONFIG = InputConfig(
    max_length=1_000_000,
    max_line_length=50_000,
)

# AI mode: stricter limits — transformer context windows are bounded
AI_MODE_CONFIG = InputConfig(
    max_length=100_000,
    max_line_length=10_000,
)

# API / backend proxy: conservative limits for untrusted network input
API_MODE_CONFIG = InputConfig(
    max_length=50_000,
    max_line_length=5_000,
)


def validate_input(
    text: str,
    config: Optional[InputConfig] = None,
    raise_on_error: bool = True,
) -> str:
    """
    Validate and sanitise input text.

    Always call this at every public entry point before running regex or ML.

    Args:
        text:           The input string to validate.
        config:         Validation constraints.  Defaults to a safe baseline.
        raise_on_error: When True (default) raise on violations.
                        When False, silently truncate / sanitise instead.

    Returns:
        The validated (and possibly sanitised) string.

    Raises:
        InputValidationError: For None input or binary-looking data
                              (when raise_on_error=True).
        InputTooLongError:    When text exceeds max_length
                              (when raise_on_error=True).
    """
    if config is None:
        config = InputConfig()

    # ------------------------------------------------------------------ None
    if text is None:
        if raise_on_error:
            raise InputValidationError("Input text cannot be None.")
        return ""

    # -------------------------------------------------------- type coercion
    if not isinstance(text, str):
        text = str(text)

    # --------------------------------------------------------- null bytes
    if config.strip_null_bytes and "\x00" in text:
        text = text.replace("\x00", "")

    # --------------------------------------------------------- binary check
    if config.reject_binary:
        sample = text[:1_000]
        non_printable = sum(
            1 for c in sample if ord(c) < 32 and c not in "\n\r\t"
        )
        if non_printable > max(10, len(sample) // 10):
            if raise_on_error:
                raise InputValidationError(
                    "Input appears to contain binary data and cannot be processed."
                )
            return ""

    # ------------------------------------------------------- total length
    if len(text) > config.max_length:
        if raise_on_error:
            raise InputTooLongError(
                f"Input length {len(text):,} exceeds the maximum of "
                f"{config.max_length:,} characters."
            )
        text = text[: config.max_length]

    # ---------------------------------------------------- per-line length
    if config.max_line_length:
        lines = text.split("\n")
        if any(len(line) > config.max_line_length for line in lines):
            if raise_on_error:
                raise InputValidationError(
                    f"A line in the input exceeds the maximum line length of "
                    f"{config.max_line_length:,} characters."
                )
            text = "\n".join(line[: config.max_line_length] for line in lines)

    return text


def validate_input_soft(
    text: str,
    config: Optional[InputConfig] = None,
) -> str:
    """
    Validate input without raising — truncates / sanitises silently.

    Use this when you want best-effort processing rather than hard failures
    (e.g. internal pipelines where the caller already validated upstream).
    """
    return validate_input(text, config, raise_on_error=False)
