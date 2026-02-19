"""
Input validation for DoS/ReDoS protection.

File: zero_harm_ai_detectors/input_validation.py
"""
import re
from dataclasses import dataclass
from typing import Optional


class InputValidationError(Exception):
    """Base exception for input validation errors."""
    pass


class InputTooLongError(InputValidationError):
    """Input exceeds maximum length."""
    pass


@dataclass
class InputConfig:
    """Configuration for input validation."""
    max_length: int = 100_000
    max_line_length: int = 10_000
    strip_null_bytes: bool = True
    normalize_unicode: bool = True
    reject_binary: bool = True


# Preset configurations
REGEX_MODE_CONFIG = InputConfig(
    max_length=1_000_000,  # 1MB for regex
    max_line_length=50_000,
)

AI_MODE_CONFIG = InputConfig(
    max_length=100_000,  # 100KB for AI (model context limits)
    max_line_length=10_000,
)

API_MODE_CONFIG = InputConfig(
    max_length=50_000,  # 50KB for API
    max_line_length=5_000,
)


def validate_input(
    text: str,
    config: Optional[InputConfig] = None,
    raise_on_error: bool = True,
) -> str:
    """
    Validate and sanitize input text.
    
    Args:
        text: Input text to validate
        config: Validation configuration
        raise_on_error: If True, raise exceptions; if False, truncate/sanitize
    
    Returns:
        Validated/sanitized text
    
    Raises:
        InputTooLongError: If text exceeds max_length and raise_on_error=True
        InputValidationError: For other validation failures
    """
    if config is None:
        config = InputConfig()
    
    # Check for None
    if text is None:
        if raise_on_error:
            raise InputValidationError("Input cannot be None")
        return ""
    
    # Check length
    if len(text) > config.max_length:
        if raise_on_error:
            raise InputTooLongError(
                f"Input length {len(text)} exceeds maximum {config.max_length}"
            )
        text = text[:config.max_length]
    
    # Strip null bytes
    if config.strip_null_bytes:
        text = text.replace("\x00", "")
    
    # Check for binary content
    if config.reject_binary:
        # Check for high concentration of non-printable characters
        non_printable = sum(1 for c in text[:1000] if ord(c) < 32 and c not in '\n\r\t')
        if non_printable > 100:  # More than 10% non-printable
            if raise_on_error:
                raise InputValidationError("Input appears to be binary data")
            return ""
    
    # Check line lengths
    lines = text.split('\n')
    if any(len(line) > config.max_line_length for line in lines):
        if raise_on_error:
            raise InputValidationError(
                f"Line length exceeds maximum {config.max_line_length}"
            )
        # Truncate long lines
        text = '\n'.join(
            line[:config.max_line_length] for line in lines
        )
    
    return text


def validate_input_soft(
    text: str,
    config: Optional[InputConfig] = None,
) -> str:
    """
    Validate input without raising exceptions (truncates/sanitizes instead).
    
    Args:
        text: Input text
        config: Validation configuration
    
    Returns:
        Sanitized text
    """
    return validate_input(text, config, raise_on_error=False)
