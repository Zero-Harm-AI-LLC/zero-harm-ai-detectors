#!/usr/bin/env python3
"""
Zero Harm AI Detectors - Basic Usage Examples

Demonstrates the unified API with mode='regex' (default) and mode='ai'.

File: examples/basic_usage.py
"""
from zero_harm_ai_detectors import (
    detect,
    DetectionResult,
    RedactionStrategy,
    AI_AVAILABLE,
)


def main():
    print("=" * 70)
    print("Zero Harm AI Detectors - Basic Usage Examples")
    print("=" * 70)
    print(f"\nAI Mode Available: {AI_AVAILABLE}")
    if not AI_AVAILABLE:
        print("(Install with `pip install zero_harm_ai_detectors[ai]` for AI mode)\n")

    # ================================================================
    # Example 1: Basic Detection (Regex Mode - Default)
    # ================================================================
    print("-" * 70)
    print("Example 1: Basic Detection (mode='regex', default)")
    print("-" * 70)

    text = "Contact john.smith@example.com or call 555-123-4567"
    result = detect(text)  # mode="regex" is default

    print(f"Input:    {text}")
    print(f"Redacted: {result.redacted_text}")
    print(f"Mode:     {result.mode}")
    print(f"Found {len(result.detections)} detections:")
    for det in result.detections:
        print(f"  - {det.type}: '{det.text}' (confidence: {det.confidence:.0%})")

    # ================================================================
    # Example 2: Secrets Detection
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 2: Secrets Detection")
    print("-" * 70)

    text = "API key: sk-1234567890abcdef1234567890abcdef"
    result = detect(text)

    print(f"Input:    {text}")
    print(f"Redacted: {result.redacted_text}")
    for det in result.detections:
        print(f"  - {det.type}: '{det.text[:30]}...'")

    # ================================================================
    # Example 3: Harmful Content Detection
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 3: Harmful Content Detection")
    print("-" * 70)

    text = "I hate you, you stupid idiot!"
    result = detect(text)

    print(f"Input:    {text}")
    print(f"Harmful:  {result.harmful}")
    print(f"Severity: {result.severity}")

    # ================================================================
    # Example 4: Mixed Content
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 4: Mixed Content (PII + Secrets)")
    print("-" * 70)

    text = """
    Contact: john@example.com
    Phone: 555-123-4567
    SSN: 123-45-6789
    API Key: sk-abc123def456abc123def456abc123def456
    """
    result = detect(text)

    print(f"Found {len(result.detections)} total detections:")
    by_type = {}
    for det in result.detections:
        by_type.setdefault(det.type, []).append(det)
    for det_type, dets in sorted(by_type.items()):
        print(f"  {det_type}: {len(dets)} found")

    print(f"\nRedacted:\n{result.redacted_text}")

    # ================================================================
    # Example 5: Redaction Strategies
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 5: Redaction Strategies")
    print("-" * 70)

    text = "Email: john@example.com"
    strategies = ["token", "mask_all", "mask_last4", "hash"]
    
    for strategy in strategies:
        result = detect(text, redaction_strategy=strategy)
        print(f"  {strategy:10}: {result.redacted_text}")

    # ================================================================
    # Example 6: Selective Detection
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 6: Selective Detection")
    print("-" * 70)

    text = "Email john@test.com, API key sk-abc123def456abc123def456abc123def456"

    # PII only
    result = detect(text, detect_pii=True, detect_secrets=False, detect_harmful=False)
    print(f"PII only:     {[d.type for d in result.detections]}")

    # Secrets only
    result = detect(text, detect_pii=False, detect_secrets=True, detect_harmful=False)
    print(f"Secrets only: {[d.type for d in result.detections]}")

    # ================================================================
    # Example 7: AI Mode (if available)
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 7: AI Mode (better for names/locations/orgs)")
    print("-" * 70)

    text = "Contact John Smith at Microsoft in Seattle"

    if AI_AVAILABLE:
        result = detect(text, mode="ai")
        print(f"Input:    {text}")
        print(f"Mode:     {result.mode}")
        print(f"Found {len(result.detections)} detections:")
        for det in result.detections:
            method = det.metadata.get("method", "unknown")
            print(f"  - {det.type}: '{det.text}' (method: {method})")
    else:
        print("AI mode not available. Using regex mode instead:")
        result = detect(text, mode="regex")
        print(f"Regex found: {[d.type for d in result.detections]}")
        print("(Note: Regex mode has limited person name detection ~30%)")

    # ================================================================
    # Example 8: Result Format Conversion
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 8: Result Format Conversion")
    print("-" * 70)

    text = "Contact john@example.com"
    result = detect(text)

    # Full dict format
    full = result.to_dict()
    print(f"to_dict() keys: {list(full.keys())}")
    print(f"  mode: {full['mode']}")

    # Legacy format
    legacy = result.to_legacy_dict()
    print(f"to_legacy_dict(): {legacy}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "-" * 70)
    print("Mode Comparison Summary")
    print("-" * 70)

    print("""
    | Feature          | mode='regex' | mode='ai'    |
    |------------------|--------------|--------------|
    | Speed            | 1-5ms        | 50-200ms     |
    | Email/Phone/SSN  | 95%+ ✓       | 95%+ ✓       |
    | Person Names     | ~30%         | ~90% ✓       |
    | Locations        | ❌           | ~85% ✓       |
    | Organizations    | ❌           | ~80% ✓       |
    | Dependencies     | regex only   | transformers |
    """)

    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
