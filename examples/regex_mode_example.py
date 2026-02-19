#!/usr/bin/env python3
"""
Zero Harm AI Detectors - Regex Mode Examples

Regex mode (mode='regex') provides:
- Fast detection (1-5ms per text)
- 95%+ accuracy on structured data (email, phone, SSN, credit card, secrets)
- ~30-40% accuracy on person names (use AI mode for better accuracy)
- No additional dependencies

Best for: High-volume scanning, structured data, latency-sensitive apps.

File: examples/regex_mode_example.py
"""
from zero_harm_ai_detectors import (
    detect,
    # Individual detectors
    detect_emails,
    detect_phones,
    detect_ssns,
    detect_credit_cards,
    detect_secrets_regex,
    detect_all_regex,
    # Utilities
    RedactionStrategy,
    find_secrets,
    luhn_check,
)


def main():
    print("=" * 70)
    print("Zero Harm AI Detectors - Regex Mode Examples")
    print("=" * 70)

    # ================================================================
    # Example 1: Using detect() with mode='regex' (default)
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 1: detect() with mode='regex'")
    print("-" * 70)

    text = "Email: john@example.com, Phone: 555-123-4567, SSN: 123-45-6789"
    result = detect(text, mode="regex")  # mode="regex" is default

    print(f"Input: {text}")
    print(f"Mode:  {result.mode}")
    print(f"Redacted: {result.redacted_text}")

    # ================================================================
    # Example 2: Individual Detectors
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 2: Individual Detector Functions")
    print("-" * 70)

    # Email detection
    emails = detect_emails("Contact alice@test.com or bob@example.org")
    print(f"Emails: {[e.text for e in emails]}")

    # Phone detection
    phones = detect_phones("Call 555-123-4567 or (800) 555-1234")
    print(f"Phones: {[p.text for p in phones]}")

    # SSN detection (validates format)
    ssns = detect_ssns("Valid: 123-45-6789, Invalid: 000-45-6789")
    print(f"SSNs: {[s.text for s in ssns]}")

    # Credit card detection (with Luhn validation)
    cards = detect_credit_cards("Card: 4532-0151-1283-0366")
    print(f"Credit Cards: {[c.text for c in cards]}")

    # ================================================================
    # Example 3: Secrets Detection (Three-Tier)
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 3: Secrets Detection")
    print("-" * 70)

    secrets_text = """
    OPENAI_KEY = "sk-1234567890abcdef1234567890abcdef"
    AWS_ACCESS = "AKIAIOSFODNN7EXAMPLE"
    GITHUB_PAT = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234"
    """

    secrets = detect_secrets_regex(secrets_text)
    print(f"Found {len(secrets)} secrets:")
    for s in secrets:
        print(f"  - {s.text[:35]}...")

    # No false positives on normal text
    normal_text = "The quick brown fox jumps over the lazy dog"
    no_secrets = detect_secrets_regex(normal_text)
    print(f"\nNormal text secrets: {len(no_secrets)} (should be 0)")

    # ================================================================
    # Example 4: Full Detection Pipeline
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 4: Full Detection Pipeline")
    print("-" * 70)

    text = """
    Dear Customer,
    
    Your account (john@example.com) has been verified.
    Please call 555-123-4567 if you have questions.
    Your API key: sk-1234567890abcdef1234567890abcdef
    
    Warning: Do not share your SSN (123-45-6789)!
    """

    result = detect_all_regex(
        text,
        detect_pii=True,
        detect_secrets=True,
        detect_harmful=True,
        redaction_strategy=RedactionStrategy.TOKEN,
    )

    print(f"Mode: {result.mode}")
    print(f"Detections: {len(result.detections)}")
    print(f"Harmful: {result.harmful}")
    
    print("\nBy type:")
    by_type = {}
    for d in result.detections:
        by_type.setdefault(d.type, []).append(d)
    for t, dets in sorted(by_type.items()):
        print(f"  {t}: {[d.text for d in dets]}")

    # ================================================================
    # Example 5: Performance Test
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 5: Performance (Regex is FAST)")
    print("-" * 70)

    import time

    test_text = "Email: test@example.com, Phone: 555-123-4567"
    iterations = 1000

    start = time.time()
    for _ in range(iterations):
        detect(test_text, mode="regex")
    elapsed = time.time() - start

    print(f"Processed {iterations} texts in {elapsed:.3f}s")
    print(f"Average: {elapsed/iterations*1000:.2f}ms per text")
    print(f"Throughput: {iterations/elapsed:.0f} texts/second")

    # ================================================================
    # Example 6: Utility Functions
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 6: Utility Functions")
    print("-" * 70)

    # Luhn check for credit cards
    valid_cc = "4532015112830366"
    invalid_cc = "1234567890123456"
    print(f"Luhn check {valid_cc}: {luhn_check(valid_cc)}")
    print(f"Luhn check {invalid_cc}: {luhn_check(invalid_cc)}")

    # Direct secrets finder
    secrets = find_secrets("key=sk-abc123def456abc123def456abc123def456")
    print(f"find_secrets: {[s['span'][:30] for s in secrets]}")

    print("\n" + "=" * 70)
    print("Regex mode examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
