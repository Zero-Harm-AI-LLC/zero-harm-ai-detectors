#!/usr/bin/env python3
"""
Zero Harm AI Detectors - AI Mode Examples

AI mode (mode='ai') provides:
- Enhanced person name detection: 30% regex → 85-95% AI
- Location detection: Not available in regex → 80-90% AI
- Organization detection: Not available in regex → 75-85% AI
- Better harmful content analysis with contextual understanding

Structured data (email, phone, SSN, secrets) still uses regex (95%+ accuracy).

Requirements:
    pip install 'zero_harm_ai_detectors[ai]'

File: examples/ai_mode_example.py
"""
import sys

from zero_harm_ai_detectors import (
    detect,
    AI_AVAILABLE,
    DetectionResult,
)


def main():
    print("=" * 70)
    print("Zero Harm AI Detectors - AI Mode Examples")
    print("=" * 70)

    if not AI_AVAILABLE:
        print("\n⚠️  AI mode is NOT available!")
        print("Install with: pip install 'zero_harm_ai_detectors[ai]'")
        print("\nShowing comparison with regex mode instead...\n")

    # ================================================================
    # Example 1: Person Name Detection Comparison
    # ================================================================
    print("-" * 70)
    print("Example 1: Person Name Detection (Regex vs AI)")
    print("-" * 70)

    text = "Please contact John Smith or Sarah Johnson for assistance"

    # Regex mode
    result_regex = detect(text, mode="regex")
    regex_persons = [d for d in result_regex.detections if d.type == "PERSON"]
    print(f"Regex mode: {len(regex_persons)} persons found")
    for p in regex_persons:
        print(f"  - '{p.text}' (confidence: {p.confidence:.0%})")

    # AI mode
    if AI_AVAILABLE:
        result_ai = detect(text, mode="ai")
        ai_persons = [d for d in result_ai.detections if d.type == "PERSON"]
        print(f"\nAI mode: {len(ai_persons)} persons found")
        for p in ai_persons:
            print(f"  - '{p.text}' (confidence: {p.confidence:.0%})")
    else:
        print("\nAI mode: [Not available - would typically find both names]")

    # ================================================================
    # Example 2: Location Detection (AI Only)
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 2: Location Detection (AI only)")
    print("-" * 70)

    text = "Our offices are in New York City, London, and Tokyo"

    result_regex = detect(text, mode="regex")
    regex_locs = [d for d in result_regex.detections if d.type == "LOCATION"]
    print(f"Regex mode: {len(regex_locs)} locations (regex can't detect these)")

    if AI_AVAILABLE:
        result_ai = detect(text, mode="ai")
        ai_locs = [d for d in result_ai.detections if d.type == "LOCATION"]
        print(f"AI mode: {len(ai_locs)} locations found")
        for loc in ai_locs:
            print(f"  - '{loc.text}'")
    else:
        print("AI mode: [Would detect 'New York City', 'London', 'Tokyo']")

    # ================================================================
    # Example 3: Organization Detection (AI Only)
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 3: Organization Detection (AI only)")
    print("-" * 70)

    text = "I work at Microsoft and previously worked at Google"

    result_regex = detect(text, mode="regex")
    regex_orgs = [d for d in result_regex.detections if d.type == "ORGANIZATION"]
    print(f"Regex mode: {len(regex_orgs)} organizations (regex can't detect these)")

    if AI_AVAILABLE:
        result_ai = detect(text, mode="ai")
        ai_orgs = [d for d in result_ai.detections if d.type == "ORGANIZATION"]
        print(f"AI mode: {len(ai_orgs)} organizations found")
        for org in ai_orgs:
            print(f"  - '{org.text}'")
    else:
        print("AI mode: [Would detect 'Microsoft', 'Google']")

    # ================================================================
    # Example 4: Mixed Content (AI + Regex Together)
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 4: Mixed Content (AI enhances, regex handles structured)")
    print("-" * 70)

    text = """
    Contact: John Smith
    Email: john.smith@example.com
    Company: Acme Corporation
    Location: San Francisco
    """

    if AI_AVAILABLE:
        result = detect(text, mode="ai")
        
        print("AI-detected (NER model):")
        for d in result.detections:
            if d.metadata.get("method") == "ai_ner":
                print(f"  - {d.type}: '{d.text}'")
        
        print("\nRegex-detected (patterns):")
        for d in result.detections:
            if d.metadata.get("method") == "regex":
                print(f"  - {d.type}: '{d.text}'")
    else:
        result = detect(text, mode="regex")
        print(f"Regex mode detected: {[d.type for d in result.detections]}")

    # ================================================================
    # Example 5: Custom AI Configuration
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 5: Custom AI Configuration")
    print("-" * 70)

    if AI_AVAILABLE:
        from zero_harm_ai_detectors import AIConfig
        
        config = AIConfig(
            ner_model="dslim/bert-base-NER",
            ner_threshold=0.8,  # Higher confidence threshold
            harmful_threshold=0.6,
            device="cpu",  # Use "cuda" for GPU
        )
        
        print(f"Custom config:")
        print(f"  NER model: {config.ner_model}")
        print(f"  NER threshold: {config.ner_threshold}")
        print(f"  Device: {config.device}")
        
        result = detect(
            "Contact Dr. Jane Wilson at Stanford University",
            mode="ai",
            ai_config=config,
        )
        print(f"\nDetections with custom config:")
        for d in result.detections:
            print(f"  - {d.type}: '{d.text}' ({d.confidence:.0%})")
    else:
        print("AIConfig example:")
        print("  config = AIConfig(ner_threshold=0.8, device='cuda')")
        print("  result = detect(text, mode='ai', ai_config=config)")

    # ================================================================
    # Example 6: Harmful Content Detection
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 6: Harmful Content Detection")
    print("-" * 70)

    test_cases = [
        "This is a wonderful day!",
        "I hate you so much!",
        "You're going to regret this",
    ]

    for text in test_cases:
        result_regex = detect(text, mode="regex", detect_pii=False, detect_secrets=False)
        
        if AI_AVAILABLE:
            result_ai = detect(text, mode="ai", detect_pii=False, detect_secrets=False)
            print(f"'{text[:30]}...'")
            print(f"  Regex: harmful={result_regex.harmful}, severity={result_regex.severity}")
            print(f"  AI:    harmful={result_ai.harmful}, severity={result_ai.severity}")
        else:
            print(f"'{text[:30]}...'")
            print(f"  Regex: harmful={result_regex.harmful}, severity={result_regex.severity}")

    # ================================================================
    # Example 7: Performance Comparison
    # ================================================================
    print("\n" + "-" * 70)
    print("Example 7: Performance Comparison")
    print("-" * 70)

    if AI_AVAILABLE:
        import time
        
        text = "Contact John Smith at Microsoft in Seattle"
        iterations = 10
        
        # Regex mode
        start = time.time()
        for _ in range(iterations):
            detect(text, mode="regex")
        regex_time = (time.time() - start) / iterations * 1000
        
        # AI mode
        start = time.time()
        for _ in range(iterations):
            detect(text, mode="ai")
        ai_time = (time.time() - start) / iterations * 1000
        
        print(f"Average time per detection:")
        print(f"  Regex: {regex_time:.1f}ms")
        print(f"  AI:    {ai_time:.1f}ms")
        print(f"  Ratio: {ai_time/regex_time:.0f}x slower (but more accurate)")
    else:
        print("Typical performance:")
        print("  Regex: 1-5ms per text")
        print("  AI:    50-200ms per text")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "-" * 70)
    print("Summary: When to use each mode")
    print("-" * 70)
    print("""
    Use mode='regex' when:
    - Speed is critical (1-5ms)
    - Only detecting structured data (email, phone, SSN, secrets)
    - Running on resource-constrained systems
    - High-volume batch processing
    
    Use mode='ai' when:
    - Accuracy for names/locations/orgs is important
    - Processing text with unstructured PII
    - Better harmful content analysis needed
    - Latency of 50-200ms is acceptable
    """)

    print("=" * 70)
    print("AI mode examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
