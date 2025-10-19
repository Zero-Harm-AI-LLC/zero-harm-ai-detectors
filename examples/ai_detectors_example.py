
# ==================== Example Usage ====================
from zero_harm_ai_detectors import ZeroHarmPipeline

if __name__ == "__main__":
    # Example 1: Simple usage
    print("=" * 60)
    print("Example 1: Simple Detection")
    print("=" * 60)
    
    pipeline = ZeroHarmPipeline()
    text = "Contact John Smith at john.smith@email.com or call 555-123-4567. API key: sk-abc123def456."
    
    result = pipeline.detect(text)
    print(f"Original: {result.original_text}")
    print(f"Redacted: {result.redacted_text}")
    print(f"Found {len(result.detections)} items:")
    for det in result.detections:
        print(f"  - {det.type}: '{det.text}' (confidence: {det.confidence:.2f})")
    
    print("\n" + "=" * 60)
    print("Example 2: Harmful Content")
    print("=" * 60)
    
    harmful_text = "I hate you and want to hurt you"
    result = pipeline.detect(harmful_text)
    print(f"Text: {harmful_text}")
    print(f"Harmful: {result.harmful}")
    print(f"Severity: {result.severity}")
    if result.harmful_scores:
        print(f"Top scores:")
        for label, score in sorted(result.harmful_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {label}: {score:.3f}")
