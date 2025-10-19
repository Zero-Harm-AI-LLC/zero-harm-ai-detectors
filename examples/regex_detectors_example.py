
# ==================== Example Usage ====================
from zero_harm_ai_detectors import HarmfulTextDetector

if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Hello, how are you today?",  # Clean
        "You're such a stupid idiot!",  # Insult
        "I'm going to kill you!",  # Threat
        "I fucking hate you, you worthless piece of trash!",  # Toxic + Insult
        "This is the best day ever!",  # Clean
        "You disgusting racist pig!",  # Insult
    ]
    
    detector = HarmfulTextDetector()
    
    print("=" * 70)
    print("Harmful Content Detection Tests")
    print("=" * 70)
    
    for text in test_texts:
        result = detector.detect(text)
        
        print(f"\nText: {text}")
        if result:
            data = result["HARMFUL_CONTENT"][0]
            print(f"  ✗ HARMFUL - Severity: {data['severity']}")
            print(f"  Labels: {', '.join(data['labels'])}")
            print(f"  Scores: {data['scores']}")
        else:
            print("  ✓ Clean")
    
    print("\n" + "=" * 70)
    print("API Compatibility Test")
    print("=" * 70)
    
    # Show compatibility with detect_pii format
    from zero_harm_ai_detectors import detect_pii, detect_secrets, detect_harmful
    
    text = "Email john@example.com with API key sk-abc123. You stupid idiot!"
    
    pii = detect_pii(text)
    secrets = detect_secrets(text)
    harmful = detect_harmful(text)
    
    print(f"\nText: {text}\n")
    print(f"PII: {list(pii.keys())}")
    print(f"Secrets: {list(secrets.keys())}")
    print(f"Harmful: {list(harmful.keys())}")
    
    # All have same format!
    print("\nAll results have compatible format:")
    print(f"  PII[EMAIL]: {pii['EMAIL'][0] if 'EMAIL' in pii else 'N/A'}")
    print(f"  SECRETS[SECRETS]: {secrets['SECRETS'][0] if 'SECRETS' in secrets else 'N/A'}")
    print(f"  HARMFUL[HARMFUL_CONTENT]: severity={harmful['HARMFUL_CONTENT'][0]['severity']}, labels={harmful['HARMFUL_CONTENT'][0]['labels']}")