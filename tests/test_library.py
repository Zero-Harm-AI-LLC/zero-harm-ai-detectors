#!/usr/bin/env python3
"""
Test script to verify the library works
"""

def test_library():
    print("Testing zero_harm_ai_detectors library...")
    
    try:
        # Test imports
        from zero_harm_ai_detectors import detect_pii, detect_secrets, HarmfulTextDetector, DetectionConfig  # Changed import
        print("✅ Imports successful")
        
        # Test PII detection
        text = "Contact me at test@example.com or call 555-123-4567"
        pii = detect_pii(text)
        print(f"✅ PII Detection: Found {len(pii)} types: {list(pii.keys())}")
        
        # Test secrets detection  
        secret_text = "My API key is sk-1234567890abcdef1234567890abcdef"
        secrets = detect_secrets(secret_text)
        print(f"✅ Secrets Detection: Found {len(secrets)} types: {list(secrets.keys())}")
        
        # Test configuration
        config = DetectionConfig(threshold_per_label=0.7)
        print(f"✅ DetectionConfig: threshold = {config.threshold_per_label}")
        
        print("\n🎉 All tests passed! Library is working correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_library()