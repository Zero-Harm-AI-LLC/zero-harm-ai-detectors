# Migration Guide: Upgrading to AI-Based Detection

Complete guide for migrating from regex-based v0.1.x to AI-powered v0.2.0.

## Overview

### What's Changing

| Aspect | v0.1.x (Regex) | v0.2.0 (AI) |
|--------|----------------|-------------|
| Person names | 30-40% accuracy | 85-95% accuracy |
| Speed | 1-5ms | 50-200ms |
| New types | 10 types | 13 types (adds locations, orgs) |
| API | Same | 100% backward compatible |

### Migration Paths

**Path 1: No Code Changes** (Easiest)
- Install new version
- Code works automatically with AI
- 5 minutes

**Path 2: Gradual Migration** (Recommended)
- Test AI vs regex side-by-side
- Migrate when confident
- 1-2 hours

**Path 3: Full Optimization** (Best performance)
- Use new `ZeroHarmPipeline` class
- Custom configuration
- Half day

## Path 1: Zero-Code Migration (5 minutes)

### Step 1: Install New Version

```bash
# Upgrade library
pip install --upgrade zero_harm_ai_detectors transformers torch
```

### Step 2: Test

```bash
# Run your existing tests
pytest

# Your code now uses AI automatically!
```

### Step 3: Done!

Your existing code like this:

```python
from zero_harm_ai_detectors import detect_pii, detect_secrets

pii = detect_pii("Contact John Smith")
# Now automatically uses AI! 🎉
```

works unchanged but with better accuracy.

## Path 2: Gradual Migration (1-2 hours)

### Step 1: Install and Test in Parallel

```bash
pip install --upgrade zero_harm_ai_detectors transformers torch
```

### Step 2: Compare AI vs Regex

```python
from zero_harm_ai_detectors import detect_pii

text = "Contact John Smith at Microsoft in New York"

# New AI detection
pii_ai = detect_pii(text, use_ai=True)
print(f"AI found: {list(pii_ai.keys())}")
# Output: ['PERSON', 'ORGANIZATION', 'LOCATION']

# Old regex detection
pii_regex = detect_pii(text, use_ai=False)
print(f"Regex found: {list(pii_regex.keys())}")
# Output: [] (missed everything!)
```

### Step 3: Run Side-by-Side Tests

```python
test_cases = [
    "Contact Alice Johnson",
    "Email: bob@example.com",
    "Located in San Francisco",
    "Works at Google"
]

for text in test_cases:
    ai_result = detect_pii(text, use_ai=True)
    regex_result = detect_pii(text, use_ai=False)
    
    print(f"Text: {text}")
    print(f"  AI: {list(ai_result.keys())}")
    print(f"  Regex: {list(regex_result.keys())}")
```

### Step 4: Deploy AI Version

Once confident, AI becomes default:

```python
# No need to specify use_ai=True, it's the default
pii = detect_pii(text)
```

## Path 3: Full Optimization (Half day)

### Step 1: Replace with New API

**Old Code:**
```python
from zero_harm_ai_detectors import detect_pii, detect_secrets, redact_text

def process(text):
    detected = {}
    
    pii = detect_pii(text)
    if pii:
        detected.update(pii)
    
    secrets = detect_secrets(text)
    if secrets:
        detected.update(secrets)
    
    if detected:
        redacted = redact_text(text, detected)
    else:
        redacted = text
    
    return redacted, detected
```

**New Code:**
```python
from zero_harm_ai_detectors import detect_all_threats

def process(text):
    result = detect_all_threats(text)
    return result['redacted'], result['detections']
```

### Step 2: Use Full Pipeline

**Even Better:**
```python
from zero_harm_ai_detectors import ZeroHarmPipeline, RedactionStrategy

# Load once at app startup
pipeline = ZeroHarmPipeline()

def process(text):
    result = pipeline.detect(
        text,
        redaction_strategy=RedactionStrategy.TOKEN
    )
    
    # Convert to your format
    detected = {}
    for det in result.detections:
        if det.type not in detected:
            detected[det.type] = []
        detected[det.type].append({
            "span": det.text,
            "start": det.start,
            "end": det.end,
            "confidence": det.confidence
        })
    
    return result.redacted_text, detected
```

### Step 3: Add Custom Configuration

```python
from zero_harm_ai_detectors import PipelineConfig

config = PipelineConfig(
    pii_threshold=0.8,  # Higher confidence
    harmful_threshold_per_label=0.6,
    device="cuda"  # Use GPU if available
)

pipeline = ZeroHarmPipeline(config)
```

### Step 4: Optimize for Your Use Case

```python
# Skip harmful detection if not needed
result = pipeline.detect(
    text,
    detect_pii=True,
    detect_secrets=True,
    detect_harmful=False  # Saves ~100ms
)

# Filter by confidence
high_conf = [
    d for d in result.detections
    if d.confidence >= 0.9
]
```

## Backend Integration

### Update proxy.py

**Old proxy.py:**
```python
from zero_harm_ai_detectors import detect_pii, detect_secrets

def process_prompt(prompt):
    detected = {}
    
    pii = detect_pii(prompt)
    if pii:
        detected.update(pii)
    
    secrets = detect_secrets(prompt)
    if secrets:
        detected.update(secrets)
    
    # ... redaction code ...
    
    return redacted, detected
```

**New proxy.py:**
```python
from zero_harm_ai_detectors import ZeroHarmPipeline, RedactionStrategy

# Load once
pipeline = ZeroHarmPipeline()

def process_prompt(prompt):
    result = pipeline.detect(
        prompt,
        redaction_strategy=RedactionStrategy.TOKEN
    )
    
    # Convert to backend format
    detected = {}
    for det in result.detections:
        if det.type != "HARMFUL_CONTENT":
            if det.type not in detected:
                detected[det.type] = []
            detected[det.type].append({
                "span": det.text,
                "start": det.start,
                "end": det.end
            })
    
    return result.redacted_text, detected
```

### Update requirements.txt

```txt
# Add these lines
zero_harm_ai_detectors>=0.2.0
transformers>=4.30.0
torch>=2.0.0
```

## Testing Migration

### Test Script

```python
# test_migration.py
from zero_harm_ai_detectors import detect_all_threats

def test_basic():
    """Test basic functionality"""
    result = detect_all_threats("Contact john@example.com")
    
    assert 'redacted' in result
    assert 'detections' in result
    assert len(result['detections']) > 0
    print("✅ Basic test passed")

def test_person_names():
    """Test improved person name detection"""
    result = detect_all_threats("Meet with Sarah Johnson")
    
    detections = result['detections']
    assert 'PERSON' in detections, "Person name not detected!"
    print("✅ Person name test passed")

def test_locations():
    """Test new location detection"""
    result = detect_all_threats("Office in New York City")
    
    detections = result['detections']
    assert 'LOCATION' in detections, "Location not detected!"
    print("✅ Location test passed")

def test_organizations():
    """Test new organization detection"""
    result = detect_all_threats("Works at Microsoft")
    
    detections = result['detections']
    assert 'ORGANIZATION' in detections, "Organization not detected!"
    print("✅ Organization test passed")

if __name__ == "__main__":
    test_basic()
    test_person_names()
    test_locations()
    test_organizations()
    print("\n🎉 All migration tests passed!")
```

Run tests:
```bash
python test_migration.py
```

## Troubleshooting

### Issue: ImportError

**Problem:**
```python
ImportError: cannot import name 'ZeroHarmPipeline'
```

**Solution:**
```bash
pip uninstall zero_harm_ai_detectors
pip install zero_harm_ai_detectors[ai]>=0.2.0
```

### Issue: Slow Performance

**Problem:** API responses take >1 second

**Solution:**
```python
# Skip harmful detection if not needed
result = pipeline.detect(text, detect_harmful=False)

# Or use faster model
config = PipelineConfig(
    pii_model="dslim/bert-base-NER"  # Smaller, faster
)
```

### Issue: Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Use CPU instead
config = PipelineConfig(device="cpu")
pipeline = ZeroHarmPipeline(config)
```

### Issue: Models Not Loading

**Problem:**
```
OSError: Can't load model
```

**Solution:**
```bash
# Clear cache and reinstall
rm -rf ~/.cache/huggingface
pip install --upgrade transformers torch
```

## Performance Comparison

### Before (v0.1.x)

```python
# Average: 2ms per detection
from zero_harm_ai_detectors import detect_pii

pii = detect_pii("Contact John Smith")
# Detections: {} (missed!)
```

### After (v0.2.0)

```python
# Average: 150ms per detection
from zero_harm_ai_detectors import detect_pii

pii = detect_pii("Contact John Smith")
# Detections: {'PERSON': [{'span': 'John Smith', ...}]} ✅
```

**Trade-off:** 75x slower but finds 3x more detections!

## Checklist

### Pre-Migration
- [ ] Backup current code
- [ ] Document current detection rates
- [ ] Test current system

### Migration
- [ ] Install new version with AI dependencies
- [ ] Run existing tests
- [ ] Compare AI vs regex results
- [ ] Update documentation

### Post-Migration
- [ ] Monitor performance metrics
- [ ] Check accuracy improvements
- [ ] Optimize configuration
- [ ] Update team documentation

## Support

- **Email**: info@zeroharmai.com
- **GitHub Issues**: https://github.com/Zero-Harm-AI-LLC/zero-harm-ai-detectors/issues
- **Documentation**: See README.md

---

**Summary:** The migration is smooth and backward compatible. Your code works unchanged with AI detection automatically enabled!
