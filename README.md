# Zero Harm AI Detectors

**Privacy & content safety detection with freemium tiers for GitHub App PR reviewers.**

## 🎯 Two Tiers, One API

| Feature | FREE Tier (Regex) | PAID Tier (AI-Enhanced) |
|---------|-------------------|-------------------------|
| **Email, Phone, SSN** | ✅ 99% accuracy | ✅ 99% accuracy (same) |
| **Credit Card, Secrets** | ✅ 95% accuracy | ✅ 95% accuracy (same) |
| **Person Names** | ⚠️ 30-40% accuracy | ✅ **85-95% accuracy** |
| **Locations** | ❌ Not available | ✅ **80-90% accuracy** |
| **Organizations** | ❌ Not available | ✅ **75-85% accuracy** |
| **Harmful Content** | ⚠️ Basic keywords | ✅ **Contextual AI** |
| **Speed** | ⚡ 1-5ms | 🐢 50-200ms |
| **Dependencies** | None (regex only) | transformers, torch |

## 📦 Installation

```bash
# Free tier (regex only)
pip install zero_harm_ai_detectors

# Paid tier (with AI)
pip install zero_harm_ai_detectors[ai]
```

## 🚀 Quick Start

### Unified API (Recommended)

```python
from zero_harm_ai_detectors import detect

# Free tier (default) - fast, good for structured data
result = detect("Email: john@example.com, SSN: 123-45-6789")
print(result.redacted_text)
# → Email: [REDACTED_EMAIL], SSN: [REDACTED_SSN]

# Paid tier - better for names, locations, harmful content
result = detect("Contact John Smith at Microsoft in NYC", tier="paid")
print(result.detections)
# → [Detection(PERSON), Detection(ORGANIZATION), Detection(LOCATION)]
```

### Same Result Format for Both Tiers

```python
result = detect(text, tier="free")  # or tier="paid"

# Always returns DetectionResult with:
result.original_text      # Original input
result.redacted_text      # Redacted output
result.detections         # List[Detection]
result.tier              # "free" or "paid"
result.harmful           # bool
result.severity          # "none", "low", "medium", "high"
result.harmful_scores    # Dict[str, float]

# Convert to dict
result.to_dict()         # Full format with metadata
result.to_legacy_dict()  # v0.1.x compatible format
```

## 🔍 What Gets Detected

### Structured PII (Regex - Both Tiers)
- ✉️ **Email**: 99%+ accuracy
- 📞 **Phone**: 95%+ accuracy  
- 🆔 **SSN**: 95%+ accuracy
- 💳 **Credit Card**: 90%+ (with Luhn validation)
- 🏦 **Bank Account**: Context-dependent
- 📅 **DOB**: Multiple formats
- 🚗 **Driver's License**: US state formats
- 🏥 **Medical Record Number**: Context-dependent
- 📍 **Address**: Street addresses, P.O. boxes

### Secrets (Regex - Both Tiers)
- 🔑 **API Keys**: OpenAI, AWS, Google, etc.
- 🎫 **Tokens**: GitHub, Slack, Stripe, JWT
- 🔐 **Generic Secrets**: Context + entropy detection

### AI-Enhanced (Paid Tier Only)
- 👤 **Person Names**: 85-95% accuracy (vs 30% regex)
- 📍 **Locations**: Cities, states, countries
- 🏢 **Organizations**: Companies, institutions
- ☠️ **Harmful Content**: Contextual AI analysis

## 📋 API Reference

### Main Function

```python
detect(
    text: str,
    tier: str = "free",           # "free" or "paid"
    detect_pii: bool = True,
    detect_secrets: bool = True,
    detect_harmful: bool = True,
    redaction_strategy: str = "token",  # "token", "mask_all", "mask_last4", "hash"
) -> DetectionResult
```

### Legacy Functions (v0.1.x compatible)

```python
# These still work for backward compatibility
detect_pii(text, tier="free")     # Returns Dict[str, List[Dict]]
detect_secrets(text)               # Returns Dict[str, List[Dict]]
detect_harmful(text, tier="free") # Returns Dict with harmful info
```

### Detection Result

```python
@dataclass
class DetectionResult:
    original_text: str
    redacted_text: str
    detections: List[Detection]
    tier: str
    harmful: bool
    harmful_scores: Dict[str, float]
    severity: str
    
    def to_dict(self) -> Dict
    def to_legacy_dict(self) -> Dict
    def get_pii(self) -> List[Detection]
    def get_secrets(self) -> List[Detection]

@dataclass
class Detection:
    type: str       # e.g., "EMAIL", "PERSON", "API_KEY"
    text: str       # The matched text
    start: int      # Start position
    end: int        # End position
    confidence: float
    metadata: Dict
```

## 🎨 Redaction Strategies

```python
from zero_harm_ai_detectors import RedactionStrategy

# TOKEN (default): [REDACTED_EMAIL]
result = detect(text, redaction_strategy="token")

# MASK_ALL: ****************
result = detect(text, redaction_strategy="mask_all")

# MASK_LAST4: ************.com
result = detect(text, redaction_strategy="mask_last4")

# HASH: 8d969eef6ecad3c29a3a...
result = detect(text, redaction_strategy="hash")
```

## 🔧 Advanced Usage

### Selective Detection

```python
# Only detect PII (skip secrets and harmful)
result = detect(text, detect_secrets=False, detect_harmful=False)

# Only detect secrets
result = detect(text, detect_pii=False, detect_harmful=False)
```

### Custom AI Configuration

```python
from zero_harm_ai_detectors import AIConfig, detect

config = AIConfig(
    ner_model="dslim/bert-base-NER",
    ner_threshold=0.8,  # Higher confidence
    harmful_model="unitary/multilingual-toxic-xlm-roberta",
    harmful_threshold=0.6,
    device="cuda",  # Use GPU
)

result = detect(text, tier="paid", ai_config=config)
```

### Check AI Availability

```python
from zero_harm_ai_detectors import AI_AVAILABLE, check_ai_available

if AI_AVAILABLE:
    result = detect(text, tier="paid")
else:
    result = detect(text, tier="free")  # Fallback
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    detect(text, tier=...)                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│     FREE TIER           │     │     PAID TIER           │
│  (regex_detectors.py)   │     │   (ai_detectors.py)     │
├─────────────────────────┤     ├─────────────────────────┤
│ • Email (regex)         │     │ • Email (regex) ✓       │
│ • Phone (regex)         │     │ • Phone (regex) ✓       │
│ • SSN (regex)           │     │ • SSN (regex) ✓         │
│ • Credit Card (regex)   │     │ • Credit Card (regex) ✓ │
│ • Secrets (regex)       │     │ • Secrets (regex) ✓     │
│ • Person (weak regex)   │     │ • Person (AI NER) ⭐    │
│ • Harmful (keywords)    │     │ • Location (AI NER) ⭐  │
│                         │     │ • Org (AI NER) ⭐       │
│                         │     │ • Harmful (AI) ⭐       │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DetectionResult                          │
│  (Same format for both tiers!)                              │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Benchmark

| Content Type | Free (Regex) | Paid (AI) | Improvement |
|--------------|--------------|-----------|-------------|
| Email | 99% | 99% | - |
| Phone | 95% | 95% | - |
| SSN | 95% | 95% | - |
| Person Names | 35% | 90% | **+55%** |
| Locations | 0% | 85% | **+85%** |
| Organizations | 0% | 80% | **+80%** |

## 🔄 Migration from v0.2.x

The API is backward compatible. Old code continues to work:

```python
# Old code (still works)
from zero_harm_ai_detectors import detect_pii, detect_secrets

pii = detect_pii(text)
secrets = detect_secrets(text)

# New recommended way
from zero_harm_ai_detectors import detect

result = detect(text, tier="free")  # or "paid"
```

## 🤝 GitHub App Integration

```python
from zero_harm_ai_detectors import detect

def review_pr_diff(diff_text: str, is_paid_user: bool) -> dict:
    """Review PR diff for sensitive content."""
    tier = "paid" if is_paid_user else "free"
    
    result = detect(
        diff_text,
        tier=tier,
        detect_pii=True,
        detect_secrets=True,
        detect_harmful=False,  # Usually not needed for code
    )
    
    return {
        "has_issues": len(result.detections) > 0,
        "detections": result.to_dict()["detections"],
        "tier_used": result.tier,
    }
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- The regex community for pattern libraries

---

**Made with ❤️ by [Zero Harm AI LLC](https://zeroharmai.com)**
