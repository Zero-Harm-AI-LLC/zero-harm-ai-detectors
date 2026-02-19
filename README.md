# Zero Harm AI Detectors

**Privacy & content safety detection library with two modes: regex (fast) and AI (accurate).**

[![PyPI version](https://badge.fury.io/py/zero-harm-ai-detectors.svg)](https://pypi.org/project/zero-harm-ai-detectors/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Two Modes, One API

| Feature | mode='regex' | mode='ai' |
|---------|--------------|-----------|
| **Speed** | ⚡ 1-5ms | 🐢 50-200ms |
| **Email, Phone, SSN** | ✅ 99% accuracy | ✅ 99% (same) |
| **Credit Card, Secrets** | ✅ 95% accuracy | ✅ 95% (same) |
| **Person Names** | ⚠️ 30-40% accuracy | ✅ **85-95% accuracy** |
| **Locations** | ❌ Not available | ✅ **80-90% accuracy** |
| **Organizations** | ❌ Not available | ✅ **75-85% accuracy** |
| **Harmful Content** | ⚠️ Keywords only | ✅ **Contextual AI** |
| **Dependencies** | regex only | transformers, torch |

## 📦 Installation

```bash
# Regex mode only (fast, no ML dependencies)
pip install zero_harm_ai_detectors

# With AI mode (requires ~2GB for models)
pip install 'zero_harm_ai_detectors[ai]'
```

## 🚀 Quick Start

### Unified API

```python
from zero_harm_ai_detectors import detect

# Regex mode (default) - fast, great for structured data
result = detect("Email: john@example.com, SSN: 123-45-6789")
print(result.redacted_text)
# → Email: [REDACTED_EMAIL], SSN: [REDACTED_SSN]

# AI mode - better for names, locations, organizations
result = detect("Contact John Smith at Microsoft in NYC", mode="ai")
print(result.detections)
# → [Detection(PERSON), Detection(ORGANIZATION), Detection(LOCATION)]
```

### Same Result Format for Both Modes

```python
result = detect(text, mode="regex")  # or mode="ai"

# Always returns DetectionResult with:
result.original_text      # Original input
result.redacted_text      # Redacted output
result.detections         # List[Detection]
result.mode              # "regex" or "ai"
result.harmful           # bool
result.severity          # "none", "low", "medium", "high"
result.harmful_scores    # Dict[str, float]

# Convert to dict
result.to_dict()         # Full format with metadata
result.to_legacy_dict()  # v0.1.x compatible format
```

## 📋 Detection Types

### PII Detection

| Type | Regex Accuracy | AI Accuracy | Example |
|------|---------------|-------------|---------|
| EMAIL | 99% | 99% | `john@example.com` |
| PHONE | 95% | 95% | `555-123-4567` |
| SSN | 98% | 98% | `123-45-6789` |
| CREDIT_CARD | 99% (Luhn) | 99% | `4532-0151-1283-0366` |
| BANK_ACCOUNT | 90% | 90% | `Account: 12345678` |
| DOB | 85% | 85% | `01/15/1990` |
| ADDRESS | 85% | 85% | `123 Main Street` |
| PERSON | **30-40%** | **85-95%** | `John Smith` |
| LOCATION | ❌ | **80-90%** | `New York City` |
| ORGANIZATION | ❌ | **75-85%** | `Microsoft` |

### Secrets Detection (Three-Tier)

```python
result = detect(code_snippet, detect_secrets=True)

# Detects:
# - Tier 1: Structured prefixes (sk-, ghp_, AKIA, etc.)
# - Tier 2: AWS secrets (context + entropy)
# - Tier 3: Generic secrets (context + entropy)
```

Supported secret types:
- OpenAI API keys (`sk-...`)
- GitHub tokens (`ghp_`, `gho_`, etc.)
- AWS keys (`AKIA...`, secret keys)
- Stripe keys (`sk_live_`, `sk_test_`)
- Google API keys (`AIza...`)
- Slack tokens (`xox...`)
- And many more...

### Harmful Content Detection

```python
result = detect(text, detect_harmful=True)

print(result.harmful)   # True/False
print(result.severity)  # "none", "low", "medium", "high"
print(result.harmful_scores)  # {"insult": 0.8, "threat": 0.2, ...}
```

## 🔧 Advanced Usage

### Selective Detection

```python
# Only detect PII (skip secrets and harmful)
result = detect(text, detect_pii=True, detect_secrets=False, detect_harmful=False)

# Only detect secrets
result = detect(text, detect_pii=False, detect_secrets=True, detect_harmful=False)
```

### Redaction Strategies

```python
from zero_harm_ai_detectors import detect

text = "Email: john@example.com"

# Token (default): [REDACTED_EMAIL]
detect(text, redaction_strategy="token")

# Mask all: ****************
detect(text, redaction_strategy="mask_all")

# Mask last 4: ************.com
detect(text, redaction_strategy="mask_last4")

# Hash: [HASH:a1b2c3d4e5f6]
detect(text, redaction_strategy="hash")
```

### Custom AI Configuration

```python
from zero_harm_ai_detectors import detect, AIConfig

config = AIConfig(
    ner_model="dslim/bert-base-NER",
    ner_threshold=0.8,      # Higher = fewer false positives
    harmful_threshold=0.6,
    device="cuda",          # Use GPU
)

result = detect(text, mode="ai", ai_config=config)
```

### Individual Detectors

```python
from zero_harm_ai_detectors import (
    detect_emails,
    detect_phones,
    detect_ssns,
    detect_credit_cards,
    detect_secrets_regex,
)

# Fine-grained control
emails = detect_emails("Contact alice@test.com")
phones = detect_phones("Call 555-123-4567")
secrets = detect_secrets_regex(code)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    detect(text, mode=...)                    │
├─────────────────────────────┬───────────────────────────────┤
│      mode='regex'           │         mode='ai'             │
│  ┌───────────────────────┐  │  ┌─────────────────────────┐  │
│  │   Regex Patterns      │  │  │  Regex (structured)     │  │
│  │   - Email, Phone      │  │  │  - Email, Phone, SSN    │  │
│  │   - SSN, Credit Card  │  │  │  - Credit Card, Secrets │  │
│  │   - Secrets           │  │  ├─────────────────────────┤  │
│  │   - Person (~30%)     │  │  │  AI NER (transformers)  │  │
│  │   - Harmful (keyword) │  │  │  - Person (~90%)        │  │
│  └───────────────────────┘  │  │  - Location (~85%)      │  │
│                             │  │  - Organization (~80%)  │  │
│                             │  ├─────────────────────────┤  │
│                             │  │  AI Harmful (context)   │  │
│                             │  └─────────────────────────┘  │
├─────────────────────────────┴───────────────────────────────┤
│                     DetectionResult                          │
│  (identical structure for both modes)                        │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 Integration Examples

### GitHub App PR Reviewer

```python
from zero_harm_ai_detectors import detect, AI_AVAILABLE

def review_pr(diff_text: str, is_paid_user: bool) -> dict:
    mode = "ai" if (is_paid_user and AI_AVAILABLE) else "regex"
    
    result = detect(
        diff_text,
        mode=mode,
        detect_pii=True,
        detect_secrets=True,
        detect_harmful=False,
    )
    
    return {
        "has_issues": len(result.detections) > 0,
        "should_block": any(d.type in {"API_KEY", "SECRET"} for d in result.detections),
        "detections": result.to_dict()["detections"],
    }
```

### FastAPI Endpoint

```python
from fastapi import FastAPI
from zero_harm_ai_detectors import detect

app = FastAPI()

@app.post("/scan")
def scan_text(text: str, mode: str = "regex"):
    result = detect(text, mode=mode)
    return result.to_dict()
```

## 📊 Performance Benchmarks

| Operation | Regex Mode | AI Mode |
|-----------|-----------|---------|
| Short text (50 chars) | 1-2ms | 50-100ms |
| Medium text (500 chars) | 2-3ms | 100-150ms |
| Long text (5000 chars) | 3-5ms | 150-200ms |
| Throughput | ~500/sec | ~5-10/sec |

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=zero_harm_ai_detectors

# Skip AI tests (if dependencies not installed)
pytest tests/test_core_and_regex.py -v
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔗 Links

- **Documentation**: https://zeroharm.ai/docs
- **PyPI**: https://pypi.org/project/zero-harm-ai-detectors/
- **GitHub**: https://github.com/zeroharm-ai/zero-harm-ai-detectors
- **Issues**: https://github.com/zeroharm-ai/zero-harm-ai-detectors/issues
