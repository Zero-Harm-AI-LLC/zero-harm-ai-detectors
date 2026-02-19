# Zero Harm AI Detectors

**Privacy & content safety detection with regex and AI modes.**

[![PyPI version](https://badge.fury.io/py/zero-harm-ai-detectors.svg)](https://pypi.org/project/zero-harm-ai-detectors/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Two Modes, One API

| Feature | `mode='regex'` | `mode='ai'` |
|---------|--------------|-----------|
| **Speed** | ⚡ 1–5ms | 🐢 50–200ms |
| **Email, Phone, SSN** | ✅ 95–99% | ✅ 95–99% |
| **Credit Card (Luhn)** | ✅ 99% | ✅ 99% |
| **Secrets / API Keys** | ✅ 95% | ✅ 95% |
| **Person Names** | ⚠️ 30–40% | ✅ 85–95% |
| **Locations** | ❌ | ✅ 80–90% |
| **Organisations** | ❌ | ✅ 75–85% |
| **Harmful Content** | ✅ Pattern-based | ✅ Contextual AI |
| **Extra dependencies** | None | `transformers`, `torch` |

## Installation

```bash
# Regex mode only (fast, no ML dependencies)
pip install zero_harm_ai_detectors

# With AI mode (~2 GB model download on first use)
pip install 'zero_harm_ai_detectors[ai]'
```

## Quick Start

```python
from zero_harm_ai_detectors import detect

# Regex mode (default) — fast, great for structured data
result = detect("Email: john@example.com, SSN: 123-45-6789")
print(result.redacted_text)
# → Email: [REDACTED_EMAIL], SSN: [REDACTED_SSN]

# AI mode — better for names, locations, organisations
result = detect("Contact John Smith at Microsoft in NYC", mode="ai")
print(result.detections)
# → [Detection(PERSON, ...), Detection(ORGANIZATION, ...), Detection(LOCATION, ...)]
```

## Detection Result

Both modes return an identical `DetectionResult`:

```python
result = detect(text, mode="regex")  # or mode="ai"

result.original_text   # str   — original input
result.redacted_text   # str   — sensitive content replaced
result.detections      # list  — List[Detection]
result.mode            # str   — "regex" or "ai"
result.harmful         # bool  — harmful content found
result.severity        # str   — "none" | "low" | "medium" | "high"
result.harmful_scores  # dict  — per-category scores

result.to_dict()       # full dict with all fields
result.get_pii()       # List[Detection] — PII only
result.get_secrets()   # List[Detection] — secrets only
```

## PII Detection

### Structured data (regex, 95–99% accuracy)

```python
from zero_harm_ai_detectors import (
    detect_emails,
    detect_phones,
    detect_ssns,
    detect_credit_cards,
    detect_bank_accounts,
    detect_dob,
    detect_addresses,
)

emails = detect_emails("Contact alice@test.com or bob@example.org")
# → [Detection(EMAIL, 'alice@test.com', ...), Detection(EMAIL, 'bob@example.org', ...)]

cards = detect_credit_cards("Card: 4532-0151-1283-0366")
# → [Detection(CREDIT_CARD, '4532-0151-1283-0366', confidence=0.99)]  # Luhn validated
```

### Names, locations, organisations (AI mode only)

```python
result = detect("Dr. Jane Wilson visited Stanford University in Palo Alto", mode="ai")

for d in result.detections:
    print(f"{d.type}: {d.text} ({d.confidence:.0%})")
# PERSON: Dr. Jane Wilson (92%)
# ORGANIZATION: Stanford University (88%)
# LOCATION: Palo Alto (85%)
```

## Secrets Detection

Three-tier detection, always uses regex (95%+ accuracy):

```python
from zero_harm_ai_detectors import detect_secrets_regex

secrets = detect_secrets_regex("""
    OPENAI_KEY = "sk-1234567890abcdef1234567890abcdef"
    AWS_KEY    = "AKIAIOSFODNN7EXAMPLE"
    GH_TOKEN   = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234"
""")
```

**Supported patterns:**
OpenAI · AWS (access + secret) · GitHub · Stripe · Google · Slack ·
Twilio · SendGrid · npm · PyPI · Anthropic · Generic secrets (context + entropy)

## Harmful Content Detection

```python
result = detect(text, detect_pii=False, detect_secrets=False)

print(result.harmful)         # True / False
print(result.severity)        # "none" | "low" | "medium" | "high"
print(result.harmful_scores)  # {"insult": 0.6, "threat_phrases": 0.8, ...}
```

**Severity rules:**

| Condition | Severity |
|-----------|----------|
| `identity_hate` found | **high** |
| Explicit threat phrase | **high** |
| 2+ threat words or 6+ total matches | **high** |
| 1 threat word or 4+ total matches | **medium** |
| 2+ obscene terms | **medium** |
| Any other match | **low** |

## Redaction Strategies

```python
text = "Email: john@example.com"

detect(text, redaction_strategy="token")     # Email: [REDACTED_EMAIL]
detect(text, redaction_strategy="mask_all")  # Email: ****************
detect(text, redaction_strategy="mask_last4")# Email: ************.com
detect(text, redaction_strategy="hash")      # Email: [HASH:a1b2c3d4e5f6]
```

## Selective Detection

```python
# PII only
result = detect(text, detect_secrets=False, detect_harmful=False)

# Secrets only
result = detect(text, detect_pii=False, detect_harmful=False)

# Skip harmful check for speed
result = detect(text, detect_harmful=False)
```

## AI Configuration

```python
from zero_harm_ai_detectors import detect, AIConfig

config = AIConfig(
    ner_model="dslim/bert-base-NER",
    ner_threshold=0.8,   # higher = fewer false positives
    harmful_threshold=0.6,
    device="cuda",       # or "cpu"
)

result = detect(text, mode="ai", ai_config=config)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   detect(text, mode=...)                  │
├────────────────────────┬─────────────────────────────────┤
│     mode='regex'       │          mode='ai'               │
│  regex_detectors.py    │       ai_detectors.py            │
│  ┌──────────────────┐  │  ┌──────────────────────────┐   │
│  │ Email / Phone    │  │  │ Regex (structured PII)   │   │
│  │ SSN / CC / DOB   │  │  │ + AI NER (names/locs/    │   │
│  │ Secrets          │  │  │   orgs via transformers) │   │
│  │ Harmful (regex)  │  │  │ + AI harmful (context)   │   │
│  └──────────────────┘  │  └──────────────────────────┘   │
│            ↓           │              ↓                   │
│       core_patterns.py (shared patterns, types, redaction)│
└───────────────────────────────────────────────────────────┘
                          ↓
                    DetectionResult
             (identical for both modes)
```

## Integration Example — GitHub App PR Scanner

```python
from zero_harm_ai_detectors import detect, AI_AVAILABLE

def scan_pr_diff(diff: str, is_paid_user: bool) -> dict:
    mode = "ai" if (is_paid_user and AI_AVAILABLE) else "regex"

    result = detect(
        diff,
        mode=mode,
        detect_pii=True,
        detect_secrets=True,
        detect_harmful=False,
    )

    blocking_types = {"API_KEY", "SECRET", "SSN", "CREDIT_CARD"}
    return {
        "has_issues":   len(result.detections) > 0,
        "should_block": any(d.type in blocking_types for d in result.detections),
        "detections":   result.to_dict()["detections"],
    }
```

## Performance

| Text length | Regex | AI |
|-------------|-------|----|
| ~50 chars | 1–2ms | 50–100ms |
| ~500 chars | 2–3ms | 100–150ms |
| ~5 000 chars | 3–5ms | 150–200ms |
| Throughput | ~500/sec | ~5–10/sec |

## Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=zero_harm_ai_detectors

# Skip AI tests (if dependencies not installed)
pytest tests/test_core_and_regex.py -v
```

## License

MIT — see [LICENSE](LICENSE).

## Links

- **PyPI**: https://pypi.org/project/zero-harm-ai-detectors/
- **GitHub**: https://github.com/zeroharm-ai/zero-harm-ai-detectors
- **Issues**: https://github.com/zeroharm-ai/zero-harm-ai-detectors/issues
