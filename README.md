# Zero Harm AI Detectors

**AI-powered detection of PII, secrets, and harmful content in text.**

Now with **transformer-based models** for 85-95% accuracy in person name detection!

## 🚀 What's New in v0.2.0

- **🤖 AI-Powered Detection**: Uses BERT/RoBERTa for accurate PII detection
- **🎯 Unified Pipeline**: Single `ZeroHarmPipeline` class for everything
- **📊 Confidence Scores**: Every detection includes confidence (0-1)
- **🔄 Backward Compatible**: Drop-in replacement - works with old API
- **⚡ Smart Detection**: AI accuracy + regex speed where appropriate
- **🌍 Better Support**: International names, locations, organizations

## 📦 Installation

### Quick Start (CPU)
```bash
pip install zero_harm_ai_detectors transformers torch
```

### For GPU (Faster)
```bash
pip install zero_harm_ai_detectors transformers
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Lightweight (Regex Only)
```bash
pip install zero_harm_ai_detectors
```

## 🎯 Quick Start

### One-Line Detection (Easiest)

```python
from zero_harm_ai_detectors import detect_all_threats

result = detect_all_threats(
    "Contact John Smith at john@example.com. API key: sk-abc123."
)

print(result['redacted'])
# Output: Contact [REDACTED_PERSON] at [REDACTED_EMAIL]. API key: [REDACTED_SECRET].

print(result['detections'])
# Output: {'PERSON': [...], 'EMAIL': [...], 'API_KEY': [...]}
```

### Full Pipeline (Recommended)

```python
from zero_harm_ai_detectors import ZeroHarmPipeline, RedactionStrategy

# Initialize once (loads models)
pipeline = ZeroHarmPipeline()

# Use many times
text = "Email John Smith at john@example.com or call 555-123-4567"
result = pipeline.detect(text, redaction_strategy=RedactionStrategy.TOKEN)

print(f"Original: {result.original_text}")
print(f"Redacted: {result.redacted_text}")

for det in result.detections:
    print(f"  {det.type}: {det.text} (confidence: {det.confidence:.0%})")
```

### Legacy API (Still Works!)

```python
# Old code works unchanged!
from zero_harm_ai_detectors import detect_pii, detect_secrets, redact_text

text = "Contact john@example.com with API key sk-abc123"

pii = detect_pii(text)  # Now uses AI automatically!
secrets = detect_secrets(text)

redacted = redact_text(text, {**pii, **secrets})
```

## 🎨 Features

### Detectable Content

#### PII (Personally Identifiable Information)
- ✉️ **Emails**: `john.doe@email.com`
- 📞 **Phone Numbers**: `555-123-4567`
- 🆔 **SSN**: `123-45-6789`
- 💳 **Credit Cards**: `4532-0151-1283-0366`
- 👤 **Person Names**: AI-powered, 85-95% accuracy (NEW!)
- 📍 **Locations**: Cities, states, countries (NEW!)
- 🏢 **Organizations**: Companies, institutions (NEW!)
- 🏠 **Addresses**: Street addresses, P.O. boxes
- 🏥 **Medical Records**: MRN detection
- 🚗 **Driver's Licenses**: US state formats
- 📅 **Dates of Birth**: Multiple formats

#### Secrets & Credentials
- 🔑 **API Keys**: OpenAI, AWS, Google, etc.
- 🎫 **Tokens**: GitHub, Slack, Stripe, JWT
- 🔐 **Passwords**: Pattern-based detection

#### Harmful Content
- ☠️ **Toxic Language**
- ⚔️ **Threats**
- 😡 **Insults**
- 🔞 **Obscene Content**
- 👿 **Identity Hate**

### Redaction Strategies

```python
# TOKEN: [REDACTED_EMAIL]
RedactionStrategy.TOKEN

# MASK_ALL: ********************
RedactionStrategy.MASK_ALL

# MASK_LAST4: ****************.com
RedactionStrategy.MASK_LAST4

# HASH: 8d969eef6ecad3c29a3a...
RedactionStrategy.HASH
```

## 📚 Advanced Usage

### Custom Configuration

```python
from zero_harm_ai_detectors import ZeroHarmPipeline, PipelineConfig

config = PipelineConfig(
    pii_threshold=0.8,  # Higher confidence threshold
    pii_model="Jean-Baptiste/roberta-large-ner-english",  # Better model
    harmful_threshold_per_label=0.6,
    device="cuda"  # Use GPU
)

pipeline = ZeroHarmPipeline(config)
```

### Selective Detection

```python
# Only detect PII
result = pipeline.detect(
    text,
    detect_pii=True,
    detect_secrets=False,
    detect_harmful=False
)
```

### Filtering by Confidence

```python
result = pipeline.detect(text)

# Only high-confidence detections
high_conf = [d for d in result.detections if d.confidence >= 0.9]

for det in high_conf:
    print(f"{det.type}: {det.text} ({det.confidence:.2%})")
```

### Batch Processing

```python
texts = [
    "Email: john@example.com",
    "Phone: 555-123-4567",
    "Meet Jane at Microsoft"
]

for text in texts:
    result = pipeline.detect(text)
    print(f"Text: {text}")
    print(f"Redacted: {result.redacted_text}")
```

### API Integration

```python
from flask import Flask, request, jsonify
from zero_harm_ai_detectors import ZeroHarmPipeline

app = Flask(__name__)

# Load once at startup
pipeline = ZeroHarmPipeline()

@app.route("/api/check_privacy", methods=["POST"])
def check_privacy():
    data = request.json
    text = data.get("text", "")
    
    result = pipeline.detect(text)
    
    return jsonify({
        "original": result.original_text,
        "redacted": result.redacted_text,
        "detections": result.to_dict()["detections"],
        "harmful": result.harmful,
        "severity": result.severity
    })
```

## 🔬 Comparison: Regex vs AI

| Feature | Regex (Old) | AI (New) | Winner |
|---------|-------------|----------|--------|
| Person Names | 30-40% | 85-95% | 🏆 AI |
| Locations | ❌ | 80-90% | 🏆 AI |
| Organizations | ❌ | 75-85% | 🏆 AI |
| Context Understanding | ❌ | ✅ | 🏆 AI |
| Email Detection | 99%+ | 99%+ | 🤝 Tie |
| Phone Detection | 95%+ | 95%+ | 🤝 Tie |
| Speed (single) | 1-5ms | 50-200ms | 🏆 Regex |
| False Positives | High | Low | 🏆 AI |

## ⚡ Performance

### Speed Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Pipeline loading | 5-10s | One-time at startup |
| Email detection | 50ms | AI + regex |
| Person name | 150ms | AI (transformer) |
| Full detection | 200ms | All types |

### Best Practices

```python
# ✅ Good: Load once, reuse
PIPELINE = ZeroHarmPipeline()

def process(text):
    return PIPELINE.detect(text)  # Fast!

# ❌ Bad: Load every time
def process(text):
    pipeline = ZeroHarmPipeline()  # Slow!
    return pipeline.detect(text)
```

## 🔄 Migration from v0.1.x

Your old code works without changes:

```python
# Old code (v0.1.x)
from zero_harm_ai_detectors import detect_pii, detect_secrets

pii = detect_pii("Contact john@example.com")  # Now uses AI!
secrets = detect_secrets("API key sk-abc123")

# Force old regex behavior if needed
pii = detect_pii(text, use_ai=False)
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions.

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=zero_harm_ai_detectors --cov-report=html

# Run specific test
pytest tests/test_ai_detectors.py -v
```

## 📊 Model Information

### PII Detection Models

| Model | Size | Languages | Accuracy |
|-------|------|-----------|----------|
| `dslim/bert-base-NER` | 420MB | English | 85% |
| `Jean-Baptiste/roberta-large-ner-english` | 1.3GB | English | 92% |

### Harmful Content Model

| Model | Size | Languages | Categories |
|-------|------|-----------|------------|
| `unitary/multilingual-toxic-xlm-roberta` | 1.1GB | 100+ | 6 labels |

## 💡 Use Cases

- **API Gateways**: Scan requests/responses for sensitive data
- **Chat Applications**: Prevent PII leakage
- **Data Pipelines**: Clean datasets before sharing
- **Content Moderation**: Filter harmful content
- **Compliance**: GDPR, HIPAA, PCI-DSS
- **Security**: Detect leaked credentials

## 🐛 Troubleshooting

### Models not loading
```bash
pip install transformers torch
```

### Out of memory
```python
config = PipelineConfig(device="cpu")  # Use CPU
```

### Slow performance
```python
# Skip unnecessary detection
result = pipeline.detect(text, detect_harmful=False)
```

## 📞 Support

- **Email**: info@zeroharmai.com
- **GitHub Issues**: [Create an issue](https://github.com/Zero-Harm-AI-LLC/zero-harm-ai-detectors/issues)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- PyTorch team for the ML framework

---

**Made with ❤️ by [Zero Harm AI LLC](https://zeroharmai.com)**

*Protecting privacy, one detection at a time.*
