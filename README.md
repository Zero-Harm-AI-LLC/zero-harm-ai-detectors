# Zero Harm Detectors

A comprehensive Python library for detecting and redacting personally identifiable information (PII), secrets, and harmful content in text.

## Features

- **PII Detection**: Email addresses, phone numbers, SSN, credit cards, bank accounts, dates of birth, driver's licenses, medical record numbers, person names, and addresses
- **Secrets Detection**: API keys, tokens, and other sensitive credentials
- **Harmful Content Detection**: Toxic language, threats, insults, and other harmful content using ML models
- **Flexible Redaction**: Multiple redaction strategies (mask all, mask last 4, hash)
- **High Performance**: Optimized for production use with transformer models

## Installation

```bash
pip install zero-harm-ai-detectors
```

## Quick Start

### PII Detection and Redaction

```python
from detectors import detect_pii, redact_text

text = "Contact John Doe at john.doe@email.com or call 555-123-4567"

# Detect PII
pii_results = detect_pii(text)
print(pii_results)
# Output: {'EMAIL': [{'span': 'john.doe@email.com', 'start': 22, 'end': 40}], 
#          'PHONE': [{'span': '555-123-4567', 'start': 49, 'end': 61}]}

# Redact the text
redacted = redact_text(text, pii_results, strategy="mask_all")
print(redacted)
# Output: "Contact John Doe at ****************** or call ************"
```

### Secrets Detection

```python
from zero_harm_detectors import detect_secrets

text = "My API key is sk-1234567890abcdef1234567890abcdef"

secrets = detect_secrets(text)
print(secrets)
# Output: {'SECRETS': [{'span': 'sk-1234567890abcdef1234567890abcdef', 'start': 14, 'end': 46}]}
```

### Harmful Content Detection

```python
from zero_harm_detectors import HarmfulTextDetector

detector = HarmfulTextDetector()
result = detector.detect("I hate you and want to hurt you")

print(result)
# Output: {
#     'text': 'I hate you and want to hurt you',
#     'harmful': True,
#     'severity': 'high',
#     'active_labels': ['toxic', 'threat'],
#     'scores': {'toxic': 0.95, 'threat': 0.87, 'insult': 0.23, ...}
# }
```

### Combined Detection

```python
from zero_harm_detectors import detect_pii, detect_secrets, redact_text

def scan_text(text):
    results = {}
    
    # Detect PII
    pii = detect_pii(text)
    if pii:
        results.update(pii)
    
    # Detect secrets
    secrets = detect_secrets(text)
    if secrets:
        results.update(secrets)
    
    # Redact everything found
    if results:
        redacted = redact_text(text, results)
        return redacted, results
    
    return text, results

# Example usage
text = "Email me at admin@company.com with API key sk-abc123def456"
redacted_text, findings = scan_text(text)
print(f"Original: {text}")
print(f"Redacted: {redacted_text}")
print(f"Found: {list(findings.keys())}")
```

## Advanced Usage

### Custom Detection Configuration

```python
from zero_harm_detectors import HarmfulTextDetector, DetectionConfig

# Custom configuration
config = DetectionConfig(
    threshold_per_label=0.7,  # Higher threshold for label activation
    overall_threshold=0.8,    # Higher threshold for harmful classification
    threat_min_score_on_cue=0.9  # Boost threat scores when keywords found
)

detector = HarmfulTextDetector(config)
result = detector.detect("Your text here")
```

### Custom Redaction Strategies

```python
from zero_harm_detectors import redact_text, RedactionStrategy

text = "Call me at 555-123-4567"
findings = detect_pii(text)

# Different redaction strategies
mask_all = redact_text(text, findings, strategy=RedactionStrategy.MASK_ALL)
# Output: "Call me at ************"

mask_last4 = redact_text(text, findings, strategy=RedactionStrategy.MASK_LAST4)
# Output: "Call me at ********4567"

hashed = redact_text(text, findings, strategy=RedactionStrategy.HASH)
# Output: "Call me at 8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92"
```

### Individual Detectors

```python
from zero_harm_detectors import EmailDetector, PhoneDetector

email_detector = EmailDetector()
phone_detector = PhoneDetector()

text = "Contact: john@example.com or 555-0123"

# Use individual detectors
for match in email_detector.finditer(text):
    print(f"Email found: {match.group()}")

for match in phone_detector.finditer(text):
    print(f"Phone found: {match.group()}")
```

## Supported Detection Types

### PII Types
- `EMAIL`: Email addresses
- `PHONE`: Phone numbers (US format)
- `SSN`: Social Security Numbers
- `CREDIT_CARD`: Credit card numbers (with Luhn validation)
- `BANK_ACCOUNT`: Bank account numbers
- `DOB`: Dates of birth
- `DRIVERS_LICENSE`: Driver's license numbers
- `MEDICAL_RECORD_NUMBER`: Medical record numbers
- `PERSON_NAME`: Person names
- `ADDRESS`: Street addresses

### Secret Types
- `SECRETS`: API keys, tokens, and credentials including:
  - OpenAI API keys (sk-*, sk-proj-*, sk-org-*)
  - AWS keys (AKIA*, ASIA*)
  - Google API keys (AIza*)
  - Slack tokens (xox*)
  - GitHub tokens (ghp_*)
  - Stripe keys (sk_live_*, sk_test_*)
  - JWT tokens
  - And more...

### Harmful Content Labels
- `toxic`: General toxic language
- `severe_toxic`: Severely toxic content
- `obscene`: Obscene language
- `threat`: Threatening language
- `insult`: Insulting language
- `identity_hate`: Identity-based hate speech

## Performance Notes

- The harmful content detector loads a transformer model (~500MB) on first use
- Consider using model caching in production environments
- PII and secrets detection is lightweight and fast
- For high-throughput applications, initialize detectors once and reuse

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

### Make your changes and commit
```bash
git add .
git commit -m "Add new feature"
```
### Create version tag
```bash
git tag v0.1.0
```
### Push with tags
```bash
git push origin main --tags
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support, please contact [info@zeroharmai.com](mailto:info@zeroharmai.com) or open an issue on GitHub.