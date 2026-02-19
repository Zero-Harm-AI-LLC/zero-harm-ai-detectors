# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Initial Release

**Zero Harm AI Detectors** is a privacy and content safety detection library
offering two modes through a single unified API.

#### Detection Modes

- **`mode='regex'`** — Fast pattern-based detection (1–5ms). No ML dependencies.
  Best for structured data at high throughput.
- **`mode='ai'`** — Transformer-based detection (50–200ms). Requires
  `pip install zero_harm_ai_detectors[ai]`. Best for unstructured text with
  names, locations, and organisations.

#### PII Detection

| Type | Regex | AI |
|------|-------|----|
| Email | ✅ 99% | ✅ 99% |
| Phone | ✅ 95% | ✅ 95% |
| SSN | ✅ 98% | ✅ 98% |
| Credit Card (Luhn) | ✅ 99% | ✅ 99% |
| Bank Account | ✅ 90% | ✅ 90% |
| Date of Birth | ✅ 85% | ✅ 85% |
| Address | ✅ 85% | ✅ 85% |
| Person Name | ⚠️ 30–40% | ✅ 85–95% |
| Location | ❌ | ✅ 80–90% |
| Organisation | ❌ | ✅ 75–85% |

#### Secrets Detection (Three-Tier)

- **Tier 1** — Structured prefixes: OpenAI, AWS, GitHub, Stripe, Slack, Google,
  Twilio, SendGrid, npm, PyPI, Anthropic
- **Tier 2** — AWS Secret Access Key (context + entropy)
- **Tier 3** — Generic secrets (context keyword + entropy threshold)

#### Harmful Content Detection

Six pattern categories with multi-factor severity:

| Category | Severity trigger |
|----------|-----------------|
| `identity_hate` | Always **high** |
| `threat_phrases` | Always **high** |
| `threat` (word) | **medium** / **high** depending on count |
| `toxic` | Contributes to escalation |
| `insult` | Contributes to escalation |
| `obscene` | Contributes to escalation |

#### Redaction Strategies

`token` · `mask_all` · `mask_last4` · `hash`

#### Architecture

- `core_patterns.py` — single source of truth for all patterns, validators,
  result types, and redaction utilities
- `regex_detectors.py` — regex detection mode
- `ai_detectors.py` — AI detection mode (optional dependency)
- `input_validation.py` — DoS/ReDoS protection

#### Dependencies

| Install | Dependencies |
|---------|-------------|
| `pip install zero_harm_ai_detectors` | `regex` only |
| `pip install zero_harm_ai_detectors[ai]` | + `transformers`, `torch`, `sentencepiece` |
