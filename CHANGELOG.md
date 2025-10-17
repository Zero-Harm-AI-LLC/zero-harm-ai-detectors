# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-01-XX

### Added
- **AI-Powered Detection**: Transformer-based models (BERT/RoBERTa) for PII detection
- **ZeroHarmPipeline**: Unified detection class combining PII, secrets, and harmful content detection
- **Confidence Scores**: All detections now include confidence scores (0.0-1.0) for transparency
- **New Detection Types**: 
  - Location detection (cities, states, countries) using AI
  - Organization detection (companies, institutions) using AI
  - Enhanced person name detection with contextual understanding
- **New API Functions**:
  - `detect_all_threats()`: One-line function to detect everything
  - `get_pipeline()`: Get global pipeline instance
- **Smart API Routing**: Automatically uses AI when available, falls back to regex
- **Comprehensive Documentation**:
  - `MIGRATION_GUIDE.md`: Step-by-step upgrade instructions
  - Updated `README.md` with AI features and examples
  - Complete API documentation with examples
- **Test Suite**: Comprehensive tests for AI detection (`test_ai_detectors.py`)
- **Multiple Redaction Strategies**: TOKEN, MASK_ALL, MASK_LAST4, HASH

### Changed
- **Person Name Detection**: Improved from 30-40% accuracy to 85-95% accuracy using AI models
- **Contextual Understanding**: AI models understand context (e.g., "Apple" company vs "apple" fruit)
- **Better International Support**: Improved handling of non-English names and locations
- **Performance**: AI detection takes 50-200ms (vs 1-5ms regex), but with much higher accuracy
- **Import Paths**: All imports work unchanged, but now use AI automatically when transformers installed

### Maintained
- **100% Backward Compatibility**: All v0.1.x code works unchanged
- **Legacy API Support**: Old `detect_pii()`, `detect_secrets()` functions work identically
- **Regex Fallback**: Can force regex-only detection with `use_ai=False` parameter
- **All existing detection types**: Email, phone, SSN, credit cards, etc. still work

### Performance
- **Model Loading**: One-time 5-10 seconds at application startup
- **Detection Speed**: 
  - AI-based: 50-200ms per text
  - Regex-based: 1-5ms per text (still available)
- **Memory Usage**: ~2GB for AI models
- **GPU Support**: Optional CUDA acceleration for faster processing

### Dependencies
- **New Optional Dependencies** (for AI features):
  - `transformers>=4.30.0`
  - `torch>=2.0.0`
  - `sentencepiece>=0.1.99`
- **Core Dependencies** (unchanged):
  - `regex>=2022.1.18`
  - `numpy>=1.21.0`

### Migration
- No code changes required for basic upgrade
- Install with AI: `pip install zero_harm_ai_detectors[ai]`
- See `MIGRATION_GUIDE.md` for detailed upgrade instructions

### Technical Details
- **AI Models Used**:
  - PII Detection: `dslim/bert-base-NER` (default) or `Jean-Baptiste/roberta-large-ner-english`
  - Harmful Content: `unitary/multilingual-toxic-xlm-roberta`
- **Architecture**: Combines AI-based NER with regex patterns for optimal accuracy
- **Confidence Thresholds**: Configurable per detection type

## [0.1.2] - 2025-01-XX

### Changed
- **BREAKING**: Renamed package from `zero-harm-ai-detectors` to `zero_harm_ai_detectors` for consistent import paths
- Import path changed from `from detectors import ...` to `from zero_harm_ai_detectors import ...`
- Fixed package naming confusion between PyPI name and Python import name

### Fixed
- Resolved import issues in backend integration
- Updated all documentation examples with correct import syntax

## [0.1.1] - 2025-01-XX

### Changed
- Minor changes for backend integration compatibility

### Fixed
- Resolved import issues in backend integration
- Improved error handling in detection pipeline

## [0.1.0] - 2024-XX-XX

### Added
- **Initial Release** of zero-harm-ai-detectors
- **PII Detection** for:
  - Email addresses
  - Phone numbers (US format)
  - Social Security Numbers (SSN)
  - Credit card numbers (with Luhn validation)
  - Bank account numbers
  - Dates of birth (DOB)
  - Driver's licenses (US state formats)
  - Medical record numbers (MRN)
  - Person names (regex-based, 30-40% accuracy)
  - Addresses (street addresses and P.O. boxes)
- **Secrets Detection** for:
  - API keys (OpenAI, AWS, Google, etc.)
  - Access tokens (GitHub, Slack, Stripe, JWT)
  - Generic secret patterns
- **Harmful Content Detection**:
  - Toxic language detection using transformer models
  - Threat detection with keyword boosting
  - Insult and obscene content detection
  - Identity hate speech detection
  - Severity levels (low, medium, high)
- **Redaction Strategies**:
  - Mask all characters
  - Mask last 4 characters
  - SHA-256 hashing
- **Comprehensive Test Suite**: Unit tests for all detectors
- **MIT License**: Open source under MIT license

### Detection Accuracy (v0.1.0)
- Email: 99%+
- Phone: 95%+
- SSN: 95%+
- Credit Cards: 90%+ (with Luhn validation)
- Person Names: 30-40% (regex-based)
- Secrets: 95%+

### Technical Details (v0.1.0)
- Python 3.8+ support
- Regex-based PII detection
- Transformer-based harmful content detection
- No external API calls (all processing local)

---

## Version Comparison

| Feature | v0.1.0 | v0.1.2 | v0.2.0 |
|---------|--------|--------|--------|
| Person Name Detection | 30-40% | 30-40% | 85-95% ⭐ |
| Location Detection | ❌ | ❌ | ✅ ⭐ |
| Organization Detection | ❌ | ❌ | ✅ ⭐ |
| Confidence Scores | ❌ | ❌ | ✅ ⭐ |
| Unified Pipeline | ❌ | ❌ | ✅ ⭐ |
| Detection Speed | 1-5ms | 1-5ms | 50-200ms |
| Backward Compatible | N/A | ✅ | ✅ |

---

## Upgrade Guide

### From v0.1.x to v0.2.0

**No code changes required!** Your existing code works automatically:

```bash
# Upgrade
pip install --upgrade zero_harm_ai_detectors[ai]

# Your code still works
from zero_harm_ai_detectors import detect_pii
pii = detect_pii("Contact John Smith")  # Now 90% accurate!
```

**For new code, use the unified pipeline:**

```python
from zero_harm_ai_detectors import ZeroHarmPipeline

pipeline = ZeroHarmPipeline()
result = pipeline.detect("Contact John Smith at john@example.com")
print(result.redacted_text)
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions.

---

## Links

- **GitHub Repository**: https://github.com/Zero-Harm-AI-LLC/zero-harm-ai-detectors
- **PyPI Package**: https://pypi.org/project/zero-harm-ai-detectors/
- **Documentation**: See [README.md](README.md)
- **Migration Guide**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Issues**: https://github.com/Zero-Harm-AI-LLC/zero-harm-ai-detectors/issues

---

## Notes

### Breaking Changes
- **v0.1.2**: Package name changed from hyphenated to underscored (import path changed)
- **v0.2.0**: No breaking changes (100% backward compatible)

### Deprecation Warnings
- **v0.2.0**: Legacy `HarmfulTextDetector` class is deprecated in favor of `ZeroHarmPipeline`
  - Will be removed in v1.0.0
  - Update your code to use `ZeroHarmPipeline` for future compatibility

### Future Plans
- v0.3.0: Fine-tuned models for specific industries (healthcare, finance)
- v0.3.0: Streaming API for real-time detection
- v0.4.0: Multi-language support improvements
- v1.0.0: Stable API, remove deprecated classes
