# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2025-01-XX

### Changed
- **BREAKING**: Renamed package from `zero-harm-ai-detectors` to `zero_harm_ai_detectors` for consistent import paths
- Import path changed from `from detectors import ...` to `from zero_harm_ai_detectors import ...`
- Fixed package naming confusion between PyPI name and Python import name

### Fixed
- Resolved import issues in backend integration
- Updated all documentation examples with correct import 

## [0.1.1] - 2025-01-XX

### Changed
- Minor changes to used with the backend

### Fixed
- Resolved import issues in backend integration

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of zero-harm-ai-detectors
- PII detection for emails, phones, SSN, credit cards, bank accounts, DOB, driver's licenses, medical record numbers, person names, and addresses
- Secrets detection for API keys and tokens
- Harmful content detection using transformer models
- Configurable redaction strategies (mask all, mask last 4, hash)
- Comprehensive test suite

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A