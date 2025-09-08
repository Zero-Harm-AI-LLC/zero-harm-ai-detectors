from .detectors import (
    detect_pii, 
    detect_secrets, 
    redact_text,
    default_detectors,
    EmailDetector,
    PhoneDetector,
    SSNDetector,
    CreditCardDetector,
    BankAccountDetector,
    DOBDetector,
    DriversLicenseDetector,
    MRNDetector,
    PersonNameDetector,
    AddressDetector,
    SecretsDetector,
    RedactionStrategy
)

from .harmful_detector import (
    HarmfulTextDetector,
    DetectionConfig
)

# Version will be automatically updated by GitHub Actions
__version__ = "0.1.0"

__all__ = [
    'detect_pii', 'detect_secrets', 'redact_text', 'default_detectors',
    'EmailDetector', 'PhoneDetector', 'SSNDetector', 'CreditCardDetector',
    'BankAccountDetector', 'DOBDetector', 'DriversLicenseDetector', 
    'MRNDetector', 'PersonNameDetector', 'AddressDetector', 'SecretsDetector',
    'RedactionStrategy', 'HarmfulTextDetector', 'DetectionConfig'
]