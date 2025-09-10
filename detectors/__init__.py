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

from .harmful_detectors import (
    HarmfulTextDetector,
    DetectionConfig
)

# Get version from setuptools-scm
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development without git tags
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"

__all__ = [
    'detect_pii', 'detect_secrets', 'redact_text', 'default_detectors',
    'EmailDetector', 'PhoneDetector', 'SSNDetector', 'CreditCardDetector',
    'BankAccountDetector', 'DOBDetector', 'DriversLicenseDetector', 
    'MRNDetector', 'PersonNameDetector', 'AddressDetector', 'SecretsDetector',
    'RedactionStrategy', 'HarmfulTextDetector', 'DetectionConfig'
]