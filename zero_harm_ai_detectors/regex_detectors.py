"""
Regex-based PII and secrets detection

Fast pattern-based detection for structured data.
Part of the free tier in the freemium model.

File: zero_harm_ai_detectors/regex_detectors.py
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .input_validation import validate_input

# ── Shared patterns and utilities ──────────────────────────────
from .core_patterns import (
    # Redaction
    RedactionStrategy,
    apply_redaction,
    redact_spans,
    # Validators
    luhn_check,
    # PII patterns
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_CARD_RE,
    # Secrets (full three-tier scan)
    find_secrets,
)

# Re-export RedactionStrategy so existing `from regex_detectors import RedactionStrategy` works
__all__ = [
    "RedactionStrategy",
    "EmailDetector",
    "PhoneDetector",
    "SSNDetector",
    "CreditCardDetector",
    "BankAccountDetector",
    "DOBDetector",
    "DriversLicenseDetector",
    "MRNDetector",
    "PersonNameDetector",
    "AddressDetector",
    "SecretsDetector",
    "detect_pii",
    "detect_secrets",
    "redact_text",
    "default_detectors",
]


# ---------- Base detector helpers ----------
CONTEXT_WINDOW = 30


class BaseDetector:
    type: str = "PII"

    def finditer(self, text: str) -> Iterable[Union[re.Match, Tuple[int, int]]]:
        raise NotImplementedError

    def _context(self, text: str, start: int, end: int, window: int = CONTEXT_WINDOW) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right]


class RegexDetector(BaseDetector):
    pattern: re.Pattern

    def __init__(self, pattern: re.Pattern):
        self.pattern = pattern

    def finditer(self, text: str) -> Iterable[re.Match]:
        return self.pattern.finditer(text)


# ---------- PII Detectors (use shared patterns from core_patterns) ----------

class EmailDetector(RegexDetector):
    type = "EMAIL"

    def __init__(self):
        super().__init__(EMAIL_RE)


class PhoneDetector(RegexDetector):
    type = "PHONE"

    def __init__(self):
        super().__init__(PHONE_RE)


class SSNDetector(RegexDetector):
    type = "SSN"

    def __init__(self):
        super().__init__(SSN_RE)


class CreditCardDetector(BaseDetector):
    type = "CREDIT_CARD"

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in CREDIT_CARD_RE.finditer(text):
            raw = m.group()
            digits_only = re.sub(r"\D", "", raw)
            if luhn_check(digits_only):
                yield (m.start(), m.end())


class BankAccountDetector(BaseDetector):
    type = "BANK_ACCOUNT"
    KEYWORDS = re.compile(r"\b(account|acct|routing|iban|aba|ach|bank)\b", re.I)
    DIGITS = re.compile(r"\b\d{6,17}\b")

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in self.DIGITS.finditer(text):
            ctx = self._context(text, m.start(), m.end())
            if self.KEYWORDS.search(ctx) and not re.fullmatch(r"\d{9}", m.group()):
                yield (m.start(), m.end())


class DOBDetector(BaseDetector):
    type = "DOB"
    DATE1 = re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19\d\d|20\d\d)\b")
    DATE2 = re.compile(r"\b(19\d\d|20\d\d)-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b")
    DATE3 = re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{1,2}),\s*(19\d\d|20\d\d)\b",
        re.I,
    )

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for pat in (self.DATE1, self.DATE2, self.DATE3):
            for m in pat.finditer(text):
                yield (m.start(), m.end())


class DriversLicenseDetector(BaseDetector):
    type = "DRIVERS_LICENSE"
    KEYWORDS = re.compile(
        r"\b(driver'?s?\s+license|dl\s*#?|dmv|lic#|license\s*no\.?|dlnum)\b", re.I
    )
    TOKEN = re.compile(r"\b[A-Z0-9]{6,12}\b", re.I)
    CA = re.compile(r"\b[A-Z]\d{7}\b")
    NY = re.compile(r"\b([A-Z]\d{7}|\d{9}|\d{8})\b")
    TX = re.compile(r"\b\d{8}\b")

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in self.TOKEN.finditer(text):
            start, end = m.start(), m.end()
            ctx = self._context(text, start, end)
            if (
                self.KEYWORDS.search(ctx)
                or self.CA.fullmatch(m.group())
                or self.NY.fullmatch(m.group())
                or self.TX.fullmatch(m.group())
            ):
                yield (start, end)


class MRNDetector(BaseDetector):
    type = "MEDICAL_RECORD_NUMBER"
    KEYWORDS = re.compile(r"\b(MRN|medical\s*record\s*(no\.?|number)?)\b", re.I)
    DIGITS = re.compile(r"\b[A-Z]?\d{6,12}[A-Z]?\b", re.I)

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in self.DIGITS.finditer(text):
            ctx = self._context(text, m.start(), m.end())
            if self.KEYWORDS.search(ctx):
                yield (m.start(), m.end())


class PersonNameDetector(BaseDetector):
    type = "PERSON_NAME"

    NAME = re.compile(
        r"(?:Name:\s*)?\b([A-Z][a-zA-Z]{1,}\s+[a-zA-Z]{1,}(?:\s+[a-zA-Z]{1,})?)\b"
    )

    EXCLUDES = {
        "Street", "Avenue", "Road", "Company", "Corp", "LLC", "Inc",
        "Boulevard", "Lane", "Court", "Circle", "Way", "Place", "Highway",
        "North", "South", "East", "West",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    }

    NON_NAME_PATTERNS = [
        re.compile(r"\bMy\s+[Nn]ame\s+[Ii]s\b", re.I),
        re.compile(r"\bEmail\s+[Aa]ddress\s*:?\s*", re.I),
        re.compile(r"\bEmail\s*:?\s*", re.I),
    ]

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in self.NAME.finditer(text):
            value = m.group(1).strip()
            start, end = m.start(1), m.end(1)

            context = self._context(text, start, end, window=20)
            skip_match = False
            for pattern in self.NON_NAME_PATTERNS:
                if pattern.search(context):
                    skip_match = True
                    break
            if skip_match:
                continue

            words = value.split()
            if len(words) >= 2 and words[-1] in self.EXCLUDES:
                continue

            if re.search(
                r"\b\d{3,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Drive|Dr)\b",
                context,
                re.I,
            ):
                continue

            if re.search(r"@\w+\.\w+|email\s*:|contact\s*information", context, re.I):
                continue

            yield (start, end)


# ---------- Address Detector ----------
class AddressDetector(BaseDetector):
    type = "ADDRESS"

    DIRECTION = r"(?:N|S|E|W|NE|NW|SE|SW)"
    STREET_TYPE = (
        r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|"
        r"Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|"
        r"Trl|Trail|Hwy|Highway)"
    )
    UNIT = r"(?:Apt|Unit|Ste|Suite|\#)\s*[A-Za-z0-9-]+"
    ZIP = r"(?:\d{5}(?:-\d{4})?)"

    STATE_ABBREV = (
        r"(?:AL|AK|AZ|AR|CA|CO|CT|DC|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|"
        r"MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|"
        r"SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)"
    )

    STATE_FULL = (
        r"(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|"
        r"Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|"
        r"Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
        r"Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|"
        r"New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|"
        r"North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|"
        r"South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|"
        r"Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|"
        r"District\s+of\s+Columbia)"
    )

    STATE = rf"(?:{STATE_FULL}|{STATE_ABBREV})"
    CITY = r"(?:[A-Za-z][A-Za-z .'-]+)"
    NAME_TOKEN = r"[A-Za-z0-9.'-]+"

    STREET_ADDR = re.compile(
        rf"""
        \b
        (?P<num>\d{{1,6}})
        \s+
        (?:(?P<pre_dir>{DIRECTION})\s+)?
        (?P<n>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}})
        \s+
        (?P<type>{STREET_TYPE})
        (?:\s+(?P<post_dir>{DIRECTION}))?
        (?:\s+(?P<unit>{UNIT}))?
        (?:\s*,?\s*(?P<city>{CITY})\s*,?\s*(?P<state>{STATE})\s+(?P<zip>{ZIP}))?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    POBOX = r"(?:P\.?\s*O\.?\s*Box\s*\d+)"
    POBOX_ADDR = re.compile(
        rf"""
        \b
        (?P<pobox>{POBOX})
        (?:\s*,?\s*(?P<city>{CITY})\s*,?\s*(?P<state>{STATE})\s+(?P<zip>{ZIP}))?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def finditer(self, text: str):
        for m in self.STREET_ADDR.finditer(text):
            yield (m.start(), m.end())
        for m in self.POBOX_ADDR.finditer(text):
            yield (m.start(), m.end())


# ---------- Secrets Detector (delegates to core_patterns.find_secrets) ----------
class SecretsDetector:
    """Secrets detector that delegates to the shared three-tier scan in core_patterns.

    The finditer() interface returns List[Dict] for backward compatibility
    with _locate_spans().
    """

    type = "SECRETS"

    def finditer(self, text: str) -> List[Dict[str, Any]]:
        return find_secrets(text)


# ---------- Defaults ----------
def default_detectors() -> List[BaseDetector]:
    return [
        EmailDetector(),
        PhoneDetector(),
        SSNDetector(),
        CreditCardDetector(),
        BankAccountDetector(),
        DOBDetector(),
        DriversLicenseDetector(),
        MRNDetector(),
        PersonNameDetector(),
        AddressDetector(),
    ]


# ---------- Span location helper ----------
PatternOrDetector = Union[re.Pattern, BaseDetector, RegexDetector]


def _locate_spans(text: str, pattern_or_detector: PatternOrDetector) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []

    if isinstance(pattern_or_detector, re.Pattern):
        for m in pattern_or_detector.finditer(text):
            spans.append({"span": text[m.start():m.end()], "start": m.start(), "end": m.end()})
        return spans

    if hasattr(pattern_or_detector, "finditer"):
        for obj in pattern_or_detector.finditer(text):
            if isinstance(obj, dict):
                # SecretsDetector returns dicts directly
                spans.append(obj)
            elif isinstance(obj, tuple) and len(obj) == 2:
                start, end = obj
                spans.append({"span": text[start:end], "start": start, "end": end})
            else:
                start, end = obj.start(), obj.end()
                spans.append({"span": text[start:end], "start": start, "end": end})
        return spans

    raise TypeError("pattern_or_detector must be regex or detector")


# ---------- Public API ----------

def detect_pii(
    text: str, detectors: Optional[List[BaseDetector]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Detect PII in text using regex patterns.

    Args:
        text: Input text to scan
        detectors: Optional list of detector instances (uses defaults if None)

    Returns:
        Dict mapping detection type → list of span dicts
    """
    text = validate_input(text, mode="regex")
    dets = detectors or default_detectors()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for det in dets:
        spans = _locate_spans(text, det)
        if spans:
            out.setdefault(det.type, []).extend(spans)
    return out


def detect_secrets(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Scan text for API keys and secrets using the shared three-tier scan.

    Returns grouped under ``'SECRETS'`` key for backward compatibility.
    """
    text = validate_input(text, mode="regex")
    findings = find_secrets(text)
    if findings:
        return {"SECRETS": findings}
    return {}


def redact_text(
    text: str,
    spans_or_map: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]],
    strategy: str = "mask_all",
) -> str:
    """Redact matched spans in text.

    Accepts either a flat list of span dicts or a grouped dict
    (as returned by detect_pii / detect_secrets).
    """
    try:
        strat = RedactionStrategy(strategy)
    except ValueError:
        strat = RedactionStrategy.MASK_ALL

    if isinstance(spans_or_map, dict):
        spans_list = [d for lst in spans_or_map.values() for d in lst]
    else:
        spans_list = spans_or_map

    return redact_spans(text, spans_list, strat)
