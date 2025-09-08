import re
import hashlib
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# ---------- Redaction strategy ----------
class RedactionStrategy(str, Enum):
    MASK_ALL = "mask_all"
    MASK_LAST4 = "mask_last4"
    HASH = "hash"

def _mask_all(value: str) -> str:
    return "*" * len(value)

def _mask_last4(value: str) -> str:
    keep = 4 if len(value) >= 4 else len(value)
    return "*" * (len(value) - keep) + value[-keep:]

def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _apply_strategy(value: str, strategy: RedactionStrategy) -> str:
    if strategy == RedactionStrategy.MASK_ALL:
        return _mask_all(value)
    if strategy == RedactionStrategy.MASK_LAST4:
        return _mask_last4(value)
    if strategy == RedactionStrategy.HASH:
        return _hash_value(value)
    return value

# ---------- Luhn for card validation ----------
def _luhn_check(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 12 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

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

    def __init__(self, pattern: str, flags=re.IGNORECASE):
        self.pattern = re.compile(pattern, flags)

    def finditer(self, text: str) -> Iterable[re.Match]:
        return self.pattern.finditer(text)

# ---------- Detectors ----------
class EmailDetector(RegexDetector):
    type = "EMAIL"
    def __init__(self):
        super().__init__(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

class PhoneDetector(RegexDetector):
    type = "PHONE"
    def __init__(self):
        super().__init__(r"(?:(?:\+1[-.\s]?)?\(?(?:\d{3})\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")

class SSNDetector(RegexDetector):
    type = "SSN"
    def __init__(self):
        super().__init__(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")

class CreditCardDetector(BaseDetector):
    type = "CREDIT_CARD"
    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        pattern = re.compile(r"\b(?:\d[ -]?){12,19}\b")
        for m in pattern.finditer(text):
            raw = m.group()
            digits_only = re.sub(r"\D", "", raw)
            if _luhn_check(digits_only):
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
    DATE3 = re.compile(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),\s*(19\d\d|20\d\d)\b", re.I)

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for pat in (self.DATE1, self.DATE2, self.DATE3):
            for m in pat.finditer(text):
                yield (m.start(), m.end())

class DriversLicenseDetector(BaseDetector):
    type = "DRIVERS_LICENSE"
    KEYWORDS = re.compile(r"\b(driver'?s?\s+license|dl\s*#?|dmv|lic#|license\s*no\.?|dlnum)\b", re.I)
    TOKEN = re.compile(r"\b[A-Z0-9]{6,12}\b", re.I)
    CA = re.compile(r"\b[A-Z]\d{7}\b")
    NY = re.compile(r"\b([A-Z]\d{7}|\d{9}|\d{8})\b")
    TX = re.compile(r"\b\d{8}\b")

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in self.TOKEN.finditer(text):
            start, end = m.start(), m.end()
            ctx = self._context(text, start, end)
            if self.KEYWORDS.search(ctx) or self.CA.fullmatch(m.group()) or self.NY.fullmatch(m.group()) or self.TX.fullmatch(m.group()):
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

    # Pattern that requires at least the first name part to be capitalized
    # This catches "Susan wang", "Susan Wang", but avoids "my email" 
    NAME = re.compile(
        r"(?:Name:\s*)?\b([A-Z][a-zA-Z]{1,}\s+[a-zA-Z]{1,}(?:\s+[a-zA-Z]{1,})?)\b"
    )
    
    # Extended exclusions: common non-person suffixes and words
    EXCLUDES = {
        # Address components
        "Street", "Avenue", "Road", "Company", "Corp", "LLC", "Inc", "St", "Rd", "Dr", "Ln",
        "Blvd", "Boulevard", "Lane", "Court", "Circle", "Way", "Place", "Terrace", "Trail",
        "Highway", "Parkway", "Pkwy", "Pl", "Ter", "Trl", "Hwy",
        # Directions
        "NE", "NW", "SE", "SW", "N", "S", "E", "W", "North", "South", "East", "West",
        # Common phrases/salutations
        "Dear Sir", "Dear Madam", "Thank You", "Best Regards", "Kind Regards", 
        "Yours Truly", "Very Truly", "Sincerely Yours",
        # Days and months
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August", 
        "September", "October", "November", "December",
        # Common words that aren't names
        "My", "The", "This", "That", "His", "Her", "Our", "Your", "Their"
    }
    
    # Patterns that indicate non-name content - keep this minimal
    NON_NAME_PATTERNS = [
        re.compile(r"\bMy\s+[Nn]ame\s+[Ii]s\b", re.I),   # "My name is" phrase
        re.compile(r"\bMy\s+Email\s+Is\b", re.I),        # "My Email Is" phrase  
        re.compile(r"\bThe\s+Email\s+Is\b", re.I),       # "The Email Is" phrase
        re.compile(r"\bEmail\s+Address\s+Is\b", re.I),   # "Email Address Is" phrase
        re.compile(r"\b(?:His|Her|Their)\s+[Nn]ame\s+[Ii]s\b", re.I), # "His/Her/Their name is"
    ]

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in self.NAME.finditer(text):
            value = m.group(1).strip()
            
            # Skip if it matches non-name patterns (like "My Name Is")
            skip_match = False
            for pattern in self.NON_NAME_PATTERNS:
                if pattern.search(value):
                    skip_match = True
                    break
            if skip_match:
                continue
            
            # Skip if ends with excluded tokens
            words = value.split()
            if any(words[-1] == x for x in self.EXCLUDES):
                continue
                
            # Skip if any word is in excludes (case-insensitive)
            if any(word in self.EXCLUDES for word in words):
                continue

            # Skip if looks like an address context
            start, end = m.start(1), m.end(1)
            ctx = self._context(text, start, end)
            
            # Check for address indicators
            if re.search(r"\b\d{5}\b", ctx) or re.search(r"\b(?:Street|St|Avenue|Ave|Road|Rd|Oregon|CA|NY|TX)\b", ctx, re.I):
                continue
                
            # Check for email/contact context (but allow "name is" contexts since those contain real names)
            if re.search(r"\b(?:email|@|contact|phone|address|call|write)\b", ctx, re.I):
                continue

            yield (start, end)

# ---------- Address Detector ----------
class AddressDetector(BaseDetector):
    type = "ADDRESS"

    # tokens and pieces
    DIRECTION = r"(?:N|S|E|W|NE|NW|SE|SW)"
    STREET_TYPE = r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Cir|Circle|Way|Pkwy|Parkway|Pl|Place|Ter|Terrace|Trl|Trail|Hwy|Highway)"
    UNIT = r"(?:Apt|Unit|Ste|Suite|\#)\s*[A-Za-z0-9-]+"
    ZIP = r"(?:\d{5}(?:-\d{4})?)"
    
    # State abbreviations
    STATE_ABBREV = r"(?:AL|AK|AZ|AR|CA|CO|CT|DC|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)"
    
    # Full state names
    STATE_FULL = r"(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|District\s+of\s+Columbia)"
    
    # Combined state pattern (full names first to avoid partial matches)
    STATE = rf"(?:{STATE_FULL}|{STATE_ABBREV})"
    
    CITY = r"(?:[A-Za-z][A-Za-z .'-]+)"
    NAME_TOKEN = r"[A-Za-z0-9.'-]+"

    STREET_ADDR = re.compile(
        rf"""
        \b
        (?P<num>\d{{1,6}})
        \s+
        (?:(?P<pre_dir>{DIRECTION})\s+)?           # Optional direction before street name
        (?P<n>{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}})
        \s+
        (?P<type>{STREET_TYPE})
        (?:\s+(?P<post_dir>{DIRECTION}))?          # Optional direction after street type
        (?:\s+(?P<unit>{UNIT}))?                   # Optional unit
        (?:\s*,?\s*(?P<city>{CITY})\s*,?\s*(?P<state>{STATE})\s+(?P<zip>{ZIP}))?  # Optional city, state, zip
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

class SecretsDetector(BaseDetector):
    type = "SECRETS"

    PATTERNS = [
        # --- OpenAI API keys ---
        # Legacy/standard keys: "sk-" + 32-64 base62-ish chars
        re.compile(r"\bsk-[A-Za-z0-9]{32,64}\b"),
        # Project-scoped keys: "sk-proj-" + token "-" token (lengths vary; be conservative)
        re.compile(r"\bsk-proj-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),
        # Org-scoped keys (seen in the wild; keep optional): "sk-org-" + token "-" token
        re.compile(r"\bsk-org-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),

        # --- Existing patterns you already had ---
        re.compile(r"\b(AKI|ASI)A[0-9A-Z]{16}\b"),                # AWS Access Key ID
        re.compile(r"\b[A-Za-z0-9]{16}\b"),                       # AWS Session Token
        re.compile(r"\b[0-9a-zA-Z/+]{40}\b"),                     # AWS Secret Access Key
        re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),                # Google API Key
        re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,100}\b"),       # Slack tokens
        re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),  # JWT
        re.compile(r"\bsk_(live|test)_[0-9a-zA-Z]{24}\b"),        # Stripe keys
        re.compile(r"\bghp_[0-9A-Za-z]{36}\b"),                 # GitHub PAT
        re.compile(r"\bsk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b")
    ]

    def finditer(self, text: str) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for pat in self.PATTERNS:
            for m in pat.finditer(text):
                findings.append({
                    "span": m.group(),
                    "start": m.start(),
                    "end": m.end()
                })
        return findings


# ---------- Defaults ----------
def default_detectors() -> List[BaseDetector]:
    return [
        EmailDetector(), PhoneDetector(), SSNDetector(), CreditCardDetector(),
        BankAccountDetector(), DOBDetector(), DriversLicenseDetector(),
        MRNDetector(), PersonNameDetector(), AddressDetector(),
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
        for obj in pattern_or_detector.finditer(text):  # detector
            if isinstance(obj, tuple) and len(obj) == 2:
                start, end = obj
            else:
                start, end = obj.start(), obj.end()
            spans.append({"span": text[start:end], "start": start, "end": end})
        return spans
    raise TypeError("pattern_or_detector must be regex or detector")

# ---------- Public API ----------
def detect_pii(text: str, detectors: Optional[List[BaseDetector]] = None) -> Dict[str, List[Dict[str, Any]]]:
    dets = detectors or default_detectors()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for det in dets:
        spans = _locate_spans(text, det)
        if spans:
            out.setdefault(det.type, []).extend(spans)
    return out

def redact_text(text: str, spans_or_map: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]], strategy: str = "mask_all") -> str:
    try:
        strat = RedactionStrategy(strategy)
    except ValueError:
        strat = RedactionStrategy.MASK_ALL
    if isinstance(spans_or_map, dict):
        spans_list = [d for lst in spans_or_map.values() for d in lst]
    else:
        spans_list = spans_or_map
    out = text
    for d in sorted(spans_list, key=lambda x: x["start"], reverse=True):
        s, e = d["start"], d["end"]
        original = out[s:e]
        out = out[:s] + _apply_strategy(original, strat) + out[e:]
    return out

def detect_secrets(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Scan text for API keys and secrets, return grouped under 'SECRETS'."""
    detector = SecretsDetector()
    findings = detector.finditer(text)
    if findings:
        return {"SECRETS": findings}
    return {}