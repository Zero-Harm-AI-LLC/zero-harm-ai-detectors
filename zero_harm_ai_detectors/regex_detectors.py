"""
Regex-based PII and secrets detection

Fast pattern-based detection for structured data.
Part of the free tier in the freemium model.

File: zero_harm_ai_detectors/regex_detectors.py
"""
import re
import hashlib
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .input_validation import validate_input

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
        # Require at least one separator (dash, dot, or space) between digit groups
        # This prevents false positives on continuous digit strings like credit cards
        super().__init__(
            r"(?:(?:\+1[-.\s]?)?\(?(?:\d{3})\)?[-.\s]\d{3}[-.\s]\d{4})\b"
        )

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
    NAME = re.compile(
        r"(?:Name:\s*)?\b([A-Z][a-zA-Z]{1,}\s+[a-zA-Z]{1,}(?:\s+[a-zA-Z]{1,})?)\b"
    )
    
    # More conservative exclusions - only obvious non-names
    EXCLUDES = {
        # Address/business components (only endings)
        "Street", "Avenue", "Road", "Company", "Corp", "LLC", "Inc", 
        "Boulevard", "Lane", "Court", "Circle", "Way", "Place", "Highway",
        # Directions (only endings)  
        "North", "South", "East", "West",
        # Days and months (only endings)
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August", 
        "September", "October", "November", "December",
    }
    
    # Patterns that indicate non-name content
    NON_NAME_PATTERNS = [
        re.compile(r"\bMy\s+[Nn]ame\s+[Ii]s\b", re.I),
        re.compile(r"\bEmail\s+[Aa]ddress\s*:?\s*", re.I),
        re.compile(r"\bEmail\s*:?\s*", re.I),
    ]

    def finditer(self, text: str) -> Iterable[Tuple[int, int]]:
        for m in self.NAME.finditer(text):
            value = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            
            # Skip if it matches non-name patterns in the surrounding context
            context = self._context(text, start, end, window=20)
            skip_match = False
            for pattern in self.NON_NAME_PATTERNS:
                if pattern.search(context):
                    skip_match = True
                    break
            if skip_match:
                continue
            
            # Only skip if the LAST word is clearly an excluded term
            words = value.split()
            if len(words) >= 2 and words[-1] in self.EXCLUDES:
                continue
                
            # Skip obvious address context (be more specific)
            if re.search(r"\b\d{3,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Drive|Dr)\b", context, re.I):
                continue
                
            # Skip if surrounded by obvious email context
            if re.search(r"@\w+\.\w+|email\s*:|contact\s*information", context, re.I):
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

# ---------- Secrets Detector ----------
"""
Changes:
1. Removed wildcard 16-char and 40-char patterns that caused massive false positives
2. Added context-aware detection for AWS secret keys
3. Added explicit patterns for real AWS session tokens
4. Added generic high-entropy secret detection with keyword context requirement
5. Each pattern is documented with what it actually matches
"""
class SecretsDetector:
    """
    Detector for API keys, tokens, and credentials.

    Design principles:
    - Structured secrets (sk-..., ghp_..., AIza...) are matched by prefix → high confidence
    - Unstructured secrets (AWS secret key, generic passwords) require keyword context
    - No pattern should match arbitrary alphanumeric strings without context
    """

    type = "SECRETS"

    # ── Structured secrets: matched by distinctive prefix ──────────────

    STRUCTURED_PATTERNS: List[re.Pattern] = [
        # OpenAI legacy/standard keys: sk- + 32-64 alphanumeric chars
        re.compile(r"\bsk-[A-Za-z0-9]{32,64}\b"),
        # OpenAI project-scoped keys
        re.compile(r"\bsk-proj-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),
        # OpenAI org-scoped keys
        re.compile(r"\bsk-org-[A-Za-z0-9]{16,}-[A-Za-z0-9]{16,}\b"),
        # OpenAI legacy format with known middle segment
        re.compile(r"\bsk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b"),
        # AWS Access Key ID (always starts with AKIA or ASIA)
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        # Google API Key (always starts with AIza)
        re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
        # Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-, xoxs-)
        re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,100}\b"),
        # JWT (three base64url segments separated by dots)
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
        ),
        # Stripe keys (sk_live_ or sk_test_)
        re.compile(r"\bsk_(?:live|test)_[0-9a-zA-Z]{24,}\b"),
        # GitHub Personal Access Token
        re.compile(r"\bghp_[0-9A-Za-z]{36}\b"),
        # GitHub Fine-grained PAT
        re.compile(r"\bgithub_pat_[0-9A-Za-z_]{36,}\b"),
        # GitLab tokens
        re.compile(r"\bglpat-[0-9A-Za-z\-]{20,}\b"),
        # Azure keys (32 hex or base64 with known prefix patterns)
        re.compile(r"\b[0-9a-fA-F]{32,}\b(?=\s*['\"]?\s*[;,}\)])"),  # hex in code context
        # npm tokens
        re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    ]

    # ── Context-dependent secrets: require nearby keywords ─────────────

    # Keywords that indicate a secret is nearby
    SECRET_CONTEXT_KEYWORDS = re.compile(
        r"("
        r"secret[_\s-]?key|access[_\s-]?key|api[_\s-]?key|api[_\s-]?secret|"
        r"private[_\s-]?key|auth[_\s-]?token|bearer|credentials?|password|passwd|"
        r"aws[_\s-]?secret|client[_\s-]?secret|signing[_\s-]?key|"
        r"encryption[_\s-]?key|master[_\s-]?key|service[_\s-]?key|"
        r"secret|token|apikey|api_key"
        r")",
        re.IGNORECASE,
    )

    # AWS Secret Access Key: exactly 40 chars of base64-ish, but ONLY with context
    # Use lookaround instead of \b since / and + break word boundaries
    AWS_SECRET_KEY = re.compile(r"(?<![0-9a-zA-Z/+])[0-9a-zA-Z/+]{40}(?![0-9a-zA-Z/+])")

    # Generic high-entropy string (20-64 chars) — only flagged with keyword context
    GENERIC_SECRET = re.compile(r"\b[A-Za-z0-9/+=_\-]{20,64}\b")

    # Context window (chars before/after) to search for keywords
    CONTEXT_WINDOW = 80

    def _has_secret_context(self, text: str, start: int, end: int) -> bool:
        """Check if a match is near secret-related keywords."""
        ctx_start = max(0, start - self.CONTEXT_WINDOW)
        ctx_end = min(len(text), end + self.CONTEXT_WINDOW)
        context = text[ctx_start:ctx_end]
        return bool(self.SECRET_CONTEXT_KEYWORDS.search(context))

    @staticmethod
    def _shannon_entropy(s: str) -> float:
        """Calculate Shannon entropy of a string (bits per character)."""
        if not s:
            return 0.0
        freq: Dict[str, int] = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        length = len(s)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in freq.values()
        )

    # Minimum entropy (bits/char) to consider a string "secret-like"
    # English text ≈ 3.5-4.0, random base64 ≈ 5.5-6.0
    MIN_ENTROPY = 4.5

    def finditer(self, text: str) -> List[Dict[str, Any]]:
        """
        Find all secrets in text.

        Returns list of dicts with 'span', 'start', 'end' keys.
        """
        findings: List[Dict[str, Any]] = []
        seen_ranges: set = set()  # avoid duplicates

        def _add(start: int, end: int, span: str) -> None:
            key = (start, end)
            if key not in seen_ranges:
                seen_ranges.add(key)
                findings.append({"span": span, "start": start, "end": end})

        # 1. Structured patterns — high confidence, no context needed
        for pat in self.STRUCTURED_PATTERNS:
            for m in pat.finditer(text):
                _add(m.start(), m.end(), m.group())

        # 2. AWS Secret Key pattern — only with context
        for m in self.AWS_SECRET_KEY.finditer(text):
            if self._has_secret_context(text, m.start(), m.end()):
                # Also check entropy to avoid matching normal base64 text
                if self._shannon_entropy(m.group()) >= self.MIN_ENTROPY:
                    _add(m.start(), m.end(), m.group())

        # 3. Generic secrets — keyword context AND high entropy required
        for m in self.GENERIC_SECRET.finditer(text):
            span = m.group()
            # Skip if already found by structured patterns
            if (m.start(), m.end()) in seen_ranges:
                continue
            # Skip short matches (structured patterns handle those)
            if len(span) < 20:
                continue
            # Require both context and entropy
            if (
                self._has_secret_context(text, m.start(), m.end())
                and self._shannon_entropy(span) >= self.MIN_ENTROPY
            ):
                _add(m.start(), m.end(), span)

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
    text = validate_input(text, mode='regex')  # text validation for regex mode
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
    text = validate_input(text, mode='regex')  # text validation for regex mode
    detector = SecretsDetector()
    findings = detector.finditer(text)
    if findings:
        return {"SECRETS": findings}
    return {}

