import html
import re
import unicodedata

# Zero-width and invisible Unicode characters to strip
_ZERO_WIDTH_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f"  # zero-width space/joiners/marks
    "\u00ad"  # soft hyphen
    "\u2060\u2061\u2062\u2063\u2064"  # word joiner, invisible operators
    "\ufeff"  # BOM / zero-width no-break space
    "\ufffe\uffff]"  # non-characters
)

# Control characters except common whitespace (\t, \n, \r)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# URLs (http/https/ftp, plus bare www.)
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"'\])]+|ftp://[^\s<>\"'\])]+|www\.[^\s<>\"'\])]+",
    re.IGNORECASE,
)

# Common news boilerplate patterns
_BOILERPLATE_PATTERNS = [
    re.compile(
        r"(?i)^(read more|continue reading|click here to read more)[\s.…]*$",
        re.MULTILINE,
    ),
    re.compile(r"(?i)^(share this|share on|share via)[\s:].*$", re.MULTILINE),
    re.compile(r"(?i)^(subscribe|sign up|register)[\s:].*newsletter.*$", re.MULTILINE),
    re.compile(r"(?i)^(advertisement|sponsored|promoted)[\s.]*$", re.MULTILINE),
    re.compile(r"(?i)^(follow us on|like us on|join us on)[\s:].*$", re.MULTILINE),
    re.compile(r"(?i)^\[?\d+\s*(photos?|images?|videos?)\]?\s*$", re.MULTILINE),
    re.compile(r"(?i)^(photo|image|video|source)\s*:\s*.*$", re.MULTILINE),
    re.compile(r"(?i)^(all rights reserved|copyright\s*©?).*$", re.MULTILINE),
    re.compile(r"(?i)^(related|also read|see also|more from)\s*:?\s*$", re.MULTILINE),
]

# Social media artifacts
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_MENTION_PATTERN = re.compile(r"@[\w.]+")

# HTML tags (for residual tags that newspaper4k may leave)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Consecutive blank lines (3+ newlines → 2 newlines)
_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")


def _clean_html(text: str) -> str:
    text = _HTML_TAG_PATTERN.sub("", text)
    text = html.unescape(text)
    return text


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_CHARS.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    return text


def _remove_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


def _remove_urls(text: str) -> str:
    return _URL_PATTERN.sub("", text)


def _clean_social_media(text: str) -> str:
    text = _HASHTAG_PATTERN.sub(r"\1", text)
    text = _MENTION_PATTERN.sub("", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = _EXCESSIVE_NEWLINES.sub("\n\n", text)
    lines = text.split("\n")
    lines = [" ".join(line.split()) for line in lines]
    text = "\n".join(lines)
    return text.strip()


def clean_text(text: str) -> str:
    text = _clean_html(text)
    text = _normalize_unicode(text)
    text = _remove_boilerplate(text)
    text = _remove_urls(text)
    text = _clean_social_media(text)
    text = _normalize_whitespace(text)
    return text
