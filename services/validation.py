import re

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_ORIGIN_ALLOWED_RE = re.compile(r"[^0-9A-Za-zÀ-ÿ _./-]")


def _normalize_text(value: str, max_len: int) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = _CONTROL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    if max_len and len(text) > max_len:
        text = text[:max_len]
    return text


def sanitize_name(value: str) -> str:
    return _normalize_text(value, 80)


def sanitize_origin(value: str) -> str:
    text = _normalize_text(value, 80)
    if not text:
        return ""
    return _ORIGIN_ALLOWED_RE.sub("", text)


def sanitize_phone(value: str) -> str:
    text = _normalize_text(value, 40)
    if not text:
        return ""
    if text.startswith("+"):
        digits = "+" + re.sub(r"\D", "", text[1:])
    else:
        digits = re.sub(r"\D", "", text)
    digits = digits[:20]
    if len(digits.replace("+", "")) < 8:
        return ""
    return digits
