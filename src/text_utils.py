from __future__ import annotations

import re
from typing import Iterable

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover - optional dependency
    word_tokenize = None


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def combine_text_fields(title: object, description: object) -> str:
    title_text = normalize_whitespace(str(title or ""))
    desc_text = normalize_whitespace(str(description or ""))
    parts = [part for part in (title_text, desc_text) if part]
    return " ".join(parts)


def segment_vietnamese_text(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    if word_tokenize is not None:
        try:
            tokens = word_tokenize(text, format="text")
            return normalize_whitespace(tokens)
        except Exception:
            pass
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return " ".join(tokens)


def safe_text_series(values: Iterable[object]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and pd_isna(value)):
            cleaned.append("")
        else:
            cleaned.append(str(value))
    return cleaned


def pd_isna(value: object) -> bool:
    import math
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False
